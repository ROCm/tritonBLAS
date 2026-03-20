#!/usr/bin/env python3
"""Collect all data needed for the overlap analysis plots.

Outputs JSON files consumed by the plotting script.
"""
import torch
import tritonblas
import statistics
import json
import os
import sys
import time

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
s = torch.cuda.Stream(device=dev)

OUT_DIR = "results/plot_data"
os.makedirs(OUT_DIR, exist_ok=True)


def bench_single(fn, reset_fn, stream, warmup=20, steps=50):
    """Return list of per-iter times (ms) with same-stream reset."""
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            if reset_fn:
                reset_fn()
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        if reset_fn:
            with torch.cuda.stream(stream):
                reset_fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(stream)
        with torch.cuda.stream(stream):
            fn()
        en.record(stream)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


def make_ws(M, N, K, dev, dtype):
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    C = torch.empty(M, N, dtype=dtype, device=dev)
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
    cfg = tritonblas.matmul_preamble(sel)
    grid = sel._hardware.N_CU
    total_tiles = (M // sel.block_m) * (N // sel.block_n)
    def fn():
        tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
    def reset():
        cfg.reset(work_stealing=True)
    return fn, reset, {"grid": grid, "total_tiles": total_tiles,
                       "block_m": sel.block_m, "block_n": sel.block_n, "block_k": sel.block_k}


def make_torch(M, N, K, dev, dtype):
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    C = torch.empty(M, N, dtype=dtype, device=dev)
    def fn():
        torch.matmul(A, B, out=C)
    return fn, None, {}


# ===================================================================
# TASK 1: Multi-size overlap penalty (1K-16K)
# ===================================================================
def collect_multisize():
    print("=== Collecting multi-size data ===")
    sizes = [1024, 2048, 4096, 8192, 12288, 16384]
    data = {"sizes": [], "ws": [], "torch": []}

    for sz in sizes:
        M = N = K = sz
        flops = 2.0 * M * N * K
        print(f"  Size {sz}x{sz}x{sz}...")

        for backend_name, make_fn in [("ws", make_ws), ("torch", make_torch)]:
            fn, reset_fn, info = make_fn(M, N, K, dev, dtype)
            times = bench_single(fn, reset_fn, s, warmup=20, steps=100)
            med = statistics.median(times)
            tflops = flops / (med * 1e-3) / 1e12

            entry = {
                "size": sz, "median_ms": med, "mean_ms": statistics.mean(times),
                "min_ms": min(times), "max_ms": max(times),
                "tflops": tflops, **info,
            }
            data[backend_name].append(entry)
            print(f"    {backend_name}: {med:.3f} ms ({tflops:.1f} TF)")

        if sz not in data["sizes"]:
            data["sizes"].append(sz)

        del fn, reset_fn
        torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/multisize_alone.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {OUT_DIR}/multisize_alone.json")


# ===================================================================
# TASK 2: 200-step full distribution at 8K
# ===================================================================
def collect_distribution_8k():
    print("\n=== Collecting 200-step distribution at 8K ===")
    M = N = K = 8192
    flops = 2.0 * M * N * K
    data = {}

    for backend_name, make_fn in [("ws", make_ws), ("torch", make_torch)]:
        fn, reset_fn, info = make_fn(M, N, K, dev, dtype)
        times = bench_single(fn, reset_fn, s, warmup=20, steps=200)
        data[f"{backend_name}_warm"] = times
        print(f"  {backend_name} warm: med={statistics.median(times):.3f}")

        # Rotating
        N_BUFS = 4
        rot_fns = []
        for _ in range(N_BUFS):
            rfn, rrst, _ = make_fn(M, N, K, dev, dtype)
            rot_fns.append((rfn, rrst))
        for j in range(max(20, N_BUFS)):
            idx = j % N_BUFS
            with torch.cuda.stream(s):
                if rot_fns[idx][1]:
                    rot_fns[idx][1]()
                rot_fns[idx][0]()
        torch.cuda.synchronize()

        rot_times = []
        for i in range(200):
            idx = i % N_BUFS
            if rot_fns[idx][1]:
                with torch.cuda.stream(s):
                    rot_fns[idx][1]()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(s)
            with torch.cuda.stream(s):
                rot_fns[idx][0]()
            en.record(s)
            torch.cuda.synchronize()
            rot_times.append(st.elapsed_time(en))
        data[f"{backend_name}_rotating"] = rot_times
        print(f"  {backend_name} rotating: med={statistics.median(rot_times):.3f}")

        del fn, reset_fn, rot_fns
        torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/distribution_8k.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {OUT_DIR}/distribution_8k.json")


# ===================================================================
# TASK 3: CU masking sweep (8K and 4K)
# ===================================================================
def collect_cu_sweep(sz):
    print(f"\n=== Collecting CU sweep at {sz}x{sz}x{sz} ===")
    M = N = K = sz
    flops = 2.0 * M * N * K

    cu_counts = list(range(24, 305, 8))
    if 304 not in cu_counts:
        cu_counts.append(304)
    cu_counts.sort()

    data = {"size": sz, "cu_counts": cu_counts, "ws": [], "torch": []}

    for n_cus in cu_counts:
        # CU mask: n_cus CUs enabled out of 304
        # MI300X: 304 CUs = 38 per XCD * 8 XCDs
        # ROC_GLOBAL_CU_MASK uses a 64-bit mask per SE
        # For simplicity, we use the tritonblas CU limiting
        mask_hex = (1 << n_cus) - 1 if n_cus < 64 else None

        for backend_name in ["ws", "torch"]:
            try:
                if backend_name == "ws":
                    A = torch.randn(M, K, dtype=dtype, device=dev)
                    B = torch.randn(K, N, dtype=dtype, device=dev)
                    C = torch.empty(M, N, dtype=dtype, device=dev)
                    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev,
                                                            total_cus=n_cus)
                    cfg = tritonblas.matmul_preamble(sel)
                    def fn(a=A, b=B, c=C, s_=sel, cf=cfg):
                        tritonblas.matmul_lt(a, b, c, s_, cf, work_stealing=True)
                    def reset(cf=cfg):
                        cf.reset(work_stealing=True)
                    times = bench_single(fn, reset, s, warmup=10, steps=30)
                else:
                    os.environ["ROC_GLOBAL_CU_MASK"] = hex((1 << n_cus) - 1) if n_cus <= 64 else "0xffffffffffffffff"
                    A = torch.randn(M, K, dtype=dtype, device=dev)
                    B = torch.randn(K, N, dtype=dtype, device=dev)
                    C = torch.empty(M, N, dtype=dtype, device=dev)
                    def fn(a=A, b=B, c=C):
                        torch.matmul(a, b, out=c)
                    times = bench_single(fn, None, s, warmup=10, steps=30)
                    if "ROC_GLOBAL_CU_MASK" in os.environ:
                        del os.environ["ROC_GLOBAL_CU_MASK"]

                med = statistics.median(times)
                tflops = flops / (med * 1e-3) / 1e12
                data[backend_name].append({"cus": n_cus, "median_ms": med, "tflops": tflops})
                if n_cus % 40 == 0 or n_cus == 304:
                    print(f"  {backend_name} CUs={n_cus}: {med:.3f} ms ({tflops:.1f} TF)")
            except Exception as e:
                print(f"  {backend_name} CUs={n_cus}: FAILED ({str(e)[:60]})")
                data[backend_name].append({"cus": n_cus, "median_ms": None, "tflops": None})

            torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/cu_sweep_{sz}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {OUT_DIR}/cu_sweep_{sz}.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "multisize"):
        collect_multisize()
    if mode in ("all", "dist"):
        collect_distribution_8k()
    if mode in ("all", "cu8k"):
        collect_cu_sweep(8192)
    if mode in ("all", "cu4k"):
        collect_cu_sweep(4096)

#!/usr/bin/env python3
"""Final data collection:
1. Global atomic vs per-XCD sweep for each size
2. Rotating-buffer overlap comparison (realistic training scenario)

Run with: torchrun --nproc_per_node=8 benchmarks/collect_final.py
"""
import torch
import torch.distributed as dist
import tritonblas
from tritonblas.config import COUNTER_STRIDE, matmul_preamble
from tritonblas.kernels import ws_persistent_matmul
import statistics
import json
import os

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
s = torch.cuda.Stream(device=dev)
cs = torch.cuda.Stream(device=dev)

OUT = "results/plot_data"
os.makedirs(OUT, exist_ok=True)

SIZES = [1024, 2048, 4096, 8192, 12288, 16384]
N_BUFS = 4


def bench_rotating_alone(fns, stream, steps=100):
    """Rotating-buffer alone benchmark."""
    n = len(fns)
    for j in range(max(20, n)):
        with torch.cuda.stream(stream):
            if fns[j % n][1]:
                fns[j % n][1]()
            fns[j % n][0]()
    torch.cuda.synchronize()
    times = []
    for i in range(steps):
        idx = i % n
        if fns[idx][1]:
            with torch.cuda.stream(stream):
                fns[idx][1]()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(stream)
        with torch.cuda.stream(stream):
            fns[idx][0]()
        en.record(stream)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


def bench_rotating_overlap(fns, comm_fn, stream, cstream, steps=100):
    """Rotating-buffer + RCCL overlap benchmark."""
    n = len(fns)
    for j in range(max(20, n)):
        if fns[j % n][1]:
            with torch.cuda.stream(stream):
                fns[j % n][1]()
        with torch.cuda.stream(cstream):
            comm_fn()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000)
            fns[j % n][0]()
        torch.cuda.synchronize()

    times = []
    for i in range(steps):
        idx = i % n
        if fns[idx][1]:
            with torch.cuda.stream(stream):
                fns[idx][1]()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(cstream):
            comm_fn()
        st.record(stream)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000)
            fns[idx][0]()
        en.record(stream)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


# ===================================================================
# Part 1: Global atomic sweep for each size (rank 0 only)
# ===================================================================
if rank == 0:
    print("=" * 70)
    print("  Part 1: Global atomic vs per-XCD sweep")
    print("=" * 70)
    cpx_results = {}
    for sz in SIZES:
        M = N = K = sz
        FLOPS = 2.0 * M * N * K
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

        configs = [
            ("global", True, 1),
            ("cpx=1", False, 1),
            ("cpx=2", False, 2),
            ("cpx=4", False, 4),
            ("cpx=8", False, 8),
            ("cpx=16", False, 16),
        ]
        best_label, best_ms = None, 999
        size_results = []

        for label, ga, cpx in configs:
            sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
            sel.COUNTERS_PER_XCD = cpx
            cfg = matmul_preamble(sel)
            grids = sel._hardware.N_CU
            num_xcds = sel.num_sms

            def fn(a=A, b=B, c=C, s_=sel, cf=cfg, _ga=ga, _cpx=cpx):
                ws_persistent_matmul[(grids,)](
                    a, b, c, None, None, None, cf.tile_counter,
                    M, N, K, a.stride(0), b.stride(1), c.stride(0), c.stride(1), 0,
                    stride_ak=a.stride(1), stride_bk=b.stride(0),
                    BLOCK_SIZE_M=sel.block_m, BLOCK_SIZE_N=sel.block_n, BLOCK_SIZE_K=sel.block_k,
                    GROUP_SIZE_M=sel.group_m, NUM_SMS=grids, NUM_XCDS=num_xcds,
                    COUNTERS_PER_XCD=_cpx, COUNTER_STRIDE=COUNTER_STRIDE,
                    BIAS=False, EVEN_K=(K % sel.block_k == 0),
                    CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
                    QUANTIZED=False, GLOBAL_ATOMIC=_ga,
                    num_stages=2, num_warps=8, waves_per_eu=1,
                    matrix_instr_nonkdim=16, kpack=1,
                    mask_ptr=torch.ones(sel._N_CU, dtype=torch.int32, device=dev),
                )

            def rst(cf=cfg):
                cf.reset(work_stealing=True)

            for _ in range(10):
                with torch.cuda.stream(s):
                    rst()
                    fn()
            torch.cuda.synchronize()

            times = []
            for _ in range(30):
                with torch.cuda.stream(s):
                    rst()
                torch.cuda.synchronize()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record(s)
                with torch.cuda.stream(s):
                    fn()
                en.record(s)
                torch.cuda.synchronize()
                times.append(st.elapsed_time(en))

            med = statistics.median(times)
            tf = FLOPS / (med * 1e-3) / 1e12
            tiles = (M // sel.block_m) * (N // sel.block_n)
            total_locks = 1 if ga else num_xcds * cpx
            if med < best_ms:
                best_label, best_ms = label, med
            size_results.append({
                "label": label, "global": ga, "cpx": cpx,
                "ms": med, "tflops": tf, "locks": total_locks,
                "tiles_per_lock": tiles / total_locks,
            })

        cpx_results[str(sz)] = {"best": best_label, "results": size_results}
        print(f"  {sz:>6}: best={best_label} ({best_ms:.3f}ms)")
        for r in size_results:
            mark = " <<<" if r["label"] == best_label else ""
            print(f"    {r['label']:<10} {r['ms']:.3f}ms  {r['tflops']:.0f}TF  "
                  f"locks={r['locks']}  tiles/lock={r['tiles_per_lock']:.1f}{mark}")

        del A, B, C
        torch.cuda.empty_cache()

    with open(f"{OUT}/cpx_sweep_with_global.json", "w") as f:
        json.dump(cpx_results, f, indent=2)
    print(f"\nSaved cpx_sweep_with_global.json")

dist.barrier()

# ===================================================================
# Part 2: Rotating-buffer overlap comparison
# ===================================================================
if rank == 0:
    print("\n" + "=" * 70)
    print("  Part 2: Rotating-buffer overlap (realistic training)")
    print("=" * 70)

# Optimal CPX from Part 1 (or pre-determined)
OPT_CPX = {1024: 16, 2048: 8, 4096: 16, 8192: 4, 12288: 1, 16384: 2}

rot_results = {}
for sz in SIZES:
    M = N = K = sz
    FLOPS = 2.0 * M * N * K
    comm = torch.randn(sz, sz, dtype=torch.bfloat16, device=dev)

    def comm_fn(c=comm):
        dist.all_reduce(c)

    for bk in ["ws", "torch"]:
        # Build rotating buffer sets
        rot_fns = []
        for _ in range(N_BUFS):
            A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
            C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            if bk == "ws":
                sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
                sel.COUNTERS_PER_XCD = OPT_CPX[sz]
                cfg = matmul_preamble(sel)

                def fn(a=A, b=B, c=C, s_=sel, cf=cfg):
                    tritonblas.matmul_lt(a, b, c, s_, cf, work_stealing=True)

                def rst(cf=cfg):
                    cf.reset(work_stealing=True)

                rot_fns.append((fn, rst))
            else:
                def fn(a=A, b=B, c=C):
                    torch.matmul(a, b, out=c)
                rot_fns.append((fn, None))

        # Rotating alone
        alone_times = bench_rotating_alone(rot_fns, s)
        # Rotating + RCCL overlap
        ovl_times = bench_rotating_overlap(rot_fns, comm_fn, s, cs)

        key = f"{bk}_{sz}"
        am = statistics.median(alone_times)
        om = statistics.median(ovl_times)
        penalty = (om / am - 1) * 100
        tf_a = FLOPS / (am * 1e-3) / 1e12
        tf_o = FLOPS / (om * 1e-3) / 1e12

        rot_results[key] = {
            "size": sz, "backend": bk,
            "alone_median": am, "overlap_median": om,
            "alone_tflops": tf_a, "overlap_tflops": tf_o,
            "penalty_pct": penalty,
            "alone_all": alone_times, "overlap_all": ovl_times,
        }
        if rank == 0:
            print(f"  {bk:>5} {sz:>6}: alone={am:.3f}ms({tf_a:.0f}TF) "
                  f"ovl={om:.3f}ms({tf_o:.0f}TF) penalty={penalty:+.1f}%")

        del rot_fns
        torch.cuda.empty_cache()
    del comm
    torch.cuda.empty_cache()

if rank == 0:
    with open(f"{OUT}/rotating_overlap.json", "w") as f:
        json.dump(rot_results, f, indent=2)
    print(f"\nSaved rotating_overlap.json")

dist.destroy_process_group()

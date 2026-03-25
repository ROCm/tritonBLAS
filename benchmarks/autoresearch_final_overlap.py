#!/usr/bin/env python3
"""
Autoresearch: FINAL overlap comparison with all Phase 4-6 optimizations.

Tests both gm=8 (default) and gm=4 (Phase 6b marginal winner), plus
the optimal overlap split ratios from Phase 5b.

Run: torchrun --nproc_per_node=8 benchmarks/autoresearch_final_overlap.py
"""
import json, os, statistics, sys
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

WARMUP = 15
STEPS = 60
N_ROT = 4
COMM_SIZE = (16384, 16384)
dtype = torch.bfloat16


def make_ws(A, B, C, gm=8, split_frac=None):
    M, K = A.shape; _, N = B.shape; dev = A.device
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev, streamk=False)
    BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
    num_xcds = sel.num_sms; n_cu = sel._N_CU
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    even_k = K % BLK_K == 0

    if split_frac is not None:
        lp = int(total_tiles * split_frac) // num_xcds
    else:
        lp, _ = sel.hierarchical_split(num_xcds)
    lp = max(lp, 1)
    gt = total_tiles - lp * num_xcds

    tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=dev, dtype=torch.int32)
    gc = torch.zeros(COUNTER_STRIDE, device=dev, dtype=torch.int32)
    mask = torch.ones(n_cu, dtype=torch.int32, device=dev)

    def fn():
        ws_hierarchical_matmul[(n_cu,)](
            A, B, C, None, None, None, tc, gc,
            M, N, K, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
            stride_ak=A.stride(1), stride_bk=B.stride(0),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
        )
    def reset(): tc.zero_(); gc.zero_()
    return fn, reset


def bench_alone(fn, rfn, stream, warmup=WARMUP, steps=STEPS):
    for _ in range(warmup):
        if rfn: rfn()
        with torch.cuda.stream(stream): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        if rfn: rfn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(stream)
        with torch.cuda.stream(stream): fn()
        e.record(stream)
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench_overlap(fn, rfn, comm_fn, mm_stream, co_stream, warmup=WARMUP, steps=STEPS):
    for _ in range(warmup):
        if rfn: rfn()
        co_stream.wait_stream(mm_stream)
        with torch.cuda.stream(co_stream): comm_fn()
        mm_stream.wait_stream(co_stream)
        with torch.cuda.stream(mm_stream): fn()
    torch.cuda.synchronize()

    walls, mms, cos = [], [], []
    for _ in range(steps):
        if rfn: rfn()
        torch.cuda.synchronize()
        sw = torch.cuda.Event(enable_timing=True)
        ew = torch.cuda.Event(enable_timing=True)
        sm = torch.cuda.Event(enable_timing=True)
        em = torch.cuda.Event(enable_timing=True)
        sc = torch.cuda.Event(enable_timing=True)
        ec = torch.cuda.Event(enable_timing=True)

        sw.record()
        sc.record(co_stream)
        with torch.cuda.stream(co_stream): comm_fn()
        ec.record(co_stream)
        sm.record(mm_stream)
        with torch.cuda.stream(mm_stream): fn()
        em.record(mm_stream)
        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)
        ew.record()
        torch.cuda.synchronize()
        walls.append(sw.elapsed_time(ew))
        mms.append(sm.elapsed_time(em))
        cos.append(sc.elapsed_time(ec))
    return statistics.median(walls), statistics.median(mms), statistics.median(cos)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)
    comm_buf = torch.randn(*COMM_SIZE, dtype=dtype, device=dev)
    def comm_fn(): dist.all_reduce(comm_buf)

    # WS configs: (label, gm, split_frac)
    ws_configs = [
        ("WS (adaptive)", 8, None),
        ("WS (50/50)", 8, 0.5),
        ("WS (70/30)", 8, 0.7),
        ("WS gm=4 (adaptive)", 4, None),
    ]

    sizes = [4096, 8192, 12288, 16384]
    all_results = {}

    for sz in sizes:
        M = N = K = sz
        flops = 2.0 * M * N * K

        # torch
        tA = torch.randn(M, K, dtype=dtype, device=dev)
        tB = torch.randn(K, N, dtype=dtype, device=dev)
        torch_fn = lambda: torch.matmul(tA, tB)
        torch_alone = bench_alone(torch_fn, None, mm_stream)
        torch_wall, torch_mm, _ = bench_overlap(torch_fn, None, comm_fn, mm_stream, co_stream)
        torch_penalty = (torch_mm - torch_alone) / torch_alone * 100

        results_for_size = {
            "torch": {"alone": torch_alone, "wall": torch_wall, "mm_overlap": torch_mm,
                      "penalty": torch_penalty}
        }

        if rank == 0:
            print(f"\n{'='*90}")
            print(f"  {sz}x{sz}x{sz} BF16   RCCL: {COMM_SIZE[0]}x{COMM_SIZE[1]}")
            print(f"{'='*90}")
            tf_alone = flops / (torch_alone * 1e-3) / 1e12
            print(f"  torch: alone={torch_alone:.3f}ms ({tf_alone:.0f}TF)  wall={torch_wall:.3f}ms  "
                  f"penalty={torch_penalty:+.1f}%")
            print(f"\n  {'Config':<22s}  {'alone':>8s}  {'wall':>8s}  {'penalty':>8s}  "
                  f"{'delta':>8s}  {'winner':>6s}")
            print("  " + "-" * 70)

        for label, gm, sf in ws_configs:
            A = torch.randn(M, K, dtype=dtype, device=dev)
            B = torch.randn(K, N, dtype=dtype, device=dev)
            C = torch.zeros(M, N, dtype=dtype, device=dev)
            fn, rfn = make_ws(A, B, C, gm=gm, split_frac=sf)

            ws_alone = bench_alone(fn, rfn, mm_stream)
            ws_wall, ws_mm, comm_med = bench_overlap(fn, rfn, comm_fn, mm_stream, co_stream)
            ws_penalty = (ws_mm - ws_alone) / ws_alone * 100
            delta = ws_wall - torch_wall
            winner = "WS" if delta < 0 else "torch"

            if rank == 0:
                print(f"  {label:<22s}  {ws_alone:>8.3f}  {ws_wall:>8.3f}  {ws_penalty:>+7.1f}%  "
                      f"{delta:>+8.3f}  {winner:>6s}")

            results_for_size[label] = {
                "alone": ws_alone, "wall": ws_wall, "mm_overlap": ws_mm,
                "penalty": ws_penalty, "delta": delta,
            }
            del A, B, C; torch.cuda.empty_cache()

        all_results[str(sz)] = results_for_size

    if rank == 0:
        os.makedirs("results/autoresearch", exist_ok=True)
        with open("results/autoresearch/final_overlap.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*90}")
        print("  FINAL OVERLAP SUMMARY (best WS variant per size)")
        print(f"{'='*90}")
        print(f"  {'Size':<8s}  {'torch wall':>10s}  {'best WS':>10s}  {'config':>24s}  {'delta':>10s}")
        for k, r in all_results.items():
            tw = r["torch"]["wall"]
            best_label, best_wall = None, float('inf')
            for name, data in r.items():
                if name == "torch": continue
                if data["wall"] < best_wall:
                    best_wall = data["wall"]
                    best_label = name
            delta = best_wall - tw
            winner = "WS" if delta < 0 else "torch"
            print(f"  {k:<8s}  {tw:>10.3f}  {best_wall:>10.3f}  {best_label:>24s}  "
                  f"{delta:>+10.3f} ({winner})")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

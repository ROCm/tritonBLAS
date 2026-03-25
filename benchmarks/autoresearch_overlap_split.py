#!/usr/bin/env python3
"""
Autoresearch: Test different split ratios specifically during RCCL overlap.
The isolation-optimal split may differ from the overlap-optimal split.

Run: torchrun --nproc_per_node=8 benchmarks/autoresearch_overlap_split.py
"""
import os, statistics, sys
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

WARMUP = 10
STEPS = 30
COMM_SIZE = (16384, 16384)
dtype = torch.bfloat16


def make_ws(A, B, C, local_frac):
    M, K = A.shape; _, N = B.shape; dev = A.device
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev, streamk=False)
    BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
    num_xcds = sel.num_sms; gm = sel.group_m; n_cu = sel._N_CU
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    even_k = K % BLK_K == 0

    if local_frac >= 1.0:
        lp = total_tiles // num_xcds
    else:
        lp = int(total_tiles * local_frac) // num_xcds
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

    walls, mms = [], []
    for _ in range(steps):
        if rfn: rfn()
        torch.cuda.synchronize()
        sw = torch.cuda.Event(enable_timing=True)
        ew = torch.cuda.Event(enable_timing=True)
        sm = torch.cuda.Event(enable_timing=True)
        em = torch.cuda.Event(enable_timing=True)
        sw.record()
        with torch.cuda.stream(co_stream): comm_fn()
        sm.record(mm_stream)
        with torch.cuda.stream(mm_stream): fn()
        em.record(mm_stream)
        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)
        ew.record()
        torch.cuda.synchronize()
        walls.append(sw.elapsed_time(ew))
        mms.append(sm.elapsed_time(em))
    return statistics.median(walls), statistics.median(mms)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)
    comm_buf = torch.randn(*COMM_SIZE, dtype=dtype, device=dev)
    def comm_fn(): dist.all_reduce(comm_buf)

    for sz in [12288, 16384]:
        M = N = K = sz
        flops = 2.0 * M * N * K

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"  {sz}x{sz}x{sz} — Split ratio sweep during RCCL overlap")
            print(f"{'='*80}")

        # torch baseline
        tA = torch.randn(M, K, dtype=dtype, device=dev)
        tB = torch.randn(K, N, dtype=dtype, device=dev)
        torch_fn = lambda: torch.matmul(tA, tB)
        torch_alone = bench_alone(torch_fn, None, mm_stream)
        torch_wall, torch_mm = bench_overlap(torch_fn, None, comm_fn, mm_stream, co_stream)
        torch_penalty = (torch_mm - torch_alone) / torch_alone * 100

        if rank == 0:
            print(f"\n  torch.matmul:  alone={torch_alone:.3f}ms  wall={torch_wall:.3f}ms  "
                  f"penalty={torch_penalty:+.1f}%")
            print(f"\n  {'Split':<12s}  {'alone':>8s}  {'wall':>8s}  {'penalty':>8s}  "
                  f"{'vs torch':>10s}  {'delta':>8s}")
            print("  " + "-" * 65)

        fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0]
        for frac in fracs:
            A = torch.randn(M, K, dtype=dtype, device=dev)
            B = torch.randn(K, N, dtype=dtype, device=dev)
            C = torch.zeros(M, N, dtype=dtype, device=dev)
            fn, rfn = make_ws(A, B, C, frac)

            alone = bench_alone(fn, rfn, mm_stream)
            wall, mm_dur = bench_overlap(fn, rfn, comm_fn, mm_stream, co_stream)
            penalty = (mm_dur - alone) / alone * 100
            delta = wall - torch_wall

            if rank == 0:
                label = f"{int(frac*100)}/{int((1-frac)*100)}"
                winner = "WS" if delta < 0 else "torch"
                print(f"  {label:<12s}  {alone:>8.3f}  {wall:>8.3f}  {penalty:>+7.1f}%  "
                      f"  {winner:>6s}  {delta:>+8.3f}")

            del A, B, C; torch.cuda.empty_cache()

    if rank == 0:
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

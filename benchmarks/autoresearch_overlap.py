#!/usr/bin/env python3
"""
Autoresearch: Overlap comparison — WS Hierarchical (tuned) vs torch.matmul.

Run with torchrun:
  torchrun --nproc_per_node=8 benchmarks/autoresearch_overlap.py

Measures for each problem size:
  1. GEMM alone (no RCCL)
  2. RCCL alone (no GEMM)
  3. Overlapped (GEMM + RCCL concurrently)
  4. Overlap penalty = (overlap_gemm - alone_gemm) / alone_gemm * 100
  5. Wall-clock comparison
"""
import json, os, statistics, sys
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

WARMUP = 15
STEPS = 50
N_ROT = 4
COMM_SIZE = (16384, 16384)
dtype = torch.bfloat16


def make_ws_hierarchical(A, B, C):
    M, K = A.shape; _, N = B.shape; dev = A.device
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev, streamk=False)
    BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
    num_xcds = sel.num_sms; gm = sel.group_m; n_cu = sel._N_CU
    even_k = K % BLK_K == 0
    lp, gt = sel.hierarchical_split(num_xcds)
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
    def reset():
        tc.zero_(); gc.zero_()
    return fn, reset


def time_alone(fns, rfns, stream, warmup, steps):
    n = len(fns)
    for j in range(max(warmup, n)):
        idx = j % n
        with torch.cuda.stream(stream):
            if rfns[idx]: rfns[idx]()
            fns[idx]()
    torch.cuda.synchronize()
    times = []
    for i in range(steps):
        idx = i % n
        if rfns[idx]: rfns[idx]()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(stream)
        with torch.cuda.stream(stream):
            fns[idx]()
        e.record(stream)
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def time_overlap(fns, rfns, comm_fn, mm_stream, co_stream, warmup, steps):
    n = len(fns)
    for j in range(max(warmup, n)):
        idx = j % n
        with torch.cuda.stream(mm_stream):
            if rfns[idx]: rfns[idx]()
        co_stream.wait_stream(mm_stream)
        with torch.cuda.stream(co_stream): comm_fn()
        mm_stream.wait_stream(co_stream)
        with torch.cuda.stream(mm_stream): fns[idx]()
    torch.cuda.synchronize()

    walls, mms, cos = [], [], []
    for i in range(steps):
        idx = i % n
        if rfns[idx]: rfns[idx]()
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
        with torch.cuda.stream(mm_stream): fns[idx]()
        em.record(mm_stream)
        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)
        ew.record()
        torch.cuda.synchronize()
        walls.append(sw.elapsed_time(ew))
        mms.append(sm.elapsed_time(em))
        cos.append(sc.elapsed_time(ec))
    return walls, mms, cos


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)
    comm_buf = torch.randn(*COMM_SIZE, dtype=dtype, device=dev)
    def comm_fn():
        dist.all_reduce(comm_buf)

    sizes = [4096, 8192, 12288, 16384]
    all_results = {}

    for sz in sizes:
        M = N = K = sz
        flops = 2.0 * M * N * K

        # --- torch.matmul ---
        torch_fns, torch_rfns = [], []
        for _ in range(N_ROT):
            rA = torch.randn(M, K, dtype=dtype, device=dev)
            rB = torch.randn(K, N, dtype=dtype, device=dev)
            torch_fns.append(lambda a=rA, b=rB: torch.matmul(a, b))
            torch_rfns.append(None)

        torch_alone = time_alone(torch_fns, torch_rfns, mm_stream, WARMUP, STEPS)
        torch_alone_med = statistics.median(torch_alone)
        torch_walls, torch_mms, torch_cos = time_overlap(
            torch_fns, torch_rfns, comm_fn, mm_stream, co_stream, WARMUP, STEPS)
        torch_wall_med = statistics.median(torch_walls)
        torch_mm_med = statistics.median(torch_mms)
        torch_penalty = (torch_mm_med - torch_alone_med) / torch_alone_med * 100

        # --- WS Hierarchical ---
        ws_fns, ws_rfns = [], []
        for _ in range(N_ROT):
            rA = torch.randn(M, K, dtype=dtype, device=dev)
            rB = torch.randn(K, N, dtype=dtype, device=dev)
            rC = torch.zeros(M, N, dtype=dtype, device=dev)
            fn, rfn = make_ws_hierarchical(rA, rB, rC)
            ws_fns.append(fn); ws_rfns.append(rfn)

        ws_alone = time_alone(ws_fns, ws_rfns, mm_stream, WARMUP, STEPS)
        ws_alone_med = statistics.median(ws_alone)
        ws_walls, ws_mms, ws_cos = time_overlap(
            ws_fns, ws_rfns, comm_fn, mm_stream, co_stream, WARMUP, STEPS)
        ws_wall_med = statistics.median(ws_walls)
        ws_mm_med = statistics.median(ws_mms)
        ws_penalty = (ws_mm_med - ws_alone_med) / ws_alone_med * 100
        comm_med = statistics.median(ws_cos)

        if rank == 0:
            alone_tf_t = flops / (torch_alone_med * 1e-3) / 1e12
            alone_tf_w = flops / (ws_alone_med * 1e-3) / 1e12
            overlap_tf_t = flops / (torch_mm_med * 1e-3) / 1e12
            overlap_tf_w = flops / (ws_mm_med * 1e-3) / 1e12

            wall_winner = "WS" if ws_wall_med < torch_wall_med else "torch"
            wall_diff = abs(ws_wall_med - torch_wall_med)

            print(f"\n{'='*80}")
            print(f"  {sz}x{sz}x{sz} BF16   RCCL: {COMM_SIZE[0]}x{COMM_SIZE[1]}")
            print(f"{'='*80}")
            print(f"  {'':30s}  {'torch.matmul':>14s}  {'WS Hierarchical':>14s}")
            print(f"  {'GEMM alone (ms)':30s}  {torch_alone_med:>14.3f}  {ws_alone_med:>14.3f}")
            print(f"  {'GEMM alone (TF)':30s}  {alone_tf_t:>14.0f}  {alone_tf_w:>14.0f}")
            print(f"  {'GEMM during overlap (ms)':30s}  {torch_mm_med:>14.3f}  {ws_mm_med:>14.3f}")
            print(f"  {'GEMM during overlap (TF)':30s}  {overlap_tf_t:>14.0f}  {overlap_tf_w:>14.0f}")
            print(f"  {'Overlap penalty':30s}  {torch_penalty:>+13.1f}%  {ws_penalty:>+13.1f}%")
            print(f"  {'Wall-clock (ms)':30s}  {torch_wall_med:>14.3f}  {ws_wall_med:>14.3f}")
            print(f"  {'RCCL alone (ms)':30s}  {comm_med:>14.3f}")
            print(f"  {'Winner':30s}  {wall_winner:>14s}  ({wall_diff:.3f} ms)")

        all_results[str(sz)] = {
            "torch": {
                "alone_ms": torch_alone_med,
                "overlap_gemm_ms": torch_mm_med,
                "wall_ms": torch_wall_med,
                "penalty_pct": torch_penalty,
            },
            "ws_hierarchical": {
                "alone_ms": ws_alone_med,
                "overlap_gemm_ms": ws_mm_med,
                "wall_ms": ws_wall_med,
                "penalty_pct": ws_penalty,
            },
            "comm_ms": comm_med,
        }

    if rank == 0:
        os.makedirs("results/autoresearch", exist_ok=True)
        with open("results/autoresearch/overlap_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"  OVERLAP SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Size':<8s}  {'torch wall':>10s}  {'WS wall':>10s}  {'Winner':>8s}  {'Delta':>10s}")
        for k, r in all_results.items():
            tw = r["torch"]["wall_ms"]; ww = r["ws_hierarchical"]["wall_ms"]
            winner = "WS" if ww < tw else "torch"
            delta = ww - tw
            print(f"  {k:<8s}  {tw:>10.3f}  {ww:>10.3f}  {winner:>8s}  {delta:>+10.3f} ms")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

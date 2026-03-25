#!/usr/bin/env python3
"""
Inflection-point overlap experiment: run RCCL + GEMM overlap with WS
grid-limited to various CU counts, and compare against torch.matmul.

Usage:
  torchrun --nproc_per_node=8 benchmarks/inflection_overlap.py \
      --active-cus 240 256 272 288 304 \
      --gemm-m 12288 --gemm-n 12288 --gemm-k 12288 \
      --comm-size 16384 16384 \
      --nccl-max-nchannels 32 \
      --steps 100

Outputs JSON to results/overlap_data/inflection_overlap.json
"""
import argparse
import json
import os
import statistics
import sys
import time as pytime
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul


def make_hierarchical_matmul_gridlimited(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, active_cus: int,
    use_mask: bool = True, grid_tiles: bool = False,
) -> Tuple[Callable, Callable]:
    """Create WS Hierarchical matmul.

    Args:
        active_cus: Number of active CUs (used for mask when use_mask=True).
        use_mask: When False, USE_MASK constexpr is compiled out entirely.
        grid_tiles: When True, launch grid = total_tiles (not n_cu).
    """
    M, K = A.shape
    _, N = B.shape
    dev = A.device

    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev, streamk=False)
    BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    num_xcds = sel.num_sms
    gsize_m = sel.group_m
    n_cu = sel._N_CU
    even_k = K % BLK_K == 0
    local_per_xcd = total_tiles // num_xcds
    global_tiles = total_tiles - local_per_xcd * num_xcds

    grid_size = total_tiles if grid_tiles else n_cu

    tile_counter = torch.zeros(num_xcds * COUNTER_STRIDE, device=dev, dtype=torch.int32)
    global_counter = torch.zeros(COUNTER_STRIDE, device=dev, dtype=torch.int32)

    if use_mask:
        mask = torch.zeros(n_cu, dtype=torch.int32, device=dev)
        mask[:active_cus] = 1
    else:
        mask = None

    def matmul_fn():
        ws_hierarchical_matmul[(grid_size,)](
            A, B, C, None, None, None,
            tile_counter, global_counter,
            M, N, K, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
            stride_ak=A.stride(1), stride_bk=B.stride(0),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m, NUM_SMS=grid_size, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1,
            USE_MASK=use_mask, mask_ptr=mask,
        )

    def reset_fn():
        tile_counter.zero_()
        global_counter.zero_()

    return matmul_fn, reset_fn


def time_overlap_rotating(
    matmul_fns, reset_fns, comm_fn, matmul_stream, comm_stream,
    n_warmup, n_steps,
) -> Tuple[List[float], List[float], List[float]]:
    """Overlapped GEMM+comm with rotating buffers."""
    n_bufs = len(matmul_fns)

    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]: reset_fns[idx]()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(matmul_stream):
            matmul_fns[idx]()
    torch.cuda.synchronize()

    wall_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    wall_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    co_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    co_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        idx = i % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]: reset_fns[idx]()
        torch.cuda.synchronize()

        matmul_stream.wait_stream(torch.cuda.current_stream())
        comm_stream.wait_stream(torch.cuda.current_stream())

        wall_s[i].record()
        co_s[i].record(comm_stream)

        with torch.cuda.stream(comm_stream):
            comm_fn()

        mm_s[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            torch.cuda._sleep(100_000)
            matmul_fns[idx]()

        mm_e[i].record(matmul_stream)
        co_e[i].record(comm_stream)

        torch.cuda.current_stream().wait_stream(matmul_stream)
        torch.cuda.current_stream().wait_stream(comm_stream)
        wall_e[i].record()

    torch.cuda.synchronize()

    wall = [s.elapsed_time(e) for s, e in zip(wall_s, wall_e)]
    mm = [s.elapsed_time(e) for s, e in zip(mm_s, mm_e)]
    co = [s.elapsed_time(e) for s, e in zip(co_s, co_e)]
    return wall, mm, co


def time_rotating_alone(matmul_fns, reset_fns, matmul_stream, n_warmup, n_steps):
    """GEMM alone with rotating buffers."""
    n_bufs = len(matmul_fns)
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]: reset_fns[idx]()
            matmul_fns[idx]()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        idx = i % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]: reset_fns[idx]()
        torch.cuda.synchronize()
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fns[idx]()
        ends[i].record(matmul_stream)
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-m", type=int, default=12288)
    parser.add_argument("--gemm-n", type=int, default=12288)
    parser.add_argument("--gemm-k", type=int, default=12288)
    parser.add_argument("--comm-size", type=int, nargs="+", default=[16384, 16384])
    parser.add_argument("--nccl-max-nchannels", type=int, default=32)
    parser.add_argument("--active-cus", type=int, nargs="+",
                        default=[192, 208, 224, 240, 256, 272, 288, 304])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str,
                        default="results/overlap_data/inflection_overlap.json")
    args = parser.parse_args()

    os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    M, N, K = args.gemm_m, args.gemm_n, args.gemm_k
    flops = 2.0 * M * N * K
    N_ROT = 4

    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    matmul_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    # ---- Measure comm alone ----
    for _ in range(args.warmup):
        with torch.cuda.stream(comm_stream):
            comm_fn()
    torch.cuda.synchronize()
    c_starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    c_ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    for i in range(args.steps):
        c_starts[i].record(comm_stream)
        with torch.cuda.stream(comm_stream):
            comm_fn()
        c_ends[i].record(comm_stream)
    torch.cuda.synchronize()
    comm_times = [s.elapsed_time(e) for s, e in zip(c_starts, c_ends)]
    comm_med = statistics.median(comm_times)

    if rank == 0:
        print(f"{'='*72}")
        print(f"Inflection Overlap: {M}x{N}x{K} BF16 + all_reduce({args.comm_size})")
        print(f"  NCCL_MAX_NCHANNELS={args.nccl_max_nchannels}  world_size={world_size}")
        print(f"  comm alone: {comm_med:.3f} ms")
        print(f"{'='*72}")

    # ---- torch.matmul baseline (304 CUs, no masking) ----
    torch_rot_fns, torch_rot_rfns = [], []
    for _ in range(N_ROT):
        rA = torch.randn(M, K, dtype=dtype, device=dev)
        rB = torch.randn(K, N, dtype=dtype, device=dev)
        torch_rot_fns.append(lambda a=rA, b=rB: torch.matmul(a, b))
        torch_rot_rfns.append(None)

    # torch alone (rotating)
    torch_alone = time_rotating_alone(
        torch_rot_fns, torch_rot_rfns, matmul_stream, args.warmup, args.steps)
    torch_alone_med = statistics.median(torch_alone)
    torch_alone_tf = flops / (torch_alone_med * 1e-3) / 1e12

    # torch overlap (rotating)
    torch_wall, torch_mm, torch_co = time_overlap_rotating(
        torch_rot_fns, torch_rot_rfns, comm_fn,
        matmul_stream, comm_stream, args.warmup, args.steps)
    torch_wall_med = statistics.median(torch_wall)
    torch_mm_med = statistics.median(torch_mm)
    torch_mm_tf = flops / (torch_mm_med * 1e-3) / 1e12
    torch_penalty = (torch_mm_med - torch_alone_med) / torch_alone_med * 100

    if rank == 0:
        print(f"\n--- torch.matmul (304 CUs) ---")
        print(f"  alone:    {torch_alone_med:.3f} ms  ({torch_alone_tf:.0f} TF)")
        print(f"  overlap:  GEMM={torch_mm_med:.3f} ms ({torch_mm_tf:.0f} TF)  wall={torch_wall_med:.3f} ms")
        print(f"  penalty:  {torch_penalty:+.1f}%")

    # ---- WS Hierarchical at various CU counts ----
    results = {
        "meta": {
            "M": M, "N": N, "K": K, "dtype": "bf16",
            "comm_size": args.comm_size,
            "nccl_max_nchannels": args.nccl_max_nchannels,
            "world_size": world_size,
            "steps": args.steps,
            "timestamp": datetime.now().isoformat(),
        },
        "comm_alone_ms": comm_med,
        "torch": {
            "alone_ms": torch_alone_med, "alone_tf": torch_alone_tf,
            "overlap_gemm_ms": torch_mm_med, "overlap_gemm_tf": torch_mm_tf,
            "overlap_wall_ms": torch_wall_med,
            "penalty_pct": torch_penalty,
            "alone_all": torch_alone, "overlap_gemm_all": torch_mm,
            "overlap_wall_all": torch_wall,
        },
        "ws_hierarchical": {},
        "ws_nomask": {},
    }

    def run_ws_variant(label, active_cus, use_mask, grid_tiles=False):
        """Benchmark one WS config and return results dict."""
        ws_rot_fns, ws_rot_rfns = [], []
        for _ in range(N_ROT):
            rA = torch.randn(M, K, dtype=dtype, device=dev)
            rB = torch.randn(K, N, dtype=dtype, device=dev)
            rC = torch.zeros(M, N, dtype=dtype, device=dev)
            mfn, rfn = make_hierarchical_matmul_gridlimited(
                rA, rB, rC, active_cus, use_mask=use_mask,
                grid_tiles=grid_tiles)
            ws_rot_fns.append(mfn)
            ws_rot_rfns.append(rfn)

        ws_alone = time_rotating_alone(
            ws_rot_fns, ws_rot_rfns, matmul_stream, args.warmup, args.steps)
        ws_alone_med = statistics.median(ws_alone)
        ws_alone_tf = flops / (ws_alone_med * 1e-3) / 1e12

        ws_wall, ws_mm, ws_co = time_overlap_rotating(
            ws_rot_fns, ws_rot_rfns, comm_fn,
            matmul_stream, comm_stream, args.warmup, args.steps)
        ws_wall_med = statistics.median(ws_wall)
        ws_mm_med = statistics.median(ws_mm)
        ws_mm_tf = flops / (ws_mm_med * 1e-3) / 1e12
        ws_penalty = (ws_mm_med - ws_alone_med) / ws_alone_med * 100
        vs_torch = ws_wall_med - torch_wall_med

        if rank == 0:
            tag = " <<< WINS" if vs_torch < 0 else ""
            print(f"  {label:<22s}  {ws_alone_med:>9.3f}  {ws_alone_tf:>9.0f}  "
                  f"{ws_mm_med:>10.3f}  {ws_mm_tf:>8.0f}  {ws_wall_med:>8.3f}  "
                  f"{ws_penalty:>+7.1f}%  {vs_torch:>+13.3f} ms{tag}")

        out = {
            "alone_ms": ws_alone_med, "alone_tf": ws_alone_tf,
            "overlap_gemm_ms": ws_mm_med, "overlap_gemm_tf": ws_mm_tf,
            "overlap_wall_ms": ws_wall_med,
            "penalty_pct": ws_penalty,
            "vs_torch_wall_ms": vs_torch,
            "alone_all": ws_alone, "overlap_gemm_all": ws_mm,
            "overlap_wall_all": ws_wall,
        }
        del ws_rot_fns, ws_rot_rfns
        torch.cuda.empty_cache()
        return out

    if rank == 0:
        print(f"\n--- WS Hierarchical ---")
        print(f"  {'variant':<22s}  {'alone ms':>9s}  {'alone TF':>9s}  "
              f"{'olap GEMM':>10s}  {'olap TF':>8s}  {'wall ms':>8s}  "
              f"{'penalty':>8s}  {'vs torch wall':>14s}")
        print("  " + "-" * 100)

    # Run grid=tiles, no-mask variant (grid = total output tiles)
    sel_tmp = tritonblas.OrigamiMatmulSelector(M, N, K, dtype, dtype, dtype, dev, streamk=False)
    total_tiles = triton.cdiv(M, sel_tmp.block_m) * triton.cdiv(N, sel_tmp.block_n)
    results["ws_grid_tiles"] = {}
    results["ws_grid_tiles"]["all"] = run_ws_variant(
        f"grid={total_tiles} tiles, no-mask", 304, use_mask=False, grid_tiles=True)

    # Run no-mask variant (grid = 304 CUs, no mask)
    results["ws_nomask"]["304"] = run_ws_variant(
        "grid=304 CUs, no-mask", 304, use_mask=False)

    # Then run masked variants for comparison
    for active_cus in args.active_cus:
        results["ws_hierarchical"][str(active_cus)] = run_ws_variant(
            f"grid=304, masked ({active_cus})", active_cus, use_mask=True)

    if rank == 0:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

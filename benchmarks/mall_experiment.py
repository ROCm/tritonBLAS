#!/usr/bin/env python3
"""
MALL/LLC + L2 Counter Experiment for Overlap Analysis

Standalone script for collecting hardware performance counters via rocprofv3
to measure cache behavior during compute-communication overlap on MI300X.

Modes:
  gemm-alone     GEMM in isolation (single GPU, baseline)
  gemm-rotating  GEMM with rotating buffers (cold L2, single GPU)
  gemm-rccl      GEMM + all_reduce overlap (multi-GPU)

Usage with rocprofv3:
  # MALL/HBM counters
  rocprofv3 --pmc MALL_BANDWIDTH_ALL HBM_READ_BYTES HBM_WRITE_BYTES \
      -o out -d results/mall_alone --output-format csv \
      -- python3 benchmarks/mall_experiment.py gemm-alone --m 8192 --n 8192 --k 8192

  # L2 counters
  rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_DRAM_sum TCC_EA0_WRREQ_DRAM_sum \
      -o out -d results/l2_alone --output-format csv \
      -- python3 benchmarks/mall_experiment.py gemm-alone --m 8192 --n 8192 --k 8192

  # RCCL overlap
  rocprofv3 --pmc MALL_BANDWIDTH_ALL HBM_READ_BYTES HBM_WRITE_BYTES \
      -o out -d results/mall_rccl --output-format csv \
      -- torchrun --nproc_per_node=8 benchmarks/mall_experiment.py gemm-rccl
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def gemm_alone(args):
    """GEMM in isolation — warm L2 baseline."""
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16

    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)

    # Warmup
    for _ in range(args.warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.steps):
        torch.matmul(A, B)
    e.record()
    torch.cuda.synchronize()
    print(f"GEMM alone (warm L2): {s.elapsed_time(e) / args.steps:.3f} ms/iter "
          f"({args.m}x{args.n}x{args.k}, {args.steps} iters)")


def gemm_rotating(args):
    """GEMM with rotating buffers — cold L2 per iteration."""
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16

    N_BUFS = args.n_bufs
    As = [torch.randn(args.m, args.k, dtype=dtype, device=dev) for _ in range(N_BUFS)]
    Bs = [torch.randn(args.k, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]

    # Warmup all buffer sets
    for j in range(max(args.warmup, N_BUFS)):
        torch.matmul(As[j % N_BUFS], Bs[j % N_BUFS])
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(args.steps):
        idx = i % N_BUFS
        torch.matmul(As[idx], Bs[idx])
    e.record()
    torch.cuda.synchronize()
    print(f"GEMM rotating ({N_BUFS} bufs, cold L2): {s.elapsed_time(e) / args.steps:.3f} ms/iter "
          f"({args.m}x{args.n}x{args.k}, {args.steps} iters)")


def gemm_rccl(args):
    """GEMM + RCCL all_reduce overlap."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    comm_tensor = torch.randn(args.comm_m, args.comm_n, dtype=dtype, device=dev)

    gemm_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    # Warmup
    for _ in range(args.warmup):
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            torch.matmul(A, B)
        torch.cuda.synchronize()

    # Timed overlap
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.steps):
        torch.cuda.synchronize()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            torch.matmul(A, B)
    e.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"GEMM + RCCL overlap: {s.elapsed_time(e) / args.steps:.3f} ms/iter "
              f"({args.m}x{args.n}x{args.k}, comm={args.comm_m}x{args.comm_n}, "
              f"{world_size} GPUs, {args.steps} iters)")

    dist.destroy_process_group()


def gemm_rccl_rotating(args):
    """GEMM + RCCL all_reduce overlap with rotating buffers."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    N_BUFS = args.n_bufs
    As = [torch.randn(args.m, args.k, dtype=dtype, device=dev) for _ in range(N_BUFS)]
    Bs = [torch.randn(args.k, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]
    comm_tensor = torch.randn(args.comm_m, args.comm_n, dtype=dtype, device=dev)

    gemm_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    # Warmup
    for j in range(max(args.warmup, N_BUFS)):
        idx = j % N_BUFS
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            torch.matmul(As[idx], Bs[idx])
        torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(args.steps):
        idx = i % N_BUFS
        torch.cuda.synchronize()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            torch.matmul(As[idx], Bs[idx])
    e.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"GEMM + RCCL rotating ({N_BUFS} bufs): {s.elapsed_time(e) / args.steps:.3f} ms/iter "
              f"({args.m}x{args.n}x{args.k}, comm={args.comm_m}x{args.comm_n}, "
              f"{world_size} GPUs, {args.steps} iters)")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="MALL/L2 Counter Experiment")
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, fn in [("gemm-alone", gemm_alone), ("gemm-rotating", gemm_rotating),
                     ("gemm-rccl", gemm_rccl), ("gemm-rccl-rotating", gemm_rccl_rotating)]:
        p = sub.add_parser(name)
        p.add_argument("--m", type=int, default=8192)
        p.add_argument("--n", type=int, default=8192)
        p.add_argument("--k", type=int, default=8192)
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--steps", type=int, default=20)
        if "rotating" in name:
            p.add_argument("--n-bufs", type=int, default=4)
        if "rccl" in name:
            p.add_argument("--comm-m", type=int, default=8192)
            p.add_argument("--comm-n", type=int, default=8192)

    args = parser.parse_args()
    dispatch = {
        "gemm-alone": gemm_alone,
        "gemm-rotating": gemm_rotating,
        "gemm-rccl": gemm_rccl,
        "gemm-rccl-rotating": gemm_rccl_rotating,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

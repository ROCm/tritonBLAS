#!/usr/bin/env python3
"""
Minimal L2 cache profiling script.

Runs the WS GEMM kernel alone vs. with concurrent memory-streaming traffic
(simulating RCCL L2 pollution). Designed to be wrapped with rocprof to
collect TCC_HIT / TCC_MISS hardware counters.

Usage:
    # Isolated GEMM (no pollution):
    rocprof --pmc TCC_HIT TCC_MISS TCC_READ TCC_WRITEBACK \
        python benchmarks/profile_l2.py --mode gemm-alone

    # GEMM + background memory traffic:
    rocprof --pmc TCC_HIT TCC_MISS TCC_READ TCC_WRITEBACK \
        python benchmarks/profile_l2.py --mode gemm-polluted --pollution-mb 512

    # GEMM + actual RCCL (needs torchrun):
    rocprof --pmc TCC_HIT TCC_MISS TCC_READ TCC_WRITEBACK \
        torchrun --nproc_per_node=8 benchmarks/profile_l2.py --mode gemm-rccl
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


def _make_ws_matmul(A, B, C):
    M, K = A.shape
    _, N = B.shape
    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, A.dtype, B.dtype, C.dtype, A.device,
        streamk=False,
    )
    cfg = tritonblas.matmul_preamble(selector)

    def matmul_fn():
        tritonblas.matmul_lt(
            A, B, C, selector, cfg,
            enable_streamk=False, work_stealing=True,
        )

    def reset_fn():
        cfg.reset(streamk=False, work_stealing=True)

    return matmul_fn, reset_fn


@torch.jit.script
def _stream_kernel(buf: torch.Tensor, iters: int) -> torch.Tensor:
    """Simple streaming kernel that pollutes L2 by reading/writing a big buffer."""
    for _ in range(iters):
        buf = buf + 1.0
    return buf


def run_gemm_alone(A, B, C, n_warmup, n_steps):
    matmul_fn, reset_fn = _make_ws_matmul(A, B, C)
    # Warmup
    for _ in range(n_warmup):
        reset_fn()
        matmul_fn()
    torch.cuda.synchronize()

    # Timed region
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        reset_fn()
        matmul_fn()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    print(f"GEMM alone: {elapsed / n_steps:.3f} ms/iter ({n_steps} iters)")


def run_gemm_polluted(A, B, C, pollution_mb, n_warmup, n_steps):
    matmul_fn, reset_fn = _make_ws_matmul(A, B, C)
    dev = A.device
    # Allocate a big buffer for L2 pollution (bf16 = 2 bytes per element)
    n_elements = (pollution_mb * 1024 * 1024) // 2
    pollution_buf = torch.randn(n_elements, dtype=torch.bfloat16, device=dev)

    pollution_stream = torch.cuda.Stream(device=dev)
    gemm_stream = torch.cuda.Stream(device=dev)

    # Warmup
    for _ in range(n_warmup):
        reset_fn()
        with torch.cuda.stream(pollution_stream):
            pollution_buf.add_(1.0)
        with torch.cuda.stream(gemm_stream):
            matmul_fn()
        torch.cuda.synchronize()

    # Timed region
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        reset_fn()
        with torch.cuda.stream(pollution_stream):
            # Read + write the entire buffer to thrash L2
            pollution_buf.add_(1.0)
        with torch.cuda.stream(gemm_stream):
            matmul_fn()
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    print(f"GEMM + {pollution_mb}MB pollution: {elapsed / n_steps:.3f} ms/iter ({n_steps} iters)")


def run_gemm_rccl(A, B, C, comm_tensor, n_warmup, n_steps):
    import torch.distributed as dist
    matmul_fn, reset_fn = _make_ws_matmul(A, B, C)
    dev = A.device

    comm_stream = torch.cuda.Stream(device=dev)
    gemm_stream = torch.cuda.Stream(device=dev)

    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    # Warmup
    for _ in range(n_warmup):
        reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            matmul_fn()
        torch.cuda.synchronize()

    # Timed region
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(gemm_stream):
            torch.cuda._sleep(100_000)
            matmul_fn()
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    rank = dist.get_rank()
    if rank == 0:
        print(f"GEMM + RCCL all_reduce: {elapsed / n_steps:.3f} ms/iter ({n_steps} iters)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gemm-alone", "gemm-polluted", "gemm-rccl"],
                   default="gemm-alone")
    p.add_argument("--gemm-m", type=int, default=8192, dest="m")
    p.add_argument("--gemm-n", type=int, default=8192, dest="n")
    p.add_argument("--gemm-k", type=int, default=8192, dest="k")
    p.add_argument("--pollution-mb", type=int, default=512,
                   help="Size of L2-pollution buffer in MB (for gemm-polluted mode)")
    p.add_argument("--comm-size", type=int, nargs="+", default=[16384, 16384])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--nccl-max-nchannels", type=int, default=None)
    args = p.parse_args()

    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)

    if args.mode == "gemm-rccl":
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    C = torch.empty(args.m, args.n, dtype=dtype, device=dev)

    if args.mode == "gemm-alone":
        run_gemm_alone(A, B, C, args.warmup, args.steps)
    elif args.mode == "gemm-polluted":
        run_gemm_polluted(A, B, C, args.pollution_mb, args.warmup, args.steps)
    elif args.mode == "gemm-rccl":
        comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
        run_gemm_rccl(A, B, C, comm_tensor, args.warmup, args.steps)
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

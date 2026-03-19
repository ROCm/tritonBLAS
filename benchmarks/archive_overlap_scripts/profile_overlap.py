#!/usr/bin/env python3
"""
Profile the overlap benchmark with torch.profiler to capture kernel-level timeline.
Exports Chrome trace JSON for rank 0 that shows RCCL + GEMM kernel interleaving.

Usage:
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nproc_per_node=8 benchmarks/profile_overlap.py \
        --matmul-backend ws \
        --gemm-m 8192 --gemm-n 8192 --gemm-k 8192 \
        --comm-size 4096 4096 \
        --nccl-max-nchannels 32 \
        --steps 5
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, schedule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


def _make_tritonblas_matmul(A, B, C, backend):
    M, K = A.shape
    _, N = B.shape
    enable_streamk = backend == "streamk"
    work_stealing = backend in ("ws", "ws-global")
    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, A.dtype, B.dtype, C.dtype, A.device, streamk=enable_streamk)
    cfg = tritonblas.matmul_preamble(selector)
    if backend == "ws-global":
        cfg.global_atomic = True

    def matmul_fn():
        tritonblas.matmul_lt(A, B, C, selector, cfg,
                             enable_streamk=enable_streamk, work_stealing=work_stealing)
    def reset_fn():
        cfg.reset(streamk=enable_streamk, work_stealing=work_stealing)
    return matmul_fn, reset_fn


def _make_torch_matmul(A, B):
    def matmul_fn():
        torch.matmul(A, B)
    def reset_fn():
        pass
    return matmul_fn, reset_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matmul-backend", choices=["ws", "ws-global", "persistent", "torch"],
                   default="ws")
    p.add_argument("--gemm-m", type=int, default=8192, dest="m")
    p.add_argument("--gemm-n", type=int, default=8192, dest="n")
    p.add_argument("--gemm-k", type=int, default=8192, dest="k")
    p.add_argument("--comm-size", type=int, nargs="+", default=[4096, 4096])
    p.add_argument("--nccl-max-nchannels", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--output-dir", default="/tmp/overlap_profile")
    args = p.parse_args()

    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)

    backend = args.matmul_backend
    if backend == "torch":
        matmul_fn, reset_fn = _make_torch_matmul(A, B)
    else:
        matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, backend)

    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    matmul_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    # Warmup
    for _ in range(args.warmup):
        reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
    torch.cuda.synchronize()
    dist.barrier()

    # ---- Phase 1: GEMM alone (profiled) ----
    if rank == 0:
        print(f"Profiling GEMM alone ({backend})...")

    mm_alone_events = []
    for i in range(args.steps):
        reset_fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        e.record(matmul_stream)
        torch.cuda.synchronize()
        mm_alone_events.append((s, e))

    dist.barrier()

    # ---- Phase 2: Overlapped (profiled) ----
    if rank == 0:
        print(f"Profiling overlap ({backend} + all_reduce)...")

    mm_overlap_events = []

    # Only profile rank 0
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        )
        prof.__enter__()

    for i in range(args.steps):
        reset_fn()
        torch.cuda.synchronize()

        matmul_stream.wait_stream(torch.cuda.current_stream())
        comm_stream.wait_stream(torch.cuda.current_stream())

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(matmul_stream)

        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(matmul_stream):
            torch.cuda._sleep(100_000)
            matmul_fn()

        e.record(matmul_stream)
        torch.cuda.synchronize()
        mm_overlap_events.append((s, e))

    if rank == 0:
        prof.__exit__(None, None, None)
        trace_path = os.path.join(args.output_dir, f"trace_{backend}.json")
        prof.export_chrome_trace(trace_path)
        print(f"Trace exported to: {trace_path}")

        # Also export the key table from profiler
        key_avg_path = os.path.join(args.output_dir, f"key_averages_{backend}.txt")
        with open(key_avg_path, "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"Key averages exported to: {key_avg_path}")

    # Print timing summary
    if rank == 0:
        alone_durs = [s.elapsed_time(e) for s, e in mm_alone_events]
        overlap_durs = [s.elapsed_time(e) for s, e in mm_overlap_events]
        print(f"\n{'='*60}")
        print(f"Backend: {backend}")
        print(f"GEMM alone (ms):  " + ", ".join(f"{d:.3f}" for d in alone_durs))
        print(f"  median: {sorted(alone_durs)[len(alone_durs)//2]:.3f} ms")
        print(f"GEMM overlap (ms): " + ", ".join(f"{d:.3f}" for d in overlap_durs))
        print(f"  median: {sorted(overlap_durs)[len(overlap_durs)//2]:.3f} ms")
        alone_med = sorted(alone_durs)[len(alone_durs)//2]
        over_med = sorted(overlap_durs)[len(overlap_durs)//2]
        print(f"  slowdown: {over_med/alone_med:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

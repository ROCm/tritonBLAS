#!/usr/bin/env python3
"""
Capture kernel dispatch traces during overlap with controlled CU pressure.

Two CU-hog modes:
  - ALU: pure compute (no memory pressure) → tests CU scheduling impact
  - MEM: memory-streaming (reads/writes large buffer) → tests HBM bandwidth contention

Usage:
    # WS + ALU hog:
    rocprofv3 --kernel-trace -d /tmp/trace_ws_alu -f csv \
        -- python3 benchmarks/trace_overlap.py --backend ws --hog-mode alu

    # WS + MEM hog:
    rocprofv3 --kernel-trace -d /tmp/trace_ws_mem -f csv \
        -- python3 benchmarks/trace_overlap.py --backend ws --hog-mode mem

    # WS alone:
    rocprofv3 --kernel-trace -d /tmp/trace_ws_alone -f csv \
        -- python3 benchmarks/trace_overlap.py --backend ws --no-overlap
"""
import argparse
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


@triton.jit
def cu_hog_alu_kernel(out_ptr, n_iters, BLOCK: tl.constexpr):
    """Pure ALU CU-hog (no memory pressure)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = (offs + pid).to(tl.float32)
    i = 0
    while i < n_iters:
        acc = acc * 1.00001 + 0.00001
        i += 1
    tl.store(out_ptr + pid * BLOCK + offs, acc)


@triton.jit
def cu_hog_mem_kernel(buf_ptr, n_iters, stride, BLOCK: tl.constexpr):
    """Memory-bound CU-hog: reads/writes a large buffer in a loop."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * stride + offs
    i = 0
    while i < n_iters:
        vals = tl.load(buf_ptr + base)
        vals = vals + 0.001
        tl.store(buf_ptr + base, vals)
        base = base + BLOCK
        wrap_mask = base >= (pid + 1) * stride
        base = tl.where(wrap_mask, pid * stride + offs, base)
        i += 1


def make_matmul(A, B, C, backend):
    M, K = A.shape
    _, N = B.shape
    enable_streamk = False
    work_stealing = backend in ("ws", "ws-global")

    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, A.dtype, B.dtype, C.dtype, A.device,
        streamk=enable_streamk,
    )
    cfg = tritonblas.matmul_preamble(selector)
    if backend == "ws-global":
        cfg.global_atomic = True

    def matmul_fn():
        tritonblas.matmul_lt(A, B, C, selector, cfg,
                             enable_streamk=enable_streamk,
                             work_stealing=work_stealing)

    def reset_fn():
        cfg.reset(streamk=enable_streamk, work_stealing=work_stealing)

    return matmul_fn, reset_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ws", "ws-global", "persistent"],
                   default="ws")
    p.add_argument("--no-overlap", action="store_true")
    p.add_argument("--hog-mode", choices=["alu", "mem"], default="alu",
                   help="CU-hog type: alu (pure compute) or mem (memory streaming)")
    p.add_argument("--hog-wgs", type=int, default=32)
    p.add_argument("--hog-alu-iters", type=int, default=100_000,
                   help="Iterations for ALU hog (~2ms on MI300X)")
    p.add_argument("--hog-mem-iters", type=int, default=9_000,
                   help="Iterations for MEM hog (~2.3ms on MI300X)")
    p.add_argument("--gemm-m", type=int, default=8192)
    p.add_argument("--gemm-n", type=int, default=8192)
    p.add_argument("--gemm-k", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16

    A = torch.randn(args.gemm_m, args.gemm_k, dtype=dtype, device=dev)
    B = torch.randn(args.gemm_k, args.gemm_n, dtype=dtype, device=dev)
    C = torch.empty(args.gemm_m, args.gemm_n, dtype=dtype, device=dev)

    HOG_BLOCK = 256
    hog_alu_out = torch.empty(args.hog_wgs * HOG_BLOCK, dtype=torch.float32, device=dev)
    # 1MB per WG for memory hog
    MEM_STRIDE = 512 * 1024
    hog_mem_buf = torch.randn(args.hog_wgs * MEM_STRIDE, dtype=torch.bfloat16, device=dev)

    def launch_hog(stream):
        with torch.cuda.stream(stream):
            if args.hog_mode == "alu":
                cu_hog_alu_kernel[(args.hog_wgs,)](
                    hog_alu_out, args.hog_alu_iters, BLOCK=HOG_BLOCK)
            else:
                cu_hog_mem_kernel[(args.hog_wgs,)](
                    hog_mem_buf, args.hog_mem_iters, MEM_STRIDE, BLOCK=HOG_BLOCK)

    matmul_fn, reset_fn = make_matmul(A, B, C, args.backend)
    gemm_stream = torch.cuda.Stream(device=dev)
    hog_stream = torch.cuda.Stream(device=dev)

    # Warmup
    for _ in range(args.warmup):
        reset_fn()
        matmul_fn()
    launch_hog(hog_stream)
    torch.cuda.synchronize()

    # Also time with CUDA events for comparison
    mm_events = []

    if args.no_overlap:
        for i in range(args.steps):
            reset_fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(gemm_stream)
            with torch.cuda.stream(gemm_stream):
                matmul_fn()
            e.record(gemm_stream)
            torch.cuda.synchronize()
            mm_events.append((s, e))
    else:
        for i in range(args.steps):
            reset_fn()
            torch.cuda.synchronize()
            # Launch hog first
            launch_hog(hog_stream)
            # Then GEMM with small sleep
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(gemm_stream)
            with torch.cuda.stream(gemm_stream):
                torch.cuda._sleep(100_000)
                matmul_fn()
            e.record(gemm_stream)
            torch.cuda.synchronize()
            mm_events.append((s, e))

    # Print CUDA event timings
    durs = [s.elapsed_time(e) for s, e in mm_events]
    mode_str = "alone" if args.no_overlap else f"overlap-{args.hog_mode}"
    print(f"\n{args.backend} ({mode_str}): CUDA event GEMM durations (ms):")
    for i, d in enumerate(durs):
        print(f"  iter {i}: {d:.3f} ms")
    avg = sum(durs) / len(durs)
    print(f"  avg: {avg:.3f} ms")


if __name__ == "__main__":
    main()

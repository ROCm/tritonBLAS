#!/usr/bin/env python3
"""
Compute–Communication Overlap Benchmark

Measures how well GEMM and NCCL collectives overlap on AMD MI300X.

Supports four matmul backends:
    torch       torch.matmul (hipBLASLt)
    persistent  tritonBLAS persistent GEMM
    streamk     tritonBLAS Stream-K GEMM
    ws          tritonBLAS work-stealing GEMM

Per-iteration CUDA-event timing produces min / max / mean / median
statistics for each of the three phases:
    1. GEMM alone
    2. Collective alone
    3. GEMM + Collective overlapped

Usage (torchrun):
    torchrun --nproc_per_node=8 benchmarks/overlap.py \
        --matmul-backend ws \
        --m 8192 --n 8192 --k 8192 \
        --comm-size 8192 8192 \
        --collective all_reduce \
        --nccl-max-nchannels 16 \
        --steps 200 \
        --output-csv overlap_results.csv
"""
import argparse
import csv
import os
import statistics
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-ALU CU-hog kernel (near-zero memory footprint)
# ---------------------------------------------------------------------------

@triton.jit
def _cu_hog_alu_kernel(out_ptr, n_iters, BLOCK: tl.constexpr):
    """Pure ALU CU-hog — keeps CUs busy with FMA but touches no L2.

    Only memory access: one BLOCK-wide store at the end (~1 KB per WG).
    Total footprint for 32 WGs = 32 KB — negligible vs 32 MB L2 per XCD.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = (offs + pid).to(tl.float32)
    i = 0
    while i < n_iters:
        acc = acc * 1.00001 + 0.00001
        i += 1
    tl.store(out_ptr + pid * BLOCK + offs, acc)


# ---------------------------------------------------------------------------
# Matmul back-end helpers
# ---------------------------------------------------------------------------

def _make_tritonblas_matmul(A, B, C, backend):
    """Return (matmul_fn, reset_fn) for a tritonBLAS kernel variant."""
    M, K = A.shape
    _, N = B.shape

    enable_streamk = backend == "streamk"
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


def _make_torch_matmul(A, B):
    """Return (matmul_fn, reset_fn) for torch.matmul."""
    def matmul_fn():
        torch.matmul(A, B)

    def reset_fn():
        pass

    return matmul_fn, reset_fn


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------

COLLECTIVES = {
    "all_reduce": lambda t, **_: dist.all_reduce(t),
    "all_gather": lambda t, **kw: dist.all_gather_into_tensor(kw["out"], t),
    "all_to_all": lambda t, **kw: dist.all_to_all(kw["out_list"], kw["in_list"]),
}


def _make_collective(name, comm_tensor, world_size):
    """Return a callable that performs the chosen collective."""
    if name == "all_gather":
        out = torch.empty(
            world_size, *comm_tensor.shape,
            dtype=comm_tensor.dtype, device=comm_tensor.device,
        ).flatten(0, 1) if comm_tensor.dim() == 1 else torch.empty(
            comm_tensor.shape[0] * world_size, *comm_tensor.shape[1:],
            dtype=comm_tensor.dtype, device=comm_tensor.device,
        )
        return lambda: COLLECTIVES[name](comm_tensor, out=out)
    elif name == "all_to_all":
        in_list = [comm_tensor.clone() for _ in range(world_size)]
        out_list = [torch.empty_like(comm_tensor) for _ in range(world_size)]
        return lambda: COLLECTIVES[name](comm_tensor, out_list=out_list, in_list=in_list)
    else:
        return lambda: COLLECTIVES[name](comm_tensor)


# ---------------------------------------------------------------------------
# Per-iteration timing utilities
# ---------------------------------------------------------------------------

def _time_per_iter(fn, stream, n_warmup, n_steps, reset_fn=None):
    """Return a list of per-iteration times (ms) on *stream*."""
    # Warmup
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(stream):
            fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        torch.cuda.synchronize()
        starts[i].record(stream)
        with torch.cuda.stream(stream):
            fn()
        ends[i].record(stream)
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_serial(matmul_fn, comm_fn, matmul_stream, comm_stream,
                 n_warmup, n_steps, reset_fn=None):
    """Run NCCL then GEMM serially (zero overlap). Returns GEMM times."""
    # Warmup
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        torch.cuda.synchronize()
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        # Run NCCL first, wait for it to COMPLETELY finish
        with torch.cuda.stream(comm_stream):
            comm_fn()
        torch.cuda.synchronize()
        # Now run GEMM (NCCL is 100% done, zero temporal overlap)
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        ends[i].record(matmul_stream)
        torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_l2_flush(matmul_fn, matmul_stream, n_warmup, n_steps, reset_fn=None):
    """Run GEMM with L2 cache flush between iterations (no NCCL).
    
    Reads a 512MB buffer between iterations to evict GEMM data from L2.
    This isolates the L2 pollution effect from any NCCL-specific behavior.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    # 512 MB buffer (16x larger than each XCD's 32MB L2)
    n_elems = 512 * 1024 * 1024 // 2  # bf16 elements
    flush_buf = torch.randn(n_elems, dtype=torch.bfloat16, device=dev)
    flush_stream = torch.cuda.Stream(device=dev)

    # Warmup
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        # Flush L2 by reading the entire 512MB buffer
        with torch.cuda.stream(flush_stream):
            flush_buf.add_(0.001)
        torch.cuda.synchronize()
        # Now run GEMM with cold L2
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        ends[i].record(matmul_stream)
        torch.cuda.synchronize()

    del flush_buf
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_alu_hog(matmul_fn, matmul_stream, n_warmup, n_steps, reset_fn=None,
                  hog_wgs=32, hog_iters=100_000):
    """Run pure-ALU CU-hog serially BEFORE each GEMM iteration.

    The CU-hog keeps 'hog_wgs' CUs busy for ~2 ms doing pure FMA work,
    but its total memory footprint is only hog_wgs * 256 * 4 = ~32 KB.
    This CANNOT meaningfully pollute L2 (32 MB per XCD).

    If GEMM time ≈ gemm_alone  → L2 pollution is the root cause.
    If GEMM time ≈ l2flush     → something else (TLB, GPU state, etc.).
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    hog_block = 256
    hog_out = torch.empty(hog_wgs * hog_block, dtype=torch.float32, device=dev)
    hog_stream = torch.cuda.Stream(device=dev)

    # Warmup hog + GEMM
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(hog_stream):
            _cu_hog_alu_kernel[(hog_wgs,)](hog_out, hog_iters, BLOCK=hog_block)
        torch.cuda.synchronize()
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        # Run ALU CU-hog and wait for it to finish
        with torch.cuda.stream(hog_stream):
            _cu_hog_alu_kernel[(hog_wgs,)](hog_out, hog_iters, BLOCK=hog_block)
        torch.cuda.synchronize()
        # Now run GEMM — L2 should still be warm from previous GEMM
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        ends[i].record(matmul_stream)
        torch.cuda.synchronize()

    del hog_out
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_rotating(matmul_fns, reset_fns, matmul_stream, n_warmup, n_steps):
    """GEMM alone with rotating input buffers — naturally cold L2 each iteration.

    Cycles through *n_bufs* independent (A, B, C) sets so that each GEMM
    iteration operates on data that is NOT resident in L2 from the previous
    iteration.  This gives a realistic single-dispatch GEMM latency without
    any artificial cache-flush kernel.
    """
    n_bufs = len(matmul_fns)

    # Warmup every buffer set
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        with torch.cuda.stream(matmul_stream):
            matmul_fns[idx]()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        idx = i % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        torch.cuda.synchronize()
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fns[idx]()
        ends[i].record(matmul_stream)
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_overlap(matmul_fn, comm_fn, matmul_stream, comm_stream,
                  n_warmup, n_steps, reset_fn=None):
    """Return per-iteration (wall, matmul, comm) times for overlapped runs."""
    import time as _time

    # Warmup — schedule comm first so RCCL gets CUs before GEMM
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
    torch.cuda.synchronize()

    wall_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    wall_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    co_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    co_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    # Host-side timing for matmul_fn dispatch
    host_mm_times = []
    host_comm_times = []

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        torch.cuda.synchronize()

        # Ensure both streams wait for the default stream
        matmul_stream.wait_stream(torch.cuda.current_stream())
        comm_stream.wait_stream(torch.cuda.current_stream())

        wall_s[i].record()
        co_s[i].record(comm_stream)

        # Schedule comm first so RCCL acquires CUs before GEMM launches
        h_comm_s = _time.perf_counter_ns()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        h_comm_e = _time.perf_counter_ns()
        host_comm_times.append((h_comm_e - h_comm_s) / 1e6)  # ms

        # Record mm_s AFTER comm submission so we don't capture
        # host-side RCCL dispatch overhead in the GEMM measurement.
        mm_s[i].record(matmul_stream)
        h_mm_s = _time.perf_counter_ns()
        with torch.cuda.stream(matmul_stream):
            # GPU-side sleep (~100 us) to let comm kernels get scheduled first
            torch.cuda._sleep(100_000)
            matmul_fn()
        h_mm_e = _time.perf_counter_ns()
        host_mm_times.append((h_mm_e - h_mm_s) / 1e6)  # ms

        mm_e[i].record(matmul_stream)
        co_e[i].record(comm_stream)

        # Wait for both streams to finish before recording wall end
        torch.cuda.current_stream().wait_stream(matmul_stream)
        torch.cuda.current_stream().wait_stream(comm_stream)
        wall_e[i].record()

    torch.cuda.synchronize()

    wall = [s.elapsed_time(e) for s, e in zip(wall_s, wall_e)]
    mm = [s.elapsed_time(e) for s, e in zip(mm_s, mm_e)]
    co = [s.elapsed_time(e) for s, e in zip(co_s, co_e)]
    return wall, mm, co, host_mm_times, host_comm_times


def _time_overlap_rotating(matmul_fns, reset_fns, comm_fn,
                           matmul_stream, comm_stream, n_warmup, n_steps):
    """Overlapped GEMM+comm with rotating GEMM buffers (cold L2 each iter).

    Identical scheduling to _time_overlap but cycles through N independent
    (A, B, C) buffer sets so each GEMM iteration sees cold L2.
    """
    import time as _time
    n_bufs = len(matmul_fns)

    # Warmup
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
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
        if reset_fns[idx]:
            reset_fns[idx]()
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


def _stats(times):
    """Return dict with min / max / mean / median over a list of times."""
    return {
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(
    matmul_fn,
    comm_fn,
    matmul_stream,
    comm_stream,
    reset_fn=None,
    rotating_matmul_fns=None,
    rotating_reset_fns=None,
    n_warmup=10,
    n_steps=200,
):
    """Run isolated + overlapped benchmarks and return per-iteration stats."""
    torch.cuda.empty_cache()

    # Phase 1 – GEMM alone (warm L2 — same buffer every iteration)
    gemm_times = _time_per_iter(matmul_fn, matmul_stream, n_warmup, n_steps,
                                reset_fn=reset_fn)
    # Phase 2 – Collective alone
    comm_times = _time_per_iter(comm_fn, comm_stream, n_warmup, n_steps)
    # Phase 3 – GEMM with rotating buffers (naturally cold L2)
    if rotating_matmul_fns is not None:
        rotating_gemm_times = _time_rotating(
            rotating_matmul_fns, rotating_reset_fns,
            matmul_stream, n_warmup, n_steps,
        )
    else:
        rotating_gemm_times = None
    # Phase 4 – Serial (NCCL then GEMM, zero overlap)
    serial_gemm_times = _time_serial(
        matmul_fn, comm_fn, matmul_stream, comm_stream,
        n_warmup, n_steps, reset_fn=reset_fn,
    )
    # Phase 5 – Overlapped (warm L2 — single buffer)
    result = _time_overlap(
        matmul_fn, comm_fn, matmul_stream, comm_stream,
        n_warmup, n_steps, reset_fn=reset_fn,
    )
    wall_times, overlap_gemm_times, overlap_comm_times = result[0], result[1], result[2]
    host_mm_times = result[3] if len(result) > 3 else []
    host_comm_times = result[4] if len(result) > 4 else []

    # Phase 6 – Overlapped with rotating buffers (cold L2)
    if rotating_matmul_fns is not None:
        rot_result = _time_overlap_rotating(
            rotating_matmul_fns, rotating_reset_fns, comm_fn,
            matmul_stream, comm_stream, n_warmup, n_steps,
        )
        rot_wall, rot_mm, rot_co = rot_result
    else:
        rot_wall = rot_mm = rot_co = None

    ret = {
        "gemm_alone": _stats(gemm_times),
        "comm_alone": _stats(comm_times),
        "serial_gemm": _stats(serial_gemm_times),
        "overlap_wall": _stats(wall_times),
        "overlap_gemm": _stats(overlap_gemm_times),
        "overlap_comm": _stats(overlap_comm_times),
        "_raw": {
            "gemm_alone": gemm_times,
            "comm_alone": comm_times,
            "overlap_wall": wall_times,
            "overlap_gemm": overlap_gemm_times,
            "overlap_comm": overlap_comm_times,
        },
    }
    if rotating_gemm_times is not None:
        ret["rotating_gemm"] = _stats(rotating_gemm_times)
        ret["_raw"]["rotating_gemm"] = rotating_gemm_times
    if rot_mm is not None:
        ret["overlap_rot_wall"] = _stats(rot_wall)
        ret["overlap_rot_gemm"] = _stats(rot_mm)
        ret["overlap_rot_comm"] = _stats(rot_co)
        ret["_raw"]["overlap_rot_wall"] = rot_wall
        ret["_raw"]["overlap_rot_gemm"] = rot_mm
        ret["_raw"]["overlap_rot_comm"] = rot_co
    if host_mm_times:
        ret["host_mm_dispatch"] = _stats(host_mm_times)
    if host_comm_times:
        ret["host_comm_dispatch"] = _stats(host_comm_times)
    return ret


# ---------------------------------------------------------------------------
# CLI / entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute–Communication Overlap Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Matrix sizes
    p.add_argument("--gemm-m", type=int, default=8192, dest="m")
    p.add_argument("--gemm-n", type=int, default=8192, dest="n")
    p.add_argument("--gemm-k", type=int, default=8192, dest="k")
    p.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp16"],
                    help="GEMM data type")
    # Communication
    p.add_argument("--comm-size", type=int, nargs="+", default=[8192, 8192],
                    help="Collective tensor shape, e.g. --comm-size 8192 8192")
    p.add_argument("--collective", type=str, default="all_reduce",
                    choices=list(COLLECTIVES.keys()))
    # Matmul backend
    p.add_argument("--matmul-backend", type=str, default="torch",
                    choices=["torch", "persistent", "streamk", "ws", "ws-global"],
                    help="GEMM kernel: torch | persistent | streamk | ws | ws-global")
    # NCCL tuning
    p.add_argument("--nccl-max-nchannels", type=int, default=None,
                    help="Set NCCL_MAX_NCHANNELS env var before benchmark "
                         "(controls how many CUs RCCL uses)")
    # Benchmark parameters
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    # Output
    p.add_argument("--output-csv", type=str, default=None,
                    help="Append results to this CSV file")
    return p.parse_args()


def _print_phase(name, stats):
    """Pretty-print one phase's statistics."""
    print(f"  {name:25s}  "
          f"min={stats['min']:8.3f}  mean={stats['mean']:8.3f}  "
          f"median={stats['median']:8.3f}  max={stats['max']:8.3f}  (ms)")


def main():
    args = parse_args()

    # ---- NCCL channel control ----
    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)

    # ---- Distributed init ----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Allocate tensors ----
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = torch.device("cuda", local_rank)

    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)

    # ---- Build collective callable (shared across backends) ----
    comm_fn = _make_collective(args.collective, comm_tensor, world_size)

    # ---- Streams ----
    matmul_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    nccl_ch = os.environ.get("NCCL_MAX_NCHANNELS", "unset")

    # ---- Build matmul callable (primary — single buffer, warm L2) ----
    if args.matmul_backend == "torch":
        matmul_fn, reset_fn = _make_torch_matmul(A, B)
    else:
        C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
        matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, args.matmul_backend)

    # ---- Build rotating buffer sets (cold L2 — different data each iter) ----
    # 4 buffer sets: each 8K×8K bf16 GEMM uses ~384 MB (A+B+C).
    # After 1 intervening GEMM on a different set, the previous set's data
    # is evicted from the 32 MB per-XCD L2.
    N_ROTATING = 4
    rot_matmul_fns = []
    rot_reset_fns = []
    for _ in range(N_ROTATING):
        rA = torch.randn(args.m, args.k, dtype=dtype, device=dev)
        rB = torch.randn(args.k, args.n, dtype=dtype, device=dev)
        if args.matmul_backend == "torch":
            mfn, rfn = _make_torch_matmul(rA, rB)
        else:
            rC = torch.empty(args.m, args.n, dtype=dtype, device=dev)
            mfn, rfn = _make_tritonblas_matmul(rA, rB, rC, args.matmul_backend)
        rot_matmul_fns.append(mfn)
        rot_reset_fns.append(rfn)

    # ---- Run ----
    results = benchmark(
        matmul_fn, comm_fn,
        matmul_stream, comm_stream,
        reset_fn=reset_fn,
        rotating_matmul_fns=rot_matmul_fns,
        rotating_reset_fns=rot_reset_fns,
        n_warmup=args.warmup,
        n_steps=args.steps,
    )

    # ---- Report (rank 0 only) ----
    if rank == 0:
        print(f"\n{'=' * 72}")
        print(f"Overlap Benchmark — {args.matmul_backend} GEMM "
              f"({args.m}x{args.n}x{args.k} {args.dtype}) "
              f"+ {args.collective} (comm {args.comm_size})")
        print(f"  NCCL_MAX_NCHANNELS = {nccl_ch}   |   "
              f"warmup = {args.warmup}   steps = {args.steps}")
        print(f"{'=' * 72}")

        _print_phase("GEMM alone (warm L2)", results["gemm_alone"])
        _print_phase("Comm alone", results["comm_alone"])
        if "rotating_gemm" in results:
            _print_phase("GEMM rotating bufs", results["rotating_gemm"])
        _print_phase("Serial (GEMM after NCCL)", results["serial_gemm"])
        _print_phase("Overlap (wall)", results["overlap_wall"])
        _print_phase("Overlap (GEMM)", results["overlap_gemm"])
        _print_phase("Overlap (Comm)", results["overlap_comm"])
        if "overlap_rot_gemm" in results:
            _print_phase("Overlap rot (wall)", results["overlap_rot_wall"])
            _print_phase("Overlap rot (GEMM)", results["overlap_rot_gemm"])
            _print_phase("Overlap rot (Comm)", results["overlap_rot_comm"])

        # Overlap efficiency
        ideal = max(results["gemm_alone"]["mean"],
                    results["comm_alone"]["mean"])
        actual = results["overlap_wall"]["mean"]
        efficiency = ideal / actual * 100 if actual > 0 else 0
        slowdown_gemm = (results["overlap_gemm"]["mean"]
                         / results["gemm_alone"]["mean"])
        slowdown_comm = (results["overlap_comm"]["mean"]
                         / results["comm_alone"]["mean"])

        print(f"\n  Overlap efficiency:        {efficiency:.1f}%  "
              f"(ideal {ideal:.3f} ms, actual {actual:.3f} ms)")
        print(f"  GEMM slowdown (overlap):   {slowdown_gemm:.2f}x")
        print(f"  Comm slowdown (overlap):   {slowdown_comm:.2f}x")

        # Host-side dispatch timing
        if "host_mm_dispatch" in results:
            hm = results["host_mm_dispatch"]
            hc = results["host_comm_dispatch"]
            print(f"\n  Host matmul dispatch       min={hm['min']:>7.3f}  mean={hm['mean']:>7.3f}  median={hm['median']:>7.3f}  max={hm['max']:>7.3f}  (ms)")
            print(f"  Host comm dispatch         min={hc['min']:>7.3f}  mean={hc['mean']:>7.3f}  median={hc['median']:>7.3f}  max={hc['max']:>7.3f}  (ms)")
        print()

        # ---- CSV export ----
        if args.output_csv:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "matmul_backend": args.matmul_backend,
                "m": args.m, "n": args.n, "k": args.k,
                "dtype": args.dtype,
                "collective": args.collective,
                "comm_shape": "x".join(str(s) for s in args.comm_size),
                "nccl_max_nchannels": nccl_ch,
                "warmup": args.warmup,
                "steps": args.steps,
                "world_size": world_size,
            }
            # Flatten stats into the row
            for phase in ["gemm_alone", "comm_alone", "overlap_wall",
                          "overlap_gemm", "overlap_comm"]:
                for stat, val in results[phase].items():
                    row[f"{phase}_{stat}"] = f"{val:.4f}"
            row["overlap_efficiency_pct"] = f"{efficiency:.1f}"
            row["gemm_slowdown"] = f"{slowdown_gemm:.3f}"
            row["comm_slowdown"] = f"{slowdown_comm:.3f}"

            file_exists = os.path.exists(args.output_csv)
            with open(args.output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f"Results appended to {args.output_csv}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

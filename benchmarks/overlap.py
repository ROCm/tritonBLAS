#!/usr/bin/env python3
"""
Compute–Communication Overlap Benchmark Suite

A unified tool for measuring and analyzing GEMM-communication overlap on AMD MI300X.

Modes:
  standard       Basic overlap measurement (GEMM alone, Comm alone, Overlapped)
  l2-profile     L2 cache profiling (wrap with rocprof for hardware counters)
  se-sweep       SE oversubscription shape sweep
  trace          Kernel trace capture with CU-hog (wrap with rocprofv3)
  chrome-trace   Chrome trace profiling with torch.profiler
  calibrate-hog  Calibrate CU-hog kernel durations
  grid-sweep     Grid size sweep for work-stealing (single GPU)

Examples:
  # Standard overlap measurement with rotating buffers
  torchrun --nproc_per_node=8 overlap.py standard \\
      --backend ws --m 8192 --n 8192 --k 8192 \\
      --comm-size 8192 8192 --collective all_reduce \\
      --steps 200 --output-csv results.csv

  # L2 cache profiling with rocprof
  rocprof --pmc TCC_HIT TCC_MISS torchrun --nproc_per_node=8 overlap.py l2-profile \\
      --profile-mode gemm-rccl --backend ws

  # SE oversubscription sweep
  torchrun --nproc_per_node=8 overlap.py se-sweep \\
      --backends ws persistent torch --shapes-preset all

See --help for each mode for detailed options.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time as pytime
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


# ==============================================================================
# CU-Hog Kernels
# ==============================================================================

@triton.jit
def _cu_hog_alu_kernel(out_ptr, n_iters, BLOCK: tl.constexpr):
    """Pure ALU CU-hog — keeps CUs busy with FMA but minimal memory footprint.
    
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


@triton.jit
def _cu_hog_mem_kernel(buf_ptr, n_iters, stride, BLOCK: tl.constexpr):
    """Memory-bound CU-hog: reads/writes a large buffer in a loop.
    
    Each WG streams through 'stride' elements per iteration.
    """
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


# ==============================================================================
# Matmul Backend Factories
# ==============================================================================

def _make_tritonblas_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    backend: str,
    total_cus: Optional[int] = None,
) -> Tuple[Callable, Callable]:
    """Return (matmul_fn, reset_fn) for a tritonBLAS kernel variant.
    
    Args:
        A, B, C: Input/output tensors
        backend: "persistent" | "streamk" | "ws" | "ws-global"
        total_cus: Override total CU count (for grid-sweep)
    
    Returns:
        (matmul_fn, reset_fn) callables
    """
    M, K = A.shape
    _, N = B.shape

    enable_streamk = backend == "streamk"
    work_stealing = backend in ("ws", "ws-global")

    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, A.dtype, B.dtype, C.dtype, A.device,
        streamk=enable_streamk,
        total_cus=total_cus,
    )
    cfg = tritonblas.matmul_preamble(selector)
    if backend == "ws-global":
        cfg.global_atomic = True

    def matmul_fn():
        tritonblas.matmul_lt(
            A, B, C, selector, cfg,
            enable_streamk=enable_streamk,
            work_stealing=work_stealing,
        )

    def reset_fn():
        cfg.reset(streamk=enable_streamk, work_stealing=work_stealing)

    return matmul_fn, reset_fn


def _make_torch_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> Tuple[Callable, Callable]:
    """Return (matmul_fn, reset_fn) for torch.matmul.
    
    Args:
        A, B: Input tensors
    
    Returns:
        (matmul_fn, reset_fn) callables
    """
    def matmul_fn():
        torch.matmul(A, B)

    def reset_fn():
        pass

    return matmul_fn, reset_fn


# ==============================================================================
# Collective Helpers
# ==============================================================================

COLLECTIVES = {
    "all_reduce": lambda t, **_: dist.all_reduce(t),
    "all_gather": lambda t, **kw: dist.all_gather_into_tensor(kw["out"], t),
    "all_to_all": lambda t, **kw: dist.all_to_all(kw["out_list"], kw["in_list"]),
}


def _make_collective(name: str, comm_tensor: torch.Tensor, world_size: int) -> Callable:
    """Return a callable that performs the chosen collective.
    
    Args:
        name: "all_reduce" | "all_gather" | "all_to_all"
        comm_tensor: Communication tensor
        world_size: Number of ranks
    
    Returns:
        Callable that performs the collective
    """
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


# ==============================================================================
# Timing Utilities
# ==============================================================================

def _time_per_iter(
    fn: Callable,
    stream: torch.cuda.Stream,
    n_warmup: int,
    n_steps: int,
    reset_fn: Optional[Callable] = None,
) -> List[float]:
    """Return a list of per-iteration times (ms) on *stream*.
    
    Args:
        fn: Function to time
        stream: CUDA stream
        n_warmup: Warmup iterations
        n_steps: Timed iterations
        reset_fn: Optional reset function called before each iteration
    
    Returns:
        List of per-iteration times in ms
    """
    # Warmup
    for _ in range(n_warmup):
        with torch.cuda.stream(stream):
            if reset_fn:
                reset_fn()
            fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        with torch.cuda.stream(stream):
            if reset_fn:
                reset_fn()
        torch.cuda.synchronize()
        starts[i].record(stream)
        with torch.cuda.stream(stream):
            fn()
        ends[i].record(stream)
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_serial(
    matmul_fn: Callable,
    comm_fn: Callable,
    matmul_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    n_warmup: int,
    n_steps: int,
    reset_fn: Optional[Callable] = None,
) -> List[float]:
    """Run NCCL then GEMM serially (zero overlap). Returns GEMM times.
    
    Args:
        matmul_fn: GEMM function
        comm_fn: Communication function
        matmul_stream: GEMM stream
        comm_stream: Communication stream
        n_warmup: Warmup iterations
        n_steps: Timed iterations
        reset_fn: Optional GEMM reset function
    
    Returns:
        List of GEMM times in ms (after NCCL completes)
    """
    # Warmup
    for _ in range(n_warmup):
        with torch.cuda.stream(matmul_stream):
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
        with torch.cuda.stream(matmul_stream):
            if reset_fn:
                reset_fn()
        with torch.cuda.stream(comm_stream):
            comm_fn()
        torch.cuda.synchronize()
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fn()
        ends[i].record(matmul_stream)
        torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_rotating(
    matmul_fns: List[Callable],
    reset_fns: List[Optional[Callable]],
    matmul_stream: torch.cuda.Stream,
    n_warmup: int,
    n_steps: int,
) -> List[float]:
    """GEMM alone with rotating input buffers — naturally cold L2 each iteration.
    
    Cycles through *n_bufs* independent (A, B, C) sets so that each GEMM
    iteration operates on data that is NOT resident in L2 from the previous
    iteration.  This gives a realistic single-dispatch GEMM latency without
    any artificial cache-flush kernel.
    
    Args:
        matmul_fns: List of matmul callables (one per buffer set)
        reset_fns: List of reset callables (one per buffer set)
        matmul_stream: CUDA stream
        n_warmup: Warmup iterations
        n_steps: Timed iterations
    
    Returns:
        List of per-iteration times in ms
    """
    n_bufs = len(matmul_fns)

    # Warmup every buffer set
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]:
                reset_fns[idx]()
            matmul_fns[idx]()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        idx = i % n_bufs
        with torch.cuda.stream(matmul_stream):
            if reset_fns[idx]:
                reset_fns[idx]()
        torch.cuda.synchronize()
        starts[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            matmul_fns[idx]()
        ends[i].record(matmul_stream)
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_overlap(
    matmul_fn: Callable,
    comm_fn: Callable,
    matmul_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    n_warmup: int,
    n_steps: int,
    reset_fn: Optional[Callable] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Return per-iteration (wall, matmul, comm) times for overlapped runs.
    
    Args:
        matmul_fn: GEMM function
        comm_fn: Communication function
        matmul_stream: GEMM stream
        comm_stream: Communication stream
        n_warmup: Warmup iterations
        n_steps: Timed iterations
        reset_fn: Optional GEMM reset function
    
    Returns:
        (wall_times, matmul_times, comm_times) in ms
    """
    # Warmup — schedule comm first so RCCL gets CUs before GEMM
    for _ in range(n_warmup):
        with torch.cuda.stream(matmul_stream):
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

    for i in range(n_steps):
        with torch.cuda.stream(matmul_stream):
            if reset_fn:
                reset_fn()
        torch.cuda.synchronize()

        matmul_stream.wait_stream(torch.cuda.current_stream())
        comm_stream.wait_stream(torch.cuda.current_stream())

        wall_s[i].record()
        co_s[i].record(comm_stream)

        # Schedule comm first so RCCL acquires CUs before GEMM launches
        with torch.cuda.stream(comm_stream):
            comm_fn()

        mm_s[i].record(matmul_stream)
        with torch.cuda.stream(matmul_stream):
            # GPU-side sleep (~100 us) to let comm kernels get scheduled first
            torch.cuda._sleep(100_000)
            matmul_fn()

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
    return wall, mm, co


def _time_overlap_rotating(
    matmul_fns: List[Callable],
    reset_fns: List[Optional[Callable]],
    comm_fn: Callable,
    matmul_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    n_warmup: int,
    n_steps: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Overlapped GEMM+comm with rotating GEMM buffers (cold L2 each iter).
    
    Identical scheduling to _time_overlap but cycles through N independent
    (A, B, C) buffer sets so each GEMM iteration sees cold L2.
    
    Args:
        matmul_fns: List of matmul callables (one per buffer set)
        reset_fns: List of reset callables (one per buffer set)
        comm_fn: Communication function
        matmul_stream: GEMM stream
        comm_stream: Communication stream
        n_warmup: Warmup iterations
        n_steps: Timed iterations
    
    Returns:
        (wall_times, matmul_times, comm_times) in ms
    """
    n_bufs = len(matmul_fns)

    # Warmup
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        with torch.cuda.stream(matmul_stream):
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
        with torch.cuda.stream(matmul_stream):
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


def _stats(times: List[float]) -> Dict[str, float]:
    """Return dict with min / max / mean / median over a list of times.
    
    Args:
        times: List of timing measurements
    
    Returns:
        Dict with keys: min, max, mean, median
    """
    return {
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
    }


# ==============================================================================
# MODE IMPLEMENTATIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# MODE: standard - Basic overlap measurement
# ------------------------------------------------------------------------------

def mode_standard(args):
    """Run standard overlap benchmark with GEMM alone, Comm alone, and Overlapped phases."""
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

    # ---- Build collective callable ----
    comm_fn = _make_collective(args.collective, comm_tensor, world_size)

    # ---- Streams ----
    matmul_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    nccl_ch = os.environ.get("NCCL_MAX_NCHANNELS", "unset")

    # ---- Build matmul callable (primary — single buffer) ----
    if args.backend == "torch":
        matmul_fn, reset_fn = _make_torch_matmul(A, B)
    else:
        C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
        matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, args.backend)

    # ---- Build rotating buffer sets ----
    N_ROTATING = 4
    rot_matmul_fns = []
    rot_reset_fns = []
    for _ in range(N_ROTATING):
        rA = torch.randn(args.m, args.k, dtype=dtype, device=dev)
        rB = torch.randn(args.k, args.n, dtype=dtype, device=dev)
        if args.backend == "torch":
            mfn, rfn = _make_torch_matmul(rA, rB)
        else:
            rC = torch.empty(args.m, args.n, dtype=dtype, device=dev)
            mfn, rfn = _make_tritonblas_matmul(rA, rB, rC, args.backend)
        rot_matmul_fns.append(mfn)
        rot_reset_fns.append(rfn)

    # ---- Run benchmark ----
    torch.cuda.empty_cache()

    # Phase 1 – GEMM alone (warm L2)
    gemm_times = _time_per_iter(matmul_fn, matmul_stream, args.warmup, args.steps,
                                reset_fn=reset_fn)
    # Phase 2 – Collective alone
    comm_times = _time_per_iter(comm_fn, comm_stream, args.warmup, args.steps)
    # Phase 3 – GEMM with rotating buffers
    rotating_gemm_times = _time_rotating(
        rot_matmul_fns, rot_reset_fns,
        matmul_stream, args.warmup, args.steps,
    )
    # Phase 4 – Serial (NCCL then GEMM)
    serial_gemm_times = _time_serial(
        matmul_fn, comm_fn, matmul_stream, comm_stream,
        args.warmup, args.steps, reset_fn=reset_fn,
    )
    # Phase 5 – Overlapped (warm L2)
    wall_times, overlap_gemm_times, overlap_comm_times = _time_overlap(
        matmul_fn, comm_fn, matmul_stream, comm_stream,
        args.warmup, args.steps, reset_fn=reset_fn,
    )
    # Phase 6 – Overlapped with rotating buffers
    rot_wall, rot_mm, rot_co = _time_overlap_rotating(
        rot_matmul_fns, rot_reset_fns, comm_fn,
        matmul_stream, comm_stream, args.warmup, args.steps,
    )

    results = {
        "gemm_alone": _stats(gemm_times),
        "comm_alone": _stats(comm_times),
        "rotating_gemm": _stats(rotating_gemm_times),
        "serial_gemm": _stats(serial_gemm_times),
        "overlap_wall": _stats(wall_times),
        "overlap_gemm": _stats(overlap_gemm_times),
        "overlap_comm": _stats(overlap_comm_times),
        "overlap_rot_wall": _stats(rot_wall),
        "overlap_rot_gemm": _stats(rot_mm),
        "overlap_rot_comm": _stats(rot_co),
    }

    # ---- Report (rank 0 only) ----
    if rank == 0:
        _print_standard_results(args, results, nccl_ch, world_size)

    dist.destroy_process_group()


def _print_standard_results(args, results, nccl_ch, world_size):
    """Print standard mode results to console."""
    print(f"\n{'=' * 72}")
    print(f"Overlap Benchmark — {args.backend} GEMM "
          f"({args.m}x{args.n}x{args.k} {args.dtype}) "
          f"+ {args.collective} (comm {args.comm_size})")
    print(f"  NCCL_MAX_NCHANNELS = {nccl_ch}   |   "
          f"warmup = {args.warmup}   steps = {args.steps}")
    print(f"{'=' * 72}")

    def _print_phase(name, stats):
        print(f"  {name:25s}  "
              f"min={stats['min']:8.3f}  mean={stats['mean']:8.3f}  "
              f"median={stats['median']:8.3f}  max={stats['max']:8.3f}  (ms)")

    _print_phase("GEMM alone (warm L2)", results["gemm_alone"])
    _print_phase("Comm alone", results["comm_alone"])
    _print_phase("GEMM rotating bufs", results["rotating_gemm"])
    _print_phase("Serial (GEMM after NCCL)", results["serial_gemm"])
    _print_phase("Overlap (wall)", results["overlap_wall"])
    _print_phase("Overlap (GEMM)", results["overlap_gemm"])
    _print_phase("Overlap (Comm)", results["overlap_comm"])
    _print_phase("Overlap rot (wall)", results["overlap_rot_wall"])
    _print_phase("Overlap rot (GEMM)", results["overlap_rot_gemm"])
    _print_phase("Overlap rot (Comm)", results["overlap_rot_comm"])

    # Overlap metrics
    ideal = max(results["gemm_alone"]["mean"], results["comm_alone"]["mean"])
    actual = results["overlap_wall"]["mean"]
    efficiency = ideal / actual * 100 if actual > 0 else 0
    slowdown_gemm_warm = results["overlap_gemm"]["mean"] / results["gemm_alone"]["mean"]
    slowdown_gemm_rot = results["overlap_rot_gemm"]["mean"] / results["rotating_gemm"]["mean"]
    slowdown_comm = results["overlap_comm"]["mean"] / results["comm_alone"]["mean"]

    print(f"\n  Overlap efficiency:              {efficiency:.1f}%  "
          f"(ideal {ideal:.3f} ms, actual {actual:.3f} ms)")
    print(f"  GEMM slowdown (vs warm L2):      {slowdown_gemm_warm:.2f}x")
    print(f"  GEMM slowdown (vs rotating):     {slowdown_gemm_rot:.2f}x  ← correct baseline")
    print(f"  Comm slowdown (overlap):         {slowdown_comm:.2f}x")
    print()

    # ---- CSV export ----
    if args.output_csv:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backend": args.backend,
            "m": args.m, "n": args.n, "k": args.k,
            "dtype": args.dtype,
            "collective": args.collective,
            "comm_shape": "x".join(str(s) for s in args.comm_size),
            "nccl_max_nchannels": nccl_ch,
            "warmup": args.warmup,
            "steps": args.steps,
            "world_size": world_size,
        }
        # Flatten stats
        for phase in ["gemm_alone", "comm_alone", "rotating_gemm", "serial_gemm",
                      "overlap_wall", "overlap_gemm", "overlap_comm",
                      "overlap_rot_wall", "overlap_rot_gemm", "overlap_rot_comm"]:
            for stat, val in results[phase].items():
                row[f"{phase}_{stat}"] = f"{val:.4f}"
        row["overlap_efficiency_pct"] = f"{efficiency:.1f}"
        row["gemm_slowdown_vs_warm"] = f"{slowdown_gemm_warm:.3f}"
        row["gemm_slowdown_vs_rotating"] = f"{slowdown_gemm_rot:.3f}"
        row["comm_slowdown"] = f"{slowdown_comm:.3f}"

        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Results appended to {args.output_csv}\n")


# ------------------------------------------------------------------------------
# MODE: l2-profile - L2 cache profiling (wrap with rocprof)
# ------------------------------------------------------------------------------

def mode_l2_profile(args):
    """Run GEMM for L2 cache profiling (wrap with rocprof for hardware counters)."""
    SINGLE_GPU_MODES = ["gemm-alone", "gemm-polluted", "gemm-rotating"]
    DISTRIBUTED_MODES = ["gemm-rccl", "gemm-rccl-rotating"]

    if args.profile_mode in SINGLE_GPU_MODES:
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
    else:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
    
    matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, args.backend)
    
    if args.profile_mode == "gemm-alone":
        # Simple isolated GEMM
        for _ in range(args.warmup):
            reset_fn()
            matmul_fn()
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            reset_fn()
            matmul_fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        print(f"GEMM alone: {elapsed / args.steps:.3f} ms/iter ({args.steps} iters)")
    
    elif args.profile_mode == "gemm-polluted":
        # GEMM + concurrent memory streaming
        n_elements = (args.pollution_mb * 1024 * 1024) // 2
        pollution_buf = torch.randn(n_elements, dtype=torch.bfloat16, device=dev)
        
        pollution_stream = torch.cuda.Stream(device=dev)
        gemm_stream = torch.cuda.Stream(device=dev)
        
        for _ in range(args.warmup):
            reset_fn()
            with torch.cuda.stream(pollution_stream):
                pollution_buf.add_(1.0)
            with torch.cuda.stream(gemm_stream):
                matmul_fn()
            torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            reset_fn()
            with torch.cuda.stream(pollution_stream):
                pollution_buf.add_(1.0)
            with torch.cuda.stream(gemm_stream):
                matmul_fn()
            torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        print(f"GEMM + {args.pollution_mb}MB pollution: {elapsed / args.steps:.3f} ms/iter ({args.steps} iters)")
    
    elif args.profile_mode == "gemm-rccl":
        # GEMM + RCCL overlap
        comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
        
        comm_stream = torch.cuda.Stream(device=dev)
        gemm_stream = torch.cuda.Stream(device=dev)
        
        def comm_fn():
            dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
        
        for _ in range(args.warmup):
            reset_fn()
            with torch.cuda.stream(comm_stream):
                comm_fn()
            with torch.cuda.stream(gemm_stream):
                torch.cuda._sleep(100_000)
                matmul_fn()
            torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
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
            print(f"GEMM + RCCL all_reduce: {elapsed / args.steps:.3f} ms/iter ({args.steps} iters)")
        
        dist.destroy_process_group()

    elif args.profile_mode == "gemm-rotating":
        N_BUFS = 4
        rot_As = [torch.randn(args.m, args.k, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_Bs = [torch.randn(args.k, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_Cs = [torch.empty(args.m, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_fns = [_make_tritonblas_matmul(a, b, c, args.backend)
                    for a, b, c in zip(rot_As, rot_Bs, rot_Cs)]

        for j in range(max(args.warmup, N_BUFS)):
            idx = j % N_BUFS
            rot_fns[idx][1]()  # reset
            rot_fns[idx][0]()  # matmul
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(args.steps):
            idx = i % N_BUFS
            rot_fns[idx][1]()
            rot_fns[idx][0]()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        print(f"GEMM rotating: {elapsed / args.steps:.3f} ms/iter ({args.steps} iters, {N_BUFS} bufs)")

    elif args.profile_mode == "gemm-rccl-rotating":
        N_BUFS = 4
        rot_As = [torch.randn(args.m, args.k, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_Bs = [torch.randn(args.k, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_Cs = [torch.empty(args.m, args.n, dtype=dtype, device=dev) for _ in range(N_BUFS)]
        rot_fns = [_make_tritonblas_matmul(a, b, c, args.backend)
                    for a, b, c in zip(rot_As, rot_Bs, rot_Cs)]

        comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
        comm_stream = torch.cuda.Stream(device=dev)
        gemm_stream = torch.cuda.Stream(device=dev)

        def comm_fn():
            dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

        for j in range(max(args.warmup, N_BUFS)):
            idx = j % N_BUFS
            rot_fns[idx][1]()
            with torch.cuda.stream(comm_stream):
                comm_fn()
            with torch.cuda.stream(gemm_stream):
                torch.cuda._sleep(100_000)
                rot_fns[idx][0]()
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(args.steps):
            idx = i % N_BUFS
            rot_fns[idx][1]()
            with torch.cuda.stream(comm_stream):
                comm_fn()
            with torch.cuda.stream(gemm_stream):
                torch.cuda._sleep(100_000)
                rot_fns[idx][0]()
            torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        rank = dist.get_rank()
        if rank == 0:
            print(f"GEMM + RCCL rotating: {elapsed / args.steps:.3f} ms/iter ({args.steps} iters, {N_BUFS} bufs)")

        dist.destroy_process_group()


# ------------------------------------------------------------------------------
# MODE: calibrate-hog - Calibrate CU-hog kernel durations
# ------------------------------------------------------------------------------

def mode_calibrate_hog(args):
    """Calibrate CU-hog kernels to determine iteration counts for target durations."""
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    
    BLOCK = 256
    N_WGS = 32
    
    # ALU hog
    out_alu = torch.empty(N_WGS * BLOCK, dtype=torch.float32, device=dev)
    
    # Memory hog: 1MB per WG
    MEM_STRIDE = 512 * 1024
    buf_mem = torch.randn(N_WGS * MEM_STRIDE, dtype=torch.bfloat16, device=dev)
    
    # Warmup
    _cu_hog_alu_kernel[(N_WGS,)](out_alu, 1000, BLOCK=BLOCK)
    _cu_hog_mem_kernel[(N_WGS,)](buf_mem, 100, MEM_STRIDE, BLOCK=BLOCK)
    torch.cuda.synchronize()
    
    print("=== ALU-only CU-hog (32 WGs) ===")
    for iters in [50000, 100000, 150000, 200000]:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _cu_hog_alu_kernel[(N_WGS,)](out_alu, iters, BLOCK=BLOCK)
        e.record()
        torch.cuda.synchronize()
        print(f'  N_ITERS={iters:>8d}  duration={s.elapsed_time(e):>7.3f} ms')
    
    print("\n=== Memory-streaming CU-hog (32 WGs, 32MB buffer) ===")
    for iters in [5000, 7000, 8000, 9000, 10000, 12000, 15000, 20000]:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _cu_hog_mem_kernel[(N_WGS,)](buf_mem, iters, MEM_STRIDE, BLOCK=BLOCK)
        e.record()
        torch.cuda.synchronize()
        print(f'  N_ITERS={iters:>8d}  duration={s.elapsed_time(e):>7.3f} ms')
    
    print("\nUse these iteration counts to target specific durations in trace mode.")


# ------------------------------------------------------------------------------
# MODE: trace - Kernel trace with CU-hog (wrap with rocprofv3)
# ------------------------------------------------------------------------------

def mode_trace(args):
    """Capture kernel traces with optional CU-hog for timeline analysis."""
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
    
    matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, args.backend)
    
    HOG_BLOCK = 256
    hog_alu_out = torch.empty(args.hog_wgs * HOG_BLOCK, dtype=torch.float32, device=dev)
    MEM_STRIDE = 512 * 1024
    hog_mem_buf = torch.randn(args.hog_wgs * MEM_STRIDE, dtype=torch.bfloat16, device=dev)
    
    def launch_hog(stream):
        with torch.cuda.stream(stream):
            if args.hog_mode == "alu":
                _cu_hog_alu_kernel[(args.hog_wgs,)](
                    hog_alu_out, args.hog_alu_iters, BLOCK=HOG_BLOCK)
            else:  # mem
                _cu_hog_mem_kernel[(args.hog_wgs,)](
                    hog_mem_buf, args.hog_mem_iters, MEM_STRIDE, BLOCK=HOG_BLOCK)
    
    gemm_stream = torch.cuda.Stream(device=dev)
    hog_stream = torch.cuda.Stream(device=dev)
    
    # Warmup
    for _ in range(args.warmup):
        reset_fn()
        matmul_fn()
    launch_hog(hog_stream)
    torch.cuda.synchronize()
    
    mm_events = []
    
    if args.no_overlap:
        # Standalone GEMM
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
        # Overlapped with CU-hog
        for i in range(args.steps):
            reset_fn()
            torch.cuda.synchronize()
            launch_hog(hog_stream)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(gemm_stream)
            with torch.cuda.stream(gemm_stream):
                torch.cuda._sleep(100_000)
                matmul_fn()
            e.record(gemm_stream)
            torch.cuda.synchronize()
            mm_events.append((s, e))
    
    # Print timings
    durs = [s.elapsed_time(e) for s, e in mm_events]
    mode_str = "alone" if args.no_overlap else f"overlap-{args.hog_mode}"
    print(f"\n{args.backend} ({mode_str}): CUDA event GEMM durations (ms):")
    for i, d in enumerate(durs):
        print(f"  iter {i}: {d:.3f} ms")
    avg = sum(durs) / len(durs)
    print(f"  avg: {avg:.3f} ms")


# ------------------------------------------------------------------------------
# MODE: chrome-trace - Chrome trace profiling with torch.profiler
# ------------------------------------------------------------------------------

def mode_chrome_trace(args):
    """Generate Chrome trace JSON for visualization."""
    from torch.profiler import profile, ProfilerActivity
    
    # NCCL setup
    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
    
    if args.backend == "torch":
        matmul_fn, reset_fn = _make_torch_matmul(A, B)
    else:
        C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
        matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, args.backend)
    
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
    
    # Profiling phase (rank 0 only)
    mm_overlap_events = []
    
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
        trace_path = os.path.join(args.output_dir, f"trace_{args.backend}.json")
        prof.export_chrome_trace(trace_path)
        print(f"Trace exported to: {trace_path}")
        
        key_avg_path = os.path.join(args.output_dir, f"key_averages_{args.backend}.txt")
        with open(key_avg_path, "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"Key averages exported to: {key_avg_path}")
        
        overlap_durs = [s.elapsed_time(e) for s, e in mm_overlap_events]
        print(f"\nGEMM overlap durations: " + ", ".join(f"{d:.3f}" for d in overlap_durs))
        print(f"  median: {sorted(overlap_durs)[len(overlap_durs)//2]:.3f} ms\n")
    
    dist.destroy_process_group()


# ------------------------------------------------------------------------------
# MODE: grid-sweep - Grid size sweep for work-stealing
# ------------------------------------------------------------------------------

def mode_grid_sweep(args):
    """Sweep work-stealing GEMM over different grid sizes."""
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    A = torch.randn(args.m, args.k, dtype=dtype, device=dev)
    B = torch.randn(args.k, args.n, dtype=dtype, device=dev)
    C = torch.empty(args.m, args.n, dtype=dtype, device=dev)
    
    # Create selector to get tile info (use default hardware CUs)
    selector = tritonblas.OrigamiMatmulSelector(
        args.m, args.n, args.k, dtype, dtype, dtype, dev, streamk=False)
    
    N_CU = selector._hardware.N_CU
    BLK_M, BLK_N, BLK_K = selector.block_m, selector.block_n, selector.block_k
    num_pid_m = (args.m + BLK_M - 1) // BLK_M
    num_pid_n = (args.n + BLK_N - 1) // BLK_N
    total_tiles = num_pid_m * num_pid_n
    
    print(f"Hardware CUs: {N_CU}")
    print(f"Tile size: {BLK_M}x{BLK_N}x{BLK_K}")
    print(f"Total tiles: {total_tiles}")
    print()
    
    # Default grid_sizes if not provided
    if not args.grid_sizes:
        args.grid_sizes = [304, 296, 288, 280, 272, 264, 256, 240, 224, 200, 176, 152, 128]
    
    print(f"{'Grid (WGs)':<12s} {'Median (ms)':<12s} {'Slowdown':<10s}")
    print("-" * 36)
    
    base = None
    for grid_size in args.grid_sizes:
        # Temporarily override the hardware CU count for this grid size
        # Note: We monkey-patch the selector's hardware rather than creating new selector
        # to avoid hardware capability detection issues with non-standard CU counts
        orig_n_cu = selector._hardware.N_CU
        selector._hardware.N_CU = grid_size
        
        # Build matmul (will use the overridden CU count)
        matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, "ws")
        
        # Restore original CU count
        selector._hardware.N_CU = orig_n_cu
        
        # Warmup
        for _ in range(10):
            reset_fn()
            matmul_fn()
        torch.cuda.synchronize()
        
        times = []
        for _ in range(args.steps):
            reset_fn()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            matmul_fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        
        med = statistics.median(times)
        if base is None:
            base = med
        slow = med / base
        print(f"{grid_size:<12d} {med:<12.3f} {slow:<10.2f}x")


# ------------------------------------------------------------------------------
# MODE: se-sweep - SE oversubscription shape sweep
# ------------------------------------------------------------------------------

# Shape presets
SE_SHAPES_SMALL = [
    (3584, 3584, 4096),
    (3840, 3840, 4096),
    (4096, 4096, 4096),
    (4352, 4352, 4096),
]

SE_SHAPES_LARGE = [
    (8192, 8192, 8192),
    (8448, 8448, 8192),
    (8704, 8704, 8192),
    (12288, 12288, 8192),
    (12544, 12544, 8192),
    (16384, 16384, 8192),
    (16640, 16640, 8192),
]

SE_SHAPES_ALL = SE_SHAPES_SMALL + SE_SHAPES_LARGE

# Tensile macro-tile sizes (from TENSILE_DB=0x8040)
TENSILE_MT = {
    (3584, 3584, 4096): (256, 176),
    (3840, 3840, 4096): (256, 192),
    (4096, 4096, 4096): (256, 224),
    (4352, 4352, 4096): (256, 128),
    (8192, 8192, 8192): (256, 224),
    (8448, 8448, 8192): (256, 160),
    (8704, 8704, 8192): (512, 128),
    (12288, 12288, 8192): (512, 128),
    (12544, 12544, 8192): (256, 304),
    (16384, 16384, 8192): (512, 160),
    (16640, 16640, 8192): (256, 192),
}

N_ROTATING = 4


def mode_se_sweep(args):
    """Run SE oversubscription shape sweep."""
    # NCCL setup
    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    # Determine shapes
    if args.shapes_preset == "small":
        shapes = SE_SHAPES_SMALL
    elif args.shapes_preset == "large":
        shapes = SE_SHAPES_LARGE
    elif args.shapes_preset == "all":
        shapes = SE_SHAPES_ALL
    else:  # custom
        shapes = args.custom_shapes if args.custom_shapes else SE_SHAPES_ALL
    
    # Communication tensor for overlap
    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
    comm_stream = torch.cuda.Stream(device=dev)
    
    def comm_fn():
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
    
    results = []
    
    for M, N, K in shapes:
        for backend in args.backends:
            matmul_stream = torch.cuda.Stream(device=dev)
            
            # Warmup + single buffer
            A = torch.randn(M, K, dtype=dtype, device=dev)
            B = torch.randn(K, N, dtype=dtype, device=dev)
            
            if backend == "torch":
                matmul_fn, reset_fn = _make_torch_matmul(A, B)
                # Get Tensile tile info
                key = (M, N, K)
                if key in TENSILE_MT:
                    tm, tn = TENSILE_MT[key]
                    tiles = math.ceil(M / tm) * math.ceil(N / tn)
                else:
                    tm, tn, tiles = 0, 0, 0
            else:
                C = torch.empty(M, N, dtype=dtype, device=dev)
                matmul_fn, reset_fn = _make_tritonblas_matmul(A, B, C, backend)
                # Get tile info from selector
                selector = tritonblas.OrigamiMatmulSelector(
                    M, N, K, dtype, dtype, dtype, dev, streamk=False)
                tm, tn = selector.block_m, selector.block_n
                tiles = triton.cdiv(M, tm) * triton.cdiv(N, tn)
            
            # Rotating buffers
            rot_fns, rot_rfns = [], []
            for _ in range(N_ROTATING):
                rA = torch.randn(M, K, dtype=dtype, device=dev)
                rB = torch.randn(K, N, dtype=dtype, device=dev)
                if backend == "torch":
                    mfn, rfn = _make_torch_matmul(rA, rB)
                else:
                    rC = torch.empty(M, N, dtype=dtype, device=dev)
                    mfn, rfn = _make_tritonblas_matmul(rA, rB, rC, backend)
                rot_fns.append(mfn)
                rot_rfns.append(rfn)
            
            # Measure: alone (rotating baseline)
            alone_times = _time_rotating(rot_fns, rot_rfns, matmul_stream, 10, args.steps)
            
            # Measure: overlap (rotating)
            ovlp_wall, ovlp_mm, ovlp_co = _time_overlap_rotating(
                rot_fns, rot_rfns, comm_fn, matmul_stream, comm_stream, 10, args.steps)
            
            # Compute stats
            alone_mean = statistics.mean(alone_times)
            alone_max = max(alone_times[1:])  # Exclude first
            ovlp_mean = statistics.mean(ovlp_mm)
            ovlp_max = max(ovlp_mm[1:])
            slowdown = ovlp_mean / alone_mean if alone_mean > 0 else 0
            
            result = {
                "shape": f"{M}x{N}x{K}",
                "M": M, "N": N, "K": K,
                "backend": backend,
                "tile": f"{tm}x{tn}",
                "tiles": tiles,
                "tiles_mod8": tiles % 8,
                "tiles_mod32": tiles % 32,
                "alone_mean": alone_mean,
                "alone_max": alone_max,
                "overlap_mean": ovlp_mean,
                "overlap_max": ovlp_max,
                "slowdown": slowdown,
            }
            results.append(result)
            
            if rank == 0:
                print(f"{backend:10s} {M}x{N}x{K}  tiles={tiles:5d} (%8={tiles%8}, %32={tiles%32:2d})  "
                      f"alone={alone_mean:.3f}ms  ovlp={ovlp_mean:.3f}ms  slow={slowdown:.2f}x")
    
    # CSV output
    if rank == 0 and args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nResults written to {args.output_csv}")
    
    dist.destroy_process_group()


# ==============================================================================
# Argument Parsing
# ==============================================================================

def add_common_args(parser):
    """Add common arguments shared across modes."""
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup iterations")
    parser.add_argument("--steps", type=int, default=200,
                       help="Timed iterations")


def add_gemm_args(parser):
    """Add GEMM-related arguments."""
    parser.add_argument("--m", "--gemm-m", type=int, default=8192, dest="m",
                       help="GEMM M dimension")
    parser.add_argument("--n", "--gemm-n", type=int, default=8192, dest="n",
                       help="GEMM N dimension")
    parser.add_argument("--k", "--gemm-k", type=int, default=8192, dest="k",
                       help="GEMM K dimension")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["bf16", "fp16"],
                       help="GEMM data type")


def add_backend_arg(parser, required=False):
    """Add backend selection argument."""
    parser.add_argument("--backend", "--matmul-backend",
                       type=str,
                       default="torch" if not required else None,
                       required=required,
                       dest="backend",
                       choices=["torch", "persistent", "streamk", "ws", "ws-global"],
                       help="GEMM backend")


def add_comm_args(parser):
    """Add communication-related arguments."""
    parser.add_argument("--comm-size", type=int, nargs="+", default=[8192, 8192],
                       help="Collective tensor shape (e.g., --comm-size 8192 8192)")
    parser.add_argument("--collective", type=str, default="all_reduce",
                       choices=list(COLLECTIVES.keys()),
                       help="Collective operation")
    parser.add_argument("--nccl-max-nchannels", type=int, default=None,
                       help="Set NCCL_MAX_NCHANNELS (controls CU usage by RCCL)")


def parse_args():
    """Parse command-line arguments with mode-based subcommands."""
    parser = argparse.ArgumentParser(
        description="Compute–Communication Overlap Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Benchmark mode")
    
    # --- MODE: standard ---
    p_std = subparsers.add_parser("standard",
                                  help="Basic overlap measurement",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gemm_args(p_std)
    add_backend_arg(p_std)
    add_comm_args(p_std)
    add_common_args(p_std)
    p_std.add_argument("--output-csv", type=str, default=None,
                      help="Append results to CSV file")
    
    # --- MODE: l2-profile ---
    p_l2 = subparsers.add_parser("l2-profile",
                                 help="L2 cache profiling (wrap with rocprof)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gemm_args(p_l2)
    add_backend_arg(p_l2)
    add_common_args(p_l2)
    p_l2.add_argument("--profile-mode", type=str, required=True,
                     choices=["gemm-alone", "gemm-polluted", "gemm-rccl",
                              "gemm-rotating", "gemm-rccl-rotating"],
                     help="Profiling mode")
    p_l2.add_argument("--pollution-mb", type=int, default=512,
                     help="Size of L2-pollution buffer in MB (for gemm-polluted mode)")
    p_l2.add_argument("--comm-size", type=int, nargs="+", default=[16384, 16384],
                     help="Collective tensor shape (for gemm-rccl mode)")
    p_l2.add_argument("--nccl-max-nchannels", type=int, default=None,
                     help="Set NCCL_MAX_NCHANNELS (for gemm-rccl mode)")
    
    # --- MODE: calibrate-hog ---
    p_cal = subparsers.add_parser("calibrate-hog",
                                  help="Calibrate CU-hog kernel durations",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # No additional args needed
    
    # --- MODE: trace ---
    p_trace = subparsers.add_parser("trace",
                                    help="Kernel trace capture with CU-hog (wrap with rocprofv3)",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gemm_args(p_trace)
    add_backend_arg(p_trace)
    add_common_args(p_trace)
    p_trace.add_argument("--no-overlap", action="store_true",
                        help="Run GEMM alone (no CU-hog)")
    p_trace.add_argument("--hog-mode", type=str, default="alu",
                        choices=["alu", "mem"],
                        help="CU-hog type: alu (pure compute) or mem (memory streaming)")
    p_trace.add_argument("--hog-wgs", type=int, default=32,
                        help="Number of workgroups for CU-hog")
    p_trace.add_argument("--hog-alu-iters", type=int, default=100_000,
                        help="Iterations for ALU hog (~2ms on MI300X)")
    p_trace.add_argument("--hog-mem-iters", type=int, default=9_000,
                        help="Iterations for MEM hog (~2.3ms on MI300X)")
    
    # --- MODE: chrome-trace ---
    p_chrome = subparsers.add_parser("chrome-trace",
                                     help="Chrome trace profiling with torch.profiler",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gemm_args(p_chrome)
    add_backend_arg(p_chrome)
    add_comm_args(p_chrome)
    add_common_args(p_chrome)
    p_chrome.add_argument("--output-dir", type=str, default="/tmp/overlap_profile",
                         help="Directory for trace files")
    
    # --- MODE: grid-sweep ---
    p_grid = subparsers.add_parser("grid-sweep",
                                   help="Grid size sweep for work-stealing (single GPU)",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gemm_args(p_grid)
    p_grid.add_argument("--grid-sizes", type=int, nargs="+", default=None,
                       help="List of grid sizes (WG counts) to test")
    p_grid.add_argument("--steps", type=int, default=50,
                       help="Timed iterations per grid size")
    
    # --- MODE: se-sweep ---
    p_se = subparsers.add_parser("se-sweep",
                                 help="SE oversubscription shape sweep",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_comm_args(p_se)
    p_se.add_argument("--backends", type=str, nargs="+",
                     default=["ws", "persistent", "torch"],
                     choices=["torch", "persistent", "streamk", "ws", "ws-global"],
                     help="GEMM backends to test")
    p_se.add_argument("--shapes-preset", type=str, default="all",
                     choices=["small", "large", "all", "custom"],
                     help="Shape preset to sweep")
    p_se.add_argument("--custom-shapes", type=str, default=None,
                     help="JSON list of (M,N,K) tuples for custom shapes")
    p_se.add_argument("--dtype", type=str, default="bf16",
                     choices=["bf16", "fp16"],
                     help="GEMM data type")
    p_se.add_argument("--warmup", type=int, default=10,
                     help="Warmup iterations")
    p_se.add_argument("--steps", type=int, default=100,
                     help="Timed iterations")
    p_se.add_argument("--output-csv", type=str, default=None,
                     help="CSV output file")
    
    return parser.parse_args()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point with mode dispatch."""
    args = parse_args()
    
    # Parse custom shapes if provided
    if hasattr(args, 'custom_shapes') and args.custom_shapes:
        args.custom_shapes = json.loads(args.custom_shapes)
    
    mode_dispatch = {
        "standard": mode_standard,
        "l2-profile": mode_l2_profile,
        "calibrate-hog": mode_calibrate_hog,
        "trace": mode_trace,
        "chrome-trace": mode_chrome_trace,
        "grid-sweep": mode_grid_sweep,
        "se-sweep": mode_se_sweep,
    }
    
    mode_fn = mode_dispatch.get(args.mode)
    if mode_fn is None:
        print(f"Error: Mode '{args.mode}' not implemented", file=sys.stderr)
        sys.exit(1)
    
    mode_fn(args)


if __name__ == "__main__":
    main()

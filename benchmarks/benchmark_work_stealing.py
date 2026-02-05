#!/usr/bin/env python3
"""
Benchmark: work-stealing persistent GEMM vs. static persistent GEMM vs. Stream-K vs. torch.matmul

Uses importlib bootstrap to load kernels directly, bypassing the full tritonblas
import (which requires triton.constexpr_function not available in older builds).

Usage:
    HIP_VISIBLE_DEVICES=6 python benchmarks/benchmark_work_stealing.py
"""

import os
import sys
import time
import types
import importlib.util
import torch
import triton
import triton.language as tl
from math import ceil

# ---------------------------------------------------------------------------
# Bootstrap: load kernels without triggering stages/__init__.py
# ---------------------------------------------------------------------------
_root = os.path.join(os.path.dirname(__file__), "..", "include", "tritonblas")
_kernels_dir = os.path.join(_root, "kernels")
_stages_dir = os.path.join(_kernels_dir, "stages")
_indexing_dir = os.path.join(_stages_dir, "indexing")


def _load_module(fqn, filepath, package_path=None):
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    if package_path is not None:
        mod.__path__ = [package_path]
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_stub_package(fqn, path):
    pkg = types.ModuleType(fqn)
    pkg.__path__ = [path]
    pkg.__package__ = fqn
    sys.modules[fqn] = pkg
    return pkg


# Stub packages
_make_stub_package("tritonblas", _root)
_make_stub_package("tritonblas.kernels", _kernels_dir)
_make_stub_package("tritonblas.kernels.stages", _stages_dir)
_make_stub_package("tritonblas.kernels.stages.indexing", _indexing_dir)

# Load pid_transforms (pure @triton.jit, no constexpr_function)
_load_module(
    "tritonblas.kernels.stages.indexing.pid_transforms",
    os.path.join(_indexing_dir, "pid_transforms.py"),
)

# Load kernels
_mono_mod = _load_module(
    "tritonblas.kernels.persistent_gemm_monolithic",
    os.path.join(_kernels_dir, "persistent_gemm_monolithic.py"),
)
_ws_mod = _load_module(
    "tritonblas.kernels.persistent_gemm_work_stealing",
    os.path.join(_kernels_dir, "persistent_gemm_work_stealing.py"),
)
_sk_mod = _load_module(
    "tritonblas.kernels.streamk_gemm",
    os.path.join(_kernels_dir, "streamk_gemm.py"),
)

persistent_matmul = _mono_mod.persistent_matmul
ws_persistent_matmul = _ws_mod.ws_persistent_matmul
streamk_matmul = _sk_mod.streamk_matmul


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
def _common_params(A, B, C, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS):
    M, K = A.shape
    _, N = B.shape
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0
    chunk_size = GROUP_M * GROUP_M
    chunk_size = min(chunk_size, max(1, total_tiles // NUM_XCDS))
    return M, K, N, total_tiles, even_k, chunk_size


def launch_persistent(A, B, C, BLK_M=128, BLK_N=128, BLK_K=64, GROUP_M=8, NUM_XCDS=8):
    """Original static-partition persistent GEMM (monolithic)."""
    M, K, N, total_tiles, even_k, chunk_size = _common_params(
        A, B, C, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS
    )
    grids = total_tiles
    persistent_matmul[(grids,)](
        A, B, C,
        None, None, None,  # scale, bias
        M, N, K,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=GROUP_M, NUM_SMS=grids, NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size, BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1,
    )


def launch_work_stealing(A, B, C, tile_counter, num_sms,
                         BLK_M=128, BLK_N=128, BLK_K=64, GROUP_M=8, NUM_XCDS=8):
    """Work-stealing persistent GEMM (atomic counter)."""
    M, K, N, total_tiles, even_k, chunk_size = _common_params(
        A, B, C, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS
    )
    grids = num_sms
    tile_counter.zero_()
    ws_persistent_matmul[(grids,)](
        A, B, C,
        None, None, None,  # scale, bias
        tile_counter,
        M, N, K,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=GROUP_M, NUM_SMS=grids, NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size, BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1,
    )


def launch_streamk(A, B, C, locks, P, sk_grid,
                   BLK_M=128, BLK_N=128, BLK_K=64, GROUP_M=8, NUM_XCDS=8):
    """Stream-K persistent GEMM."""
    M, K, N, total_tiles, even_k, chunk_size = _common_params(
        A, B, C, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS
    )
    # StreamK tiles = remainder tiles that need cooperative decomposition
    streamk_tiles = total_tiles % sk_grid if sk_grid > 0 else 0

    chunk_size_sk = GROUP_M * GROUP_M
    chunk_size_sk = min(chunk_size_sk, max(1, sk_grid // NUM_XCDS))

    locks[:sk_grid].zero_()
    streamk_matmul[(sk_grid,)](
        A, B, C,
        None, None, None,  # scale, bias
        P[:sk_grid, :BLK_M * BLK_N],
        locks[:sk_grid],
        M, N, K,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=GROUP_M, NUM_SMS=sk_grid, NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size_sk, STREAMK_TILES=streamk_tiles,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1,
    )


def launch_torch(A, B, C):
    """torch.matmul (rocBLAS/hipBLAS backend)."""
    torch.matmul(A, B, out=C)


# ---------------------------------------------------------------------------
# Simple Stream-K grid heuristic (mirrors origami logic)
# ---------------------------------------------------------------------------
def compute_sk_grid(M, N, K, BLK_M, BLK_N, BLK_K, cu_count):
    tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
    sk_grid = tiles
    split_factors = [8, 6, 4, 3, 2, 1]
    tile_fractions = [0.0, 0.5, 0.125, 0.2, 0.25, 1.0 / 3.0]
    iters_per_tile = max(1, ceil(K / BLK_K))

    if tiles > cu_count:
        min_even_tiles = tiles / cu_count
        for frac in tile_fractions:
            frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)
            if frac_grid <= cu_count:
                sk_grid = frac_grid
                break
    elif tiles < cu_count:
        for factor in split_factors:
            split_grid = tiles * factor
            iters_per_cu = iters_per_tile // factor
            if split_grid <= cu_count and iters_per_cu >= 8:
                sk_grid = split_grid
                break

    if tiles % sk_grid != 0:
        sk_grid = tiles

    if tiles >= cu_count:
        last_wave_remainder = tiles % cu_count
        if last_wave_remainder < 128 and last_wave_remainder > 0 and cu_count in [304, 80, 64]:
            sk_grid = 256 if cu_count == 304 else 64

    return sk_grid


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench(fn, warmup=25, iters=50):
    """Return median runtime in ms using triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=iters)


def main():
    torch.manual_seed(42)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    NUM_SMS = props.multi_processor_count
    NUM_XCDS = 8  # MI300X

    print(f"Device      : {props.name}")
    print(f"CUs (SMs)   : {NUM_SMS}")
    print(f"HIP_VISIBLE : {os.environ.get('HIP_VISIBLE_DEVICES', '<not set>')}")
    print()

    # Pre-allocate work-stealing counter + Stream-K buffers once
    tile_counter = torch.zeros(1, device="cuda", dtype=torch.int32)
    max_grid = NUM_SMS * 2  # generous upper bound for SK grid
    block_area = 128 * 128
    locks = torch.zeros(max_grid, device="cuda", dtype=torch.uint8)
    P = torch.zeros(max_grid, block_area, device="cuda", dtype=torch.float32)

    BLK_M, BLK_N, BLK_K, GROUP_M = 128, 128, 64, 8
    dtype = torch.float16

    # Problem sizes to benchmark
    sizes = [
        # Square
        (256,   256,   256),
        (512,   512,   512),
        (1024,  1024,  1024),
        (2048,  2048,  2048),
        (4096,  4096,  4096),
        (8192,  8192,  8192),
        # Rectangular (common LLM shapes)
        (1,     4096,  4096),
        (4,     4096,  4096),
        (16,    4096,  4096),
        (32,    4096,  4096),
        (64,    4096,  4096),
        (128,   4096,  4096),
        (256,   4096,  4096),
        (512,   4096,  4096),
        (1024,  4096,  4096),
        (2048,  4096,  4096),
        (4096,  4096,  11008),
        (4096,  11008, 4096),
        (8192,  8192,  4096),
        (8192,  4096,  8192),
    ]

    # Header
    hdr = (
        f"{'M':>6} {'N':>6} {'K':>6} │ "
        f"{'Persistent':>12} {'WorkSteal':>12} {'StreamK':>12} {'torch.mm':>12} │ "
        f"{'WS/Pers':>8} {'WS/SK':>8} {'WS/Torch':>8}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(f"{'':>20} │ {'── Time (ms) ──':^51} │ {'── Speedup ──':^26}")
    print(hdr)
    print(sep)

    results = []

    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype).T  # K x N contiguous
        C_pers = torch.zeros(M, N, device="cuda", dtype=dtype)
        C_ws   = torch.zeros(M, N, device="cuda", dtype=dtype)
        C_sk   = torch.zeros(M, N, device="cuda", dtype=dtype)
        C_ref  = torch.zeros(M, N, device="cuda", dtype=dtype)

        even_k = K % BLK_K == 0
        total_tiles_m = triton.cdiv(M, BLK_M)
        total_tiles_n = triton.cdiv(N, BLK_N)
        total_tiles = total_tiles_m * total_tiles_n

        # Skip tiny sizes where tiles < 1 (e.g. M=1 with BLK_M=128 still gives 1 tile)
        sk_grid = compute_sk_grid(M, N, K, BLK_M, BLK_N, BLK_K, NUM_SMS)
        # Clamp stream-K grid to our pre-allocated buffer size
        sk_grid = min(sk_grid, max_grid)

        # ── Benchmark each variant ──────────────────────────────────
        try:
            ms_pers = bench(lambda: launch_persistent(
                A, B, C_pers, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS))
        except Exception as e:
            ms_pers = float("nan")

        try:
            ms_ws = bench(lambda: launch_work_stealing(
                A, B, C_ws, tile_counter, NUM_SMS, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS))
        except Exception as e:
            ms_ws = float("nan")

        try:
            ms_sk = bench(lambda: launch_streamk(
                A, B, C_sk, locks, P, sk_grid, BLK_M, BLK_N, BLK_K, GROUP_M, NUM_XCDS))
        except Exception as e:
            ms_sk = float("nan")

        ms_torch = bench(lambda: launch_torch(A, B, C_ref))

        # ── Speedups (> 1.0 means work-stealing is faster) ─────────
        su_pers  = ms_pers / ms_ws  if ms_ws > 0 else float("nan")
        su_sk    = ms_sk   / ms_ws  if ms_ws > 0 else float("nan")
        su_torch = ms_torch / ms_ws if ms_ws > 0 else float("nan")

        # ── TFLOP/s ────────────────────────────────────────────────
        flops = 2.0 * M * N * K
        def to_tflops(ms):
            return flops / (ms * 1e-3) / 1e12 if ms > 0 else 0

        row = {
            "M": M, "N": N, "K": K,
            "persistent_ms": ms_pers,
            "work_stealing_ms": ms_ws,
            "streamk_ms": ms_sk,
            "torch_ms": ms_torch,
            "persistent_tflops": to_tflops(ms_pers),
            "work_stealing_tflops": to_tflops(ms_ws),
            "streamk_tflops": to_tflops(ms_sk),
            "torch_tflops": to_tflops(ms_torch),
            "speedup_vs_pers": su_pers,
            "speedup_vs_sk": su_sk,
            "speedup_vs_torch": su_torch,
        }
        results.append(row)

        # Format ms; mark NaN
        def fmt_ms(v):
            return f"{v:12.4f}" if v == v else f"{'N/A':>12}"
        def fmt_su(v):
            return f"{v:8.3f}" if v == v else f"{'N/A':>8}"

        print(
            f"{M:>6} {N:>6} {K:>6} │ "
            f"{fmt_ms(ms_pers)} {fmt_ms(ms_ws)} {fmt_ms(ms_sk)} {fmt_ms(ms_torch)} │ "
            f"{fmt_su(su_pers)} {fmt_su(su_sk)} {fmt_su(su_torch)}"
        )

    print(sep)

    # ── Summary in TFLOP/s ──────────────────────────────────────────
    print()
    print(sep)
    print(f"{'':>20} │ {'── TFLOP/s ──':^51} │")
    hdr2 = (
        f"{'M':>6} {'N':>6} {'K':>6} │ "
        f"{'Persistent':>12} {'WorkSteal':>12} {'StreamK':>12} {'torch.mm':>12} │"
    )
    print(hdr2)
    print(sep)
    for r in results:
        def fmt_tf(v):
            return f"{v:12.2f}" if v > 0 else f"{'N/A':>12}"
        print(
            f"{r['M']:>6} {r['N']:>6} {r['K']:>6} │ "
            f"{fmt_tf(r['persistent_tflops'])} {fmt_tf(r['work_stealing_tflops'])} "
            f"{fmt_tf(r['streamk_tflops'])} {fmt_tf(r['torch_tflops'])} │"
        )
    print(sep)

    # ── Geometric mean speedup ──────────────────────────────────────
    import math
    valid_pers = [r["speedup_vs_pers"] for r in results if r["speedup_vs_pers"] == r["speedup_vs_pers"] and r["speedup_vs_pers"] > 0]
    valid_sk   = [r["speedup_vs_sk"]   for r in results if r["speedup_vs_sk"]   == r["speedup_vs_sk"]   and r["speedup_vs_sk"] > 0]
    valid_torch= [r["speedup_vs_torch"]for r in results if r["speedup_vs_torch"]== r["speedup_vs_torch"]and r["speedup_vs_torch"] > 0]

    def geomean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    print()
    print("Geometric-mean speedup of Work-Stealing over:")
    print(f"  Persistent (static) : {geomean(valid_pers):.4f}x")
    print(f"  Stream-K            : {geomean(valid_sk):.4f}x")
    print(f"  torch.matmul        : {geomean(valid_torch):.4f}x")
    print()


if __name__ == "__main__":
    main()

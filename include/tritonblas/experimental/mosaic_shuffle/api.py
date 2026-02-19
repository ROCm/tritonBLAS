"""
High-level API for mosaic shuffle GEMM scheduling.

Three strategies:
- matmul_random: Feistel permutation of individual tiles
- matmul_l2_aware: Mosaic 2-level L2 tiling + Feistel shuffle
- matmul_llc_and_l2_aware: Mosaic 3-level deterministic hierarchy
"""

import torch
import triton
from typing import Optional, Union, Tuple

from .config import L2AwareConfig, LLCAndL2AwareConfig
from .permutation import compute_permutation_table
from .kernels import (
    persistent_matmul_random,
    persistent_matmul_l2_aware,
    persistent_matmul_llc_and_l2_aware,
    persistent_matmul_debug_map_random,
    persistent_matmul_debug_map_l2_aware,
    persistent_matmul_debug_map_llc_and_l2_aware,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_config(selector):
    """Extract block sizes from a tritonBLAS selector (duck-typed)."""
    try:
        return (
            selector.block_m,
            selector.block_n,
            selector.block_k,
            selector.group_m,
            selector.num_sms,
        )
    except AttributeError:
        try:
            return selector.get_config()
        except AttributeError:
            raise TypeError(
                f"Selector must have block_m/block_n/block_k/group_m/num_sms "
                f"properties or a get_config() method. Got: {type(selector)}"
            )


def _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, num_sms, even_k):
    """Build the common kernel argument dict."""
    return {
        "A": a,
        "B": b,
        "C": c,
        "bias_ptr": None,
        "M": M,
        "N": N,
        "K": K,
        "stride_am": a.stride(0),
        "stride_bn": b.stride(1),
        "stride_cm": c.stride(0),
        "stride_cn": c.stride(1),
        "stride_bias": 0,
        "trace_start_ptr": None,
        "trace_end_ptr": None,
        "trace_pid_ptr": None,
        "trace_xcd_ptr": None,
        "stride_ak": a.stride(1),
        "stride_bk": b.stride(0),
        "BLOCK_SIZE_M": BLK_M,
        "BLOCK_SIZE_N": BLK_N,
        "BLOCK_SIZE_K": BLK_K,
        "NUM_SMS": num_sms,
        "NUM_XCDS": 8,
        "BIAS": False,
        "EVEN_K": even_k,
    }


def _setup_trace_buffers(device, total_tiles):
    return {
        "trace_start_ptr": torch.zeros(total_tiles, dtype=torch.int64, device=device),
        "trace_end_ptr": torch.zeros(total_tiles, dtype=torch.int64, device=device),
        "trace_pid_ptr": torch.zeros(total_tiles, dtype=torch.int32, device=device),
        "trace_xcd_ptr": torch.zeros(total_tiles, dtype=torch.int32, device=device),
    }


def _collect_trace_data(trace_bufs, total_tiles, total_programs, total_blocks_M,
                        total_blocks_N, M, N, K, BLK_M, BLK_N, BLK_K, num_xcds=8):
    torch.cuda.synchronize()
    return {
        "start": trace_bufs["trace_start_ptr"].cpu(),
        "end": trace_bufs["trace_end_ptr"].cpu(),
        "pid": trace_bufs["trace_pid_ptr"].cpu(),
        "xcd": trace_bufs["trace_xcd_ptr"].cpu(),
        "total_tiles": total_tiles,
        "total_programs": total_programs,
        "num_pid_m": total_blocks_M,
        "num_pid_n": total_blocks_N,
        "M": M,
        "N": N,
        "K": K,
        "BLOCK_SIZE_M": BLK_M,
        "BLOCK_SIZE_N": BLK_N,
        "BLOCK_SIZE_K": BLK_K,
        "GROUP_SIZE_M": 0,
        "NUM_XCDS": num_xcds,
    }


def _get_selector(a, b, c, M, N, K, selector):
    if selector is None:
        from ..origami import OrigamiMatmulSelector
        selector = OrigamiMatmulSelector(M, N, K, a.dtype, b.dtype, c.dtype, a.device)
    return selector


def _count_l2_tiles(num_pid_m: int, num_pid_n: int, tile_y: int, tile_x: int) -> int:
    quantized_m = (num_pid_m // tile_y) * tile_y
    quantized_n = (num_pid_n // tile_x) * tile_x
    return (quantized_m // tile_y) * (quantized_n // tile_x)


# ---------------------------------------------------------------------------
# Strategy 1: random
# ---------------------------------------------------------------------------

def matmul_random(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    seed: int = 42,
    selector=None,
    trace: bool = False,
):
    """
    GEMM with random permutation of all individual output tiles.

    Every tile in the grid is independently permuted -- no L2 tile grouping.
    The permutation is computed on the host via a Feistel network and uploaded
    as a lookup table to the GPU.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _get_selector(a, b, c, M, N, K, selector)
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    perm_table = compute_permutation_table(max(total_tiles, 2), seed, device=a.device)

    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs["perm_table_ptr"] = perm_table

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    persistent_matmul_random[(total_tiles,)](
        **kernel_kwargs,
        num_stages=2,
        num_warps=8,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    if trace:
        trace_data = _collect_trace_data(
            trace_bufs, total_tiles, total_tiles, total_blocks_M, total_blocks_N,
            M, N, K, BLK_M, BLK_N, BLK_K,
        )
        return c, trace_data
    return c


# ---------------------------------------------------------------------------
# Strategy 2: l2_aware
# ---------------------------------------------------------------------------

def matmul_l2_aware(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    config: L2AwareConfig,
    selector=None,
    trace: bool = False,
):
    """
    GEMM with mosaic-style L2 tiling and random shuffle of L2 tile groups.

    Individual tiles within each L2 group are traversed according to
    config.inner_ordering for cache locality. The L2 groups themselves
    are randomly permuted via a host-computed lookup table.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _get_selector(a, b, c, M, N, K, selector)
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    num_l2_tiles = _count_l2_tiles(total_blocks_M, total_blocks_N, config.tile_y, config.tile_x)
    if num_l2_tiles < 2:
        raise ValueError(
            f"L2-aware strategy requires >= 2 L2 tiles in the quantized region. "
            f"Got {num_l2_tiles} (grid: {total_blocks_M}x{total_blocks_N}, "
            f"tile: {config.tile_y}x{config.tile_x})"
        )

    perm_table = compute_permutation_table(num_l2_tiles, config.seed, device=a.device)

    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs.update({
        "perm_table_ptr": perm_table,
        "TILE_Y": config.tile_y,
        "TILE_X": config.tile_x,
        "INNER_ORDER": config.inner_ordering,
        "CHUNK_SIZE": config.tile_y * config.tile_x,
    })

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    persistent_matmul_l2_aware[(total_tiles,)](
        **kernel_kwargs,
        num_stages=2,
        num_warps=8,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    if trace:
        trace_data = _collect_trace_data(
            trace_bufs, total_tiles, total_tiles, total_blocks_M, total_blocks_N,
            M, N, K, BLK_M, BLK_N, BLK_K,
        )
        return c, trace_data
    return c


# ---------------------------------------------------------------------------
# Strategy 3: llc_and_l2_aware
# ---------------------------------------------------------------------------

def matmul_llc_and_l2_aware(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    config: LLCAndL2AwareConfig,
    selector=None,
    trace: bool = False,
):
    """
    GEMM with mosaic 3-level deterministic hierarchy (LayoutRank2Depth3).

    No random permutation -- purely deterministic traversal optimized
    for LLC and L2 cache locality.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _get_selector(a, b, c, M, N, K, selector)
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs.update(config.to_kernel_kwargs())

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    persistent_matmul_llc_and_l2_aware[(total_tiles,)](
        **kernel_kwargs,
        num_stages=2,
        num_warps=8,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    if trace:
        trace_data = _collect_trace_data(
            trace_bufs, total_tiles, total_tiles, total_blocks_M, total_blocks_N,
            M, N, K, BLK_M, BLK_N, BLK_K,
        )
        return c, trace_data
    return c


# ---------------------------------------------------------------------------
# Workgroup map visualization
# ---------------------------------------------------------------------------

def compute_workgroup_map(
    m: int,
    n: int,
    k: int,
    strategy: str,
    config=None,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    selector=None,
) -> torch.Tensor:
    """
    Generate workgroup-to-tile mapping for visualization.

    Returns a (num_pid_m, num_pid_n) tensor where each element is the
    workgroup ID assigned to that output tile.

    Args:
        strategy: One of "random", "l2_aware", "llc_and_l2_aware"
        config: L2AwareConfig or LLCAndL2AwareConfig (strategy-dependent)
        seed: Feistel seed (for random / l2_aware)
    """
    device = torch.device("cuda")

    if selector is None:
        from ..origami import OrigamiMatmulSelector
        selector = OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, device)

    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)
    num_pid_m = triton.cdiv(m, BLK_M)
    num_pid_n = triton.cdiv(n, BLK_N)
    total_tiles = num_pid_m * num_pid_n
    workgroup_map = torch.zeros((total_tiles,), dtype=torch.int32, device=device)

    common_kwargs = {
        "workgroup_map": workgroup_map,
        "M": m,
        "N": n,
        "BLOCK_SIZE_M": BLK_M,
        "BLOCK_SIZE_N": BLK_N,
        "NUM_SMS": total_tiles,
        "NUM_XCDS": 8,
    }

    if strategy == "random":
        perm_table = compute_permutation_table(max(total_tiles, 2), seed, device=device)
        persistent_matmul_debug_map_random[(total_tiles,)](
            **common_kwargs,
            perm_table_ptr=perm_table,
        )

    elif strategy == "l2_aware":
        if config is None:
            raise ValueError("l2_aware strategy requires an L2AwareConfig")
        num_l2_tiles = _count_l2_tiles(num_pid_m, num_pid_n, config.tile_y, config.tile_x)
        if num_l2_tiles < 2:
            raise ValueError(
                f"l2_aware requires >= 2 L2 tiles, got {num_l2_tiles} "
                f"(grid: {num_pid_m}x{num_pid_n}, tile: {config.tile_y}x{config.tile_x})"
            )
        perm_table = compute_permutation_table(num_l2_tiles, config.seed, device=device)
        persistent_matmul_debug_map_l2_aware[(total_tiles,)](
            **common_kwargs,
            perm_table_ptr=perm_table,
            TILE_Y=config.tile_y,
            TILE_X=config.tile_x,
            INNER_ORDER=config.inner_ordering,
            CHUNK_SIZE=config.tile_y * config.tile_x,
        )

    elif strategy == "llc_and_l2_aware":
        if config is None:
            raise ValueError("llc_and_l2_aware strategy requires an LLCAndL2AwareConfig")
        persistent_matmul_debug_map_llc_and_l2_aware[(total_tiles,)](
            **common_kwargs,
            **config.to_kernel_kwargs(),
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Expected 'random', 'l2_aware', or 'llc_and_l2_aware'"
        )

    return workgroup_map.view(num_pid_m, num_pid_n)


def get_wg_mapping(
    m: int,
    n: int,
    k: int,
    strategy: str,
    config=None,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    selector=None,
) -> list:
    """
    Compute workgroup -> (tile_m, tile_n) mapping as a list of dicts.

    Returns [{"wgid": int, "tile_m": int, "tile_n": int}, ...] ordered by wgid.
    Compatible with layout_viewer schedule JSON format.
    """
    wg_map_2d = compute_workgroup_map(m, n, k, strategy, config, seed, dtype, selector)
    num_pid_m, num_pid_n = wg_map_2d.shape
    wg_map_flat = wg_map_2d.flatten().cpu()
    total = wg_map_flat.numel()

    mapping = [None] * total
    for lin_idx in range(total):
        wgid = int(wg_map_flat[lin_idx].item())
        mapping[wgid] = {
            "wgid": wgid,
            "tile_m": lin_idx // num_pid_n,
            "tile_n": lin_idx % num_pid_n,
        }
    return mapping

"""
High-level API for random grid GEMM scheduling.

This module provides user-facing functions that wrap the low-level Triton kernels.
"""

import torch
import triton
from typing import Optional

from .kernels import (
    persistent_matmul_shuffled,
    persistent_matmul_workgroup_shuffled,
    persistent_matmul_hierarchical,
    persistent_matmul_debug_map_shuffled,
    persistent_matmul_debug_map_workgroup_shuffled,
    persistent_matmul_debug_map_hierarchical,
)
from .config import HierarchicalPersistentConfig
from .utils import (
    _choose_lcg_params,
    _choose_lcg_params_allow_single_tile,
    _count_quantized_l2_tiles,
)


def _extract_config(selector):
    """
    Extract kernel configuration from a selector object.

    Supports both MatmulHeuristicResult (old fork) and OrigamiMatmulSelector (new fork)
    using duck typing.

    Args:
        selector: Selector object with block_m, block_n, block_k, group_m, num_sms properties

    Returns:
        Tuple of (block_m, block_n, block_k, group_m, num_sms)
    """
    try:
        # Try OrigamiMatmulSelector API
        return (
            selector.block_m,
            selector.block_n,
            selector.block_k,
            selector.group_m,
            selector.num_sms,
        )
    except AttributeError:
        # Try MatmulHeuristicResult API (old fork)
        try:
            return selector.get_config()
        except AttributeError:
            raise TypeError(
                f"Selector object must have block_m/block_n/block_k/group_m/num_sms "
                f"properties or a get_config() method. Got type: {type(selector)}"
            )


def _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, num_sms, even_k):
    """
    Create common kernel keyword arguments.

    Args:
        a, b, c: Input/output tensors
        M, N, K: Matrix dimensions
        BLK_M, BLK_N, BLK_K: Tile sizes
        num_sms: Number of streaming multiprocessors
        even_k: Whether K is evenly divisible by BLK_K

    Returns:
        Dictionary of kernel arguments
    """
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
        "NUM_XCDS": 8,  # MI300X has 8 XCDs
        "BIAS": False,
        "EVEN_K": even_k,
    }


def _setup_trace_buffers(device, total_tiles):
    """Allocate device buffers for kernel tracing."""
    return {
        "trace_start_ptr": torch.zeros(total_tiles, dtype=torch.int64, device=device),
        "trace_end_ptr": torch.zeros(total_tiles, dtype=torch.int64, device=device),
        "trace_pid_ptr": torch.zeros(total_tiles, dtype=torch.int32, device=device),
        "trace_xcd_ptr": torch.zeros(total_tiles, dtype=torch.int32, device=device),
    }


def _collect_trace_data(trace_bufs, total_tiles, total_programs, total_blocks_M, total_blocks_N,
                        M, N, K, BLK_M, BLK_N, BLK_K, gsize_m, num_xcds=8):
    """Synchronize and collect trace data from device buffers into a host dict."""
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
        "GROUP_SIZE_M": gsize_m,
        "NUM_XCDS": num_xcds,
    }


def matmul_random(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector=None,
    shuffle_seed: Optional[int] = None,
    trace: bool = False,
    **kwargs,
):
    """
    GEMM with random L2-aware tile scheduling.

    This mode uses a Linear Congruential Generator (LCG) to shuffle L2 tiles randomly,
    improving load balancing at high occupancy. Requires at least 2 L2 tiles in the
    quantized grid region.

    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N)
        c: Output matrix C (M x N), will be overwritten
        selector: Optional OrigamiMatmulSelector for tile configuration.
                  If None, auto-created based on problem size.
        shuffle_seed: Random seed for reproducible shuffling. If None, uses default RNG.
        trace: If True, collect per-tile timing/wgid/xcd data. Returns (c, trace_data).
        **kwargs: Additional kernel arguments (for future extensions)

    Returns:
        Output tensor c, or (c, trace_data) when trace=True

    Raises:
        ValueError: If the grid has fewer than 2 L2 tiles in quantized region

    Example:
        >>> a = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
        >>> c = torch.zeros(2048, 2048, dtype=torch.bfloat16, device='cuda')
        >>> matmul_random(a, b, c, shuffle_seed=42)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Auto-create selector if needed
    if selector is None:
        from ..origami import OrigamiMatmulSelector

        selector = OrigamiMatmulSelector(M, N, K, a.dtype, b.dtype, c.dtype, a.device)

    # Extract configuration
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    # Compute grid and tile information
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0
    num_l2_tiles = _count_quantized_l2_tiles(total_blocks_M, total_blocks_N, gsize_m)

    # Validate grid size for random scheduling
    if num_l2_tiles <= 1:
        raise ValueError(
            f"Random workgroup schedule requires at least two full L2 tiles in the quantized region. "
            f"Got {num_l2_tiles} L2 tiles (grid: {total_blocks_M}x{total_blocks_N}, group_size: {gsize_m})"
        )

    # Generate LCG parameters
    a_lcg, c_lcg = _choose_lcg_params(num_l2_tiles, seed=shuffle_seed)

    # Build kernel arguments
    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs.update({"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m})

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    # Launch kernel
    persistent_matmul_shuffled[(total_tiles,)](
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
            M, N, K, BLK_M, BLK_N, BLK_K, gsize_m,
        )
        return c, trace_data

    return c


def matmul_workgroup_shuffle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector=None,
    shuffle_seed: Optional[int] = None,
    trace: bool = False,
    **kwargs,
):
    """
    GEMM with global workgroup tile shuffling.

    This mode applies LCG shuffling to all tiles globally, regardless of L2 structure.
    Works with any grid size, including single-tile grids.

    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N)
        c: Output matrix C (M x N), will be overwritten
        selector: Optional OrigamiMatmulSelector for tile configuration
        shuffle_seed: Random seed for reproducible shuffling
        trace: If True, collect per-tile timing/wgid/xcd data. Returns (c, trace_data).
        **kwargs: Additional kernel arguments

    Returns:
        Output tensor c, or (c, trace_data) when trace=True

    Example:
        >>> a = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
        >>> c = torch.zeros(1024, 1024, dtype=torch.bfloat16, device='cuda')
        >>> matmul_workgroup_shuffle(a, b, c, shuffle_seed=123)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Auto-create selector if needed
    if selector is None:
        from ..origami import OrigamiMatmulSelector

        selector = OrigamiMatmulSelector(M, N, K, a.dtype, b.dtype, c.dtype, a.device)

    # Extract configuration
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    # Compute grid information
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    # Generate LCG parameters (handles single-tile grids gracefully)
    a_lcg, c_lcg = _choose_lcg_params_allow_single_tile(total_tiles, seed=shuffle_seed)

    # Build kernel arguments
    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs.update({"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m})

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    # Launch kernel
    persistent_matmul_workgroup_shuffled[(total_tiles,)](
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
            M, N, K, BLK_M, BLK_N, BLK_K, gsize_m,
        )
        return c, trace_data

    return c


def matmul_hierarchical(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    config: HierarchicalPersistentConfig,
    selector=None,
    trace: bool = False,
    **kwargs,
):
    """
    GEMM with hierarchical cache-aware tile scheduling.

    This mode uses a 3-level hierarchical traversal order specified by the config object.
    Useful for controlling cache locality at multiple levels of the memory hierarchy.

    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N)
        c: Output matrix C (M x N), will be overwritten
        config: HierarchicalPersistentConfig specifying the hierarchy parameters
        selector: Optional OrigamiMatmulSelector for tile configuration
        trace: If True, collect per-tile timing/wgid/xcd data. Returns (c, trace_data).
        **kwargs: Additional kernel arguments

    Returns:
        Output tensor c, or (c, trace_data) when trace=True

    Example:
        >>> from tritonblas.experimental.random_grid import HierarchicalPersistentConfig
        >>> config = HierarchicalPersistentConfig(
        ...     ordering0=0, ordering1=1, ordering2=2,
        ...     L3Y=4, L3X=4, L2Y=8, L2X=8
        ... )
        >>> a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
        >>> c = torch.zeros(4096, 4096, dtype=torch.bfloat16, device='cuda')
        >>> matmul_hierarchical(a, b, c, config=config)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Auto-create selector if needed
    if selector is None:
        from ..origami import OrigamiMatmulSelector

        selector = OrigamiMatmulSelector(M, N, K, a.dtype, b.dtype, c.dtype, a.device)

    # Extract configuration
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    # Compute grid information
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    # Build kernel arguments
    kernel_kwargs = _make_kernel_kwargs(a, b, c, M, N, K, BLK_M, BLK_N, BLK_K, total_tiles, even_k)
    kernel_kwargs.update(config.to_kernel_kwargs())

    if trace:
        trace_bufs = _setup_trace_buffers(a.device, total_tiles)
        kernel_kwargs.update(trace_bufs)
        kernel_kwargs["TRACE"] = True

    # Launch kernel
    persistent_matmul_hierarchical[(total_tiles,)](
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
            M, N, K, BLK_M, BLK_N, BLK_K, gsize_m,
        )
        return c, trace_data

    return c


def compute_workgroup_map(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.bfloat16,
    schedule_mode: str = "random",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
    selector=None,
):
    """
    Generate workgroup-to-tile mapping for visualization.

    This function doesn't compute a matrix multiplication - instead, it returns
    a 2D map showing which workgroup ID is assigned to each output tile. Useful
    for debugging and visualizing scheduling strategies.

    Args:
        m, n, k: Matrix dimensions
        dtype: Data type for selector creation
        schedule_mode: One of "random", "workgroup_shuffle", "hierarchical"
        shuffle_seed: Random seed (for random/workgroup_shuffle modes)
        hierarchical_config: Config object (for hierarchical mode)
        selector: Optional pre-created selector

    Returns:
        torch.Tensor of shape (num_pid_m, num_pid_n) with workgroup IDs

    Example:
        >>> wg_map = compute_workgroup_map(
        ...     2048, 2048, 2048, schedule_mode="random", shuffle_seed=42
        ... )
        >>> print(wg_map.shape)  # (num_tiles_m, num_tiles_n)
    """
    device = torch.device("cuda")

    # Create selector if needed
    if selector is None:
        from ..origami import OrigamiMatmulSelector

        selector = OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, device)

    # Extract configuration
    BLK_M, BLK_N, BLK_K, gsize_m, num_sms = _extract_config(selector)

    # Compute grid
    num_pid_m = triton.cdiv(m, BLK_M)
    num_pid_n = triton.cdiv(n, BLK_N)
    total_tiles = num_pid_m * num_pid_n

    # Allocate output map
    workgroup_map = torch.zeros((total_tiles,), dtype=torch.int32, device=device)

    # Select kernel and prepare arguments
    if schedule_mode == "random":
        num_l2_tiles = _count_quantized_l2_tiles(num_pid_m, num_pid_n, gsize_m)
        if num_l2_tiles <= 1:
            raise ValueError(
                f"Random mode requires at least 2 L2 tiles. "
                f"Got {num_l2_tiles} tiles (grid: {num_pid_m}x{num_pid_n}, group_size: {gsize_m})"
            )
        a_lcg, c_lcg = _choose_lcg_params(num_l2_tiles, seed=shuffle_seed)
        kernel = persistent_matmul_debug_map_shuffled
        kernel_kwargs = {"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m}

    elif schedule_mode == "workgroup_shuffle":
        a_lcg, c_lcg = _choose_lcg_params_allow_single_tile(total_tiles, seed=shuffle_seed)
        kernel = persistent_matmul_debug_map_workgroup_shuffled
        kernel_kwargs = {"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m}

    elif schedule_mode == "hierarchical":
        if hierarchical_config is None:
            raise ValueError("hierarchical_config required for hierarchical mode")
        kernel = persistent_matmul_debug_map_hierarchical
        kernel_kwargs = hierarchical_config.to_kernel_kwargs()

    else:
        raise ValueError(
            f"Unknown schedule_mode '{schedule_mode}'. "
            f"Expected 'random', 'workgroup_shuffle', or 'hierarchical'"
        )

    # Common kernel arguments
    kernel_kwargs.update(
        {
            "workgroup_map": workgroup_map,
            "M": m,
            "N": n,
            "BLOCK_SIZE_M": BLK_M,
            "BLOCK_SIZE_N": BLK_N,
            "NUM_SMS": total_tiles,
            "NUM_XCDS": 8,
        }
    )

    # Launch debug kernel
    kernel[(total_tiles,)](**kernel_kwargs)

    # Reshape to 2D grid
    return workgroup_map.view(num_pid_m, num_pid_n)


def get_wg_mapping(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.bfloat16,
    schedule_mode: str = "random",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
    selector=None,
) -> list:
    """
    Compute workgroup -> (tile_m, tile_n) mapping for random-grid strategies.

    Returns a list of dicts suitable for JSON serialization, matching the format
    of ``get_persistent_wg_mapping`` from ``tritonblas.matmul``: one entry per
    workgroup with keys wgid, tile_m, tile_n.

    Args:
        m, n, k: Matrix dimensions
        dtype: Data type for selector creation
        schedule_mode: One of "random", "workgroup_shuffle", "hierarchical"
        shuffle_seed: Random seed (for random/workgroup_shuffle modes)
        hierarchical_config: Config object (for hierarchical mode)
        selector: Optional pre-created selector

    Returns:
        List[Dict] with entries {"wgid": int, "tile_m": int, "tile_n": int},
        one per workgroup, ordered by wgid.
    """
    wg_map_2d = compute_workgroup_map(
        m, n, k, dtype, schedule_mode,
        shuffle_seed, hierarchical_config, selector,
    )
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

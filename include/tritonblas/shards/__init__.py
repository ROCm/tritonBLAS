"""
Composable shard abstractions for tritonblas.

Key abstractions:
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)
- tile_ptr: Device function for computing tile pointers (works with mixed types)
- GemmContext: Accumulator context for GEMM operations
- Grid: Encapsulates 2D grid setup and tile iteration
- ScaleView: Quantization scale view
- TensorView: Matrix view (for all-constexpr cases only)

Example usage (k_complete - simple):
    from tritonblas.shards import Grid, Tile, GemmContext, tile_ptr

    @triton.jit
    def kernel(A, B, C, M, N, K, stride_am, stride_bn, stride_cm, stride_cn, 
               stride_ak: tl.constexpr, stride_bk: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               EVEN_K: tl.constexpr, ...):
        
        grid = Grid(M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, num_xcds=NUM_XCDS)
        ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K, even_k=EVEN_K)
        
        start_tile, total_tiles = grid.get_tile_range()
        for tile_id in range(start_tile, total_tiles, grid.stride):
            pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
            out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
            
            # Full K loop in one call
            acc = ctx.k_complete(
                A, stride_am, stride_ak, M, K,
                B, stride_bk, stride_bn, N,
                out_tile,
            )
            
            c_ptrs, c_mask = tile_ptr(C, stride_cm, stride_cn, M, N, out_tile)
            tl.store(c_ptrs, acc, mask=c_mask)

Example usage (k_step - manual loop for advanced control):
    # For prefetching, overlapping, or custom iteration patterns
    ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K)
    acc = ctx.init_accumulator()
    
    for k_idx in range(num_k_tiles):
        acc = ctx.k_step(
            A, stride_am, stride_ak, M, K,
            B, stride_bk, stride_bn, N,
            out_tile, k_idx, acc,
            boundary=(k_idx == num_k_tiles - 1 and not EVEN_K),
        )
"""

# Aggregate classes
from .tile import Tile
from .tensor_view import TensorView, tile_ptr, tile_ptr_1d, make_tile_ptr
from .gemm_context import GemmContext
from .grid import Grid, GemmGrid
from .scale_view import ScaleView

# Device functions (work with any Triton version)
from .core import (
    tile_layout,
    tile_coords,
    chiplet_pid,
    grid_info,
    gemm_k_loop,
    apply_scales,
)

__all__ = [
    # Aggregate classes (require @triton.constexpr_function support)
    'Tile',
    'TensorView',
    'GemmContext',
    'Grid',
    'GemmGrid',
    'ScaleView',
    # Device functions (work with any type combinations)
    'tile_ptr',
    'tile_ptr_1d',
    'make_tile_ptr',
    # Legacy device functions from core.py
    'tile_layout',
    'tile_coords',
    'chiplet_pid',
    'grid_info',
    'gemm_k_loop',
    'apply_scales',
]

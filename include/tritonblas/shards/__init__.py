"""
Composable shard abstractions for tritonblas.

Key abstractions:
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)
- TensorView: Matrix view with tile_ptr(tile) returning (ptrs, mask)
- GemmContext: Accumulator context for GEMM operations
- Grid: Encapsulates 2D grid setup and tile iteration
- ScaleView: Quantization scale view

Example usage:
    from tritonblas.shards import Grid, Tile, TensorView, GemmContext

    @triton.jit
    def kernel(A, B, C, M, N, K, ...):
        # Tensor views
        A_view = TensorView(A, stride_am, stride_ak, M, K)
        B_view = TensorView(B, stride_bk, stride_bn, K, N)
        C_view = TensorView(C, stride_cm, stride_cn, M, N)
        
        # Grid for iteration
        grid = Grid(M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS)
        
        for tile_id in range(grid.start_tile, grid.total_tiles, grid.stride):
            pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
            
            # Output tile
            out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
            
            # GEMM context
            ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K)
            
            # Execute full GEMM loop (K inferred from views)
            acc = ctx.execute(A_view, B_view, out_tile)
            
            # Store result
            c_ptrs, c_mask = C_view.tile_ptr(out_tile)
            tl.store(c_ptrs, acc, mask=c_mask)
"""

# Aggregate classes
from .tile import Tile
from .tensor_view import TensorView
from .gemm_context import GemmContext
from .grid import Grid, GemmGrid
from .scale_view import ScaleView

# Helper functions
from .core import tile_layout, tile_coords

__all__ = [
    # Aggregate classes
    'Tile',
    'TensorView',
    'GemmContext',
    'Grid',
    'GemmGrid',
    'ScaleView',
    # Helper functions
    'tile_layout',
    'tile_coords',
]

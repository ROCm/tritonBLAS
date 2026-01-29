"""
Composable shard abstractions for tritonblas.

Key abstractions:
- ScheduleContext: Unified scheduling that hides loop complexity with tile_range()/get_tile()
- InputView, OutputView: Generic matrix views with load()/store() methods
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)
- GemmContext: Accumulator context with gemm()/k_complete() methods
- Grid: Encapsulates 2D grid setup and tile iteration (legacy, use ScheduleContext)

Example usage (ScheduleContext - recommended):
    from tritonblas.shards import ScheduleContext, Tile, GemmContext, InputView, tile_ptr

    @triton.jit
    def kernel(A, B, C, M, N, K, stride_am, stride_bn, stride_cm, stride_cn, 
               stride_ak: tl.constexpr, stride_bk: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               EVEN_K: tl.constexpr, ...):
        
        # Create matrix views
        tensorA = InputView(A, stride_am, stride_ak, M, K)
        tensorB = InputView(B, stride_bk, stride_bn, K, N)
        
        sched = ScheduleContext(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, 
                                GROUP_SIZE_M, NUM_SMS, num_xcds=NUM_XCDS)
        ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K, even_k=EVEN_K)
        
        start, total, stride = sched.tile_range()
        for tile_id in range(start, total, stride):
            pid_m, pid_n = sched.get_tile(tile_id)
            out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
            
            acc = ctx.k_complete(tensorA, tensorB, out_tile)
            
            c_ptrs, c_mask = tile_ptr(C, stride_cm, stride_cn, M, N, out_tile)
            tl.store(c_ptrs, acc, mask=c_mask)

Example usage (Stream-K with iter_range):
    sched = ScheduleContext(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                            GROUP_SIZE_M, NUM_SMS, streamk_tiles=STREAMK_TILES)
    
    start, end = sched.iter_range()
    for iter_id in range(start, end):
        pid_m, pid_n, k_iter = sched.get_iter(iter_id)
        # Process single K iteration at (pid_m, pid_n, k_iter)
        ...
"""

# Aggregate classes
from .tile import Tile
from .tensor_view import TensorView, tile_ptr, tile_ptr_1d, make_tile_ptr
from .gemm_context import GemmContext
from .grid import Grid, GemmGrid, chiplet_transform, chiplet_transform_chunked
from .scale_view import ScaleView
from .schedule import ScheduleContext
from .matrix_view import (
    InputView,
    OutputView,
    # Legacy aliases
    A_View,
    B_View,
    C_View,
    InputTensorA,
    InputTensorB,
    MatrixView,
)

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
    'ScheduleContext',
    'InputView',
    'OutputView',
    'Tile',
    'TensorView',
    'GemmContext',
    'Grid',
    'GemmGrid',
    'ScaleView',
    # Legacy aliases for matrix views
    'A_View',
    'B_View',
    'C_View',
    # Generic tensor device functions
    'tile_ptr',
    'tile_ptr_1d',
    'make_tile_ptr',
    'chiplet_transform',
    'chiplet_transform_chunked',
    # Legacy device functions from core.py
    'tile_layout',
    'tile_coords',
    'chiplet_pid',
    'grid_info',
    'gemm_k_loop',
    'apply_scales',
    # Legacy aliases
    'InputTensorA',
    'InputTensorB',
    'MatrixView',
]

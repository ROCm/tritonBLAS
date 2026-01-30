"""
Composable shard abstractions for tritonblas.

Key abstractions:
- GemmConfig: Bundle all performance parameters (block sizes, num_sms, etc.)
- ScheduleContext: Unified scheduling that hides loop complexity with tile_range()/get_tile()
- GemmContext: Accumulator context with gemm()/k_complete() methods
- InputView, OutputView: Generic matrix views with tile_ptrs()/load()/store() methods
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)
- Grid: Encapsulates 2D grid setup and tile iteration (legacy, use ScheduleContext)

Example usage (recommended):
    from tritonblas.shards import (
        GemmConfig, ScheduleContext, GemmContext, 
        Tile, InputView, OutputView
    )

    @triton.jit
    def kernel(A, B, C, M, N, K, stride_am, stride_bn, stride_cm, stride_cn, 
               stride_ak: tl.constexpr, stride_bk: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
               EVEN_K: tl.constexpr, ...):
        
        # Create matrix views
        tensorA = InputView(A, stride_ak, stride_am, M, K)
        tensorB = InputView(B, stride_bk, stride_bn, K, N)
        tensorC = OutputView(C, stride_cm, stride_cn, M, N)
        
        # Construct GemmConfig on device with ALL parameters
        config = GemmConfig(
            BLOCK_M, BLOCK_N, BLOCK_K, NUM_SMS, NUM_XCDS, GROUP_SIZE_M,
            even_k=EVEN_K,  # computation options go in config too
        )
        
        # Create contexts from GemmConfig - just pass config
        sched = ScheduleContext(M, N, K, config)
        ctx = GemmContext(config)  # no extra args needed
        
        # Persistent loop
        start, total, stride = sched.persistent_tile_range()
        for tile_id in range(start, total, stride):
            pid_m, pid_n = sched.get_tile(tile_id)
            out_tile = Tile(pid_m, pid_n, config.block_m, config.block_n)
            
            acc = ctx.k_complete(tensorA, tensorB, out_tile)
            
            c_ptrs, c_mask = tensorC.tile_ptrs(out_tile)
            tl.store(c_ptrs, acc.to(C.type.element_ty), mask=c_mask)

Example usage (Stream-K with iter_range):
    # Construct GemmConfig on device
    config = GemmConfig(BLOCK_M, BLOCK_N, BLOCK_K, NUM_SMS, NUM_XCDS, GROUP_SIZE_M)
    sched = ScheduleContext(M, N, K, config, streamk_tiles=STREAMK_TILES)
    
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
from .gemm_config import GemmConfig
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
    'GemmConfig',
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

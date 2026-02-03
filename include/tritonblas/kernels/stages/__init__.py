"""
Composable stage abstractions for tritonblas GEMM kernels.

Key abstractions:
- GemmContext: Accumulator context with reduce_axis() plus all config parameters
- ScheduleContext: Unified scheduling with persistent_tile_range()/get_tile()
- InputView, OutputView: Matrix views with tile_ptrs() for memory access
- ScaleView, BiasView: Epilogue views for quantized GEMM (scale and bias)
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)

Factory functions:
- make_input_view, make_tensor_view: Create InputView for A and B matrices
- make_output_view: Create OutputView for C matrix with epilogue support
- make_scale_view: Create ScaleView for quantization scales
- make_bias_view: Create BiasView for bias addition

Example usage:
    from tritonblas.kernels.stages import (
        ScheduleContext, GemmContext, 
        make_tensor_view, make_output_view,
        make_scale_view, make_bias_view,
    )

    @triton.jit
    def kernel(A, B, C, A_scale_ptr, B_scale_ptr, bias_ptr, M, N, K, 
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               stride_bias, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
               BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
               NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
               EVEN_K: tl.constexpr, BIAS: tl.constexpr, ...):
        
        # Create matrix views - just describe your matrices
        tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
        tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
        
        # Create epilogue views (optional scale and bias)
        scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N) if A_scale_ptr is not None else None
        bias_view = make_bias_view(bias_ptr, M, stride_bias) if BIAS else None
        
        # Construct GemmContext on device with ALL parameters
        ctx = GemmContext(
            BLOCK_M, BLOCK_N, BLOCK_K, NUM_SMS, NUM_XCDS, GROUP_SIZE_M,
            even_k=EVEN_K,
        )
        
        # Create schedule from GemmContext
        sched = ScheduleContext(M, N, K, ctx)
        
        # Persistent loop
        start, total, stride = sched.persistent_tile_range()
        for tile_id in range(start, total, stride):
            out_tile = sched.get_tile_from_idx(tile_id)
            
            # Compute GEMM
            acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
            
            # Store with epilogue: scale -> bias -> convert -> store
            tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
"""

# Core aggregates
from .tile import Tile
from .gemm_context import GemmContext
from .schedule import ScheduleContext
from .matrix_view import (
    InputView, OutputView, ScaleView, BiasView,
    make_input_view, make_tensor_view, make_output_view,
    make_scale_view, make_bias_view,
)

# Grid utilities (used by streamk_gemm, fp4_matmul, persistent_gemm_monolithic)
from .grid import chiplet_transform, chiplet_transform_chunked

__all__ = [
    # Core aggregates
    'Tile',
    'GemmContext',
    'ScheduleContext',
    'InputView',
    'OutputView',
    'ScaleView',
    'BiasView',
    # Factory functions
    'make_input_view',
    'make_tensor_view',
    'make_output_view',
    'make_scale_view',
    'make_bias_view',
    # Grid utilities
    'chiplet_transform',
    'chiplet_transform_chunked',
]

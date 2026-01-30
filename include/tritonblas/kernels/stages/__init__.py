"""
Composable stage abstractions for tritonblas GEMM kernels.

Key abstractions:
- GemmConfig: Bundle all performance parameters (block sizes, num_sms, etc.)
- ScheduleContext: Unified scheduling with persistent_tile_range()/get_tile()
- GemmContext: Accumulator context with k_complete() method
- InputView, OutputView: Matrix views with tile_ptrs() for memory access
- Tile: 2D tile with coordinates (pid_m, pid_n) and shape (block_m, block_n)

Example usage:
    from tritonblas.kernels.stages import (
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
            even_k=EVEN_K,
        )
        
        # Create contexts from GemmConfig
        sched = ScheduleContext(M, N, K, config)
        ctx = GemmContext(config)
        
        # Persistent loop
        start, total, stride = sched.persistent_tile_range()
        for tile_id in range(start, total, stride):
            pid_m, pid_n = sched.get_tile(tile_id)
            out_tile = Tile(pid_m, pid_n, config.block_m, config.block_n)
            
            acc = ctx.k_complete(tensorA, tensorB, out_tile)
            
            c_ptrs, c_mask = tensorC.tile_ptrs(out_tile)
            tl.store(c_ptrs, acc.to(C.type.element_ty), mask=c_mask)
"""

# Core aggregates
from .tile import Tile
from .gemm_context import GemmContext
from .gemm_config import GemmConfig
from .schedule import ScheduleContext
from .matrix_view import InputView, OutputView

# Grid utilities (used by streamk_gemm, fp4_matmul, persistent_gemm_monolithic)
from .grid import chiplet_transform, chiplet_transform_chunked

__all__ = [
    # Core aggregates
    'Tile',
    'GemmConfig',
    'GemmContext',
    'ScheduleContext',
    'InputView',
    'OutputView',
    # Grid utilities
    'chiplet_transform',
    'chiplet_transform_chunked',
]

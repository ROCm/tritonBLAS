import triton
import triton.language as tl
import torch

from .shards import prologue_global, gemm_loop_tile, epilogue_tile

@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """
    Persistent matmul kernel using composable shards.
    
    This kernel has been refactored to use the tritonblas.shards module for better
    code reuse and maintainability. The functionality remains identical to the
    original implementation.
    """
    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Determine output dtype for accumulator
    OUTPUT_IS_INT8 = C.type.element_ty == tl.int8
    
    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # Global prologue: setup and get tile grid info
    pid, num_pid_m, num_pid_n, total_tiles = prologue_global(
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        NUM_SMS, NUM_XCDS, CHUNK_SIZE,
        USE_CHIPLET_PID
    )

    # Persistent loop: process multiple tiles per thread block
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # GEMM loop: compute the tile
        acc, rm, rn, pid_m, pid_n = gemm_loop_tile(
            A, B, M, N, K,
            stride_am, stride_bn, stride_ak, stride_bk,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            CACHE_MODIFIER_A, CACHE_MODIFIER_B,
            GROUP_SIZE_M,
            QUANTIZED, ALLOW_TF32,
            OUTPUT_IS_INT8,
            EVEN_K,
            tile_id, num_pid_m, num_pid_n
        )
        
        # Epilogue: post-process and store
        # Note: The shard handles bias_ptr=None correctly, so we pass it directly
        # regardless of the BIAS flag
        epilogue_tile(
            acc, rm, rn, pid_m, pid_n,
            C, A_scale_ptr, B_scale_ptr, bias_ptr if BIAS else None,
            M, N, stride_cm, stride_cn, stride_bias,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
            QUANTIZED
        )

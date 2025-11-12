"""
GEMM loop functions for composable Triton GEMM kernels.
Handles the main matrix multiplication loop including K-dimension tiling.
"""
import triton
import triton.language as tl

from .utils import dot_acc, load_block
from .prologue import tile_coords, make_bases


@triton.jit
def gemm_loop_tile(
    A, B,
    M, N, K,
    stride_am, stride_bn,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr, CACHE_MODIFIER_B: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    QUANTIZED: tl.constexpr, ALLOW_TF32: tl.constexpr,
    OUTPUT_IS_INT8: tl.constexpr,
    EVEN_K: tl.constexpr,
    tile_id, num_pid_m, num_pid_n,
):
    """
    Execute GEMM loop for a single tile, including K-dimension iteration and tail handling.
    
    Args:
        A, B: Input matrix pointers
        M, N, K: Matrix dimensions
        stride_am, stride_bn: Strides for A (M dim) and B (N dim)
        stride_ak, stride_bk: Strides for A and B in K dimension
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes
        CACHE_MODIFIER_A, CACHE_MODIFIER_B: Cache modifiers for loads
        GROUP_SIZE_M: Number of M tiles to group together
        QUANTIZED: Whether using quantized inputs
        ALLOW_TF32: Whether to allow TF32 precision
        OUTPUT_IS_INT8: Whether output is int8 (determines accumulator dtype)
        tile_id: Linear tile index
        num_pid_m, num_pid_n: Number of tiles in M and N dimensions
    
    Returns:
        Tuple of (acc, rm, rn, pid_m, pid_n) where:
            acc: Accumulated result for this tile
            rm: Row indices for this tile
            rn: Column indices for this tile
            pid_m, pid_n: Tile coordinates
    """
    pid_m, pid_n = tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    
    rm, rn, A_BASE, B_BASE = make_bases(
        A, B, pid_m, pid_n, M, N,
        stride_am, stride_bn, stride_ak, stride_bk,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    tl.assume(loop_k > 1)

    # Initialize accumulator - always use float32 for compatibility
    # The original uses acc_dtype computed at kernel level, but in a function
    # we need consistent types. The conversion to output dtype happens in epilogue.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop: iterate over K dimension in BLOCK_SIZE_K chunks
    for _ in range(0, loop_k):
        a, b = load_block(A_BASE, B_BASE, stride_ak, stride_bk, CACHE_MODIFIER_A, CACHE_MODIFIER_B)
        acc = dot_acc(acc, a, b, QUANTIZED, ALLOW_TF32)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    # Tail handling: process remaining K elements with masking if K is not evenly divisible
    if not EVEN_K:
        k0 = loop_k * BLOCK_SIZE_K
        rk = k0 + tl.arange(0, BLOCK_SIZE_K)
        A_T = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_T = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if stride_ak == 1:
            A_T = tl.multiple_of(A_T, (1, 16))
        else:
            A_T = tl.multiple_of(A_T, (16, 1))
        if stride_bk == 1:
            B_T = tl.multiple_of(B_T, (16, 1))
        else:
            B_T = tl.multiple_of(B_T, (1, 16))

        a = tl.load(A_T, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(B_T, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
        acc = dot_acc(acc, a, b, QUANTIZED, ALLOW_TF32)

    return acc, rm, rn, pid_m, pid_n

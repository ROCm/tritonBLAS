"""
GEMM loop algorithm for composable Triton GEMM kernels.

This module provides the core GEMM loop that iterates over the K dimension,
loading tiles and accumulating the matrix multiplication result.
"""

import triton
import triton.language as tl

from ..memory import load
from . import multiply_accumulate


@triton.jit
def gemm_loop(
    A,
    B,
    row_indices,
    col_indices,
    acc,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    BLOCK_SIZE_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Execute the main GEMM loop over the K dimension.

    This function encapsulates the loads, and MAC operations of a gemm loop over K.

    Handles both the main loop and K tail (if K is not evenly divisible by BLOCK_SIZE_K).

    Parameters:
    -----------
    A : tensor
        Input matrix A
    B : tensor
        Input matrix B
    row_indices : tensor
        Row indices for the current tile
    col_indices : tensor
        Column indices for the current tile
    acc : tensor
        Accumulator tensor (will be updated in-place conceptually)
    M : int
        Total M dimension size (for boundary masking on A)
    N : int
        Total N dimension size (for boundary masking on B)
    K : int
        Total K dimension size
    stride_am : int
        Stride for A in M dimension
    stride_ak : int
        Stride for A in K dimension
    stride_bn : int
        Stride for B in N dimension
    stride_bk : int
        Stride for B in K dimension
    BLOCK_SIZE_K : constexpr int
        Block size in K dimension
    CACHE_MODIFIER_A : constexpr
        Cache modifier for loading A
    CACHE_MODIFIER_B : constexpr
        Cache modifier for loading B
    QUANTIZED : constexpr bool
        Whether using quantized computation
    ALLOW_TF32 : constexpr bool
        Whether to allow TF32 computation
    EVEN_K : constexpr bool
        Whether K is evenly divisible by BLOCK_SIZE_K

    Returns:
    --------
    acc : tensor
        Updated accumulator after all K iterations
    """
    # Compute loop bounds
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    tl.assume(loop_k > 0)

    # Main loop over K dimension
    for k_iter in range(loop_k):
        k0 = k_iter * BLOCK_SIZE_K

        # Load - Address math + global -> CU load
        # Pass M/N for boundary masking on partial tiles
        a = load(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=False, is_row_major=True, major_dim_size=M)
        b = load(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=False, is_row_major=False, major_dim_size=N)

        # Compute - Math only
        acc = multiply_accumulate(acc, a, b, QUANTIZED, ALLOW_TF32)

    # Handle K tail if needed
    if not EVEN_K:
        k0 = loop_k * BLOCK_SIZE_K

        # Load with masking
        a = load(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=True, is_row_major=True, major_dim_size=M)
        b = load(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=True, is_row_major=False, major_dim_size=N)

        # Compute
        acc = multiply_accumulate(acc, a, b, QUANTIZED, ALLOW_TF32)

    return acc

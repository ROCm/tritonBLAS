"""
GEMM loop algorithm for composable Triton GEMM kernels.

This module provides the core GEMM loop that iterates over the K dimension,
loading tiles and accumulating the matrix multiplication result.
"""

import triton
import triton.language as tl


@triton.jit
def gemm_loop(
    A,
    B,
    row_indices,
    col_indices,
    acc,
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
    load_func: tl.constexpr,
    multiply_accumulate_func: tl.constexpr,
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
    load_func : constexpr function
        Function to use for loading tiles
    multiply_accumulate_func : constexpr function
        Function to use for multiply-accumulate operation
    
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
        
        # Load - Address math + global → CU load
        a = load_func(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=False, is_row_major=True)
        b = load_func(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=False, is_row_major=False)
        
        # Compute - Math only
        acc = multiply_accumulate_func(acc, a, b, QUANTIZED, ALLOW_TF32)
    
    # Handle K tail if needed
    if not EVEN_K:
        k0 = loop_k * BLOCK_SIZE_K
        
        # Load with masking
        a = load_func(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=True, is_row_major=True)
        b = load_func(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=True, is_row_major=False)
        
        # Compute
        acc = multiply_accumulate_func(acc, a, b, QUANTIZED, ALLOW_TF32)
    
    return acc

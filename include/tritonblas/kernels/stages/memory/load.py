"""
Load phase for composable Triton GEMM kernels.
Phase 2: Address math + global → register movement.
"""
import triton
import triton.language as tl


@triton.jit
def load(
    matrix_ptr,
    indices, k0,
    stride_major, stride_k: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    K,
    CACHE_MODIFIER: tl.constexpr,
    mask_k: tl.constexpr = False,
    is_row_major: tl.constexpr = True,
):
    """
    Load a single tile from global memory.
    
    This function loads one tile (either A or B) from global memory.
    
    Args:
        matrix_ptr: Pointer to the matrix
        indices: Row indices (for A) or column indices (for B)
        k0: Starting K offset for this iteration
        stride_major: Stride in the major dimension (M for A, N for B)
        stride_k: Stride in K dimension
        BLOCK_SIZE_K: Tile size in K dimension
        K: Total K dimension (for masking)
        CACHE_MODIFIER: Cache modifier for load
        mask_k: Whether to apply K-dimension masking (for tail handling)
        is_row_major: Whether this is a row-major load (A=True) or column-major (B=False)
    
    Returns:
        Loaded tile: [BLOCK_SIZE_M, BLOCK_SIZE_K] for A or [BLOCK_SIZE_K, BLOCK_SIZE_N] for B
    """
    # Compute K indices
    rk = k0 + tl.arange(0, BLOCK_SIZE_K)
    
    # Compute addresses based on layout
    if is_row_major:
        # For A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        ptrs = matrix_ptr + indices[:, None] * stride_major + rk[None, :] * stride_k
        # Apply alignment hints
        if stride_k == 1:
            ptrs = tl.multiple_of(ptrs, (1, 16))
        else:
            ptrs = tl.multiple_of(ptrs, (16, 1))
        # Load with optional K masking
        if mask_k:
            tile = tl.load(ptrs, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
        else:
            tile = tl.load(ptrs, cache_modifier=CACHE_MODIFIER)
    else:
        # For B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        ptrs = matrix_ptr + rk[:, None] * stride_k + indices[None, :] * stride_major
        # Apply alignment hints
        if stride_k == 1:
            ptrs = tl.multiple_of(ptrs, (16, 1))
        else:
            ptrs = tl.multiple_of(ptrs, (1, 16))
        # Load with optional K masking
        if mask_k:
            tile = tl.load(ptrs, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER)
        else:
            tile = tl.load(ptrs, cache_modifier=CACHE_MODIFIER)
    
    return tile

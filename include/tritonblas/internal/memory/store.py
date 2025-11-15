"""
Store phase for composable Triton GEMM kernels.
Phase 5: Register → global movement - write results back to memory.
"""
import triton
import triton.language as tl


@triton.jit
def store(
    C, result,
    row_indices, col_indices,
    M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Phase 5: Store - Write tile from registers to global memory.
    
    This phase handles all write-back logic - address computation and stores.
    All write-back logic lives here, separate from math.
    
    Args:
        C: Output matrix pointer
        result: Tile data to store [BLOCK_SIZE_M, BLOCK_SIZE_N]
        row_indices: Row indices [BLOCK_SIZE_M] (already formatted with max_contiguous/multiple_of)
        col_indices: Column indices [BLOCK_SIZE_N] (already formatted with max_contiguous/multiple_of)
        M, N: Matrix dimensions
        stride_cm, stride_cn: Strides for C in M and N dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
    """
    # Compute output addresses
    C_ptrs = C + row_indices[:, None] * stride_cm + col_indices[None, :] * stride_cn
    
    # Apply boundary mask
    mask = (row_indices[:, None] < M) & (col_indices[None, :] < N)
    
    # Store to global memory
    tl.store(C_ptrs, result, mask=mask)

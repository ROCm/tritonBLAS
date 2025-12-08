import triton
import triton.language as tl
import torch

# @triton.jit
# def build_H(SIZE: tl.constexpr, dtype: tl.constexpr):
#     r"""
#     Construct small Hadamard matrices, in such a way that Triton can optimize the code away.
#     This uses the identity $H_{i,j} = (-1)^{i \cdot j}$, 
#     where the operation $\cdot$ is the BITWISE dot product of integers.
#     """
#     tl.static_assert(0 < SIZE)
#     tl.static_assert(SIZE <= 64)

#     i = tl.arange(0, SIZE)
#     j = tl.arange(0, SIZE)    
#     matching_bits = (i[:, None] & j)

#     bit_sum = tl.zeros_like(matching_bits)
#     for i in tl.static_range(5):
#         bit_sum += matching_bits & 1
#         matching_bits >>= 1

#     # map odd to -1, even to 1
#     H = 2 * ((bit_sum % 2) == 0) - 1
#     return H.cast(dtype)

@triton.jit
def build_H(SIZE: tl.constexpr, dtype: tl.constexpr):
    r"""
    Construct Hadamard matrix using H_{i,j} = (-1)^{popcount(i & j)}.
    
    This computes the bitwise dot product of row index i and column index j:
    - popcount(i & j) counts the number of matching 1-bits
    - If count is even: H_{i,j} = 1
    - If count is odd: H_{i,j} = -1
    
    Args:
        SIZE: Matrix dimension (must be power of 2, max 64)
        dtype: Output data type
        
    Returns:
        SIZE x SIZE Hadamard matrix
    """
    tl.static_assert(0 < SIZE)
    tl.static_assert(SIZE <= 64)
    
    # Create row and column indices
    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)
    
    # Compute bitwise AND for all (i, j) pairs
    matching_bits = i[:, None] & j[None, :]
    
    # Count set bits (popcount) - simple iterative approach
    bit_sum = tl.zeros_like(matching_bits)
    temp = matching_bits
    for _ in tl.static_range(6):  # 6 iterations for up to 64 bits
        bit_sum += temp & 1
        temp >>= 1
    
    # Map: even popcount -> +1, odd popcount -> -1
    H = 1 - 2 * (bit_sum & 1)
    H = H.to(dtype)
    norm_factor = 1.0 / tl.math.sqrt(float(SIZE))
    H = H * norm_factor
    
    return H


@triton.jit
def hadamard_blocked_kernel(
    A_ptr,           # Pointer to input matrix A [M, K]
    Out_ptr,         # Pointer to output matrix [M, K]
    M,               # Number of rows in A
    K,               # Number of columns in A
    stride_am,       # Stride of A in M dimension
    stride_ak,       # Stride of A in K dimension
    stride_om,       # Stride of output in M dimension
    stride_ok,       # Stride of output in K dimension
    BLOCK_SIZE: tl.constexpr,  # Block size (32)
):
    """
    Kernel that applies Hadamard transformation to each 32x32 block of A.
    
    Each program processes one 32x32 block independently:
    Output[m_block, k_block] = A[m_block, k_block] @ H
    """
    # Get program IDs for M and K dimensions
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Compute starting indices for this block
    m_start = pid_m * BLOCK_SIZE
    k_start = pid_k * BLOCK_SIZE
    
    # Create offset ranges
    m_offs = m_start + tl.arange(0, BLOCK_SIZE)
    k_offs = k_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for boundary conditions
    m_mask = m_offs < M
    k_mask = k_offs < K
    
    # Load A block [BLOCK_SIZE, BLOCK_SIZE]
    a_ptrs = A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
    a_block = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Materialize Hadamard matrix [BLOCK_SIZE, BLOCK_SIZE]
    h_block = build_H(BLOCK_SIZE, a_block.dtype)
    
    # Perform matrix multiplication: A_block @ H_block
    # This is a single 32x32 @ 32x32 operation
    result = tl.dot(a_block, h_block)
    
    # Store result to output
    out_ptrs = Out_ptr + m_offs[:, None] * stride_om + k_offs[None, :] * stride_ok
    tl.store(out_ptrs, result, mask=m_mask[:, None] & k_mask[None, :])


def hadamard_blocked_fast(A: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transformation to each 32x32 block of matrix A.
    
    Args:
        A: Input matrix of shape [M, K]
    
    Returns:
        Output matrix of shape [M, K] with each 32x32 block transformed
    """
    assert A.is_cuda,  "Tensors must be on CUDA"
    
    M, K = A.shape
    
    # Allocate output with same shape as A
    Out = torch.zeros_like(A)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions - one program per 32x32 block
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE))
    
    # Launch kernel
    hadamard_blocked_kernel[grid](
        A, Out,
        M, K,
        A.stride(0), A.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return Out
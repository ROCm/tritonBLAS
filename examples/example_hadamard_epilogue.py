"""
Example demonstrating Hadamard rotation epilogue in tritonBLAS.

This example shows how to:
1. Define a custom Hadamard rotation epilogue function
2. Apply it to the GEMM output accumulator
3. Use it for randomized numerical linear algebra

This is useful for privacy-preserving computations and randomized algorithms.
"""
import torch
import triton
import triton.language as tl
from include.tritonblas.kernels.persistent_gemm import persistent_matmul


# ============================================================================
# Define Hadamard Rotation Epilogue
# ============================================================================

@triton.jit
def build_hadamard(SIZE: tl.constexpr):
    """
    Construct Hadamard matrix using H_{i,j} = (-1)^{popcount(i & j)}.
    
    This computes the bitwise dot product of row index i and column index j:
    - popcount(i & j) counts the number of matching 1-bits
    - If count is even: H_{i,j} = 1
    - If count is odd: H_{i,j} = -1
    
    Args:
        SIZE: Matrix dimension (must be power of 2, 16 <= SIZE <= 64)
        
    Returns:
        SIZE x SIZE normalized Hadamard matrix
    """
    tl.static_assert(16 <= SIZE)
    tl.static_assert(SIZE <= 64)

    # Create row and column indices
    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)

    # Compute bitwise AND for all (i, j) pairs
    matching_bits = i[:, None] & j[None, :]

    # Count set bits (popcount) - iterative approach
    bit_sum = tl.zeros_like(matching_bits)
    temp = matching_bits
    for _ in tl.static_range(6):  # 6 iterations for up to 64 bits
        bit_sum += temp & 1
        temp >>= 1

    # Map: even popcount -> +1, odd popcount -> -1
    H = 1 - 2 * (bit_sum & 1)

    # Normalize by sqrt(SIZE)
    H = H / tl.math.sqrt(float(SIZE))
    return H


@triton.jit
def hadamard_rotation(acc, BLOCK_SIZE: tl.constexpr = 16):
    """
    Apply Hadamard rotation to the accumulator in blocks.
    
    This epilogue applies a Hadamard transformation to blocks of the accumulator:
    For each BLOCK_SIZE x BLOCK_SIZE block: result = block @ H
    
    Constraints:
    - BLOCK_SIZE must be a power of 2
    - 16 <= BLOCK_SIZE <= 64
    - BLOCK_SIZE must evenly divide both accumulator dimensions
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        BLOCK_SIZE: Size of Hadamard blocks (default: 32)
    
    Returns:
        Accumulator with Hadamard rotation applied to each block
    """
    # Get accumulator dimensions
    M = acc.shape[0]
    N = acc.shape[1]
    
    # Static assertions for valid block size
    tl.static_assert(16 <= BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE <= 64)
    tl.static_assert(BLOCK_SIZE <= M)
    tl.static_assert(BLOCK_SIZE <= N)
    
    # Build Hadamard matrix once
    H = build_hadamard(BLOCK_SIZE)
    
    # Process each block
    result = tl.zeros_like(acc)
    
    # Iterate over blocks in M dimension
    for m_block in tl.static_range(M // BLOCK_SIZE):
        m_start = m_block * BLOCK_SIZE
        m_end = m_start + BLOCK_SIZE
        
        # Iterate over blocks in N dimension
        for n_block in tl.static_range(N // BLOCK_SIZE):
            n_start = n_block * BLOCK_SIZE
            n_end = n_start + BLOCK_SIZE
            
            # Extract block
            block = acc[m_start:m_end, n_start:n_end]
            
            # Apply Hadamard: block @ H
            rotated = tl.dot(block, H.to(block.dtype))
            
            # Store result
            result[m_start:m_end, n_start:n_end] = rotated
    
    return result


def matmul_with_hadamard(A, B, block_size=32):
    """
    Perform matrix multiplication with Hadamard rotation epilogue.
    
    Args:
        A: Input matrix A [M, K]
        B: Input matrix B [K, N] (transposed)
        block_size: Size of Hadamard blocks (must be 16, 32, or 64)
    
    Returns:
        Output matrix C [M, N] with Hadamard rotation applied
    """
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device="cuda", dtype=A.dtype)
    
    # Get device properties
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    # Block sizes must be compatible with Hadamard block size
    # For this example, we use 128x128 tiles which are divisible by 32
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Verify dimensions are compatible
    assert BLOCK_SIZE_M % block_size == 0, f"BLOCK_SIZE_M ({BLOCK_SIZE_M}) must be divisible by block_size ({block_size})"
    assert BLOCK_SIZE_N % block_size == 0, f"BLOCK_SIZE_N ({BLOCK_SIZE_N}) must be divisible by block_size ({block_size})"
    
    # Define grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Create epilogue function with specific block size
    @triton.jit
    def hadamard_epilogue(acc):
        return hadamard_rotation(acc, BLOCK_SIZE=block_size)
    
    # Launch kernel with Hadamard epilogue
    persistent_matmul[grid](
        A, B, C,
        None, None,  # No quantization scales
        A,           # Dummy bias pointer (not used)
        M, N, K,
        A.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        0,  # stride_bias (not used)
        A.stride(1), B.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=num_sms,
        NUM_XCDS=1,
        CHUNK_SIZE=1,
        BIAS=False,
        EVEN_K=(K % BLOCK_SIZE_K == 0),
        CACHE_MODIFIER_A=".cg",
        CACHE_MODIFIER_B=".cg",
        epilogue_fn=hadamard_epilogue,
        QUANTIZED=False,
    )
    
    return C


def build_hadamard_torch(size):
    """Build Hadamard matrix in PyTorch for verification."""
    i = torch.arange(size, dtype=torch.int32)
    j = torch.arange(size, dtype=torch.int32)
    
    # Compute bitwise AND for all (i, j) pairs
    matching_bits = i[:, None] & j[None, :]
    
    # Count set bits (popcount)
    bit_sum = torch.zeros_like(matching_bits)
    temp = matching_bits.clone()
    for _ in range(6):  # 6 iterations for up to 64 bits
        bit_sum += temp & 1
        temp >>= 1
    
    # Map: even popcount -> +1, odd popcount -> -1
    H = 1 - 2 * (bit_sum & 1)
    
    # Normalize
    H = H.float() / (size ** 0.5)
    return H


def apply_hadamard_torch(matrix, block_size=32):
    """Apply Hadamard rotation in PyTorch for verification."""
    M, N = matrix.shape
    result = torch.zeros_like(matrix)
    
    # Build Hadamard matrix
    H = build_hadamard_torch(block_size).to(matrix.device, matrix.dtype)
    
    # Apply to each block
    for m_block in range(M // block_size):
        m_start = m_block * block_size
        m_end = m_start + block_size
        
        for n_block in range(N // block_size):
            n_start = n_block * block_size
            n_end = n_start + block_size
            
            # Extract block and apply Hadamard
            block = matrix[m_start:m_end, n_start:n_end]
            result[m_start:m_end, n_start:n_end] = block @ H
    
    return result


def main():
    print("\n" + "="*70)
    print("Hadamard Rotation Epilogue Example")
    print("="*70 + "\n")
    
    # Problem size (must be divisible by block size)
    M, N, K = 512, 512, 512
    block_size = 32
    
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Hadamard block size: {block_size}x{block_size}\n")
    
    # Allocate input matrices
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16).T
    
    # Run GEMM with Hadamard epilogue
    print("Running GEMM with Hadamard rotation epilogue...")
    C_triton = matmul_with_hadamard(A, B, block_size=block_size)
    
    # Compute reference: GEMM then Hadamard
    print("Computing PyTorch reference...")
    C_gemm = torch.matmul(A, B)
    C_torch = apply_hadamard_torch(C_gemm, block_size=block_size)
    
    # Compare results
    max_diff = torch.max(torch.abs(C_triton - C_torch)).item()
    mean_diff = torch.mean(torch.abs(C_triton - C_torch)).item()
    
    print(f"\nResults:")
    print("-" * 70)
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    print(f"Mean difference from PyTorch: {mean_diff:.6f}")
    print(f"Output shape: {C_triton.shape}")
    print(f"\nSample output (first 3x3):\n{C_triton[:3, :3]}")
    
    # Verify Hadamard properties
    print(f"\nHadamard Matrix Properties:")
    print("-" * 70)
    H = build_hadamard_torch(block_size).to("cuda", torch.float16)
    H_squared = H @ H.T
    identity_diff = torch.max(torch.abs(H_squared - torch.eye(block_size, device="cuda", dtype=torch.float16))).item()
    print(f"H @ H^T ≈ I (max diff from identity): {identity_diff:.6f}")
    print(f"Hadamard matrix is orthogonal: {identity_diff < 0.01}")
    
    print("\n" + "="*70)
    print("Key Points:")
    print("="*70)
    print("1. Hadamard rotation is applied to blocks of the accumulator")
    print("2. Block size must be power of 2 between 16 and 64")
    print("3. Accumulator dimensions must be divisible by block size")
    print("4. Hadamard matrices are orthogonal: H @ H^T = I")
    print("5. Useful for randomized algorithms and privacy-preserving ML")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

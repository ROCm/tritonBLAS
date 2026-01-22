"""
Example demonstrating Hadamard rotation epilogue in tritonBLAS.

This example shows how to apply a Hadamard transformation to the entire
output accumulator tile. This is useful for randomized numerical linear algebra
and privacy-preserving computations.
"""
import torch
import triton
import triton.language as tl
from tritonblas.kernels.persistent_gemm import persistent_matmul


# ============================================================================
# Define Hadamard Rotation Epilogue
# ============================================================================

@triton.jit
def build_hadamard(SIZE: tl.constexpr):
    """
    Construct Hadamard matrix using H_{i,j} = (-1)^{popcount(i & j)}.
    
    Args:
        SIZE: Matrix dimension (must be power of 2, 16 <= SIZE <= 64)
        
    Returns:
        SIZE x SIZE normalized Hadamard matrix
    """
    tl.static_assert(16 <= SIZE)

    # Create row and column indices
    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)

    # Compute bitwise AND for all (i, j) pairs
    matching_bits = i[:, None] & j[None, :]

    # Count set bits (popcount)
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
def is_power_of_two(n: tl.constexpr) -> tl.constexpr:
    """Check if n is a power of 2."""
    return (n & (n - 1)) == 0 and n > 0


@triton.jit
def hadamard_rotation_square(acc):
    """
    Apply Hadamard rotation to the entire square accumulator.
    
    This works for any square accumulator with power-of-2 dimensions
    between 16 and 64.
    
    Args:
        acc: Accumulator tensor [SIZE, SIZE] where SIZE is power of 2, 16 <= SIZE <= 64
    
    Returns:
        Accumulator with Hadamard transformation applied: acc @ H
    """
    SIZE:tl.constexpr = acc.shape[0]
    
    # Static assertions to enforce layout constraints
    tl.static_assert(acc.shape[0] == acc.shape[1], "Accumulator must be square")
    tl.static_assert((SIZE  & (SIZE  - 1)) == 0, "Accumulator size must be power of 2")
    tl.static_assert(SIZE >= 16, "Accumulator size must be >= 16")
    
    # Build Hadamard matrix and apply
    H = build_hadamard(SIZE)
    return tl.dot(acc, H.to(acc.dtype))


# ============================================================================
# Helper Function
# ============================================================================

def matmul_with_hadamard(A, B, tile_size=256):
    """
    Perform matrix multiplication with Hadamard rotation epilogue.
    
    Args:
        A: Input matrix A [M, K]
        B: Input matrix B [K, N] (transposed)
        tile_size: Square tile size for GEMM (must be 32 or 64)
    
    Returns:
        Output matrix C [M, N] with Hadamard transformation applied to each tile
    """
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device="cuda", dtype=A.dtype)
    
    # Get device properties
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    # Use square tiles
    BLOCK_SIZE_M = tile_size
    BLOCK_SIZE_N = tile_size
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Define grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
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
        epilogue_fn=hadamard_rotation_square,
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
    for _ in range(6):
        bit_sum += temp & 1
        temp >>= 1
    
    # Map: even popcount -> +1, odd popcount -> -1
    H = 1 - 2 * (bit_sum & 1)
    
    # Normalize
    H = H.float() / (size ** 0.5)
    return H


def apply_hadamard_torch(matrix, tile_size):
    """Apply Hadamard rotation to each tile in PyTorch using bmm."""
    M, N = matrix.shape
    num_tiles_m = M // tile_size
    num_tiles_n = N // tile_size
    
    # Build Hadamard matrix
    H = build_hadamard_torch(tile_size).to(matrix.device, matrix.dtype)
    
    # Reshape matrix into tiles: (num_tiles_m, num_tiles_n, tile_size, tile_size)
    matrix_tiled = matrix.reshape(num_tiles_m, tile_size, num_tiles_n, tile_size)
    matrix_tiled = matrix_tiled.permute(0, 2, 1, 3)  # (num_tiles_m, num_tiles_n, tile_size, tile_size)
    matrix_tiled = matrix_tiled.reshape(-1, tile_size, tile_size)  # (num_tiles_m * num_tiles_n, tile_size, tile_size)
    
    # Apply Hadamard to all tiles at once using bmm
    result_tiled = torch.bmm(matrix_tiled, H.unsqueeze(0).expand(matrix_tiled.shape[0], -1, -1))
    
    # Reshape back to original shape
    result_tiled = result_tiled.reshape(num_tiles_m, num_tiles_n, tile_size, tile_size)
    result_tiled = result_tiled.permute(0, 2, 1, 3)  # (num_tiles_m, tile_size, num_tiles_n, tile_size)
    result = result_tiled.reshape(M, N)
    
    return result


def main():
    print("\n" + "="*70)
    print("Hadamard Rotation Epilogue Example")
    print("="*70 + "\n")
    
    # Problem size (must be divisible by tile size)
    M, N, K = 8192, 8192, 8192
    tile_size = 128
    
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Square tile size: {tile_size}x{tile_size}")
    print(f"Hadamard applied to entire accumulator tile\n")
    
    # Allocate input matrices
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16).T
    
    # Run GEMM with Hadamard epilogue
    print("Running GEMM with Hadamard rotation epilogue...")
    C_triton = matmul_with_hadamard(A, B, tile_size=tile_size)
    
    # Compute reference: GEMM then Hadamard
    print("Computing PyTorch reference...")
    C_gemm = torch.matmul(A, B)
    C_torch = apply_hadamard_torch(C_gemm, tile_size=tile_size)
    
    # Compare results
    max_diff = torch.max(torch.abs(C_triton - C_torch)).item()
    mean_diff = torch.mean(torch.abs(C_triton - C_torch)).item()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    print(f"\nResults:")
    print("-" * 70)
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    print(f"Mean difference from PyTorch: {mean_diff:.6f}")
    print(f"Output shape: {C_triton.shape}")
    print(f"\nSample output (first 3x3):\n{C_triton[:3, :3]}")
    
    # Verify Hadamard properties
    print(f"\nHadamard Matrix Properties:")
    print("-" * 70)
    H = build_hadamard_torch(tile_size).to("cuda", torch.float16)
    H_squared = H @ H.T
    identity_diff = torch.max(torch.abs(H_squared - torch.eye(tile_size, device="cuda", dtype=torch.float16))).item()
    print(f"H @ H^T ≈ I (max diff from identity): {identity_diff:.6f}")
    print(f"Hadamard matrix is orthogonal: {identity_diff < 0.01}")
    
    print("\n" + "="*70)
    print("Key Points:")
    print("="*70)
    print("1. Hadamard rotation applied to entire square accumulator tile")
    print("2. Tile must be square and power-of-2 (16, 32, or 64)")
    print("3. Generic implementation works for any valid square tile size")
    print("4. Hadamard matrices are orthogonal: H @ H^T = I")
    print("5. Useful for randomized algorithms and privacy-preserving ML")
    print("="*70 + "\n")
    
    # ========================================================================
    # Performance Benchmark
    # ========================================================================
    print("="*70)
    print("Performance Benchmark")
    print("="*70 + "\n")
    
    from triton.testing import do_bench
    
    # Benchmark GEMM without epilogue
    print("Benchmarking GEMM without epilogue...")
    
    def gemm_no_epilogue():
        C_no_epi = torch.zeros((M, N), device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(M, tile_size) * triton.cdiv(N, tile_size),)
        persistent_matmul[grid](
            A, B, C_no_epi,
            None, None, A,
            M, N, K,
            A.stride(0), B.stride(1),
            C_no_epi.stride(0), C_no_epi.stride(1),
            0, A.stride(1), B.stride(0),
            BLOCK_SIZE_M=tile_size, BLOCK_SIZE_N=tile_size, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8, NUM_SMS=num_sms, NUM_XCDS=8, CHUNK_SIZE=8,
            BIAS=False, EVEN_K=(K % 32 == 0),
            CACHE_MODIFIER_A=".cg", CACHE_MODIFIER_B=".cg",
            epilogue_fn=None,
            QUANTIZED=False,
        )
        return C_no_epi
    
    time_no_epilogue = do_bench(gemm_no_epilogue)
    
    # Benchmark GEMM with Hadamard epilogue
    print("Benchmarking GEMM with Hadamard epilogue...")
    
    def gemm_with_epilogue():
        return matmul_with_hadamard(A, B, tile_size=tile_size)
    
    time_with_epilogue = do_bench(gemm_with_epilogue)
    
    # Benchmark separate GEMM + Hadamard
    print("Benchmarking GEMM + separate Hadamard kernel...")
    
    def gemm_then_hadamard():
        C_temp = torch.matmul(A, B)
        return apply_hadamard_torch(C_temp, tile_size=tile_size)
    
    time_separate = do_bench(gemm_then_hadamard)
    
    print(f"\nPerformance Results:")
    print("-" * 70)
    print(f"GEMM without epilogue:        {time_no_epilogue:.3f} ms")
    print(f"GEMM with Hadamard epilogue:  {time_with_epilogue:.3f} ms")
    print(f"GEMM + separate Hadamard:     {time_separate:.3f} ms")
    print(f"\nOverhead of epilogue:         {time_with_epilogue - time_no_epilogue:.3f} ms ({((time_with_epilogue/time_no_epilogue - 1) * 100):.1f}%)")
    print(f"Speedup vs separate kernels:  {time_separate / time_with_epilogue:.2f}x")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

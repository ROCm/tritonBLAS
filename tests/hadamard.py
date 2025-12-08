"""
Standalone Triton kernel for blocked Hadamard transformation.
This kernel applies a 32x32 Hadamard transformation to each 32x32 subblock of an MxK matrix A.
The output is also MxK, with each 32x32 block rotated by the Hadamard matrix.
For each 32x32 block A[i:i+32, j:j+32], we compute:
    Output[i:i+32, j:j+32] = A[i:i+32, j:j+32] @ H
where H is the 32x32 Hadamard matrix.
"""

import triton
import triton.language as tl
import torch
import time
import numpy as np


@triton.jit
def hadamard_blocked_kernel(
    A_ptr,           # Pointer to input matrix A [M, K]
    H_ptr,           # Pointer to Hadamard matrix [32, 32]
    Out_ptr,         # Pointer to output matrix [M, K]
    M,               # Number of rows in A
    K,               # Number of columns in A
    stride_am,       # Stride of A in M dimension
    stride_ak,       # Stride of A in K dimension
    stride_hrow,     # Stride of H in row dimension
    stride_hcol,     # Stride of H in column dimension
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
    
    # Load Hadamard matrix [BLOCK_SIZE, BLOCK_SIZE]
    h_row_offs = tl.arange(0, BLOCK_SIZE)
    h_col_offs = tl.arange(0, BLOCK_SIZE)
    h_ptrs = H_ptr + h_row_offs[:, None] * stride_hrow + h_col_offs[None, :] * stride_hcol
    h_block = tl.load(h_ptrs)
    
    # Perform matrix multiplication: A_block @ H_block
    # This is a single 32x32 @ 32x32 operation
    result = tl.dot(a_block, h_block)
    
    # Store result to output
    out_ptrs = Out_ptr + m_offs[:, None] * stride_om + k_offs[None, :] * stride_ok
    tl.store(out_ptrs, result, mask=m_mask[:, None] & k_mask[None, :])


def hadamard_blocked_transform(A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transformation to each 32x32 block of matrix A.
    
    Args:
        A: Input matrix of shape [M, K]
        H: Hadamard matrix of shape [32, 32]
    
    Returns:
        Output matrix of shape [M, K] with each 32x32 block transformed
    """
    assert A.is_cuda and H.is_cuda, "Tensors must be on CUDA"
    assert H.shape == (32, 32), f"Hadamard must be 32x32, got {H.shape}"
    
    M, K = A.shape
    
    # Allocate output with same shape as A
    Out = torch.zeros_like(A)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions - one program per 32x32 block
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE))
    
    # Launch kernel
    hadamard_blocked_kernel[grid](
        A, H, Out,
        M, K,
        A.stride(0), A.stride(1),
        H.stride(0), H.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return Out

def generate_hadamard_matrix(n: int) -> torch.Tensor:
    """
    Generate a Hadamard matrix of size n x n using Sylvester's construction.
    n must be a power of 2.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    
    if n == 1:
        return torch.tensor([[1.0]])
    
    # Recursive construction
    H_half = generate_hadamard_matrix(n // 2)
    H = torch.zeros((n, n))
    half = n // 2
    
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half
    
    # Note: cannot normalize when applying recursion!
    return H


def hadamard_blocked_transform_torch(A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation using batched matrix multiplication.
    
    Reshapes A into blocks and performs batched GEMM with the Hadamard matrix.
    """
    M, K = A.shape
    
    # Pad to multiples of 32 if necessary
    M_pad = ((M + 31) // 32) * 32
    K_pad = ((K + 31) // 32) * 32
    
    if M_pad != M or K_pad != K:
        A_padded = torch.zeros((M_pad, K_pad), device=A.device, dtype=A.dtype)
        A_padded[:M, :K] = A
    else:
        A_padded = A
    
    # Reshape into blocks: [M_pad, K_pad] -> [M_blocks, 32, K_blocks, 32]
    M_blocks = M_pad // 32
    K_blocks = K_pad // 32
    
    # Reshape: [M_pad, K_pad] -> [M_blocks, 32, K_blocks, 32] -> [M_blocks, K_blocks, 32, 32]
    A_blocks = A_padded.reshape(M_blocks, 32, K_blocks, 32).permute(0, 2, 1, 3)
    
    # Flatten batch dimensions: [M_blocks * K_blocks, 32, 32]
    A_blocks_flat = A_blocks.reshape(-1, 32, 32)
    
    # Batched matrix multiplication: [batch, 32, 32] @ [32, 32] -> [batch, 32, 32]
    Out_blocks_flat = torch.bmm(A_blocks_flat, H.unsqueeze(0).expand(A_blocks_flat.shape[0], -1, -1))
    
    # Reshape back: [batch, 32, 32] -> [M_blocks, K_blocks, 32, 32]
    Out_blocks = Out_blocks_flat.reshape(M_blocks, K_blocks, 32, 32)
    
    # Permute and reshape: [M_blocks, K_blocks, 32, 32] -> [M_blocks, 32, K_blocks, 32] -> [M_pad, K_pad]
    Out_padded = Out_blocks.permute(0, 2, 1, 3).reshape(M_pad, K_pad)
    
    # Remove padding if necessary
    if M_pad != M or K_pad != K:
        return Out_padded[:M, :K]
    else:
        return Out_padded


def test_correctness():
    """Test correctness against PyTorch reference implementation."""
    print("=" * 80)
    print("Testing Correctness")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test dimensions
    M, K = 128, 128
    
    # Generate test data
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    H = generate_hadamard_matrix(32).cuda()
    
    # Triton implementation
    Out_triton = hadamard_blocked_transform(A, H)
    
    # PyTorch reference using batched GEMM
    Out_torch = hadamard_blocked_transform_torch(A, H)
    
    # Compare results
    max_diff = torch.max(torch.abs(Out_triton - Out_torch)).item()
    mean_diff = torch.mean(torch.abs(Out_triton - Out_torch)).item()
    
    print(f"Input shape: A={A.shape}, H={H.shape}")
    print(f"Output shape: {Out_triton.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-4:
        print("✓ Correctness test PASSED")
    else:
        print("✗ Correctness test FAILED")
        print(f"\nSample values:")
        print(f"Triton output [0:3, 0:3]:\n{Out_triton[0:3, 0:3]}")
        print(f"PyTorch output [0:3, 0:3]:\n{Out_torch[0:3, 0:3]}")
    
    return max_diff < 1e-4


def benchmark_performance():
    """Benchmark performance against PyTorch implementation."""
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    
    # Test configurations
    configs = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]
    
    num_warmup = 10
    num_iterations = 100
    
    print(f"\nRunning {num_warmup} warmup iterations and {num_iterations} timed iterations")
    print(f"{'M':>6} {'K':>6} {'Triton (ms)':>12} {'PyTorch (ms)':>13} {'Speedup':>8} {'TFLOPS':>8}")
    print("-" * 80)
    
    for M, K in configs:
        # Generate test data
        A = torch.randn((M, K), device='cuda', dtype=torch.float32)
        H = generate_hadamard_matrix(32).cuda()
        
        # Warmup - Triton
        for _ in range(num_warmup):
            _ = hadamard_blocked_transform(A, H)
        torch.cuda.synchronize()
        
        # Benchmark - Triton
        start = time.time()
        for _ in range(num_iterations):
            Out_triton = hadamard_blocked_transform(A, H)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations * 1000
        
        # Warmup - PyTorch (batched GEMM)
        for _ in range(num_warmup):
            _ = hadamard_blocked_transform_torch(A, H)
        torch.cuda.synchronize()
        
        # Benchmark - PyTorch (batched GEMM)
        start = time.time()
        for _ in range(num_iterations):
            Out_torch = hadamard_blocked_transform_torch(A, H)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iterations * 1000
        
        # Calculate metrics
        speedup = torch_time / triton_time
        
        # FLOPS calculation: (M/32) * (K/32) blocks, each doing 32*32*32*2 FLOPs
        num_blocks = ((M + 31) // 32) * ((K + 31) // 32)
        flops = num_blocks * 32 * 32 * 32 * 2
        tflops = (flops / (triton_time * 1e-3)) / 1e12
        
        print(f"{M:6d} {K:6d} {triton_time:12.4f} {torch_time:13.4f} {speedup:8.2f}x {tflops:8.2f}")
    
    print("=" * 80)


def main():
    """Main function to run tests and benchmarks."""
    print("\n" + "=" * 80)
    print("Hadamard Blocked Transformation - Triton Implementation")
    print("=" * 80)
    print("\nThis kernel applies a 32x32 Hadamard transformation to each")
    print("32x32 subblock of the input matrix A, producing an output of")
    print("the same size as A.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nERROR: CUDA is not available. This kernel requires a CUDA GPU.")
        return
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Run correctness test
    passed = test_correctness()
    
    if passed:
        # Run performance benchmark
        benchmark_performance()
    else:
        print("\nSkipping performance benchmark due to correctness test failure.")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
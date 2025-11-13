"""
Test for persistent_matmul with bias support comparing against torch.nn.functional.linear.

This test demonstrates how to use the bias feature in persistent_matmul
and compares the results directly against PyTorch's fused linear operation.

NOTE: The persistent_matmul kernel applies bias per row (M dimension):
      C[i, j] = (A @ B)[i, j] + bias[i]
      
      torch.nn.functional.linear applies bias per output feature (N dimension):
      output = input @ weight.T + bias  where bias has shape (out_features,)

To match torch.linear with persistent_matmul, we transpose the computation.

To run this test:
    1. Start the Docker container:
       docker-compose up -d
       docker attach tritonBLAS-dev
    
    2. Inside the container:
       export PYTHONPATH=/workspace/include:$PYTHONPATH
       python3 -m pytest tests/test_matmul_bias.py -v
       
    Or run directly:
       python3 tests/test_matmul_bias.py
"""

import pytest
import torch
import triton
from tritonblas.internal.persistent_matmul import persistent_matmul
from tritonblas.origami import MatmulHeuristicResult


@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    [
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
        (1024, 2048, 4096),
        (2048, 4096, 8192),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_persistent_matmul_bias_vs_torch_linear(batch_size, in_features, out_features, dtype):
    """
    Test persistent_matmul with bias against torch.nn.functional.linear (fused operation).
    
    torch.nn.functional.linear computes:
        output = input @ weight.T + bias
        where:
        - input: (batch_size, in_features)
        - weight: (out_features, in_features)
        - bias: (out_features,)
        - output: (batch_size, out_features)
    
    The persistent_matmul kernel computes:
        C = A @ B + bias[:, None]
        where bias is (M,) and is broadcast across columns
    
    To match torch.linear, we transpose the computation:
        weight @ input.T + bias[:, None] = (out_features, batch_size)
        Then transpose the result to get (batch_size, out_features)
    """
    # Create input tensors for torch.linear
    input_tensor = torch.randn((batch_size, in_features), device="cuda", dtype=dtype)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=dtype)
    bias_linear = torch.randn((out_features,), device="cuda", dtype=dtype)
    
    # Compute torch.nn.functional.linear (FUSED operation - this is our reference)
    output_torch = torch.nn.functional.linear(input_tensor, weight, bias_linear)
    
    A = weight  # (out_features, in_features)
    B = input_tensor.T  # (in_features, batch_size)
    bias = bias_linear  # (out_features,)
    
    m, k = A.shape  # m=out_features, k=in_features
    _, n = B.shape  # n=batch_size
    
    # Allocate output tensor
    C_triton = torch.zeros((m, n), device="cuda", dtype=dtype)
    
    # Get heuristic configuration
    selector = MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    # Calculate grid parameters
    total_blocks_M = triton.cdiv(m, BLK_M)
    total_blocks_N = triton.cdiv(n, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = k % BLK_K == 0
    
    # Set chunk size
    num_xcds = 8
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, total_programs // num_xcds)
    
    # Kernel configuration
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None
    
    # Run persistent_matmul with bias enabled
    persistent_matmul[(total_tiles,)](
        A,
        B,
        C_triton,
        None,  # A_scale_ptr (not quantized)
        None,  # B_scale_ptr (not quantized)
        bias,  # bias_ptr - ENABLED
        m,
        n,
        k,
        A.stride(0),
        B.stride(1),
        C_triton.stride(0),
        C_triton.stride(1),
        bias.stride(0),  # stride_bias
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_programs,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=True,  # BIAS ENABLED
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        QUANTIZED=False,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )
    
    # Transpose to match torch.linear output shape
    output_triton = C_triton.T  # (batch_size, out_features)
    
    # Compare results with appropriate tolerances
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-4
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2
    
    # Calculate differences for debugging
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    mean_diff = torch.mean(torch.abs(output_torch - output_triton)).item()
    rel_diff = max_diff / (torch.max(torch.abs(output_torch)).item() + 1e-8)
    
    print(f"batch={batch_size}, in_feat={in_features}, out_feat={out_features}, dtype={dtype}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Rel diff: {rel_diff:.6e}")
    
    try:
        torch.testing.assert_close(output_triton, output_torch, atol=atol, rtol=rtol)
        print(f"  ✓ Test PASSED - TritonBLAS matches torch.nn.functional.linear")
    except AssertionError as e:
        print(f"  ✗ Test FAILED - TritonBLAS does NOT match torch.nn.functional.linear")
        print(f"  Torch linear output sample: {output_torch[0, :5]}")
        print(f"  Triton output sample: {output_triton[0, :5]}")
        print(f"  Difference sample: {(output_torch - output_triton)[0, :5]}")
        raise


@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    [
        (256, 512, 1024),
        (1024, 2048, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
)
def test_persistent_matmul_bias_detailed_comparison(batch_size, in_features, out_features, dtype):
    """
    Detailed comparison showing where TritonBLAS succeeds or fails compared to torch.linear.
    This test provides more diagnostic information.
    """
    # Create input tensors
    input_tensor = torch.randn((batch_size, in_features), device="cuda", dtype=dtype)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=dtype)
    bias_linear = torch.randn((out_features,), device="cuda", dtype=dtype)
    
    # Compute torch.nn.functional.linear (reference)
    output_torch = torch.nn.functional.linear(input_tensor, weight, bias_linear)
    
    # Compute using persistent_matmul
    A = weight
    B = input_tensor.T
    bias = bias_linear
    
    m, k = A.shape
    _, n = B.shape
    
    C_triton = torch.zeros((m, n), device="cuda", dtype=dtype)
    
    selector = MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    total_blocks_M = triton.cdiv(m, BLK_M)
    total_blocks_N = triton.cdiv(n, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = k % BLK_K == 0
    
    num_xcds = 8
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, total_programs // num_xcds)
    
    persistent_matmul[(total_tiles,)](
        A, B, C_triton,
        None, None, bias,
        m, n, k,
        A.stride(0), B.stride(1),
        C_triton.stride(0), C_triton.stride(1),
        bias.stride(0),
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_programs,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=True,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        num_stages=2,
        num_warps=8,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    
    output_triton = C_triton.T
    
    # Detailed analysis
    diff = output_torch - output_triton
    abs_diff = torch.abs(diff)
    
    print(f"\n{'='*80}")
    print(f"Detailed Comparison: batch={batch_size}, in={in_features}, out={out_features}, dtype={dtype}")
    print(f"{'='*80}")
    print(f"Torch linear output stats:")
    print(f"  Min: {output_torch.min():.6f}, Max: {output_torch.max():.6f}")
    print(f"  Mean: {output_torch.mean():.6f}, Std: {output_torch.std():.6f}")
    print(f"\nTritonBLAS output stats:")
    print(f"  Min: {output_triton.min():.6f}, Max: {output_triton.max():.6f}")
    print(f"  Mean: {output_triton.mean():.6f}, Std: {output_triton.std():.6f}")
    print(f"\nDifference stats:")
    print(f"  Max abs diff: {abs_diff.max():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean():.6e}")
    print(f"  Median abs diff: {abs_diff.median():.6e}")
    print(f"  95th percentile abs diff: {torch.quantile(abs_diff.float(), 0.95):.6e}")
    print(f"  99th percentile abs diff: {torch.quantile(abs_diff.float(), 0.99):.6e}")
    
    # Check if differences are within tolerance
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2
    
    within_tolerance = torch.allclose(output_triton, output_torch, atol=atol, rtol=rtol)
    
    if within_tolerance:
        print(f"\n✓ SUCCESS: TritonBLAS matches torch.linear within tolerance")
        print(f"  (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILURE: TritonBLAS does NOT match torch.linear")
        print(f"  (atol={atol}, rtol={rtol})")
        print(f"\nSample outputs (first 5 elements of first batch):")
        print(f"  Torch:  {output_torch[0, :5]}")
        print(f"  Triton: {output_triton[0, :5]}")
        print(f"  Diff:   {diff[0, :5]}")
    
    print(f"{'='*80}\n")
    
    torch.testing.assert_close(output_triton, output_torch, atol=atol, rtol=rtol)


if __name__ == "__main__":
    print("=" * 80)
    print("Testing persistent_matmul with bias vs torch.nn.functional.linear")
    print("=" * 80)
    
    print("\nTest 1: Basic comparison against torch.linear (fused operation)")
    print("-" * 80)
    test_persistent_matmul_bias_vs_torch_linear(256, 512, 1024, torch.float16)
    
    print("\nTest 2: Detailed diagnostic comparison")
    print("-" * 80)
    test_persistent_matmul_bias_detailed_comparison(256, 512, 1024, torch.float16)
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("TritonBLAS persistent_matmul with bias matches torch.nn.functional.linear")
    print("=" * 80)

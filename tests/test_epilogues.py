"""
Tests for epilogue functions in tritonBLAS.
"""
import pytest
import torch
import triton
from tritonblas.kernels.persistent_gemm import persistent_matmul
from tritonblas.kernels.stages.algorithms.epilogue import (
    relu, gelu, silu, tanh, sigmoid, leaky_relu, identity
)


def run_matmul_with_epilogue(M, N, K, epilogue_fn=None, bias=None, dtype=torch.float16):
    """
    Helper function to run matrix multiplication with epilogue.
    
    Args:
        M, N, K: Matrix dimensions
        epilogue_fn: Epilogue function to apply
        bias: Optional bias vector
        dtype: Data type for tensors
    
    Returns:
        Output tensor from Triton kernel
    """
    # Allocate tensors
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(N, K, device="cuda", dtype=dtype).T
    C = torch.zeros((M, N), device="cuda", dtype=dtype)
    
    # Get device properties
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    # Setup bias
    has_bias = bias is not None
    if has_bias:
        bias_ptr = bias
        stride_bias = bias.stride(0)
    else:
        bias_ptr = A  # Dummy pointer
        stride_bias = 0
    
    # Fixed block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Define grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    persistent_matmul[grid](
        A, B, C,
        None, None,  # No quantization scales
        bias_ptr,
        M, N, K,
        A.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        stride_bias,
        A.stride(1), B.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=num_sms,
        NUM_XCDS=1,
        CHUNK_SIZE=1,
        BIAS=has_bias,
        EVEN_K=(K % BLOCK_SIZE_K == 0),
        CACHE_MODIFIER_A=".cg",
        CACHE_MODIFIER_B=".cg",
        epilogue_fn=epilogue_fn,
        QUANTIZED=False,
    )
    
    return C, A, B


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
    (128, 256, 512),
])
def test_identity_epilogue(M, N, K):
    """Test identity epilogue (no activation)."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=identity)
    C_torch = torch.matmul(A, B)
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"Identity epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_relu_epilogue(M, N, K):
    """Test ReLU epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=relu)
    C_torch = torch.relu(torch.matmul(A, B))
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"ReLU epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_gelu_epilogue(M, N, K):
    """Test GELU epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=gelu)
    C_torch = torch.nn.functional.gelu(torch.matmul(A, B), approximate='tanh')
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"GELU epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_silu_epilogue(M, N, K):
    """Test SiLU epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=silu)
    C_torch = torch.nn.functional.silu(torch.matmul(A, B))
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"SiLU epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_tanh_epilogue(M, N, K):
    """Test Tanh epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=tanh)
    C_torch = torch.tanh(torch.matmul(A, B))
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"Tanh epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_sigmoid_epilogue(M, N, K):
    """Test Sigmoid epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=sigmoid)
    C_torch = torch.sigmoid(torch.matmul(A, B))
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"Sigmoid epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
def test_leaky_relu_epilogue(M, N, K):
    """Test Leaky ReLU epilogue."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=leaky_relu)
    C_torch = torch.nn.functional.leaky_relu(torch.matmul(A, B), negative_slope=0.01)
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"Leaky ReLU epilogue failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
])
def test_epilogue_with_bias(M, N, K):
    """Test epilogue with bias addition."""
    bias = torch.randn(M, device="cuda", dtype=torch.float16)
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=relu, bias=bias)
    
    C_torch = torch.matmul(A, B) + bias.unsqueeze(1)
    C_torch = torch.relu(C_torch)
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"ReLU epilogue with bias failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
])
def test_no_epilogue(M, N, K):
    """Test that None epilogue works correctly."""
    C_triton, A, B = run_matmul_with_epilogue(M, N, K, epilogue_fn=None)
    C_torch = torch.matmul(A, B)
    
    assert torch.allclose(C_triton, C_torch, rtol=1e-2, atol=1e-2), \
        f"No epilogue (None) failed: max_diff={torch.max(torch.abs(C_triton - C_torch))}"


if __name__ == "__main__":
    # Run tests manually
    print("Running epilogue tests...")
    
    test_functions = [
        test_identity_epilogue,
        test_relu_epilogue,
        test_gelu_epilogue,
        test_silu_epilogue,
        test_tanh_epilogue,
        test_sigmoid_epilogue,
        test_leaky_relu_epilogue,
        test_epilogue_with_bias,
        test_no_epilogue,
    ]
    
    for test_func in test_functions:
        try:
            test_func(256, 256, 256)
            print(f"✓ {test_func.__name__} PASSED")
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
    
    print("\nAll tests completed!")

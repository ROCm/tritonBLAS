import torch
import tritonblas
import pytest


@pytest.mark.parametrize("M,N,K", [
    (128, 256, 256),   # Standard GEMV case
    (1, 256, 256),     # M=1 (row vector result)
    (256, 1, 256),     # N=1 (column vector input)
    (256, 256, 1),     # K=1 (minimal reduction)
])
def test_gemv_basic(M, N, K):
    """Test basic GEMV operation: y = A @ x"""
    device = "cuda"
    dtype = torch.float16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    # Compute with tritonblas
    y_triton = tritonblas.mv(A, x)
    
    # Compute reference with PyTorch
    y_torch = A @ x
    
    # Check results
    assert y_triton.shape == (M,), f"Expected shape ({M},), got {y_triton.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Results don't match. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ Basic GEMV test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    (128, 256, 256),   # Standard transposed case
    (1, 256, 256),     # M=1
    (256, 1, 256),     # N=1
])
def test_gemv_transposed(M, N, K):
    """Test transposed GEMV operation: y = A.T @ x"""
    device = "cuda"
    dtype = torch.float16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(M, device=device, dtype=dtype)
    
    # Compute with tritonblas
    y_triton = tritonblas.mv(A, x, transpose=True)
    
    # Compute reference with PyTorch
    y_torch = A.T @ x
    
    # Check results
    assert y_triton.shape == (N,), f"Expected shape ({N},), got {y_triton.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Results don't match. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ Transposed GEMV test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    (64, 128, 128),    # Standard case
    (1, 128, 128),     # M=1
    (64, 1, 64),       # N=1
])
def test_gemv_with_preallocated_output(M, N, K):
    """Test GEMV with pre-allocated output buffer"""
    device = "cuda"
    dtype = torch.float16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    c = torch.zeros(M, device=device, dtype=dtype)
    
    # Compute with tritonblas
    c_triton = tritonblas.mv(A, x, c=c)
    
    # Compute reference with PyTorch
    y_torch = A @ x
    
    # Check that the same buffer was used
    assert c_triton is c, "Output buffer was not reused"
    
    # Check results
    assert torch.allclose(c_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Results don't match. Max diff: {(c_triton - y_torch).abs().max()}"
    
    print(f"✓ GEMV with pre-allocated output test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    # All configs must have at least one dimension = 1 for true GEMV
    (1, 64, 64),       # M=1, small
    (1, 128, 128),     # M=1, medium
    (1, 512, 512),     # M=1, large
    (1, 2048, 2048),   # M=1, very large
    (64, 1, 64),       # N=1, small
    (128, 1, 128),     # N=1, medium
    (512, 1, 512),     # N=1, large
    (2048, 1, 2048),   # N=1, very large
    (100, 200, 1),     # K=1, non-power-of-2
    (333, 777, 1),     # K=1, odd sizes
])
def test_gemv_various_sizes(M, N, K):
    """Test GEMV with various matrix sizes (all with at least one dim=1)"""
    device = "cuda"
    dtype = torch.float16
    
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    y_triton = tritonblas.mv(A, x)
    y_torch = A @ x
    
    assert torch.allclose(y_triton, y_torch, rtol=5e-2, atol=5e-2), \
        f"Failed for size M={M}, N={N}, K={K}. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ GEMV various sizes test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    (128, 256, 256),   # Standard case
    (1, 256, 256),     # M=1
])
def test_gemv_bfloat16(M, N, K):
    """Test GEMV with bfloat16 dtype"""
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    # Compute with tritonblas
    y_triton = tritonblas.mv(A, x)
    
    # Compute reference with PyTorch
    y_torch = A @ x
    
    # Check results (bfloat16 has lower precision)
    assert torch.allclose(y_triton, y_torch, rtol=1e-1, atol=1e-1), \
        f"Results don't match. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ GEMV bfloat16 test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    (128, 256, 256),   # Matrix @ Vector
    (1, 256, 256),     # M=1
])
def test_matmul_with_vector(M, N, K):
    """Test that matmul automatically uses GEMV for vector inputs"""
    device = "cuda"
    dtype = torch.float16
    
    # Test case 1: Matrix @ Vector
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    y_triton = tritonblas.matmul(A, x)
    y_torch = A @ x
    
    assert y_triton.shape == (M,), f"Expected shape ({M},), got {y_triton.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Matrix @ Vector failed. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ matmul(Matrix, Vector) test passed (M={M}, N={N}, K={K})")
    
    # Test case 2: Vector @ Matrix (should compute b.T @ a)
    a = torch.randn(M, device=device, dtype=dtype)
    B = torch.randn(M, N, device=device, dtype=dtype)
    
    y_triton = tritonblas.matmul(a, B)
    y_torch = a @ B
    
    assert y_triton.shape == (N,), f"Expected shape ({N},), got {y_triton.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Vector @ Matrix failed. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ matmul(Vector, Matrix) test passed (M={M}, N={N}, K={K})")


def test_matmul_vector_error_handling():
    """Test that matmul raises error for vector @ vector"""
    device = "cuda"
    dtype = torch.float16
    
    a = torch.randn(128, device=device, dtype=dtype)
    b = torch.randn(128, device=device, dtype=dtype)
    
    with pytest.raises(ValueError, match="Both inputs are vectors"):
        tritonblas.matmul(a, b)
    
    print(f"✓ matmul vector error handling test passed")


def test_gemv_dimension_mismatch():
    """Test that GEMV raises error for dimension mismatch"""
    device = "cuda"
    dtype = torch.float16
    
    A = torch.randn(128, 256, device=device, dtype=dtype)
    x = torch.randn(128, device=device, dtype=dtype)  # Wrong size
    
    with pytest.raises(AssertionError):
        tritonblas.mv(A, x)
    
    print(f"✓ GEMV dimension mismatch test passed")


@pytest.mark.parametrize("M,N,K", [
    (2048, 4096, 4096),   # Large
    (1, 4096, 4096),      # M=1 large
])
def test_gemv_large_matrix(M, N, K):
    """Test GEMV with larger matrices"""
    device = "cuda"
    dtype = torch.float16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    # Compute with tritonblas
    y_triton = tritonblas.mv(A, x)
    
    # Compute reference with PyTorch
    y_torch = A @ x
    
    # Check results - relaxed tolerance for large matrices with atomic operations
    assert torch.allclose(y_triton, y_torch, rtol=5e-2, atol=5e-2), \
        f"Results don't match. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ Large matrix GEMV test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,N,K", [
    (128, 256, 512),   # Standard case
    (1, 256, 512),     # M=1 (row vector result)
    (128, 1, 512),     # N=1 (column vector input)
    (128, 256, 1),     # K=1 (minimal reduction)
    (1, 1, 512),       # M=1, N=1
    (1, 256, 1),       # M=1, K=1
    (128, 1, 1),       # N=1, K=1
])
def test_gemv_edge_cases(M, N, K):
    """Test GEMV with various edge cases including dimensions of 1"""
    device = "cuda"
    dtype = torch.float16
    
    # Create test data
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N, device=device, dtype=dtype)
    
    # Compute with tritonblas
    y_triton = tritonblas.mv(A, x)
    
    # Compute reference with PyTorch
    y_torch = A @ x
    
    # Check results
    assert y_triton.shape == (M,), f"Expected shape ({M},), got {y_triton.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Results don't match for M={M}, N={N}, K={K}. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ GEMV edge case test passed (M={M}, N={N}, K={K})")


@pytest.mark.parametrize("M,K", [
    (128, 512),   # Standard case
    (1, 512),     # M=1 (single row)
    (256, 1),     # K=1 (minimal reduction)
    (1024, 2048), # Large
])
def test_matmul_with_column_vectors(M, K):
    """Test that matmul handles column vectors (K, 1) correctly"""
    device = "cuda"
    dtype = torch.float16
    
    # Test A @ b where b is a column vector (K, 1)
    A = torch.randn(M, K, device=device, dtype=dtype)
    b_col = torch.randn(K, 1, device=device, dtype=dtype)
    
    y_triton = tritonblas.matmul(A, b_col)
    y_torch = A @ b_col
    
    # Result should match PyTorch's output shape (M, 1)
    assert y_triton.shape == y_torch.shape, f"Shape mismatch: {y_triton.shape} vs {y_torch.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=5e-2, atol=5e-2), \
        f"Results don't match for M={M}, K={K}. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ matmul with column vector test passed (M={M}, K={K})")


@pytest.mark.parametrize("K,N", [
    (128, 256),   # Standard case
    (512, 1),     # N=1 (single column)
    (1, 512),     # K=1 (minimal reduction)
    (2048, 1024), # Large
])
def test_matmul_with_row_vectors(K, N):
    """Test that matmul handles row vectors (1, K) correctly"""
    device = "cuda"
    dtype = torch.float16
    
    # Test a @ B where a is a row vector (1, K)
    a_row = torch.randn(1, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    
    y_triton = tritonblas.matmul(a_row, B)
    y_torch = a_row @ B
    
    # Result should match PyTorch's output shape (1, N)
    assert y_triton.shape == y_torch.shape, f"Shape mismatch: {y_triton.shape} vs {y_torch.shape}"
    assert torch.allclose(y_triton, y_torch, rtol=5e-2, atol=5e-2), \
        f"Results don't match for K={K}, N={N}. Max diff: {(y_triton - y_torch).abs().max()}"
    
    print(f"✓ matmul with row vector test passed (K={K}, N={N})")


def benchmark_gemv():
    """Benchmark tritonblas GEMV vs PyTorch"""
    import time
    
    device = "cuda"
    dtype = torch.float16
    num_warmup = 10
    num_iters = 100
    
    test_configs = [
        # M, N, K - At least one dimension must be 1 for GEMV
        (1, 256, 256),        # M=1, small
        (1, 512, 512),        # M=1, medium
        (1, 1024, 1024),      # M=1, large
        (1, 2048, 2048),      # M=1, very large
        (1, 4096, 4096),      # M=1, huge
        (1, 8192, 8192),      # M=1, massive
        (128, 1, 128),        # N=1, small
        (512, 1, 512),        # N=1, medium
        (2048, 1, 2048),      # N=1, large
        (8, 1, 7200000),      # N=1, extreme case
        (256, 512, 1),        # K=1, small
        (1024, 2048, 1),      # K=1, large
    ]
    
    print("\n" + "="*90)
    print("Performance Comparison: tritonblas.mv vs torch.matmul")
    print("="*90)
    print(f"{'M':>6} {'N':>8} {'K':>8} {'PyTorch (ms)':>15} {'tritonblas (ms)':>17} {'Speedup':>10}")
    print("-"*90)
    
    for M, N, K in test_configs:
        # Create test data
        A = torch.randn(M, N, device=device, dtype=dtype)
        x = torch.randn(N, device=device, dtype=dtype)
        y_torch = torch.zeros(M, device=device, dtype=dtype)
        y_triton = torch.zeros(M, device=device, dtype=dtype)
        
        # Warmup PyTorch
        for _ in range(num_warmup):
            torch.matmul(A, x, out=y_torch)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(num_iters):
            torch.matmul(A, x, out=y_torch)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / num_iters * 1000
        
        # Warmup tritonblas
        for _ in range(num_warmup):
            tritonblas.mv(A, x, c=y_triton)
        torch.cuda.synchronize()
        
        # Benchmark tritonblas
        start = time.perf_counter()
        for _ in range(num_iters):
            tritonblas.mv(A, x, c=y_triton)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_iters * 1000
        
        speedup = torch_time / triton_time
        print(f"{M:>6} {N:>8} {K:>8} {torch_time:>15.4f} {triton_time:>17.4f} {speedup:>10.2f}x")
    
    print("="*90)


if __name__ == "__main__":
    print("Running GEMV tests...\n")
    
    # Run parameterized tests manually
    print("Running basic GEMV tests...")
    for M, N, K in [(128, 256, 256), (1, 256, 256), (256, 1, 256), (256, 256, 1)]:
        test_gemv_basic(M, N, K)
    
    print("\nRunning transposed GEMV tests...")
    for M, N, K in [(128, 256, 256), (1, 256, 256), (256, 1, 256)]:
        test_gemv_transposed(M, N, K)
    
    print("\nRunning preallocated output tests...")
    for M, N, K in [(64, 128, 128), (1, 128, 128), (64, 1, 64)]:
        test_gemv_with_preallocated_output(M, N, K)
    
    print("\nRunning various sizes tests...")
    for M, N, K in [(1, 64, 64), (1, 128, 128), (1, 512, 512), (1, 2048, 2048), (64, 1, 64), (128, 1, 128), (512, 1, 512), (2048, 1, 2048), (100, 200, 1), (333, 777, 1)]:
        test_gemv_various_sizes(M, N, K)
    
    print("\nRunning bfloat16 tests...")
    for M, N, K in [(128, 256, 256), (1, 256, 256)]:
        test_gemv_bfloat16(M, N, K)
    
    print("\nRunning matmul with vector tests...")
    for M, N, K in [(128, 256, 256), (1, 256, 256)]:
        test_matmul_with_vector(M, N, K)
    
    print("\nRunning error handling tests...")
    test_matmul_vector_error_handling()
    test_gemv_dimension_mismatch()
    
    print("\nRunning large matrix tests...")
    for M, N, K in [(2048, 4096, 4096), (1, 4096, 4096)]:
        test_gemv_large_matrix(M, N, K)
    
    print("\nRunning edge case tests...")
    for M, N, K in [(128, 256, 512), (1, 256, 512), (128, 1, 512), (128, 256, 1), (1, 1, 512), (1, 256, 1), (128, 1, 1), (8, 1, 7200000)]:
        test_gemv_edge_cases(M, N, K)
    
    print("\nRunning column vector tests...")
    for M, K in [(128, 512), (1, 512), (256, 1), (1024, 2048)]:
        test_matmul_with_column_vectors(M, K)
    
    print("\nRunning row vector tests...")
    for K, N in [(128, 256), (512, 1), (1, 512), (2048, 1024)]:
        test_matmul_with_row_vectors(K, N)
    
    print("\n✅ All GEMV tests passed!")
    
    # Run performance benchmark
    benchmark_gemv()

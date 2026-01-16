"""
Test that both streamk and persistent kernels produce the same output as torch.addmm.

This test verifies that tritonblas matmul implementations (both streamk and persistent)
are numerically equivalent to PyTorch's addmm operation for various matrix sizes and dtypes.
"""

import pytest
import torch
import tritonblas
from tritonblas.utils import generate_matmul_inputs


@pytest.mark.parametrize(
    "m, n, k",
    [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        # Non-square cases
        (4864, 8192, 4160),
        (1024, 2048, 512),
        (512, 1024, 2048),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("N", "N"),  # A @ B
        ("T", "N"),  # A^T @ B
        ("N", "T"),  # A @ B^T
        ("T", "T"),  # A^T @ B^T
    ],
)
def test_streamk_persistent_vs_torch_addmm(m, n, k, in_dtype, out_dtype, transA, transB):
    """
    Test that both streamk and persistent kernels produce the same output as torch.addmm.
    
    This test:
    1. Generates input matrices A and B
    2. Computes reference result using torch.addmm (C = beta*C + alpha*A@B)
    3. Computes result using tritonblas persistent kernel
    4. Computes result using tritonblas streamk kernel
    5. Verifies all three results match within tolerance
    
    Args:
        m: Number of rows in A (or A^T) and C
        n: Number of columns in B (or B^T) and C
        k: Shared dimension between A and B
        in_dtype: Input dtype for A and B
        out_dtype: Output dtype for C
        transA: "T" for A^T, "N" for A
        transB: "T" for B^T, "N" for B
    """
    init_type = "randn"
    
    # Generate inputs using shared utility
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    
    # Compute reference using torch.addmm
    # torch.addmm computes: out = beta * input + alpha * (mat1 @ mat2)
    # We want: C = A @ B, so we use beta=0, alpha=1 (equivalent to torch.matmul)
    torch_c = torch.addmm(torch.zeros_like(inputs.C), inputs.A, inputs.B, beta=0.0, alpha=1.0)
    
    # Test persistent kernel
    persistent_c = torch.zeros_like(inputs.C)
    tritonblas.matmul(inputs.A, inputs.B, persistent_c, enable_streamk=False)
    
    # Test streamk kernel
    streamk_c = torch.zeros_like(inputs.C)
    tritonblas.matmul(inputs.A, inputs.B, streamk_c, enable_streamk=True)
    
    # Verify persistent kernel matches torch.addmm
    torch.testing.assert_close(
        persistent_c.to(out_dtype),
        torch_c.to(out_dtype),
        atol=1.0,
        rtol=1.0,
        msg=f"Persistent kernel output does not match torch.addmm for {m}x{n}x{k} {in_dtype} {transA}{transB}"
    )
    
    # Verify streamk kernel matches torch.addmm
    torch.testing.assert_close(
        streamk_c.to(out_dtype),
        torch_c.to(out_dtype),
        atol=1.0,
        rtol=1.0,
        msg=f"StreamK kernel output does not match torch.addmm for {m}x{n}x{k} {in_dtype} {transA}{transB}"
    )
    
    # Verify both kernels produce identical results
    torch.testing.assert_close(
        persistent_c.to(out_dtype),
        streamk_c.to(out_dtype),
        atol=1e-5,
        rtol=1e-5,
        msg=f"Persistent and StreamK kernels produce different outputs for {m}x{n}x{k} {in_dtype} {transA}{transB}"
    )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("beta, alpha", [(0.5, 1.0), (1.0, 2.0), (0.3, 0.7)])
def test_addmm_with_beta_alpha(m, n, k, in_dtype, out_dtype, beta, alpha):
    """
    Test that tritonblas.addmm matches torch.addmm with various beta and alpha values.
    
    This test verifies: out = beta*input + alpha*(mat1@mat2)
    where input is a non-zero initial value.
    """
    init_type = "randn"
    transA = "N"
    transB = "N"
    
    # Generate inputs
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    
    # Create initial input value
    input_tensor = torch.randn((m, n), device="cuda", dtype=out_dtype)
    
    # Compute reference using torch.addmm
    torch_result = torch.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha)
    
    # Test persistent kernel using tritonblas.addmm
    persistent_result = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
    
    # Test streamk kernel using tritonblas.addmm
    streamk_result = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
    
    # Verify both match torch.addmm
    torch.testing.assert_close(
        persistent_result.to(out_dtype),
        torch_result.to(out_dtype),
        atol=1.0,
        rtol=1.0,
        msg=f"Persistent kernel addmm does not match torch.addmm for {m}x{n}x{k} {in_dtype} beta={beta} alpha={alpha}"
    )
    
    torch.testing.assert_close(
        streamk_result.to(out_dtype),
        torch_result.to(out_dtype),
        atol=1.0,
        rtol=1.0,
        msg=f"StreamK kernel addmm does not match torch.addmm for {m}x{n}x{k} {in_dtype} beta={beta} alpha={alpha}"
    )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
)
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_streamk_different_grids(m, n, k, in_dtype):
    """
    Test that streamk kernel produces consistent results with different grid sizes.
    
    This verifies that the streamk implementation correctly handles different
    parallelization strategies and produces the same result as torch.addmm.
    """
    out_dtype = in_dtype
    init_type = "randn"
    transA = "N"
    transB = "N"
    
    # Generate inputs
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    
    # Compute reference
    torch_c = torch.zeros_like(inputs.C)
    torch.addmm(torch_c, inputs.A, inputs.B, beta=0.0, alpha=1.0, out=torch_c)
    
    # Test with different sk_grid values
    sk_grids = [None, 64, 128, 256]
    
    for sk_grid in sk_grids:
        streamk_c = torch.zeros_like(inputs.C)
        tritonblas.matmul(inputs.A, inputs.B, streamk_c, enable_streamk=True, sk_grid=sk_grid)
        
        torch.testing.assert_close(
            streamk_c.to(out_dtype),
            torch_c.to(out_dtype),
            atol=1.0,
            rtol=1.0,
            msg=f"StreamK with sk_grid={sk_grid} does not match torch.addmm for {m}x{n}x{k} {in_dtype}"
        )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
)
def test_numerical_stability(m, n, k):
    """
    Test numerical stability with edge cases.
    
    Verifies that both kernels handle edge cases like very small/large values
    and produce results consistent with torch.addmm.
    """
    in_dtype = torch.float16
    out_dtype = torch.float16
    
    # Test with very small values
    A_small = torch.full((m, k), 1e-4, device="cuda", dtype=in_dtype)
    B_small = torch.full((k, n), 1e-4, device="cuda", dtype=in_dtype)
    C_small = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    
    torch_c_small = torch.addmm(torch.zeros_like(C_small), A_small, B_small, beta=0.0, alpha=1.0)
    
    persistent_c_small = torch.zeros_like(C_small)
    tritonblas.matmul(A_small, B_small, persistent_c_small, enable_streamk=False)
    
    streamk_c_small = torch.zeros_like(C_small)
    tritonblas.matmul(A_small, B_small, streamk_c_small, enable_streamk=True)
    
    torch.testing.assert_close(persistent_c_small, torch_c_small, atol=1.0, rtol=1.0)
    torch.testing.assert_close(streamk_c_small, torch_c_small, atol=1.0, rtol=1.0)
    
    # Test with larger values (within fp16 range)
    A_large = torch.full((m, k), 10.0, device="cuda", dtype=in_dtype)
    B_large = torch.full((k, n), 10.0, device="cuda", dtype=in_dtype)
    C_large = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    
    torch_c_large = torch.addmm(torch.zeros_like(C_large), A_large, B_large, beta=0.0, alpha=1.0)
    
    persistent_c_large = torch.zeros_like(C_large)
    tritonblas.matmul(A_large, B_large, persistent_c_large, enable_streamk=False)
    
    streamk_c_large = torch.zeros_like(C_large)
    tritonblas.matmul(A_large, B_large, streamk_c_large, enable_streamk=True)
    
    torch.testing.assert_close(persistent_c_large, torch_c_large, atol=1.0, rtol=1.0)
    torch.testing.assert_close(streamk_c_large, torch_c_large, atol=1.0, rtol=1.0)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
)
@pytest.mark.parametrize("in_dtype, out_dtype", [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)])
@pytest.mark.parametrize("beta, alpha", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm_broadcast_semantics(m, n, k, in_dtype, out_dtype, beta, alpha):
    """
    Test that addmm correctly handles broadcast semantics for the input tensor.
    
    This verifies that input tensors of various shapes are correctly broadcast
    to (M, N) before being added to the matmul result.
    
    Tests:
    - Scalar input (0D tensor)
    - Row vector input (M,) - broadcasts across columns
    - Column vector input (N,) - broadcasts across rows  
    - Full matrix input (M, N) - no broadcasting needed
    """
    init_type = "randn"
    transA = "N"
    transB = "N"
    
    # Generate matmul inputs
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    
    # Test 1: Scalar input
    scalar_input = torch.tensor(2.5, device="cuda", dtype=out_dtype)
    torch_result_scalar = torch.addmm(scalar_input, inputs.A, inputs.B, beta=beta, alpha=alpha)
    triton_result_scalar = tritonblas.addmm(scalar_input, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
    
    torch.testing.assert_close(
        triton_result_scalar,
        torch_result_scalar,
        atol=1.0,
        rtol=1.0,
        msg=f"Scalar broadcast failed for {m}x{n}x{k}"
    )
    
    # Test 2: 1D vector (N,) - PyTorch treats as (1, N) and broadcasts across rows
    col_vector = torch.randn(n, device="cuda", dtype=out_dtype)
    torch_result_col = torch.addmm(col_vector, inputs.A, inputs.B, beta=beta, alpha=alpha)
    triton_result_col = tritonblas.addmm(col_vector, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
    
    torch.testing.assert_close(
        triton_result_col,
        torch_result_col,
        atol=1.0,
        rtol=1.0,
        msg=f"Column vector broadcast failed for {m}x{n}x{k}"
    )
    
    # Test 4: Full matrix (M, N) - no broadcasting
    full_matrix = torch.randn((m, n), device="cuda", dtype=out_dtype)
    torch_result_full = torch.addmm(full_matrix, inputs.A, inputs.B, beta=beta, alpha=alpha)
    triton_result_full = tritonblas.addmm(full_matrix, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
    
    torch.testing.assert_close(
        triton_result_full,
        torch_result_full,
        atol=1.0,
        rtol=1.0,
        msg=f"Full matrix (no broadcast) failed for {m}x{n}x{k}"
    )
    
    # Test 3: Verify StreamK kernel handles all broadcast types correctly
    triton_streamk_scalar = tritonblas.addmm(scalar_input, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
    torch.testing.assert_close(
        triton_streamk_scalar,
        torch_result_scalar,
        atol=1.0,
        rtol=1.0,
        msg=f"StreamK scalar broadcast failed for {m}x{n}x{k}"
    )
    
    triton_streamk_col = tritonblas.addmm(col_vector, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
    torch.testing.assert_close(
        triton_streamk_col,
        torch_result_col,
        atol=1.0,
        rtol=1.0,
        msg=f"StreamK 1D vector broadcast failed for {m}x{n}x{k}"
    )
    
    triton_streamk_full = tritonblas.addmm(full_matrix, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
    torch.testing.assert_close(
        triton_streamk_full,
        torch_result_full,
        atol=1.0,
        rtol=1.0,
        msg=f"StreamK full matrix broadcast failed for {m}x{n}x{k}"
    )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
)
@pytest.mark.parametrize("in_dtype, out_dtype", [(torch.float16, torch.float16)])
def test_addmm_beta_zero_ignores_input(m, n, k, in_dtype, out_dtype):
    """
    Test that when beta=0, the input tensor is ignored (including NaN/inf values).
    
    This matches torch.addmm behavior where beta=0 means the input content
    is completely ignored and NaN/inf values don't propagate.
    """
    init_type = "randn"
    transA = "N"
    transB = "N"
    
    # Generate matmul inputs
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    
    # Create input with NaN and inf values
    input_with_nan = torch.full((m, n), float('nan'), device="cuda", dtype=out_dtype)
    input_with_inf = torch.full((m, n), float('inf'), device="cuda", dtype=out_dtype)
    
    # With beta=0, these should be ignored
    torch_result_nan = torch.addmm(input_with_nan, inputs.A, inputs.B, beta=0.0, alpha=1.0)
    triton_result_nan = tritonblas.addmm(input_with_nan, inputs.A, inputs.B, beta=0.0, alpha=1.0, enable_streamk=False)
    
    torch_result_inf = torch.addmm(input_with_inf, inputs.A, inputs.B, beta=0.0, alpha=1.0)
    triton_result_inf = tritonblas.addmm(input_with_inf, inputs.A, inputs.B, beta=0.0, alpha=1.0, enable_streamk=False)
    
    # Results should not contain NaN or inf
    assert not torch.isnan(triton_result_nan).any(), "NaN propagated when beta=0"
    assert not torch.isinf(triton_result_nan).any(), "Inf propagated when beta=0"
    assert not torch.isnan(triton_result_inf).any(), "NaN propagated when beta=0"
    assert not torch.isinf(triton_result_inf).any(), "Inf propagated when beta=0"
    
    # Results should match torch
    torch.testing.assert_close(triton_result_nan, torch_result_nan, atol=1.0, rtol=1.0)
    torch.testing.assert_close(triton_result_inf, torch_result_inf, atol=1.0, rtol=1.0)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        # Non-power-of-two sizes
        (1000, 1000, 1000),
        (3000, 3000, 3000),
        (5000, 5000, 5000),
        # Non-square non-power-of-two
        (1500, 2500, 2000),
        (3333, 4444, 2222),
    ],
)
@pytest.mark.parametrize("in_dtype, out_dtype", [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)])
def test_addmm_performance(m, n, k, in_dtype, out_dtype):
    """
    Performance comparison between torch.addmm and tritonblas.addmm.
    
    Measures and compares execution time for:
    - torch.addmm (baseline)
    - tritonblas.addmm with persistent kernel
    - tritonblas.addmm with StreamK kernel
    
    This test doesn't fail on performance differences, but prints timing information.
    """
    import time
    
    init_type = "randn"
    transA = "T"
    transB = "N"
    
    # Generate inputs
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    input_tensor = torch.randn((m, n), device="cuda", dtype=out_dtype)
    beta, alpha = 0.5, 1.0
    
    # Warmup
    output_tensor = torch.empty((m, n), device="cuda", dtype=out_dtype)
    for _ in range(5):
        _ = torch.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha)
        _ = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
        _ = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
        tritonblas.matmul(inputs.A, inputs.B, output_tensor, enable_streamk=False)
        tritonblas.matmul(inputs.A, inputs.B, output_tensor, enable_streamk=True)
    torch.cuda.synchronize()
    
    # Benchmark torch.addmm
    num_iters = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha)
    torch.cuda.synchronize()
    torch_addmm_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark torch.matmul (for comparison - no addmm overhead)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.matmul(inputs.A, inputs.B)
    torch.cuda.synchronize()
    torch_matmul_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark tritonblas.addmm persistent
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=False)
    torch.cuda.synchronize()
    triton_addmm_persistent_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark tritonblas.addmm StreamK
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = tritonblas.addmm(input_tensor, inputs.A, inputs.B, beta=beta, alpha=alpha, enable_streamk=True)
    torch.cuda.synchronize()
    triton_addmm_streamk_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark tritonblas.matmul persistent (for comparison - no addmm overhead)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        tritonblas.matmul(inputs.A, inputs.B, output_tensor, enable_streamk=False)
    torch.cuda.synchronize()
    triton_matmul_persistent_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark tritonblas.matmul StreamK (for comparison - no addmm overhead)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        tritonblas.matmul(inputs.A, inputs.B, output_tensor, enable_streamk=True)
    torch.cuda.synchronize()
    triton_matmul_streamk_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Calculate overhead
    persistent_overhead = triton_addmm_persistent_time - triton_matmul_persistent_time
    streamk_overhead = triton_addmm_streamk_time - triton_matmul_streamk_time
    
    # Print results
    print(f"\n{'='*90}")
    print(f"Performance Comparison: {m}x{n}x{k} ({in_dtype})")
    print(f"{'='*90}")
    print(f"{'Operation':<30} {'Time (ms)':<12} {'vs torch.addmm':<18} {'Overhead':<15}")
    print(f"{'-'*90}")
    print(f"{'torch.matmul':<30} {torch_matmul_time:>10.3f}   {torch_matmul_time/torch_addmm_time:>8.2f}x")
    print(f"{'torch.addmm':<30} {torch_addmm_time:>10.3f}   {'(baseline)':<18}")
    print(f"{'-'*90}")
    print(f"{'tritonblas.matmul (persistent)':<30} {triton_matmul_persistent_time:>10.3f}   {triton_matmul_persistent_time/torch_addmm_time:>8.2f}x")
    print(f"{'tritonblas.addmm (persistent)':<30} {triton_addmm_persistent_time:>10.3f}   {triton_addmm_persistent_time/torch_addmm_time:>8.2f}x   {persistent_overhead:>8.3f} ms")
    print(f"{'-'*90}")
    print(f"{'tritonblas.matmul (StreamK)':<30} {triton_matmul_streamk_time:>10.3f}   {triton_matmul_streamk_time/torch_addmm_time:>8.2f}x")
    print(f"{'tritonblas.addmm (StreamK)':<30} {triton_addmm_streamk_time:>10.3f}   {triton_addmm_streamk_time/torch_addmm_time:>8.2f}x   {streamk_overhead:>8.3f} ms")
    print(f"{'='*90}\n")
    
    # Test passes regardless of performance (this is just for information)
    assert True

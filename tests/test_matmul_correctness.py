"""
Tests for tritonblas.matmul with torch.autograd and torch.compile support

Tests cover:
1. Forward pass correctness against torch.mm
2. Backward pass gradient correctness against torch.mm
3. In-place (out=...) functionality and autograd restrictions
4. Edge cases including small dimensions
5. torch.compile compatibility
6. Persistent vs. StreamK compatibility
"""

import pytest
import torch
import tritonblas

from conftest import (
    STANDARD_DIMS, EDGE_CASE_DIMS, SKINNY_DIMS,
    DTYPES, USE_COMPILE, ENABLE_STREAMK, ENABLE_WORK_STEALING,
    MULTITRIAL_NUM_TRIALS,
)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", STANDARD_DIMS + EDGE_CASE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_forward_correctness(m, n, k, dtype, enable_streamk, work_stealing, use_compile):
    """Test that tritonblas.matmul forward pass matches torch.mm."""
    torch.manual_seed(42)

    a = torch.randn(m, k, device='cuda', dtype=dtype)
    b = torch.randn(k, n, device='cuda', dtype=dtype)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    # tritonblas result
    result = matmul_fn(a, b, enable_streamk=enable_streamk, work_stealing=work_stealing)

    # torch reference
    expected = torch.mm(a, b)

    # Check forward correctness with relaxed tolerance for low precision
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("trial", range(MULTITRIAL_NUM_TRIALS))
@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", STANDARD_DIMS + EDGE_CASE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_backward_correctness(m, n, k, dtype, enable_streamk, work_stealing, use_compile, trial):
    """Test that tritonblas.matmul backward pass produces correct gradients."""
    torch.manual_seed(42 + trial)

    # Create inputs with requires_grad for tritonblas
    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)

    # Clone for torch reference
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    # Forward pass
    result = matmul_fn(a, b, enable_streamk=enable_streamk, work_stealing=work_stealing)
    result_ref = torch.mm(a_ref, b_ref)

    # Backward pass with same upstream gradient
    grad_output = torch.randn_like(result)
    result.backward(grad_output)
    result_ref.backward(grad_output)

    # Check gradients match
    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1,
                               msg="a gradient mismatch")
    torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-1, rtol=1e-1,
                               msg="b gradient mismatch")


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", SKINNY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_skinny_matrices(m, n, k, dtype, enable_streamk, work_stealing, use_compile):
    """Test matmul with skinny matrices (large K dimension)."""
    torch.manual_seed(42)

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    # Forward
    result = matmul_fn(a, b, enable_streamk=enable_streamk, work_stealing=work_stealing)
    result_ref = torch.mm(a_ref, b_ref)

    torch.testing.assert_close(result, result_ref, atol=1e-1, rtol=1e-1)

    # Backward
    result.sum().backward()
    result_ref.sum().backward()

    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_matmul_inplace_with_grad_raises(enable_streamk, work_stealing, use_compile):
    """Test that matmul with out=... raises RuntimeError when autograd is enabled."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    with pytest.raises(RuntimeError, match="don't support automatic differentiation"):
        matmul_fn(a, b, out=out, enable_streamk=enable_streamk, work_stealing=work_stealing)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_matmul_inplace_without_grad_works(enable_streamk, work_stealing, use_compile):
    """Test that matmul with out=... works when autograd is disabled."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    # Should work with torch.no_grad()
    with torch.no_grad():
        result = matmul_fn(a, b, out=out, enable_streamk=enable_streamk, work_stealing=work_stealing)

    # In-place path returns None (custom ops don't support aliasing)
    assert result is None, "in-place matmul should return None"

    # Verify correctness against torch
    expected = torch.mm(a, b)
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_matmul_inplace_output_correctness(enable_streamk, work_stealing, use_compile):
    """Test that matmul in-place mode produces correct results."""
    torch.manual_seed(42)
    m, n, k = 128, 256, 512
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype)
    b = torch.randn(k, n, device='cuda', dtype=dtype)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    with torch.no_grad():
        matmul_fn(a, b, out=out, enable_streamk=enable_streamk, work_stealing=work_stealing)

    expected = torch.mm(a, b)
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_matmul_no_grad_tensors(enable_streamk, work_stealing, use_compile):
    """Test matmul works when input tensors don't require grad."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=False)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=False)

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    result = matmul_fn(a, b, enable_streamk=enable_streamk, work_stealing=work_stealing)
    expected = torch.mm(a, b)

    torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("work_stealing", ENABLE_WORK_STEALING)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_matmul_partial_grad(enable_streamk, work_stealing, use_compile):
    """Test matmul when only some inputs require grad."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    # Only a requires grad
    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=False)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone()

    matmul_fn = tritonblas.matmul
    if use_compile:
        matmul_fn = torch.compile(tritonblas.matmul, fullgraph=True)

    result = matmul_fn(a, b, enable_streamk=enable_streamk, work_stealing=work_stealing)
    result_ref = torch.mm(a_ref, b_ref)

    result.sum().backward()
    result_ref.sum().backward()

    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1)

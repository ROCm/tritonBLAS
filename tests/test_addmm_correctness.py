"""
Tests for tritonblas.addmm with torch.autograd and torch.compile support

Tests cover:
1. Forward pass correctness against torch.addmm
2. Backward pass gradient correctness against torch.addmm
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
    DTYPES, USE_COMPILE, ENABLE_STREAMK, MULTITRIAL_NUM_TRIALS,
)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", STANDARD_DIMS + EDGE_CASE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_forward_correctness(m, n, k, dtype, enable_streamk, use_compile):
    """Test that tritonblas.addmm forward pass matches torch.addmm."""
    torch.manual_seed(42)

    a = torch.randn(m, k, device='cuda', dtype=dtype)
    b = torch.randn(k, n, device='cuda', dtype=dtype)
    bias = torch.randn(n, device='cuda', dtype=dtype)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    # tritonblas result
    result = addmm_fn(bias, a, b, enable_streamk=enable_streamk)

    # torch reference
    expected = torch.addmm(bias, a, b)

    # Check forward correctness with relaxed tolerance for low precision
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("trial", range(MULTITRIAL_NUM_TRIALS))
@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", STANDARD_DIMS + EDGE_CASE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_backward_correctness(m, n, k, dtype, enable_streamk, use_compile, trial):
    """Test that tritonblas.addmm backward pass produces correct gradients."""
    torch.manual_seed(42 + trial)

    # Create inputs with requires_grad for tritonblas
    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=True)

    # Clone for torch reference
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    # Forward pass
    result = addmm_fn(bias, a, b, enable_streamk=enable_streamk)
    result_ref = torch.addmm(bias_ref, a_ref, b_ref)

    # Backward pass with same upstream gradient
    grad_output = torch.randn_like(result)
    result.backward(grad_output)
    result_ref.backward(grad_output)

    # Check gradients match
    torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-1, rtol=1e-1,
                               msg="bias gradient mismatch")
    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1,
                               msg="a gradient mismatch")
    torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-1, rtol=1e-1,
                               msg="b gradient mismatch")


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
@pytest.mark.parametrize("m, n, k", SKINNY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_skinny_matrices(m, n, k, dtype, enable_streamk, use_compile):
    """Test addmm with skinny matrices (large K dimension)."""
    torch.manual_seed(42)

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=True)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    # Forward
    result = addmm_fn(bias, a, b, enable_streamk=enable_streamk)
    result_ref = torch.addmm(bias_ref, a_ref, b_ref)

    torch.testing.assert_close(result, result_ref, atol=1e-1, rtol=1e-1)

    # Backward
    result.sum().backward()
    result_ref.sum().backward()

    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_addmm_inplace_with_grad_raises(enable_streamk, use_compile):
    """Test that addmm with out=... raises RuntimeError when autograd is enabled."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=True)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    with pytest.raises(RuntimeError, match="don't support automatic differentiation"):
        addmm_fn(bias, a, b, out=out, enable_streamk=enable_streamk)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_addmm_inplace_without_grad_works(enable_streamk, use_compile):
    """Test that addmm with out=... works when autograd is disabled."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=True)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=True)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    # Should work with torch.no_grad()
    with torch.no_grad():
        result = addmm_fn(bias, a, b, out=out, enable_streamk=enable_streamk)

    # In-place path returns None (custom ops don't support aliasing)
    assert result is None, "in-place addmm should return None"

    # Verify correctness against torch
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_addmm_inplace_output_correctness(enable_streamk, use_compile):
    """Test that addmm in-place mode produces correct results."""
    torch.manual_seed(42)
    m, n, k = 128, 256, 512
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype)
    b = torch.randn(k, n, device='cuda', dtype=dtype)
    bias = torch.randn(n, device='cuda', dtype=dtype)
    out = torch.empty(m, n, device='cuda', dtype=dtype)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    with torch.no_grad():
        addmm_fn(bias, a, b, out=out, enable_streamk=enable_streamk)

    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_addmm_no_grad_tensors(enable_streamk, use_compile):
    """Test addmm works when input tensors don't require grad."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=False)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=False)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=False)

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    result = addmm_fn(bias, a, b, enable_streamk=enable_streamk)
    expected = torch.addmm(bias, a, b)

    torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("use_compile", USE_COMPILE)
@pytest.mark.parametrize("enable_streamk", ENABLE_STREAMK)
def test_addmm_partial_grad(enable_streamk, use_compile):
    """Test addmm when only some inputs require grad."""
    torch.manual_seed(42)
    m, n, k = 64, 64, 64
    dtype = torch.bfloat16

    # Only a requires grad
    a = torch.randn(m, k, device='cuda', dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device='cuda', dtype=dtype, requires_grad=False)
    bias = torch.randn(n, device='cuda', dtype=dtype, requires_grad=False)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone()
    bias_ref = bias.detach().clone()

    addmm_fn = tritonblas.addmm
    if use_compile:
        addmm_fn = torch.compile(tritonblas.addmm, fullgraph=True)

    result = addmm_fn(bias, a, b, enable_streamk=enable_streamk)
    result_ref = torch.addmm(bias_ref, a_ref, b_ref)

    result.sum().backward()
    result_ref.sum().backward()

    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-1, rtol=1e-1)

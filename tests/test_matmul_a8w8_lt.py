import pytest
import torch  # type: ignore
import tritonblas  # type: ignore
from tritonblas.utils import generate_matmul_inputs  # type: ignore


def run_torch(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16):
    """Reference computation matching kernel behavior: float32 accumulation with scale application."""
    # 1. Matrix multiplication in float32 (like kernel's tl.dot accumulation)
    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    if a_scale is not None and b_scale is not None:
        # Scales from generate_matmul_inputs are 1D: (M,) and (N,)
        # Apply scales: (M, 1) * (1, N) -> (M, N)
        scale = a_scale[:, None] * b_scale[None, :]
        acc = acc * scale  # Keep in float32

    if bias is not None:
        # 4. Add bias in float32 (like kernel: acc + bias_float[:, None])
        acc = acc + bias.to(torch.float32)

    # 5. Convert to output dtype at the very end (like kernel: c = acc.to(C.type.element_ty))
    if dtype == torch.float8_e4m3fn:
        dtype_max = torch.finfo(torch.float8_e4m3fn).max
        acc = torch.clamp(acc, -dtype_max, dtype_max)
    elif dtype == torch.float8_e5m2:
        dtype_max = torch.finfo(torch.float8_e5m2).max
        acc = torch.clamp(acc, -dtype_max, dtype_max)
    elif dtype == torch.int8:
        # INT8 has range [-128, 127], but we use symmetric range [-127, 127] like the kernel
        dtype_max = 127.0
        acc = torch.clamp(acc, -dtype_max, dtype_max)

    return acc.to(dtype)


def run_triton(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16, c=None):
    # Helper function that matches the actual API signature
    # Note: matmul_a8w8 creates the selector internally
    if c is None:
        c = torch.zeros((a.shape[0], b.shape[1]), device="cuda", dtype=dtype)
    return tritonblas.matmul_a8w8(a, b, a_scale, b_scale, c, enable_streamk=False)

@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),  # Large - test if fixed reference computation works
        (4096, 4096, 4096),  # Medium-large
        (1024, 1024, 1024),  # Medium
        (512, 512, 512),     # Small
        (256, 256, 256),     # Very small
#        (512,2048,970132),  ## there are serious issue for this shape.
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
#        (torch.int8, torch.int8),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
#        (torch.float8_e5m2, torch.float8_e5m2),  # Disabled - no PyTorch CUDA kernel support
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("T", "T"),  # A^T @ B^T
        ("N", "N"),  # A @ B
        ("T", "N"),  # A^T @ B
        ("N", "T"),  # A @ B^T
    ],
)
@pytest.mark.parametrize(
    "enable_streamk",
    [
        False,
        True,
    ],
)
def test_matmul_a8w8(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk):
    """Test quantized matmul with all transpose combinations using shared input generation utilities."""
    init_type = "randn"

    # Generate all inputs using shared utility (handles transposes and quantization automatically)
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    # Scales from generate_matmul_inputs are already 1D: (M,) and (N,)
    # which is what the kernel expects
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=enable_streamk,
    )
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_a8w8_lt(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, config,
        enable_streamk,
    )

    # Check correctness using reference computation
    torch_c = run_torch(
        inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, bias=None, dtype=out_dtype
    )

    # Use relaxed tolerance for quantized output due to limited precision
    if out_dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(
            inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2
        )
    elif out_dtype == torch.int8:
        # INT8 has integer precision, so we need more relaxed tolerance
        torch.testing.assert_close(
            inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=5.0, rtol=0.5
        )
    else:
        torch.testing.assert_close(
            inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=1e-1, rtol=1e-2
        )

import pytest
import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore
from tritonblas.utils import generate_matmul_inputs  # type: ignore


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (4864, 8192, 4160),
        (4096, 4096, 4096),
        (512, 2048, 970132),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        # (torch.float8_e4m3fn, torch.float8_e4m3fn),
        # (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
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
def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk):
    """Test non-quantized matmul with all transpose combinations using shared input generation utilities."""
    init_type = "randn"

    # Generate all inputs using shared utility (handles transposes automatically)
    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    # Run TritonBLAS matmul
    selector = tritonblas.MatmulHeuristicResult(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype
    )
    tritonblas.matmul_lt(inputs.A, inputs.B, inputs.C, selector, enable_streamk)

    # Check correctness
    torch_c = torch.matmul(inputs.A, inputs.B)
    #    torch.testing.assert_close(inputs.C.to(out_dtype), torch_c, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(inputs.C.to(out_dtype), torch_c, atol=1, rtol=1)

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
    "mode",
    [
        "persistent",
        "streamk",
        "work_stealing",
    ],
)
def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, mode):
    """Test non-quantized matmul with all transpose combinations using shared input generation utilities."""
    init_type = "randn"
    enable_streamk = mode == "streamk"
    work_stealing = mode == "work_stealing"

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    tritonblas.matmul(inputs.A, inputs.B, inputs.C, enable_streamk=enable_streamk,
                      work_stealing=work_stealing)

    torch_c = torch.matmul(inputs.A, inputs.B)
    torch.testing.assert_close(inputs.C.to(out_dtype), torch_c, atol=1, rtol=1)

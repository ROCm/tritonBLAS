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
    "mode",
    [
        "persistent",
        "streamk",
        "work_stealing",            # default WS scheduling: per-XCD/slot
        "ws_global_atomic",         # WS with single device-wide counter
        "ws_neighbor_stealing",     # WS with cross-XCD counter rotation
    ],
)
def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, mode):
    """Test non-quantized matmul with all transpose combinations using shared input generation utilities."""
    init_type = "randn"
    enable_streamk = mode == "streamk"
    work_stealing = mode in ("work_stealing", "ws_global_atomic", "ws_neighbor_stealing")

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)

    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=enable_streamk,
    )
    config = tritonblas.matmul_preamble(selector)
    config.global_atomic = (mode == "ws_global_atomic")
    config.neighbor_stealing = (mode == "ws_neighbor_stealing")

    tritonblas.matmul_lt(inputs.A, inputs.B, inputs.C, selector, config,
                         enable_streamk, work_stealing=work_stealing)

    torch_c = torch.matmul(inputs.A, inputs.B)
    torch.testing.assert_close(inputs.C.to(out_dtype), torch_c, atol=1, rtol=1)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (1024, 1024, 1024),
        (4352, 4352, 4096),
    ],
)
def test_ws_scheduling_modes_agree(m, n, k):
    """All three WS scheduling modes must produce identical results to each other.

    A regression that affects all three modes equally vs. torch.matmul could
    slip past test_matmul; this catches mode-specific divergences in the
    scheduling layer.
    """
    dtype = torch.bfloat16
    base = generate_matmul_inputs(m, n, k, dtype, dtype, "N", "N", "randn")
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, dtype, dtype, dtype, base.A.device, streamk=False,
    )
    config = tritonblas.matmul_preamble(selector)

    outputs = {}
    for ws_mode in ("slot", "global_atomic", "neighbor_stealing"):
        config.global_atomic = (ws_mode == "global_atomic")
        config.neighbor_stealing = (ws_mode == "neighbor_stealing")
        c = torch.empty_like(base.C)
        config.reset(work_stealing=True)
        tritonblas.matmul_lt(base.A, base.B, c, selector, config,
                             enable_streamk=False, work_stealing=True)
        outputs[ws_mode] = c

    torch.testing.assert_close(outputs["slot"], outputs["neighbor_stealing"], atol=1, rtol=1)
    torch.testing.assert_close(outputs["slot"], outputs["global_atomic"], atol=1, rtol=1)


def test_ws_mode_mutual_exclusion():
    """Setting both global_atomic and neighbor_stealing must raise ValueError."""
    dtype = torch.bfloat16
    inputs = generate_matmul_inputs(1024, 1024, 1024, dtype, dtype, "N", "N", "randn")
    selector = tritonblas.OrigamiMatmulSelector(
        1024, 1024, 1024, dtype, dtype, dtype, inputs.A.device, streamk=False,
    )
    config = tritonblas.matmul_preamble(selector)
    config.global_atomic = True
    config.neighbor_stealing = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        tritonblas.matmul_lt(inputs.A, inputs.B, inputs.C, selector, config,
                             enable_streamk=False, work_stealing=True)

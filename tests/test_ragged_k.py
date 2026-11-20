"""Persistent GEMM on ragged shapes: K not a multiple of BLOCK_K, partial M/N tiles, all four operand layouts.

The K loop skips wrapping K indices into bounds; these shapes check that the masked K tail and the
M/N edge tiles still produce correct results whichever operand has K as its strided dimension.
"""

import pytest
import torch
import tritonblas
from tritonblas.matmul import persistent_matmul_lt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.mark.parametrize(
    "m, n, k",
    [
        (15, 17, 33),
        (127, 255, 100),
        (1000, 777, 333),
        (513, 7168, 384),
        (4097, 2049, 1031),
        (2048, 3584, 390),
        (300, 5000, 4160),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("a_col_major", [False, True])
@pytest.mark.parametrize("b_col_major", [False, True])
def test_persistent_matmul_ragged(m, n, k, dtype, a_col_major, b_col_major):
    torch.manual_seed(0)
    a = (
        torch.randn((k, m), device="cuda", dtype=dtype).t()
        if a_col_major
        else torch.randn((m, k), device="cuda", dtype=dtype)
    )
    b = (
        torch.randn((n, k), device="cuda", dtype=dtype).t()
        if b_col_major
        else torch.randn((k, n), device="cuda", dtype=dtype)
    )
    c = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, a.device)

    persistent_matmul_lt(a, b, c, selector)

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c.float(), ref, atol=0.05 * k**0.5, rtol=1e-2)

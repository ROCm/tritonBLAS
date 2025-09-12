import pytest
import torch
import triton
import tritonblas


@pytest.mark.parametrize(
    "m, n, k, total_programs_streamk, in_dtype, out_dtype",
    [
        (8192, 8192, 8192, 304, torch.float16, torch.float16),
        (4864, 8192, 4160, 304, torch.float16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "BLK_M, BLK_N, BLK_K",
    [
        (256, 256, 64),
        (128, 128, 64),
        (256, 128, 64),
    ],
)
@pytest.mark.parametrize("gsize_m", [1])
@pytest.mark.parametrize("group_size", [1, 2, 4, 6, 8])
def test_grouped_gemm(m, n, k, total_programs_streamk, in_dtype, out_dtype, BLK_M, BLK_N, BLK_K, gsize_m, group_size):
    
    group_A = []
    group_B = []
    group_C = []
    torch_result = []

    for i in range(group_size):
        A = torch.randn(m, k, device="cuda", dtype=in_dtype)
        B = torch.randn(k, n, device="cuda", dtype=in_dtype)
        C = torch.empty((m, n), device="cuda", dtype=out_dtype) 
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        torch_result.append(torch.matmul(A, B))

    tritonblas.grouped_gemm(
        group_A,
        group_B,
        group_C,
        BLK_M,
        BLK_N,
        BLK_K,
    )
    for i in range(group_size):
        torch.testing.assert_close(torch_result[i], group_C[i], atol=0.5, rtol=0.5)

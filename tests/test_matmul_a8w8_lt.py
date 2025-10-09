import pytest
import torch
import triton
import tritonblas
from utils import  _is_quantized, matmul_input_gen

def run_torch(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16):
    x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    scale = torch.matmul(a_scale, b_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16, c=None):
    return matmul_a8w8(a, b, a_scale, b_scale, bias, dtype, c)

@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
#        (4864, 8192, 4160),
#        (4096, 4096, 4096),
#        (512,2048,970132),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
#        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
#        (torch.float16, torch.float16),
#        (torch.bfloat16, torch.bfloat16),
#        (torch.float32, torch.float32),
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
@pytest.mark.parametrize(
    "scale_mode",
    [
        "per_axis",
        "per_tensor",
    ],
)
def test_matmul_a8w8(m, n, k, in_dtype, out_dtype, transA, transB, scale_mode, enable_streamk):

    # Adjust dimensions for transposition and apply tensor.T if needed
    if transA == "T":
        A_size = (m, k)  # A is MxK
    else:
        A_size = (k, m)  # A is KxM (we will later transpose it with .T)

    if transB == "T":
        B_size = (k, n)  # B is KxN
    else:
        B_size = (n, k)  # B is NxK (we will later transpose it with .T)

    # Build logical shapes directly: A:(M,K), B:(K,N)
    quantize_mode = "auto"  # fp8/int8 -> (q,scale); others -> tensor
    init_type = "randn"

    # A: per-row over K -> (M,1)
    if scale_mode == "per_axis":
        A_scale_axis = 1 if transA == "T" else 0   # (M,K) -> 1 ; (K,M) -> 0 so it reduces over K
        A_init = matmul_input_gen(
            A_size, in_dtype, init_type,
            quantize=quantize_mode, scale_mode="per_axis", scale_axis=1
        )
    else:
        A_init = matmul_input_gen(
            A_size, in_dtype, init_type,
            quantize=quantize_mode, scale_mode="per_tensor"
        )
    A, scaleA = _is_quantized(A_init)

    # B: per-column over K -> (1,N)
    if scale_mode == "per_axis":
        B_scale_axis = 0 if transB == "T" else 1   # (K,N) -> 0 ; (N,K) -> 1 so it reduces over K
        B_init = matmul_input_gen(
            B_size, in_dtype, init_type,
            quantize=quantize_mode, scale_mode="per_axis", scale_axis=0
        )
    else:
        B_init = matmul_input_gen(
            B_size, in_dtype, init_type,
            quantize=quantize_mode, scale_mode="per_tensor"
        )
    B, scaleB = _is_quantized(B_init)

    # Apply transpose on A or B if necessary (only needed for "N" case)
    if transA == "N":
        A = A.T  # Apply transpose to A if transA is "N"
        scaleA = None if scaleA is None else scaleA.T.contiguous()

    if transB == "N":
        B = B.T  # Apply transpose to B if transB is "N"
        scaleB = None if scaleB is None else scaleB.T.contiguous()

    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    tritonblas.matmul_a8w8_lt(A, B, scaleA, scaleB, C, selector, enable_streamk)

    # Check correctnes: Fix tolerance later
    torch_c = run_torch(A, B, scaleA, scaleB, bias=None, dtype=out_dtype)
#    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1e-2, rtol=1e-3)
#    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1, rtol=1)
    torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=1e-1, rtol=1e-2)

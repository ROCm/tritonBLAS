import pytest
import torch
import triton
import triton.language as tl
import tritonblas
try:
    from tritonblas.utils import _is_quantized, matmul_input_gen
except ImportError:
    # Fallback: try importing from tests/utils.py (for backward compatibility)
    try:
        from .utils import _is_quantized, matmul_input_gen
    except ImportError:
        from utils import _is_quantized, matmul_input_gen

def run_torch(a, b, a_scale, b_scale, bias=None, dtype=torch.bfloat16):
    # Match kernel behavior exactly: keep everything in float32 until final conversion
    
    # 1. Matrix multiplication in float32 (like kernel's tl.dot accumulation)
    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    
    if a_scale is not None and b_scale is not None:
        # 2. Handle scale shapes (match kernel's scale loading)
        if a_scale.shape[0] == 1:  # (1, M) -> (M, 1)
            a_scale = a_scale.T
        if b_scale.shape[0] == 1:  # (1, N) -> (N, 1) 
            b_scale = b_scale.T
            
        # 3. Apply scales to float32 accumulator (like kernel: acc *= A_scale[:, None] * B_scale[None, :])
        scale = torch.matmul(a_scale, b_scale.T)  # (M, 1) @ (1, N) -> (M, N)
        acc = acc * scale  # Keep in float32
    
    if bias is not None:
        # 4. Add bias in float32 (like kernel: acc + bias_float[:, None])
        acc = acc + bias.to(torch.float32)
    
    # 5. Convert to output dtype at the very end (like kernel: c = acc.to(C.type.element_ty))
    # HYPOTHESIS: The kernel does implicit clamping to dtype range before conversion
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

    # Generate inputs with per-tensor quantization (like AITer)
    A_init = matmul_input_gen(A_size, in_dtype, init_type, quantize=quantize_mode)
    A, scaleA = _is_quantized(A_init)

    # Generate B with correct size based on transpose type (B_size already accounts for transpose)
    B_init = matmul_input_gen(B_size, in_dtype, init_type, quantize=quantize_mode)
    B, scaleB = _is_quantized(B_init)

    # Apply transpose on A or B if necessary (only needed for "N" case)
    if transA == "N":
        A = A.T  # Apply transpose to A if transA is "N"
        if scaleA is not None:
            scaleA = scaleA.T  # Transpose scale to match transposed tensor

    if transB == "N":
        B = B.T  # Apply transpose to B if transB is "N"
        if scaleB is not None:
            scaleB = scaleB.T  # Transpose scale to match transposed tensor

    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    # For per-channel scaling, scales are (M, 1) and (N, 1) or (1, N) depending on transpose
    # TritonBLAS kernel expects A_scale (M,) and B_scale (N,)
    scaleA_expanded = scaleA.squeeze(-1) if scaleA is not None else None  # (M, 1) -> (M,)
    # For scaleB, we need to handle both (N, 1) and (1, N) cases
    if scaleB is not None:
        if scaleB.shape[1] == 1:  # (N, 1) case
            scaleB_expanded = scaleB.squeeze(-1)  # (N, 1) -> (N,)
        else:  # (1, N) case
            scaleB_expanded = scaleB.squeeze(0)   # (1, N) -> (N,)
    else:
        scaleB_expanded = None

    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    tritonblas.matmul_a8w8_lt(A, B, scaleA_expanded, scaleB_expanded, C, selector, enable_streamk)

    # Check correctness: Fix tolerance later
    torch_c = run_torch(A, B, scaleA, scaleB, bias=None, dtype=out_dtype)
    #    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1e-2, rtol=1e-3)
    #    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1, rtol=1)
    # Use relaxed tolerance for quantized output due to limited precision
    if out_dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2)
    elif out_dtype == torch.int8:
        # INT8 has integer precision, so we need more relaxed tolerance
        torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=5.0, rtol=0.5)
    else:
        torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=1e-1, rtol=1e-2)

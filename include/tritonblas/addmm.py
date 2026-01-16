"""
addmm operation for tritonBLAS.

Implements torch.addmm compatible API with zero-copy in-kernel broadcasting.
"""
import torch
from typing import Optional
from .matmul import _make_matmul_selector, persistent_matmul_lt, streamk_matmul_lt


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    enable_streamk=False,
    sk_grid=None,
    out: torch.Tensor = None,
):
    """
    Performs the addmm operation: out = beta * input + alpha * (mat1 @ mat2)
    
    This function matches the signature and behavior of torch.addmm, supporting
    broadcast semantics for the input tensor. All broadcasting is handled in the
    kernel itself - no memory copies are performed.
    
    The input tensor is added to the final result with broadcasting:
    - If mat1 is (M x K) and mat2 is (K x N), then input must be broadcastable 
      with (M x N) and out will be (M x N)
    - input can be a scalar, vector, or matrix that broadcasts to (M x N)
    
    Formula: out = β * input + α * (mat1 @ mat2)
    
    If beta is 0, then input will be ignored (and NaN/inf in it won't propagate).
    
    Broadcasting is handled in the kernel using constexpr flags:
    - Scalars (0D): Kernel loads single value and broadcasts (C_SCALAR=True)
    - 1D vectors (N,): Kernel loads and broadcasts across rows (C_COL_BROADCAST=True)
    - 2D matrices: Kernel loads normally with stride-based broadcasting
    
    Note: PyTorch only accepts 1D tensors of size N (not M). A 1D tensor is
    treated as shape (1, N) and broadcasts across rows to (M, N).
    
    Args:
        input: Tensor to be added (broadcastable to M x N)
        mat1: First matrix for multiplication (shape: M x K)
        mat2: Second matrix for multiplication (shape: K x N)
        beta: Scalar multiplier for input (default: 1.0)
        alpha: Scalar multiplier for mat1 @ mat2 (default: 1.0)
        enable_streamk: Whether to use StreamK kernel (default: False)
        sk_grid: StreamK grid size (default: None)
        out: Output tensor (default: None, creates new tensor)
    
    Returns:
        Output tensor containing beta * input + alpha * (mat1 @ mat2)
    
    Examples:
        >>> import torch
        >>> import tritonblas
        >>> 
        >>> A = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
        >>> B = torch.randn(512, 2048, device='cuda', dtype=torch.float16)
        >>> 
        >>> # Scalar bias
        >>> result = tritonblas.addmm(torch.tensor(2.5, device='cuda'), A, B)
        >>> 
        >>> # Vector bias (N,) - broadcasts as (1, N) -> (M, N)
        >>> bias = torch.randn(2048, device='cuda', dtype=torch.float16)
        >>> result = tritonblas.addmm(bias, A, B, beta=0.5, alpha=1.0)
        >>> 
        >>> # Full matrix
        >>> C = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
        >>> result = tritonblas.addmm(C, A, B, beta=0.3, alpha=0.7)
        >>> 
        >>> # StreamK kernel
        >>> result = tritonblas.addmm(bias, A, B, enable_streamk=True)
    """
    assert mat1.shape[1] == mat2.shape[0], "Incompatible Dimensions"
    M, K = mat1.shape
    _, N = mat2.shape
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    
    # Determine broadcast mode based on input shape
    # NO copy operations - all broadcasting handled in kernel
    c_row_broadcast = False
    c_col_broadcast = False
    c_scalar = False
    
    if beta == 0.0:
        # Input is ignored - pass out as dummy (won't be read)
        c_input = out
    elif input.dim() == 0:  # scalar
        # Scalar - kernel will load single value and broadcast
        c_scalar = True
        # Convert to 1D tensor so we can get strides
        c_input = input.view(1)
    elif input.dim() == 1:
        # PyTorch treats 1D tensors as (1, N) for broadcasting
        # So a 1D tensor must have size N and broadcasts across rows
        if input.shape[0] == N:
            # Column vector (N,) - kernel will broadcast across rows
            c_col_broadcast = True
            c_input = input
        else:
            raise ValueError(f"1D input must have size N={N}, got {input.shape[0]}")
    elif input.dim() == 2:  # 2D matrix
        # Pass 2D input directly - kernel handles any broadcasting via strides
        c_input = input
    else:
        raise ValueError(f"input must be 0D, 1D, or 2D, got {input.dim()}D")
    
    # Call kernel - broadcasting handled by constexpr flags
    # C is for reading (input), OUT is for writing (output)
    selector = _make_matmul_selector(M, N, K, mat1.dtype, mat2.dtype, out.dtype, mat1.device, streamk=enable_streamk)
    
    if enable_streamk:
        streamk_matmul_lt(mat1, mat2, c_input, out, selector, sk_grid=sk_grid, 
                         c_row_broadcast=c_row_broadcast, c_col_broadcast=c_col_broadcast,
                         c_scalar=c_scalar, alpha=alpha, beta=beta)
    else:
        persistent_matmul_lt(mat1, mat2, c_input, out, selector,
                            c_row_broadcast=c_row_broadcast, c_col_broadcast=c_col_broadcast,
                            c_scalar=c_scalar, alpha=alpha, beta=beta)
    
    return out

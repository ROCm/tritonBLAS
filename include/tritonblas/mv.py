import torch
import triton
from .kernels.gemv import gemv_kernel_m1, gemv_kernel_n1, gemv_kernel_k1, gemv_kernel_general


def gemv(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    transpose: bool = False,
    block_size_m: int = 32,
    block_size_n: int = 32,
) -> torch.Tensor:
    """
    Matrix-vector multiplication: y = A @ x or y = A.T @ x
    
    Args:
        A: Input matrix of shape (M, N)
        x: Input vector of shape (N,) for normal mode or (M,) for transpose mode
        y: Output vector (required). Must be preallocated.
        transpose: If True, compute A.T @ x instead of A @ x
        block_size_m: Block size for M dimension
        block_size_n: Block size for N dimension
    
    Returns:
        Output vector y
    """
    # Handle transpose by transposing the matrix
    if transpose:
        A = A.T
    
    M, N = A.shape

    # Validate input dimensions
    assert x.shape[0] == N, f"For A @ x, x must have shape ({N},), got {x.shape}"
    assert y.shape[0] == M, f"Output y must have shape ({M},), got {y.shape}"
    
    # Get strides
    stride_am, stride_an = A.stride()
    stride_x = x.stride(0)
    stride_y = y.stride(0)
    
    # Select specialized kernel based on dimensions
    if M == 1:
        # M=1: dot product case
        y.zero_()  # Required for atomic_add
        grid = (triton.cdiv(N, block_size_n),)
        gemv_kernel_m1[grid](
            A, x, y,
            N,
            stride_an, stride_x,
            BLOCK_SIZE_N=block_size_n,
        )
    elif N == 1:
        # N=1: scalar multiplication case
        grid = (triton.cdiv(M, block_size_m),)
        gemv_kernel_n1[grid](
            A, x, y,
            M,
            stride_am, stride_y,
            BLOCK_SIZE_M=block_size_m,
        )
    elif N <= 16:
        # K=1 or very small N: use specialized kernel
        grid = (triton.cdiv(M, block_size_m),)
        gemv_kernel_k1[grid](
            A, x, y,
            M, N,
            stride_am, stride_an,
            stride_x, stride_y,
            BLOCK_SIZE_M=block_size_m,
        )
    else:
        # General case
        grid = (triton.cdiv(M, block_size_m),)
        gemv_kernel_general[grid](
            A, x, y,
            M, N,
            stride_am, stride_an,
            stride_x, stride_y,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
        )
    
    return y


def mv(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor = None,
    transpose: bool = False,
) -> torch.Tensor:
    """
    Matrix-vector multiplication with automatic detection: y = A @ x or C = A @ B
    
    Handles:
    - Standard GEMV: A @ x where A is (M, N) and x is (N,)
    - Transposed GEMV: A.T @ x where A is (M, N) and x is (M,)
    - Vector @ Matrix: x @ B
    - Skinny matrices: (M, K) @ (K, 1) or (1, K) @ (K, N)
    
    Args:
        a: Input matrix/vector A of shape (M, N) or (N,) or (M, K)
        b: Input matrix/vector x of shape (N,) or (M,) or (K, N) or (K, 1)
        c: Output (optional). If None, will be allocated.
        transpose: If True, compute A.T @ x (only valid when a is 2D and b is 1D)
    
    Returns:
        Output vector/matrix c, or None if not a GEMV case
    """
    # Check if inputs are vectors
    a_is_vector = a.dim() == 1
    b_is_vector = b.dim() == 1
    
    # Case 1: Both are 1D vectors
    if a_is_vector and b_is_vector:
        raise ValueError("Both inputs are vectors. Use torch.dot for vector dot product.")
    
    # Case 2: a is 2D matrix, b is 1D vector with transpose flag
    if not a_is_vector and b_is_vector and transpose:
        # A.T @ x where A is (M, N) and x is (M,)
        M, N = a.shape
        assert b.shape[0] == M, f"For A.T @ x, x must have shape ({M},), got {b.shape}"
        # Allocate output if needed
        if c is None:
            c = torch.zeros(N, device=a.device, dtype=a.dtype)
        return gemv(a, b, c, transpose=True)
    
    # Case 3: a is 1D vector, b is 2D matrix
    if a_is_vector:
        # a is vector, b is matrix: compute a.T @ b (equivalent to b.T @ a)
        K = a.shape[0]
        _, N = b.shape
        assert b.shape[0] == K, f"Incompatible dimensions: a has shape ({K},), b has shape {b.shape}"
        # Allocate output if needed
        if c is None:
            c = torch.zeros(N, device=a.device, dtype=a.dtype)
        return gemv(b, a, c, transpose=True)
    
    # Case 4: a is 2D matrix, b is 1D vector (no transpose)
    if b_is_vector:
        # a is matrix, b is vector: compute a @ b
        M, K = a.shape
        assert K == b.shape[0], f"Incompatible dimensions: a has shape {a.shape}, b has shape ({b.shape[0]},)"
        # Allocate output if needed
        if c is None:
            c = torch.zeros(M, device=a.device, dtype=a.dtype)
        return gemv(a, b, c, transpose=False)
    
    # Case 5: Both are 2D - check for skinny matrices
    M_a, N_a = a.shape
    M_b, N_b = b.shape
    
    # Check dimensions are compatible
    if N_a != M_b:
        # Not compatible for matrix multiplication
        return None
    
    # Case 5a: Matrix @ column vector (N=1)
    if N_b == 1:
        b_squeezed = b.squeeze(1)  # (K, 1) -> (K,)
        # Allocate output if needed
        if c is None:
            c = torch.empty((M_a, 1), device=a.device, dtype=a.dtype)
        y_1d = c.squeeze(1)  # View as 1D
        gemv(a, b_squeezed, y_1d, transpose=False)
        return c
    
    # Case 5b: Row vector @ matrix (M=1)
    if M_a == 1:
        a_squeezed = a.squeeze(0)  # (1, K) -> (K,)
        # Allocate output if needed
        if c is None:
            c = torch.empty((1, N_b), device=a.device, dtype=a.dtype)
        y_1d = c.squeeze(0)  # View as 1D
        gemv(b, a_squeezed, y_1d, transpose=True)
        return c
    
    # Not a GEMV case
    return None

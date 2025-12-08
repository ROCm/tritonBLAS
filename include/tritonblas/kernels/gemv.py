import triton
import triton.language as tl
import torch


@triton.jit
def gemv_kernel_m1(
    # Pointers to matrices
    A_ptr,
    x_ptr,
    y_ptr,
    # Matrix dimensions
    N,
    # Strides
    stride_an,
    stride_x,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized GEMV kernel for M=1 case: y = A @ x where A is (1, N)
    This is a simple dot product.
    """
    pid = tl.program_id(axis=0)
    
    # Compute offsets
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load A block: shape (BLOCK_SIZE_N,)
    a_ptrs = A_ptr + offs_n * stride_an
    a_mask = offs_n < N
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    
    # Load x block: shape (BLOCK_SIZE_N,)
    x_ptrs = x_ptr + offs_n * stride_x
    x = tl.load(x_ptrs, mask=a_mask, other=0.0)
    
    # Compute partial dot product
    partial = tl.sum(a.to(tl.float32) * x.to(tl.float32))
    
    # Atomic add to single output element
    tl.atomic_add(y_ptr, partial.to(y_ptr.dtype.element_ty), sem='relaxed')


@triton.jit
def gemv_kernel_n1(
    # Pointers to matrices
    A_ptr,
    x_ptr,
    y_ptr,
    # Matrix dimensions
    M,
    # Strides
    stride_am,
    stride_y,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Optimized GEMV kernel for N=1 case: y = A @ x where A is (M, 1) and x is (1,)
    This is a simple scalar multiplication.
    """
    pid = tl.program_id(axis=0)
    
    # Compute offsets
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Load scalar x
    x_val = tl.load(x_ptr).to(tl.float32)
    
    # Load A block: shape (BLOCK_SIZE_M,)
    a_ptrs = A_ptr + offs_m * stride_am
    a_mask = offs_m < M
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    
    # Multiply by scalar
    result = a.to(tl.float32) * x_val
    
    # Store result
    y_ptrs = y_ptr + offs_m * stride_y
    tl.store(y_ptrs, result.to(y_ptr.dtype.element_ty), mask=a_mask)


@triton.jit
def gemv_kernel_k1(
    # Pointers to matrices
    A_ptr,
    x_ptr,
    y_ptr,
    # Matrix dimensions
    M,
    N,
    # Strides
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Optimized GEMV kernel for K=1 case: y = A @ x where A is (M, N) and x is (N,)
    But K=1 means this is just element-wise multiplication of A columns by x elements.
    """
    pid = tl.program_id(axis=0)
    
    # Compute offsets
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over N dimension
    for n in range(N):
        # Load A column element
        a_ptrs = A_ptr + offs_m * stride_am + n * stride_an
        a_mask = offs_m < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load x element
        x_val = tl.load(x_ptr + n * stride_x)
        
        # Accumulate
        acc += a.to(tl.float32) * x_val.to(tl.float32)
    
    # Store result
    y_ptrs = y_ptr + offs_m * stride_y
    y_mask = offs_m < M
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


@triton.jit
def gemv_kernel_general(
    # Pointers to matrices
    A_ptr,
    x_ptr,
    y_ptr,
    # Matrix dimensions
    M,
    N,
    # Strides
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    General GEMV kernel: y = A @ x
    A is of shape (M, N)
    x is of shape (N,)
    y is of shape (M,)
    
    Uses 1D grid over M dimension, loops over N sequentially.
    """
    # Get program ID
    pid_m = tl.program_id(axis=0)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over N dimension in blocks
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Compute current N offsets
        n_offs = n * BLOCK_SIZE_N + offs_n
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + n_offs[None, :] * stride_an
        a_mask = (offs_m[:, None] < M) & (n_offs[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load x block: shape (BLOCK_SIZE_N,)
        x_ptrs = x_ptr + n_offs * stride_x
        x_mask = n_offs < N
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Accumulate: sum over N dimension
        acc += tl.sum(a.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    
    # Store result
    y_ptrs = y_ptr + offs_m * stride_y
    y_mask = offs_m < M
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)

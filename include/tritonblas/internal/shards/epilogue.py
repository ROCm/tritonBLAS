"""
Epilogue functions for composable Triton GEMM kernels.
Handles post-GEMM operations like quantization scaling, bias addition, and output storage.
"""
import triton
import triton.language as tl


@triton.jit
def apply_quant_scales(acc, A_scale_ptr, B_scale_ptr, pid_m, pid_n, M, N,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Apply per-channel quantization scales to the accumulator.
    
    Args:
        acc: Accumulator tensor to scale
        A_scale_ptr: Pointer to A's per-row scales
        B_scale_ptr: Pointer to B's per-column scales
        pid_m, pid_n: Tile coordinates
        M, N: Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
    
    Returns:
        Scaled accumulator
    """
    rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
    rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
    a_s = tl.load(A_scale_ptr + rm_s)
    b_s = tl.load(B_scale_ptr + rn_s)
    return acc * (a_s[:, None] * b_s[None, :])


@triton.jit
def add_bias(acc_like, bias_ptr, rm, M, stride_bias, QUANTIZED: tl.constexpr):
    """
    Add bias to the accumulator.
    
    Args:
        acc_like: Accumulator-like tensor to add bias to
        bias_ptr: Pointer to bias vector (or None/0 if no bias)
        rm: Row indices for this tile
        M: Matrix M dimension
        stride_bias: Stride for bias vector
        QUANTIZED: Whether in quantized mode (affects type handling)
    
    Returns:
        Accumulator with bias added
    """
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
        if QUANTIZED:
            acc_like = acc_like + bias[:, None].to(tl.float32)
        else:
            acc_like = acc_like + bias[:, None]
    return acc_like


@triton.jit
def store_tile(C, c, rm, rn, M, N, stride_cm, stride_cn,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Store a tile to the output matrix C.
    
    Args:
        C: Output matrix pointer
        c: Tile data to store
        rm: Row indices for this tile (already formatted with max_contiguous/multiple_of)
        rn: Column indices for this tile (already formatted with max_contiguous/multiple_of)
        M, N: Matrix dimensions
        stride_cm, stride_cn: Strides for C in M and N dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
    """
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, c, mask)


@triton.jit
def epilogue_tile(
    acc, rm, rn, pid_m, pid_n,
    C, A_scale_ptr, B_scale_ptr, bias_ptr,
    M, N, stride_cm, stride_cn, stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED: tl.constexpr,
):
    """
    Complete epilogue for a single tile: apply scales, add bias, convert type, and store.
    
    Args:
        acc: Accumulated result for this tile
        rm, rn: Row and column indices for this tile
        pid_m, pid_n: Tile coordinates
        C: Output matrix pointer
        A_scale_ptr: Pointer to A's per-row scales (or None/0 if not quantized)
        B_scale_ptr: Pointer to B's per-column scales (or None/0 if not quantized)
        bias_ptr: Pointer to bias vector (or None/0 if no bias)
        M, N: Matrix dimensions
        stride_cm, stride_cn: Strides for C in M and N dimensions
        stride_bias: Stride for bias vector
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
        QUANTIZED: Whether using quantized inputs
    """
    if QUANTIZED:
        acc = apply_quant_scales(acc, A_scale_ptr, B_scale_ptr, pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = add_bias(acc, bias_ptr, rm, M, stride_bias, QUANTIZED)
    c = acc.to(C.type.element_ty)
    store_tile(C, c, rm, rn, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)

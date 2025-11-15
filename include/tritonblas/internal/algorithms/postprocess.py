"""
Postprocess phase for composable Triton GEMM kernels.
Phase 4: Final math operations - scales, bias, activation, type conversion.
"""
import triton
import triton.language as tl

from .binary import apply_scales, add_vector
from .unary import convert_dtype


@triton.jit
def postprocess(
    acc,
    A_scale_ptr, B_scale_ptr, bias_ptr,
    pid_m, pid_n, row_indices,
    M, N,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED: tl.constexpr,
    output_dtype,
):
    """
    Phase 4: Postprocess - Apply scales, bias, activation, and type conversion.
    
    This function handles loading auxiliary data (scales, bias) and applying
    binary/unary operations to the accumulator.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        A_scale_ptr: Pointer to A's per-row scales (or None/0 if not quantized)
        B_scale_ptr: Pointer to B's per-column scales (or None/0 if not quantized)
        bias_ptr: Pointer to bias vector (or None/0 if no bias)
        pid_m, pid_n: Tile coordinates
        row_indices: Row indices [BLOCK_SIZE_M]
        M, N: Matrix dimensions
        stride_bias: Stride for bias vector
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
        QUANTIZED: Whether using quantized inputs
        output_dtype: Target output data type
    
    Returns:
        Processed result ready for storage [BLOCK_SIZE_M, BLOCK_SIZE_N]
    """
    # Load and apply quantization scales if needed
    if QUANTIZED:
        # Load scales
        row_scale_indices = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
        col_scale_indices = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        a_scales = tl.load(A_scale_ptr + row_scale_indices)
        b_scales = tl.load(B_scale_ptr + col_scale_indices)
        # Apply scales (pure binary operation)
        acc = apply_scales(acc, a_scales, b_scales)
    
    # Load and add bias if provided
    if bias_ptr is not None:
        # Load bias
        bias_vector = tl.load(bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0)
        # Add bias (pure binary operation)
        acc = add_vector(acc, bias_vector, QUANTIZED)
    
    # Convert to output dtype (pure unary operation)
    result = convert_dtype(acc, output_dtype)
    
    return result

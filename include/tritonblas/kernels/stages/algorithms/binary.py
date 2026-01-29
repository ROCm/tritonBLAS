"""
Binary algorithms for composable Triton GEMM kernels.
Pure operations that work on two tiles (no memory operations).
"""
import triton
import triton.language as tl


@triton.jit
def multiply_accumulate(
    acc, a, b,
    QUANTIZED: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Binary operation: Multiply-Accumulate (GEMM core operation).
    
    Performs acc += dot(a, b) with appropriate precision.
    This is a pure math operation with no memory operations.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        a: A tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        b: B tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
        QUANTIZED: Whether using quantized inputs (affects precision)
        ALLOW_TF32: Whether to allow TF32 precision (if not quantized)
    
    Returns:
        Updated accumulator after acc += dot(a, b)
    """
    if QUANTIZED:
        # Use IEEE precision for quantized inputs
        return acc + tl.dot(a, b, input_precision="ieee")
    else:
        # Use TF32 if allowed, otherwise full precision
        return acc + tl.dot(a, b, allow_tf32=ALLOW_TF32)


@triton.jit
def apply_scales(acc, a_scales, b_scales):
    """
    Binary operation: Apply quantization scales to accumulator.
    
    Multiplies the accumulator by per-row A scales and per-column B scales.
    Pure operation on tiles - no memory operations.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        a_scales: A's per-row scales [BLOCK_SIZE_M]
        b_scales: B's per-column scales [BLOCK_SIZE_N]
    
    Returns:
        Scaled accumulator
    """
    return acc * (a_scales[:, None] * b_scales[None, :])


@triton.jit
def add_vector(acc, bias_vector, QUANTIZED: tl.constexpr):
    """
    Binary operation: Add a vector to accumulator.
    
    Adds a per-row vector to the accumulator.
    Pure operation on tiles - no memory operations.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        bias_vector: Bias vector [BLOCK_SIZE_M]
        QUANTIZED: Whether using quantized inputs (affects dtype handling)
    
    Returns:
        Accumulator with bias added
    """
    if QUANTIZED:
        return acc + bias_vector[None, :].to(tl.float32)
    else:
        return acc + bias_vector[None, :]

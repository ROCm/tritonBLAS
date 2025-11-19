"""
Unary algorithms for composable Triton GEMM kernels.
Operations that work on a single tile (e.g., scaling, bias, activation).
"""
import triton
import triton.language as tl


@triton.jit
def convert_dtype(acc, output_dtype):
    """
    Unary operation: Convert accumulator to output dtype.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        output_dtype: Target output data type
    
    Returns:
        Accumulator converted to output_dtype
    """
    return acc.to(output_dtype)


@triton.jit
def apply_activation(acc, activation: tl.constexpr = "none"):
    """
    Unary operation: Apply activation function to accumulator.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        activation: Activation function name ("none", "relu", "gelu", etc.)
    
    Returns:
        Accumulator with activation applied
    """
    if activation == "relu":
        return tl.maximum(acc, 0.0)
    elif activation == "gelu":
        # Approximate GELU
        return acc * 0.5 * (1.0 + tl.libdevice.tanh(0.797885 * (acc + 0.044715 * acc * acc * acc)))
    else:  # "none"
        return acc

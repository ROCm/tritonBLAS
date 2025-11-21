"""
Unary algorithms for composable Triton GEMM kernels.
Operations that work on a single tile (e.g., scaling, bias, type conversion).
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

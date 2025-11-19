"""
Algorithms module for composable Triton GEMM kernels.

This module provides reusable mathematical operations that can be composed
to build custom GEMM kernels.

## Binary Operations (algorithms.binary)
Operations that work on two tiles:
- multiply_accumulate: Core GEMM operation (acc += dot(a, b))

## Unary Operations (algorithms.unary)
Operations that work on a single tile (usually the accumulator):
- apply_quantization_scales: Apply per-row/column quantization scales
- add_bias: Add bias vector
- convert_dtype: Convert to output dtype
- apply_activation: Apply activation function (relu, gelu, etc.)

Example usage:
    from tritonblas.shards.algorithms.binary import multiply_accumulate
    from tritonblas.shards.algorithms.unary import add_bias, convert_dtype
    
    # In your kernel:
    acc = multiply_accumulate(acc, a, b, QUANTIZED, ALLOW_TF32)
    acc = add_bias(acc, bias_ptr, rm, M, stride_bias, QUANTIZED)
    result = convert_dtype(acc, output_dtype)
"""

from .binary import (
    multiply_accumulate,
    apply_scales,
    add_vector,
)
from .unary import (
    convert_dtype,
    apply_activation,
)
from .postprocess import postprocess
from .gemm_loop import gemm_loop

__all__ = [
    # Binary operations
    'multiply_accumulate',
    'apply_scales',
    'add_vector',
    # Unary operations
    'convert_dtype',
    'apply_activation',
    # Composition
    'postprocess',
    'gemm_loop',
]

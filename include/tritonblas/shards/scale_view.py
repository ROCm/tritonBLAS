# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
ScaleView aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class ScaleView:
    """
    Quantization scale view for int8/fp8 operations.
    
    Example usage:
        scales = ScaleView(A_scale_ptr, B_scale_ptr)
        acc = scales.apply(acc, rm, rn, M, N)
    """
    
    a_scale_ptr: tl.tensor
    b_scale_ptr: tl.tensor
    stride_a: tl.constexpr
    stride_b: tl.constexpr
    
    @triton.constexpr_function
    def __init__(self, a_scale_ptr, b_scale_ptr, stride_a=1, stride_b=1):
        self.a_scale_ptr = a_scale_ptr
        self.b_scale_ptr = b_scale_ptr
        self.stride_a = tl.constexpr(stride_a)
        self.stride_b = tl.constexpr(stride_b)
    
    @triton.jit
    def apply(self, acc, rm, rn, M, N):
        """
        Apply quantization scales to accumulator.
        
        Args:
            acc: Accumulator tensor [BLOCK_M, BLOCK_N]
            rm: Row indices [BLOCK_M]
            rn: Column indices [BLOCK_N]
            M, N: Matrix dimensions for bounds checking
        
        Returns:
            Scaled accumulator
        """
        a_scales = tl.load(self.a_scale_ptr + rm * self.stride_a, mask=rm < M, other=1.0)
        b_scales = tl.load(self.b_scale_ptr + rn * self.stride_b, mask=rn < N, other=1.0)
        acc = acc.to(tl.float32)
        acc = acc * a_scales[:, None]
        acc = acc * b_scales[None, :]
        return acc

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GemmConfig aggregate for bundling GEMM performance parameters.

Provides a single object to pass around instead of many block size parameters.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class GemmConfig:
    """
    GEMM configuration parameters.
    
    Bundles together all compile-time GEMM parameters:
    - Block sizes (M, N, K)
    - Hardware configuration (NUM_SMS, NUM_XCDS)
    - Scheduling parameters (GROUP_SIZE_M, CHUNK_SIZE)
    - Cache modifiers
    - Computation options (acc_dtype, allow_tf32, even_k, quantized)
    
    Example usage:
        config = GemmConfig(
            block_m=128, block_n=256, block_k=64,
            num_sms=NUM_SMS, num_xcds=NUM_XCDS,
            group_size_m=8, even_k=EVEN_K,
        )
        
        # Use in ScheduleContext and GemmContext
        sched = ScheduleContext(M, N, K, config)
        ctx = GemmContext(config)
    """
    
    # Block sizes
    block_m: tl.constexpr
    block_n: tl.constexpr
    block_k: tl.constexpr
    
    # Hardware config
    num_sms: tl.constexpr
    num_xcds: tl.constexpr
    
    # Scheduling
    group_size_m: tl.constexpr
    chunk_size: tl.constexpr
    
    # Cache modifiers
    cache_modifier_a: tl.constexpr
    cache_modifier_b: tl.constexpr
    
    # Computation options
    acc_dtype: tl.constexpr
    allow_tf32: tl.constexpr
    even_k: tl.constexpr
    quantized: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        block_m,
        block_n,
        block_k,
        num_sms,
        num_xcds=1,
        group_size_m=8,
        chunk_size=1,
        cache_modifier_a=".cg",
        cache_modifier_b=".cg",
        acc_dtype=tl.float32,
        allow_tf32=True,
        even_k=True,
        quantized=False,
    ):
        """
        Create a GEMM configuration.
        
        Args:
            block_m: Block size M (constexpr)
            block_n: Block size N (constexpr)
            block_k: Block size K (constexpr)
            num_sms: Number of SMs/CUs (constexpr)
            num_xcds: Number of XCDs for chiplet transform (default: 1)
            group_size_m: Group size for tile scheduling (default: 8)
            chunk_size: Chunk size for chiplet scheduling (default: 1)
            cache_modifier_a: Cache modifier for A loads (default: ".cg")
            cache_modifier_b: Cache modifier for B loads (default: ".cg")
            acc_dtype: Accumulator dtype (default: tl.float32)
            allow_tf32: Allow TF32 for matmul (default: True)
            even_k: Whether K is evenly divisible by BLOCK_K (default: True)
            quantized: Use int32 accumulation for quantized inputs (default: False)
        """
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.block_k = tl.constexpr(block_k)
        self.num_sms = tl.constexpr(num_sms)
        self.num_xcds = tl.constexpr(num_xcds)
        self.group_size_m = tl.constexpr(group_size_m)
        self.chunk_size = tl.constexpr(chunk_size)
        self.cache_modifier_a = tl.constexpr(cache_modifier_a)
        self.cache_modifier_b = tl.constexpr(cache_modifier_b)
        self.acc_dtype = tl.constexpr(acc_dtype)
        self.allow_tf32 = tl.constexpr(allow_tf32)
        self.even_k = tl.constexpr(even_k)
        self.quantized = tl.constexpr(quantized)

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GemmContext aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile
from .tensor_view import tile_ptr


@aggregate
class GemmContext:
    """
    GEMM accumulator context with all configuration options.
    
    Provides two execution modes:
    - k_step(): Single BLOCK_K iteration (one dot product)
    - k_complete(): Full K loop
    
    Example usage (k_complete - simple):
        ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K, even_k=EVEN_K)
        acc = ctx.k_complete(
            A, stride_am, stride_ak, M, K,
            B, stride_bk, stride_bn, N,
            out_tile
        )
    
    Example usage (k_step - manual loop for advanced control):
        ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K)
        acc = ctx.init_accumulator()
        
        for k_idx in range(num_k_tiles):
            acc = ctx.k_step(
                A, stride_am, stride_ak, M, K,
                B, stride_bk, stride_bn, N,
                out_tile, k_idx, acc
            )
    """
    
    block_m: tl.constexpr
    block_n: tl.constexpr
    block_k: tl.constexpr
    acc_dtype: tl.constexpr
    allow_tf32: tl.constexpr
    even_k: tl.constexpr
    quantized: tl.constexpr
    cache_modifier_a: tl.constexpr
    cache_modifier_b: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        block_m,
        block_n,
        block_k,
        acc_dtype=tl.float32,
        allow_tf32=True,
        even_k=True,
        quantized=False,
        cache_modifier_a=".cg",
        cache_modifier_b=".cg",
    ):
        """
        Create a GEMM context.
        
        Args:
            block_m: Block size M (constexpr)
            block_n: Block size N (constexpr)
            block_k: Block size K (constexpr)
            acc_dtype: Accumulator dtype (default: float32)
            allow_tf32: Allow TF32 for matmul (default: True)
            even_k: Whether K is evenly divisible by BLOCK_K (default: True)
            quantized: Use int32 accumulation for quantized inputs (default: False)
            cache_modifier_a: Cache modifier for A loads (default: ".cg")
            cache_modifier_b: Cache modifier for B loads (default: ".cg")
        """
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.block_k = tl.constexpr(block_k)
        self.acc_dtype = tl.constexpr(acc_dtype)
        self.allow_tf32 = tl.constexpr(allow_tf32)
        self.even_k = tl.constexpr(even_k)
        self.quantized = tl.constexpr(quantized)
        self.cache_modifier_a = tl.constexpr(cache_modifier_a)
        self.cache_modifier_b = tl.constexpr(cache_modifier_b)
    
    @triton.jit
    def init_accumulator(self):
        """
        Initialize and return a zero accumulator.
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N] initialized to zeros
        
        Example:
            acc = ctx.init_accumulator()
        """
        return tl.zeros((self.block_m, self.block_n), dtype=self.acc_dtype)
    
    @triton.jit
    def k_step(
        self,
        # Matrix A parameters [M, K]
        A,
        stride_am,
        stride_ak,
        M,
        K,
        # Matrix B parameters [K, N]
        B,
        stride_bk,
        stride_bn,
        N,
        # Tile info
        out_tile: Tile,
        k_idx,
        # Accumulator
        acc,
        # Whether this is a boundary iteration (needs masking)
        boundary: tl.constexpr = False,
    ):
        """
        Execute a single K step (one BLOCK_K iteration).
        
        Args:
            A: Pointer to matrix A [M, K]
            stride_am: Stride for A in M dimension (runtime)
            stride_ak: Stride for A in K dimension (constexpr)
            M: Number of rows in A
            K: Number of columns in A / rows in B
            B: Pointer to matrix B [K, N]
            stride_bk: Stride for B in K dimension (constexpr)
            stride_bn: Stride for B in N dimension (runtime)
            N: Number of columns in B
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
            k_idx: Current K tile index
            acc: Accumulator to add to
            boundary: Whether this is a boundary iteration needing masking
        
        Returns:
            Updated accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            acc = ctx.init_accumulator()
            for k_idx in range(num_k_tiles):
                acc = ctx.k_step(A, stride_am, stride_ak, M, K,
                                 B, stride_bk, stride_bn, N,
                                 out_tile, k_idx, acc)
        """
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n
        
        # Row and column indices for output tile
        rm = pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = pid_n * self.block_n + tl.arange(0, self.block_n)
        rm = tl.max_contiguous(tl.multiple_of(rm % M, self.block_m), self.block_m)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, self.block_n), self.block_n)
        
        # K indices for this iteration
        rk = k_idx * self.block_k + tl.arange(0, self.block_k)
        
        # Compute pointers for A [M, K] and B [K, N]
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        
        # Load tiles
        if boundary:
            # For K boundary, only mask the K dimension
            a = tl.load(a_ptrs, mask=rk[None, :] < K, other=0.0, cache_modifier=self.cache_modifier_a)
            b = tl.load(b_ptrs, mask=rk[:, None] < K, other=0.0, cache_modifier=self.cache_modifier_b)
        else:
            a = tl.load(a_ptrs, cache_modifier=self.cache_modifier_a)
            b = tl.load(b_ptrs, cache_modifier=self.cache_modifier_b)
        
        # Accumulate
        if self.quantized:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=self.allow_tf32)
        
        return acc
    
    @triton.jit
    def k_complete(
        self,
        # Matrix A parameters [M, K]
        # A,
        # stride_am,
        # stride_ak,
        # M,
        # K,
        tensorA,
        tensorB,
        # Matrix B parameters [K, N]
        # B,
        # stride_bk,
        # stride_bn,
        # N,
        # Output tile
        out_tile: Tile,
    ):
        """
        Execute the full GEMM K loop and return the accumulator.
        
        Args:
            A: Pointer to matrix A [M, K]
            stride_am: Stride for A in M dimension (runtime)
            stride_ak: Stride for A in K dimension (constexpr)
            M: Number of rows in A
            K: Number of columns in A / rows in B
            B: Pointer to matrix B [K, N]
            stride_bk: Stride for B in K dimension (constexpr)
            stride_bn: Stride for B in N dimension (runtime)
            N: Number of columns in B
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K, even_k=True)
            acc = ctx.k_complete(A, stride_am, stride_ak, M, K,
                                 B, stride_bk, stride_bn, N, out_tile)
        """
        #Unwrap tensor descriptor
        #TODO just call the descriptor when we use it.
        A = tensorA.ptr
        stride_am = tensorA.stride_am
        stride_ak = tensorA.stride_ak
        M = tensorA.M
        K = tensorA.K
        B = tensorB.ptr
        stride_bk = tensorB.stride_bk
        stride_bn = tensorB.stride_bn
        N = tensorB.N
        
        # Initialize accumulator
        acc = self.init_accumulator()
        
        # Compute K loop bounds
        num_k_tiles = tl.cdiv(K, self.block_k)
        if not self.even_k:
            num_k_tiles -= 1
        tl.assume(num_k_tiles > 0)
        
        # Main K loop
        for k_idx in range(num_k_tiles):
            acc = self.k_step(
                A, stride_am, stride_ak, M, K,
                B, stride_bk, stride_bn, N,
                out_tile, k_idx, acc,
                boundary=False,
            )
        
        # Handle K tail if needed
        if not self.even_k:
            k_idx = num_k_tiles
            acc = self.k_step(
                A, stride_am, stride_ak, M, K,
                B, stride_bk, stride_bn, N,
                out_tile, k_idx, acc,
                boundary=True,
            )
        
        return acc

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GemmContext aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile
from .tensor_view import TensorView


@aggregate
class GemmContext:
    """
    GEMM accumulator context.
    
    Example usage:
        ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K)
        
        # Manual control:
        for k in range(0, K, BLOCK_K):
            a_tile = Tile(pid_m, k // BLOCK_K, BLOCK_M, BLOCK_K)
            b_tile = Tile(k // BLOCK_K, pid_n, BLOCK_K, BLOCK_N)
            
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            b_ptrs, b_mask = B_view.tile_ptr(b_tile)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            ctx.dot_accumulate(a, b)
        
        acc = ctx.get_accumulator()
        
        # OR one-liner:
        acc = ctx.execute(A_view, B_view, out_tile)
    """
    
    block_m: tl.constexpr
    block_n: tl.constexpr
    block_k: tl.constexpr
    acc_dtype: tl.constexpr
    allow_tf32: tl.constexpr
    acc: tl.tensor  # Accumulator [BLOCK_M, BLOCK_N]
    
    @triton.constexpr_function
    def __init__(
        self,
        block_m,
        block_n,
        block_k,
        acc_dtype=tl.float32,
        allow_tf32=True,
    ):
        """
        Create a GEMM context.
        
        Args:
            block_m: Block size M (constexpr)
            block_n: Block size N (constexpr)
            block_k: Block size K (constexpr)
            acc_dtype: Accumulator dtype (default: float32)
            allow_tf32: Allow TF32 for matmul (default: True)
        """
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.block_k = tl.constexpr(block_k)
        self.acc_dtype = tl.constexpr(acc_dtype)
        self.allow_tf32 = tl.constexpr(allow_tf32)
        self.acc = tl.zeros((block_m, block_n), dtype=acc_dtype)
    
    @triton.jit
    def dot_accumulate(self, a, b):
        """
        Accumulate a @ b into the context.
        
        Args:
            a: Input tile A [BLOCK_M, BLOCK_K]
            b: Input tile B [BLOCK_K, BLOCK_N]
        """
        self.acc += tl.dot(a, b, allow_tf32=self.allow_tf32)
    
    @triton.jit
    def dot_accumulate_int(self, a, b):
        """
        Accumulate a @ b for integer inputs (quantized).
        
        Args:
            a: Input tile A [BLOCK_M, BLOCK_K]
            b: Input tile B [BLOCK_K, BLOCK_N]
        """
        self.acc += tl.dot(a, b, out_dtype=tl.int32)
    
    @triton.jit
    def get_accumulator(self):
        """
        Get the accumulated result.
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        """
        return self.acc
    
    @triton.jit
    def execute(
        self,
        A_view: TensorView,
        B_view: TensorView,
        out_tile: Tile,
        EVEN_K: tl.constexpr = True,
    ):
        """
        Execute the full GEMM loop and return the accumulator.
        
        Args:
            A_view: TensorView for matrix A [M, K] - K is dim_minor
            B_view: TensorView for matrix B [K, N] - K is dim_major
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
            EVEN_K: Whether K is evenly divisible by BLOCK_K
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            ctx = GemmContext(BLOCK_M, BLOCK_N, BLOCK_K)
            acc = ctx.execute(A_view, B_view, out_tile)
        """
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n
        
        # K is in A_view.dim_minor (A is [M, K])
        K = A_view.dim_minor
        
        # Compute K loop bounds
        num_k_tiles = tl.cdiv(K, self.block_k)
        if not EVEN_K:
            num_k_tiles -= 1
        tl.assume(num_k_tiles > 0)
        
        # Main K loop
        for k_idx in range(num_k_tiles):
            # Input tiles at k offset
            a_tile = Tile(pid_m, k_idx, self.block_m, self.block_k)
            b_tile = Tile(k_idx, pid_n, self.block_k, self.block_n)
            
            # Get pointers and masks
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            b_ptrs, b_mask = B_view.tile_ptr(b_tile)
            
            # Load tiles
            a = tl.load(a_ptrs, cache_modifier=".cg")
            b = tl.load(b_ptrs, cache_modifier=".cg")
            
            # Accumulate
            self.dot_accumulate(a, b)
        
        # Handle K tail if needed
        if not EVEN_K:
            k_idx = num_k_tiles
            a_tile = Tile(pid_m, k_idx, self.block_m, self.block_k)
            b_tile = Tile(k_idx, pid_n, self.block_k, self.block_n)
            
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            b_ptrs, b_mask = B_view.tile_ptr(b_tile)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            self.dot_accumulate(a, b)
        
        return self.acc

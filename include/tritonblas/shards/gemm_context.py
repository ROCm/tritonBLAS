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
from .matrix_view import InputView
from .gemm_config import GemmConfig


@aggregate
class GemmContext:
    """
    GEMM accumulator context with all configuration options.
    
    Provides two execution modes:
    - k_step(): Single BLOCK_K iteration (one dot product)
    - k_complete(): Full K loop
    
    Example usage (k_complete - simple):
        # InputView takes: (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
        # For A [M, K]: K is reduction, M is free
        # For B [K, N]: K is reduction, N is free
        tensorA = InputView(A, stride_ak, stride_am, M, K)
        tensorB = InputView(B, stride_bk, stride_bn, K, N)
        
        config = GemmConfig(block_m=128, block_n=256, block_k=64, num_sms=NUM_SMS, even_k=EVEN_K)
        ctx = GemmContext(config)
        acc = ctx.k_complete(tensorA, tensorB, out_tile)
    
    Example usage (k_step - manual loop for advanced control):
        tensorA = InputView(A, stride_ak, stride_am, M, K)
        tensorB = InputView(B, stride_bk, stride_bn, K, N)
        
        config = GemmConfig(block_m=128, block_n=256, block_k=64, num_sms=NUM_SMS)
        ctx = GemmContext(config)
        acc = ctx.init_accumulator()
        
        for k_idx in range(num_k_tiles):
            acc = ctx.k_step(tensorA, tensorB, out_tile, k_idx, acc)
    """
    
    # Store the config (contains all parameters)
    config: GemmConfig
    
    @triton.constexpr_function
    def __init__(self, config: GemmConfig):
        """
        Create a GEMM context from a GemmConfig.
        
        All parameters come from the config:
        - block_m, block_n, block_k
        - cache_modifier_a, cache_modifier_b
        - acc_dtype, allow_tf32, even_k, quantized
        
        Args:
            config: GemmConfig with all GEMM parameters
        """
        self.config = config
    
    @triton.jit
    def init_accumulator(self):
        """
        Initialize and return a zero accumulator.
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N] initialized to zeros
        
        Example:
            acc = ctx.init_accumulator()
        """
        return tl.zeros((self.config.block_m, self.config.block_n), dtype=self.config.acc_dtype)
    
    @triton.jit
    def k_step(
        self,
        A: InputView,
        B: InputView,
        out_tile: Tile,
        k_idx,
        acc,
        boundary: tl.constexpr = False,
    ):
        """
        Execute a single K step (one BLOCK_K iteration).
        
        Args:
            A: InputView with (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
               For A [M, K]: rows=M, cols=K, stride_reduction_dim=stride_ak, stride_free_dim=stride_am
            B: InputView with (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
               For B [K, N]: rows=K, cols=N, stride_reduction_dim=stride_bk, stride_free_dim=stride_bn
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
            k_idx: Current K tile index
            acc: Accumulator to add to
            boundary: Whether this is a boundary iteration needing masking
        
        Returns:
            Updated accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            A = InputView(A_ptr, stride_ak, stride_am, M, K)
            B = InputView(B_ptr, stride_bk, stride_bn, K, N)
            acc = ctx.init_accumulator()
            for k_idx in range(num_k_tiles):
                acc = ctx.k_step(A, B, out_tile, k_idx, acc)
        """
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n
        
        # Create tiles for A and B at this K iteration
        # A [M, K]: tile at (pid_m, k_idx) with shape (BLOCK_M, BLOCK_K)
        # B [K, N]: tile at (k_idx, pid_n) with shape (BLOCK_K, BLOCK_N)
        a_tile = Tile(pid_m, k_idx, self.config.block_m, self.config.block_k)
        b_tile = Tile(k_idx, pid_n, self.config.block_k, self.config.block_n)
        
        # Get pointers using tile_ptrs with transpose flag
        # A: transpose=False (row=free, col=reduction)
        # B: transpose=True (row=reduction, col=free)
        a_ptrs, a_mask = A.tile_ptrs(a_tile, transpose=False)
        b_ptrs, b_mask = B.tile_ptrs(b_tile, transpose=True)
        
        # Load tiles
        if boundary:
            # Use masks for K boundary
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=self.config.cache_modifier_a)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=self.config.cache_modifier_b)
        else:
            a = tl.load(a_ptrs, cache_modifier=self.config.cache_modifier_a)
            b = tl.load(b_ptrs, cache_modifier=self.config.cache_modifier_b)
        
        # Accumulate
        if self.config.quantized:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=self.config.allow_tf32)
        
        return acc
    
    @triton.jit
    def k_complete(
        self,
        A: InputView,
        B: InputView,
        out_tile: Tile,
    ):
        """
        Execute the full GEMM K loop and return the accumulator.
        
        Args:
            A: InputView with (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
               For A [M, K]: rows=M, cols=K, stride_reduction_dim=stride_ak, stride_free_dim=stride_am
            B: InputView with (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
               For B [K, N]: rows=K, cols=N, stride_reduction_dim=stride_bk, stride_free_dim=stride_bn
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            A = InputView(A_ptr, stride_ak, stride_am, M, K)
            B = InputView(B_ptr, stride_bk, stride_bn, K, N)
            config = GemmConfig(block_m=128, block_n=256, block_k=64, num_sms=NUM_SMS, even_k=True)
            ctx = GemmContext(config)
            acc = ctx.k_complete(A, B, out_tile)
        """
        # Initialize accumulator
        acc = self.init_accumulator()
        
        # Compute K loop bounds (K dimension is A.cols or B.rows)
        num_k_tiles = tl.cdiv(A.cols, self.config.block_k)
        if not self.config.even_k:
            num_k_tiles -= 1
        tl.assume(num_k_tiles > 0)
        
        # Main K loop
        for k_idx in range(num_k_tiles):
            acc = self.k_step(A, B, out_tile, k_idx, acc, boundary=False)
        
        # Handle K tail if needed
        if not self.config.even_k:
            k_idx = num_k_tiles
            acc = self.k_step(A, B, out_tile, k_idx, acc, boundary=True)
        
        return acc

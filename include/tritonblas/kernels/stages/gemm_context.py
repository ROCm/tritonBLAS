# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
GemmContext aggregate for tritonblas shards.

Provides the K-loop execution context for GEMM operations, managing
the accumulator and iteration over the reduction dimension.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile
from .matrix_view import InputView
from .gemm_config import GemmConfig


@aggregate
class GemmContext:
    """
    GEMM accumulator context with all configuration options.
    
    Provides two execution modes:
    - k_step(): Single BLOCK_K iteration (one dot product)
    - k_complete(): Full K loop
    
    TILE CREATION CONVENTION:
    -------------------------
    For A [M, K] and B [K, N]:
    - A tiles: (pid_m, k_idx) with shape (BLOCK_M, BLOCK_K)
    - B tiles: (k_idx, pid_n) with shape (BLOCK_K, BLOCK_N)
    
    The InputView handles the pointer arithmetic based on its stored layout.
    
    Example usage (k_complete - simple):
        tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
        
        config = GemmConfig(block_m=128, block_n=256, block_k=64, ...)
        ctx = GemmContext(config)
        acc = ctx.k_complete(tensorA, tensorB, out_tile)
    
    Example usage (k_step - manual loop for advanced control):
        tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
        
        config = GemmConfig(...)
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
        
        Creates tiles for A and B at the given K index and loads them using
        the InputView's tile_ptrs method (which handles layout internally).
        
        Args:
            A: InputView for matrix A [M, K] with strides already stored
            B: InputView for matrix B [K, N] with strides already stored
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
            k_idx: Current K tile index
            acc: Accumulator to add to
            boundary: Whether this is a boundary iteration needing masking
        
        Returns:
            Updated accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            A = make_tensor_view(A_ptr, M, K, stride_am, stride_ak)
            B = make_tensor_view(B_ptr, K, N, stride_bk, stride_bn)
            acc = ctx.init_accumulator()
            for k_idx in range(num_k_tiles):
                acc = ctx.k_step(A, B, out_tile, k_idx, acc)
        """
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n
        
        # ═══════════════════════════════════════════════════════════════════
        # TILE CREATION: Create tiles for A and B at this K iteration
        # ═══════════════════════════════════════════════════════════════════
        # A [M, K]: tile at (pid_m, k_idx) with shape (BLOCK_M, BLOCK_K)
        # B [K, N]: tile at (k_idx, pid_n) with shape (BLOCK_K, BLOCK_N)
        a_tile = Tile(pid_m, k_idx, self.config.block_m, self.config.block_k)
        b_tile = Tile(k_idx, pid_n, self.config.block_k, self.config.block_n)
        
        # ═══════════════════════════════════════════════════════════════════
        # POINTER COMPUTATION: InputView handles layout internally
        # ═══════════════════════════════════════════════════════════════════
        # The InputView's tile_ptrs method uses its stored stride_row and
        # stride_col to compute correct pointers for any layout.
        a_ptrs, a_mask = A.tile_ptrs(a_tile)
        b_ptrs, b_mask = B.tile_ptrs(b_tile)
        
        # ═══════════════════════════════════════════════════════════════════
        # LOAD TILES
        # ═══════════════════════════════════════════════════════════════════
        if boundary:
            # Use masks for K boundary
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=self.config.cache_modifier_a)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=self.config.cache_modifier_b)
        else:
            a = tl.load(a_ptrs, cache_modifier=self.config.cache_modifier_a)
            b = tl.load(b_ptrs, cache_modifier=self.config.cache_modifier_b)
        
        # ═══════════════════════════════════════════════════════════════════
        # ACCUMULATE
        # ═══════════════════════════════════════════════════════════════════
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
        
        Iterates over all K tiles, loading from A and B using their stored
        layout information, and accumulates the dot products.
        
        Args:
            A: InputView for matrix A [M, K] with strides already stored
            B: InputView for matrix B [K, N] with strides already stored
            out_tile: Output Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
        
        Returns:
            Accumulator tensor [BLOCK_M, BLOCK_N]
        
        Example:
            A = make_tensor_view(A_ptr, M, K, stride_am, stride_ak)
            B = make_tensor_view(B_ptr, K, N, stride_bk, stride_bn)
            config = GemmConfig(block_m=128, block_n=256, block_k=64, ...)
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

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class Tile:
    """
    2D tile with coordinates and shape.
    
    Stores runtime coordinates (pid_m, pid_n) and compile-time block sizes.
    
    Example usage:
        # Output tile
        out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
        
        # Input A tile at k offset
        a_tile = Tile(pid_m, k // BLOCK_K, BLOCK_M, BLOCK_K)
        
        # Input B tile at k offset  
        b_tile = Tile(k // BLOCK_K, pid_n, BLOCK_K, BLOCK_N)
    """
    
    pid_m: tl.tensor  # Tile coordinate in M dimension
    pid_n: tl.tensor  # Tile coordinate in N dimension
    block_m: tl.constexpr  # Block size M
    block_n: tl.constexpr  # Block size N
    
    @triton.constexpr_function
    def __init__(self, pid_m, pid_n, block_m, block_n):
        """
        Create a tile with runtime coordinates and compile-time sizes.
        
        Args:
            pid_m: Tile coordinate in M dimension
            pid_n: Tile coordinate in N dimension
            block_m: Block size in M dimension (constexpr)
            block_n: Block size in N dimension (constexpr)
        """
        self.pid_m = pid_m
        self.pid_n = pid_n
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
    
    @triton.jit
    def indices(self):
        """
        Compute row and column indices for this tile.
        
        Returns:
            rm, rn: Row indices [BLOCK_M], column indices [BLOCK_N]
        """
        rm = self.pid_m * self.block_m + tl.arange(0, self.block_m)
        rn = self.pid_n * self.block_n + tl.arange(0, self.block_n)
        return rm, rn
    
    @triton.jit
    def layout(self, M, N):
        """
        Compute memory layout with bounds checking.
        
        Args:
            M: Total rows
            N: Total columns
        
        Returns:
            rm, rn, mask: Row indices, column indices, bounds mask
        """
        rm, rn = self.indices()
        rm = tl.max_contiguous(tl.multiple_of(rm % M, self.block_m), self.block_m)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, self.block_n), self.block_n)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        return rm, rn, mask

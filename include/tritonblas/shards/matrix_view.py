# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Matrix view aggregates for tritonblas shards.

Provides typed matrix view classes with load/store methods:
- InputView: Generic input matrix with load() method
- OutputView: Generic output matrix with store() method

Usage:
    A = InputView(A_ptr, stride_am, stride_ak, M, K)  # A [M, K]
    B = InputView(B_ptr, stride_bk, stride_bn, K, N)  # B [K, N]
    C = OutputView(C_ptr, stride_cm, stride_cn, M, N) # C [M, N]
    
    a_tile = Tile(pid_m, k_idx, BLOCK_M, BLOCK_K)
    b_tile = Tile(k_idx, pid_n, BLOCK_K, BLOCK_N)
    out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
    
    a = A.load(a_tile)
    b = B.load(b_tile)
    C.store(result, out_tile)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile


@aggregate
class InputView:
    """
    Generic input tensor descriptor with load() method.
    
    Can be used for any 2D input matrix (A, B, etc.).
    Uses constexpr stride_row for vectorization efficiency.
    
    Fields:
    - ptr: Pointer to matrix data
    - stride_row: Stride in row dimension (constexpr for vectorization)
    - stride_col: Stride in column dimension (runtime tensor)
    - rows, cols: Matrix dimensions
    
    Example:
        A = InputView(A_ptr, stride_am, stride_ak, M, K)
        B = InputView(B_ptr, stride_bk, stride_bn, K, N)
        
        a_tile = Tile(pid_m, k_idx, BLOCK_M, BLOCK_K)
        data = A.load(a_tile)
    """
    ptr: tl.tensor
    stride_reduction_dim: tl.constexpr
    stride_free_dim: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, stride_reduction_dim, stride_free_dim, rows, cols):
        """
        Create InputView.
        
        Args:
            ptr: Base pointer to matrix
            stride_reduction_dim: Stride in reduction dimension (constexpr)
            stride_free_dim: Stride in free dimension (runtime)
            rows: Number of rows
            cols: Number of columns
        """
        self.ptr = ptr
        self.stride_reduction_dim = tl.constexpr(stride_reduction_dim)
        self.stride_free_dim = stride_free_dim
        self.rows = rows
        self.cols = cols
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Get pointer array and mask for a tile.
        
        Args:
            tile: Tile with (pid_row, pid_col, BLOCK_ROW, BLOCK_COL)
        
        Returns:
            ptrs, mask: Pointer array and bounds mask
        """
        r_row, r_col, mask = tile.layout(self.rows, self.cols)
        ptrs = self.ptr + r_row[:, None] * self.stride_row + r_col[None, :] * self.stride_col
        return ptrs, mask
    
    @triton.jit
    def load(self, tile: Tile, boundary: tl.constexpr = False, cache_modifier: tl.constexpr = ".cg"):
        """
        Load a tile from this matrix.
        
        Args:
            tile: Tile with coordinates and shape
            boundary: Whether to apply boundary masking
            cache_modifier: Cache modifier for load
        
        Returns:
            Loaded tile data
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


@aggregate
class OutputView:
    """
    Generic output tensor descriptor with store() method.
    
    Can be used for any 2D output matrix.
    Uses runtime strides for both dimensions (supports any layout).
    
    Fields:
    - ptr: Pointer to matrix data
    - stride_row: Stride in row dimension (runtime tensor)
    - stride_col: Stride in column dimension (runtime tensor)
    - rows, cols: Matrix dimensions
    
    Example:
        C = OutputView(C_ptr, stride_cm, stride_cn, M, N)
        out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
        C.store(result, out_tile)
    """
    ptr: tl.tensor
    stride_row: tl.tensor
    stride_col: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, stride_row, stride_col, rows, cols):
        """
        Create OutputView.
        
        Args:
            ptr: Base pointer to matrix
            stride_row: Stride in row dimension (runtime)
            stride_col: Stride in column dimension (runtime)
            rows: Number of rows
            cols: Number of columns
        """
        self.ptr = ptr
        self.stride_row = stride_row
        self.stride_col = stride_col
        self.rows = rows
        self.cols = cols
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Get pointer array and mask for a tile.
        
        Args:
            tile: Tile with (pid_row, pid_col, BLOCK_ROW, BLOCK_COL)
        
        Returns:
            ptrs, mask: Pointer array and bounds mask
        """
        r_row, r_col, mask = tile.layout(self.rows, self.cols)
        ptrs = self.ptr + r_row[:, None] * self.stride_row + r_col[None, :] * self.stride_col
        return ptrs, mask
    
    @triton.jit
    def store(self, data, tile: Tile, mask=None):
        """
        Store data to a tile in this matrix.
        
        Args:
            data: Data to store
            tile: Tile with coordinates and shape
            mask: Optional mask (if None, computes from bounds)
        """
        ptrs, bounds_mask = self.tile_ptrs(tile)
        if mask is None:
            mask = bounds_mask
        tl.store(ptrs, data, mask=mask)
    
    @triton.jit
    def load(self, tile: Tile, boundary: tl.constexpr = False, cache_modifier: tl.constexpr = ".cg"):
        """
        Load a tile from this matrix (for read-modify-write patterns).
        
        Args:
            tile: Tile with coordinates and shape
            boundary: Whether to apply boundary masking
            cache_modifier: Cache modifier for load
        
        Returns:
            Loaded tile data
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

A_View = InputView
B_View = InputView
C_View = OutputView
MatrixView = OutputView
InputTensorA = InputView
InputTensorB = InputView

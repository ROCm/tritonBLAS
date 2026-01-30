# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Matrix view aggregates for tritonblas shards.

Provides InputView and OutputView aggregates that encapsulate matrix pointers,
dimensions, and strides. The kernel writer just describes their matrix and
uses it - no need to worry about layout flags or transpose.

Usage:
    # A [M, K] with strides stride_am, stride_ak
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    
    # B [K, N] with strides stride_bk, stride_bn
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    
    # C [M, N] with strides stride_cm, stride_cn
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # Use them
    acc = ctx.k_complete(tensorA, tensorB, out_tile)
    tensorC.store(result, out_tile)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile


@aggregate
class InputView:
    """
    Input matrix view for GEMM.
    
    Stores the matrix pointer, dimensions, and both strides.
    The tile_ptrs() method computes pointers using the general formula
    that works for any memory layout.
    
    Fields:
        ptr: Base pointer to matrix data
        rows: Number of rows
        cols: Number of columns
        stride_row: Stride when moving along rows (first dimension)
        stride_col: Stride when moving along columns (second dimension)
    """
    ptr: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    stride_row: tl.tensor
    stride_col: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, rows, cols, stride_row, stride_col):
        self.ptr = ptr
        self.rows = rows
        self.cols = cols
        self.stride_row = stride_row
        self.stride_col = stride_col
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Compute pointer array and bounds mask for a tile.
        
        Uses the general formula: ptr[i,j] = base + i*stride_row + j*stride_col
        This works for any memory layout (row-major, col-major, or other).
        
        Args:
            tile: Tile object with (pid_row, pid_col, block_row, block_col)
            
        Returns:
            ptrs: 2D pointer array [BLOCK_ROW, BLOCK_COL]
            mask: 2D boolean mask for boundary handling
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
            boundary: If True, apply boundary masking for partial tiles
            cache_modifier: Cache modifier for load instruction
        
        Returns:
            Loaded tile data [BLOCK_ROW, BLOCK_COL]
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


@aggregate
class OutputView:
    """
    Output matrix view for GEMM.
    
    Same design as InputView - stores pointer, dimensions, and strides.
    Provides store() in addition to load() for writing results.
    """
    ptr: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    stride_row: tl.tensor
    stride_col: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, rows, cols, stride_row, stride_col):
        self.ptr = ptr
        self.rows = rows
        self.cols = cols
        self.stride_row = stride_row
        self.stride_col = stride_col
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Compute pointer array and bounds mask for a tile.
        """
        r_row, r_col, mask = tile.layout(self.rows, self.cols)
        ptrs = self.ptr + r_row[:, None] * self.stride_row + r_col[None, :] * self.stride_col
        return ptrs, mask
    
    @triton.jit
    def store(self, data, tile: Tile, mask=None):
        """
        Store data to a tile in this matrix.
        
        Args:
            data: Data to store [BLOCK_ROW, BLOCK_COL]
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
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


# =============================================================================
# Factory Functions
# =============================================================================

@triton.jit
def make_input_view(ptr, rows, cols, stride_row, stride_col) -> InputView:
    """
    Create an InputView with automatic stride type coercion.
    
    This factory ensures strides are always tensor-typed, handling the case 
    where contiguous dimensions have stride=1 (Python int) while other 
    dimensions have tensor-typed strides.
    
    Args:
        ptr: Base pointer to matrix data
        rows: Number of rows (first dimension) - must be a tensor
        cols: Number of columns (second dimension)
        stride_row: Stride when moving along rows
        stride_col: Stride when moving along columns
    
    Returns:
        InputView with all fields as tensors
    
    Example:
        # A is [M, K] matrix - strides can be int or tensor
        tensorA = make_input_view(A, M, K, stride_am, stride_ak)
        
        # B is [K, N] matrix
        tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # TYPE PROMOTION TRICK
    # ═══════════════════════════════════════════════════════════════════════
    # Triton aggregates require strongly-typed fields (tl.tensor). However,
    # strides can be either Python ints (stride=1 for contiguous dimensions)
    # or Triton tensors (stride>1 from kernel params).
    #
    # The pattern `stride + 0 * rows` promotes any int to a tensor:
    #   - 0 * rows produces a tensor with value 0 (since rows is a tensor)
    #   - stride + (tensor 0) = tensor with stride's value
    #
    # This has ZERO runtime cost - the compiler constant-folds 0*x and x+0.
    # ═══════════════════════════════════════════════════════════════════════
    stride_row_t = stride_row + 0 * rows
    stride_col_t = stride_col + 0 * rows
    
    return InputView(ptr, rows, cols, stride_row_t, stride_col_t)


@triton.jit
def make_output_view(ptr, rows, cols, stride_row, stride_col) -> OutputView:
    """
    Create an OutputView with automatic stride type coercion.
    
    Same as make_input_view() but returns an OutputView which has
    store() method in addition to load().
    
    Args:
        ptr: Base pointer to matrix data
        rows: Number of rows (first dimension) - must be a tensor
        cols: Number of columns (second dimension)
        stride_row: Stride when moving along rows
        stride_col: Stride when moving along columns
    
    Returns:
        OutputView with all fields as tensors
    
    Example:
        # C is [M, N] output matrix
        tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # TYPE PROMOTION TRICK - See make_input_view() for detailed explanation
    # ═══════════════════════════════════════════════════════════════════════
    stride_row_t = stride_row + 0 * rows
    stride_col_t = stride_col + 0 * rows
    
    return OutputView(ptr, rows, cols, stride_row_t, stride_col_t)


# Alias for backward compatibility
make_tensor_view = make_input_view


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

A_View = InputView
B_View = InputView
C_View = OutputView
MatrixView = OutputView
InputTensorA = InputView
InputTensorB = InputView

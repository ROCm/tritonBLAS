# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
TensorView aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile


@aggregate
class TensorView:
    """
    Tensor view with tile_ptr() method returning pointers and mask.
    
    Example usage:
        A_view = TensorView(A, stride_am, stride_ak, M, K)
        
        a_tile = Tile(pid_m, k // BLOCK_K, BLOCK_M, BLOCK_K)
        a_ptrs, a_mask = A_view.tile_ptr(a_tile)
        
        # User does the load
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    """
    
    ptr: tl.tensor
    stride_major: tl.constexpr
    stride_minor: tl.constexpr
    dim_major: tl.tensor  # Total size in major dimension (for bounds)
    dim_minor: tl.tensor  # Total size in minor dimension (for bounds)
    
    @triton.constexpr_function
    def __init__(self, ptr, stride_major, stride_minor, dim_major, dim_minor):
        """
        Create a tensor view.
        
        Args:
            ptr: Pointer to tensor data
            stride_major: Stride in major (row) dimension
            stride_minor: Stride in minor (column) dimension
            dim_major: Total size in major dimension (M or K)
            dim_minor: Total size in minor dimension (K or N)
        """
        self.ptr = ptr
        self.stride_major = tl.constexpr(stride_major)
        self.stride_minor = tl.constexpr(stride_minor)
        self.dim_major = dim_major
        self.dim_minor = dim_minor
    
    @triton.jit
    def tile_ptr(self, tile: Tile):
        """
        Get pointer array and mask for a tile.
        
        Args:
            tile: Tile with coordinates and shape
        
        Returns:
            ptrs, mask: Pointer array [BLOCK_M, BLOCK_N] and bounds mask
        
        Example:
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        """
        rm, rn, mask = tile.layout(self.dim_major, self.dim_minor)
        ptrs = self.ptr + rm[:, None] * self.stride_major + rn[None, :] * self.stride_minor
        
        # Add alignment hints for better performance
        if self.stride_minor == 1:
            ptrs = tl.multiple_of(ptrs, (1, 16))
        else:
            ptrs = tl.multiple_of(ptrs, (16, 1))
        
        return ptrs, mask
    
    @triton.jit
    def tile_ptr_1d(self, tile: Tile):
        """
        Get pointer array and mask for 1D access (row vector).
        
        Args:
            tile: Tile (uses only row dimension)
        
        Returns:
            ptrs, mask: Pointer array [BLOCK_M] and bounds mask
        """
        rm, _ = tile.indices()
        rm = tl.max_contiguous(tl.multiple_of(rm % self.dim_major, tile.block_m), tile.block_m)
        mask = rm < self.dim_major
        ptrs = self.ptr + rm * self.stride_major
        return ptrs, mask

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
TensorView functions and classes for tritonblas shards.

Due to Triton's strict type system for aggregates, we provide:
1. Device functions (tile_ptr, make_tile_ptr) - Work with any stride types
2. Specialized aggregate classes for specific use cases

For GEMM kernels with mixed runtime/constexpr strides, use the device functions.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile


# =============================================================================
# Device Functions - Always work regardless of stride types
# =============================================================================


@triton.jit
def tile_ptr(
    ptr,
    stride_major,
    stride_minor,
    dim_major,
    dim_minor,
    tile: Tile,
):
    """
    Compute pointer array and mask for a tile.
    
    This device function works with ANY combination of runtime/constexpr strides.
    
    Args:
        ptr: Base pointer to tensor data
        stride_major: Stride in major (row) dimension (runtime or constexpr)
        stride_minor: Stride in minor (col) dimension (runtime or constexpr)
        dim_major: Total size in major dimension (for bounds)
        dim_minor: Total size in minor dimension (for bounds)
        tile: Tile with coordinates and shape
    
    Returns:
        ptrs, mask: Pointer array [BLOCK_M, BLOCK_N] and bounds mask
    
    Example:
        a_tile = Tile(pid_m, k_idx, BLOCK_M, BLOCK_K)
        a_ptrs, a_mask = tile_ptr(A, stride_am, stride_ak, M, K, a_tile)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    """
    rm, rn, mask = tile.layout(dim_major, dim_minor)
    ptrs = ptr + rm[:, None] * stride_major + rn[None, :] * stride_minor
    return ptrs, mask


@triton.jit
def tile_ptr_1d(
    ptr,
    stride,
    dim,
    tile: Tile,
):
    """
    Compute pointer array and mask for 1D access (row vector).
    
    Args:
        ptr: Base pointer to tensor data
        stride: Stride in the dimension
        dim: Total size in the dimension (for bounds)
        tile: Tile (uses only row dimension for indexing)
    
    Returns:
        ptrs, mask: Pointer array [BLOCK_M] and bounds mask
    """
    rm, _ = tile.indices()
    rm = tl.max_contiguous(tl.multiple_of(rm % dim, tile.block_m), tile.block_m)
    mask = rm < dim
    ptrs = ptr + rm * stride
    return ptrs, mask


# =============================================================================
# Helper function for creating tile pointers inline
# =============================================================================


@triton.jit
def make_tile_ptr(
    ptr,
    stride_major,
    stride_minor,
    dim_major,
    dim_minor,
    pid_major,
    pid_minor,
    BLOCK_MAJOR: tl.constexpr,
    BLOCK_MINOR: tl.constexpr,
):
    """
    Create tile pointer and mask directly from coordinates.
    
    This is a convenience function that creates a Tile internally.
    
    Args:
        ptr: Base pointer to tensor data
        stride_major: Stride in major dimension
        stride_minor: Stride in minor dimension
        dim_major: Total size in major dimension
        dim_minor: Total size in minor dimension
        pid_major: Tile coordinate in major dimension
        pid_minor: Tile coordinate in minor dimension
        BLOCK_MAJOR: Block size in major dimension (constexpr)
        BLOCK_MINOR: Block size in minor dimension (constexpr)
    
    Returns:
        ptrs, mask: Pointer array and bounds mask
    
    Example:
        a_ptrs, a_mask = make_tile_ptr(A, stride_am, stride_ak, M, K,
                                        pid_m, k_idx, BLOCK_M, BLOCK_K)
    """
    tile = Tile(pid_major, pid_minor, BLOCK_MAJOR, BLOCK_MINOR)
    return tile_ptr(ptr, stride_major, stride_minor, dim_major, dim_minor, tile)


# =============================================================================
# TensorView aggregate - For cases with constexpr dimensions/strides only
# =============================================================================


@aggregate
class TensorView:
    """
    Tensor view for cases where ALL dimensions and strides are constexpr.
    
    NOTE: For GEMM kernels with mixed runtime/constexpr strides, use the
    tile_ptr() device function instead.
    
    This class is useful when all parameters can be constexpr (e.g., in
    specialized kernels or when strides are known at compile time).
    
    Example usage (all constexpr):
        @triton.jit
        def kernel(ptr, M: tl.constexpr, K: tl.constexpr,
                   stride_m: tl.constexpr, stride_k: tl.constexpr, ...):
            view = TensorView(ptr, stride_m, stride_k, M, K)
            ptrs, mask = view.tile_ptr(tile)
    """
    
    ptr: tl.tensor
    stride_major: tl.constexpr
    stride_minor: tl.constexpr
    dim_major: tl.constexpr
    dim_minor: tl.constexpr
    
    @triton.constexpr_function
    def __init__(self, ptr, stride_major, stride_minor, dim_major, dim_minor):
        """
        Create a tensor view with constexpr dimensions and strides.
        
        Args:
            ptr: Pointer to tensor data
            stride_major: Stride in major dimension (must be constexpr)
            stride_minor: Stride in minor dimension (must be constexpr)
            dim_major: Total size in major dimension (must be constexpr)
            dim_minor: Total size in minor dimension (must be constexpr)
        """
        self.ptr = ptr
        self.stride_major = tl.constexpr(stride_major)
        self.stride_minor = tl.constexpr(stride_minor)
        self.dim_major = tl.constexpr(dim_major)
        self.dim_minor = tl.constexpr(dim_minor)
    
    @triton.jit
    def tile_ptr(self, tile: Tile):
        """
        Get pointer array and mask for a tile.
        
        Args:
            tile: Tile with coordinates and shape
        
        Returns:
            ptrs, mask: Pointer array [BLOCK_M, BLOCK_N] and bounds mask
        """
        rm, rn, mask = tile.layout(self.dim_major, self.dim_minor)
        ptrs = self.ptr + rm[:, None] * self.stride_major + rn[None, :] * self.stride_minor
        return ptrs, mask

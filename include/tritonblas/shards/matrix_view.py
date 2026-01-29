# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Matrix view device functions for tritonblas shards.

Due to Triton's strict type system for aggregates (requiring exact type matches),
we use device functions instead of aggregate classes for matrix views.
Device functions work with any combination of tensor/constexpr/int types.
"""

import triton
import triton.language as tl

from .tile import Tile


# =============================================================================
# Device functions for matrix tile operations
# These work with ANY stride types (tensor, constexpr, or int)
# =============================================================================


@triton.jit
def a_tile_ptrs(
    ptr,
    stride_m,
    stride_k,
    M,
    K,
    tile: Tile,
):
    """
    Get pointer array and mask for a tile of matrix A [M, K].
    
    Works with any stride types (tensor, constexpr, or int).
    
    Args:
        ptr: Base pointer to matrix A
        stride_m: Stride in M dimension (any type)
        stride_k: Stride in K dimension (any type)
        M: Number of rows
        K: Number of columns
        tile: Tile with (pid_m, pid_k, BLOCK_M, BLOCK_K)
    
    Returns:
        ptrs, mask: Pointer array [BLOCK_M, BLOCK_K] and bounds mask
    """
    rm, rk, mask = tile.layout(M, K)
    ptrs = ptr + rm[:, None] * stride_m + rk[None, :] * stride_k
    return ptrs, mask


@triton.jit
def a_load_tile(
    ptr,
    stride_m,
    stride_k,
    M,
    K,
    tile: Tile,
    boundary: tl.constexpr = False,
    cache_modifier: tl.constexpr = ".cg",
):
    """
    Load a tile from matrix A [M, K].
    
    Args:
        ptr: Base pointer to matrix A
        stride_m: Stride in M dimension
        stride_k: Stride in K dimension
        M: Number of rows
        K: Number of columns
        tile: Tile with (pid_m, pid_k, BLOCK_M, BLOCK_K)
        boundary: Whether to apply boundary masking
        cache_modifier: Cache modifier for load
    
    Returns:
        Loaded tile data [BLOCK_M, BLOCK_K]
    """
    ptrs, mask = a_tile_ptrs(ptr, stride_m, stride_k, M, K, tile)
    if boundary:
        return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
    else:
        return tl.load(ptrs, cache_modifier=cache_modifier)


@triton.jit
def b_tile_ptrs(
    ptr,
    stride_k,
    stride_n,
    K,
    N,
    tile: Tile,
):
    """
    Get pointer array and mask for a tile of matrix B [K, N].
    
    Works with any stride types (tensor, constexpr, or int).
    
    Args:
        ptr: Base pointer to matrix B
        stride_k: Stride in K dimension (any type)
        stride_n: Stride in N dimension (any type)
        K: Number of rows
        N: Number of columns
        tile: Tile with (pid_k, pid_n, BLOCK_K, BLOCK_N)
    
    Returns:
        ptrs, mask: Pointer array [BLOCK_K, BLOCK_N] and bounds mask
    """
    rk, rn, mask = tile.layout(K, N)
    ptrs = ptr + rk[:, None] * stride_k + rn[None, :] * stride_n
    return ptrs, mask


@triton.jit
def b_load_tile(
    ptr,
    stride_k,
    stride_n,
    K,
    N,
    tile: Tile,
    boundary: tl.constexpr = False,
    cache_modifier: tl.constexpr = ".cg",
):
    """
    Load a tile from matrix B [K, N].
    
    Args:
        ptr: Base pointer to matrix B
        stride_k: Stride in K dimension
        stride_n: Stride in N dimension
        K: Number of rows
        N: Number of columns
        tile: Tile with (pid_k, pid_n, BLOCK_K, BLOCK_N)
        boundary: Whether to apply boundary masking
        cache_modifier: Cache modifier for load
    
    Returns:
        Loaded tile data [BLOCK_K, BLOCK_N]
    """
    ptrs, mask = b_tile_ptrs(ptr, stride_k, stride_n, K, N, tile)
    if boundary:
        return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
    else:
        return tl.load(ptrs, cache_modifier=cache_modifier)


@triton.jit
def c_tile_ptrs(
    ptr,
    stride_m,
    stride_n,
    M,
    N,
    tile: Tile,
):
    """
    Get pointer array and mask for a tile of matrix C [M, N].
    
    Works with any stride types (tensor, constexpr, or int).
    
    Args:
        ptr: Base pointer to matrix C
        stride_m: Stride in M dimension (any type)
        stride_n: Stride in N dimension (any type)
        M: Number of rows
        N: Number of columns
        tile: Tile with (pid_m, pid_n, BLOCK_M, BLOCK_N)
    
    Returns:
        ptrs, mask: Pointer array [BLOCK_M, BLOCK_N] and bounds mask
    """
    rm, rn, mask = tile.layout(M, N)
    ptrs = ptr + rm[:, None] * stride_m + rn[None, :] * stride_n
    return ptrs, mask


@triton.jit
def c_store_tile(
    ptr,
    stride_m,
    stride_n,
    M,
    N,
    data,
    tile: Tile,
    mask=None,
):
    """
    Store data to a tile in matrix C [M, N].
    
    Args:
        ptr: Base pointer to matrix C
        stride_m: Stride in M dimension
        stride_n: Stride in N dimension
        M: Number of rows
        N: Number of columns
        data: Data to store [BLOCK_M, BLOCK_N]
        tile: Tile with coordinates and shape
        mask: Optional mask (if None, computes from bounds)
    """
    ptrs, bounds_mask = c_tile_ptrs(ptr, stride_m, stride_n, M, N, tile)
    if mask is None:
        mask = bounds_mask
    tl.store(ptrs, data, mask=mask)


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

# MatrixView is just an alias for tile_ptr from tensor_view.py
# Import it to provide compatibility
from .tensor_view import tile_ptr as MatrixView

# Placeholder classes that can't work due to strict typing
A_View = None
B_View = None
C_View = None
GemmMatrices = None

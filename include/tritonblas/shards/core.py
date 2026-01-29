# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Device functions for tritonblas shards.

These functions work with any Triton version and provide the same functionality
as the aggregate classes but without requiring @triton.constexpr_function.

Usage in kernels:
    from tritonblas.shards import (
        tile_layout, tile_coords, tile_ptr, chiplet_pid,
        gemm_k_loop
    )
"""

import triton
import triton.language as tl


# ============================================================
# Tile operations
# ============================================================

@triton.jit
def tile_layout(pid_m, pid_n, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Compute memory layout for a tile.
    
    Args:
        pid_m, pid_n: Tile coordinates
        M, N: Matrix dimensions
        BLOCK_M, BLOCK_N: Tile sizes (constexpr)
    
    Returns:
        rm: Row indices [BLOCK_M]
        rn: Column indices [BLOCK_N]
        mask: Bounds mask [BLOCK_M, BLOCK_N]
    """
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    return rm, rn, mask


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """
    Compute tile coordinates from linear tile ID with swizzling.
    
    Args:
        tile_id: Linear tile index
        num_pid_m, num_pid_n: Number of tiles in each dimension
        GROUP_SIZE_M: Group size for swizzling (constexpr)
    
    Returns:
        pid_m, pid_n: Tile coordinates
    """
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    return pid_m, pid_n


@triton.jit
def tile_ptr(
    ptr,
    pid_m, pid_n,
    stride_major, stride_minor,
    dim_major, dim_minor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute pointer tensor and mask for a tile.
    
    Args:
        ptr: Base pointer
        pid_m, pid_n: Tile coordinates
        stride_major, stride_minor: Strides (major=rows, minor=cols for row-major)
        dim_major, dim_minor: Dimensions (major=M, minor=K for A matrix)
        BLOCK_M, BLOCK_N: Tile sizes (constexpr)
    
    Returns:
        ptrs: Pointer tensor [BLOCK_M, BLOCK_N]
        mask: Bounds mask [BLOCK_M, BLOCK_N]
    """
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    mask = (rm[:, None] < dim_major) & (rn[None, :] < dim_minor)
    offset = rm[:, None] * stride_major + rn[None, :] * stride_minor
    ptrs = ptr + offset
    return ptrs, mask


# ============================================================
# Grid operations
# ============================================================

@triton.jit
def chiplet_pid(
    pid,
    num_workgroups: tl.constexpr,
    num_xcds: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """
    Transform PID for chiplet-aware mapping on multi-XCD AMD GPUs.
    
    Args:
        pid: Original program ID
        num_workgroups: Total number of workgroups (constexpr)
        num_xcds: Number of XCDs/chiplets (constexpr)
        chunk_size: Chunk size for mapping (constexpr)
    
    Returns:
        Transformed PID for better L2 cache locality
    """
    if num_xcds == 1:
        return pid
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@triton.jit
def grid_info(
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr = 1,
    CHUNK_SIZE: tl.constexpr = 1,
):
    """
    Compute grid information for persistent kernel loop.
    
    Args:
        M, N: Matrix dimensions
        BLOCK_M, BLOCK_N: Block sizes (constexpr)
        NUM_SMS: Number of SMs (constexpr)
        NUM_XCDS: Number of XCDs (constexpr, default 1)
        CHUNK_SIZE: Chunk size for chiplet mapping (constexpr, default 1)
    
    Returns:
        start_tile: Starting tile for this workgroup
        total_tiles: Total number of tiles
        stride: Stride for tile iteration
        num_pid_m, num_pid_n: Number of tiles in each dimension
    """
    pid = tl.program_id(0)
    if NUM_XCDS > 1:
        pid = chiplet_pid(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    
    return pid, total_tiles, NUM_SMS, num_pid_m, num_pid_n


# ============================================================
# GEMM operations
# ============================================================

@triton.jit
def gemm_k_loop(
    A, B,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    M, N, K,
    pid_m, pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
):
    """
    Execute the GEMM K-loop and return the accumulator.
    
    Args:
        A, B: Input matrix pointers
        stride_am, stride_ak: A matrix strides
        stride_bk, stride_bn: B matrix strides
        M, N, K: Matrix dimensions
        pid_m, pid_n: Output tile coordinates
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes (constexpr)
        EVEN_K: Whether K is evenly divisible by BLOCK_K (constexpr)
        QUANTIZED: Whether to use int32 accumulation (constexpr)
        ALLOW_TF32: Whether to allow TF32 (constexpr)
        CACHE_MODIFIER_A, CACHE_MODIFIER_B: Cache modifiers (constexpr)
    
    Returns:
        acc: Accumulator tensor [BLOCK_M, BLOCK_N]
    """
    # Initialize accumulator
    acc_dtype = tl.int32 if QUANTIZED else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    
    # Compute K loop bounds
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        num_k_tiles = num_k_tiles - 1
    tl.assume(num_k_tiles > 0)
    
    # Base pointers for A and B tiles
    a_base = A + pid_m * BLOCK_M * stride_am
    b_base = B + pid_n * BLOCK_N * stride_bn
    
    # Row/column indices
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # Main K loop
    for k_idx in range(num_k_tiles):
        k_offset = k_idx * BLOCK_K
        
        # A tile pointers [BLOCK_M, BLOCK_K]
        a_ptrs = a_base + rm[:, None] * stride_am + (k_offset + rk[None, :]) * stride_ak
        
        # B tile pointers [BLOCK_K, BLOCK_N]
        b_ptrs = b_base + (k_offset + rk[:, None]) * stride_bk + rn[None, :] * stride_bn
        
        # Load tiles
        a = tl.load(a_ptrs, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(b_ptrs, cache_modifier=CACHE_MODIFIER_B)
        
        # Accumulate
        if QUANTIZED:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
    
    # Handle K tail if needed
    if not EVEN_K:
        k_offset = num_k_tiles * BLOCK_K
        
        # A tile pointers with mask
        a_ptrs = a_base + rm[:, None] * stride_am + (k_offset + rk[None, :]) * stride_ak
        a_mask = (rm[:, None] < M) & ((k_offset + rk[None, :]) < K)
        
        # B tile pointers with mask
        b_ptrs = b_base + (k_offset + rk[:, None]) * stride_bk + rn[None, :] * stride_bn
        b_mask = ((k_offset + rk[:, None]) < K) & (rn[None, :] < N)
        
        # Load with masking
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=CACHE_MODIFIER_B)
        
        if QUANTIZED:
            acc += tl.dot(a, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
    
    return acc


@triton.jit
def apply_scales(acc, A_scale_ptr, B_scale_ptr, rm, rn, M, N):
    """
    Apply quantization scales to accumulator.
    
    Args:
        acc: Accumulator tensor [BLOCK_M, BLOCK_N]
        A_scale_ptr, B_scale_ptr: Scale pointers
        rm, rn: Row/column indices
        M, N: Matrix dimensions
    
    Returns:
        Scaled accumulator
    """
    a_scales = tl.load(A_scale_ptr + rm, mask=rm < M, other=1.0)
    b_scales = tl.load(B_scale_ptr + rn, mask=rn < N, other=1.0)
    acc = acc.to(tl.float32)
    acc = acc * a_scales[:, None]
    acc = acc * b_scales[None, :]
    return acc

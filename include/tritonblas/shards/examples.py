# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example kernels demonstrating the aggregate-based shards API.

Key pattern:
    1. TensorView.tile_ptr(tile) returns (ptrs, mask)
    2. User does tl.load(ptrs, mask=mask, other=0.0) explicitly
    3. GemmContext.accumulate(a, b) for accumulation
    4. Separate tiles for A and B with K offsets
"""

import triton
import triton.language as tl

from .tile import Tile
from .tensor_view import TensorView
from .gemm_context import GemmContext
from .grid import Grid
from .scale_view import ScaleView


@triton.jit
def persistent_matmul_with_shards(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
):
    """
    Persistent GEMM kernel using the aggregate-based shards API.
    
    Key design points:
    - TensorView.tile_ptr(tile) returns (ptrs, mask)
    - User does tl.load explicitly
    - GemmContext accumulates
    """
    acc_dtype = tl.int32 if QUANTIZED else tl.float32
    
    # Tensor views
    A_view = TensorView(A, stride_am, stride_ak, M, K)
    B_view = TensorView(B, stride_bk, stride_bn, K, N)
    C_view = TensorView(C, stride_cm, stride_cn, M, N)
    
    # Grid
    grid = Grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, NUM_SMS)
    
    # Scale view if quantized
    if A_scale_ptr is not None:
        scales = ScaleView(A_scale_ptr, B_scale_ptr)
    
    # Tile loop
    for tile_id in range(grid.start_tile, grid.total_tiles, grid.stride):
        pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
        
        # Output tile
        out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        
        # GEMM context
        ctx = GemmContext(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, acc_dtype, ALLOW_TF32)
        
        # K loop
        num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            num_k_tiles -= 1
        
        for k_idx in range(num_k_tiles):
            # Input tiles at k offset
            a_tile = Tile(pid_m, k_idx, BLOCK_SIZE_M, BLOCK_SIZE_K)
            b_tile = Tile(k_idx, pid_n, BLOCK_SIZE_K, BLOCK_SIZE_N)
            
            # Get pointers and masks
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            b_ptrs, b_mask = B_view.tile_ptr(b_tile)
            
            # User does the load
            a = tl.load(a_ptrs, cache_modifier=".cg")
            b = tl.load(b_ptrs, cache_modifier=".cg")
            
            # Accumulate
            if QUANTIZED:
                ctx.dot_accumulate_int(a, b)
            else:
                ctx.dot_accumulate(a, b)
        
        # Handle K tail if needed
        if not EVEN_K:
            k_idx = num_k_tiles
            a_tile = Tile(pid_m, k_idx, BLOCK_SIZE_M, BLOCK_SIZE_K)
            b_tile = Tile(k_idx, pid_n, BLOCK_SIZE_K, BLOCK_SIZE_N)
            
            a_ptrs, a_mask = A_view.tile_ptr(a_tile)
            b_ptrs, b_mask = B_view.tile_ptr(b_tile)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            if QUANTIZED:
                ctx.dot_accumulate_int(a, b)
            else:
                ctx.dot_accumulate(a, b)
        
        # Get accumulator
        acc = ctx.get_accumulator()
        
        # Get output layout
        rm, rn, mask = out_tile.layout(M, N)
        
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            acc = scales.apply(acc, rm, rn, M, N)
        
        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            if QUANTIZED:
                acc = acc + bias_vector[:, None].to(tl.float32)
            else:
                acc = acc + bias_vector[:, None]
        
        # Store result
        result = acc.to(C.type.element_ty)
        c_ptrs, c_mask = C_view.tile_ptr(out_tile)
        tl.store(c_ptrs, result, mask=c_mask)


@triton.jit
def simple_matmul_with_shards(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak: tl.constexpr,
    stride_bn,
    stride_bk: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Simple GEMM kernel using ctx.execute() for the full GEMM loop.
    
    Pattern:
        1. Create TensorViews for A, B, C
        2. Create output Tile
        3. ctx.execute(A_view, B_view, out_tile, K) -> acc
        4. tl.store(c_ptrs, acc, mask=c_mask)
    """
    # Tensor views
    A_view = TensorView(A, stride_am, stride_ak, M, K)
    B_view = TensorView(B, stride_bk, stride_bn, K, N)
    C_view = TensorView(C, stride_cm, stride_cn, M, N)
    
    # Grid
    grid = Grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, NUM_SMS)
    
    for tile_id in range(grid.start_tile, grid.total_tiles, grid.stride):
        pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
        
        # Output tile Information
        out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        
        # GEMM context, compute according indices
        ctx = GemmContext(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        
        # Execute full GEMM loop - returns accumulator (K is inferred from views)
        acc = ctx.execute(A_view, B_view, out_tile)
        
        # Store
        result = acc.to(C.type.element_ty)
        c_ptrs, c_mask = C_view.tile_ptr(out_tile)
        tl.store(c_ptrs, result, mask=c_mask)


@triton.jit
def elementwise_with_tile(
    X,
    Y,
    M,
    N,
    stride_xm,
    stride_xn: tl.constexpr,
    stride_ym,
    stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Example showing tile_ptr usage for non-GEMM operations.
    """
    # Tensor views
    X_view = TensorView(X, stride_xm, stride_xn, M, N)
    Y_view = TensorView(Y, stride_ym, stride_yn, M, N)
    
    # Grid
    grid = Grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, NUM_SMS)
    
    for tile_id in range(grid.start_tile, grid.total_tiles, grid.stride):
        pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
        
        # Tile
        tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        
        # Get pointers and mask
        x_ptrs, x_mask = X_view.tile_ptr(tile)
        
        # Load
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Compute
        y = x * 2.0
        
        # Store
        y_ptrs, y_mask = Y_view.tile_ptr(tile)
        tl.store(y_ptrs, y, mask=y_mask)

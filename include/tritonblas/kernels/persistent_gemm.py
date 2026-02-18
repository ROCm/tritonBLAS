# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent GEMM kernel using tritonblas aggregates.

This kernel uses InputView and OutputView aggregates that store matrix layout
information, enabling clean separation between pointer computation and GEMM logic.
"""

import triton
import triton.language as tl
import torch

from tritonblas.kernels.stages import (
    ScheduleContext,
    GemmContext,
    make_input_view,
    make_output_view,
    make_scale_view,
    make_bias_view,
)


@triton.jit
def _read_realtime():
    """Read GPU wall clock timestamp from s_memrealtime (100MHz constant clock)."""
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp


@triton.jit
def _get_xcc_id():
    """Get XCC (GPU chiplet) ID for the current workgroup."""
    xcc_id = tl.inline_asm_elementwise(
        asm="s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
        constraints=("=s"),
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    return xcc_id


@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    # Trace buffers (None when TRACE is False)
    trace_start_ptr,
    trace_end_ptr,
    trace_pid_ptr,
    trace_xcd_ptr,
    # Performance parameters (used to construct GemmContext on device)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    # Mosaic scheduling parameters
    MOSAIC_MODE: tl.constexpr = 0,
    MOSAIC_META_Y: tl.constexpr = 1,
    MOSAIC_META_X: tl.constexpr = 1,
    MOSAIC_META_ORDERING: tl.constexpr = 0,
    MOSAIC_L2_TILE_Y: tl.constexpr = 1,
    MOSAIC_L2_TILE_X: tl.constexpr = 1,
    MOSAIC_L2_ORDERING: tl.constexpr = 0,
    MOSAIC_HAS_L3: tl.constexpr = False,
    MOSAIC_L3_TILE_Y: tl.constexpr = 1,
    MOSAIC_L3_TILE_X: tl.constexpr = 1,
    MOSAIC_L3_ORDERING: tl.constexpr = 0,
    # Trace flag (compile-time: zero overhead when False)
    TRACE: tl.constexpr = False,
):
    """
    Persistent GEMM kernel using GemmContext aggregate.
    
    Matrix views are created using factory functions that handle any memory layout.
    Just describe your matrices and their strides - no layout flags needed.
    
    STRIDE PARAMETERS:
    ------------------
    - stride_am: A's stride when moving one M row down
    - stride_ak: A's stride in K dimension
    - stride_bk: B's stride in K dimension
    - stride_bn: B's stride when moving one N column right
    - stride_cm: C's stride when moving one M row down
    - stride_cn: C's stride in N dimension
    """

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE MATRIX VIEWS
    # ════════════════════════════════════════════════════════════════════════
    # Factory functions handle stride type coercion automatically
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE EPILOGUE VIEWS (optional scale and bias)
    # ════════════════════════════════════════════════════════════════════════
    scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N) if A_scale_ptr is not None else None
    bias_view = make_bias_view(bias_ptr, M, stride_bias) if BIAS else None
    
    # ════════════════════════════════════════════════════════════════════════
    # CONSTRUCT GEMM CONTEXT TO MANAGE MATH RELEVANT CONTEXT
    # ════════════════════════════════════════════════════════════════════════
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
        # Mosaic parameters
        MOSAIC_MODE,
        MOSAIC_META_Y, MOSAIC_META_X,
        MOSAIC_META_ORDERING,
        MOSAIC_L2_TILE_Y, MOSAIC_L2_TILE_X, MOSAIC_L2_ORDERING,
        MOSAIC_HAS_L3,
        MOSAIC_L3_TILE_Y, MOSAIC_L3_TILE_X, MOSAIC_L3_ORDERING,
    )
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE SCHEDULE CONTEXT FROM GEMM CONTEXT TO MANAGE OUTER LOOP ITERATION
    # ════════════════════════════════════════════════════════════════════════
    sched = ScheduleContext(M, N, K, ctx)
    
    # ════════════════════════════════════════════════════════════════════════
    # PERSISTENT LOOP: Process multiple tiles per workgroup
    # ════════════════════════════════════════════════════════════════════════
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        # ════════════════════════════════════════════════════
        # Compute tile coordinates based on scheduling mode
        # ════════════════════════════════════════════════════
        out_tile = sched.get_tile_from_idx(tile_id)

        if TRACE:
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            flat_tile_id = out_tile.pid_m * num_pid_n + out_tile.pid_n
            tl.store(trace_start_ptr + flat_tile_id, _read_realtime())
            tl.store(trace_pid_ptr + flat_tile_id, tl.program_id(0))
            tl.store(trace_xcd_ptr + flat_tile_id, _get_xcc_id())
        
        # ════════════════════════════════════════════════════════════════════
        # COMPUTE GEMM: K-loop handled by GemmContext
        # ════════════════════════════════════════════════════════════════════
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        
        # ════════════════════════════════════════════════════════════════════
        # STORE RESULT: Epilogue (scale, bias, convert) handled by OutputView
        # Store Accumulator to output matrix C at pointers defined by out_tile
        # ════════════════════════════════════════════════════════════════════
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)

        if TRACE:
            tl.store(trace_end_ptr + flat_tile_id, _read_realtime())

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
        # Get schedule aware output tile to be processed this loop iteration
        # ════════════════════════════════════════════════════
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # ════════════════════════════════════════════════════════════════════
        # COMPUTE GEMM: K-loop handled by GemmContext
        # ════════════════════════════════════════════════════════════════════
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        
        # ════════════════════════════════════════════════════════════════════
        # STORE RESULT: Epilogue (scale, bias, convert) handled by OutputView
        # Store Accumulator to output matrix C at pointers defined by out_tile
        # ════════════════════════════════════════════════════════════════════
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)

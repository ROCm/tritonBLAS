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
    Tile, 
    GemmContext, 
    GemmConfig,
    make_input_view,
    make_output_view,
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
    # Performance parameters (used to construct GemmConfig on device)
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
    Persistent GEMM kernel using GemmConfig aggregate.
    
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
    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    
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
    # CONSTRUCT GEMM CONFIG
    # ════════════════════════════════════════════════════════════════════════
    config = GemmConfig(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE SCHEDULE AND GEMM CONTEXTS
    # ════════════════════════════════════════════════════════════════════════
    sched = ScheduleContext(M, N, K, config)
    ctx = GemmContext(config)
    
    # ════════════════════════════════════════════════════════════════════════
    # PERSISTENT LOOP: Process multiple tiles per workgroup
    # ════════════════════════════════════════════════════════════════════════
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        pid_m, pid_n = sched.get_tile(tile_id)
        out_tile = Tile(pid_m, pid_n, config.block_m, config.block_n)
        
        # ════════════════════════════════════════════════════════════════════
        # COMPUTE GEMM: K-loop handled by GemmContext
        # ════════════════════════════════════════════════════════════════════
        acc = ctx.k_complete(tensorA, tensorB, out_tile)
        
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            acc = out_tile.scale(acc, A_scale_ptr, B_scale_ptr, M, N)
        
        # Add bias if provided
        if BIAS:
            acc = out_tile.bias(acc, bias_ptr, M, stride_bias)
        
        # ════════════════════════════════════════════════════════════════════
        # STORE RESULT
        # ════════════════════════════════════════════════════════════════════
        result = acc.to(C.type.element_ty)
        c_ptrs, c_mask = tensorC.tile_ptrs(out_tile)
        tl.store(c_ptrs, result, mask=c_mask)

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Persistent GEMM kernel for chained matrix multiplications.

This kernel performs two consecutive GEMMs in a fused manner:
    C = A @ B
    E = C @ D

Where the intermediate result C is computed and immediately used as input
for the second GEMM operation.
"""

import triton
import triton.language as tl
import torch

from tritonblas.kernels.stages import (
    ScheduleContext,
    GemmContext,
    make_input_view,
    make_output_view,
)

import mosaic

@triton.jit()
def fused_persistent_matmul(
    A,
    B,
    C,
    D,
    E,
    M,
    N,
    K,
    P,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dn,
    stride_dp,
    stride_em,
    stride_ep,
    locks,
    alpha_xcd_map,
    beta_xcd_map,
    # Performance parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS_ALPHA: tl.constexpr,
    NUM_SMS_BETA: tl.constexpr,
    NUM_SMS_TOTAL: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    EVEN_K: tl.constexpr,
    SHOW_MAP: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    # Mosaic scheduling parameters (for API compatibility, not used in fused kernel yet)
    MOSAIC_MODE: tl.constexpr = 0,
    MOSAIC_META_ORDERING: tl.constexpr = 0,
    MOSAIC_L2_TILE_Y: tl.constexpr = 1,
    MOSAIC_L2_TILE_X: tl.constexpr = 1,
    MOSAIC_L2_ORDERING: tl.constexpr = 0,
    MOSAIC_HAS_L3: tl.constexpr = False,
    MOSAIC_L3_TILE_Y: tl.constexpr = 1,
    MOSAIC_L3_TILE_X: tl.constexpr = 1,
    MOSAIC_L3_ORDERING: tl.constexpr = 0,
):
    """
    Fused persistent GEMM kernel for chained matrix multiplications.
    
    Computes:
        C = A @ B  (First GEMM: M×K @ K×N = M×N)
        E = C @ D  (Second GEMM: M×N @ N×P = M×P)
    
    Parameters:
    -----------
    A : Input matrix (M × K)
    B : Input matrix (K × N)
    C : Intermediate output matrix (M × N)
    D : Input matrix (N × P)
    E : Final output matrix (M × P)
    
    M, N, K, P : Matrix dimensions
    
    stride_am, stride_ak : Strides for matrix A
    stride_bk, stride_bn : Strides for matrix B
    stride_cm, stride_cn : Strides for matrix C
    stride_dn, stride_dp : Strides for matrix D
    stride_em, stride_ep : Strides for matrix E
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K : Tile sizes
    GROUP_SIZE_M : Swizzle parameter for M dimension
    NUM_SMS : Total number of streaming multiprocessors
    NUM_XCDS : Number of XCDs (chiplets)
    CHUNK_SIZE : Chunk size for workgroup mapping
    CACHE_MODIFIER_A, CACHE_MODIFIER_B : Cache modifiers for loads
    EVEN_K : Whether K is evenly divisible by BLOCK_SIZE_K
    ALLOW_TF32 : Whether to allow TF32 for FP32 inputs
    """
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # ════════════════════════════════════════════════════════════════════════
    # SPATIAL PARTITION: Use mosaic spatial partitioning
    # ════════════════════════════════════════════════════════════════════════
    xid, local_xid, wg_per_xcd_alpha, wg_per_xcd_beta, is_alpha, pid_alpha, is_beta, pid_beta = mosaic.spatial_partition(
        tl.program_id(0), NUM_SMS_ALPHA, NUM_SMS_BETA, NUM_SMS_TOTAL, NUM_XCDS
    )

    # ════════════════════════════════════════════════════════════════════════
    # CREATE MATRIX VIEWS
    # ════════════════════════════════════════════════════════════════════════
    # Factory functions handle stride type coercion automatically
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    tensorC_input = make_input_view(C, M, N, stride_cm, stride_cn)
    tensorD = make_input_view(D, N, P, stride_dn, stride_dp)
    tensorE = make_output_view(E, M, P, stride_em, stride_ep)

    # ════════════════════════════════════════════════════════════════════════
    # CREATE EPILOGUE VIEWS (optional scale and bias)
    # ════════════════════════════════════════════════════════════════════════
    scale_view = None
    bias_view = None

    if is_alpha:
        # This workgroup is part of the `alpha` partition.

        # ════════════════════════════════════════════════════════════════════════
        # CONSTRUCT GEMM CONTEXT TO MANAGE MATH RELEVANT CONTEXT
        # ════════════════════════════════════════════════════════════════════════
        alpha_ctx = GemmContext(
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            NUM_SMS_ALPHA, NUM_XCDS,
            GROUP_SIZE_M, CHUNK_SIZE,
            CACHE_MODIFIER_A, CACHE_MODIFIER_B,
            acc_dtype, ALLOW_TF32, EVEN_K, False,
        )

        # ════════════════════════════════════════════════════════════════════════
        # CREATE SCHEDULE CONTEXT FROM GEMM CONTEXT TO MANAGE OUTER LOOP ITERATION
        # ════════════════════════════════════════════════════════════════════════
        alpha_sched = ScheduleContext(M, N, K, alpha_ctx)

        # ════════════════════════════════════════════════════════════════════════
        # PERSISTENT LOOP: Process multiple tiles per workgroup
        # Use row-major tile iteration (no swizzling/transforms)
        # ════════════════════════════════════════════════════════════════════════
        num_pid_n_alpha = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_m_alpha = tl.cdiv(M, BLOCK_SIZE_M)
        total_tiles_alpha = num_pid_m_alpha * num_pid_n_alpha

        # pid_alpha is mapped to [0, NUM_SMS_ALPHA)
        for tile_id in range(pid_alpha, total_tiles_alpha, NUM_SMS_ALPHA):
            out_tile = alpha_sched.get_tile_from_idx(tile_id)

            # Debug: Record which XCD is processing this tile
            if SHOW_MAP:
                offset = out_tile.pid_m * num_pid_n_alpha + out_tile.pid_n
                tl.store(alpha_xcd_map + offset, xid.to(tl.int8))

            # ════════════════════════════════════════════════════════════════════
            # COMPUTE GEMM: K-loop handled by GemmContext
            # ════════════════════════════════════════════════════════════════════
            acc = alpha_ctx.reduce_axis(tensorA, tensorB, out_tile)

            # ════════════════════════════════════════════════════════════════════
            # STORE RESULT: Epilogue (scale, bias, convert) handled by OutputView
            # Store Accumulator to output matrix C at pointers defined by out_tile
            # ════════════════════════════════════════════════════════════════════
            tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)

            tl.debug_barrier()

            # Signal that this tile is ready (using row-major tile_id)
            # tl.atomic_xchg(locks + (out_tile.pid_m * num_pid_n_alpha + out_tile.pid_n), 1)
            tl.store(locks + tile_id, 1)

    else:
        # This workgroup is part of the beta partition.

        # ════════════════════════════════════════════════════════════════════════
        # CONSTRUCT GEMM CONTEXT TO MANAGE MATH RELEVANT CONTEXT
        # ════════════════════════════════════════════════════════════════════════
        beta_ctx = GemmContext(
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            NUM_SMS_BETA, NUM_XCDS,
            GROUP_SIZE_M, CHUNK_SIZE,
            CACHE_MODIFIER_A, CACHE_MODIFIER_B,
            acc_dtype, ALLOW_TF32, EVEN_K, False,
        )

        pid_m_beta = tl.cdiv(M, BLOCK_SIZE_M)
        pid_p_beta = tl.cdiv(P, BLOCK_SIZE_N)

        alpha_tiles_per_beta_tile = NUM_SMS_ALPHA // NUM_SMS_BETA

        # ════════════════════════════════════════════════════════════════════════
        # CREATE SCHEDULE CONTEXT FROM GEMM CONTEXT TO MANAGE OUTER LOOP ITERATION
        # ════════════════════════════════════════════════════════════════════════
        beta_sched = ScheduleContext(M, P, N, beta_ctx)

        # ════════════════════════════════════════════════════════════════════════
        # PERSISTENT LOOP: Process multiple tiles per workgroup
        # Use row-major tile iteration (no swizzling/transforms)
        # ════════════════════════════════════════════════════════════════════════
        num_pid_n_alpha = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_p_beta = tl.cdiv(P, BLOCK_SIZE_N)
        num_pid_m_beta = tl.cdiv(M, BLOCK_SIZE_M)
        total_tiles_beta = num_pid_m_beta * num_pid_p_beta
        for tile_id in range(pid_beta, total_tiles_beta, NUM_SMS_BETA):
            # ════════════════════════════════════════════════════
            # Compute tile coordinates using ROW-MAJOR indexing (no swizzling)
            # ════════════════════════════════════════════════════
            out_tile = beta_sched.get_tile_from_idx(tile_id)

            # Debug: Record which XCD is processing this tile
            if SHOW_MAP:
                offset = out_tile.pid_m * num_pid_p_beta + out_tile.pid_n
                tl.store(beta_xcd_map + offset, xid.to(tl.int8))

            # We need to wait on all the input tiles from C that this output tile depends on.
            # Beta tile at (pid_m, pid_p) needs all alpha tiles in row pid_m.
            # Use row-major tile indexing (matching alpha's lock storage).
            row_start_tile = out_tile.pid_m * num_pid_n_alpha
            row_end_tile = row_start_tile + num_pid_n_alpha
            for dep_tile in range(row_start_tile, row_end_tile):
                # while tl.atomic_cas(locks + dep_tile, 1, 1) != 1:
                #     pass
                while tl.load(locks + dep_tile, cache_modifier=".cv", volatile=True) != 1:
                    pass

            # ════════════════════════════════════════════════════════════════════
            # COMPUTE GEMM: K-loop handled by GemmContext
            # ════════════════════════════════════════════════════════════════════
            acc = beta_ctx.reduce_axis(tensorC_input, tensorD, out_tile)

            # ════════════════════════════════════════════════════════════════════
            # STORE RESULT: Epilogue (scale, bias, convert) handled by OutputView
            # Store Accumulator to output matrix C at pointers defined by out_tile
            # ════════════════════════════════════════════════════════════════════
            tensorE.store(acc, out_tile, scale=scale_view, bias=bias_view)
    

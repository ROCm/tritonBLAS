# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Workgroup-to-tile mapping kernel for persistent GEMM.

Reuses the exact same ScheduleContext/GemmContext logic as persistent_gemm
to compute, for each workgroup, the (first) output tile (pid_m, pid_n) it
would process. Used for debugging and validation (e.g. --extract-layouts).
Output format matches C++ extractor: one (tile_m, tile_n) per workgroup.
"""

import triton
import triton.language as tl
import torch

from tritonblas.kernels.stages import ScheduleContext, GemmContext


@triton.jit()
def persistent_wg_mapping(
    tile_m_out,
    tile_n_out,
    M,
    N,
    K,
    # Same constexpr as persistent_matmul (no matrix pointers)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
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
):
    """
    Compute workgroup -> (tile_m, tile_n) mapping using the same logic as
    persistent_matmul. Each workgroup writes its first (and with grids=total_tiles,
    only) tile to tile_m_out[wgid], tile_n_out[wgid].
    """
    acc_dtype = tl.float32

    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
        MOSAIC_MODE,
        MOSAIC_META_Y, MOSAIC_META_X,
        MOSAIC_META_ORDERING,
        MOSAIC_L2_TILE_Y, MOSAIC_L2_TILE_X, MOSAIC_L2_ORDERING,
        MOSAIC_HAS_L3,
        MOSAIC_L3_TILE_Y, MOSAIC_L3_TILE_X, MOSAIC_L3_ORDERING,
    )

    sched = ScheduleContext(M, N, K, ctx)

    start_tile, total_tiles, stride = sched.persistent_tile_range()
    out_tile = sched.get_tile_from_idx(start_tile)

    wgid = tl.program_id(0)
    tl.store(tile_m_out + wgid, out_tile.pid_m.to(tl.int32))
    tl.store(tile_n_out + wgid, out_tile.pid_n.to(tl.int32))

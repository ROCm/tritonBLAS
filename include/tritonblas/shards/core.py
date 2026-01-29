# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Helper functions for tritonblas shards.

The aggregate classes are in their own files:
- tile.py: Tile
- tensor_view.py: TensorView
- gemm_context.py: GemmContext
- grid.py: Grid, GemmGrid
- scale_view.py: ScaleView
"""

import triton
import triton.language as tl


@triton.jit
def tile_layout(pid_m, pid_n, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Compute memory layout for a tile."""
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    return rm, rn, mask


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """Compute tile coordinates from linear tile ID."""
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

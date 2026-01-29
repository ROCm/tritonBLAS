# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Grid aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class Grid:
    """
    2D grid context for tile iteration.
    
    Example usage:
        grid = Grid(M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS)
        
        for tile_id in range(grid.start_tile, grid.total_tiles, grid.stride):
            pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
            out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
    """
    
    M: tl.tensor
    N: tl.tensor
    num_pid_m: tl.tensor
    num_pid_n: tl.tensor
    total_tiles: tl.tensor
    start_tile: tl.tensor
    stride: tl.constexpr
    block_m: tl.constexpr
    block_n: tl.constexpr
    group_size_m: tl.constexpr
    
    @triton.constexpr_function
    def __init__(self, M, N, block_m, block_n, group_size_m, num_sms):
        self.M = M
        self.N = N
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.group_size_m = tl.constexpr(group_size_m)
        self.stride = tl.constexpr(num_sms)
        
        self.num_pid_m = tl.cdiv(M, block_m)
        self.num_pid_n = tl.cdiv(N, block_n)
        self.total_tiles = self.num_pid_m * self.num_pid_n
        self.start_tile = tl.program_id(0)
    
    @triton.jit
    def tile_idx_to_coord(self, tile_id):
        """Convert linear tile ID to (pid_m, pid_n) coordinates."""
        num_pid_in_group = self.group_size_m * self.num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * self.group_size_m
        group_size_m = tl.minimum(self.num_pid_m - first_pid_m, self.group_size_m)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        return pid_m, pid_n


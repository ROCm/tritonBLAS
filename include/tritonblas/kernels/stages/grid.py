# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Grid aggregate for tritonblas shards.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


@triton.jit
def chiplet_transform(
    pid,
    num_workgroups: tl.constexpr,
    num_xcds: tl.constexpr,
):
    """Transform PID for basic chiplet-aware mapping."""
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + tl.minimum(xcd, extra_sms)
    return offset + pos_in_xcd


@triton.jit
def chiplet_transform_chunked(
    pid,
    num_workgroups: tl.constexpr,
    num_xcds: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """Transform PID for chunked chiplet-aware mapping."""
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@aggregate
class Grid:
    """
    2D grid context for tile iteration with optional chiplet-aware mapping.
    
    Example usage:
        grid = Grid(M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, num_xcds=NUM_XCDS)
        
        start_tile, total_tiles = grid.get_tile_range()
        for tile_id in range(start_tile, total_tiles, grid.stride):
            pid_m, pid_n = grid.tile_idx_to_coord(tile_id)
            out_tile = Tile(pid_m, pid_n, BLOCK_M, BLOCK_N)
    """
    
    M: tl.tensor
    N: tl.tensor
    stride: tl.constexpr
    block_m: tl.constexpr
    block_n: tl.constexpr
    group_size_m: tl.constexpr
    num_xcds: tl.constexpr
    chunk_size: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        M,
        N,
        block_m,
        block_n,
        group_size_m,
        num_sms,
        num_xcds=1,
        chunk_size=1,
    ):
        """
        Create a Grid.
        
        Args:
            M, N: Problem dimensions
            block_m, block_n: Block sizes
            group_size_m: Group size for swizzling
            num_sms: Number of SMs (stride for persistent loop)
            num_xcds: Number of chiplets/XCDs (default: 1, no chiplet mapping)
            chunk_size: Chunk size for chiplet mapping (default: 1)
        """
        self.M = M
        self.N = N
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.group_size_m = tl.constexpr(group_size_m)
        self.stride = tl.constexpr(num_sms)
        self.num_xcds = tl.constexpr(num_xcds)
        self.chunk_size = tl.constexpr(chunk_size)
    
    @triton.jit
    def get_tile_range(self):
        """
        Get the tile range for this workgroup.
        
        Returns:
            start_tile: Starting tile ID with chiplet-aware mapping
            total_tiles: Total number of tiles
        """
        num_pid_m = tl.cdiv(self.M, self.block_m)
        num_pid_n = tl.cdiv(self.N, self.block_n)
        total_tiles = num_pid_m * num_pid_n
        
        pid = tl.program_id(0)
        # Always apply transform - when num_xcds=1 and chunk_size=1, returns pid unchanged
        pid = chiplet_transform_chunked(pid, self.stride, self.num_xcds, self.chunk_size)
        
        return pid, total_tiles
    
    @triton.jit
    def tile_idx_to_coord(self, tile_id):
        """Convert linear tile ID to (pid_m, pid_n) coordinates."""
        num_pid_m = tl.cdiv(self.M, self.block_m)
        num_pid_n = tl.cdiv(self.N, self.block_n)
        num_pid_in_group = self.group_size_m * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * self.group_size_m
        group_size_m = tl.minimum(num_pid_m - first_pid_m, self.group_size_m)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        return pid_m, pid_n


# Legacy alias
GemmGrid = Grid

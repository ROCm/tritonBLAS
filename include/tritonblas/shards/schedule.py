# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
ScheduleContext aggregate for tritonblas shards.

Provides a simple iterator interface that hides the complexity of persistent
GEMM and Stream-K scheduling. Just call next_tile() or next_iter() to get
the work unit coordinates.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .grid import chiplet_transform_chunked


@aggregate
class ScheduleContext:
    """
    Unified scheduling context that hides persistent GEMM loop complexity.
    
    Two simple iteration patterns:
    - Tile loop: for tile_id in range(start, total, stride) with get_tile(tile_id)
    - Iter loop: for iter_id in range(start, end) with get_iter(iter_id)
    
    Example usage (persistent GEMM):
        sched = ScheduleContext(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, 
                                GROUP_SIZE_M, NUM_SMS, num_xcds=NUM_XCDS)
        
        start, total, stride = sched.tile_range()
        for tile_id in range(start, total, stride):
            pid_m, pid_n = sched.get_tile(tile_id)
            # Process full tile at (pid_m, pid_n)
    
    Example usage (Stream-K):
        sched = ScheduleContext(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                GROUP_SIZE_M, NUM_SMS, streamk_tiles=STREAMK_TILES)
        
        start, end = sched.iter_range()
        for iter_id in range(start, end):
            pid_m, pid_n, k_iter = sched.get_iter(iter_id)
            # Process single K iteration at (pid_m, pid_n, k_iter)
    """
    
    # Problem dimensions
    M: tl.tensor
    N: tl.tensor
    K: tl.tensor
    
    # Block sizes
    block_m: tl.constexpr
    block_n: tl.constexpr
    block_k: tl.constexpr
    
    # Grid configuration
    group_size_m: tl.constexpr
    num_sms: tl.constexpr
    num_xcds: tl.constexpr
    chunk_size: tl.constexpr
    
    # Stream-K specific
    streamk_tiles: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        M,
        N,
        K,
        block_m,
        block_n,
        block_k,
        group_size_m,
        num_sms,
        num_xcds=1,
        chunk_size=1,
        streamk_tiles=0,
    ):
        """
        Create a ScheduleContext.
        
        Args:
            M, N, K: Problem dimensions
            block_m, block_n, block_k: Block sizes
            group_size_m: Group size for tile swizzling
            num_sms: Number of SMs/workgroups
            num_xcds: Number of chiplets/XCDs (default: 1)
            chunk_size: Chunk size for chiplet mapping (default: 1)
            streamk_tiles: Number of tiles for Stream-K (0 = persistent only)
        """
        self.M = M
        self.N = N
        self.K = K
        self.block_m = tl.constexpr(block_m)
        self.block_n = tl.constexpr(block_n)
        self.block_k = tl.constexpr(block_k)
        self.group_size_m = tl.constexpr(group_size_m)
        self.num_sms = tl.constexpr(num_sms)
        self.num_xcds = tl.constexpr(num_xcds)
        self.chunk_size = tl.constexpr(chunk_size)
        self.streamk_tiles = tl.constexpr(streamk_tiles)
    
    # ================================================================
    # Tile-level iteration (for persistent GEMM)
    # ================================================================
    
    @triton.jit
    def tile_range(self):
        """
        Get tile iteration range for this workgroup.
        
        Returns:
            (start, total, stride): Use as range(start, total, stride)
        """
        num_pid_m = tl.cdiv(self.M, self.block_m)
        num_pid_n = tl.cdiv(self.N, self.block_n)
        total_tiles = num_pid_m * num_pid_n
        
        # Get transformed program ID
        pid = tl.program_id(0)
        if self.num_xcds != 1:
            pid = chiplet_transform_chunked(pid, self.num_sms, self.num_xcds, self.chunk_size)
        
        return pid, total_tiles, self.num_sms
    
    @triton.jit
    def get_tile(self, tile_id):
        """
        Get tile coordinates for a given tile ID.
        
        Args:
            tile_id: Linear tile index
            
        Returns:
            (pid_m, pid_n): Tile coordinates
        """
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
    
    # ================================================================
    # Iteration-level iteration (for Stream-K)
    # ================================================================
    
    @triton.jit
    def iter_range(self):
        """
        Get iteration range for this workgroup (Stream-K mode).
        
        Returns:
            (start_iter, end_iter): Iteration range [start, end)
        """
        num_pid_m = tl.cdiv(self.M, self.block_m)
        num_pid_n = tl.cdiv(self.N, self.block_n)
        total_tiles = num_pid_m * num_pid_n
        iters_per_tile = tl.cdiv(self.K, self.block_k)
        
        # Get transformed program ID
        pid = tl.program_id(0)
        if self.num_xcds != 1:
            pid = chiplet_transform_chunked(pid, self.num_sms, self.num_xcds, self.chunk_size)
        
        total_full_tiles = total_tiles - self.streamk_tiles
        total_streamk_iters = self.streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // self.num_sms
        streamk_remainder_iters = total_streamk_iters % self.num_sms
        
        start_iter = (
            total_full_tiles * iters_per_tile +
            pid * streamk_iters_pcu +
            tl.minimum(pid, streamk_remainder_iters)
        )
        end_iter = (
            total_full_tiles * iters_per_tile +
            (pid + 1) * streamk_iters_pcu +
            tl.minimum(pid + 1, streamk_remainder_iters)
        )
        
        return start_iter, end_iter
    
    @triton.jit
    def get_iter(self, global_iter):
        """
        Get coordinates for a given global iteration.
        
        Args:
            global_iter: Global iteration index
            
        Returns:
            (pid_m, pid_n, k_iter): Tile coordinates and K iteration index
        """
        iters_per_tile = tl.cdiv(self.K, self.block_k)
        
        # Convert global iteration to (tile_id, k_iter)
        tile_id = global_iter // iters_per_tile
        k_iter = global_iter % iters_per_tile
        
        # Convert tile_id to (pid_m, pid_n)
        pid_m, pid_n = self.get_tile(tile_id)
        
        return pid_m, pid_n, k_iter
    
    @triton.jit
    def iters_per_tile(self):
        """Number of K iterations per tile."""
        return tl.cdiv(self.K, self.block_k)
    
    @triton.jit
    def total_tiles(self):
        """Total number of tiles."""
        num_pid_m = tl.cdiv(self.M, self.block_m)
        num_pid_n = tl.cdiv(self.N, self.block_n)
        return num_pid_m * num_pid_n

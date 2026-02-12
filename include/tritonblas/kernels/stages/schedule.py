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
from .gemm_context import GemmContext
from .tile import Tile

import mosaic

@aggregate
class ScheduleContext:
    """
    Unified scheduling context that hides persistent GEMM loop complexity.

    Two simple iteration patterns:
    - Tile loop: for tile_id in range(start, total, stride) with get_tile(tile_id)
    - Iter loop: for iter_id in range(start, end) with get_iter(iter_id)

    Example usage (persistent GEMM):
        ctx = GemmContext(block_m=128, block_n=256, block_k=64,
                          num_sms=NUM_SMS, num_xcds=NUM_XCDS)
        sched = ScheduleContext(M, N, K, ctx)

        start, total, stride = sched.persistent_tile_range()
        for tile_id in range(start, total, stride):
            out_tile = sched.get_tile_from_idx(tile_id)
            # Process full tile

    Example usage (Stream-K):
        ctx = GemmContext(block_m=128, block_n=256, block_k=64, num_sms=NUM_SMS)
        sched = ScheduleContext(M, N, K, ctx, streamk_tiles=STREAMK_TILES)

        start, end = sched.iter_range()
        for iter_id in range(start, end):
            pid_m, pid_n, k_iter = sched.get_iter(iter_id)
            # Process single K iteration at (pid_m, pid_n, k_iter)
    """

    # Problem dimensions
    M: tl.tensor
    N: tl.tensor
    K: tl.tensor

    # GemmContext with all block sizes and scheduling params
    ctx: GemmContext

    # Stream-K specific
    streamk_tiles: tl.constexpr
    
    @triton.constexpr_function
    def __init__(
        self,
        M,
        N,
        K,
        ctx: GemmContext,
        streamk_tiles=0,
    ):
        """
        Create a ScheduleContext from a GemmContext.

        Args:
            M, N, K: Problem dimensions
            ctx: GemmContext with block sizes and scheduling parameters
            streamk_tiles: Number of tiles for Stream-K (0 = persistent only)
        """
        self.M = M
        self.N = N
        self.K = K
        self.ctx = ctx
        self.streamk_tiles = tl.constexpr(streamk_tiles)
    
    # ================================================================
    # Tile-level iteration (for persistent GEMM)
    # ================================================================
    
    @triton.jit
    def persistent_tile_range(self):
        """
        Get tile iteration range for this workgroup (persistent GEMM).

        Applies chiplet transform based on chunk_size:
            chunk_size > 0: chiplet_transform_chunked (chunked assignment)
            chunk_size = 0: chiplet_transform (contiguous assignment)

        Returns:
            (start, total, stride): Use as range(start, total, stride)

            start, total, stride = sched.persistent_tile_range()
            for tile_id in range(start, total, stride):
                out_tile = sched.get_tile_from_idx(tile_id)
                ...
        """
        num_pid_m = tl.cdiv(self.M, self.ctx.block_m)
        num_pid_n = tl.cdiv(self.N, self.ctx.block_n)
        total_tiles = num_pid_m * num_pid_n

        # Get transformed program ID with chiplet transform
        pid = tl.program_id(0)
        # Only apply chiplet transform if NOT in default mode (mode 2)
        if self.ctx.num_xcds != 1 and self.ctx.mosaic_mode != 2:
            if self.ctx.chunk_size == 0:
                # Use chiplet_transform (contiguous assignment to XCDs)
                pid = mosaic.chiplet_transform(pid, self.ctx.num_sms, self.ctx.num_xcds)
            else:
                # Use chiplet_transform_chunked (chunked assignment)
                pid = mosaic.chiplet_transform_chunked(pid, self.ctx.num_sms, self.ctx.num_xcds, self.ctx.chunk_size)

        return pid, total_tiles, self.ctx.num_sms
    
    @triton.jit
    def get_tile_from_idx(self, tile_id):
        """
        Get a Tile for a given tile ID using the configured scheduling mode.

        Scheduling modes:
            0 (baseline): wgm_transform (current tritonBLAS behavior)
            1 (mosaic): Hierarchical layouts with gemm_grid_transform
            2 (default): Raw row-major assignment, no transforms

        Args:
            tile_id: Linear tile index

        Returns:
            Tile: Tile object with computed coordinates and ctx block sizes
        """
        num_pid_m = tl.cdiv(self.M, self.ctx.block_m)
        num_pid_n = tl.cdiv(self.N, self.ctx.block_n)

        if self.ctx.mosaic_mode == 1:  # Mosaic mode (hierarchical layouts)
            # Create hierarchical layout and apply gemm_grid_transform
            if self.ctx.mosaic_has_l3:
                # 3-level layout: LayoutRank2Depth3
                layout = mosaic.LayoutRank2Depth3(
                    self.ctx.mosaic_meta_y, self.ctx.mosaic_meta_x,            # outer
                    self.ctx.mosaic_l3_tile_y, self.ctx.mosaic_l3_tile_x,      # middle
                    self.ctx.mosaic_l2_tile_y, self.ctx.mosaic_l2_tile_x,      # inner
                    self.ctx.mosaic_meta_ordering,
                    self.ctx.mosaic_l3_ordering,
                    self.ctx.mosaic_l2_ordering,
                )
            else:
                # 2-level layout: LayoutRank2Depth2
                layout = mosaic.LayoutRank2Depth2(
                    self.ctx.mosaic_meta_y, self.ctx.mosaic_meta_x,            # outer
                    self.ctx.mosaic_l2_tile_y, self.ctx.mosaic_l2_tile_x,      # inner
                    self.ctx.mosaic_meta_ordering,
                    self.ctx.mosaic_l2_ordering,
                )

            # Apply gemm_grid_transform
            offset = layout.gemm_grid_transform(tile_id, num_pid_m, num_pid_n)
            pid_m = offset // num_pid_n
            pid_n = offset % num_pid_n

        elif self.ctx.mosaic_mode == 2:  # Default mode (no transforms)
            # Raw row-major assignment
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

        else:  # mosaic_mode == 0: Baseline mode (current tritonBLAS behavior)
            # Use wgm_transform (grouped scheduling)
            pid_m, pid_n = mosaic.wgm_transform(tile_id, num_pid_m, num_pid_n, self.ctx.group_size_m)

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        return Tile(pid_m, pid_n, self.ctx.block_m, self.ctx.block_n)
    
    @triton.jit
    def get_tile_from_coord(self, pid_m, pid_n):
        """
        Get a Tile from 2D coordinates.
        
        Args:
            pid_m: Tile coordinate in M dimension
            pid_n: Tile coordinate in N dimension
            
        Returns:
            Tile: Tile object with the given coordinates and ctx block sizes
        """
        return Tile(pid_m, pid_n, self.ctx.block_m, self.ctx.block_n)
    
    @triton.jit
    def _tile_idx_to_coord(self, tile_id):
        """
        Internal: Convert tile ID to coordinates (returns tuple).
        
        Args:
            tile_id: Linear tile index
            
        Returns:
            (pid_m, pid_n): Tile coordinates as tuple
        """
        num_pid_m = tl.cdiv(self.M, self.ctx.block_m)
        num_pid_n = tl.cdiv(self.N, self.ctx.block_n)
        
        num_pid_in_group = self.ctx.group_size_m * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * self.ctx.group_size_m
        group_size_m = tl.minimum(num_pid_m - first_pid_m, self.ctx.group_size_m)
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
        num_pid_m = tl.cdiv(self.M, self.ctx.block_m)
        num_pid_n = tl.cdiv(self.N, self.ctx.block_n)
        total_tiles = num_pid_m * num_pid_n
        iters_per_tile = tl.cdiv(self.K, self.ctx.block_k)
        
        # Get transformed program ID
        pid = tl.program_id(0)
        if self.ctx.num_xcds != 1:
            pid = chiplet_transform_chunked(pid, self.ctx.num_sms, self.ctx.num_xcds, self.ctx.chunk_size)
        
        total_full_tiles = total_tiles - self.streamk_tiles
        total_streamk_iters = self.streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // self.ctx.num_sms
        streamk_remainder_iters = total_streamk_iters % self.ctx.num_sms
        
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
        iters_per_tile = tl.cdiv(self.K, self.ctx.block_k)
        
        # Convert global iteration to (tile_id, k_iter)
        tile_id = global_iter // iters_per_tile
        k_iter = global_iter % iters_per_tile
        
        # Convert tile_id to (pid_m, pid_n)
        pid_m, pid_n = self._tile_idx_to_coord(tile_id)
        
        return pid_m, pid_n, k_iter
    
    @triton.jit
    def iters_per_tile(self):
        """Number of K iterations per tile."""
        return tl.cdiv(self.K, self.ctx.block_k)
    
    @triton.jit
    def total_tiles(self):
        """Total number of tiles."""
        num_pid_m = tl.cdiv(self.M, self.ctx.block_m)
        num_pid_n = tl.cdiv(self.N, self.ctx.block_n)
        return num_pid_m * num_pid_n

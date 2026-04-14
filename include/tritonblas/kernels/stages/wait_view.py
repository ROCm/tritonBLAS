# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Wait view aggregate for polling readiness flags before loading tiles.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate

from .tile import Tile


@_aggregate
class WaitView:
    """
    Wait view for polling flags before loading tiles.

    Consumer-side counterpart to :class:`SignalView`. Polls a flag until it
    reaches the expected value, then returns. This enables producer-consumer
    patterns like all-gather + matmul where compute must wait for data to
    arrive before entering the GEMM K-loop.
    """
    flag_ptr: tl.tensor
    expected_ptr: tl.tensor

    @triton.constexpr_function
    def __init__(self, flag_ptr, expected_ptr):
        self.flag_ptr = flag_ptr
        self.expected_ptr = expected_ptr

    @triton.jit
    def wait_for_tile(
        self,
        tile: Tile,
        M,
        N,
        num_flags: tl.constexpr,
        map_type: tl.constexpr,
        block_group_m: tl.constexpr,
        block_group_n: tl.constexpr,
        expected_inc: tl.constexpr,
    ):
        """
        Poll the flag corresponding to this tile until it reaches the target.
        """
        num_pid_m = tl.cdiv(M, tile.block_m)
        num_pid_n = tl.cdiv(N, tile.block_n)
        tile_id = tile.pid_m * num_pid_n + tile.pid_n

        if map_type == 1:  # "row"
            flag_id = tile.pid_m
        elif map_type == 2:  # "col"
            flag_id = tile.pid_n
        elif map_type == 3:  # "block"
            group_m = tile.pid_m // block_group_m
            group_n = tile.pid_n // block_group_n
            num_groups_n = tl.cdiv(num_pid_n, block_group_n)
            flag_id = group_m * num_groups_n + group_n
        elif map_type == 4:  # "modulo"
            flag_id = tile_id % num_flags
        else:  # 0 = "identity"
            flag_id = tile_id

        base_value = tl.atomic_add(
            self.expected_ptr + tile_id,
            expected_inc,
            sem="relaxed",
            scope="gpu",
        )
        expected_value = base_value + expected_inc
        while tl.load(self.flag_ptr + flag_id, cache_modifier=".cv", volatile=True) < expected_value:
            pass


@triton.jit
def make_wait_view(flag_ptr, expected_ptr):
    """
    Factory function for WaitView.

    Creates a WaitView aggregate wrapping a flag pointer for consumer-side
    waiting in producer-consumer patterns.
    """
    return WaitView(flag_ptr, expected_ptr)

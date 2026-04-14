# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Signal view aggregate for tracking tile writes to HBM.

Provides :class:`SignalView` aggregate that enables optional tile-level
tracking/signaling by atomically incrementing counters when tiles complete.
Multiple tiles can map to the same signal via coordinate-based formulas.

Example
-------

.. code-block:: python

    # Create signal buffer in Python
    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    signal_buffer = torch.zeros(num_tiles_m, device='cuda', dtype=torch.int32)

    # In kernel: track one signal per row
    signal_view = make_signal_view(signal_buffer)

    # After storing tile result
    tensorC.store(acc, out_tile, signal=signal_view)
    # -> Atomically increments signal[pid_m]
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate

from .tile import Tile


@_aggregate
class SignalView:
    """
    Signal view for tracking tile writes to HBM.

    Maps tiles to signals via coordinate-based formulas. Supports:

    - ``"row"``: All tiles in same M-row share a signal (signal_id = pid_m)
    - ``"col"``: All tiles in same N-column share a signal (signal_id = pid_n)
    - ``"block"``: Tiles in (block_m_group × block_n_group) regions share a signal
    - ``"modulo"``: signal_id = tile_id % num_signals (uniform distribution)
    - ``"identity"``: signal_id = tile_id (one signal per tile)

    Attributes
    ----------
    signal_ptr : tl.tensor
        Pointer to signal array (int32[num_signals])

    Note: Configuration (num_signals, map_type, block_group_m/n) is passed
    as constexpr to apply(), not stored in the aggregate.
    """
    signal_ptr: tl.tensor

    @triton.constexpr_function
    def __init__(self, signal_ptr):
        self.signal_ptr = signal_ptr

    @triton.jit
    def apply(
        self,
        tile: Tile,
        M,
        N,
        num_signals: tl.constexpr,
        map_type: tl.constexpr,
        block_group_m: tl.constexpr,
        block_group_n: tl.constexpr,
    ):
        """
        Increment the signal corresponding to this tile.

        Computes signal_id from tile coordinates (pid_m, pid_n) based on
        the configured mapping type, then atomically increments that signal.

        Map type encoding:
        - 0: identity (one signal per tile)
        - 1: row (all tiles in same M-row share signal)
        - 2: col (all tiles in same N-column share signal)
        - 3: block (tiles in spatial groups share signal)
        - 4: modulo (distribute tiles across signals)
        """
        num_pid_m = tl.cdiv(M, tile.block_m)
        num_pid_n = tl.cdiv(N, tile.block_n)

        if map_type == 1:  # "row"
            signal_id = tile.pid_m
        elif map_type == 2:  # "col"
            signal_id = tile.pid_n
        elif map_type == 3:  # "block"
            group_m = tile.pid_m // block_group_m
            group_n = tile.pid_n // block_group_n
            num_groups_n = tl.cdiv(num_pid_n, block_group_n)
            signal_id = group_m * num_groups_n + group_n
        elif map_type == 4:  # "modulo"
            tile_id = tile.pid_m * num_pid_n + tile.pid_n
            signal_id = tile_id % num_signals
        else:  # 0 = "identity" (default)
            tile_id = tile.pid_m * num_pid_n + tile.pid_n
            signal_id = tile_id

        tl.atomic_add(self.signal_ptr + signal_id, 1, scope="gpu")


@triton.jit
def make_signal_view(signal_ptr):
    """
    Factory function for SignalView.

    Creates a SignalView aggregate wrapping a signal pointer.
    Configuration (num_signals, map_type, etc.) is passed separately
    to apply() as constexpr parameters.
    """
    return SignalView(signal_ptr)

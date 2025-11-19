"""
Indexing operations for Triton GEMM kernels.

This module provides functions for tile coordinate calculation, PID mapping,
and index computation.
"""

from .utils import (
    pid_identity,
    pid_chiplet_chunked,
    dot_acc,
)
from .prologue import grid_setup
from .preamble import idx2coord, tile_coords, compute_scale_indices
from .pid_transforms import chiplet_transform_chunked

__all__ = [
    'pid_identity',
    'pid_chiplet_chunked',
    'dot_acc',
    'grid_setup',
    'idx2coord',
    'tile_coords',
    'compute_scale_indices',
    'chiplet_transform_chunked',
]

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
from .prologue import grid_index
from .preamble import preamble, tile_coords, compute_scale_indices

__all__ = [
    'pid_identity',
    'pid_chiplet_chunked',
    'dot_acc',
    'grid_index',
    'preamble',
    'tile_coords',
    'compute_scale_indices',
]

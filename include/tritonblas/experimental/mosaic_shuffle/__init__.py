"""
Mosaic Shuffle: GPU-friendly tile scheduling strategies for GEMM.

Three strategies with consistent naming:

- **random**: Random permutation of individual output tiles (host-computed lookup table).
  No tile grouping -- every tile is independently shuffled.
  API: ``matmul_random(a, b, c, seed=42)``

- **l2_aware**: Mosaic 2-level L2 tiling with random shuffle of tile groups.
  Inner tiles preserve L2 cache locality; outer groups are randomly permuted
  via a host-computed lookup table.
  API: ``matmul_l2_aware(a, b, c, config=L2AwareConfig(...))``

- **llc_and_l2_aware**: Mosaic 3-level deterministic hierarchy (LayoutRank2Depth3).
  No random permutation -- purely deterministic traversal for LLC + L2 locality.
  API: ``matmul_llc_and_l2_aware(a, b, c, config=LLCAndL2AwareConfig(...))``
"""

from .api import (
    matmul_random,
    matmul_l2_aware,
    matmul_llc_and_l2_aware,
    compute_workgroup_map,
    get_wg_mapping,
)
from .config import L2AwareConfig, LLCAndL2AwareConfig
from .permutation import compute_permutation_table

__all__ = [
    "matmul_random",
    "matmul_l2_aware",
    "matmul_llc_and_l2_aware",
    "compute_workgroup_map",
    "get_wg_mapping",
    "L2AwareConfig",
    "LLCAndL2AwareConfig",
    "compute_permutation_table",
]

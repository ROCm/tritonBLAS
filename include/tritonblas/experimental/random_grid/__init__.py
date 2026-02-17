"""
Random Grid Scheduling for GEMM Kernels

This experimental module provides random workgroup scheduling strategies for GEMM operations.
It includes three scheduling modes:

1. random: L2-aware tile shuffling using Linear Congruential Generator (LCG)
2. workgroup_shuffle: Global tile shuffling for all grid sizes
3. hierarchical: Multi-level cache-aware scheduling with configurable hierarchies

Example usage:
    ```python
    from tritonblas.experimental.random_grid import matmul_random
    import torch

    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(2048, 2048, dtype=torch.bfloat16, device='cuda')

    # Use random L2-aware scheduling
    matmul_random(a, b, c, shuffle_seed=42)
    ```

Note: This is experimental code and the API may change.
"""

from .api import (
    matmul_random,
    matmul_workgroup_shuffle,
    matmul_hierarchical,
    compute_workgroup_map,
    get_wg_mapping,
)
from .config import HierarchicalPersistentConfig
from .utils import choose_lcg_shuffle_params

__all__ = [
    'matmul_random',
    'matmul_workgroup_shuffle',
    'matmul_hierarchical',
    'compute_workgroup_map',
    'get_wg_mapping',
    'HierarchicalPersistentConfig',
    'choose_lcg_shuffle_params',
]

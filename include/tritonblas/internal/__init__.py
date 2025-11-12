"""
Internal modules for tritonblas.

This package contains internal implementation details including:
- Persistent matmul kernels
- StreamK matmul kernels
- PID transformation utilities
- Composable kernel shards
"""

# Import shards submodule to make it accessible
from . import shards

__all__ = ['shards']

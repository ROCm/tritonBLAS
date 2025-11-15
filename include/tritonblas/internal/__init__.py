"""
Internal modules for tritonblas.

This package contains internal implementation details including:
- Persistent matmul kernels
- StreamK matmul kernels
- PID transformation utilities
- Composable kernel components (indexing, algorithms, memory)
"""

# Import submodules to make them accessible
from . import indexing, algorithms, memory

__all__ = ['indexing', 'algorithms', 'memory']

"""
Composable kernel stages for tritonblas.

This package contains reusable building blocks for constructing GEMM kernels:
- algorithms: Core computational stages (gemm_loop, binary ops, unary ops, postprocess)
- indexing: Coordinate mapping and PID transformations
- memory: Load and store operations
"""

# Export stage submodules
from . import algorithms, indexing, memory

__all__ = ['algorithms', 'indexing', 'memory']

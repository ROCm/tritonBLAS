"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages
- persistent_gemm_unrolled: Unrolled persistent GEMM kernel (legacy, for debugging)
- streamk_gemm: Stream-K GEMM kernel for load balancing
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_UNROLLED: Set to '1' or 'true' to use the unrolled persistent kernel instead of the composable stages version
"""

import os

# Check environment variable to determine which persistent kernel to use
_use_unrolled = os.environ.get('TBLAS_USE_UNROLLED', '').lower() in ('1', 'true', 'yes')

if _use_unrolled:
    # Use unrolled version (legacy implementation)
    from .persistent_gemm_unrolled import persistent_matmul
else:
    # Use composable stages version (default)
    from .persistent_gemm import persistent_matmul

# Stream-K kernel is always the same
from .streamk_gemm import streamk_matmul

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'streamk_matmul', 'stages']

import sys
from .matmul import matmul, matmul_a8w8
from .matmul import matmul_lt, matmul_a8w8_lt
from .origami import MatmulHeuristicResult

# Import kernels
from . import kernels

# Import stages from kernels and register as '_' for clean import syntax
from .kernels import stages

# Register stages as '_' in sys.modules
sys.modules['tritonblas._'] = stages
sys.modules['tritonblas._.algorithms'] = stages.algorithms
sys.modules['tritonblas._.indexing'] = stages.indexing
sys.modules['tritonblas._.memory'] = stages.memory

# Also register kernels
sys.modules['tritonblas.kernels'] = kernels

__all__ = [
    'matmul',
    'matmul_a8w8',
    'matmul_lt',
    'matmul_a8w8_lt',
    'MatmulHeuristicResult',
    'kernels',
]

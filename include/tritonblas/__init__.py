import sys
from .matmul import matmul, matmul_a8w8
from .matmul import matmul_lt, matmul_a8w8_lt
from .origami import MatmulHeuristicResult

# Import indexing, algorithms, and memory as submodules
from .internal import indexing, algorithms, memory

# Register in sys.modules so they can be imported as tritonblas.{module}
sys.modules['tritonblas.indexing'] = indexing
sys.modules['tritonblas.algorithms'] = algorithms
sys.modules['tritonblas.memory'] = memory

__all__ = [
    'matmul',
    'matmul_a8w8',
    'matmul_lt',
    'matmul_a8w8_lt',
    'MatmulHeuristicResult',
    'indexing',
    'algorithms',
    'memory',
]

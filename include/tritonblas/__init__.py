import sys
from .matmul import matmul, matmul_a8w8
from .matmul import matmul_lt, matmul_a8w8_lt
from .origami import MatmulHeuristicResult

# Import shards as a submodule to make it accessible as tritonblas.shards
from .internal import shards

# Register shards in sys.modules so it can be imported as tritonblas.shards
sys.modules['tritonblas.shards'] = shards

__all__ = [
    'matmul',
    'matmul_a8w8',
    'matmul_lt',
    'matmul_a8w8_lt',
    'MatmulHeuristicResult',
    'shards',
]

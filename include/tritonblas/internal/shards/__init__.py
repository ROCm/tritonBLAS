"""
Composable shards for Triton GEMM kernels.

This module provides reusable components (shards) that can be composed to build
custom GEMM kernel variants with minimal code duplication. Each shard handles
a specific phase of the GEMM computation:

- utils: Utility functions (PID mapping, dot accumulation, block loading)
- prologue: Setup and initialization (global prologue, tile coordinates, base pointers)
- gemm_loop: Main matrix multiplication loop with K-dimension tiling
- epilogue: Post-processing (quantization scaling, bias addition, output storage)

Example usage:
    from tritonblas.shards import prologue_global, gemm_loop_tile, epilogue_tile
    
    @triton.jit
    def my_custom_matmul_kernel(...):
        pid, num_pid_m, num_pid_n, total_tiles = prologue_global(...)
        for tile_id in range(pid, total_tiles, NUM_SMS):
            acc, rm, rn, pid_m, pid_n = gemm_loop_tile(...)
            epilogue_tile(...)
"""

# Utility functions
from .utils import (
    pid_identity,
    pid_chiplet_chunked,
    dot_acc,
    load_block,
)

# Prologue functions
from .prologue import (
    prologue_global,
    tile_coords,
    make_bases,
)

# GEMM loop functions
from .gemm_loop import (
    gemm_loop_tile,
)

# Epilogue functions
from .epilogue import (
    apply_quant_scales,
    add_bias,
    store_tile,
    epilogue_tile,
)

__all__ = [
    # Utils
    'pid_identity',
    'pid_chiplet_chunked',
    'dot_acc',
    'load_block',
    # Prologue
    'prologue_global',
    'tile_coords',
    'make_bases',
    # GEMM loop
    'gemm_loop_tile',
    # Epilogue
    'apply_quant_scales',
    'add_bias',
    'store_tile',
    'epilogue_tile',
]

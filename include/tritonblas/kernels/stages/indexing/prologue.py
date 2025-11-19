"""
Prologue functions for composable Triton GEMM kernels.
Handles global setup and tile coordinate calculation.
"""
import triton
import triton.language as tl

from .utils import pid_identity, pid_chiplet_chunked


@triton.jit
def grid_setup(
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    USE_CHIPLET_PID: tl.constexpr,
):
    """
    Set up grid dimensions and get program ID.
    
    Args:
        M, N, K: Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
        NUM_SMS: Number of streaming multiprocessors
        NUM_XCDS: Number of chiplets/XCDs
        CHUNK_SIZE: Chunk size for chiplet mapping
        USE_CHIPLET_PID: Whether to use chiplet-aware PID mapping
    
    Returns:
        Tuple of (pid, num_pid_m, num_pid_n, total_tiles)
    """
    pid = tl.program_id(0)
    pid = pid_chiplet_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE) if USE_CHIPLET_PID \
          else pid_identity(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    return pid, num_pid_m, num_pid_n, total_tiles

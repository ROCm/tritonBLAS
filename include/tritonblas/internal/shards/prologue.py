"""
Prologue functions for composable Triton GEMM kernels.
Handles global setup, tile coordinate calculation, and base pointer initialization.
"""
import triton
import triton.language as tl

from .utils import pid_identity, pid_chiplet_chunked


@triton.jit
def prologue_global(
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    USE_CHIPLET_PID: tl.constexpr,
):
    """
    Global prologue: get program ID and compute tile grid dimensions.
    
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


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """
    Compute tile coordinates (pid_m, pid_n) from a linear tile ID.
    Uses grouped ordering for better L2 cache locality.
    
    Args:
        tile_id: Linear tile index
        num_pid_m: Number of tiles in M dimension
        num_pid_n: Number of tiles in N dimension
        GROUP_SIZE_M: Number of M tiles to group together
    
    Returns:
        Tuple of (pid_m, pid_n)
    """
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def make_bases(
    A, B, pid_m, pid_n,
    M, N,
    stride_am, stride_bn,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute base pointers for A and B tiles, along with row/column indices.
    
    Args:
        A, B: Input matrix pointers
        pid_m, pid_n: Tile coordinates
        M, N: Matrix dimensions
        stride_am, stride_bn: Strides for A (M dim) and B (N dim)
        stride_ak, stride_bk: Strides for A and B in K dimension
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes
    
    Returns:
        Tuple of (rm, rn, A_BASE, B_BASE) where:
            rm: Row indices for this M tile
            rn: Column indices for this N tile
            A_BASE: Base pointer for A tile
            B_BASE: Base pointer for B tile
    """
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    return rm, rn, A_BASE, B_BASE

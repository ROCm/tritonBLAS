"""
Preamble phase for composable Triton GEMM kernels.
Phase 1: Indexing only - determines tile responsibility and initializes accumulator.
"""
import triton
import triton.language as tl


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """
    Compute tile coordinates (output_coord_m, output_coord_n) from a linear tile ID.
    Uses grouped ordering for better L2 cache locality.
    
    Args:
        tile_id: Linear tile index
        num_pid_m: Number of tiles in M dimension
        num_pid_n: Number of tiles in N dimension
        GROUP_SIZE_M: Number of M tiles to group together
    
    Returns:
        Tuple of (output_coord_m, output_coord_n) - the M and N coordinates of the output tile
    """
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    output_coord_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    output_coord_n = (tile_id % num_pid_in_group) // group_size_m
    return output_coord_m, output_coord_n


@triton.jit
def idx2coord(
    tile_id, num_pid_m, num_pid_n,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Convert tile index to coordinates and initialize accumulator.
    
    This phase only does indexing - no pointer math, no loads, no compute.
    
    Args:
        tile_id: Linear tile index
        num_pid_m, num_pid_n: Number of tiles in M and N dimensions
        M, N: Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
        GROUP_SIZE_M: Number of M tiles to group together
    
    Returns:
        Tuple of (output_coord_m, output_coord_n, row_indices, col_indices, acc) where:
            output_coord_m, output_coord_n: M and N coordinates of the output tile
            row_indices: Row indices for this tile [BLOCK_SIZE_M]
            col_indices: Column indices for this tile [BLOCK_SIZE_N]
            acc: Initialized accumulator [BLOCK_SIZE_M, BLOCK_SIZE_N]
    """
    # Compute tile coordinates
    output_coord_m, output_coord_n = tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
    tl.assume(output_coord_m >= 0)
    tl.assume(output_coord_n >= 0)
    
    # Compute logical indices (no pointer math yet)
    row_indices = (output_coord_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    col_indices = (output_coord_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # Apply max_contiguous and multiple_of hints for better codegen
    row_indices = tl.max_contiguous(tl.multiple_of(row_indices, BLOCK_SIZE_M), BLOCK_SIZE_M)
    col_indices = tl.max_contiguous(tl.multiple_of(col_indices, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # Initialize accumulator - always use float32 for compatibility
    # The conversion to output dtype happens in postprocess phase
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    return output_coord_m, output_coord_n, row_indices, col_indices, acc


@triton.jit
def compute_scale_indices(output_coord_m, output_coord_n, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Compute indices for loading quantization scales.
    
    Args:
        output_coord_m, output_coord_n: M and N coordinates of the output tile
        M, N: Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes
    
    Returns:
        Tuple of (row_scale_indices, col_scale_indices)
    """
    row_scale_indices = output_coord_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
    col_scale_indices = output_coord_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
    return row_scale_indices, col_scale_indices

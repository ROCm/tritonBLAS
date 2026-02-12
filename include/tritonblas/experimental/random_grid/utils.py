"""
Utility functions for random grid scheduling.

This module provides Linear Congruential Generator (LCG) parameter selection
for shuffling workgroup assignments in GEMM kernels.
"""

import math
import random
from typing import Optional, Tuple


def _is_power_of_two(x: int) -> bool:
    """Check if x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


def _choose_lcg_params(
    n: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    """
    Choose LCG parameters (a, c) for shuffling n items.

    The LCG formula is: next = (a * current + c) % n

    For power-of-2 n:
        - a must be odd and not equal to 1 (prefers a ≡ 5 mod 4)
        - c can be any value in [0, n)

    For non-power-of-2 n:
        - gcd(a, n) must be 1 and a != 1
        - c can be any value in [0, n)

    Args:
        n: Number of items to shuffle (must be > 1)
        seed: Optional random seed
        rng: Optional random number generator

    Returns:
        Tuple of (a, c) LCG parameters

    Raises:
        ValueError: If n <= 1 or no valid parameters exist
    """
    if n <= 1:
        raise ValueError("shuffle requires at least two tiles in the grid")

    rng = rng or (random.Random(seed) if seed is not None else random.Random())

    if _is_power_of_two(n):
        # For power-of-2, prefer a ≡ 5 mod 4 (a = 5, 9, 13, ...)
        valid_as = [a for a in range(5, n, 4)]
        if not valid_as:
            # Fallback: any odd number != 1
            valid_as = [a for a in range(1, n, 2) if a != 1]
        valid_cs = list(range(n))
    else:
        # For non-power-of-2, require gcd(a, n) = 1 and a != 1
        valid_as = [a for a in range(1, n) if math.gcd(a, n) == 1 and a != 1]
        if not valid_as:
            # Fallback: allow a = 1 if nothing else works
            valid_as = [a for a in range(1, n) if math.gcd(a, n) == 1]
        valid_cs = list(range(n))

    if not valid_as or not valid_cs:
        raise ValueError("No valid LCG parameters for given n")

    a = rng.choice(valid_as)
    c = rng.choice(valid_cs)
    return a, c


def choose_lcg_shuffle_params(
    num_tiles: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    """
    Public API for choosing LCG shuffle parameters.

    Args:
        num_tiles: Number of tiles to shuffle (must be > 1)
        seed: Optional random seed for reproducibility
        rng: Optional random number generator

    Returns:
        Tuple of (a, c) LCG parameters suitable for shuffling num_tiles items

    Example:
        >>> a, c = choose_lcg_shuffle_params(num_tiles=256, seed=42)
        >>> # Use a and c in LCG: shuffled_id = (a * tile_id + c) % num_tiles
    """
    return _choose_lcg_params(num_tiles, seed=seed, rng=rng)


def _choose_lcg_params_allow_single_tile(
    num_tiles: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    """
    Choose LCG parameters, returning (0, 0) for single-tile grids.

    This variant is used for workgroup_shuffle mode which needs to handle
    grids with only a single tile gracefully.

    Args:
        num_tiles: Number of tiles to shuffle
        seed: Optional random seed
        rng: Optional random number generator

    Returns:
        Tuple of (a, c) LCG parameters, or (0, 0) if num_tiles <= 1
    """
    if num_tiles <= 1:
        return 0, 0
    return _choose_lcg_params(num_tiles, seed=seed, rng=rng)


def _count_quantized_l2_tiles(num_pid_m: int, num_pid_n: int, tile_dim: int) -> int:
    """
    Count the number of L2 tiles in the quantized region of the grid.

    The grid is quantized to the nearest multiple of tile_dim in both dimensions.
    This is used to determine if random L2 shuffling is applicable.

    Args:
        num_pid_m: Number of tiles in M dimension
        num_pid_n: Number of tiles in N dimension
        tile_dim: L2 tile dimension (typically GROUP_SIZE_M)

    Returns:
        Number of L2 tiles in the quantized region
    """
    if tile_dim <= 0:
        return 0
    quantized_m = (num_pid_m // tile_dim) * tile_dim
    quantized_n = (num_pid_n // tile_dim) * tile_dim
    tiles_per_row = quantized_n // tile_dim
    tiles_per_col = quantized_m // tile_dim
    return tiles_per_row * tiles_per_col

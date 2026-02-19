"""
Configuration dataclasses for mosaic shuffle scheduling strategies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class L2AwareConfig:
    """
    Configuration for L2-aware tile scheduling with Feistel shuffle.

    Tiles in the output grid are grouped into rectangular L2 tiles of shape
    (tile_y x tile_x). The inner ordering within each L2 tile is configurable.
    L2 tile groups are then randomly permuted using a Feistel network seeded
    by `seed`, while preserving intra-tile locality.

    Attributes:
        tile_y: L2 tile height in output tiles (not elements)
        tile_x: L2 tile width in output tiles
        inner_ordering: Traversal order within each L2 tile (0=row_major, 1=col_major)
        seed: Feistel permutation seed for reproducibility
    """
    tile_y: int
    tile_x: int
    inner_ordering: int = 0
    seed: int = 42

    def __post_init__(self):
        if self.tile_y <= 0:
            raise ValueError(f"tile_y must be > 0, got {self.tile_y}")
        if self.tile_x <= 0:
            raise ValueError(f"tile_x must be > 0, got {self.tile_x}")
        if self.inner_ordering not in (0, 1):
            raise ValueError(f"inner_ordering must be 0 (row_major) or 1 (col_major), got {self.inner_ordering}")


@dataclass(frozen=True)
class LLCAndL2AwareConfig:
    """
    Configuration for 3-level deterministic hierarchical scheduling.

    Mirrors mosaic's LayoutRank2Depth3: three nested levels of tiling with
    configurable orderings at each level. No random permutation -- purely
    deterministic traversal optimized for LLC and L2 cache locality.

    Level hierarchy (inner to outer):
        - Level 2 (inner/L2): L2Y x L2X tiles, ordering2
        - Level 1 (middle/LLC): L3Y x L3X tiles, ordering1
        - Level 0 (outer): computed from grid / (L2 * L3), ordering0

    Attributes:
        L2Y: Inner (L2) tile height
        L2X: Inner (L2) tile width
        L3Y: Middle (LLC) tile height
        L3X: Middle (LLC) tile width
        ordering0: Outer level traversal order (0=row_major, 1=col_major)
        ordering1: Middle (LLC) level traversal order
        ordering2: Inner (L2) level traversal order
    """
    L2Y: int
    L2X: int
    L3Y: int
    L3X: int
    ordering0: int
    ordering1: int
    ordering2: int

    def __post_init__(self):
        for name in ("L2Y", "L2X", "L3Y", "L3X"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        for name in ("ordering0", "ordering1", "ordering2"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 (row_major) or 1 (col_major), got {value}")

    @property
    def chunk_size(self) -> int:
        """Number of tiles per L2 chunk (for XCD chunking)."""
        return self.L2Y * self.L2X

    def to_kernel_kwargs(self) -> dict:
        return {
            "ordering0": self.ordering0,
            "ordering1": self.ordering1,
            "ordering2": self.ordering2,
            "L3Y": self.L3Y,
            "L3X": self.L3X,
            "L2Y": self.L2Y,
            "L2X": self.L2X,
            "chunk_size": self.chunk_size,
        }

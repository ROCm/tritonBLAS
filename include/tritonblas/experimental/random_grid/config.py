"""
Configuration dataclasses for random grid scheduling.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HierarchicalPersistentConfig:
    """
    Configuration for hierarchical cache-aware GEMM scheduling.

    This config defines a 3-level hierarchy for tile traversal:
    - Level 3 (L3): Coarsest level, typically cache-aware blocks
    - Level 2 (L2): Mid-level tiles
    - Level 1: Finest level, individual work tiles

    Each level has configurable traversal order (ordering0/1/2).

    Attributes:
        ordering0: Traversal order at level 0 (finest)
        ordering1: Traversal order at level 1 (mid)
        ordering2: Traversal order at level 2 (coarsest)
        L3Y: Level 3 tile height
        L3X: Level 3 tile width
        L2Y: Level 2 tile height
        L2X: Level 2 tile width

    Example:
        >>> config = HierarchicalPersistentConfig(
        ...     ordering0=0, ordering1=1, ordering2=2,
        ...     L3Y=4, L3X=4, L2Y=8, L2X=8
        ... )
        >>> print(config.chunk_size)  # L2Y * L2X = 64
    """

    ordering0: int
    ordering1: int
    ordering2: int
    L3Y: int
    L3X: int
    L2Y: int
    L2X: int

    def __post_init__(self):
        """Validate that all dimension fields are positive."""
        dim_fields = ("L3Y", "L3X", "L2Y", "L2X")
        for name in dim_fields:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0 for hierarchical schedule (got {value}).")

    @property
    def chunk_size(self) -> int:
        """
        Compute chunk size for persistent scheduling.

        Returns:
            L2Y * L2X (number of tiles per L2 chunk)
        """
        return self.L2Y * self.L2X

    def to_kernel_kwargs(self) -> Dict[str, int]:
        """
        Convert config to kernel keyword arguments.

        Returns:
            Dictionary of kernel parameters
        """
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

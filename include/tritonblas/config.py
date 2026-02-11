import torch


class MatmulConfig:
    """
    Pre-allocated GPU buffers for GEMM kernel launches.

    Create via :func:`matmul_preamble` with an ``OrigamiMatmulSelector``.
    Buffer sizes are derived from the selector's tile configuration.

    Attributes:
        device:        ``torch.device`` the buffers live on.
        tile_counter:  ``int32[num_xcds * counters_per_xcd]`` work-stealing counters.
        locks:         ``uint8[sk_grid]`` stream-K lock array.
        P:             ``float32[sk_grid, block_size]`` stream-K partial buffer.
    """

    def __init__(self, device: torch.device, tile_counter: torch.Tensor,
                 locks: torch.Tensor, P: torch.Tensor):
        self.device = device
        self.tile_counter = tile_counter
        self.locks = locks
        self.P = P

    def reset(self, streamk: bool = False, work_stealing: bool = False):
        """Reset mutable state based on the active kernel mode.

        Args:
            streamk:        Zero the stream-K lock array.
            work_stealing:  Zero the work-stealing tile counter.
        """
        if work_stealing:
            self.tile_counter.zero_()
        if streamk:
            self.locks.zero_()
            self.P.zero_()

    def __repr__(self):
        return (
            f"MatmulConfig(device={self.device!r}, "
            f"tile_counter={list(self.tile_counter.shape)}, "
            f"locks={list(self.locks.shape)}, "
            f"P={list(self.P.shape)})"
        )


def matmul_preamble(selector, device: torch.device = None) -> MatmulConfig:
    """
    Allocate all GPU-side buffers needed by the tritonBLAS GEMM kernels.

    Call this once per problem shape (or once with the largest expected shape)
    and pass the returned config into ``matmul_lt``, ``matmul_a8w8_lt``, etc.

    Args:
        selector:  An ``OrigamiMatmulSelector`` providing tile sizes, XCD count,
                   stream-K grid, and ``COUNTERS_PER_XCD``.
        device:    ``torch.device`` for buffer allocation (default: current CUDA device).

    Returns:
        A :class:`MatmulConfig` ready for kernel launches.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    num_xcds = selector._hardware.NUM_XCD
    counters_per_xcd = selector.COUNTERS_PER_XCD
    block_size = selector.block_m * selector.block_n
    sk_grid = selector.sk_grid

    tile_counter = torch.zeros(num_xcds * counters_per_xcd, device=device, dtype=torch.int32)
    locks = torch.zeros(sk_grid, device=device, dtype=torch.uint8)
    P = torch.empty(sk_grid, block_size, device=device, dtype=torch.float32)

    return MatmulConfig(device=device, tile_counter=tile_counter, locks=locks, P=P)

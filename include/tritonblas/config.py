import torch

# 256-byte separation between atomic counters to avoid false sharing
# across L2 cache lines.  Each int32 is 4 bytes -> stride = 256 / 4 = 64 elements.
COUNTER_STRIDE = 64

MAX_SK_TILES = 512


class MatmulConfig:
    """
    Pre-allocated GPU buffers for GEMM kernel launches.

    Create via :func:`matmul_preamble` with an ``OrigamiMatmulSelector``.
    Buffer sizes are derived from the selector's tile configuration.

    Attributes:
        device:           ``torch.device`` the buffers live on.
        tile_counter:     ``int32[num_counters * COUNTER_STRIDE]`` work-stealing
                          counters, padded to 256B per slot to avoid false sharing.
        global_counter:   ``int32[COUNTER_STRIDE]`` single global counter for
                          hierarchical mode's Level 2 fallback pool.
        mask:             ``int32[N_CU]`` per-CU enable mask (1=active, 0=skip).
        locks:            ``uint8[sk_grid]`` stream-K lock array.
        P:                ``float32[sk_grid, block_size]`` stream-K partial buffer.
        sk_iter_counter:  ``int32[COUNTER_STRIDE]`` global atomic for dynamic SK.
        sk_locks:         ``int32[MAX_SK_TILES]`` per-tile spin-lock.
        sk_done:          ``int32[MAX_SK_TILES]`` per-tile K-iteration completion.
        sk_P:             ``float32[MAX_SK_TILES, block_size]`` per-tile partial acc.
    """

    def __init__(self, device: torch.device, tile_counter: torch.Tensor,
                 streamk_tile_counter: torch.Tensor, locks: torch.Tensor,
                 P: torch.Tensor, global_atomic: bool = False,
                 neighbor_stealing: bool = False,
                 global_counter: torch.Tensor = None,
                 mask: torch.Tensor = None,
                 sk_iter_counter: torch.Tensor = None,
                 sk_locks: torch.Tensor = None,
                 sk_done: torch.Tensor = None,
                 sk_P: torch.Tensor = None):
        self.device = device
        self.tile_counter = tile_counter
        self.streamk_tile_counter = streamk_tile_counter
        self.locks = locks
        self.P = P
        self.global_atomic = global_atomic
        self.neighbor_stealing = neighbor_stealing
        self.global_counter = global_counter
        self.mask = mask
        self.sk_iter_counter = sk_iter_counter
        self.sk_locks = sk_locks
        self.sk_done = sk_done
        self.sk_P = sk_P

    def reset(self, streamk: bool = False, work_stealing: bool = False):
        """Reset mutable state based on the active kernel mode.

        Args:
            streamk:        Zero the stream-K lock array.
            work_stealing:  Zero the work-stealing tile counter(s).
        """
        if work_stealing:
            self.tile_counter.zero_()
            if self.global_counter is not None:
                self.global_counter.zero_()
        if streamk:
            self.locks.zero_()
            self.P.zero_()
            if self.sk_iter_counter is not None:
                self.sk_iter_counter.zero_()
            if self.sk_locks is not None:
                self.sk_locks.zero_()
            if self.sk_done is not None:
                self.sk_done.zero_()
            if self.sk_P is not None:
                self.sk_P.zero_()

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

    num_counters = num_xcds * counters_per_xcd
    tile_counter = torch.zeros(num_counters * COUNTER_STRIDE, device=device, dtype=torch.int32)
    streamk_tile_counter = torch.zeros(num_counters * COUNTER_STRIDE, device=device, dtype=torch.int32)
    locks = torch.zeros(sk_grid, device=device, dtype=torch.uint8)
    P = torch.empty(sk_grid, block_size, device=device, dtype=torch.float32)

    global_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)

    n_cu = selector._N_CU
    active_cu = selector._ACTIVE_CU
    mask = torch.ones(n_cu, dtype=torch.int32, device=device)
    if active_cu < n_cu:
        mask[active_cu:] = 0

    max_sk = MAX_SK_TILES
    sk_iter_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
    sk_locks = torch.zeros(max_sk, device=device, dtype=torch.int32)
    sk_done = torch.zeros(max_sk, device=device, dtype=torch.int32)
    sk_P = torch.zeros(max_sk, block_size, device=device, dtype=torch.float32)

    return MatmulConfig(device=device, tile_counter=tile_counter,
                        streamk_tile_counter=streamk_tile_counter,
                        locks=locks, P=P, mask=mask,
                        global_counter=global_counter,
                        sk_iter_counter=sk_iter_counter,
                        sk_locks=sk_locks, sk_done=sk_done, sk_P=sk_P)

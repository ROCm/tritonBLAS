from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

# 256-byte separation between atomic counters to avoid false sharing
# across L2 cache lines.  Each int32 is 4 bytes -> stride = 256 / 4 = 64 elements.
COUNTER_STRIDE = 64

MAX_SK_TILES = 512

# Map type encoding shared by the epilogue (counter) and prologue (wait) sync.
# 0=identity (one per tile), 1=row, 2=col, 3=block, 4=launch_wave
MAP_TYPE_ENCODING = {"identity": 0, "row": 1, "col": 2, "block": 3, "launch_wave": 4}


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


def _validate_sync_buffer(buffer: torch.Tensor, name: str) -> None:
    if not isinstance(buffer, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if buffer.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(buffer.shape)}")
    if buffer.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {buffer.dtype}")
    if buffer.numel() <= 0:
        raise ValueError(f"{name} must contain at least one element")
    if buffer.device.type != "cuda":
        raise ValueError(f"{name} must live on a CUDA device, got {buffer.device}")


class _TileMapMixin:
    """Tile-to-slot mapping shared by :class:`EpilogueConfig` and :class:`PrologueConfig`."""

    map_type: str
    block_group_m: int
    block_group_n: int

    def _validate_map_type(self) -> None:
        if self.map_type not in MAP_TYPE_ENCODING:
            raise ValueError(
                f"Invalid map_type: {self.map_type}. Must be one of {list(MAP_TYPE_ENCODING.keys())}"
            )

    def required_size(self, M: int, N: int, block_m: int, block_n: int) -> int:
        """Minimum number of counters/flags needed for an ``M x N`` problem with this mapping."""
        num_tiles_m = (M + block_m - 1) // block_m
        num_tiles_n = (N + block_n - 1) // block_n

        if self.map_type == "row":
            return num_tiles_m
        if self.map_type == "col":
            return num_tiles_n
        if self.map_type == "block":
            num_groups_m = (num_tiles_m + self.block_group_m - 1) // self.block_group_m
            num_groups_n = (num_tiles_n + self.block_group_n - 1) // self.block_group_n
            return num_groups_m * num_groups_n
        if self.map_type == "identity":
            return num_tiles_m * num_tiles_n
        if self.map_type == "launch_wave":
            return (num_tiles_m * num_tiles_n + self.block_group_m - 1) // self.block_group_m
        raise ValueError(f"Unknown map_type: {self.map_type}")


@dataclass(eq=False)
class EpilogueConfig(_TileMapMixin):
    """
    Tile-completion counters updated in the persistent GEMM epilogue.

    Packages a user-managed counter buffer with tile-to-counter mapping
    metadata. Does not allocate: reuse the same buffer across launches and call
    ``counter_buffer.zero_()`` to reset counts. Buffer size is validated at
    launch time, once matrix dimensions are known; use :meth:`required_size`
    to pre-validate.

    Attributes:
        counter_buffer: ``int32`` CUDA tensor storing the counters.
        map_type:       Mapping strategy (default ``"identity"``).

                        - ``"row"``: all tiles in the same M-row share a counter
                        - ``"col"``: all tiles in the same N-column share a counter
                        - ``"block"``: tiles in spatial groups share a counter
                        - ``"launch_wave"``: counter id = persistent launch-wave iteration
                        - ``"identity"``: one counter per tile
        block_group_m:  ``"block"``: tiles per group in M. ``"launch_wave"``: hardware
                        wave size (active CUs), used to size the launch-wave counters.
        block_group_n:  ``"block"``: tiles per group in N.

    Example:
        >>> counters = torch.zeros(num_tiles_m, device="cuda", dtype=torch.int32)
        >>> epilogue = EpilogueConfig(counters, map_type="row")
        >>> persistent_matmul_lt(a, b, c, selector, epilogue_config=epilogue)
    """

    counter_buffer: torch.Tensor
    map_type: str = "identity"
    block_group_m: int = 1
    block_group_n: int = 1

    def __post_init__(self):
        _validate_sync_buffer(self.counter_buffer, "counter_buffer")
        self._validate_map_type()

    @property
    def num_counters(self) -> int:
        return self.counter_buffer.numel()

    @staticmethod
    def resolve_launch_args(
        cfg: Optional["EpilogueConfig"], M: int, N: int, block_m: int, block_n: int
    ) -> Tuple[Tuple[Optional[torch.Tensor]], Dict[str, Any]]:
        """
        Validate ``cfg`` against the problem size and return the kernel arguments.

        ``cfg=None`` disables the counters: no pointer and empty constexprs, so the kernel
        defaults apply. Returns ``((counter_ptr,), constexprs)``.
        """
        if cfg is None:
            return (None,), {}

        required = cfg.required_size(M, N, block_m, block_n)
        if cfg.num_counters < required:
            raise ValueError(
                f"counter_buffer too small: has {cfg.num_counters} elements, "
                f"but requires {required} for M={M}, N={N}, "
                f"BLOCK_M={block_m}, BLOCK_N={block_n}, map_type='{cfg.map_type}'"
            )
        return (cfg.counter_buffer,), dict(
            ENABLE_COUNTER=True,
            COUNTER_NUM=cfg.num_counters,
            COUNTER_MAP_TYPE=MAP_TYPE_ENCODING[cfg.map_type],
            COUNTER_BLOCK_GROUP_M=cfg.block_group_m,
            COUNTER_BLOCK_GROUP_N=cfg.block_group_n,
        )


@dataclass(eq=False)
class PrologueConfig(_TileMapMixin):
    """
    Producer-consumer readiness flags waited on in the persistent GEMM prologue.

    Packages readiness buffers with the mapping metadata used by ``WaitView``.
    The mapping modes mirror :class:`EpilogueConfig` so producers and consumers
    can share the same tile-to-flag assignment. Buffer sizes are validated at
    launch time; use :meth:`required_size` to pre-validate.

    Attributes:
        wait_buffer:     ``int32`` CUDA tensor of readiness flags.
        expected_buffer: ``int32`` CUDA tensor with one expected value per tile.
        expected_inc:    Increment applied to the expected value per wait.
        map_type, block_group_m, block_group_n: see :class:`EpilogueConfig`.
    """

    wait_buffer: torch.Tensor
    expected_buffer: torch.Tensor
    expected_inc: int = 1
    map_type: str = "identity"
    block_group_m: int = 1
    block_group_n: int = 1

    def __post_init__(self):
        _validate_sync_buffer(self.wait_buffer, "wait_buffer")
        _validate_sync_buffer(self.expected_buffer, "expected_buffer")
        if self.wait_buffer.device != self.expected_buffer.device:
            raise ValueError(
                "wait_buffer and expected_buffer must live on the same device, "
                f"got {self.wait_buffer.device} and {self.expected_buffer.device}"
            )
        self._validate_map_type()

    @property
    def num_waits(self) -> int:
        return self.wait_buffer.numel()

    @staticmethod
    def resolve_launch_args(
        cfg: Optional["PrologueConfig"], M: int, N: int, block_m: int, block_n: int
    ) -> Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], Dict[str, Any]]:
        """
        Validate ``cfg`` against the problem size and return the kernel arguments.

        ``cfg=None`` disables the waits: no pointers and empty constexprs, so the kernel
        defaults apply. Returns ``((wait_ptr, expected_ptr), constexprs)``.
        """
        if cfg is None:
            return (None, None), {}

        required = cfg.required_size(M, N, block_m, block_n)
        if cfg.num_waits < required:
            raise ValueError(
                f"wait_buffer too small: has {cfg.num_waits} elements, "
                f"but requires {required} for M={M}, N={N}, "
                f"BLOCK_M={block_m}, BLOCK_N={block_n}, map_type='{cfg.map_type}'"
            )
        # expected_buffer always needs one entry per tile, regardless of map_type
        total_tiles = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
        if cfg.expected_buffer.numel() < total_tiles:
            raise ValueError(
                f"expected_buffer too small: has {cfg.expected_buffer.numel()} elements, "
                f"but requires {total_tiles} (one per tile) for M={M}, N={N}, "
                f"BLOCK_M={block_m}, BLOCK_N={block_n}"
            )
        return (cfg.wait_buffer, cfg.expected_buffer), dict(
            ENABLE_WAIT=True,
            WAIT_NUM=cfg.num_waits,
            WAIT_MAP_TYPE=MAP_TYPE_ENCODING[cfg.map_type],
            WAIT_BLOCK_GROUP_M=cfg.block_group_m,
            WAIT_BLOCK_GROUP_N=cfg.block_group_n,
            WAIT_EXPECTED_INC=cfg.expected_inc,
        )

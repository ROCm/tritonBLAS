# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
The order a matmul will visit its output tiles in, exposed to the caller.

A GEMM's tile order is chosen for L2 reuse and then kept inside the kernel. Anything that has to
act on it — a triggered collective arming descriptors, a fused-op scheduler, a cost model — must
re-derive it, and nothing enforces that the re-derivations agree. They do not fail loudly when
they disagree: the work still completes and the result is merely slow.

``schedule()`` publishes the complete producer plan instead. It builds the selector by the same
path ``matmul()`` does and mirrors :class:`~tritonblas.kernels.stages.schedule.ScheduleContext`,
so what it reports is the order the kernel will actually run rather than a reconstruction of it.
The plan also carries a coordinate ``tile_order`` and a separate ``signal_plan`` derived from final
producers. A triggered consumer therefore reads the schedule instead of independently reconstructing
any part of the firing order.

    >>> sched = tritonblas.schedule(a, b)
    >>> sched.tile_of_wg[0]            # linear tile id computed by program 0
    >>> sched.tile_order[0]             # first output (tile_m, tile_n)
    >>> sched.signal_plan.arms[0]       # first grouped SignalArm
    >>> sched.first_ready_fraction(num_cus=304)

Returns ``None`` when there is no static order to publish, which is a normal answer:

``work_stealing``
    tiles are claimed from counters at runtime; the assignment differs per launch.
``enable_streamk``
    a tile is not owned by one workgroup, so there is no permutation over tiles at all.

Callers must handle ``None`` rather than substitute a guess. For a triggered collective the
correct response is to not attempt gated overlap.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

__all__ = [
    "SignalArm",
    "SignalPlan",
    "TileSchedule",
    "schedule",
    "schedule_for_problem",
    "schedule_for_selector",
]


def _chiplet_transform_chunked(pid, num_workgroups, num_xcds, chunk_size):
    """Host mirror of ``kernels.stages.indexing.pid_transforms.chiplet_transform_chunked``."""
    if num_xcds <= 1:
        return pid
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    xcd = pid % num_xcds
    return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk


def _group_m_swizzle(tile_id, num_pid_m, num_pid_n, group_size_m):
    """Host mirror of ``ScheduleContext.get_tile_from_idx``."""
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    span = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % span)
    pid_n = (tile_id % num_pid_in_group) // span
    return pid_m, pid_n


@dataclass(frozen=True)
class SignalArm:
    """
    One triggered-consumer queue entry: a gate, its threshold, and the tiles it releases.

    ``signal_id`` identifies the gate that the consumer polls. ``producers`` is the number of GEMM
    tiles that must increment that gate before it opens. ``tiles`` contains the ``(tile_m, tile_n)``
    coordinates whose communication descriptors can run after it opens.

    The coordinates describe membership, not production order. For example, a 2x2 group is:

    .. code-block:: python

        SignalArm(
            signal_id=3,
            tiles=((0, 0), (0, 1), (1, 0), (1, 1)),
            producers=4,
        )

    The consumer polls gate 3 for the value 4 and then releases descriptors for those four tiles.
    """

    signal_id: int
    tiles: tuple[tuple[int, int], ...]
    producers: int


@dataclass(frozen=True)
class SignalPlan:
    """
    Immutable signalling policy layered on top of a :class:`TileSchedule`.

    ``signal_of_tile`` is indexed by row-major linear tile id, not dispatch position. If a 2x2
    tile grid groups each tile column onto one signal, the map is ``(0, 1, 0, 1)``:

    .. code-block:: text

        tile coordinate     linear tile id     signal id
        (0, 0)                     0                0
        (0, 1)                     1                1
        (1, 0)                     2                0
        (1, 1)                     3                1

    ``arms`` is in the order the consumer should queue those signals. Each entry carries the gate
    id, threshold, and coordinates, so the consumer does not need to reconstruct grouping or readiness
    from the GEMM permutation.
    """

    signal_of_tile: tuple[int, ...]
    arms: tuple[SignalArm, ...]

    @property
    def producers(self):
        """
        Return producer thresholds indexed by signal id.

        The tuple index is the gate id and the value is the number of tile completions required to
        open that gate. For the two-column example in :class:`SignalPlan`, this returns ``(2, 2)``.
        Arm order does not affect this indexing: even if signal 1 is armed first, ``producers[1]``
        is still signal 1's threshold.
        """
        out = [0] * len(self.arms)
        for arm in self.arms:
            out[arm.signal_id] = arm.producers
        return tuple(out)


@dataclass(frozen=True)
class TileSchedule:
    """
    Complete static producer plan shared by a matmul and its triggered consumer.

    There are two tile representations:

    * ``tile_of_wg[position]`` is a compact row-major linear tile id. The tuple index is dispatch
      position, so ``tile_of_wg[0]`` is the first tile dispatched.
    * :attr:`tile_order` expresses the same permutation as explicit ``(tile_m, tile_n)``
      coordinates. ``tile_m`` selects a block along matrix rows (M), and ``tile_n`` selects a block
      along matrix columns (N).

    For a 2x2 grid, ``tile_of_wg == (0, 2, 1, 3)`` and
    ``tile_order == ((0, 0), (1, 0), (0, 1), (1, 1))`` both mean column-major traversal of the tile
    grid: visit the two tiles in tile column 0 before the two tiles in tile column 1.

    :attr:`signal_plan` is separate because multiple signalling layouts can be applied to the same
    GEMM tile order. Call :meth:`with_signal_layout` to derive an immutable plan for one layout.
    """

    tiles_m: int
    tiles_n: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_xcds: int
    chunk_size: int
    #: Dispatch position -> linear tile id ``pid_m * tiles_n + pid_n``.
    tile_of_wg: tuple[int, ...] = field(repr=False)
    signal_plan: SignalPlan = field(repr=False)

    @property
    def num_tiles(self):
        """Total number of output tiles, equal to ``grid_m * grid_n``."""
        return self.tiles_m * self.tiles_n

    @property
    def grid_m(self):
        """
        Number of output tiles along matrix dimension M.

        Tile row ``tile_m`` is valid when ``0 <= tile_m < grid_m``.
        """
        return self.tiles_m

    @property
    def grid_n(self):
        """
        Number of output tiles along matrix dimension N.

        Tile column ``tile_n`` is valid when ``0 <= tile_n < grid_n``.
        """
        return self.tiles_n

    @property
    def num_signals(self):
        """Number of gates in the current signal plan."""
        return len(self.signal_plan.arms)

    @property
    def tile_order(self):
        """
        Return output-tile coordinates in GEMM dispatch order.

        The tuple index is the dispatch position and the value is the output coordinate produced at
        that position. For example:

        .. code-block:: python

            schedule.tile_order == (
                (0, 0),  # dispatched first
                (1, 0),  # dispatched second
                (0, 1),  # dispatched third
                (1, 1),  # dispatched fourth
            )

        This is a column-major traversal of a 2x2 *tile grid*: M changes fastest. It does not claim
        that the matrix storage is column-major, nor that concurrently running workgroups complete
        in a deterministic total order. It is the kernel's static launch/assignment order.
        """
        return tuple((tile // self.grid_n, tile % self.grid_n) for tile in self.tile_of_wg)

    @property
    def arming_order(self):
        """
        Compatibility alias for :attr:`tile_order`.

        New consumers should use ``tile_order`` when they mean the GEMM's tile permutation.
        The triggered consumer's actual gate queue is ``signal_plan.arms`` (also exposed through
        :attr:`signal_arming_order`).
        """
        return self.tile_order

    @property
    def signal_arming_order(self):
        """
        Compatibility alias for ``signal_plan.arms``.

        The tuple index is consumer queue position; the value is a :class:`SignalArm`. Unlike
        :attr:`tile_order`, one entry can cover several output tiles sharing one gate.
        """
        return self.signal_plan.arms

    @property
    def signal_of_tile(self):
        """
        Return the gate id written by each output tile.

        This tuple is indexed by row-major linear tile id
        ``tile_m * grid_n + tile_n``. It is not indexed by dispatch position. The GEMM epilogue
        computes its linear tile id, reads this table, and increments the selected gate.
        """
        return self.signal_plan.signal_of_tile

    @property
    def producers(self):
        """
        Return gate thresholds indexed by signal id.

        ``producers[s]`` equals the number of tiles whose ``signal_of_tile`` entry is ``s``.
        The consumer waits until gate ``s`` has received that many producer increments.
        """
        return self.signal_plan.producers

    def wg_of_tile(self):
        """
        Return the inverse permutation: linear tile id to dispatch position.

        For ``tile_of_wg == (2, 0, 3, 1)``, this returns ``[1, 3, 0, 2]``: linear tile 2 appears at
        dispatch position 0, linear tile 0 at position 1, and so on. This inverse is useful when
        asking when a particular coordinate is expected to become ready.
        """
        out = [0] * self.num_tiles
        for position, tile in enumerate(self.tile_of_wg):
            out[tile] = position
        return out

    def coords(self):
        """
        Compatibility method returning :attr:`tile_order`.

        ``coords()[position]`` is the ``(tile_m, tile_n)`` coordinate assigned at that dispatch
        position.
        """
        return self.tile_order

    def with_signal_layout(self, signal_of_tile):
        """
        Return a copy with thresholds and consumer queue order derived from a tile-to-gate map.

        ``signal_of_tile[t]`` assigns row-major linear tile id ``t`` to a non-negative, densely
        numbered signal. Signal ids must therefore cover exactly ``0..num_signals-1``.

        Consider a 2x2 grid with column-major ``tile_order``:

        .. code-block:: python

            schedule.tile_order
            # ((0, 0), (1, 0), (0, 1), (1, 1))

            grouped = schedule.with_signal_layout((0, 1, 0, 1))

        Linear tile ids are row-major, so ``(0, 1, 0, 1)`` groups coordinates ``(0, 0)`` and
        ``(1, 0)`` onto signal 0, and ``(0, 1)`` and ``(1, 1)`` onto signal 1. The result has
        ``grouped.producers == (2, 2)``. Signal 0's final producer appears before signal 1's final
        producer in ``tile_order``, so ``grouped.signal_plan.arms`` queues signal 0 first.

        This method preserves the GEMM's tile order. It derives all consumer-facing signal data
        here so the consumer does not independently count producers or sort gates.
        """
        if hasattr(signal_of_tile, "tolist"):
            signal_of_tile = signal_of_tile.tolist()
        signal_of_tile = [int(signal) for signal in signal_of_tile]
        if len(signal_of_tile) != self.num_tiles:
            raise ValueError(
                f"signal_of_tile has {len(signal_of_tile)} entries; "
                f"the tile grid has {self.num_tiles}"
            )
        if not signal_of_tile or min(signal_of_tile) < 0:
            raise ValueError("signal ids must be non-negative")

        num_signals = max(signal_of_tile) + 1
        if set(signal_of_tile) != set(range(num_signals)):
            raise ValueError("signal ids must be dense from 0 through num_signals - 1")

        producers = [0] * num_signals
        ready_position = [-1] * num_signals
        tiles_of_signal = [[] for _ in range(num_signals)]
        inverse = self.wg_of_tile()
        for tile, signal in enumerate(signal_of_tile):
            producers[signal] += 1
            ready_position[signal] = max(ready_position[signal], inverse[tile])
            tiles_of_signal[signal].append((tile // self.grid_n, tile % self.grid_n))
        signal_order = sorted(range(num_signals), key=ready_position.__getitem__)
        arms = tuple(
            SignalArm(
                signal_id=signal,
                tiles=tuple(tiles_of_signal[signal]),
                producers=producers[signal],
            )
            for signal in signal_order
        )
        return replace(
            self,
            signal_plan=SignalPlan(tuple(signal_of_tile), arms),
        )

    def finish_wave(self, num_cus, position):
        """
        Which wave the workgroup at ``position`` retires in, given ``num_cus`` compute units.

        The hardware places launch index ``k`` on XCD ``k % num_xcds`` and then on the
        earliest-free CU of that die, so under uniform tile cost the ``j``-th workgroup on a die
        runs in wave ``j // cus_per_xcd``. Dispatch position alone is only a proxy for time when
        tiles greatly outnumber CUs; a grid that fits in one wave has no temporal spread at all.
        """
        cus_per_xcd = max(1, num_cus // self.num_xcds)
        return (position // self.num_xcds) // cus_per_xcd + 1

    def first_ready_fraction(self, num_cus):
        """
        How far into the GEMM the earliest output tile is finished, in ``[0, 1]``.

        ``1 - this`` is the overlap window: an upper bound on how much of a dependent collective
        could be hidden behind this GEMM. **The number to check before building gated overlap at
        all** — a grid that fits in one wave returns 1.0, meaning nothing can be hidden however
        the collective is scheduled.

        ``num_cus`` is required rather than optional. Measured in dispatch positions the answer
        is always 0 (some tile is always first), which says nothing; the question is temporal
        and needs to know how many tiles run at once.
        """
        n = self.num_tiles
        if not n:
            return 0.0
        waves = [self.finish_wave(num_cus, p) for p in range(n)]
        span = max(waves)
        return min(waves) / span if span else 0.0

    def ready_wave_of_tile(self, num_cus):
        """
        Return the estimated completion wave of every linear tile id.

        The output is indexed by row-major linear tile id, even though workgroups are launched in
        ``tile_of_wg`` order. A value of 1 means the tile is assigned in the first CU-resident wave;
        a value of 2 means it waits for one earlier wave on the same XCD, and so on.

        This is an analytical uniform-tile-cost estimate, not a measured timestamp. It captures CU
        concurrency and XCD placement but not runtime variation between tiles.
        """
        out = [0] * self.num_tiles
        for position, tile in enumerate(self.tile_of_wg):
            out[tile] = self.finish_wave(num_cus, position)
        return out

    def ready_wave_of_signal(self, num_cus):
        """
        Return the estimated wave in which each signal receives its final producer.

        The output index is the signal id. If signal 0 groups tiles whose estimated completion waves
        are ``(1, 1, 3)``, then its ready wave is 3. This maximum is the earliest wave in which
        the consumer can expect the corresponding gate to reach ``producers[0]``.
        """
        tile_waves = self.ready_wave_of_tile(num_cus)
        out = [0] * self.num_signals
        for tile, signal in enumerate(self.signal_of_tile):
            out[signal] = max(out[signal], tile_waves[tile])
        return out


def schedule_for_problem(M, N, K, a_dtype, b_dtype, out_dtype, device, *,
                         enable_streamk=False, work_stealing=False, num_stages=2,
                         signal_of_tile=None):
    """
    Build the schedule for explicit GEMM dimensions and dtypes rather than tensor objects.

    This runs the same Origami configuration selection used by ``matmul`` and then publishes the
    selected static tile order. ``M`` and ``N`` determine the output grid; ``K`` and the dtypes can
    change the selected block sizes and therefore also change that grid and order.

    ``signal_of_tile`` is optional. Without it, the result uses one signal per output tile. With it,
    :meth:`TileSchedule.with_signal_layout` derives grouped thresholds and queue order.

    Returns ``None`` for work-stealing or Stream-K because those modes assign work dynamically and
    cannot publish one static tile permutation before launch.
    """
    if work_stealing or enable_streamk:
        return None

    from .origami import OrigamiMatmulSelector

    selector = OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, out_dtype, device,
        streamk=enable_streamk, num_stages=num_stages,
    )
    return schedule_for_selector(M, N, selector, signal_of_tile=signal_of_tile)


def schedule_for_selector(M, N, selector, *, signal_of_tile=None):
    """
    Publish the tile and signal plan for an already-configured matmul selector.

    Use this when the caller already owns the exact selector that will launch the GEMM. Reading its
    effective ``block_m``, ``block_n``, ``group_m``, XCD count, and chunk size avoids selecting the
    kernel twice and guarantees that this object describes that selector's kernel mapping.

    For example, if the selector chooses 128x256 output tiles for ``M=1024, N=1024``, the result has
    ``grid_m == 8`` and ``grid_n == 4``. Every coordinate in ``tile_order`` is in that 8x4 grid.
    """
    block_m, block_n = selector.block_m, selector.block_n
    tiles_m = (M + block_m - 1) // block_m
    tiles_n = (N + block_n - 1) // block_n
    total = tiles_m * tiles_n

    # Exactly `persistent_matmul_lt`'s derivation: `group_m` is the effective mapping (Origami's
    # selection is overwritten by `_select_ws_params`), `num_sms` carries wgmxcc, and the chunk
    # is capped to one XCD's share of the grid.
    group_m = selector.group_m
    num_xcds = selector.num_sms
    chunk_size = group_m * group_m
    if num_xcds and num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total // num_xcds))
    else:
        num_xcds = 1

    tile_of_wg = []
    for pid in range(total):
        tile_id = _chiplet_transform_chunked(pid, total, num_xcds, chunk_size)
        pid_m, pid_n = _group_m_swizzle(tile_id, tiles_m, tiles_n, group_m)
        tile_of_wg.append(pid_m * tiles_n + pid_n)

    tile_schedule = TileSchedule(
        tiles_m=tiles_m, tiles_n=tiles_n,
        block_m=block_m, block_n=block_n, block_k=selector.block_k,
        group_m=group_m, num_xcds=num_xcds, chunk_size=chunk_size,
        tile_of_wg=tuple(tile_of_wg),
        signal_plan=SignalPlan((), ()),
    )
    if signal_of_tile is None:
        signal_of_tile = range(total)
    return tile_schedule.with_signal_layout(signal_of_tile)


def schedule(a, b, *, enable_streamk=False, work_stealing=False,
             signal_of_tile=None) -> Optional[TileSchedule]:
    """
    Return the complete static producer plan for ``matmul(a, b, ...)``.

    Pass the same flags you would pass to :func:`tritonblas.matmul`; the schedule depends on
    them, and on the config Origami selects for this problem. ``signal_of_tile`` may be any
    one-dimensional host or device table; producer thresholds and arming order are derived once
    and returned as part of the schedule.

    With no signal layout, every output tile owns one gate:

    .. code-block:: python

        plan = tritonblas.schedule(a, b)
        plan.signal_of_tile == tuple(range(plan.num_tiles))
        plan.producers == (1,) * plan.num_tiles

    A grouped layout can be supplied immediately or attached later:

    .. code-block:: python

        grouped = tritonblas.schedule(a, b, signal_of_tile=tile_to_gate)
        # Equivalent to:
        grouped = tritonblas.schedule(a, b).with_signal_layout(tile_to_gate)

        for arm in grouped.signal_plan.arms:
            consumer.arm(arm.signal_id, arm.producers, arm.tiles)

    Returns ``None`` when ``enable_streamk`` or ``work_stealing`` is enabled, because those modes do
    not have a fixed pre-launch tile ownership order. A consumer must treat ``None`` as "gated
    overlap unavailable", not substitute a guessed ordering.
    """
    M, K = a.shape
    _, N = b.shape
    return schedule_for_problem(
        M, N, K, a.dtype, b.dtype, a.dtype, a.device,
        enable_streamk=enable_streamk, work_stealing=work_stealing,
        signal_of_tile=signal_of_tile,
    )

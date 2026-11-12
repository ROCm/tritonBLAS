"""
Tests for tritonblas.schedule() — the published tile order.

The point of publishing the order is that something else acts on it: a triggered collective
arming descriptors, a fused-op scheduler, a cost model. None of those fail loudly when the
published order is wrong. The work still completes, the bytes are still correct, and the result
is merely slower — so a permutation that is plausible but not the kernel's is the failure mode
these tests exist to prevent.

The load-bearing test is `test_matches_the_device_scheduler`, which launches a probe kernel
built on the same `ScheduleContext` the GEMM kernels use and compares. Everything else pins
properties that a wrong-but-plausible schedule would still satisfy, so they are necessary and
nowhere near sufficient on their own.
"""

import pytest
import torch
import triton
import triton.language as tl

import tritonblas
from tritonblas.origami import OrigamiMatmulSelector
from tritonblas.schedule import schedule, schedule_for_problem

from conftest import DTYPES

# Shapes chosen so the grid is square, non-square, ragged, single-wave and many-wave. The tile
# size is Origami's choice, so these are problem dims rather than grids.
SCHEDULE_DIMS = [
    (4096, 4096, 8192),
    (8192, 8192, 4096),
    (2048, 8192, 4096),
    (8192, 2048, 4096),
    (1024, 1024, 1024),
    (1536, 3072, 2048),   # not a power of two in either dimension
]

DEVICE = torch.device("cuda:0")


def _schedule(m, n, k, dtype=torch.bfloat16):
    return schedule_for_problem(m, n, k, dtype, dtype, dtype, DEVICE)


def _arming_ids(schedule):
    return tuple(arm.signal_id for arm in schedule.signal_plan.arms)


# ---------------------------------------------------------------------------
# Against the device: the only test that can catch a wrong permutation
# ---------------------------------------------------------------------------


@triton.jit
def _probe_kernel(out_m, out_n, M, N, K,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                  NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr):
    """Record the tile each program would compute, via the scheduler the GEMM kernels use."""
    from tritonblas.kernels.stages.gemm_context import GemmContext
    from tritonblas.kernels.stages.schedule import ScheduleContext

    ctx = GemmContext(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                      NUM_SMS, NUM_XCDS, GROUP_SIZE_M, CHUNK_SIZE)
    sched = ScheduleContext(M, N, K, ctx)
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        tile = sched.get_tile_from_idx(tile_id)
        tl.store(out_m + tl.program_id(0), tile.pid_m)
        tl.store(out_n + tl.program_id(0), tile.pid_n)


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_matches_the_device_scheduler(m, n, k):
    """
    The published order must be the order the kernel runs.

    Launched exactly as `persistent_matmul_lt` launches: grid and NUM_SMS both `total_tiles`,
    which is what `matmul.py` uses on the default path.
    """
    s = _schedule(m, n, k)
    out_m = torch.empty(s.num_tiles, dtype=torch.int32, device=DEVICE)
    out_n = torch.empty(s.num_tiles, dtype=torch.int32, device=DEVICE)
    _probe_kernel[(s.num_tiles,)](
        out_m, out_n, m, n, k,
        BLOCK_SIZE_M=s.block_m, BLOCK_SIZE_N=s.block_n, BLOCK_SIZE_K=s.block_k,
        GROUP_SIZE_M=s.group_m, NUM_SMS=s.num_tiles, NUM_XCDS=s.num_xcds,
        CHUNK_SIZE=s.chunk_size, num_warps=1,
    )
    torch.cuda.synchronize()

    from_device = [int(a) * s.tiles_n + int(b) for a, b in zip(out_m.tolist(), out_n.tolist())]
    assert from_device == s.tile_of_wg


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_launch_parameters_are_the_ones_matmul_uses(m, n, k):
    """
    The order depends on four numbers, and they must be the ones that reach the kernel.

    `group_m` in particular: `OrigamiMatmulSelector` asks Origami for a workgroup mapping and
    then overwrites it in `_select_ws_params`, so the effective value is the property, not
    Origami's selection. `num_sms` carries `wgmxcc` despite the name, and the chunk is capped
    to one XCD's share of the grid — all mirrored from `persistent_matmul_lt`.
    """
    s = _schedule(m, n, k)
    selector = OrigamiMatmulSelector(m, n, k, torch.bfloat16, torch.bfloat16,
                                     torch.bfloat16, DEVICE)
    assert (s.block_m, s.block_n, s.block_k) == (selector.block_m, selector.block_n,
                                                 selector.block_k)
    assert s.group_m == selector.group_m
    expected_xcds = selector.num_sms if selector.num_sms and selector.num_sms > 0 else 1
    assert s.num_xcds == expected_xcds
    expected_chunk = min(s.group_m * s.group_m, max(1, s.num_tiles // expected_xcds))
    assert s.chunk_size == expected_chunk


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_is_a_permutation_of_the_tile_grid(m, n, k):
    """Every tile computed exactly once. A map that drops or doubles one is unusable."""
    s = _schedule(m, n, k)
    assert sorted(s.tile_of_wg) == list(range(s.num_tiles))
    assert s.num_tiles == s.tiles_m * s.tiles_n


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_wg_of_tile_is_the_inverse(m, n, k):
    s = _schedule(m, n, k)
    inverse = s.wg_of_tile()
    for position, tile in enumerate(s.tile_of_wg):
        assert inverse[tile] == position


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_coords_agree_with_the_linear_ids(m, n, k):
    s = _schedule(m, n, k)
    assert s.arming_order == s.coords()
    for (pid_m, pid_n), tile in zip(s.coords(), s.tile_of_wg):
        assert pid_m * s.tiles_n + pid_n == tile
        assert 0 <= pid_m < s.tiles_m and 0 <= pid_n < s.tiles_n


def test_default_schedule_is_a_complete_per_tile_signal_plan():
    """A triggered consumer can use the default schedule without deriving another table."""
    s = _schedule(4096, 4096, 8192)
    assert s.num_signals == s.num_tiles
    assert s.signal_of_tile == tuple(range(s.num_tiles))
    assert s.producers == (1,) * s.num_tiles
    assert _arming_ids(s) == s.tile_of_wg
    assert s.signal_arming_order == s.signal_plan.arms
    for arm in s.signal_plan.arms:
        assert arm.producers == 1
        assert arm.tiles == ((arm.signal_id // s.grid_n, arm.signal_id % s.grid_n),)


def test_grouped_signal_plan_contains_thresholds_and_firing_order():
    s = _schedule(8192, 8192, 4096)
    group_size = 7
    signal_of_tile = [0] * s.num_tiles
    for position, tile in enumerate(s.tile_of_wg):
        signal_of_tile[tile] = position // group_size

    grouped = s.with_signal_layout(signal_of_tile)
    expected_signals = triton.cdiv(s.num_tiles, group_size)
    assert grouped.num_signals == expected_signals
    assert grouped.producers[:-1] == (group_size,) * (expected_signals - 1)
    assert grouped.producers[-1] == (s.num_tiles % group_size or group_size)
    assert _arming_ids(grouped) == tuple(range(expected_signals))

    inverse = grouped.wg_of_tile()
    ready = [
        max(inverse[tile] for tile, owner in enumerate(grouped.signal_of_tile) if owner == signal)
        for signal in range(grouped.num_signals)
    ]
    assert _arming_ids(grouped) == tuple(
        sorted(range(grouped.num_signals), key=ready.__getitem__)
    )
    assert grouped.arming_order == s.arming_order
    for arm in grouped.signal_plan.arms:
        expected_tiles = tuple(
            (tile // grouped.grid_n, tile % grouped.grid_n)
            for tile, signal in enumerate(grouped.signal_of_tile)
            if signal == arm.signal_id
        )
        assert arm.tiles == expected_tiles
        assert arm.producers == len(expected_tiles)


def test_invalid_signal_plans_are_rejected():
    s = _schedule(4096, 4096, 8192)
    with pytest.raises(ValueError, match="tile grid"):
        s.with_signal_layout([0])
    with pytest.raises(ValueError, match="non-negative"):
        s.with_signal_layout([-1] * s.num_tiles)
    sparse = [0] * s.num_tiles
    sparse[-1] = 2
    with pytest.raises(ValueError, match="dense"):
        s.with_signal_layout(sparse)


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_is_deterministic(m, n, k):
    """Same problem, same order — otherwise nothing downstream can cache or compare it."""
    assert _schedule(m, n, k).tile_of_wg == _schedule(m, n, k).tile_of_wg


@pytest.mark.parametrize("dtype", DTYPES)
def test_accepts_the_supported_dtypes(dtype):
    s = _schedule(2048, 2048, 2048, dtype=dtype)
    assert s is not None and s.num_tiles > 0


def test_schedule_from_tensors_matches_schedule_from_dims():
    a = torch.randn(4096, 8192, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(8192, 4096, device=DEVICE, dtype=torch.bfloat16)
    assert schedule(a, b).tile_of_wg == _schedule(4096, 4096, 8192).tile_of_wg


def test_schedule_accepts_the_device_signal_table_used_by_matmul():
    a = torch.randn(4096, 8192, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(8192, 4096, device=DEVICE, dtype=torch.bfloat16)
    base = schedule(a, b)
    signal_of_tile = torch.arange(base.num_tiles, dtype=torch.int32, device=DEVICE) // 3
    planned = schedule(a, b, signal_of_tile=signal_of_tile)
    assert planned.signal_of_tile == tuple(signal_of_tile.cpu().tolist())
    assert sum(planned.producers) == planned.num_tiles
    assert sorted(_arming_ids(planned)) == list(range(planned.num_signals))


# ---------------------------------------------------------------------------
# No static order is a normal answer
# ---------------------------------------------------------------------------


def test_dynamic_dispatch_returns_none():
    """
    Work-stealing claims tiles from counters at runtime and Stream-K does not give a tile a
    single owner, so neither has a permutation to publish. Callers must handle `None` rather
    than substitute a guess — for a triggered collective the right response is to not attempt
    gated overlap at all.
    """
    a = torch.randn(2048, 2048, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=DEVICE, dtype=torch.bfloat16)
    assert schedule(a, b) is not None
    assert schedule(a, b, work_stealing=True) is None
    assert schedule(a, b, enable_streamk=True) is None
    assert schedule(a, b, work_stealing=True, enable_streamk=True) is None


# ---------------------------------------------------------------------------
# Readiness: the number that decides whether overlap is worth attempting
# ---------------------------------------------------------------------------


def test_a_single_wave_grid_has_no_overlap_window():
    """
    The check this API exists to make possible.

    When the grid fits in one wave every tile starts and finishes together, so no dependent
    consumer can be overlapped however it is scheduled — `first_ready_fraction` must return 1.0
    and not something encouraging. Reading readiness off dispatch position instead would report
    a window here that does not exist.
    """
    s = _schedule(4096, 4096, 8192)
    num_cus = s.num_tiles * 2          # comfortably more CUs than tiles
    assert s.first_ready_fraction(num_cus=num_cus) == pytest.approx(1.0)


def test_a_many_wave_grid_has_a_window():
    s = _schedule(8192, 8192, 4096)
    num_cus = max(1, s.num_tiles // 8)  # ~8 waves
    fraction = s.first_ready_fraction(num_cus=num_cus)
    assert 0.0 < fraction < 0.5, fraction


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_more_cus_never_widens_the_window(m, n, k):
    """Fewer waves can only make readiness *later* relative to the whole GEMM, never earlier."""
    s = _schedule(m, n, k)
    coarse = s.first_ready_fraction(num_cus=max(s.num_xcds, s.num_tiles // 16))
    fine = s.first_ready_fraction(num_cus=max(s.num_xcds, s.num_tiles * 2))
    assert coarse <= fine + 1e-9


def test_first_ready_fraction_requires_a_cu_count():
    """
    Deliberately not optional. Measured in dispatch positions the answer is always 0 — some tile
    is always first — which says nothing; the question is temporal.
    """
    s = _schedule(2048, 2048, 2048)
    with pytest.raises(TypeError):
        s.first_ready_fraction()


@pytest.mark.parametrize("m, n, k", SCHEDULE_DIMS)
def test_ready_waves_are_consistent_and_cover_every_tile(m, n, k):
    s = _schedule(m, n, k)
    num_cus = max(s.num_xcds, s.num_tiles // 4)
    waves = s.ready_wave_of_tile(num_cus)
    assert len(waves) == s.num_tiles
    assert min(waves) >= 1
    for position, tile in enumerate(s.tile_of_wg):
        assert waves[tile] == s.finish_wave(num_cus, position)


def test_signal_ready_wave_is_the_last_producer_wave():
    s = _schedule(8192, 8192, 4096)
    grouped = s.with_signal_layout([tile // 5 for tile in range(s.num_tiles)])
    num_cus = max(s.num_xcds, s.num_tiles // 4)
    tile_waves = grouped.ready_wave_of_tile(num_cus)
    signal_waves = grouped.ready_wave_of_signal(num_cus)
    for signal in range(grouped.num_signals):
        expected = max(
            tile_waves[tile]
            for tile, owner in enumerate(grouped.signal_of_tile)
            if owner == signal
        )
        assert signal_waves[signal] == expected


def test_finish_wave_is_monotone_in_dispatch_position():
    """Later launch index, never an earlier wave — the property readiness ordering rests on."""
    s = _schedule(8192, 8192, 4096)
    num_cus = max(s.num_xcds, s.num_tiles // 4)
    waves = [s.finish_wave(num_cus, p) for p in range(s.num_tiles)]
    assert waves == sorted(waves)

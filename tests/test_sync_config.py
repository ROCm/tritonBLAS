"""Tests for EpilogueConfig (tile counters) and PrologueConfig (tile waits)."""

import pytest
import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore
from tritonblas import EpilogueConfig, PrologueConfig  # type: ignore
from tritonblas.matmul import _make_matmul_selector, persistent_matmul_lt  # type: ignore

# Non-square tile grid so row/col/block mappings are distinguishable.
M, N, K = 512, 768, 256
DTYPE = torch.float16


def _i32(n, fill=0):
    return torch.full((n,), fill, device="cuda", dtype=torch.int32)


@pytest.fixture(scope="module")
def problem():
    a = torch.randn(M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=DTYPE)
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, DTYPE, a.device)
    tm = triton.cdiv(M, selector.block_m)
    tn = triton.cdiv(N, selector.block_n)
    return a, b, a @ b, selector, tm, tn


def _slot(map_type, tile_id, tn, gm, gn):
    pid_m, pid_n = divmod(tile_id, tn)
    if map_type == "identity":
        return tile_id
    if map_type == "row":
        return pid_m
    if map_type == "col":
        return pid_n
    if map_type == "block":
        return (pid_m // gm) * triton.cdiv(tn, gn) + pid_n // gn
    if map_type == "launch_wave":
        return tile_id // gm
    raise AssertionError(map_type)


MAPPINGS = [
    ("identity", 1, 1),
    ("row", 1, 1),
    ("col", 1, 1),
    ("block", 2, 2),
    ("block", 3, 5),
    ("launch_wave", 16, 1),
]


@pytest.mark.parametrize("map_type, gm, gn", MAPPINGS)
def test_epilogue_counter_values(problem, map_type, gm, gn):
    a, b, ref, selector, tm, tn = problem
    size = EpilogueConfig(_i32(1), map_type, gm, gn).required_size(M, N, selector.block_m, selector.block_n)
    cfg = EpilogueConfig(_i32(size), map_type, gm, gn)
    c = torch.empty(M, N, device="cuda", dtype=DTYPE)

    persistent_matmul_lt(a, b, c, selector, epilogue_config=cfg)
    torch.cuda.synchronize()

    expected = [0] * size
    for tile_id in range(tm * tn):
        expected[_slot(map_type, tile_id, tn, gm, gn)] += 1
    assert cfg.counter_buffer.tolist() == expected
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_epilogue_counters_accumulate_until_reset(problem):
    a, b, _, selector, tm, tn = problem
    cfg = EpilogueConfig(_i32(tm * tn))
    c = torch.empty(M, N, device="cuda", dtype=DTYPE)

    persistent_matmul_lt(a, b, c, selector, epilogue_config=cfg)
    persistent_matmul_lt(a, b, c, selector, epilogue_config=cfg)
    torch.cuda.synchronize()
    assert cfg.counter_buffer.tolist() == [2] * (tm * tn)

    cfg.counter_buffer.zero_()
    persistent_matmul_lt(a, b, c, selector, epilogue_config=cfg)
    torch.cuda.synchronize()
    assert cfg.counter_buffer.tolist() == [1] * (tm * tn)


@pytest.mark.parametrize("map_type, gm, gn", [m for m in MAPPINGS if m[0] != "launch_wave"])
def test_prologue_wait_completes_across_launches(problem, map_type, gm, gn):
    a, b, ref, selector, tm, tn = problem
    inc = 2
    size = PrologueConfig(_i32(1), _i32(1), inc, map_type, gm, gn).required_size(
        M, N, selector.block_m, selector.block_n
    )
    wait, expected = _i32(size), _i32(tm * tn)
    cfg = PrologueConfig(wait, expected, inc, map_type, gm, gn)
    c = torch.empty(M, N, device="cuda", dtype=DTYPE)

    # The kernel bumps expected[tile] by `inc` per launch and waits for flag >= expected, so
    # a producer that bumps the flags by `inc` per round lets every launch complete.
    for launch in range(1, 4):
        wait.fill_(launch * inc)
        c.zero_()
        persistent_matmul_lt(a, b, c, selector, prologue_config=cfg)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
        assert expected.tolist() == [launch * inc] * (tm * tn)


def test_epilogue_and_prologue_together(problem):
    a, b, ref, selector, tm, tn = problem
    epilogue = EpilogueConfig(_i32(tm * tn))
    prologue = PrologueConfig(_i32(tm * tn, fill=1), _i32(tm * tn))
    c = torch.empty(M, N, device="cuda", dtype=DTYPE)

    persistent_matmul_lt(a, b, c, selector, epilogue_config=epilogue, prologue_config=prologue)
    torch.cuda.synchronize()
    assert epilogue.counter_buffer.tolist() == [1] * (tm * tn)
    assert prologue.expected_buffer.tolist() == [1] * (tm * tn)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_matmul_lt_forwards_configs(problem):
    a, b, ref, selector, tm, tn = problem
    cfg = EpilogueConfig(_i32(tm * tn))
    c = torch.empty(M, N, device="cuda", dtype=DTYPE)

    tritonblas.matmul_lt(a, b, c, selector, None, epilogue_config=cfg)
    torch.cuda.synchronize()
    assert cfg.counter_buffer.tolist() == [1] * (tm * tn)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


class TestInvalidConfigs:
    def test_undersized_counter_buffer(self, problem):
        a, b, _, selector, _, _ = problem
        c = torch.empty(M, N, device="cuda", dtype=DTYPE)
        with pytest.raises(ValueError, match="counter_buffer too small"):
            persistent_matmul_lt(a, b, c, selector, epilogue_config=EpilogueConfig(_i32(1)))

    def test_undersized_wait_buffer(self, problem):
        a, b, _, selector, tm, tn = problem
        c = torch.empty(M, N, device="cuda", dtype=DTYPE)
        with pytest.raises(ValueError, match="wait_buffer too small"):
            persistent_matmul_lt(a, b, c, selector, prologue_config=PrologueConfig(_i32(1), _i32(tm * tn)))

    def test_undersized_expected_buffer(self, problem):
        a, b, _, selector, tm, tn = problem
        c = torch.empty(M, N, device="cuda", dtype=DTYPE)
        # One flag is enough for a single-slot mapping, but expected still needs one entry per tile.
        cfg = PrologueConfig(_i32(tm * tn), _i32(1))
        with pytest.raises(ValueError, match="expected_buffer too small"):
            persistent_matmul_lt(a, b, c, selector, prologue_config=cfg)

    def test_work_stealing_rejects_configs(self, problem):
        a, b, _, selector, tm, tn = problem
        c = torch.empty(M, N, device="cuda", dtype=DTYPE)
        for kwargs in (
            {"epilogue_config": EpilogueConfig(_i32(tm * tn))},
            {"prologue_config": PrologueConfig(_i32(tm * tn), _i32(tm * tn))},
        ):
            with pytest.raises(NotImplementedError):
                persistent_matmul_lt(a, b, c, selector, work_stealing=True, **kwargs)

    def test_matmul_lt_streamk_rejects_configs(self, problem):
        a, b, _, selector, tm, tn = problem
        c = torch.empty(M, N, device="cuda", dtype=DTYPE)
        with pytest.raises(NotImplementedError):
            tritonblas.matmul_lt(
                a, b, c, selector, None, enable_streamk=True, epilogue_config=EpilogueConfig(_i32(tm * tn))
            )

    @pytest.mark.parametrize("group", [0, -1])
    @pytest.mark.parametrize("field", ["block_group_m", "block_group_n"])
    def test_non_positive_group_size(self, field, group):
        with pytest.raises(ValueError, match=field):
            EpilogueConfig(_i32(4), "block", **{field: group})
        with pytest.raises(ValueError, match=field):
            PrologueConfig(_i32(4), _i32(4), 1, "block", **{field: group})

    @pytest.mark.parametrize("inc", [0, -1])
    def test_non_positive_expected_inc(self, inc):
        with pytest.raises(ValueError, match="expected_inc"):
            PrologueConfig(_i32(4), _i32(4), expected_inc=inc)

    def test_launch_wave_rejected_for_prologue(self):
        with pytest.raises(ValueError, match="launch_wave"):
            PrologueConfig(_i32(4), _i32(4), map_type="launch_wave", block_group_m=4)

    def test_bad_map_type(self):
        with pytest.raises(ValueError, match="Invalid map_type"):
            EpilogueConfig(_i32(4), map_type="bogus")

    def test_bad_buffer(self):
        with pytest.raises(TypeError, match="int32"):
            EpilogueConfig(torch.zeros(4, device="cuda"))
        with pytest.raises(ValueError, match="1D"):
            EpilogueConfig(torch.zeros(2, 2, device="cuda", dtype=torch.int32))
        with pytest.raises(ValueError, match="CUDA device"):
            PrologueConfig(_i32(4), torch.zeros(4, dtype=torch.int32, device="cpu"))

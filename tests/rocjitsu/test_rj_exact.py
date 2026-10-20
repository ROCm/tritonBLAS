"""Bit-exact tritonBLAS correctness tests, sized to run under rocjitsu (also valid on a real GPU).

Inputs are integers in [-2, 2], so the fp32 accumulation is exact for any K used here and the
only rounding is the final fp32 -> out_dtype conversion. The reference is that same conversion
of an exact CPU result, so PASS means every element matches bit for bit. Tolerances hide
emulator bugs as easily as kernel bugs, so there are none.

Covers paths upstream only exercises at >=4096^3 (test_matmul.py, test_matmul_lt.py): operand
layouts (transposes), fp32, matmul_lt with an explicit selector, plus out= into a NaN-poisoned
buffer so an unwritten tile fails loudly, and M/N/K tails around the tile sizes.
"""

import os
import subprocess
import sys

import pytest
import torch
import tritonblas

MODES = ["persistent", "streamk", "work_stealing"]

# (m, n, k): tails around the 16/32/64/128/256 tile edges, K smaller than BLOCK_K, and a
# few multi-tile shapes so persistent loops and Stream-K splits actually iterate. K=1 is its
# own test below (test_k1_known_bug) because it faults the GPU.
SHAPES = [
    (1, 1, 2),
    (1, 256, 64),
    (256, 1, 64),
    (17, 33, 2),
    (31, 47, 13),
    (64, 64, 64),
    (127, 129, 65),
    (255, 257, 96),
    (256, 256, 512),
    (300, 200, 1000),
]
SMOKE_SHAPES = [(1, 1, 2), (31, 47, 13), (127, 129, 65), (256, 256, 512)]

DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
]

# A is m x k and B is k x n either way; "col" makes it a transposed view of a contiguous buffer.
LAYOUTS = [("row", "row"), ("col", "row"), ("row", "col"), ("col", "col")]


def _ints(rows, cols, dtype, layout, gen):
    shape = (rows, cols) if layout == "row" else (cols, rows)
    t = torch.randint(-2, 3, shape, generator=gen, dtype=torch.int32).to(dtype)
    t = t.cuda()
    return t if layout == "row" else t.T


def _inputs(m, n, k, in_dtype, layout_a="row", layout_b="row", seed=0):
    gen = torch.Generator().manual_seed(seed)
    a = _ints(m, k, in_dtype, layout_a, gen)
    b = _ints(k, n, in_dtype, layout_b, gen)
    return a, b


def _ref(a, b, out_dtype, bias=None):
    c = a.cpu().double() @ b.cpu().double()
    if bias is not None:
        c = c + bias.cpu().double()[None, :]
    return c.float().to(out_dtype)


def _assert_exact(got, ref):
    got = got.cpu()
    assert got.shape == ref.shape
    nan = torch.isnan(got.float())
    assert not nan.any(), "%d of %d outputs never written (NaN poison survived)" % (nan.sum(), nan.numel())
    bad = got != ref
    if bad.any():
        idx = bad.nonzero()[0].tolist()
        err = (got.double() - ref.double()).abs().max().item()
        pytest.fail(
            "%d of %d elements differ, max_abs_err=%g, first at %s: got %s want %s"
            % (bad.sum(), bad.numel(), err, idx, got[tuple(idx)].item(), ref[tuple(idx)].item())
        )


def _flags(mode):
    return dict(enable_streamk=mode == "streamk", work_stealing=mode == "work_stealing")


def _smoke(shape):
    return [] if shape in SMOKE_SHAPES else [pytest.mark.rj_full]


def _shape_params(shapes=SHAPES):
    return [pytest.param(*s, marks=_smoke(s), id="%dx%dx%d" % s) for s in shapes]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("in_dtype, out_dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("m, n, k", _shape_params())
def test_matmul_exact(m, n, k, in_dtype, out_dtype, mode):
    a, b = _inputs(m, n, k, in_dtype)
    _assert_exact(tritonblas.matmul(a, b, **_flags(mode)), _ref(a, b, out_dtype))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout_a, layout_b", LAYOUTS, ids=["-".join(l) for l in LAYOUTS])
@pytest.mark.parametrize("in_dtype, out_dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("m, n, k", _shape_params([(31, 47, 13), (127, 129, 65), (256, 256, 512)]))
def test_matmul_lt_layouts_exact(m, n, k, in_dtype, out_dtype, layout_a, layout_b, mode):
    a, b = _inputs(m, n, k, in_dtype, layout_a, layout_b)
    c = torch.full((m, n), float("nan"), dtype=out_dtype, device="cuda")
    streamk = mode == "streamk"
    selector = tritonblas.OrigamiMatmulSelector(m, n, k, a.dtype, b.dtype, c.dtype, a.device, streamk=streamk)
    config = tritonblas.matmul_preamble(selector)
    tritonblas.matmul_lt(a, b, c, selector, config, streamk, work_stealing=mode == "work_stealing")
    _assert_exact(c, _ref(a, b, out_dtype))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("in_dtype, out_dtype", DTYPES[:2], ids=["bf16", "fp16"])
@pytest.mark.parametrize("m, n, k", _shape_params())
def test_matmul_out_poisoned_exact(m, n, k, in_dtype, out_dtype, mode):
    a, b = _inputs(m, n, k, in_dtype, seed=1)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device="cuda")
    tritonblas.matmul(a, b, out=out, **_flags(mode))
    _assert_exact(out, _ref(a, b, out_dtype))


# Stream-K's non-quantized epilogue converts acc to the output dtype and then adds the bias in
# that dtype (kernels/streamk_gemm.py), so it rounds twice and can land 1 ulp off torch.addmm at a
# tie. Only some shapes hit a tie (300x200x1000 bf16 does), hence non-strict.
_SK_BIAS_ROUNDING = pytest.mark.xfail(
    strict=False,
    reason=("Stream-K addmm adds bias after rounding to the output dtype (double rounding, 1 ulp at ties)"),
)
ADDMM_MODES = [pytest.param(m, marks=[_SK_BIAS_ROUNDING] if m == "streamk" else []) for m in MODES]


@pytest.mark.parametrize("mode", ADDMM_MODES)
@pytest.mark.parametrize("in_dtype, out_dtype", DTYPES[:2], ids=["bf16", "fp16"])
@pytest.mark.parametrize("m, n, k", _shape_params())
def test_addmm_exact(m, n, k, in_dtype, out_dtype, mode):
    a, b = _inputs(m, n, k, in_dtype, seed=2)
    gen = torch.Generator().manual_seed(3)
    bias = torch.randint(-8, 9, (n,), generator=gen, dtype=torch.int32).to(out_dtype).cuda()
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device="cuda")
    tritonblas.addmm(bias, a, b, out=out, **_flags(mode))
    _assert_exact(out, _ref(a, b, out_dtype, bias))


@pytest.mark.parametrize("mode", MODES)
def test_repeat_launch_deterministic(mode):
    """Stream-K / work-stealing keep cross-launch counters; a second launch must not see stale state."""
    m, n, k = 255, 257, 96
    a, b = _inputs(m, n, k, torch.bfloat16, seed=4)
    ref = _ref(a, b, torch.bfloat16)
    for _ in range(3):
        _assert_exact(tritonblas.matmul(a, b, **_flags(mode)), ref)


_K1_SCRIPT = """
import sys, torch, tritonblas
m, n, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
a = torch.randint(-2, 3, (m, 1)).to(torch.bfloat16).cuda()
b = torch.randint(-2, 3, (1, n)).to(torch.bfloat16).cuda()
c = tritonblas.matmul(a, b, enable_streamk=mode == "streamk", work_stealing=mode == "work_stealing")
torch.cuda.synchronize()
ref = (a.cpu().double() @ b.cpu().double()).to(torch.bfloat16)
sys.exit(0 if torch.equal(c.cpu(), ref) else 3)
"""


_K1_BUG = pytest.mark.xfail(
    strict=True,
    reason=(
        "K=1: Triton specializes K==1 to a constant, so GemmContext.reduce_axis computes "
        "num_k_tiles = cdiv(1, BLOCK_K) - 1 = 0 and tl.assume(num_k_tiles > 0) becomes assume(false); "
        "the kernel body is dropped. gfx950: memory fault (HSA 0x1016); gfx942: wrong result. "
        "Same pattern in streamk_gemm.py (tl.assume(loop_k > 1)). Remove this xfail once fixed."
    ),
)


# 64x64 fails in every mode on gfx950 (memory fault) and gfx942 (wrong result). At 1x1 the outcome
# of the violated assume is codegen-dependent (e.g. persistent happens to pass on gfx942), so those
# cases are non-strict.
_K1_MAYBE = pytest.mark.xfail(strict=False, reason=_K1_BUG.kwargs["reason"] + " (1x1: arch-dependent)")


@pytest.mark.parametrize(
    "m, n, mode",
    [
        pytest.param(m, n, mode, id="%dx%d-%s" % (m, n, mode), marks=[_K1_BUG if m > 1 else _K1_MAYBE])
        for m, n in [(1, 1), (64, 64)]
        for mode in MODES
    ],
)
def test_k1_known_bug(m, n, mode):
    # Subprocess: a GPU fault aborts the process, which would otherwise take pytest down with it.
    p = subprocess.run(
        [sys.executable, "-c", _K1_SCRIPT, str(m), str(n), mode],
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ,
    )
    assert p.returncode == 0, "rc=%d (%s)" % (
        p.returncode,
        "fault"
        if p.returncode < 0 or p.returncode == 134
        else "wrong result"
        if p.returncode == 3
        else p.stderr[-300:],
    )

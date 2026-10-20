"""pytest plugin: run tritonBLAS's tests under rocjitsu (CPU-emulated AMD GPU) without editing them.

rocjitsu executes GPU ISA on the CPU at roughly 1-1.5 GFLOP/s (gfx942/gfx950), so the
parametrizations that use 4096^3+ shapes would take minutes to hours each. This plugin
deselects any test whose (m, n, k) parameters exceed a FLOP budget, drops performance tests
(timing under emulation is meaningless), and marks the known failures below.

Load with:  pytest -p pytest_rj --rj-max-gflop 2.2 ...   (tests/rocjitsu/run-tier.sh does this)
"""

import os

import pytest

# Timing tests: rocjitsu has no timing model, a number from it is not a measurement.
_PERF_FILES = {"test_performance.py"}

# Known failures under rocjitsu, marked so the tiers stay green and a new failure stands out.
# --rj-run-known-failures runs them as plain tests.
#
# Stream-K + work stealing (enable_streamk=True, work_stealing=True) gives wrong results under
# rocjitsu (up to ~all of the output). Which parametrizations fail depends on what ran earlier in
# the same process: the same shape passes in a fresh process and fails mid-session, and the set
# differs between CI shards and machines. So every such case is xfail(strict=False) rather than a
# fixed list. Not yet reproduced on hardware.
_SK_WS_REASON = (
    "Stream-K + work stealing gives wrong results under rocjitsu, depending on prior launches in "
    "the process; unconfirmed on hardware"
)
# On gfx942 the tests' torch.mm reference (hipBLASLt) never completes for the skinny shapes under
# rocjitsu (tritonBLAS itself finishes in ~1 s), so each would burn the whole per-test timeout.
_GFX942_SKINNY_REASON = "torch.mm reference (hipBLASLt) hangs under rocjitsu on gfx942 for skinny shapes"


def pytest_addoption(parser):
    g = parser.getgroup("rocjitsu")
    g.addoption(
        "--rj-max-gflop",
        type=float,
        default=float(os.environ.get("RJ_MAX_GFLOP") or "2.2"),
        help="deselect tests whose 2*m*n*k exceeds this many GFLOP (default 2.2)",
    )
    g.addoption(
        "--rj-no-compile",
        action="store_true",
        help="deselect use_compile=True parametrizations (inductor adds ~minutes of JIT)",
    )
    g.addoption(
        "--rj-shard",
        default=None,
        metavar="I/N",
        help="after filtering, keep every Nth test starting at I (0-based) to split across CI jobs",
    )
    g.addoption(
        "--rj-run-known-failures",
        action="store_true",
        help="run the known rocjitsu failures as plain tests instead of marking them xfail",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "rj_full: larger rocjitsu exact-test shapes, skipped in the smoke tier")


def _gflop(params):
    m, n, k = params.get("m"), params.get("n"), params.get("k")
    if not all(isinstance(v, int) for v in (m, n, k)):
        return None
    return 2.0 * m * n * k / 1e9


def _arch():
    try:
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return "unknown"


def _mark_known_failures(item, params, arch):
    name = item.originalname
    if arch == "gfx942" and "skinny" in name:
        item.add_marker(pytest.mark.skip(reason=_GFX942_SKINNY_REASON))
        return
    if params.get("enable_streamk") is True and params.get("work_stealing") is True:
        item.add_marker(pytest.mark.xfail(strict=False, reason=_SK_WS_REASON))


def pytest_collection_modifyitems(config, items):
    arch = None if config.getoption("--rj-run-known-failures") else _arch()
    budget = config.getoption("--rj-max-gflop")
    no_compile = config.getoption("--rj-no-compile")
    keep, drop = [], []
    reasons = {"over-budget": 0, "performance": 0, "compile": 0}
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        g = _gflop(params)
        if item.path.name in _PERF_FILES or item.get_closest_marker("performance"):
            reasons["performance"] += 1
            drop.append(item)
        elif g is not None and g > budget:
            reasons["over-budget"] += 1
            drop.append(item)
        elif no_compile and params.get("use_compile") is True:
            reasons["compile"] += 1
            drop.append(item)
        else:
            if arch:
                _mark_known_failures(item, params, arch)
            keep.append(item)
    shard = config.getoption("--rj-shard")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        reasons["other-shard"] = sum(1 for j in range(len(keep)) if j % n != i)
        drop += [it for j, it in enumerate(keep) if j % n != i]
        keep = [it for j, it in enumerate(keep) if j % n == i]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
    config._rj_summary = (budget, reasons, len(keep))


def pytest_report_header(config):
    return "rocjitsu: functional emulation only, no timing. Budget %.2f GFLOP/test" % config.getoption("--rj-max-gflop")


def pytest_terminal_summary(terminalreporter, config):
    s = getattr(config, "_rj_summary", None)
    if s:
        budget, reasons, kept = s
        terminalreporter.write_line(
            "rocjitsu: selected %d; deselected %d over %.2f GFLOP, %d performance, %d compile, %d other shards"
            % (
                kept,
                reasons["over-budget"],
                budget,
                reasons["performance"],
                reasons["compile"],
                reasons.get("other-shard", 0),
            )
        )

"""K-1397 P13 skinny_N256 K-COMPLEMENT productionisation pin tests — torch-free.

Validates the K-1402 productionisation of the K-1397 RETRY-winning 12-cell
``_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12`` frozenset stacked as the
8th-position envelope on top of the K-1389 productionised 73-cell baseline
(envelope grows 73 → 85 cells; +12 admits).

These tests pin:
  * cardinality of the new frozenset (12 cells; total post-K-1397 = 85)
  * exact membership of the K-1397 12 cells (M ∈ {2048, 4096, 8192} ×
    N=256 × K ∈ {2048, 32768} × {bf16, fp16})
  * cross-frozenset disjointness from P8, K971_ROUTE_TABLE, P12, K-1367 P13
  * dispatch precedence: every K-1397 admit routes to hipBLASLt under default
    carve-out settings (no streamk, no work-stealing, matched dtype)
  * 4 boundary perturbations (K=4096, K=16384, N=128, N=512) MUST NOT match
    the K-1397 strict-equality frozenset — guards against axis over-relaxation
  * carve-out short-circuits (streamk / work_stealing / mixed dtype) preserve
    the load-bearing dispatch contract

Imports the actual shipped predicate module (``_route_predicate``) so the
real runtime path is exercised; no exec/string-parse copies.
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    K971_ROUTE_TABLE,
    _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4,
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _P8_MFMA_ISSUE_STALL_ROUTEOUT,
    _k1397_p13_skinny_n256_routeout,
    k971_route_decision,
)


# ---------------------------------------------------------------------------
# Cardinality pins.  Any deviation indicates an authoring typo against the
# K-1365 4-bucket decomposition or the K-1397 K-extreme-corners scoping.
# ---------------------------------------------------------------------------
def test_k1397_p13_skinny_n256_kcomplement_cardinality_12():
    assert len(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12) == 12


def test_envelope_total_post_k1397_is_85():
    """P8 (51) + K-1361 P12 (4) + K-1367 P13 (18) + K-1397 P13 (12) = 85
    strict-equality cells.

    Same accounting note as K-1389: K971_ROUTE_TABLE is the 5th-position
    envelope and is counted separately because it gates on the LDS-bank-
    conflict mechanism rather than the MFMA-issue-stall / square_mid /
    skinny_N128 / skinny_N256 mechanisms.
    """
    total = (
        len(_P8_MFMA_ISSUE_STALL_ROUTEOUT)
        + len(_K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)
        + len(_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)
        + len(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)
    )
    assert total == 85, (
        f"Expected 51 + 4 + 18 + 12 = 85 strict-equality cells across "
        f"P8/P12/P13(N128)/P13(N256); got {total}.  Check that no frozenset "
        f"has been mutated.")


# ---------------------------------------------------------------------------
# Exact-membership pin.  Enumerates every key so any accidental drop / typo
# / re-ordering is caught at CI time.
# ---------------------------------------------------------------------------
def _expected_k1397_12_cells():
    """Construct the K-1397 P13 12-cell expected set from the bucket rule."""
    return frozenset({
        (M, 256, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    })


def test_k1397_membership_exactly_12_kcomplement_cells():
    expected = _expected_k1397_12_cells()
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == expected, (
        "K-1397 P13 frozenset diverges from the K-COMPLEMENT bucket rule: "
        "M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,32768} × {bf16,fp16}.\n"
        f"  missing : {sorted(expected - _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12)}\n"
        f"  unknown : {sorted(_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 - expected)}")


# ---------------------------------------------------------------------------
# Disjointness pins.  Cheap insurance per
# R-1329.K-AXIS-PROJECTION-DISJOINTNESS-ASSERTS-ARE-CHEAP-INSURANCE.
# ---------------------------------------------------------------------------
def test_k1397_disjoint_from_p8():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _P8_MFMA_ISSUE_STALL_ROUTEOUT)


def test_k1397_disjoint_from_k971_route_table():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        K971_ROUTE_TABLE)


def test_k1397_disjoint_from_p12_square_mid():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4)


def test_k1397_disjoint_from_k1367_p13_skinny_n128():
    assert _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12.isdisjoint(
        _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18)


# ---------------------------------------------------------------------------
# Per-cell predicate behaviour: every admit cell fires.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_k1397_12_cells()))
def test_k1397_predicate_admits_every_kcomplement_cell(M, N, K, dtype):
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is True


# ---------------------------------------------------------------------------
# Full dispatch decision — every K-1397 admit routes to hipBLASLt under
# default carve-out settings (no streamk, no work-stealing, matched dtype).
# Spec requirement (1): "asserting all 12 cells route to hipBLASLt".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K,dtype", sorted(_expected_k1397_12_cells()))
def test_full_dispatch_routes_every_k1397_cell_to_hbl(M, N, K, dtype):
    """K-1397 P13 admits MUST route to hipBLASLt under default carve-out
    settings (no streamk, no work-stealing, matched dtype).
    """
    assert k971_route_decision(M, N, K, dtype, dtype, False, False) is True


# ---------------------------------------------------------------------------
# Boundary controls — 4 axis-perturbed cells that MUST NOT match the
# K-1397 strict-equality frozenset.  Spec requirement (2): "4 boundary-
# perturbed cells (K=4096, K=16384, N=128, N=512) do NOT match".
#
# Tested against the K-1397-specific frozenset / per-cell predicate ONLY,
# following the K-1389 boundary-control convention (test_k1389_p13_does_not_
# spill_into_adjacent_n128_kcompl_cells): end-to-end k971_route_decision may
# still route some of these to hipBLASLt via EARLIER predicates (R-K979 P5,
# K-1367 P13 N=128) and that is intentional and out of scope for K-1397.
# This test isolates the K-1397-specific no-spill claim.
# ---------------------------------------------------------------------------
_K1397_BOUNDARY_NON_ADMIT_CELLS = [
    # K=4096 — between the K=2048 and K=32768 cohort corners; same M,N=256.
    # MUST NOT match: K=4096 is the triton-favoured intermediate K band
    # (K-1227 wrapper-bound region; K-1397 measurement showed CI95 straddles
    # 1.0× — left to triton.persistent_matmul).
    (2048, 256,  4096, "torch.bfloat16"),
    (4096, 256,  4096, "torch.float16"),
    # K=16384 — between K=2048 and K=32768; same M,N=256.  MUST NOT match:
    # K=16384 is also in the triton-favoured intermediate K band (K-1397
    # measurement showed CI95 straddles 1.0× — left to triton).
    (8192, 256, 16384, "torch.bfloat16"),
    # N=128 — off the N=256 column-narrow regime; covered by the K-1367 P13
    # N=128 envelope (a separate, EARLIER predicate).  MUST NOT match the
    # K-1397 N=256 frozenset: the two cohorts are disjoint by N projection.
    (4096, 128,  2048, "torch.bfloat16"),
    # N=512 — wider than the N=256 column-narrow regime; persistent_matmul
    # tile efficiency improves past N=256 (MFMA tile efficiency threshold).
    # MUST NOT match: K-1397 cohort is N=256-only.
    (4096, 512,  2048, "torch.bfloat16"),
]


@pytest.mark.parametrize("M,N,K,dtype", _K1397_BOUNDARY_NON_ADMIT_CELLS)
def test_k1397_p13_does_not_spill_into_boundary_perturbed_cells(M, N, K, dtype):
    """K-1397 spec boundary control: the 4 axis-perturbed cells (K=4096,
    K=16384, N=128, N=512) MUST NOT be admitted by the K-1397 P13
    strict-equality frozenset _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 nor
    by the K-1397 per-cell predicate ``_k1397_p13_skinny_n256_routeout``.

    NOTE (mirrors K-1389 boundary-control convention): end-to-end
    ``k971_route_decision`` may still return True on some of these cells via
    EARLIER predicates (R-K979 P5 closed-form structural pathology, or the
    K-1367 P13 N=128 frozenset for the N=128 boundary).  That is intentional
    and out of scope for K-1397 — this task is a strict-equality additive
    extension at the 8th-position envelope, not a re-tuning of any earlier
    predicate.  The test isolates the K-1397-specific no-spill claim from
    the end-to-end routing decision.
    """
    assert (M, N, K, dtype) not in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12, (
        f"Boundary control cell ({M},{N},{K},{dtype}) is in the K-1397 "
        f"admit frozenset; the K-1397 boundary table is mis-specified or "
        f"the frozenset spilled into a perturbed axis.")
    assert _k1397_p13_skinny_n256_routeout(M, N, K, dtype) is False, (
        f"K-1397 boundary perturbation cell ({M},{N},{K},{dtype}) was "
        f"admitted by the K-1397 per-cell predicate; strict-equality "
        f"frozenset must NOT spill into perturbed-axis cells.")


# ---------------------------------------------------------------------------
# Carve-out short-circuits — a K-1397 admit cell must NOT route to hipBLASLt
# when streamk / work-stealing / mismatched-dtype is requested.  Mirrors
# the K-1367/K-1389 carve-out test; load-bearing for the matmul.py kwarg
# surface (PR description Test Plan).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_streamk,work_stealing,b_dtype", [
    (True,  False, "torch.bfloat16"),  # streamk requested
    (False, True,  "torch.bfloat16"),  # work-stealing requested
    (False, False, "torch.float16"),   # mixed dtype (a bf16, b fp16)
])
def test_k1397_admit_cell_carveout_short_circuits(
        enable_streamk, work_stealing, b_dtype):
    # Pick a known K-1397 admit cell — M=4096,N=256,K=32768,bf16.
    a_dtype = "torch.bfloat16"
    M, N, K = 4096, 256, 32768
    assert (M, N, K, a_dtype) in _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12
    routed = k971_route_decision(M, N, K, a_dtype, b_dtype,
                                 enable_streamk, work_stealing)
    assert routed is False, (
        f"K-1397 admit cell ({M},{N},{K},{a_dtype}) must NOT route to "
        f"hipBLASLt when carve-out applies (streamk={enable_streamk}, "
        f"work_stealing={work_stealing}, b_dtype={b_dtype}); short-circuit "
        f"in _k971_route_to_hbl is load-bearing.")


# ---------------------------------------------------------------------------
# Regression firewall — every cell in the K-1335 / K-1322 / K-905 / K-1361
# / K-1367 prior envelopes still routes to hipBLASLt after the K-1397 stack.
# Per R-1322.STRICT-EQUALITY-UNION-PRESERVES-PRIOR-ADMIT-INVARIANCE; the new
# 8th-position envelope is unioned via short-circuit `if ... return True`
# so prior admits cannot be removed by the diff.
# ---------------------------------------------------------------------------
def test_no_regression_on_k971_route_table_prior_admits():
    for (M, N, K, dt) in K971_ROUTE_TABLE:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K971_ROUTE_TABLE prior admit ({M},{N},{K},{dt}) was lost after "
            f"K-1397 P13 stack; the K-1397 productionisation must preserve "
            f"every prior 5th-position envelope cell (regression firewall).")


def test_no_regression_on_p8_prior_admits():
    for (M, N, K, dt) in _P8_MFMA_ISSUE_STALL_ROUTEOUT:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"P8 prior admit ({M},{N},{K},{dt}) was lost after K-1397 P13 "
            f"stack; the K-1397 productionisation must preserve every prior "
            f"P8 sub-frozenset cell (regression firewall).")


def test_no_regression_on_k1367_p13_n128_prior_admits():
    for (M, N, K, dt) in _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18:
        assert k971_route_decision(M, N, K, dt, dt, False, False) is True, (
            f"K-1367/K-1389 P13 N=128 prior admit ({M},{N},{K},{dt}) was "
            f"lost after K-1397 P13 stack; the K-1397 productionisation must "
            f"preserve every prior 7th-position envelope cell (regression "
            f"firewall).")

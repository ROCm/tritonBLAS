#!/usr/bin/env python3
"""
Tests for mosaic_shuffle scheduling strategies.

Test categories:
  - Feistel bijectivity (CPU, no GPU)
  - Randomness quality metrics (CPU)
  - GEMM numerical correctness (GPU, requires CUDA)
  - Workgroup map validity (GPU)
  - Trace integration (GPU)

Run with:
    pytest test_mosaic_shuffle.py -v
    pytest test_mosaic_shuffle.py -v -k "not gpu"  # CPU-only tests
"""

import math
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
_tritonblas_path = _repo_root / "external" / "tritonBLAS" / "include"
if str(_tritonblas_path) not in sys.path:
    sys.path.insert(0, str(_tritonblas_path))

from tritonblas.experimental.mosaic_shuffle.permutation import (
    compute_feistel_params,
    feistel_permute_ref,
    _feistel_round_ref,
)
from tritonblas.experimental.mosaic_shuffle.config import (
    L2AwareConfig,
    LLCAndL2AwareConfig,
)


# ===================================================================
# 1. Feistel bijectivity tests (CPU)
# ===================================================================

BIJECTIVITY_N_VALUES = [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                        64, 100, 127, 128, 255, 256, 500, 1000, 1024,
                        4096, 9973, 10000]


@pytest.mark.parametrize("n", BIJECTIVITY_N_VALUES)
def test_feistel_bijectivity(n):
    """Feistel permutation must be a bijection on [0, n) for all n >= 2."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    assert sorted(perm) == list(range(n)), f"Not a bijection for n={n}"


@pytest.mark.parametrize("n", [8, 64, 256, 1000])
def test_feistel_different_seeds_differ(n):
    """Different seeds should produce different permutations."""
    perms = []
    for seed in [0, 1, 42, 123, 999]:
        half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=seed)
        perm = tuple(feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n))
        perms.append(perm)

    unique_perms = set(perms)
    assert len(unique_perms) == len(perms), (
        f"Expected {len(perms)} unique permutations for n={n}, got {len(unique_perms)}"
    )


@pytest.mark.parametrize("n", BIJECTIVITY_N_VALUES)
def test_feistel_deterministic(n):
    """Same seed must produce the same permutation."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm1 = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    perm2 = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    assert perm1 == perm2


def test_feistel_rejects_n_less_than_2():
    with pytest.raises(ValueError, match="n >= 2"):
        compute_feistel_params(1)


# ===================================================================
# 2. Randomness quality tests (CPU)
# ===================================================================

def _compute_displacement(perm):
    """Mean |perm[i] - i| normalized by n."""
    n = len(perm)
    return sum(abs(perm[i] - i) for i in range(n)) / n


def _compute_fixed_points(perm):
    """Count elements where perm[i] == i."""
    return sum(1 for i in range(len(perm)) if perm[i] == i)


def _compute_autocorrelation(perm):
    """Pearson correlation between perm[0:n-1] and perm[1:n]."""
    n = len(perm)
    if n < 3:
        return 0.0
    x = perm[:-1]
    y = perm[1:]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
def test_randomness_displacement(n):
    """Tiles should actually move: mean displacement should be significant."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    displacement = _compute_displacement(perm)
    # A random permutation of [0,n) has expected displacement ~n/3
    assert displacement > n * 0.1, (
        f"Displacement too low for n={n}: {displacement:.1f} (expected > {n * 0.1:.1f})"
    )


@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
def test_randomness_autocorrelation(n):
    """Adjacent outputs should be uncorrelated."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    corr = _compute_autocorrelation(perm)
    assert abs(corr) < 0.15, (
        f"Autocorrelation too high for n={n}: {corr:.4f} (threshold: 0.15)"
    )


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_randomness_fixed_points(n):
    """Fixed points (perm[i]==i) should be rare."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]
    fp = _compute_fixed_points(perm)
    threshold = max(n * 0.1, 5)
    assert fp < threshold, (
        f"Too many fixed points for n={n}: {fp} (threshold: {threshold})"
    )


def test_randomness_comparison_feistel_vs_lcg():
    """Print a quality comparison table (informational, doesn't fail)."""
    n = 1024
    seed = 42

    # Feistel
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=seed)
    feistel_perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]

    # LCG (for comparison)
    import random
    rng = random.Random(seed)
    a_lcg = rng.choice([a for a in range(5, n, 4)])
    c_lcg = rng.randint(0, n - 1)
    lcg_perm = [(a_lcg * i + c_lcg) % n for i in range(n)]

    feistel_disp = _compute_displacement(feistel_perm)
    feistel_corr = _compute_autocorrelation(feistel_perm)
    feistel_fp = _compute_fixed_points(feistel_perm)

    lcg_disp = _compute_displacement(lcg_perm)
    lcg_corr = _compute_autocorrelation(lcg_perm)
    lcg_fp = _compute_fixed_points(lcg_perm)

    print(f"\n{'Metric':<25} {'Feistel':>12} {'LCG':>12}")
    print("-" * 50)
    print(f"{'Displacement (mean)':<25} {feistel_disp:>12.1f} {lcg_disp:>12.1f}")
    print(f"{'Autocorrelation':<25} {feistel_corr:>12.4f} {lcg_corr:>12.4f}")
    print(f"{'Fixed points':<25} {feistel_fp:>12d} {lcg_fp:>12d}")


# ===================================================================
# 3. Config validation tests (CPU)
# ===================================================================

def test_l2_aware_config_valid():
    cfg = L2AwareConfig(tile_y=8, tile_x=4, inner_ordering=1, seed=42)
    assert cfg.tile_y == 8
    assert cfg.tile_x == 4


def test_l2_aware_config_rejects_bad_values():
    with pytest.raises(ValueError):
        L2AwareConfig(tile_y=0, tile_x=4)
    with pytest.raises(ValueError):
        L2AwareConfig(tile_y=8, tile_x=-1)
    with pytest.raises(ValueError):
        L2AwareConfig(tile_y=8, tile_x=4, inner_ordering=2)


def test_llc_and_l2_aware_config_valid():
    cfg = LLCAndL2AwareConfig(L2Y=8, L2X=4, L3Y=2, L3X=4, ordering0=0, ordering1=1, ordering2=0)
    assert cfg.chunk_size == 32
    assert cfg.to_kernel_kwargs()["L2Y"] == 8


def test_llc_and_l2_aware_config_rejects_bad_values():
    with pytest.raises(ValueError):
        LLCAndL2AwareConfig(L2Y=0, L2X=4, L3Y=2, L3X=4, ordering0=0, ordering1=0, ordering2=0)
    with pytest.raises(ValueError):
        LLCAndL2AwareConfig(L2Y=8, L2X=4, L3Y=2, L3X=4, ordering0=3, ordering1=0, ordering2=0)


# ===================================================================
# 4. Visualization / grid dump tests (CPU)
# ===================================================================

@pytest.mark.parametrize("n", [16, 64])
def test_grid_dump_layout_viewer_format(n):
    """Write a workgroup map JSON compatible with layout_viewer schedule format."""
    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed=42)
    perm = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]

    side = int(math.isqrt(n))
    mapping = []
    for wgid in range(n):
        tile_flat = perm[wgid]
        mapping.append({
            "wgid": wgid,
            "tile_m": tile_flat // side,
            "tile_n": tile_flat % side,
        })

    schedule = {
        "metadata": {
            "backend": "mosaic_shuffle_test",
            "strategy": "random",
            "grid": {
                "grid_height": side,
                "grid_width": side,
                "num_workgroups": n,
            },
        },
        "mapping": mapping,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(schedule, f, indent=2)
        path = f.name

    with open(path) as f:
        loaded = json.load(f)

    os.unlink(path)

    assert len(loaded["mapping"]) == n
    assert loaded["metadata"]["strategy"] == "random"
    wgids = [entry["wgid"] for entry in loaded["mapping"]]
    assert sorted(wgids) == list(range(n))


# ===================================================================
# 5. GPU tests (require CUDA / ROCm)
# ===================================================================

def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="No GPU available")


@requires_gpu
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (4096, 4096, 4096)])
def test_gemm_correctness_random(M, N, K):
    """matmul_random produces correct results."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import matmul_random

    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    c_ref = a @ b

    matmul_random(a, b, c, seed=42)
    rel_error = (c - c_ref).abs().max().item() / c_ref.abs().max().item()
    assert rel_error < 1e-2, f"Relative error too high: {rel_error:.2e}"


@requires_gpu
@pytest.mark.parametrize("tile_y,tile_x,inner_ordering", [(8, 4, 0), (8, 4, 1), (4, 8, 0)])
def test_gemm_correctness_l2_aware(tile_y, tile_x, inner_ordering):
    """matmul_l2_aware produces correct results with various tile configs."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import matmul_l2_aware

    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    c_ref = a @ b

    cfg = L2AwareConfig(tile_y=tile_y, tile_x=tile_x, inner_ordering=inner_ordering, seed=42)
    matmul_l2_aware(a, b, c, config=cfg)
    rel_error = (c - c_ref).abs().max().item() / c_ref.abs().max().item()
    assert rel_error < 1e-2, f"Relative error too high: {rel_error:.2e}"


@requires_gpu
def test_gemm_correctness_llc_and_l2_aware():
    """matmul_llc_and_l2_aware produces correct results."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import matmul_llc_and_l2_aware

    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    c_ref = a @ b

    cfg = LLCAndL2AwareConfig(L2Y=8, L2X=4, L3Y=2, L3X=4, ordering0=1, ordering1=1, ordering2=1)
    matmul_llc_and_l2_aware(a, b, c, config=cfg)
    rel_error = (c - c_ref).abs().max().item() / c_ref.abs().max().item()
    assert rel_error < 1e-2, f"Relative error too high: {rel_error:.2e}"


@requires_gpu
@pytest.mark.parametrize("strategy", ["random", "l2_aware", "llc_and_l2_aware"])
def test_workgroup_map_validity(strategy):
    """compute_workgroup_map returns a valid assignment."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import compute_workgroup_map

    M, N, K = 4096, 4096, 4096
    config = None
    if strategy == "l2_aware":
        config = L2AwareConfig(tile_y=8, tile_x=4, inner_ordering=1, seed=42)
    elif strategy == "llc_and_l2_aware":
        config = LLCAndL2AwareConfig(L2Y=8, L2X=4, L3Y=2, L3X=4, ordering0=1, ordering1=1, ordering2=1)

    wg_map = compute_workgroup_map(M, N, K, strategy=strategy, config=config, seed=42)
    assert wg_map.ndim == 2
    assert wg_map.shape[0] > 0 and wg_map.shape[1] > 0


@requires_gpu
@pytest.mark.parametrize("strategy", ["random", "l2_aware", "llc_and_l2_aware"])
def test_trace_integration(strategy):
    """Trace data has valid structure (start <= end for all tiles)."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import (
        matmul_random, matmul_l2_aware, matmul_llc_and_l2_aware,
    )

    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    if strategy == "random":
        _, trace_data = matmul_random(a, b, c, seed=42, trace=True)
    elif strategy == "l2_aware":
        cfg = L2AwareConfig(tile_y=8, tile_x=4, inner_ordering=1, seed=42)
        _, trace_data = matmul_l2_aware(a, b, c, config=cfg, trace=True)
    elif strategy == "llc_and_l2_aware":
        cfg = LLCAndL2AwareConfig(L2Y=8, L2X=4, L3Y=2, L3X=4, ordering0=1, ordering1=1, ordering2=1)
        _, trace_data = matmul_llc_and_l2_aware(a, b, c, config=cfg, trace=True)

    assert "start" in trace_data
    assert "end" in trace_data
    assert "pid" in trace_data
    assert "xcd" in trace_data
    assert trace_data["total_tiles"] > 0

    starts = trace_data["start"]
    ends = trace_data["end"]
    total_tiles = trace_data["total_tiles"]

    valid_count = sum(1 for i in range(total_tiles) if starts[i] <= ends[i])
    assert valid_count == total_tiles, (
        f"Some tiles have start > end: {total_tiles - valid_count} violations"
    )


@requires_gpu
def test_trace_json_dump():
    """Trace data can be unflattened to layout_viewer compatible JSON."""
    import torch
    from tritonblas.experimental.mosaic_shuffle import matmul_random

    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    _, trace_data = matmul_random(a, b, c, seed=42, trace=True)

    num_pid_n = trace_data["num_pid_n"]
    total_tiles = trace_data["total_tiles"]

    tiles = []
    for flat_id in range(total_tiles):
        tiles.append({
            "wgid": int(trace_data["pid"][flat_id]),
            "tile_m": flat_id // num_pid_n,
            "tile_n": flat_id % num_pid_n,
            "start": int(trace_data["start"][flat_id]),
            "end": int(trace_data["end"][flat_id]),
            "xcd": int(trace_data["xcd"][flat_id]),
        })

    trace_json = {
        "metadata": {
            "strategy": "mosaic_random",
            "M": trace_data["M"],
            "N": trace_data["N"],
            "K": trace_data["K"],
            "num_pid_m": trace_data["num_pid_m"],
            "num_pid_n": num_pid_n,
            "total_tiles": total_tiles,
        },
        "tiles": tiles,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(trace_json, f, indent=2)
        path = f.name

    with open(path) as f:
        loaded = json.load(f)

    os.unlink(path)

    assert len(loaded["tiles"]) == total_tiles
    assert loaded["metadata"]["strategy"] == "mosaic_random"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

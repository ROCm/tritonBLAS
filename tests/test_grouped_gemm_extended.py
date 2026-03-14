"""
Extended validation tests for the grouped GEMM kernel.

Covers:
- Large shapes (single and multi-group)
- vLLM-style inference shapes
- Many groups (16, 32)
- Extreme aspect ratios
- Tile size sweep
- Determinism (bit-identical results)
- Relative error with torch.testing.assert_close()
- Group count scaling
"""

import pytest
import torch
import tritonblas


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

FP16_ATOL = 0.5
FP16_RTOL = 0.01
BF16_ATOL = 1.0
BF16_RTOL = 0.02


def get_atol(dtype):
    if dtype == torch.float16:
        return FP16_ATOL
    elif dtype == torch.bfloat16:
        return BF16_ATOL
    return 1.0


def get_rtol(dtype):
    if dtype == torch.float16:
        return FP16_RTOL
    elif dtype == torch.bfloat16:
        return BF16_RTOL
    return 1e-3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_grouped_gemm(shapes, dtype=torch.float16, BLK_M=None, BLK_N=None, BLK_K=None):
    """
    Build tensors, call grouped_gemm, return (results, refs).

    When BLK_M/N/K are None, origami prediction is used.
    The seed is derived from the shapes so each unique problem gets
    its own reproducible random inputs regardless of test ordering.
    """
    group_a, group_b, group_c, refs = [], [], [], []
    # Derive a deterministic seed from the problem to avoid RNG-state
    # ordering issues when pytest-randomly reorders tests.
    seed = hash((tuple(shapes), dtype, BLK_M, BLK_N, BLK_K)) & 0xFFFF_FFFF
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        c = torch.empty(m, n, device="cuda", dtype=dtype)
        group_a.append(a)
        group_b.append(b)
        group_c.append(c)
        refs.append(torch.matmul(a, b))

    if BLK_M is None or BLK_N is None or BLK_K is None:
        results = tritonblas.grouped_gemm(group_a, group_b)
    else:
        results = tritonblas.grouped_gemm(group_a, group_b, group_c, BLK_M, BLK_N, BLK_K)

    return results, refs


# ---------------------------------------------------------------------------
# Large shapes
# ---------------------------------------------------------------------------

class TestLargeShapes:
    """Single and multi-group large matrix multiplications."""

    def test_single_group_4096x4096x4096(self):
        shapes = [(4096, 4096, 4096)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg=f"Single group 4096x4096x4096 failed"
        )

    @pytest.mark.parametrize("n_groups", [4])
    def test_four_groups_4096x4096x4096(self, n_groups):
        shapes = [(4096, 4096, 4096)] * n_groups
        results, refs = run_grouped_gemm(shapes)
        for i, (res, ref) in enumerate(zip(results, refs)):
            torch.testing.assert_close(
                res, ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"Group {i} of {n_groups} failed for shape 4096x4096x4096"
            )

    def test_single_group_8192x8192x4096(self):
        shapes = [(8192, 8192, 4096)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Single group 8192x8192x4096 failed"
        )


# ---------------------------------------------------------------------------
# vLLM-style shapes
# ---------------------------------------------------------------------------

class TestVLLMShapes:
    """Shapes common in LLM inference (attention projections, FFN layers)."""

    @pytest.mark.parametrize("m, n, k", [
        (1, 4096, 4096),
        (32, 4096, 11008),
        (1, 11008, 4096),
    ])
    def test_single_group_vllm(self, m, n, k):
        shapes = [(m, n, k)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg=f"vLLM single group ({m},{n},{k}) failed"
        )

    def test_multi_group_vllm(self):
        """Mixed vLLM inference shapes in a single grouped call."""
        shapes = [
            (1, 4096, 4096),
            (32, 4096, 11008),
            (1, 11008, 4096),
            (32, 11008, 4096),
        ]
        results, refs = run_grouped_gemm(shapes)
        for i, (res, ref) in enumerate(zip(results, refs)):
            torch.testing.assert_close(
                res, ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"vLLM multi-group: group {i} (shape {shapes[i]}) failed"
            )


# ---------------------------------------------------------------------------
# Many groups
# ---------------------------------------------------------------------------

class TestManyGroups:
    """Stress-test with 16 and 32 groups."""

    def test_16_groups_256x256x128(self):
        shapes = [(256, 256, 128)] * 16
        results, refs = run_grouped_gemm(shapes)
        for i, (res, ref) in enumerate(zip(results, refs)):
            torch.testing.assert_close(
                res, ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"16 groups: group {i} failed"
            )

    def test_32_groups_128x128x64(self):
        # NOTE: This test exposes an intermittent stream-K kernel bug where
        # group 29 (of 32) occasionally produces large errors (~10x) in a
        # small number of elements. This appears to be a race condition in
        # the stream-K atomic reduction. The test is intentionally kept strict
        # so the bug is caught when present.
        shapes = [(128, 128, 64)] * 32
        results, refs = run_grouped_gemm(shapes)
        for i, (res, ref) in enumerate(zip(results, refs)):
            diff = (res - ref).abs()
            max_err = diff.max().item()
            assert max_err <= FP16_ATOL, (
                f"32 groups: group {i} failed — "
                f"max_abs_err={max_err:.4f} (limit={FP16_ATOL}), "
                f"ref_range=[{ref.min().item():.3f}, {ref.max().item():.3f}]"
            )


# ---------------------------------------------------------------------------
# Extreme aspect ratios
# ---------------------------------------------------------------------------

class TestExtremeAspectRatios:
    """Edge-case shapes with very small M, N, or very large K."""

    def test_single_row(self):
        """M=1: single-row output."""
        shapes = [(1, 4096, 128)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Single-row (1, 4096, 128) failed"
        )

    def test_single_column(self):
        """N=1: single-column output."""
        shapes = [(4096, 1, 128)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Single-column (4096, 1, 128) failed"
        )

    def test_deep_k(self):
        """Large K reduction dimension."""
        shapes = [(128, 128, 4096)]
        results, refs = run_grouped_gemm(shapes)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Deep K (128, 128, 4096) failed"
        )


# ---------------------------------------------------------------------------
# Tile size sweep
# ---------------------------------------------------------------------------

class TestTileSizeSweep:
    """Same problem shape with different manually specified tile sizes."""

    SHAPE = (1024, 1024, 512)

    @pytest.mark.parametrize("BLK_M, BLK_N, BLK_K", [
        (64, 64, 32),
        (128, 128, 64),
        (256, 256, 64),
    ])
    def test_tile_sweep(self, BLK_M, BLK_N, BLK_K):
        results, refs = run_grouped_gemm(
            [self.SHAPE], BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K
        )
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL,
            msg=f"Tile {BLK_M}x{BLK_N}x{BLK_K} on shape {self.SHAPE} failed"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same inputs must produce bit-identical outputs across repeated calls."""

    SHAPES = [(512, 512, 256)] * 4

    def _build_inputs(self, dtype=torch.float16):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        group_a, group_b, group_c = [], [], []
        for m, n, k in self.SHAPES:
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype)
            c = torch.empty(m, n, device="cuda", dtype=dtype)
            group_a.append(a)
            group_b.append(b)
            group_c.append(c)
        return group_a, group_b, group_c

    def test_bit_identical_three_runs(self):
        group_a, group_b, group_c = self._build_inputs()

        run1 = tritonblas.grouped_gemm(group_a, group_b)
        run2 = tritonblas.grouped_gemm(group_a, group_b)
        run3 = tritonblas.grouped_gemm(group_a, group_b)

        for i in range(len(self.SHAPES)):
            assert torch.equal(run1[i], run2[i]), \
                f"Run1 vs Run2 differ on group {i}"
            assert torch.equal(run1[i], run3[i]), \
                f"Run1 vs Run3 differ on group {i}"


# ---------------------------------------------------------------------------
# Relative error with assert_close
# ---------------------------------------------------------------------------

class TestRelativeError:
    """Verify that torch.testing.assert_close passes with fp16 tolerances."""

    @pytest.mark.parametrize("m, n, k", [
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
    ])
    def test_fp16_assert_close(self, m, n, k):
        shapes = [(m, n, k)]
        results, refs = run_grouped_gemm(shapes, dtype=torch.float16)
        torch.testing.assert_close(
            results[0], refs[0], atol=FP16_ATOL, rtol=FP16_RTOL
        )

    @pytest.mark.parametrize("m, n, k", [
        (256, 256, 128),
        (512, 512, 256),
    ])
    def test_bf16_assert_close(self, m, n, k):
        shapes = [(m, n, k)]
        results, refs = run_grouped_gemm(shapes, dtype=torch.bfloat16)
        torch.testing.assert_close(
            results[0], refs[0], atol=BF16_ATOL, rtol=BF16_RTOL
        )


# ---------------------------------------------------------------------------
# Group count scaling
# ---------------------------------------------------------------------------

class TestGroupCountScaling:
    """Verify correctness as group count scales from 1 to 16."""

    BASE_SHAPE = (512, 512, 256)

    @pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16])
    def test_group_count_scaling(self, group_size):
        shapes = [self.BASE_SHAPE] * group_size
        results, refs = run_grouped_gemm(shapes)
        for i, (res, ref) in enumerate(zip(results, refs)):
            diff = (res - ref).abs()
            max_err = diff.max().item()
            assert max_err <= FP16_ATOL, (
                f"group_size={group_size}, group {i} failed — "
                f"max_abs_err={max_err:.4f} (limit={FP16_ATOL})"
            )

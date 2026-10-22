"""Property test: Origami must NEVER select a tile config that overflows LDS.

For every combination of:
- GEMM shape (M, N, K) covering small, medium, large, and the problematic 8192^3
- dtype (fp16, bf16, and fp8 when available)
- num_stages (2, 3, 4)
- architecture (MI300X 64KB, MI350X 160KB)

...the selected (tile, stages) config must fit in the hardware LDS limit.

This test catches the bug where Origami selects 256x256x128 stages=3 on MI355X
(needs 262KB, limit is 160KB) and similar overflow cases.
"""

import pytest
import torch

from tritonblas.origami import (
    OrigamiMatmulSelector,
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)


SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1024, 1024, 8192),
    (8192, 8192, 1024),
    (128, 256, 512),
    (4096, 4096, 8192),
    (16384, 16384, 4096),
]

DTYPES = [torch.float16, torch.bfloat16]
# Add FP8 coverage when available (requires ROCm with FP8 support)
if hasattr(torch, "float8_e4m3fnuz"):
    DTYPES.append(torch.float8_e4m3fnuz)
elif hasattr(torch, "float8_e4m3fn"):
    DTYPES.append(torch.float8_e4m3fn)

NUM_STAGES_VALUES = [2, 3, 4]

LDS_LIMITS = {
    "MI300X_64KB": 65536,
    "MI350X_160KB": 163840,
}


_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
if hasattr(torch, "float8_e4m3fnuz"):
    _DTYPE_NAMES[torch.float8_e4m3fnuz] = "fp8"
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_NAMES[torch.float8_e4m3fn] = "fp8"


def _bytes_per_elem(dtype):
    return torch.tensor([], dtype=dtype).element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestOrigamiNeverOverflowsLDS:
    """Origami must never select a config that exceeds the hardware LDS limit."""

    @pytest.fixture
    def hardware_lds(self):
        """Get the actual LDS capacity from the hardware."""
        try:
            import origami
        except ImportError:
            pytest.skip("origami not installed")
        hw = origami.get_hardware_for_device(torch.cuda.current_device())
        return hw.lds_capacity

    @pytest.mark.parametrize(
        "M,N,K",
        SHAPES,
        ids=[f"{m}x{n}x{k}" for m, n, k in SHAPES],
    )
    @pytest.mark.parametrize("dtype", DTYPES, ids=[_DTYPE_NAMES[d] for d in DTYPES])
    @pytest.mark.parametrize("num_stages", NUM_STAGES_VALUES, ids=[f"s{s}" for s in NUM_STAGES_VALUES])
    def test_selected_tile_fits_in_lds(self, M, N, K, dtype, num_stages, hardware_lds):
        """The tile Origami selects must fit in the actual hardware LDS."""
        bpe = _bytes_per_elem(dtype)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        selector = OrigamiMatmulSelector(
            M, N, K, dtype, dtype, dtype, device, num_stages=num_stages
        )

        bm = selector._result.config.mt.m
        bn = selector._result.config.mt.n
        bk = selector._result.config.mt.k
        # Use the clamped num_stages, not the requested value — the selector
        # may have reduced it to fit within the hardware LDS budget.
        clamped_stages = selector.num_stages

        actual_lds = estimate_triton_lds_bytes(bm, bn, bk, bpe, bpe, clamped_stages)

        assert actual_lds <= hardware_lds, (
            f"Origami selected {bm}x{bn}x{bk} stages={clamped_stages} "
            f"(requested {num_stages}) for {M}x{N}x{K} {dtype} "
            f"which needs {actual_lds} bytes LDS, but hardware limit is {hardware_lds} bytes "
            f"({hardware_lds // 1024} KB). "
            f"Origami must cap num_stages or reject this tile."
        )


class TestLdsCheckRejectsOverflow:
    """Pure formula tests — no GPU needed."""

    @pytest.mark.parametrize("lds_limit_name,lds_limit", LDS_LIMITS.items())
    @pytest.mark.parametrize("dtype", DTYPES, ids=[_DTYPE_NAMES[d] for d in DTYPES])
    @pytest.mark.parametrize("num_stages", NUM_STAGES_VALUES, ids=[f"s{s}" for s in NUM_STAGES_VALUES])
    def test_lds_check_consistent_with_estimate(self, lds_limit_name, lds_limit, dtype, num_stages):
        """check_triton_lds_capacity must agree with estimate_triton_lds_bytes.

        For every (tile, stages, lds_limit) combination, check_triton_lds_capacity
        must return True iff estimate_triton_lds_bytes <= lds_limit.
        """
        bpe = _bytes_per_elem(dtype)

        for bm in [64, 128, 256]:
            for bn in [64, 128, 256]:
                for bk in [32, 64, 128, 256]:
                    lds_needed = estimate_triton_lds_bytes(bm, bn, bk, bpe, bpe, num_stages)
                    passes_check = check_triton_lds_capacity(bm, bn, bk, bpe, bpe, lds_limit, num_stages)

                    if lds_needed > lds_limit:
                        assert not passes_check, (
                            f"check_triton_lds_capacity WRONGLY PASSED {bm}x{bn}x{bk} "
                            f"stages={num_stages} on {lds_limit_name}: needs {lds_needed} > {lds_limit}"
                        )
                    else:
                        assert passes_check, (
                            f"check_triton_lds_capacity WRONGLY REJECTED {bm}x{bn}x{bk} "
                            f"stages={num_stages} on {lds_limit_name}: needs {lds_needed} <= {lds_limit}"
                        )

    def test_problematic_config_rejected(self):
        """The specific config that caused the MI355X crash must be rejected.

        256x256x128 bf16 stages=3 needs 262144 bytes > 163840 (MI355X 160KB).
        """
        assert not check_triton_lds_capacity(256, 256, 128, 2, 2, 163840, 3)

    def test_problematic_config_fits_at_stages_2(self):
        """256x256x128 bf16 stages=2 should fit in 160KB LDS.

        (2-1) * (256*128*2 + 128*256*2) = 131072 <= 163840.
        """
        assert check_triton_lds_capacity(256, 256, 128, 2, 2, 163840, 2)

    def test_max_stages_formula_for_crash_config(self):
        """max_stages = floor(lds_cap / per_stage) + 1 = floor(163840/131072) + 1 = 2."""
        per_stage = 256 * 128 * 2 + 128 * 256 * 2  # 131072
        max_stages = int(163840 // per_stage) + 1
        assert max_stages == 2, f"Expected max_stages=2, got {max_stages}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestClampingWith160KBLDS:
    """Simulate MI355X 160KB LDS on any GPU to exercise the clamping path.

    The original crash: Origami selects 256x256x128 with stages=3 on MI355X,
    needing 262KB — far exceeding the 160KB LDS limit. The fix clamps
    num_stages after tile selection. This test patches hardware.lds_capacity
    to 160KB to exercise that code path on any available GPU.
    """

    @pytest.fixture(autouse=True)
    def patch_lds_capacity(self):
        """Patch hardware LDS to 160KB for the test, restore afterwards."""
        import origami
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        hw = origami.get_hardware_for_device(device.index)
        real_lds = hw.lds_capacity
        hw.lds_capacity = 163840  # 160KB, same as MI355X
        yield hw
        hw.lds_capacity = real_lds

    @pytest.mark.parametrize("num_stages", [2, 3, 4], ids=[f"s{s}" for s in [2, 3, 4]])
    def test_8192_clamp_at_160kb(self, num_stages, patch_lds_capacity):
        """8192^3 bf16 at 160KB LDS must never overflow, regardless of stages."""
        dtype = torch.bfloat16
        bpe = _bytes_per_elem(dtype)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        sel = OrigamiMatmulSelector(
            8192, 8192, 8192, dtype, dtype, dtype, device, num_stages=num_stages
        )
        usage = estimate_triton_lds_bytes(
            sel.block_m, sel.block_n, sel.block_k, bpe, bpe, sel.num_stages
        )
        assert usage <= 163840, (
            f"OVERFLOW at 160KB: {sel.block_m}x{sel.block_n}x{sel.block_k} "
            f"s={sel.num_stages} (requested {num_stages}) uses {usage} > 163840"
        )
        # num_stages must be at most the requested value
        assert sel.num_stages <= num_stages, (
            f"num_stages increased: requested {num_stages}, got {sel.num_stages}"
        )

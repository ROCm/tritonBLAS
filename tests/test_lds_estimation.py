"""Comprehensive unit tests for LDS (shared memory) estimation functions.

These functions are critical for preventing "out of resource: shared memory"
errors (issue #62). They estimate Triton's LDS usage with async_copy,
PaddedSharedEncoding, and num_stages pipelining.

Tests cover:
- _padded_size_32_4: [[32, 4]] pattern (fast path)
- _padded_size_elements_pow2: generic Triton getPaddedSize equivalent
- estimate_triton_lds_bytes: full LDS estimation with max() over padding patterns
- check_triton_lds_capacity: capacity check with short-circuit
- Architecture-dependent tests: use origami.get_hardware_for_device() to verify
  LDS capacity checks against real hardware (gfx942, gfx950, etc.)
"""

import pytest
import torch

# Import module-private functions for direct testing
from tritonblas.origami import (
    _padded_size_32_4,
    _padded_size_elements_pow2,
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)


def _get_hardware():
    """Get hardware from origami for current device. Returns hardware or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        import origami
        device_id = torch.cuda.current_device()
        return origami.get_hardware_for_device(device_id)
    except Exception:
        return None


class TestPaddedSize32_4:
    """Test _padded_size_32_4: Triton [[32, 4]] PaddedSharedEncoding pattern."""

    def test_zero_size(self):
        """Zero elements should return zero."""
        assert _padded_size_32_4(0) == 0

    def test_exact_divisibility_by_32(self):
        """When size is exactly divisible by 32, Triton subtracts one padding block."""
        # 32 elements: 1 block, 4 padding, then subtract 4 -> 0 padding
        assert _padded_size_32_4(32) == 32
        # 64 elements: 2 blocks, 8 padding, subtract 4 -> 4 padding
        assert _padded_size_32_4(64) == 68
        # 96 elements: 3 blocks, 12 padding, subtract 4 -> 8 padding
        assert _padded_size_32_4(96) == 104
        # 2048 elements: 64 blocks, 256 padding, subtract 4 -> 252 padding
        assert _padded_size_32_4(2048) == 2300

    def test_not_divisible_by_32(self):
        """When size is not divisible by 32, no subtraction occurs."""
        # 31: 0 full blocks, 0 padding
        assert _padded_size_32_4(31) == 31
        # 33: 1 block, 4 padding, 33%32 != 0 so no subtract
        assert _padded_size_32_4(33) == 37
        # 63: 1 block, 4 padding
        assert _padded_size_32_4(63) == 67

    def test_various_tile_sizes(self):
        """Common matmul tile element counts."""
        # 128*64 = 8192 (block_m*block_k for A)
        assert _padded_size_32_4(8192) == 9212
        # 64*128 = 8192 (block_k*block_n for B)
        assert _padded_size_32_4(8192) == 9212
        # 256*64 = 16384
        assert _padded_size_32_4(16384) == 18428


class TestPaddedSizeElementsPow2:
    """Test _padded_size_elements_pow2: generic Triton getPaddedSize."""

    def test_matches_32_4_for_same_pattern(self):
        """[[32, 4]] via generic should match _padded_size_32_4."""
        for size in [0, 31, 32, 33, 64, 63, 96, 2048, 8192, 16384]:
            assert _padded_size_elements_pow2(size, [(32, 4)]) == _padded_size_32_4(size)

    def test_block_k_8_pattern(self):
        """[[block_k, 8]] pattern for A tile (e.g. block_k=64)."""
        # elem_a=8192, interval=64, padding=8
        # blocks=128, block_padding=128*8=1024, 8192%64=0 so subtract 8 -> 1016
        assert _padded_size_elements_pow2(8192, [(64, 8)]) == 9208

    def test_block_n_8_pattern(self):
        """[[block_n, 8]] pattern for B tile (e.g. block_n=128)."""
        # interval=128, padding=8: (8192>>7)<<3 = 64<<3 = 512, 8192%128=0, subtract 8 -> 504
        assert _padded_size_elements_pow2(8192, [(128, 8)]) == 8696

    def test_multiple_intervals(self):
        """Multiple intervals sum padding (Triton can chain encodings)."""
        # Single (32,4) for 64
        single = _padded_size_elements_pow2(64, [(32, 4)])
        assert single == 68

    def test_edge_case_small_padding(self):
        """Small padding values."""
        # (32, 1): log2_padding=0, block_padding = (64>>5)<<0 = 2
        # 64%32=0, 2>=1, subtract 1 -> 1. return 65
        assert _padded_size_elements_pow2(64, [(32, 1)]) == 65
        # (32, 2): log2_padding=1, block_padding = (64>>5)<<1 = 4. 64%32=0, 4>=2, subtract 2 -> 2. return 66
        assert _padded_size_elements_pow2(64, [(32, 2)]) == 66


class TestEstimateTritonLdsBytes:
    """Test estimate_triton_lds_bytes: full LDS estimation."""

    def test_at_least_raw_size(self):
        """Estimate must be >= raw (unpadded) size."""
        block_m, block_n, block_k = 128, 128, 64
        bytes_a, bytes_b = 2, 2  # bf16
        raw = (block_m * block_k * bytes_a + block_k * block_n * bytes_b) * 2
        lds = estimate_triton_lds_bytes(block_m, block_n, block_k, bytes_a, bytes_b, 2)
        assert lds >= raw

    def test_num_stages_scaling(self):
        """LDS scales linearly with num_stages."""
        block_m, block_n, block_k = 128, 128, 64
        bytes_a, bytes_b = 2, 2
        lds_2 = estimate_triton_lds_bytes(block_m, block_n, block_k, bytes_a, bytes_b, 2)
        lds_3 = estimate_triton_lds_bytes(block_m, block_n, block_k, bytes_a, bytes_b, 3)
        lds_4 = estimate_triton_lds_bytes(block_m, block_n, block_k, bytes_a, bytes_b, 4)
        assert lds_3 == lds_2 * 3 // 2
        assert lds_4 == lds_2 * 2

    def test_max_logic_32_4_dominates(self):
        """When [[32, 4]] produces larger padding than [[block_k, 8]]."""
        # For 128x128x64: elem_a=8192, elem_b=8192
        # [[32,4]]: 9212 each. [[64,8]]: 9208 each. So [[32,4]] wins.
        block_m, block_n, block_k = 128, 128, 64
        lds = estimate_triton_lds_bytes(block_m, block_n, block_k, 2, 2, 2)
        # 2 * (9212*2 + 9212*2) = 2 * 36848 = 73696
        assert lds == 73696

    def test_max_logic_block_pattern_dominates(self):
        """When [[block_k, 8]] produces larger padding for A than [[32, 4]]."""
        # block_k=16: elem_a=2048. [[32,4]]: 2300. [[16,8]]: 3064 -> A uses 3064
        # block_n=128: elem_b=2048. [[32,4]]: 2300. [[128,8]]: 2168 -> B uses 2300
        block_m, block_n, block_k = 128, 128, 16
        lds = estimate_triton_lds_bytes(block_m, block_n, block_k, 2, 2, 2)
        # per_stage = 3064*2 + 2300*2 = 10728, total = 21456
        assert lds == 21456

    def test_known_config_128x128x64_bf16(self):
        """Known config: 128x128x64, bf16, num_stages=2."""
        lds = estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2)
        assert lds == 73696

    def test_known_config_64x64x32_bf16(self):
        """Known config: 64x64x32, bf16, num_stages=2."""
        lds = estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2)
        # [[32,4]]: 2300. [[32,8]] for A (block_k=32): 2552. [[64,8]] for B (block_n=64): 2296.
        # A uses max(2300, 2552)=2552, B uses max(2300, 2296)=2300. per_stage=9704, total=19408
        assert lds == 19408

    def test_fp16_same_as_bf16(self):
        """FP16 and BF16 both use 2 bytes per element."""
        block_m, block_n, block_k = 64, 64, 32
        lds_bf16 = estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2)
        lds_fp16 = estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2)
        assert lds_bf16 == lds_fp16

    def test_fp32_doubles_bf16(self):
        """FP32 uses 4 bytes, so LDS ~2x bf16 for same tile."""
        block_m, block_n, block_k = 64, 64, 32
        lds_bf16 = estimate_triton_lds_bytes(64, 64, 32, 2, 2, 2)
        lds_fp32 = estimate_triton_lds_bytes(64, 64, 32, 4, 4, 2)
        assert lds_fp32 == lds_bf16 * 2


class TestCheckTritonLdsCapacity:
    """Test check_triton_lds_capacity: capacity check with short-circuit."""

    def test_fits_within_capacity(self):
        """Config that fits should return True."""
        # 64x64x32 bf16 uses ~18400 bytes
        assert check_triton_lds_capacity(64, 64, 32, 2, 2, 65536, 2) is True

    def test_exceeds_capacity(self):
        """Config that exceeds capacity should return False."""
        # 256x256x64 bf16 is large
        assert check_triton_lds_capacity(256, 256, 64, 2, 2, 65536, 2) is False

    def test_short_circuit_raw_exceeds(self):
        """When raw size exceeds capacity, should return False without full padded calc."""
        # Very large tile - raw alone exceeds
        assert check_triton_lds_capacity(512, 512, 256, 2, 2, 1000, 2) is False

    def test_num_stages_affects_capacity(self):
        """Higher num_stages can push config over capacity."""
        # 128x128x64: 73696 bytes (2 stages), 147392 (4 stages). Use capacity 100000.
        fits_2 = check_triton_lds_capacity(128, 128, 64, 2, 2, 100000, 2)
        fits_4 = check_triton_lds_capacity(128, 128, 64, 2, 2, 100000, 4)
        assert fits_2 is True
        assert fits_4 is False


# N_CU -> expected LDS capacity (from origami common.hpp / hardware)
# gfx950: 256 CUs, 160KB; gfx942/gfx90a: 64KB
N_CU_TO_LDS = {256: 163840, 304: 65536, 80: 65536, 64: 65536, 228: 65536, 104: 65536}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLdsArchitectureDependent:
    """Architecture-dependent LDS tests using origami.get_hardware_for_device().

    These tests verify that check_triton_lds_capacity correctly filters configs
    based on the actual GPU's LDS capacity. Uses hardware.N_CU for arch detection.
    """

    def test_hardware_lds_capacity_matches_expected(self):
        """Detected hardware lds_capacity should match known N_CU values."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        n_cu = hardware.N_CU
        if n_cu not in N_CU_TO_LDS:
            pytest.skip(f"N_CU={n_cu} not in known LDS capacity map")
        expected = N_CU_TO_LDS[n_cu]
        assert hardware.lds_capacity == expected, (
            f"N_CU={n_cu}: expected lds_capacity={expected}, got {hardware.lds_capacity}"
        )

    def test_small_config_fits_on_all_archs(self):
        """64x64x32 bf16 should fit within LDS on all supported archs."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        fits = check_triton_lds_capacity(64, 64, 32, 2, 2, lds_cap, 2)
        assert fits, f"64x64x32 should fit in {lds_cap} bytes (N_CU={hardware.N_CU})"

    def test_64kb_arch_128x128x64_exceeds_lds(self):
        """On 64KB LDS (N_CU in 304,80,64,228,104), 128x128x64 bf16 should exceed LDS."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        if lds_cap != 65536:
            pytest.skip(f"Requires 64KB LDS, got {lds_cap} (N_CU={hardware.N_CU})")
        fits = check_triton_lds_capacity(128, 128, 64, 2, 2, lds_cap, 2)
        assert not fits, f"128x128x64 should NOT fit in 64KB"

    def test_160kb_arch_128x128x64_fits_lds(self):
        """On 160KB LDS (N_CU=256), 128x128x64 bf16 should fit in LDS."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        lds_cap = hardware.lds_capacity
        if lds_cap != 163840:
            pytest.skip(f"Requires 160KB LDS, got {lds_cap} (N_CU={hardware.N_CU})")
        fits = check_triton_lds_capacity(128, 128, 64, 2, 2, lds_cap, 2)
        assert fits, f"128x128x64 should fit in 160KB"

    def test_selector_filters_by_lds_on_current_arch(self):
        """OrigamiMatmulSelector should not select configs that exceed LDS."""
        hardware = _get_hardware()
        if hardware is None:
            pytest.skip("Could not get hardware")
        from tritonblas import OrigamiMatmulSelector
        m, n, k = 4096, 4096, 4096
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        selector = OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, device)
        usage = estimate_triton_lds_bytes(
            selector.block_m, selector.block_n, selector.block_k,
            2, 2, selector.num_stages
        )
        assert usage <= hardware.lds_capacity, (
            f"Selector chose {selector.block_m}x{selector.block_n}x{selector.block_k} "
            f"(LDS={usage}) which exceeds {hardware.lds_capacity} (N_CU={hardware.N_CU})"
        )

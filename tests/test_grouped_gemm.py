import pytest
import torch
import tritonblas
from tritonblas.origami import GroupedGemmSelector


# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_triton_cache():
    """Yield before test; no cache clearing needed per-test since kernels are stable."""
    yield


# --- Helpers ---

def run_grouped_gemm(shapes, dtype=torch.float16, BLK_M=128, BLK_N=128, BLK_K=64, use_origami=False):
    """Run grouped GEMM and return (results, references, max_errors)."""
    group_a, group_b, group_c, refs = [], [], [], []
    for m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        c = torch.empty(m, n, device="cuda", dtype=dtype)
        group_a.append(a)
        group_b.append(b)
        group_c.append(c)
        refs.append(torch.matmul(a, b))

    if use_origami:
        result = tritonblas.grouped_gemm(group_a, group_b)
    else:
        result = tritonblas.grouped_gemm(group_a, group_b, group_c, BLK_M, BLK_N, BLK_K)

    max_errors = []
    for i in range(len(shapes)):
        max_errors.append((result[i] - refs[i]).abs().max().item())

    return result, refs, max_errors


def get_atol(dtype):
    """Reasonable absolute tolerance for GEMM given dtype."""
    if dtype == torch.float16:
        return 0.5
    elif dtype == torch.bfloat16:
        return 1.0
    elif dtype == torch.float32:
        return 1e-3
    return 1.0


# --- Tests: Single Group ---

class TestSingleGroup:
    @pytest.mark.parametrize("m, n, k", [
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
    ])
    def test_aligned_shapes(self, m, n, k):
        _, _, max_errors = run_grouped_gemm([(m, n, k)])
        assert max_errors[0] < get_atol(torch.float16), f"max_err={max_errors[0]}"

    @pytest.mark.parametrize("m, n, k", [
        (127, 65, 33),
        (1, 64, 128),
        (255, 257, 63),
        (17, 31, 47),
    ])
    def test_non_aligned_shapes(self, m, n, k):
        _, _, max_errors = run_grouped_gemm([(m, n, k)])
        assert max_errors[0] < get_atol(torch.float16), f"max_err={max_errors[0]}"

    @pytest.mark.parametrize("m, n, k", [
        (64, 64, 64),
        (128, 256, 512),
        (256, 64, 128),
    ])
    def test_rectangular_shapes(self, m, n, k):
        _, _, max_errors = run_grouped_gemm([(m, n, k)])
        assert max_errors[0] < get_atol(torch.float16), f"max_err={max_errors[0]}"


# --- Tests: Multi Group ---

class TestMultiGroup:
    @pytest.mark.parametrize("group_size", [2, 4, 8])
    def test_same_shape_groups(self, group_size):
        shapes = [(256, 256, 128)] * group_size
        _, _, max_errors = run_grouped_gemm(shapes)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.float16), f"Group {i}: max_err={err}"

    def test_mixed_shape_groups(self):
        shapes = [(256, 512, 128), (127, 65, 33), (512, 256, 64), (64, 128, 256)]
        _, _, max_errors = run_grouped_gemm(shapes)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.float16), f"Group {i}: max_err={err}"

    def test_varied_sizes(self):
        shapes = [(128, 128, 64), (256, 256, 128), (512, 512, 256)]
        _, _, max_errors = run_grouped_gemm(shapes)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.float16), f"Group {i}: max_err={err}"


# --- Tests: Dtypes ---

class TestDtypes:
    def test_fp16(self):
        shapes = [(256, 256, 128), (128, 256, 64)]
        _, _, max_errors = run_grouped_gemm(shapes, dtype=torch.float16)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.float16), f"Group {i}: max_err={err}"

    def test_bf16(self):
        shapes = [(256, 256, 128), (128, 256, 64)]
        _, _, max_errors = run_grouped_gemm(shapes, dtype=torch.bfloat16)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.bfloat16), f"Group {i}: max_err={err}"


# --- Tests: Origami Integration ---

class TestOrigamiIntegration:
    def test_grouped_gemm_selector(self):
        sel = GroupedGemmSelector(
            [(512, 512, 256), (128, 256, 64)],
            torch.float16, torch.float16, torch.float16,
        )
        assert sel.block_m > 0
        assert sel.block_n > 0
        assert sel.block_k > 0

    def test_auto_tile_selection(self):
        shapes = [(256, 256, 128), (128, 256, 64)]
        _, _, max_errors = run_grouped_gemm(shapes, use_origami=True)
        for i, err in enumerate(max_errors):
            assert err < get_atol(torch.float16), f"Group {i}: max_err={err}"

    def test_auto_output_allocation(self):
        """Test that grouped_gemm allocates output if group_c is None."""
        a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a, b)
        result = tritonblas.grouped_gemm([a], [b])
        assert result[0].shape == (128, 128)
        max_err = (result[0] - ref).abs().max().item()
        assert max_err < get_atol(torch.float16), f"max_err={max_err}"

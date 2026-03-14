import itertools
import torch
import origami
import math
from math import ceil


# https://docs.pytorch.org/docs/stable/tensors.html
dtype_to_str = {
    torch.float32: "f32",
    torch.complex64: "c32",
    torch.complex128: "c64",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.int32: "i32",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.float8_e5m2: "f8",
    torch.float8_e4m3fn: "f8",
}
# Add FP8 FNUZ variants if available (for non-gfx950 architectures)
if hasattr(torch, "float8_e5m2fnuz"):
    dtype_to_str[torch.float8_e5m2fnuz] = "f8"
if hasattr(torch, "float8_e4m3fnuz"):
    dtype_to_str[torch.float8_e4m3fnuz] = "f8"


def _get_dtype_bits(dtype):
    """Get bit size for both float and int dtypes."""
    try:
        return torch.finfo(dtype).bits
    except TypeError:
        return torch.iinfo(dtype).bits


def _infer_mi_dimensions(hardware, a_bitsize, b_bitsize, block_mn_range, block_k_range):
    """Infer matrix instruction dimensions and adjust block ranges based on hardware and dtype.

    Returns (mi_dim, block_mn_range, block_k_range).
    """
    largest_bitsize = max(a_bitsize, b_bitsize)
    mi_dim = None

    # gfx950 (MI350X/MI355X)
    if hardware.N_CU == 256:
        if largest_bitsize == 32:
            mi_dim = origami.dim3_t(16, 16, 4)
        elif largest_bitsize == 16:
            mi_dim = origami.dim3_t(16, 16, 32)
        elif largest_bitsize <= 8:
            mi_dim = origami.dim3_t(16, 16, 128)

    # gfx942 (MI300X: 304 CUs, partitioned: 80/64 CUs)
    is_gfx942 = hardware.N_CU in [304, 80, 64]
    if is_gfx942:
        if largest_bitsize == 32:
            mi_dim = origami.dim3_t(16, 16, 4)
        elif largest_bitsize == 16:
            mi_dim = origami.dim3_t(16, 16, 16)
        elif largest_bitsize == 8:
            block_mn_range = block_mn_range + [512]
            block_k_range = block_k_range + [128, 256]
            mi_dim = origami.dim3_t(16, 16, 32)
        elif largest_bitsize < 8:
            raise ValueError("gfx942 doesn't support F4/F6")

    # MI300A (228 CUs)
    if hardware.N_CU == 228:
        if largest_bitsize == 32:
            mi_dim = origami.dim3_t(16, 16, 4)
        elif largest_bitsize == 16:
            mi_dim = origami.dim3_t(16, 16, 16)
        elif largest_bitsize == 8:
            block_mn_range = block_mn_range + [512]
            block_k_range = block_k_range + [128, 256]
            mi_dim = origami.dim3_t(16, 16, 32)
        elif largest_bitsize < 8:
            raise ValueError("MI300A doesn't support F4/F6")

    # gfx90a (MI200)
    if hardware.N_CU == 104:
        if largest_bitsize == 32:
            mi_dim = origami.dim3_t(16, 16, 4)
        elif largest_bitsize == 16:
            mi_dim = origami.dim3_t(16, 16, 16)
        elif largest_bitsize == 8:
            raise ValueError("MI200 doesn't support F8")
        elif largest_bitsize < 8:
            raise ValueError("MI200 doesn't support F4/F6")

    if mi_dim is None:
        raise ValueError(
            f"No valid matrix instruction for {a_bitsize}-bit/{b_bitsize}-bit dtypes"
        )

    return mi_dim, block_mn_range, block_k_range


def _generate_configs(block_mn_range, block_k_range, mi_dim, occupancy_range=None, streamk=False):
    """Generate a list of origami config_t objects from block size ranges."""
    if occupancy_range is None:
        occupancy_range = [1]
    configs = []
    for blk_m, blk_n, blk_k, occ in itertools.product(
        block_mn_range, block_mn_range, block_k_range, occupancy_range
    ):
        cfg = origami.config_t()
        cfg.mt = origami.dim3_t(blk_m, blk_n, blk_k)
        cfg.mi = mi_dim
        cfg.occupancy = occ
        if streamk:
            cfg.grid_selection = origami.grid_selection_t.k_split_aware
        else:
            cfg.grid_selection = origami.grid_selection_t.data_parallel
        configs.append(cfg)
    return configs


def _make_problem(m, n, k, a_dtype_str, b_dtype_str, out_dtype_str, mx_block_size=0):
    """Create an origami problem_t from shape and dtype strings."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype(a_dtype_str)
    problem.b_dtype = origami.string_to_datatype(b_dtype_str)
    problem.c_dtype = origami.string_to_datatype(out_dtype_str)
    problem.d_dtype = origami.string_to_datatype(out_dtype_str)
    problem.mi_dtype = origami.string_to_datatype(out_dtype_str)
    problem.a_mx_block_size = mx_block_size
    problem.b_mx_block_size = mx_block_size
    return problem


class MatmulHeuristicResult:
    """Analytical GEMM config selector using the modern origami API.

    Wraps origami.select_config() and origami.select_workgroup_mapping() to choose
    optimal tile sizes (block_m/n/k) and workgroup mapping for a single GEMM problem.
    """

    def __init__(
        self,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        c_dtype,
        device_index=0,
        mx_block_size=0,
        streamk=True,
    ):
        self.m = m
        self.n = n
        self.k = k

        self.hardware = origami.get_hardware_for_device(device_index)

        self.element_size_A = _get_dtype_bits(a_dtype)
        self.element_size_B = _get_dtype_bits(b_dtype)
        self.element_size_out = _get_dtype_bits(c_dtype)

        # Resolve MI dtype for instruction lookup (use narrower input)
        input_dtype_for_mi = a_dtype if self.element_size_A <= self.element_size_B else b_dtype
        self.mi_dtype = dtype_to_str.get(input_dtype_for_mi, dtype_to_str.get(c_dtype))
        self._out_dtype_str = dtype_to_str.get(c_dtype)
        self._a_dtype_str = dtype_to_str.get(a_dtype)
        self._b_dtype_str = dtype_to_str.get(b_dtype)

        self.mx_block_size = mx_block_size

        # Infer MI dimensions and adjust block ranges
        block_mn_range = [16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256, 512]
        mi_dim, block_mn_range, block_k_range = _infer_mi_dimensions(
            self.hardware, self.element_size_A, self.element_size_B,
            block_mn_range, block_k_range
        )

        # Generate configs and select best
        configs = _generate_configs(block_mn_range, block_k_range, mi_dim, streamk=streamk)
        problem = _make_problem(m, n, k, self._a_dtype_str, self._b_dtype_str,
                                self._out_dtype_str, mx_block_size)

        self._result = origami.select_config(problem, self.hardware, configs)

        # Grid model constants
        self.split_factors = [8, 6, 4, 3, 2, 1]
        self.tile_fractions = [0.0, 0.5, 0.125, 0.2, 0.25, 1.0 / 3.0]
        self.max_workspace = 128 * 1024 * 1024

        # Compute workgroup mapping
        if streamk:
            self.grid = self._compute_sk_grid()
        else:
            self.grid = self.hardware.N_CU

        wgm_result = origami.select_workgroup_mapping(
            problem, self.hardware, self._result.config, self.grid
        )
        self._xcc_wgm = wgm_result.wgmxcc
        self._wgm = wgm_result.wgm

        # Store as tuple for backward compatibility
        self.config = (
            self._result.config.mt.m,
            self._result.config.mt.n,
            self._result.config.mt.k,
            self._wgm,
        )

    def get_config(self):
        return self.config

    def get_grid(self):
        return self.grid

    @property
    def block_m(self):
        return self._result.config.mt.m

    @property
    def block_n(self):
        return self._result.config.mt.n

    @property
    def block_k(self):
        return self._result.config.mt.k

    @property
    def group_m(self):
        return self._wgm

    def partial_tile_size(self, sk_grid):
        BLK_M, BLK_N = self.block_m, self.block_n
        bytes_per_elem = self.element_size_out // 8
        return BLK_M * BLK_N * bytes_per_elem * sk_grid

    def _compute_sk_grid(self):
        M, N, K = self.m, self.n, self.k
        BLK_M, BLK_N, BLK_K = self.block_m, self.block_n, self.block_k
        cu_count = self.hardware.N_CU

        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        if tiles > cu_count:
            virt_cu_count = cu_count
            min_even_tiles = tiles / virt_cu_count
            for frac in self.tile_fractions:
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)
                if tiles % frac_grid != 0 and self.partial_tile_size(frac_grid) > self.max_workspace:
                    continue
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break
        elif tiles < cu_count:
            for factor in self.split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor
                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            is_gfx942 = cu_count in [304, 80, 64]
            if last_wave_remainder < 128 and last_wave_remainder > 0 and is_gfx942:
                sk_grid = 256 if cu_count == 304 else 64

        return sk_grid


class GroupedGemmSelector:
    """Analytical config selector for grouped GEMM using origami's grouped prediction.

    Uses origami.select_config_grouped() to find the optimal single tile configuration
    across all groups in a grouped GEMM problem.
    """

    def __init__(self, group_shapes, a_dtype, b_dtype, c_dtype, device_index=0):
        """
        Args:
            group_shapes: List of (M, N, K) tuples, one per group.
            a_dtype: torch dtype for A matrices.
            b_dtype: torch dtype for B matrices.
            c_dtype: torch dtype for C matrices.
            device_index: CUDA device index.
        """
        self.group_shapes = group_shapes
        self.hardware = origami.get_hardware_for_device(device_index)

        a_bits = _get_dtype_bits(a_dtype)
        b_bits = _get_dtype_bits(b_dtype)

        a_dtype_str = dtype_to_str.get(a_dtype)
        b_dtype_str = dtype_to_str.get(b_dtype)
        c_dtype_str = dtype_to_str.get(c_dtype)

        # Infer MI dimensions
        block_mn_range = [16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256, 512]
        mi_dim, block_mn_range, block_k_range = _infer_mi_dimensions(
            self.hardware, a_bits, b_bits, block_mn_range, block_k_range
        )

        # Generate configs
        configs = _generate_configs(block_mn_range, block_k_range, mi_dim)

        # Build grouped problem (assign list at once - nanobind returns a copy on .groups access)
        grouped_problem = origami.grouped_problem_t()
        problems = []
        for m, n, k in group_shapes:
            problems.append(_make_problem(m, n, k, a_dtype_str, b_dtype_str, c_dtype_str))
        grouped_problem.groups = problems

        # Select best config for the grouped problem
        self._result = origami.select_config_grouped(
            grouped_problem, self.hardware, configs
        )

    @property
    def block_m(self):
        return self._result.config.mt.m

    @property
    def block_n(self):
        return self._result.config.mt.n

    @property
    def block_k(self):
        return self._result.config.mt.k

    def get_config(self):
        """Return (BLK_M, BLK_N, BLK_K) tuple."""
        return (self.block_m, self.block_n, self.block_k)

import itertools
import torch
import origami
import math
from math import ceil


def _padded_size_32_4(unpadded_size: int) -> int:
    """
    Fast path: Triton [[32, 4]] PaddedSharedEncoding pattern.
    Inserts 4 padding elements after every 32 elements for bank-conflict avoidance.
    interval=32 (log2=5), padding=4 (log2=2).
    """
    # Number of 32-element blocks × 4 padding per block
    block_padding = (unpadded_size >> 5) << 2
    # Triton subtracts one padding block when size is exactly divisible by 32
    if (unpadded_size & 31) == 0 and block_padding >= 4:
        block_padding -= 4
    return unpadded_size + block_padding


def _padded_size_elements_pow2(unpadded_size: int, intervals: list[tuple[int, int]]) -> int:
    """
    Triton PaddedSharedEncodingAttr.getPaddedSize - exact C++ equivalent.
    interval and padding must be powers of two. Uses bit ops for efficiency.
    """
    total_padding = 0
    for interval, padding in intervals:
        log2_interval = (interval - 1).bit_length()
        log2_padding = (padding - 1).bit_length() if padding else 0
        block_padding = (unpadded_size >> log2_interval) << log2_padding
        if unpadded_size % interval == 0 and block_padding >= padding:
            block_padding -= padding
        total_padding += block_padding
    return unpadded_size + total_padding


def estimate_triton_lds_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: int,
    bytes_b: int,
    num_stages: int = 2,
) -> int:
    """
    Estimate Triton kernel LDS (shared memory) usage in bytes.

    Triton uses async_copy by default for matmul, which applies
    PaddedSharedEncoding for bank-conflict avoidance. This function uses the
    exact Triton formula.

    PipeliningUtility.createAlloc: num_stages buffers per tile.
    PaddedSharedEncoding [[32, 4]]: insert 4 elements after every 32
    (Dialect.cpp getPaddedSize).

    Args:
        block_m, block_n, block_k: Tile dimensions (MT_M, MT_N, MT_K).
        bytes_a, bytes_b: Bytes per element for A and B (e.g. 2 for bf16/fp16).
        num_stages: Pipeline stages (2 or 3); Triton matmul uses 2 by default.

    Returns:
        Estimated total LDS usage in bytes.
    """
    elem_a = block_m * block_k
    elem_b = block_k * block_n

    # Fast path: [[32, 4]] (no list alloc, no loop)
    padded_elems_a = _padded_size_32_4(elem_a)
    padded_elems_b = _padded_size_32_4(elem_b)
    # [[block_k, 8]] / [[block_n, 8]] for small tiles (Triton AMD tests)
    if block_k & (block_k - 1) == 0:  # power of 2
        padded_a_bk = _padded_size_elements_pow2(elem_a, [(block_k, 8)])
        if padded_a_bk > padded_elems_a:
            padded_elems_a = padded_a_bk
    if block_n & (block_n - 1) == 0:
        padded_b_bn = _padded_size_elements_pow2(elem_b, [(block_n, 8)])
        if padded_b_bn > padded_elems_b:
            padded_elems_b = padded_b_bn
    padded_per_stage = padded_elems_a * bytes_a + padded_elems_b * bytes_b

    return num_stages * padded_per_stage


def check_triton_lds_capacity(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: int,
    bytes_b: int,
    lds_capacity: int,
    num_stages: int = 2,
) -> bool:
    """Return True if estimated Triton LDS usage fits within lds_capacity."""
    raw = (block_m * block_k * bytes_a + block_k * block_n * bytes_b) * num_stages
    if raw > lds_capacity:
        return False
    usage = estimate_triton_lds_bytes(
        block_m, block_n, block_k, bytes_a, bytes_b, num_stages
    )
    return usage <= lds_capacity


class OrigamiMatmulSelector:
    @staticmethod
    def estimate_triton_lds(
        block_m: int,
        block_n: int,
        block_k: int,
        bytes_a: int,
        bytes_b: int,
        num_stages: int = 2,
    ) -> int:
        """Class-level wrapper for estimate_triton_lds_bytes."""
        return estimate_triton_lds_bytes(
            block_m, block_n, block_k, bytes_a, bytes_b, num_stages
        )

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

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        mx_block_size=0,
        streamk=False,
        num_stages: int = 2,
    ):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k
        self.streamk = streamk
        self._num_stages = num_stages
        # Save tensor dtypes as strings
        self._a_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, a_dtype)
        self._b_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, b_dtype)
        self._out_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(
            out_dtype, out_dtype
        )

        # Save MX block size
        self._mx_block_size = mx_block_size

        #####
        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = ["f4"]

        def get_dtype_bits(dtype):
            # Handle MX types (string-based)
            if dtype in mx_types:
                return origami.datatype_to_bits(origami.string_to_datatype(dtype))

            # Handle torch dtypes
            try:
                return torch.finfo(dtype).bits
            except TypeError:
                return torch.iinfo(dtype).bits

        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(a_dtype)
        self._out_dtype_bitsize = get_dtype_bits(a_dtype)

        # For matrix instruction latency lookup, use input dtype (not output dtype)
        # because the matrix instruction type is determined by input operand types
        # Example: FP8 inputs with BF16 output still uses FP8 matrix instructions
        # Set MI dtype - use string for MX types, otherwise lookup from dict
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            input_dtype_for_mi = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
            self.mi_dtype = OrigamiMatmulSelector.dtype_to_str.get(
                input_dtype_for_mi, OrigamiMatmulSelector.dtype_to_str.get(out_dtype)
            )
        #####

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)
        self._N_CU = self._hardware.N_CU

        # Create list of Origami config_t objects from defaults.
        self._block_mn_range = [16, 32, 64, 128, 256]
        self._block_k_range = [16, 32, 64, 128, 256, 512]
        self._kernel_occupancy_range = [1]
        self._configs = self._generate_default_configs()

        # Create Origami problem_t based on problem metadata (needed for fallback)
        self._problem = self._make_problem()

        # Filter configs by Triton LDS capacity (async_copy + num_stages + padding).
        # Origami's check_lds_capacity uses raw tile size only; Triton allocates
        # num_stages buffers with padding for bank conflicts.
        bytes_a = self._a_dtype_bitsize // 8
        bytes_b = self._b_dtype_bitsize // 8
        lds_cap = self._hardware.lds_capacity
        self._configs = [
            c
            for c in self._configs
            if check_triton_lds_capacity(
                c.mt.m, c.mt.n, c.mt.k, bytes_a, bytes_b, lds_cap, self._num_stages
            )
        ]
        if not self._configs:
            # Fallback: allow at least origami's raw check (no Triton multiplier)
            self._configs = self._generate_default_configs()
            self._configs = [
                c
                for c in self._configs
                if origami.check_lds_capacity(
                    self._hardware, c.mt, self._problem.a_dtype, self._problem.b_dtype
                )
            ]
        if not self._configs:
            # Last resort: use minimal tile (16x16x16) so selection can proceed
            minimal = origami.config_t()
            minimal.mt = origami.dim3_t(16, 16, 16)
            minimal.mi = self._infer_matrix_instruction_dimensions()
            minimal.occupancy = 1
            minimal.grid_selection = (
                origami.grid_selection_t.k_split_aware if self.streamk
                else origami.grid_selection_t.data_parallel
            )
            self._configs = [minimal]

        # Run Origami solution selection
        self._result = origami.select_config(
            self._problem, self._hardware, self._configs
        )

        if streamk:
            self._grid = self._compute_sk_grid()
        else:
            self._grid = self._hardware.N_CU

        # select_workgroup_mapping returns workgroup_mapping_t (wgmxccchunk, wgmxcc, wgm)
        wgm_result = origami.select_workgroup_mapping(
            self._problem, self._hardware, self._result.config, self._grid
        )
        self._xcc_workgroup_mapping = wgm_result.wgmxcc
        self._workgroup_mapping = abs(wgm_result.wgm)  # wgm can be negative for M-major
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
        return self._workgroup_mapping

    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def waves_per_eu(self):
        return self._result.config.occupancy

    @property
    def even_k(self):
        return self._k % self.block_k == 0

    @property
    def sk_grid(self):
        return self._grid

    def _compute_sk_grid(self):
        # Grid model constants for StreamK
        split_factors = [8, 6, 4, 3, 2, 1]
        tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
        max_workspace = 128 * 1024 * 1024

        M, N, K = self._m, self._n, self._k
        BLK_M, BLK_N, BLK_K = self.block_m, self.block_n, self.block_k
        cu_count = self._hardware.N_CU

        # Fallback if no better fractional split is found
        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            # if size_mapping.CUOccupancy > 1:
            # virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (
                    tiles % frac_grid != 0
                    and self._partial_tile_size(frac_grid) > max_workspace
                ):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            last_wave_occupancy = last_wave_remainder / cu_count

            # Really bad last wave, which would have originally been compensated for
            # by changing tile size, but triton tile sizes are limited
            if (
                last_wave_remainder < 128
                and last_wave_remainder > 0
                and cu_count in [304, 80, 64]
            ):  # gfx942
                sk_grid = 256 if cu_count == 304 else 64
        return sk_grid

    def _partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N = self.block_m, self.block_n

        # bytes per C element
        bytes_per_elem = self._out_dtype_bitsize // 8

        # size of one partial tile per WG
        tile_size = BLK_M * BLK_N * bytes_per_elem

        # scale by the number of partial‑tiles per WG
        return tile_size * sk_grid

    def _generate_default_configs(self):
        config_list = []

        mi = self._infer_matrix_instruction_dimensions()

        for blk_m, blk_n, blk_k, occupancy in itertools.product(
            self._block_mn_range,
            self._block_mn_range,
            self._block_k_range,
            self._kernel_occupancy_range,
        ):
            # Create special dim3_t object for BLK_* sizes
            mt = origami.dim3_t(blk_m, blk_n, blk_k)

            # Create and set new config_t values
            new_config = origami.config_t()
            new_config.mt = mt
            new_config.mi = mi
            new_config.occupancy = occupancy
            if self.streamk:
                new_config.grid_selection = origami.grid_selection_t.k_split_aware
            else:
                new_config.grid_selection = origami.grid_selection_t.data_parallel
            config_list.append(new_config)

        return config_list

    def _make_problem(self) -> origami.problem_t:
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size = size
        problem.batch = 1
        problem.a_transpose = origami.transpose_t.T
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype = a_origami_dtype
        problem.b_dtype = b_origami_dtype
        problem.c_dtype = c_origami_dtype
        problem.d_dtype = c_origami_dtype
        problem.mi_dtype = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size

        return problem

    def _infer_matrix_instruction_dimensions(self):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.  The input dtype sizes are retrieved
        from local object variables.

        Returns:
            origami.dim3_t: An Origami dimension trio containing the matrixinstruction
                dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
                sizes are not compatible with the detected hardware.
        """
        largest_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)

        mi_dim = None
        # gfx950
        if self._hardware.N_CU == 256:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6F8
            if largest_bitsize <= 8:
                if self._k % 256 == 0:
                    self._block_k_range = self._block_k_range + [256]
                else:
                    self._block_k_range = self._block_k_range + [128]
                self._block_mn_range = [32, 64, 128, 256]
                mi_dim = origami.dim3_t(16, 16, 128)
        # gfx942 (304 CUs full, 80 CUs partitioned, 64 CUs)
        is_gfx942 = self._hardware.N_CU in [304, 80, 64]
        if is_gfx942:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on gfx942
            if largest_bitsize < 8:
                raise ValueError("gfx942 doesn't support F4/F6")
        if self._hardware.N_CU == 228:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on MI300A
            if largest_bitsize < 8:
                raise ValueError("MI300A doesn't support F4/F6")
        # gfx90a
        if self._hardware.N_CU == 104:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                raise ValueError("MI200 doesn't support F8")
            if largest_bitsize < 8:
                raise ValueError("MI200 doesn't support F4/F6")
        # Architecture Detected is not valid
        if mi_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction integrated for {element_size_A}-bit or {element_size_B}-bit datatypes"
            )

        return mi_dim

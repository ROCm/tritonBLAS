import itertools
import torch
import origami
import math
from math import ceil


class OrigamiMatmulSelector:
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
        num_stages=2,
    ):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k
        self.streamk = streamk
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

        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Run Origami solution selection
        self._result = origami.select_config(
            self._problem, self._hardware, self._configs
        )

        self._num_stages = num_stages
        self._validate_lds_usage()

        hints_a, hints_b = self._detect_cache_hints()
        self._result.config.cache_hints_a = hints_a
        self._result.config.cache_hints_b = hints_b

        if streamk:
            self._grid = self._compute_sk_grid()
        else:
            self._grid = self._hardware.N_CU

        # Try both workgroup mapping modes for compatibility with Origami Versions
        try:
            _mapping_mode, self._xcc_workgroup_mapping, self._workgroup_mapping = (
                origami.select_workgroup_mapping(
                    self._problem, self._hardware, self._result.config, self._grid
                )
            )
        except ValueError:
            self._xcc_workgroup_mapping, self._workgroup_mapping = (
                origami.select_workgroup_mapping(
                    self._problem, self._hardware, self._result.config, self._grid
                )
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

    @property
    def group_m(self):
        return self._workgroup_mapping

    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping

    @property
    def waves_per_eu(self):
        return self._result.config.occupancy

    @property
    def even_k(self):
        return math.gcd(self._k, self.block_k) == self.block_k

    @property
    def cache_modifier_a(self):
        """Triton cache_modifier string for operand A loads, or None for default."""
        return self._cache_hints_to_modifier(self._result.config.cache_hints_a)

    @property
    def cache_modifier_b(self):
        """Triton cache_modifier string for operand B loads, or None for default."""
        return self._cache_hints_to_modifier(self._result.config.cache_hints_b)

    @property
    def sk_grid(self):
        return self._grid

    @property
    def num_stages(self):
        return self._num_stages

    @staticmethod
    def _cache_hints_to_modifier(cache_hints):
        """Map Origami cache_hints to Triton cache_modifier. See docs/ORIGAMI_NONTEMPORAL_CACHE_HINTS.md."""
        if cache_hints == 0:
            return None
        elif cache_hints == 1:
            return ".cg"
        else:
            return ".cs"

    @staticmethod
    def estimate_triton_lds(block_m, block_n, block_k, elem_bytes_a, elem_bytes_b, num_stages):
        """Estimate Triton's shared memory (LDS) usage for a GEMM tile.

        See docs/TRITON_LDS_ESTIMATION.md for detailed explanation.
        """
        PADDING_INTERVAL = 512
        PADDING_ELEMENTS = 16

        def padded_buffer_size(rows, cols, elem_bytes):
            raw_elements = rows * cols
            raw_bytes = raw_elements * elem_bytes
            num_padding_rows = raw_elements // PADDING_INTERVAL
            padding_bytes = num_padding_rows * PADDING_ELEMENTS * elem_bytes
            return raw_bytes + padding_bytes

        padded_a = padded_buffer_size(block_m, block_k, elem_bytes_a)
        padded_b = padded_buffer_size(block_k, block_n, elem_bytes_b)

        total_per_stage = padded_a + padded_b
        estimated = num_stages * total_per_stage
        estimated = ((estimated + 255) // 256) * 256

        return estimated

    def _validate_lds_usage(self):
        """Validate and re-select tile if it exceeds Triton's LDS budget."""
        lds_capacity = self._hardware.lds_capacity
        elem_bits_a = origami.datatype_to_bits(self._problem.a_dtype)
        elem_bits_b = origami.datatype_to_bits(self._problem.b_dtype)
        elem_bytes_a = (elem_bits_a + 7) // 8
        elem_bytes_b = (elem_bits_b + 7) // 8

        mt = self._result.config.mt
        estimated_lds = self.estimate_triton_lds(
            mt.m, mt.n, mt.k, elem_bytes_a, elem_bytes_b, self._num_stages
        )

        if estimated_lds <= lds_capacity:
            return  # Current tile is fine

        # Current tile exceeds LDS budget — filter configs and re-select
        constrained_configs = []
        for cfg in self._configs:
            est = self.estimate_triton_lds(
                cfg.mt.m, cfg.mt.n, cfg.mt.k,
                elem_bytes_a, elem_bytes_b, self._num_stages
            )
            if est <= lds_capacity:
                constrained_configs.append(cfg)

        if not constrained_configs:
            # No valid config found; keep the original and let Triton handle the error
            return

        # Re-select with constrained configs
        self._result = origami.select_config(
            self._problem, self._hardware, constrained_configs
        )

    def _detect_cache_hints(self):
        """Detect NonTemporal cache hints for A and B based on problem shape
        and the selected tile sizes.

        This mirrors the short-circuit logic in Origami's compute_total_latency
        (gemm.cpp) which decides when nontemporal loads are beneficial:
          - NTB when M is small relative to tile, B not transposed, and N >> M
          - NTA when N is small relative to tile, A transposed, and M >> N
          - Both require K and MT_K to be 128-byte aligned

        Returns:
            tuple[int, int]: (cache_hints_a, cache_hints_b) where 0 = default,
                4 = nontemporal.
        """
        M, N, K = self._m, self._n, self._k
        MT_M = self._result.config.mt.m
        MT_N = self._result.config.mt.n
        MT_K = self._result.config.mt.k

        a_bits = origami.datatype_to_bits(self._problem.a_dtype)
        b_bits = origami.datatype_to_bits(self._problem.b_dtype)

        a_trans = self._problem.a_transpose == origami.transpose_t.T
        b_trans = self._problem.b_transpose == origami.transpose_t.T

        cache_hints_a = 0
        cache_hints_b = 0

        # K and MT_K must be 128-byte aligned (1024 bits) for nontemporal loads
        K_mod_128bytes = (K * a_bits) % 1024
        MT_K_mod_128bytes = (MT_K * a_bits) % 1024

        if K_mod_128bytes == 0 and MT_K_mod_128bytes == 0:
            if (
                M <= MT_M * 2
                and not b_trans
                and M * a_bits > 0
                and (N * b_bits) // (M * a_bits) > 5
            ):
                # Skinny-M: B data is accessed with poor reuse → nontemporal B
                cache_hints_b = 4
            elif (
                N <= MT_N * 2
                and a_trans
                and N * b_bits > 0
                and (M * a_bits) // (N * b_bits) > 5
            ):
                # Skinny-N: A data is accessed with poor reuse → nontemporal A
                cache_hints_a = 4

        return cache_hints_a, cache_hints_b

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

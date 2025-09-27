"""!
@file origami.py
@brief Matrix multiplication heuristic optimization module using Origami hardware abstraction.

This module provides the MatmulHeuristicResult class which implements intelligent
tile selection and grid computation for optimal matrix multiplication performance
on AMD GPU hardware. It leverages the Origami library for hardware-aware optimization.

@author TritonBLAS Development Team
@date 2024
"""

import torch
import itertools
from math import ceil
import origami

#! Mapping from PyTorch data types to their string representations.
#! Used for interfacing with the Origami library's datatype system.
#! @see https://docs.pytorch.org/docs/stable/tensors.html
dtype_to_str = {
    torch.float32: "f32",      #!< 32-bit floating point
    torch.complex64: "c32",    #!< 32-bit complex (single precision)
    torch.complex128: "c64",   #!< 64-bit complex (double precision)
    torch.float64: "f64",      #!< 64-bit floating point (double precision)
    torch.float16: "f16",      #!< 16-bit floating point (half precision)
    torch.int32: "i32",        #!< 32-bit signed integer
    torch.bfloat16: "bf16",    #!< 16-bit brain floating point
    torch.int8: "i8",          #!< 8-bit signed integer
    torch.float8_e5m2: "f8",   #!< 8-bit floating point (5-bit exponent, 2-bit mantissa)
    torch.float8_e4m3fn: "f8", #!< 8-bit floating point (4-bit exponent, 3-bit mantissa, finite/NaN)
}


class MatmulHeuristicResult:
    """!
    @brief Heuristic-based matrix multiplication configuration optimizer.
    
    This class analyzes matrix multiplication parameters and hardware characteristics
    to determine optimal tile sizes, grid configurations, and execution strategies
    for maximum performance on AMD GPU architectures.
    
    The heuristic considers:
    - Matrix dimensions (M, N, K)
    - Data types and their bit widths
    - Hardware specifications (CU count, matrix instruction support)
    - Memory hierarchy optimization
    - Stream-K vs persistent execution modes
    
    @details
    The optimization process involves:
    1. Hardware detection and matrix instruction dimension inference
    2. Valid tile size generation based on hardware constraints
    3. Performance modeling using the Origami library
    4. Grid size computation for optimal resource utilization
    5. Stream-K grid optimization for load balancing
    
    Supported AMD GPU architectures:
    - gfx950: 256 CUs, supports FP32/FP16/BF16/FP8
    - gfx942 (MI300X): 304 CUs, supports FP32/FP16/BF16/FP8
    - gfx942 (MI300A): 228 CUs, supports FP32/FP16/BF16/FP8  
    - gfx908 (MI200): 104 CUs, supports FP32/FP16/BF16
    
    @note This class is typically instantiated automatically by the matmul functions
    and cached for performance via LRU caching mechanisms.
    """
    def __init__(
        self,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        c_dtype,
        MI_dim=None,
        mx_block_size=0,  # Number of MX datatype elements that share a scale
        streamk=True,
    ):
        """!
        @brief Initialize the matrix multiplication heuristic optimizer.
        
        Analyzes the given matrix multiplication problem and hardware to determine
        optimal execution parameters including tile sizes, grid configuration,
        and execution strategy.
        
        @param m (int): Number of rows in matrix A and output matrix C
        @param n (int): Number of columns in matrix B and output matrix C  
        @param k (int): Number of columns in matrix A and rows in matrix B (inner dimension)
        @param a_dtype (torch.dtype): Data type of input matrix A
        @param b_dtype (torch.dtype): Data type of input matrix B
        @param c_dtype (torch.dtype): Data type of output matrix C
        @param MI_dim (list[int], optional): Matrix instruction dimensions [M, N, K].
               If None, dimensions are inferred from hardware and data types.
        @param mx_block_size (int, optional): Number of MX datatype elements sharing a scale factor.
               Used for mixed-precision optimizations. Default is 0 (disabled).
        @param streamk (bool, optional): Enable Stream-K execution mode for better load balancing.
               Default is True. When False, uses persistent execution mode.
        
        @details
        Initialization process:
        1. Store matrix dimensions and data types
        2. Query hardware information via Origami
        3. Compute element sizes and infer matrix instruction dimensions
        4. Generate optimal tile configuration using performance modeling
        5. Compute grid size based on execution mode (Stream-K or persistent)
        
        @note The constructor performs significant computation including hardware
        queries and performance modeling. Results should be cached when possible.
        
        @throws ValueError: If unsupported hardware architecture is detected or
                           if data types are incompatible with detected hardware.
        
        @see _infer_matrix_instruction_dimensions() for hardware compatibility details
        @see compute_sk_grid() for Stream-K grid optimization algorithm
        """

        # Set Instance Variables
        self.m = m
        self.n = n
        self.k = k

        # Instantiate hardare information object
        self.hardware = origami.get_hardware_for_device(0)
        self.block_mn_range = [16, 32, 64, 128, 256]
        self.block_k_range = [16, 32, 64]

        self.element_size_A = torch.finfo(a_dtype).bits
        self.element_size_B = torch.finfo(b_dtype).bits
        self.element_size_out = torch.finfo(c_dtype).bits
        self.mi_dtype = dtype_to_str.get(c_dtype)

        # Infer Matrix Instruction Dimensions from datatypes
        self.MI_dim = self._infer_matrix_instruction_dimensions(
            self.element_size_A, self.element_size_B
        )

        self.kernel_occupancy = [1]  # Number of WG possibly co-resident in a CU
        self.mx_block_size = mx_block_size

        self.config = self._prepare_config()

        # Grid model constants
        self.split_factors = [8, 6, 4, 3, 2, 1]

        self.tile_fractions = [
            0.0,
            1.0 / 2.0,
            1.0 / 8.0,
            1.0 / 5.0,
            1.0 / 4.0,
            1.0 / 3.0,
        ]
        self.max_workspace = 128 * 1024 * 1024

        if streamk:
            self.grid = self.compute_sk_grid()
        else:
            self.grid = self.hardware.N_CU

    def _infer_matrix_instruction_dimensions(self, element_size_A, element_size_B):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.

        Parameters:
            element_size_A (int): The size (in bits) of the elements in matrix A.
            element_size_B (int): The size (in bits) of the elements in matrix B.

        Returns:
            list[int]: A list representing the matrix instruction dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
            sizes are not compatible with the detected hardware.
        """
        MI_dim = None
        # gfx950
        if self.hardware.N_CU == 256:
            # FP32
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            # FP16/BF16
            if max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 32]
            # F4F6F8
            if max(element_size_A, element_size_B) <= 8:
                MI_dim = [16, 16, 128]
        # gfx942
        if self.hardware.N_CU == 304:
            # FP32
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            # FP16/BF16
            if max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 16]
            # F8
            if max(element_size_A, element_size_B) == 8:
                MI_dim = [16, 16, 32]
                self.block_mn_range = self.block_mn_range + [512]
                self.block_k_range = self.block_k_range + [128, 256]

            # F4F6 -> Unsupported on MI300X
            if max(element_size_A, element_size_B) < 8:
                raise ValueError("MI300X doesn't support F4/F6")

        if self.hardware.N_CU == 228:
            # FP32
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            # FP16/BF16
            if max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 16]
            # F8
            if max(element_size_A, element_size_B) == 8:
                MI_dim = [16, 16, 32]
                self.block_mn_range = self.block_mn_range + [512]
                self.block_k_range = self.block_k_range + [128, 256]

            # F4F6 -> Unsupported on MI300A
            if max(element_size_A, element_size_B) < 8:
                raise ValueError("MI300A doesn't support F4/F6")
            
        if self.hardware.N_CU == 104:
            # FP32
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            # FP16/BF16
            if max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 16]
            if max(element_size_A, element_size_B) == 8:
                raise ValueError("MI200 doesn't support F8")

            if max(element_size_A, element_size_B) < 8:
                raise ValueError("MI200 doesn't support F4/F6")
            
        # Architecture Detected is not valid
        if MI_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction integrated for {element_size_A}-bit or {element_size_B}-bit datatypes"
            )
        return MI_dim

    def _get_valid_tiles(self):
        """!
        @brief Generate all valid tile size combinations for the current hardware configuration.
        
        Creates a Cartesian product of possible tile dimensions (M, N, K) combined with
        matrix instruction dimensions and kernel occupancy options. This provides the
        search space for tile optimization.
        
        @return list[tuple]: List of tuples containing:
                - Block size M (from block_mn_range)
                - Block size N (from block_mn_range) 
                - Block size K (from block_k_range)
                - Matrix instruction M dimension
                - Matrix instruction N dimension
                - Matrix instruction K dimension
                - Kernel occupancy (workgroups per CU)
        
        @details
        The tile dimensions are constrained by:
        - Hardware matrix instruction capabilities (MI_dim)
        - Memory hierarchy efficiency (block_mn_range, block_k_range)
        - Compute unit occupancy requirements (kernel_occupancy)
        
        @note This method generates the full search space for tile optimization.
        The actual selection is performed by _get_best_tile_size().
        
        @see _get_best_tile_size() for tile selection algorithm
        @see _infer_matrix_instruction_dimensions() for MI_dim computation
        """
        return list(
            itertools.product(
                self.block_mn_range,
                self.block_mn_range,
                self.block_k_range,
                [self.MI_dim[0]],  # MI_M
                [self.MI_dim[1]],  # MI_N
                [self.MI_dim[2]],  # MI_K
                self.kernel_occupancy,
            )
        )

    def _get_gsize_m(self, BLK_M, BLK_N, BLK_K):
        """!
        @brief Determine optimal workgroup mapping (GROUP_SIZE_M) for given tile dimensions.
        
        Uses the Origami library to select the best workgroup mapping strategy that
        minimizes memory bank conflicts and maximizes compute unit utilization
        for the specified tile configuration.
        
        @param BLK_M (int): Tile size in the M dimension (rows of A, rows of C)
        @param BLK_N (int): Tile size in the N dimension (cols of B, cols of C)  
        @param BLK_K (int): Tile size in the K dimension (cols of A, rows of B)
        
        @return int: Optimal GROUP_SIZE_M value for workgroup scheduling
        
        @details
        The workgroup mapping affects:
        - Memory access patterns and bank conflicts
        - Load balancing across compute units
        - Cache locality and reuse
        - Overall throughput and efficiency
        
        The method evaluates multiple GROUP_SIZE_M candidates [1, 2, 4, 6, 8]
        using hardware-aware performance modeling with L2 cache hit rate assumptions.
        
        @note This uses Origami's select_best_wgm function which performs detailed
        hardware modeling including memory hierarchy analysis.
        """
        results = origami.select_best_wgm(
            self.m,  # M
            self.n,  # N
            self.k,  # K
            1,  # batch
            self.hardware,  # Hardware
            BLK_M,  # MT_M
            BLK_N,  # MT_N
            BLK_K,  # MT_K
            self.MI_dim[0],  # MI_M
            self.MI_dim[1],  # MI_N
            self.MI_dim[2],  # MI_K
            [1, 2, 4, 6, 8],  # WGM List
            self.element_size_A,  # element size
            0.8,  # H_L2
            False,  # debug
            False,  # Print
        )
        return results[1]

    def _get_best_tile_size(self):
        """!
        @brief Select optimal tile dimensions using hardware-aware performance modeling.
        
        Evaluates all valid tile configurations using the Origami library's performance
        modeling to determine the tile sizes that maximize throughput for the given
        matrix multiplication problem and hardware configuration.
        
        @return tuple[int, int, int]: Optimal tile dimensions (BLK_M, BLK_N, BLK_K)
        
        @details
        The selection process:
        1. Generate all valid tile combinations via _get_valid_tiles()
        2. Use Origami's select_best_macro_tile_size() for performance modeling
        3. Apply hardware-specific heuristics for fine-tuning
        4. Return the tile configuration with highest predicted performance
        
        Performance modeling considers:
        - Memory bandwidth utilization
        - Compute unit occupancy and efficiency  
        - Cache hit rates and memory hierarchy
        - Matrix instruction utilization
        - Load balancing characteristics
        
        Hardware-specific adjustments:
        - MI300X (304 CUs): Applies heuristics for 256x256 tiles to balance
          performance vs occupancy based on empirical observations
        
        @note The modeling assumes:
        - Transposed A matrix (transA=True)
        - Non-transposed B matrix (transB=False)  
        - L2 cache hit rate of 80% (0.8)
        - Default workgroup mapping of 6
        
        @see _get_valid_tiles() for tile generation
        @see origami.select_best_macro_tile_size() for performance modeling details
        """
        valid_tiles = self._get_valid_tiles()
        results = origami.select_best_macro_tile_size(
            self.m,  # M
            self.n,  # N
            self.k,  # K
            1,  # Batch
            True,  # transA
            False,  # transB
            self.hardware,  # Hardware
            valid_tiles,  # Tile List
            self.element_size_A,  # Element Size A
            self.element_size_B,  # Element Size B
            self.element_size_out,  # Element Size Out
            origami.string_to_datatype(self.mi_dtype),  # MI Data Type
            self.mx_block_size,  # MX Block Size
            0.8,  # H_L2
            False,  # debug
            False,  # Print
            6,  # WGM
        )

        best_result = results[0]

        # Heuristic weighting for specific hardware configurations
        if self.hardware.N_CU == 304:  # MI300X
            # For 256x256 tiles, consider alternative if performance is close
            if best_result[1] == 256 and best_result[2] == 256:
                if results[0][0] * 1.00 > results[1][0]:
                    best_result = results[1]

        return (best_result[1], best_result[2], best_result[3])

    def _prepare_config(self):
        """!
        @brief Prepare complete configuration tuple with optimal tile sizes and workgroup mapping.
        
        Combines the results of tile size optimization and workgroup mapping selection
        into a complete configuration tuple that can be used by the matrix multiplication
        kernels.
        
        @return tuple[int, int, int, int]: Configuration tuple containing:
                - BLK_M: Optimal tile size in M dimension
                - BLK_N: Optimal tile size in N dimension  
                - BLK_K: Optimal tile size in K dimension
                - gsize_m: Optimal workgroup mapping (GROUP_SIZE_M)
        
        @details
        This method orchestrates the full optimization process:
        1. Determines optimal tile dimensions via _get_best_tile_size()
        2. Computes optimal workgroup mapping via _get_gsize_m()
        3. Returns complete configuration for kernel execution
        
        @see _get_best_tile_size() for tile optimization
        @see _get_gsize_m() for workgroup mapping optimization
        """
        BLK_M, BLK_N, BLK_K = self._get_best_tile_size()
        gsize_m = self._get_gsize_m(BLK_M, BLK_N, BLK_K)
        return BLK_M, BLK_N, BLK_K, gsize_m

    def get_config(self):
        """!
        @brief Get the optimal configuration tuple for matrix multiplication execution.
        
        @return tuple[int, int, int, int]: Configuration tuple (BLK_M, BLK_N, BLK_K, GROUP_SIZE_M)
        
        @details
        Returns the pre-computed optimal configuration that includes:
        - BLK_M: Tile size in M dimension (rows)
        - BLK_N: Tile size in N dimension (columns)
        - BLK_K: Tile size in K dimension (inner dimension)
        - GROUP_SIZE_M: Workgroup mapping parameter
        
        This configuration is used by the Triton kernels for optimal performance.
        """
        return self.config

    def get_grid(self):
        """!
        @brief Get the optimal grid size for kernel execution.
        
        @return int: Grid size (number of workgroups) for optimal load balancing
        
        @details
        Returns the pre-computed grid size that depends on the execution mode:
        - Stream-K mode: Uses compute_sk_grid() for dynamic load balancing
        - Persistent mode: Uses hardware CU count for static scheduling
        
        The grid size determines how work is distributed across compute units
        and affects load balancing, memory access patterns, and overall performance.
        
        @see compute_sk_grid() for Stream-K grid computation algorithm
        """
        return self.grid

    def partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N, _, GSIZE = self.get_config()

        # bytes per C element
        bytes_per_elem = self.element_size_out // 8

        # size of one partial tile per WG
        tile_size = BLK_M * BLK_N * bytes_per_elem

        # scale by the number of partial‑tiles per WG
        return tile_size * sk_grid

    def compute_sk_grid(self):
        """
        Implements the dynamic‐grid mode logic
        """
        config = self.config
        cu_count = self.hardware.N_CU
        BLK_M = config[0]
        BLK_N = config[1]
        BLK_K = config[2]
        # Fallback if no better fractional split is found
        tiles = ceil(self.m / BLK_M) * ceil(self.n / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(self.k / BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            # if size_mapping.CUOccupancy > 1:
            # virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            tile_fractions = self.tile_fractions
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (
                    tiles % frac_grid != 0
                    and self.partial_tile_size(frac_grid) > self.max_workspace
                ):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            split_factors = self.split_factors
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

        if tiles >= self.hardware.N_CU:
            last_wave_remainder = tiles % self.hardware.N_CU
            last_wave_occupancy = last_wave_remainder / self.hardware.N_CU

            # Really bad last wave, which would have originally been compensated for
            # by changing tile size, but triton tile sizes are limited
            if last_wave_remainder < 128 and last_wave_remainder > 0:
                sk_grid = 256

        return sk_grid

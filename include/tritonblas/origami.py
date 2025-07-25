import itertools
import origami

from math import ceil

class MatmulHeuristicResult:
    def __init__(
        self,
        m,
        n,
        k,
        element_size_A=16,
        element_size_B=16,
        element_size_out=32,
        MI_dim=None,
        mx_block_size=0,  # Number of MX datatype elements that share a scale
        streamk=True
    ):

        # Instantiate hardare information object
        self.hardware = origami.getHardwareForDevice(0)
        self.block_mn_range = [16, 32, 64, 128, 256]
        self.block_k_range = [16, 32, 64]
        # Infer Matrix Instruction Dimensions from datatypes
        self.MI_dim = self._infer_matrix_instruction_dimensions(
            element_size_A, element_size_B
        )
        # Set Instance Variables
        self.m = m
        self.n = n
        self.k = k

        self.element_size_A = element_size_A
        self.element_size_B = element_size_B
        self.element_size_out = element_size_out
        self.kernel_occupancy = [1]  # Number of WG possibly co-resident in a CU
        self.mx_block_size = mx_block_size

        self.config = self._prepare_config()
        if streamk:
            self.grid = self.compute_sk_grid()
        else:
            self.grid=self.hardware.N_CU

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
        # Architecture Detected is not valid
        if MI_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction integrated for {element_size_A}-bit or {element_size_B}-bit datatypes"
            )
        return MI_dim

    def _get_valid_tiles(self):
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
            self.mx_block_size,  # MX Block Size
            0.8,  # H_L2
            False,  # debug
            False,  # Print
            6,  # WGM
        )

        best_result = results[0]
        return (best_result[1], best_result[2], best_result[3])

    def _prepare_config(self):
        BLK_M, BLK_N, BLK_K = self._get_best_tile_size()
        gsize_m = self._get_gsize_m(BLK_M, BLK_N, BLK_K)
        return BLK_M, BLK_N, BLK_K, gsize_m

    def get_config(self):
        return self.config

    def get_grid(self):
        return self.grid

    def partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N, _,GSIZE = self.get_config()

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
        config=self.config
        cu_count = self.hardware.N_CU
        BLK_M = config[0]
        BLK_N = config[1]
        BLK_K = config[2]
        # Fallback if no better fractional split is found
        tiles = ceil(self.m/BLK_M) * ceil(self.n/BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(self.k/BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            #if size_mapping.CUOccupancy > 1:
                #virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            tile_fractions = [0.0, 1.0/2.0, 1.0/8.0, 1.0/5.0, 1.0/4.0, 1.0/3.0]
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (tiles % frac_grid != 0 and
                    self.partial_tile_size(frac_grid) > 128 * 1024 * 1024):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            split_factors = [8, 6, 4, 3, 2, 1]
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if (tiles % sk_grid != 0):
            sk_grid = tiles

        return sk_grid
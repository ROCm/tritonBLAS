import torch
import itertools

import origami

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


class MatmulHeuristicResult:
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
    ):

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
        # gfx1100
        if self.hardware.N_CU == 96:  # TODO: fill actual CU count
            # If gfx1100 matches gfx950 MFMA shapes:
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            if max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 32]
            if max(element_size_A, element_size_B) <= 8:
                MI_dim = [16, 16, 128]
            # If gfx1100 adds larger K for 8-bit (example):
            # if max(element_size_A, element_size_B) <= 8:
            #     MI_dim = [16, 16, 256]  # uncomment if new instruction exists
            # Optional: expand search space if hardware benefits
            # self.block_mn_range += [512]
            # self.block_k_range += [128, 256]
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
                f"No Valid Matrix Instruction integrated for {element_size_A}-bit or {element_size_B}-bit datatypes with {self.hardware.N_CU} CUs."
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
            origami.string_to_datatype(self.mi_dtype), # MI Data Type
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

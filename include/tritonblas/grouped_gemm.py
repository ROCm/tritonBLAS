import torch
import triton
import random
import functools
import time
import math
from .internal.wcc_grouped_gemm import wcc_groupgemm
from .origami import MatmulHeuristicResult

_tensor_cache = {}
current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
#TODO: 256x256 for fp16/bf16, need adjust for fp8/fp4
MAX_BLOCK_SIZE = 65536



def grouped_gemm( 
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    group_c: list[torch.Tensor],
    BLK_M: int,
    BLK_N: int,
    BLK_K: int,
    ):

    group_size = len(group_a)
    a_addrs, b_addrs, c_addrs = [], [], []
    g_sizes, g_lds = [], []

    for i in range(group_size):
        A, B, C = group_a[i], group_b[i], group_c[i]
        assert A.shape[1] == B.shape[0], "Incompatible Dimensions"
        m, k = A.shape
        _, n = B.shape
        a_addrs.append(A.data_ptr())
        b_addrs.append(B.data_ptr())
        c_addrs.append(C.data_ptr())
        g_sizes.extend([m, n, k])
        g_lds.extend([A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1)])

    d_a_ptrs = torch.tensor(a_addrs, device="cuda", dtype=torch.int64)
    d_b_ptrs = torch.tensor(b_addrs, device="cuda", dtype=torch.int64)
    d_c_ptrs = torch.tensor(c_addrs, device="cuda", dtype=torch.int64)
    d_g_sizes = torch.tensor(g_sizes, device="cuda", dtype=torch.int32)
    d_g_lds = torch.tensor(g_lds, device="cuda", dtype=torch.int32)

    grids = MAX_SMS
    locks = torch.zeros((MAX_SMS,), device="cuda", dtype=torch.int32)
    P = torch.zeros((MAX_SMS, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

    group_tiles_count = []
    total = 0
    for g in range(group_size):
        mm = math.ceil(g_sizes[g * 3] / BLK_M)
        nn = math.ceil(g_sizes[g * 3 + 1] / BLK_N)
        kk = math.ceil(g_sizes[g * 3 + 2] / BLK_K)
        gemm_tiles = nn * mm * kk
        total += gemm_tiles
        group_tiles_count.append(int(gemm_tiles))

    gemm_offsets = [0]
    for count in group_tiles_count:
        gemm_offsets.append(gemm_offsets[-1] + count)

    group_total_tiles = total
    streamk_tiles_pcu = group_total_tiles // MAX_SMS
    streamk_remainder_tiles = group_total_tiles % MAX_SMS
    d_gemm_offsets = torch.tensor(gemm_offsets, dtype=torch.int32, device="cuda")

    wcc_groupgemm[(grids,)](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,            
        d_gemm_offsets,
        d_g_lds,              
        group_size,
        P,
        locks,
        streamk_tiles_pcu=streamk_tiles_pcu,
        streamk_remainder_tiles=streamk_remainder_tiles,
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=1,
        NUM_PRGMS=MAX_SMS,
        NUM_XCDS=8,
    )
    return group_c
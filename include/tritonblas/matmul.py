import torch
import triton
import random
import functools
import time
from .internal.persistent_matmul import persistent_matmul
from .internal.streamk_matmul import streamk_matmul
from .origami import MatmulHeuristicResult
from typing import Dict, Tuple, Optional

_tensor_cache = {}
MAX_SMS = 304
# 256x256 may need adjust when using fp8/fp4
MAX_BLOCK_SIZE = 65536 

# Global pre-allocated buffers
_global_locks = torch.zeros(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.zeros(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)

# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(M: int, N: int, K: int, 
                          a_dtype: torch.dtype, 
                          b_dtype: torch.dtype, 
                          c_dtype: torch.dtype):
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, a_dtype, b_dtype, c_dtype)


def persistent_matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % BLK_K == 0

    # TODO: Separate these configs.
    # basica configs for most of compute bound sizes
    # TODO: set these values analytically?
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    # Run in Data-parallel mode.
    grids = total_tiles

    # TODO: Support other matmul algs.
    kk = persistent_matmul[(grids,)](
        a,
        b,
        c,
        None,  # TODO: Enable bias.
        M,
        N,
        K,
        a.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,  # TODO: Enable bias stride.
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_programs,
        NUM_XCDS=8,
        BIAS=False,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def streamk_matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    selector
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    iters_per_tile = triton.cdiv(K, BLK_K)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    if total_tiles >= selector.hardware.N_CU:
        total_programs_streamk = selector.hardware.N_CU
    else:
        total_programs_streamk = total_tiles

    if total_programs_streamk > 0:  # Stream-K
        # last wave may occupy less than total_programs_streamk SMs
        total_tiles_streamk = total_tiles % total_programs_streamk
        total_blocking_tiles = total_tiles - total_tiles_streamk
        total_iters_streamk = total_tiles_streamk * iters_per_tile
        total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
        total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
    else:  # all tiles are computed using classical blocking
        total_blocking_tiles = total_tiles
        total_tiles_streamk = 0
        total_full_tiles_streamk = 0
        total_partial_tiles_streamk = 0
        total_iters_streamk = 0

    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    # Use global buffers with optimized zeroing
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]

        # Optimized zeroing - use fill_ which can be faster than zero_()
        locks.fill_(0)
        P.fill_(0.0)
    else:
        locks = torch.zeros(grids, device="cuda", dtype=torch.uint8)
        P = torch.zeros(grids, block_size, device="cuda", dtype=torch.float32)


    kk = streamk_matmul[(grids,)](
        a,
        b,
        c,
        None, # TODO: Enable bias.
        P,
        locks,
        M,
        N,
        K,
        a.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0, # TODO: Enable bias stride.
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=grids,
        NUM_XCDS=8,
        STREAMK_TILES=total_tiles_streamk,
        BIAS=False,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c

def matmul_lt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    selector, enable_streamk = False):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector)
    else:
        return persistent_matmul_lt(a, b, c, selector)

def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, enable_streamk = False):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector)
    else:
        return persistent_matmul_lt(a, b, c, selector)

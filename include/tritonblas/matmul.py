import torch
import triton
import random
import functools
import time
from .internal.persistent_matmul import persistent_matmul
from .internal.streamk_matmul import streamk_matmul
from .origami import MatmulHeuristicResult

tritonblas_enable_streamk_matmul = True
_tensor_cache = {}

# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(M: int, N: int, K: int, bitsA: int, bitsB: int, bitsC: int,streamk:bool):
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, bitsA, bitsB, bitsC,streamk=streamk)


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

    ##
    # Grid Size
    ##
    total_programs_streamk = selector.get_grid()
    # if total_tiles >= selector.hardware.N_CU:
    #     total_programs_streamk = selector.hardware.N_CU
    # else:
    #     total_programs_streamk = total_tiles

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

    # Stream-K Specific Parameters
    global _tensor_cache
    if '_tensor_cache' not in globals():
        _tensor_cache = {}

    # Enhanced cache keys to avoid collisions
    locks_key = (grids, id(torch.cuda.current_stream()))
    P_key = (grids, BLK_M * BLK_N, id(torch.cuda.current_stream()))

    if locks_key not in _tensor_cache:
        _tensor_cache[locks_key] = torch.zeros((grids,), device="cuda", dtype=torch.uint8)
    else:
        # Only zero the active portion
        _tensor_cache[locks_key][:grids].zero_()

    if P_key not in _tensor_cache:
        _tensor_cache[P_key] = torch.zeros((grids, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
    else:
        # Only zero the active portion
        _tensor_cache[P_key][:grids, :BLK_M * BLK_N].zero_()

    locks = _tensor_cache[locks_key]
    P = _tensor_cache[P_key]

    # TODO: Support other matmul algs.
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

def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    # pull shape/precision out of the tensors
    bitsA = torch.finfo(a.dtype).bits
    bitsB = torch.finfo(b.dtype).bits
    bitsC = torch.finfo(c.dtype).bits

    
    if tritonblas_enable_streamk_matmul:
        selector = _make_matmul_selector(M, N, K, bitsA, bitsB, bitsC,streamk=True)
        return streamk_matmul_lt(a, b, c, selector)
    else:
        selector = _make_matmul_selector(M, N, K, bitsA, bitsB, bitsC)
        return persistent_matmul_lt(a, b, c, selector,streamk=False)

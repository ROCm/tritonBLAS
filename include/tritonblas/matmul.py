import torch
import triton
import random
import functools
import time
from .internal.persistent_matmul import persistent_matmul
from .internal.streamk_matmul import streamk_matmul
from .internal.fp4_matmul import fp4_matmul
from .origami import MatmulHeuristicResult
from typing import Dict, Tuple, Optional

_tensor_cache = {}
current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
# TODO: 256x256 for fp16/bf16, need adjust for fp8/fp4
MAX_BLOCK_SIZE = 65536

# Global pre-allocated buffers
_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
):
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, a_dtype, b_dtype, c_dtype)


def persistent_matmul_lt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector):
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
    #for skinny size like 4, 5120, 2880, use CACHE_MODIFIER=".cg"
    CACHE_MODIFIER_A= None
    CACHE_MODIFIER_B= None

    # Run in Data-parallel mode.
    grids = total_tiles

    # Set chunk size to same area as L2 tiles.
    num_xcds = 8
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, total_programs // num_xcds)

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
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def streamk_matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector, sk_grid: Optional[int] = None
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    ##
    # Grid Size
    ##
    total_programs_streamk = selector.get_grid()

    if total_programs_streamk > 0:  # Stream-K
        total_tiles_streamk = total_tiles % total_programs_streamk
    else:  # all tiles are computed using classical blocking
        total_tiles_streamk = 0

    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    #for skinny size like 4, 5120, 2880, use CACHE_MODIFIER=".cg"
    CACHE_MODIFIER_A= None
    CACHE_MODIFIER_B= None

    if sk_grid is not None:
        total_programs_streamk = sk_grid

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    # Use global buffers with optimized zeroing
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]
    else:
        locks = torch.empty(grids, device="cuda", dtype=torch.uint8)
        P = torch.empty(grids, block_size, device="cuda", dtype=torch.float32)

    # Set chunk size to same area as L2 tiles.
    num_xcds = 8
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, grids // num_xcds) 

    kk = streamk_matmul[(grids,)](
        a,
        b,
        c,
        None,  # TODO: Enable bias.
        P,
        locks,
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
        NUM_SMS=grids,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        STREAMK_TILES=total_tiles_streamk,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector, enable_streamk=False
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector)
    else:
        return persistent_matmul_lt(a, b, c, selector)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    sk_grid=None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, sk_grid=sk_grid)
    else:
        return persistent_matmul_lt(a, b, c, selector)


def matmul_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 512,
    group_size_m: int = 8,
):
    """
    FP4 matrix multiplication: C = A @ B
    
    Args:
        a: Input matrix A in FP4 format (M, K//2), packed 2 elements per uint8
        b: Input matrix B in FP4 format (N, K//2), packed 2 elements per uint8
        c: Output matrix C (M, N) in bfloat16 or float16
        a_scales: Scales for A in e8m0 format (M, K // 32)
        b_scales: Scales for B in e8m0 format (N, K // 32)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
        block_k: Block size for K dimension (must be multiple of 64 for FP4)
        group_size_m: Group size for M dimension tiling
    
    Returns:
        Output matrix C
    """
    # Get actual dimensions (accounting for packing)
    M = a.shape[0]
    K = a.shape[1] * 2  # Unpacked K dimension
    N = b.shape[0]  # B has shape (N, K//2)
    
    # Verify dimensions are compatible
    assert b.shape[1] * 2 == K, f"Incompatible Dimensions: A has K={K}, B has K={b.shape[1] * 2}"
    
    # Transpose B to match kernel expectations (kernel expects B as K x N)
    b = b.T
    
    # Ensure block_k is appropriate for FP4 (must be multiple of 64)
    assert block_k % 64 == 0, "BLOCK_K must be multiple of 64 for FP4"
    
    total_blocks_M = triton.cdiv(M, block_m)
    total_blocks_N = triton.cdiv(N, block_n)
    total_tiles = total_blocks_M * total_blocks_N
    
    # Set chunk size to same area as L2 tiles
    num_xcds = 8
    chunk_size = group_size_m * group_size_m
    chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    
    grid = (total_tiles,)
    
    fp4_matmul[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        num_stages=2,
        num_warps=8,
    )
    
    return c

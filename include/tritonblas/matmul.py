import torch
import triton
import random
import functools
import time
from .kernels import persistent_matmul, streamk_matmul
from .kernels.fp4_matmul import fp4_matmul
from .kernels.fused_gemm import fused_persistent_matmul
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
    mx_block_size = 0
):
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, a_dtype, b_dtype, c_dtype, mx_block_size=mx_block_size)


def persistent_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
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
    #for skinny size like 4, 5120, 2880, use CACHE_MODIFIER=".cg"
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

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
        a_scale if quantized else None,  # A_scale_ptr
        b_scale if quantized else None,  # B_scale_ptr
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
        QUANTIZED=quantized,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c

def streamk_matmul_lt(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    selector, 
    sk_grid: Optional[int] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
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
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

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
        a_scale if quantized else None,  # A_scale_ptr
        b_scale if quantized else None,  # B_scale_ptr
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
        QUANTIZED=quantized,
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

def matmul_a8w8_lt(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, c: torch.Tensor, selector, enable_streamk=False
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, a_scale=a_scale, b_scale=b_scale, quantized=True)
    else:
        return persistent_matmul_lt(a, b, c, selector, a_scale=a_scale, b_scale=b_scale, quantized=True)

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
    

def fused_matmul(
    a: torch.Tensor,
    b0: torch.Tensor,
    c0: torch.Tensor,
    b1: torch.Tensor,
    c1: torch.Tensor,
    kernel_alpha_selector,
    kernel_beta_selector
):
    """
    c0 = a @ b0
    c1 = c0 @ b1
    """
    assert a.shape[1] == b0.shape[0], "Incompatible Dimensions"
    assert a.shape[0] == c1.shape[0], "Incompatible Dimensions"
    assert c0.shape[1] == b1.shape[0], "Incompatible Dimensions"
    assert c0.shape[0] == c1.shape[0], "Incompatible Dimensions"
    assert c0.shape[1] == b1.shape[0], "Incompatible Dimensions"

    # Extract dimensions
    # ALPHA: C0 = A @ B0
    # A is (M, K), B0 is (K, N), C0 is (M, N)
    M = a.shape[0]
    K = a.shape[1]  # Also b0.shape[0]
    N = b0.shape[1]  # Also c0.shape[1]
    # BETA: C1 = C0 @ B1
    # C0 is (M, N), B1 is (N, P), C1 is (M, P)
    P = b1.shape[1]  # Also c1.shape[1]
    
    ALPHA_BLK_M, ALPHA_BLK_N, ALPHA_BLK_K, ALPHA_GSIZE_M = kernel_alpha_selector.get_config()
    alpha_total_blocks_m = triton.cdiv(M, ALPHA_BLK_M)
    alpha_total_blocks_n = triton.cdiv(N, ALPHA_BLK_N)
    alpha_total_tiles = alpha_total_blocks_m * alpha_total_blocks_n
    alpha_total_programs = alpha_total_tiles
    alpha_even_k = K % ALPHA_BLK_K == 0
    
    BETA_BLK_M, BETA_BLK_N, BETA_BLK_K, BETA_GSIZE_M = kernel_beta_selector.get_config()
    # Force BLK_M and BLK_N to be the same as ALPHA_BLK_M and ALPHA_BLK_N
    BETA_BLK_M = ALPHA_BLK_M
    BETA_BLK_N = ALPHA_BLK_N
    BETA_BLK_K = ALPHA_BLK_K
    beta_total_blocks_m = triton.cdiv(M, BETA_BLK_M)
    beta_total_blocks_n = triton.cdiv(P, BETA_BLK_N)
    beta_total_tiles = beta_total_blocks_m * beta_total_blocks_n
    beta_total_programs = beta_total_tiles
    beta_even_k = N % BETA_BLK_K == 0  # For BETA, K dimension is N (from C0)
    
    # EVEN_K is used by both ALPHA (for K dimension) and BETA (for N dimension)
    # Since the kernel only accepts one EVEN_K, we use the more restrictive value
    # Both branches will handle remainders correctly if either dimension is not divisible
    even_k = alpha_even_k and beta_even_k
    
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None
    grids = alpha_total_programs + beta_total_programs
    
    # Set chunk size to same area as L2 tiles
    num_xcds = 8
    chunk_size = ALPHA_GSIZE_M * ALPHA_GSIZE_M
    chunk_size = min(chunk_size, grids // num_xcds)
    
    # Initialize locks for synchronization
    locks = torch.zeros(alpha_total_tiles, device="cuda", dtype=torch.int32)
    
    # Invoke the fused kernel
    fused_persistent_matmul[(grids,)](
        a,
        b0,
        c0,
        b1,
        c1,
        locks,
        None,  # A_scale_ptr (TODO: support quantization)
        None,  # B_scale_ptr (TODO: support quantization)
        None,  # bias_ptr (TODO: enable bias)
        M,
        N,
        K,
        P,
        a.stride(0),  # stride_am
        b0.stride(1),  # stride_b0n
        c0.stride(0),  # stride_c0m
        c0.stride(1),  # stride_c0n
        b1.stride(1),  # stride_b1n
        c1.stride(0),  # stride_c1m
        c1.stride(1),  # stride_c1n
        0,  # stride_bias (TODO: enable bias)
        stride_ak=a.stride(1),
        stride_b0k=b0.stride(0),
        stride_b1k=b1.stride(0),
        BLOCK_SIZE_M=ALPHA_BLK_M,
        BLOCK_SIZE_N=ALPHA_BLK_N,
        BLOCK_SIZE_K=ALPHA_BLK_K,
        GROUP_SIZE_M=ALPHA_GSIZE_M,
        NUM_SMS=grids,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        QUANTIZED=False,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )
    
    return c0, c1





def matmul_a8w8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    sk_grid=None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, sk_grid=sk_grid, a_scale=a_scale, b_scale=b_scale, quantized=True)
    else:
        return persistent_matmul_lt(a, b, c, selector, a_scale=a_scale, b_scale=b_scale, quantized=True)

def matmul_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_m: int = None, #Overrides Origami value
    block_n: int = None, #Overrides Origami value
    block_k: int = None, #Overrides Origami value
    group_size_m: int = 8, #Overrides Origami value
    num_warps: int = 8,
    num_stages: int = 2,
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
        num_warps: Number of warps per thread block (default: 8)
        num_stages: Number of pipeline stages (default: 2)
    
    Returns:
        Output matrix C
    """


    M, K = a.shape
    _, N = b.shape
    
    if(block_m == None):
        selector = _make_matmul_selector(M, N, K, "f4", "f4", c.dtype,mx_block_size=32)
        block_m, block_n, block_k, gsize_m = selector.get_config()
        #print(f"Selected {block_m}x{block_n}x{block_k}")
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
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
    return c

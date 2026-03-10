# ruff: noqa: E402
"""
Gluon GEMM kernels for gfx1250.

Provides gluon_streamk_matmul() as the frontend for the gfx1250 StreamK kernel.
"""
import hip
hip.hip.hipInit(0)

import math
import torch
import triton

from .f16_gemm_streamk_gfx1250 import (
    streamk_gemm_tdm_pipelined_kernel_4warps,
    streamk_gemm_tdm_pipelined_kernel_8warps,
    streamk_gemm_tdm_prefetch_kernel,
)


def _compute_warp_bases(num_warps):
    """Compute WARP_BASES tuple from num_warps."""
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    return tuple(warp_bases)


def gluon_streamk_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 64,
    num_buffers: int = 2,
    num_warps: int = 4,
    num_sms: int = 8,
    transpose_b: bool = True,
    use_prefetch: bool = False,
    group_size_m: int = 8,
) -> torch.Tensor:
    """
    Launch the gluon StreamK GEMM kernel for gfx1250.

    Args:
        a: Input matrix A (M, K), fp16 or bf16, on GPU.
        b: Input matrix B. If transpose_b=True, shape is (N, K) (row-major transposed).
           If transpose_b=False, shape is (K, N).
        c: Output matrix C (M, N), float32, on GPU. Written in-place.
        block_m/block_n/block_k: Tile sizes.
        num_buffers: Number of shared memory pipeline buffers (2 for 4-warp, 3 for 8-warp).
        num_warps: 4 or 8.
        num_sms: Number of SMs to use for persistent grid.
        transpose_b: Whether B is stored transposed (N, K).
        use_prefetch: Use prefetch kernel variant (4-warp only).
        group_size_m: Tile swizzle group size.

    Returns:
        c tensor (modified in-place).
    """
    M, K = a.shape
    if transpose_b:
        N = b.shape[0]
        stride_bk, stride_bn = b.stride(1), b.stride(0)
    else:
        N = b.shape[1]
        stride_bk, stride_bn = b.stride(0), b.stride(1)

    total_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    streamk_tiles = total_tiles % num_sms

    grid = (min(num_sms, total_tiles), 1)

    # Allocate StreamK workspace buffers
    p = torch.empty(num_sms * block_m * block_n, dtype=torch.float32).cuda()
    locks = torch.empty(num_sms, dtype=torch.int32).cuda()

    warp_bases = _compute_warp_bases(num_warps)

    # Select kernel variant
    if num_warps == 8:
        kernel_fn = streamk_gemm_tdm_pipelined_kernel_8warps
    elif use_prefetch:
        kernel_fn = streamk_gemm_tdm_prefetch_kernel
    else:
        kernel_fn = streamk_gemm_tdm_pipelined_kernel_4warps

    kernel_fn[grid](
        a,
        b,
        c,
        p,
        locks,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_BUFFERS=num_buffers,
        TRANSPOSE_B=transpose_b,
        NUM_WARPS=num_warps,
        WARP_BASES=warp_bases,
        STREAMK_TILES=streamk_tiles,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        waves_per_eu=num_warps // 4,
    )

    return c


__all__ = ['gluon_streamk_matmul']

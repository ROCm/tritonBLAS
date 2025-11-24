import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked, chiplet_transform


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit
def fp4_matmul(
    A,
    B,
    C,
    A_scales,
    B_scales,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B with FP4 inputs.
    A and B inputs are in the microscale fp4 (mxfp4) e2m1 format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    Every 32 elements in the K dimension share one e8m0 scale.
    """
    
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform(pid, NUM_SMS, NUM_XCDS)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)
        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        
        A_BASE = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_BASE = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        # Create pointers for the first block of A and B scales
        offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
        A_scale_BASE = A_scales + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
        # B scales are N x K even though B operand is K x N.
        B_scale_BASE = B_scales + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Calculate loop iterations - note K is unpacked dimension
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        
        for k in range(0, loop_k):
            a_scales = tl.load(A_scale_BASE)
            b_scales = tl.load(B_scale_BASE)
            
            # Load with masks to handle boundaries
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                # offs_k is in packed space (BLOCK_SIZE_K // 2), K is unpacked
                a = tl.load(A_BASE, mask=offs_k[None, :] < (K // 2) - k * (BLOCK_SIZE_K // 2), other=0)
                b = tl.load(B_BASE, mask=offs_k[:, None] < (K // 2) - k * (BLOCK_SIZE_K // 2), other=0)
            
            accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")
            
            # Advance the ptrs to the next K block.
            A_BASE += (BLOCK_SIZE_K // 2) * stride_ak
            B_BASE += (BLOCK_SIZE_K // 2) * stride_bk
            A_scale_BASE += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            B_scale_BASE += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(C.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        C_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(C_ptrs, c, mask=c_mask)

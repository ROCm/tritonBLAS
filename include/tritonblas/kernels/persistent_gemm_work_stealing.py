"""
Work-stealing persistent GEMM kernel.

Instead of statically partitioning tiles across workgroups (for tile_id in
range(pid, total_tiles, NUM_SMS)), each WG dynamically grabs the next
available tile via a global atomic counter.  This naturally load-balances
when some WGs arrive late to the party.
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked

@triton.jit()
def ws_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    tile_counter,  # Global atomic counter for work-stealing (int32[1])
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # ── Work-stealing with chiplet swizzle ────────────────────────────────
    # 1. Grab a raw tile index from a single global atomic counter.
    # 2. Swizzle it through chiplet_transform_chunked so that consecutive
    #    tile_ids land on the same XCD → better L2 locality.
    # 3. The GROUP_SIZE_M decomposition below turns the swizzled tile_id
    #    into (pid_m, pid_n).
    tile_id = tl.atomic_add(tile_counter, 1)
    for _ in range(total_tiles):
        if tile_id < total_tiles:
            # Chiplet-aware swizzle
            if NUM_XCDS != 1:
                tile_id = chiplet_transform_chunked(tile_id, total_tiles, NUM_XCDS, CHUNK_SIZE)

            # GROUP_SIZE_M swizzle → (pid_m, pid_n)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rk = tl.arange(0, BLOCK_SIZE_K)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            if BIAS:
                bias_ = bias_ptr + rm * stride_bias
                bias = tl.load(bias_, mask=rm < M, other=0.0)

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1
            tl.assume(loop_k > 1)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

                # Conditional dot product precision based on quantization mode
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                k = loop_k
                rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                if stride_ak == 1:
                    A_BASE = tl.multiple_of(A_BASE, (1, 16))
                else:
                    A_BASE = tl.multiple_of(A_BASE, (16, 1))

                if stride_bk == 1:
                    B_BASE = tl.multiple_of(B_BASE, (16, 1))
                else:
                    B_BASE = tl.multiple_of(B_BASE, (1, 16))
                a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)

                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            # Conditional scaling for quantized mode
            if QUANTIZED:
                # Create pointers for the scale tensors and load them
                rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
                rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
                A_scale = tl.load(A_scale_ptr + rm_A_scale)
                B_scale = tl.load(B_scale_ptr + rn_B_scale)
                acc *= A_scale[:, None] * B_scale[None, :]

            # Unified bias handling
            if BIAS:
                if QUANTIZED:
                    bias_float = bias.to(tl.float32)
                    c = acc + bias_float[:, None]
                    c = c.to(C.type.element_ty)
                else:
                    c = acc.to(C.type.element_ty)
                    c += bias[:, None]
            else:
                c = acc.to(C.type.element_ty)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, c_mask)

            # Grab next tile
            tile_id = tl.atomic_add(tile_counter, 1)

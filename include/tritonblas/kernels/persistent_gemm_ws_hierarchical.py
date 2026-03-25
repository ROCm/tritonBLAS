"""
Hierarchical Work-Stealing persistent GEMM kernel.

Level 1 (per-XCD): Single counter per XCD, no slots. WGs steal tiles from
  their own XCD's L2-local tile region. Eliminates slot starvation.

Level 2 (global): When a WG's XCD tiles are exhausted, it steals from a
  global counter that covers the remaining ~10% of tiles. This absorbs
  wave quantization across all XCDs.

Tile space split:
  [0, local_per_xcd * NUM_XCDS)  → per-XCD tiles (L2-local, ~90%)
  [local_per_xcd * NUM_XCDS, total_tiles) → global pool (~10%)
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform


@triton.jit()
def ws_hierarchical_matmul(
    A, B, C,
    A_scale_ptr, B_scale_ptr, bias_ptr,
    tile_counter,   # int32[NUM_XCDS * COUNTER_STRIDE]: per-XCD counters
    global_counter, # int32[COUNTER_STRIDE]: single global fallback counter
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn, stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    LOCAL_TILES_PER_XCD: tl.constexpr,
    GLOBAL_TILES: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    USE_MASK: tl.constexpr = True,
    mask_ptr=None,
):
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS

    if USE_MASK:
        mask = tl.load(mask_ptr + pid)
        if mask == 0:
            return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_local = LOCAL_TILES_PER_XCD * NUM_XCDS

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    has_remainder = not EVEN_K
    if has_remainder:
        loop_k -= 1
    tl.assume(loop_k > 1)

    # ================================================================
    # Level 1: Per-XCD stealing (L2-local tiles)
    # ================================================================
    xcd_base = xcd_id * LOCAL_TILES_PER_XCD
    local_counter = tile_counter + xcd_id * COUNTER_STRIDE

    raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")

    while raw_idx < LOCAL_TILES_PER_XCD:
        tile_id = xcd_base + raw_idx

        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

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
            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if has_remainder:
            k = loop_k
            rk_rem = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
            B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1: A_REM = tl.multiple_of(A_REM, (1, 16))
            else: A_REM = tl.multiple_of(A_REM, (16, 1))
            if stride_bk == 1: B_REM = tl.multiple_of(B_REM, (16, 1))
            else: B_REM = tl.multiple_of(B_REM, (1, 16))
            a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            if QUANTIZED: acc += tl.dot(a, b, input_precision="ieee")
            else: acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        if QUANTIZED:
            rm_A = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            acc *= tl.load(A_scale_ptr + rm_A)[:, None] * tl.load(B_scale_ptr + rn_B)[None, :]

        if BIAS:
            if QUANTIZED: c = (acc + bias.to(tl.float32)[:, None]).to(C.type.element_ty)
            else: c = acc.to(C.type.element_ty); c += bias[:, None]
        else:
            c = acc.to(C.type.element_ty)

        next_raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")

        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

        raw_idx = next_raw_idx

    # ================================================================
    # Level 2: Global fallback stealing (cross-XCD, wave balancing)
    # ================================================================
    if GLOBAL_TILES == 0:
        return

    raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")

    while raw_idx < GLOBAL_TILES:
        tile_id = total_local + raw_idx

        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

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
            if QUANTIZED: acc += tl.dot(a, b, input_precision="ieee")
            else: acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if has_remainder:
            k = loop_k
            rk_rem = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
            B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1: A_REM = tl.multiple_of(A_REM, (1, 16))
            else: A_REM = tl.multiple_of(A_REM, (16, 1))
            if stride_bk == 1: B_REM = tl.multiple_of(B_REM, (16, 1))
            else: B_REM = tl.multiple_of(B_REM, (1, 16))
            a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            if QUANTIZED: acc += tl.dot(a, b, input_precision="ieee")
            else: acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        if QUANTIZED:
            rm_A = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            acc *= tl.load(A_scale_ptr + rm_A)[:, None] * tl.load(B_scale_ptr + rn_B)[None, :]

        if BIAS:
            if QUANTIZED: c = (acc + bias.to(tl.float32)[:, None]).to(C.type.element_ty)
            else: c = acc.to(C.type.element_ty); c += bias[:, None]
        else:
            c = acc.to(C.type.element_ty)

        next_raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")

        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

        raw_idx = next_raw_idx

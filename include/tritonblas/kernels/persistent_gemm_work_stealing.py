"""
Work-stealing persistent GEMM kernel with per-XCD atomic counters.

Instead of statically partitioning tiles across workgroups (for tile_id in
range(pid, total_tiles, NUM_SMS)), each WG dynamically grabs the next
available tile via an atomic counter that is local to its XCD.

PIDs are assigned round-robin across XCDs:
    pid 0 → XCD 0, pid 1 → XCD 1, …, pid 7 → XCD 7, pid 8 → XCD 0, …

Supports three scheduling modes (selected via constexpr flags):

1. Per-XCD/Slot (default):
   Each XCD has COUNTERS_PER_XCD independent counters. The XCD's tile region
   is sub-partitioned across slots to reduce atomic contention.

2. Global Atomic (GLOBAL_ATOMIC=True):
   Single device-wide counter with chiplet_transform swizzle to maintain
   L2 locality despite global ordering.

3. Hierarchical (HIERARCHICAL=True):
   Level 1: Per-XCD single counter — WGs steal from their XCD's L2-local
   tile region (LOCAL_TILES_PER_XCD tiles per XCD, ~90% of work).
   Level 2: Global fallback counter — once an XCD exhausts its local pool,
   WGs steal from a shared global_counter covering the remaining tiles.
   This absorbs wave quantization imbalance across XCDs.
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform


@triton.jit()
def ws_persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,
    B_scale_ptr,
    bias_ptr,
    tile_counter,
    global_counter,     # Only used when HIERARCHICAL=True; pass None otherwise
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
    COUNTERS_PER_XCD: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
    GLOBAL_ATOMIC: tl.constexpr = False,
    HIERARCHICAL: tl.constexpr = False,
    LOCAL_TILES_PER_XCD: tl.constexpr = 0,
    GLOBAL_TILES: tl.constexpr = 0,
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

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    if HIERARCHICAL:
        # ================================================================
        # Level 1: Per-XCD stealing (L2-local tiles)
        # ================================================================
        total_local = LOCAL_TILES_PER_XCD * NUM_XCDS
        xcd_base = xcd_id * LOCAL_TILES_PER_XCD
        local_counter = tile_counter + xcd_id * COUNTER_STRIDE

        raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")
        while raw_idx < LOCAL_TILES_PER_XCD:
            tile_id = xcd_base + raw_idx
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            rm_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rk = tl.arange(0, BLOCK_SIZE_K)
            mask_m = rm_raw < M
            mask_n = rn_raw < N

            rm = tl.max_contiguous(tl.multiple_of(rm_raw % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn_raw % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            if BIAS:
                bias_ = bias_ptr + rn * stride_bias
                bias = tl.load(bias_, mask=mask_n, other=0.0)

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1
            tl.assume(loop_k > 1)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                k = loop_k
                rk_rem = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                if stride_ak == 1:
                    A_REM = tl.multiple_of(A_REM, (1, 16))
                else:
                    A_REM = tl.multiple_of(A_REM, (16, 1))
                if stride_bk == 1:
                    B_REM = tl.multiple_of(B_REM, (16, 1))
                else:
                    B_REM = tl.multiple_of(B_REM, (1, 16))
                a = tl.load(A_REM, mask=mask_m[:, None] & (rk_rem[None, :] < K), other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_REM, mask=mask_n[None, :] & (rk_rem[:, None] < K), other=0.0, cache_modifier=CACHE_MODIFIER_B)
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            if QUANTIZED:
                rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                A_scale = tl.load(A_scale_ptr + rm_A_scale, mask=mask_m, other=1.0)
                B_scale = tl.load(B_scale_ptr + rn_B_scale, mask=mask_n, other=1.0)
                acc *= A_scale[:, None] * B_scale[None, :]

            if BIAS:
                if QUANTIZED:
                    c = (acc + bias.to(tl.float32)[None, :]).to(C.type.element_ty)
                else:
                    c = acc.to(C.type.element_ty)
                    c += bias[None, :]
            else:
                c = acc.to(C.type.element_ty)

            c_mask = mask_m[:, None] & mask_n[None, :]
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, c_mask)

            raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")

        # ================================================================
        # Level 2: Global fallback stealing (cross-XCD, wave balancing)
        # ================================================================
        if GLOBAL_TILES > 0:
            raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")
            while raw_idx < GLOBAL_TILES:
                tile_id = total_local + raw_idx
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m
                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                rm_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                rk = tl.arange(0, BLOCK_SIZE_K)
                mask_m = rm_raw < M
                mask_n = rn_raw < N

                rm = tl.max_contiguous(tl.multiple_of(rm_raw % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn_raw % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

                if BIAS:
                    bias_ = bias_ptr + rn * stride_bias
                    bias = tl.load(bias_, mask=mask_n, other=0.0)

                loop_k = tl.cdiv(K, BLOCK_SIZE_K)
                if not EVEN_K:
                    loop_k -= 1
                tl.assume(loop_k > 1)

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                for k in range(0, loop_k):
                    if stride_ak == 1:
                        a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                    else:
                        a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                    if stride_bk == 1:
                        b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                    else:
                        b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                    if QUANTIZED:
                        acc += tl.dot(a, b, input_precision="ieee")
                    else:
                        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                    A_BASE += BLOCK_SIZE_K * stride_ak
                    B_BASE += BLOCK_SIZE_K * stride_bk

                if not EVEN_K:
                    k = loop_k
                    rk_rem = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                    B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                    if stride_ak == 1:
                        A_REM = tl.multiple_of(A_REM, (1, 16))
                    else:
                        A_REM = tl.multiple_of(A_REM, (16, 1))
                    if stride_bk == 1:
                        B_REM = tl.multiple_of(B_REM, (16, 1))
                    else:
                        B_REM = tl.multiple_of(B_REM, (1, 16))
                    a = tl.load(A_REM, mask=mask_m[:, None] & (rk_rem[None, :] < K), other=0.0, cache_modifier=CACHE_MODIFIER_A)
                    b = tl.load(B_REM, mask=mask_n[None, :] & (rk_rem[:, None] < K), other=0.0, cache_modifier=CACHE_MODIFIER_B)
                    if QUANTIZED:
                        acc += tl.dot(a, b, input_precision="ieee")
                    else:
                        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

                if QUANTIZED:
                    rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    A_scale = tl.load(A_scale_ptr + rm_A_scale, mask=mask_m, other=1.0)
                    B_scale = tl.load(B_scale_ptr + rn_B_scale, mask=mask_n, other=1.0)
                    acc *= A_scale[:, None] * B_scale[None, :]

                if BIAS:
                    if QUANTIZED:
                        c = (acc + bias.to(tl.float32)[None, :]).to(C.type.element_ty)
                    else:
                        c = acc.to(C.type.element_ty)
                        c += bias[None, :]
                else:
                    c = acc.to(C.type.element_ty)

                c_mask = mask_m[:, None] & mask_n[None, :]
                C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
                tl.store(C_, c, c_mask)

                raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")
    else:
        # ================================================================
        # Flat work-stealing: per-XCD/slot or global atomic
        # ================================================================
        tiles_per_xcd = tl.cdiv(total_tiles, NUM_XCDS)

        if GLOBAL_ATOMIC:
            counter_ptr = tile_counter
            bound = total_tiles
        else:
            local_wg_id = pid // NUM_XCDS
            slot = local_wg_id % COUNTERS_PER_XCD

            xcd_base = xcd_id * tiles_per_xcd
            xcd_end = tl.minimum(xcd_base + tiles_per_xcd, total_tiles)
            tiles_this_xcd = xcd_end - xcd_base

            tiles_per_slot = tl.cdiv(tiles_this_xcd, COUNTERS_PER_XCD)
            slot_base = slot * tiles_per_slot
            slot_end = tl.minimum(slot_base + tiles_per_slot, tiles_this_xcd)
            bound = slot_end - slot_base

            counter_ptr = tile_counter + (xcd_id * COUNTERS_PER_XCD + slot) * COUNTER_STRIDE

        raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")

        while raw_idx < bound:
            if GLOBAL_ATOMIC:
                tile_id = chiplet_transform(raw_idx, total_tiles, NUM_XCDS)
            else:
                tile_id = xcd_base + slot_base + raw_idx

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            rm_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rk = tl.arange(0, BLOCK_SIZE_K)
            mask_m = rm_raw < M
            mask_n = rn_raw < N

            rm = tl.max_contiguous(tl.multiple_of(rm_raw % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn_raw % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            if BIAS:
                bias_ = bias_ptr + rn * stride_bias
                bias = tl.load(bias_, mask=mask_n, other=0.0)

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1
            tl.assume(loop_k > 1)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                k = loop_k
                rk_rem = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                if stride_ak == 1:
                    A_REM = tl.multiple_of(A_REM, (1, 16))
                else:
                    A_REM = tl.multiple_of(A_REM, (16, 1))
                if stride_bk == 1:
                    B_REM = tl.multiple_of(B_REM, (16, 1))
                else:
                    B_REM = tl.multiple_of(B_REM, (1, 16))
                a = tl.load(A_REM, mask=mask_m[:, None] & (rk_rem[None, :] < K), other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_REM, mask=mask_n[None, :] & (rk_rem[:, None] < K), other=0.0, cache_modifier=CACHE_MODIFIER_B)
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            if QUANTIZED:
                rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                A_scale = tl.load(A_scale_ptr + rm_A_scale, mask=mask_m, other=1.0)
                B_scale = tl.load(B_scale_ptr + rn_B_scale, mask=mask_n, other=1.0)
                acc *= A_scale[:, None] * B_scale[None, :]

            if BIAS:
                if QUANTIZED:
                    c = (acc + bias.to(tl.float32)[None, :]).to(C.type.element_ty)
                else:
                    c = acc.to(C.type.element_ty)
                    c += bias[None, :]
            else:
                c = acc.to(C.type.element_ty)

            c_mask = mask_m[:, None] & mask_n[None, :]
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, c_mask)

            raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")

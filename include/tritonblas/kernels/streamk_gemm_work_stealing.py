"""
Hybrid Stream-K + Work-Stealing GEMM kernel.

Phase 1 (SK): Static pid-based K-iteration assignment for remainder tiles.
  Runs FIRST so all WGs enter simultaneously — no spin-wait stalls.
  Uses ACTIVE_CUS for balanced iteration distribution.

Phase 2 (WS DP): Dynamic work-stealing via per-XCD/slot atomic counters.
  Handles total_tiles - STREAMK_TILES tiles with full K-loop each.
  Slot count clamped to active WGs per XCD to prevent starvation.

Grid: always NUM_SMS (full hardware). mask_ptr gates inactive WGs.
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform
from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.jit()
def ws_streamk_matmul(
    A, B, C,
    A_scale_ptr, B_scale_ptr, bias_ptr,
    tile_counter,   # Per-XCD×slot atomic counters for WS DP tiles
    P,              # Pid-indexed partial accumulator for SK reduction
    locks,          # Pid-indexed lock/signal array for SK reduction
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn, stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ACTIVE_CUS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    STREAMK_TILES: tl.constexpr,
    COUNTERS_PER_XCD: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    GLOBAL_ATOMIC: tl.constexpr = False,
    mask_ptr=None,
):
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS
    orig_local_wg_id = pid // NUM_XCDS

    mask = tl.load(mask_ptr + pid)
    if mask == 0:
        return

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    has_remainder = not EVEN_K
    if has_remainder:
        loop_k -= 1
    tl.assume(loop_k > 1)

    # ================================================================
    # Phase 1: Static Stream-K for remainder tiles (runs FIRST)
    # ================================================================
    if STREAMK_TILES > 0:
        rm1 = tl.arange(0, BLOCK_SIZE_M)
        rn1 = tl.arange(0, BLOCK_SIZE_N)
        rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)

        tl.assume(pid >= 0)
        iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
        total_streamk_iters = STREAMK_TILES * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // ACTIVE_CUS
        streamk_remainder_iters = total_streamk_iters % ACTIVE_CUS

        if pid < ACTIVE_CUS:
            start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
            last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)

            while start_iter < last_iter:
                remainder = start_iter % iters_per_tile
                end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
                tile_id = start_iter // iters_per_tile
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m
                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                rk2 = tl.arange(0, BLOCK_SIZE_K)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
                A_BASE = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
                B_BASE = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
                if stride_ak == 1: A_BASE = tl.multiple_of(A_BASE, (1, 16))
                else: A_BASE = tl.multiple_of(A_BASE, (16, 1))
                if stride_bk == 1: B_BASE = tl.multiple_of(B_BASE, (16, 1))
                else: B_BASE = tl.multiple_of(B_BASE, (1, 16))

                if BIAS:
                    bias_ = bias_ptr + rm * stride_bias
                    bias = tl.load(bias_, mask=rm < M, other=0.0)

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                for current_iter in range(start_iter, end_iter):
                    if EVEN_K:
                        a = tl.load(A_BASE, cache_modifier=CACHE_MODIFIER_A)
                        b = tl.load(B_BASE, cache_modifier=CACHE_MODIFIER_B)
                    else:
                        gk = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                        k_mask = gk + rk2 < K
                        a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                        b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)
                    if QUANTIZED: acc += tl.dot(a, b, input_precision="ieee")
                    else: acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                    A_BASE += BLOCK_SIZE_K * stride_ak
                    B_BASE += BLOCK_SIZE_K * stride_bk

                if QUANTIZED:
                    rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
                    rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
                    acc *= tl.load(A_scale_ptr + rm_A_scale)[:, None] * tl.load(B_scale_ptr + rn_B_scale)[None, :]

                tile_iter = tile_id * iters_per_tile
                if start_iter != tile_iter:
                    P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                    tl.store(P_, acc, cache_modifier=".wt")
                    tl.debug_barrier()
                    tl.store(locks + pid, 1, cache_modifier=".wt")
                else:
                    next_pid = pid + 1
                    tile_iter_end = tile_iter + iters_per_tile
                    end = end_iter
                    acc_m = tl.reshape(acc, (2, BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
                    acc_mp = tl.permute(acc_m, (1, 2, 0))
                    acc_top, acc_bot = tl.split(acc_mp)
                    acc_top = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
                    acc_bot = tl.reshape(acc_bot, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
                    at_r = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
                    at_p = tl.permute(at_r, (0, 2, 1))
                    acc00, acc01 = tl.split(at_p)
                    ab_r = tl.reshape(acc_bot, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
                    ab_p = tl.permute(ab_r, (0, 2, 1))
                    acc10, acc11 = tl.split(ab_p)
                    acc00 = tl.reshape(acc00, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
                    acc01 = tl.reshape(acc01, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
                    acc10 = tl.reshape(acc10, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
                    acc11 = tl.reshape(acc11, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
                    while (end < tile_iter_end and next_pid < ACTIVE_CUS):
                        while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                            pass
                        Pb = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                        acc00 += tl.load(Pb + tl.arange(0, BLOCK_SIZE_M//2)[:, None]*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N//2)[None, :], cache_modifier=".cv")
                        acc01 += tl.load(Pb + tl.arange(0, BLOCK_SIZE_M//2)[:, None]*BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N//2)[None, :]+BLOCK_SIZE_N//2), cache_modifier=".cv")
                        acc10 += tl.load(Pb + (tl.arange(0, BLOCK_SIZE_M//2)[:, None]+BLOCK_SIZE_M//2)*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N//2)[None, :], cache_modifier=".cv")
                        acc11 += tl.load(Pb + (tl.arange(0, BLOCK_SIZE_M//2)[:, None]+BLOCK_SIZE_M//2)*BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N//2)[None, :]+BLOCK_SIZE_N//2), cache_modifier=".cv")
                        end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                        next_pid += 1
                    if BIAS:
                        br = tl.reshape(bias, (BLOCK_SIZE_M, 1))
                        bt, bb = br[:BLOCK_SIZE_M//2], br[BLOCK_SIZE_M//2:]
                        if QUANTIZED: acc00 += bt.to(tl.float32); acc01 += bt.to(tl.float32); acc10 += bb.to(tl.float32); acc11 += bb.to(tl.float32)
                        else: acc00 += bt; acc01 += bt; acc10 += bb; acc11 += bb
                    c00, c01 = acc00.to(C.type.element_ty), acc01.to(C.type.element_ty)
                    c10, c11 = acc10.to(C.type.element_ty), acc11.to(C.type.element_ty)
                    rm_t = (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M//2)) % M
                    rm_b = (pid_m*BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M//2, BLOCK_SIZE_M)) % M
                    rn_l = (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N//2)) % N
                    rn_r = (pid_n*BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N//2, BLOCK_SIZE_N)) % N
                    tl.store(C+rm_t[:,None]*stride_cm+rn_l[None,:]*stride_cn, c00, mask=(rm_t<M)[:,None]&(rn_l<N)[None,:])
                    tl.store(C+rm_t[:,None]*stride_cm+rn_r[None,:]*stride_cn, c01, mask=(rm_t<M)[:,None]&(rn_r<N)[None,:])
                    tl.store(C+rm_b[:,None]*stride_cm+rn_l[None,:]*stride_cn, c10, mask=(rm_b<M)[:,None]&(rn_l<N)[None,:])
                    tl.store(C+rm_b[:,None]*stride_cm+rn_r[None,:]*stride_cn, c11, mask=(rm_b<M)[:,None]&(rn_r<N)[None,:])
                start_iter = end_iter

    # ================================================================
    # Phase 2: Work-stealing DP tiles (full K-loop each)
    #   Slot count clamped to active WGs per XCD to prevent starvation.
    # ================================================================
    tiles_per_xcd = tl.cdiv(total_full_tiles, NUM_XCDS)
    if GLOBAL_ATOMIC:
        counter_ptr = tile_counter
        bound = total_full_tiles
    else:
        active_per_xcd = ACTIVE_CUS // NUM_XCDS
        eff_cpx = min(COUNTERS_PER_XCD, active_per_xcd)
        eff_cpx = max(eff_cpx, 1)
        slot = orig_local_wg_id % eff_cpx
        xcd_base = xcd_id * tiles_per_xcd
        xcd_end = tl.minimum(xcd_base + tiles_per_xcd, total_full_tiles)
        tiles_this_xcd = xcd_end - xcd_base
        tiles_per_slot = tl.cdiv(tiles_this_xcd, eff_cpx)
        slot_base = slot * tiles_per_slot
        slot_end = tl.minimum(slot_base + tiles_per_slot, tiles_this_xcd)
        bound = slot_end - slot_base
        counter_ptr = tile_counter + (xcd_id * eff_cpx + slot) * COUNTER_STRIDE

    raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")
    while raw_idx < bound:
        if GLOBAL_ATOMIC: tile_id = chiplet_transform(raw_idx, total_full_tiles, NUM_XCDS)
        else: tile_id = xcd_base + slot_base + raw_idx
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
            if stride_ak == 1: a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else: a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)
            if stride_bk == 1: b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else: b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)
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
        else: c = acc.to(C.type.element_ty)
        next_raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
        raw_idx = next_raw_idx

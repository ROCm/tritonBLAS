"""
Stream-K FP4 GEMM kernel.

Combines the Stream-K load-balancing pattern from streamk_gemm with FP4
(microscale fp4 e2m1) matrix multiplication from fp4_matmul.
A and B are packed 2 FP4 values per uint8; scales are e8m0, one per 32 K elements.
"""

import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit
def fp4_streamk_matmul(
    A,
    B,
    C,
    A_scales,
    B_scales,
    P,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
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
    STREAMK_TILES: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Stream-K kernel for FP4 matmul C = A x B.
    A and B are in microscale fp4 (mxfp4) e2m1 format, packed 2 per uint8.
    A_scales and B_scales are e8m0, one per 32 elements in K.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

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

    SCALE_GROUP_SIZE: tl.constexpr = 32
    # FP4: packed K has K//2 elements; each iter covers BLOCK_SIZE_K elements (unpacked)
    PACKED_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    SCALES_PER_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

    # -------------------------------------------------------------------------
    # Full tiles loop
    # -------------------------------------------------------------------------
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
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
        rk = tl.arange(0, PACKED_BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        rks = tl.arange(0, SCALES_PER_BLOCK_K)
        A_scale_BASE = A_scales + rm[:, None] * stride_asm + rks[None, :] * stride_ask
        B_scale_BASE = B_scales + rn[:, None] * stride_bsn + rks[None, :] * stride_bsk

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k >= 0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, loop_k):
            if BLOCK_SIZE_M >= 32 and BLOCK_SIZE_N >= 32 and BLOCK_SIZE_K >= 256:
                a_scales = (
                    tl.load(A_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_M // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = (
                    tl.load(B_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            else:
                a_scales = tl.load(A_scale_BASE)
                b_scales = tl.load(B_scale_BASE)

            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))

            acc += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

            A_BASE += PACKED_BLOCK_K * stride_ak
            B_BASE += PACKED_BLOCK_K * stride_bk
            A_scale_BASE += SCALES_PER_BLOCK_K * stride_ask
            B_scale_BASE += SCALES_PER_BLOCK_K * stride_bsk

        if not EVEN_K:
            k = loop_k
            remaining_packed = (K // 2) - k * PACKED_BLOCK_K
            rk_peel = tl.arange(0, PACKED_BLOCK_K)
            A_BASE = A + rm[:, None] * stride_am + (k * PACKED_BLOCK_K + rk_peel)[None, :] * stride_ak
            B_BASE = B + (k * PACKED_BLOCK_K + rk_peel)[:, None] * stride_bk + rn[None, :] * stride_bn
            A_scale_BASE = A_scales + rm[:, None] * stride_asm + (k * SCALES_PER_BLOCK_K + tl.arange(0, SCALES_PER_BLOCK_K))[None, :] * stride_ask
            B_scale_BASE = B_scales + rn[:, None] * stride_bsn + (k * SCALES_PER_BLOCK_K + tl.arange(0, SCALES_PER_BLOCK_K))[None, :] * stride_bsk
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))
            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            if BLOCK_SIZE_M >= 32 and BLOCK_SIZE_N >= 32 and BLOCK_SIZE_K >= 256:
                a_scales = (
                    tl.load(A_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_M // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = (
                    tl.load(B_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            else:
                a_scales = tl.load(A_scale_BASE)
                b_scales = tl.load(B_scale_BASE)
            a = tl.load(A_BASE, mask=rk_peel[None, :] < remaining_packed, other=0)
            b = tl.load(B_BASE, mask=rk_peel[:, None] < remaining_packed, other=0)
            acc += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        c = acc.to(C.type.element_ty)
        rm_out = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm_out = tl.max_contiguous(tl.multiple_of(rm_out, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_out = tl.max_contiguous(tl.multiple_of(rn_out, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm_out[:, None] < M) & (rn_out[None, :] < N)
        C_ = C + rm_out[:, None] * stride_cm + rn_out[None, :] * stride_cn
        tl.store(C_, c, mask=mask)

    if STREAMK_TILES == 0:
        return

    # -------------------------------------------------------------------------
    # Stream-K: initialize P buffer and locks
    # -------------------------------------------------------------------------
    rm1 = tl.arange(0, BLOCK_SIZE_M)
    rn1 = tl.arange(0, BLOCK_SIZE_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
    P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
    tl.store(P_, 0.0, cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(locks + pid, 0, cache_modifier=".wt")

    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(
        pid, streamk_remainder_iters
    )
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(
        pid + 1, streamk_remainder_iters
    )

    # -------------------------------------------------------------------------
    # Stream-K main loop
    # -------------------------------------------------------------------------
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
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
        rk = tl.arange(0, PACKED_BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Base pointers for this tile, offset by remainder iters
        k_offset_packed = remainder * PACKED_BLOCK_K
        k_offset_scales = remainder * SCALES_PER_BLOCK_K
        A_BASE = (
            A
            + rm[:, None] * stride_am
            + (rk[None, :] + k_offset_packed) * stride_ak
        )
        B_BASE = (
            B
            + (rk[:, None] + k_offset_packed) * stride_bk
            + rn[None, :] * stride_bn
        )
        rks = tl.arange(0, SCALES_PER_BLOCK_K) + k_offset_scales
        A_scale_BASE = A_scales + rm[:, None] * stride_asm + rks[None, :] * stride_ask
        B_scale_BASE = B_scales + rn[:, None] * stride_bsn + rks[None, :] * stride_bsk

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for current_iter in range(start_iter, end_iter):
            if BLOCK_SIZE_M >= 32 and BLOCK_SIZE_N >= 32 and BLOCK_SIZE_K >= 256:
                a_scales = (
                    tl.load(A_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_M // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = (
                    tl.load(B_scale_BASE)
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            else:
                a_scales = tl.load(A_scale_BASE)
                b_scales = tl.load(B_scale_BASE)

            if EVEN_K:
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
            else:
                global_k_packed = (current_iter % iters_per_tile) * PACKED_BLOCK_K
                remaining = (K // 2) - global_k_packed
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=rk[None, :] < remaining, other=0)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=rk[None, :] < remaining, other=0)
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=rk[:, None] < remaining, other=0)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=rk[:, None] < remaining, other=0)

            acc += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

            A_BASE += PACKED_BLOCK_K * stride_ak
            B_BASE += PACKED_BLOCK_K * stride_bk
            A_scale_BASE += SCALES_PER_BLOCK_K * stride_ask
            B_scale_BASE += SCALES_PER_BLOCK_K * stride_bsk

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            # Partial tile: store accumulator and signal completion
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store(locks + pid, 1, cache_modifier=".wt")
        else:
            # Complete tile: aggregate from other PEs and store result
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            # Split accumulator into 4 quadrants for efficient aggregation
            acc_m_reshaped = tl.reshape(acc, (2, BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_m_permuted = tl.permute(acc_m_reshaped, (1, 2, 0))
            acc_top, acc_bottom = tl.split(acc_m_permuted)
            acc_top = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_bottom = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))

            acc_top_reshaped = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_top_permuted = tl.permute(acc_top_reshaped, (0, 2, 1))
            acc00, acc01 = tl.split(acc_top_permuted)
            acc_bottom_reshaped = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_bottom_permuted = tl.permute(acc_bottom_reshaped, (0, 2, 1))
            acc10, acc11 = tl.split(acc_bottom_permuted)
            acc00 = tl.reshape(acc00, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc01 = tl.reshape(acc01, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc10 = tl.reshape(acc10, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc11 = tl.reshape(acc11, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))

            while end < tile_iter_end and next_pid < NUM_SMS:
                while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_base = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N

                P_00 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + tl.arange(
                    0, BLOCK_SIZE_N // 2
                )[None, :]
                acc00 += tl.load(P_00, cache_modifier=".cv")

                P_01 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + (
                    tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2
                )
                acc01 += tl.load(P_01, cache_modifier=".cv")

                P_10 = P_base + (
                    tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2
                ) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc10 += tl.load(P_10, cache_modifier=".cv")

                P_11 = P_base + (
                    tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2
                ) * BLOCK_SIZE_N + (
                    tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2
                )
                acc11 += tl.load(P_11, cache_modifier=".cv")

                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c00 = acc00.to(C.type.element_ty)
            c01 = acc01.to(C.type.element_ty)
            c10 = acc10.to(C.type.element_ty)
            c11 = acc11.to(C.type.element_ty)

            rm_top = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)) % M
            rm_bottom = (pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)) % M
            rn_left = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)) % N
            rn_right = (pid_n * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N // 2, BLOCK_SIZE_N)) % N

            mask00 = (rm_top < M)[:, None] & (rn_left < N)[None, :]
            tl.store(
                C + rm_top[:, None] * stride_cm + rn_left[None, :] * stride_cn,
                c00,
                mask=mask00,
            )
            mask01 = (rm_top < M)[:, None] & (rn_right < N)[None, :]
            tl.store(
                C + rm_top[:, None] * stride_cm + rn_right[None, :] * stride_cn,
                c01,
                mask=mask01,
            )
            mask10 = (rm_bottom < M)[:, None] & (rn_left < N)[None, :]
            tl.store(
                C + rm_bottom[:, None] * stride_cm + rn_left[None, :] * stride_cn,
                c10,
                mask=mask10,
            )
            mask11 = (rm_bottom < M)[:, None] & (rn_right < N)[None, :]
            tl.store(
                C + rm_bottom[:, None] * stride_cm + rn_right[None, :] * stride_cn,
                c11,
                mask=mask11,
            )

        start_iter = end_iter

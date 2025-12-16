import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked

@triton.jit()
def streamk_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    P,
    locks,
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
    STREAMK_TILES: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # Full tiles loop
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # NOTE: Do NOT use modulo here - it breaks boundary masking for partial tiles
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        # Compute masks for boundary handling
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

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
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)

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
            a = tl.load(A_BASE, mask=mask_m & (rk[None, :] < K), other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_BASE, mask=mask_n & (rk[:, None] < K), other=0.0, cache_modifier=CACHE_MODIFIER_B)

            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        # Conditional scaling for quantized mode
        if QUANTIZED:
            # Create pointers for the scale tensors and load them
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            A_scale = tl.load(A_scale_ptr + rm_A_scale, mask=rm_A_scale < M, other=0.0)
            B_scale = tl.load(B_scale_ptr + rn_B_scale, mask=rn_B_scale < N, other=0.0)
            acc *= A_scale[:, None] * B_scale[None, :]

        # Unified bias handling for full tiles
        if BIAS:
            if QUANTIZED:
                # For quantized mode: convert bias to float32, add to acc, then convert to output dtype
                bias_float = bias.to(tl.float32)
                c = acc + bias_float[:, None]
                c = c.to(C.type.element_ty)
            else:
                # For non-quantized mode: convert acc to output dtype, then add bias
                c = acc.to(C.type.element_ty)
                c += bias[:, None]
        else:
            c = acc.to(C.type.element_ty)

        # Store using original rm/rn (without modulo) for correct masking
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=mask)

    if STREAMK_TILES == 0:
        return

    # Initialize shared memory buffers for Stream-K
    rm1 = tl.arange(0, BLOCK_SIZE_M)
    rn1 = tl.arange(0, BLOCK_SIZE_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
    P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
    tl.store(P_, 0.0, cache_modifier=".wt")
    tl.store(locks + pid, 0, cache_modifier=".wt")

    tl.assume(pid >= 0)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(
        pid + 1, streamk_remainder_iters)

    # Stream-K main loop
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

        # NOTE: Do NOT use modulo here - it breaks boundary masking for partial tiles
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))

        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))

        # Compute masks for boundary handling
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE, mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=mask_m & k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, mask=mask_n & k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)

            # Conditional dot product precision for Stream-K loop
            if QUANTIZED:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        # Conditional scaling for quantized mode in Stream-K section
        if QUANTIZED:
            # Create pointers for the scale tensors and load them
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            A_scale = tl.load(A_scale_ptr + rm_A_scale, mask=rm_A_scale < M, other=0.0)
            B_scale = tl.load(B_scale_ptr + rn_B_scale, mask=rn_B_scale < N, other=0.0)
            acc *= A_scale[:, None] * B_scale[None, :]

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
            #leave atomic_xchg/atomc_cas implementation here for gfx940
            #as it doesn't support cache_modifier.
            # tl.store(P_, acc)
            # tl.debug_barrier()
            # tl.atomic_xchg(locks + pid, 1)
        else:
            # Complete tile: aggregate from other PEs and store result
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            # Split accumulator into 4 quadrants for efficient aggregation
            # First split in M direction
            acc_m_reshaped = tl.reshape(acc, (2, BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_m_permuted = tl.permute(acc_m_reshaped, (1, 2, 0))  # (M//2, N, 2)
            acc_top, acc_bottom = tl.split(acc_m_permuted)  # Split along last dimension

            # Remove singleton dimension - each is now (M//2, N)
            acc_top = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_bottom = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))

            # Now split each half in N direction
            acc_top_reshaped = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_top_permuted = tl.permute(acc_top_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc00, acc01 = tl.split(acc_top_permuted)  # Split along last dimension

            acc_bottom_reshaped = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_bottom_permuted = tl.permute(acc_bottom_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc10, acc11 = tl.split(acc_bottom_permuted)  # Split along last dimension

            # Remove singleton dimensions - each is now (M//2, N//2)
            acc00 = tl.reshape(acc00, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc01 = tl.reshape(acc01, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc10 = tl.reshape(acc10, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc11 = tl.reshape(acc11, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))

            # Aggregate from other processing elements
            while (end < tile_iter_end and next_pid < NUM_SMS):
                #while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)

                # Load P in 4 quadrants
                P_base = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N

                # Quadrant 00 (top-left)
                P_00 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc00 += tl.load(P_00, cache_modifier=".cv")

                # Quadrant 01 (top-right)
                P_01 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc01 += tl.load(P_01, cache_modifier=".cv")

                # Quadrant 10 (bottom-left)
                P_10 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc10 += tl.load(P_10, cache_modifier=".cv")

                # Quadrant 11 (bottom-right)
                P_11 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc11 += tl.load(P_11, cache_modifier=".cv")

                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            # Unified bias handling for Stream-K section
            if BIAS:
                # Split bias for top and bottom halves
                bias_top = bias[:BLOCK_SIZE_M // 2]
                bias_bottom = bias[BLOCK_SIZE_M // 2:]

                bias_top_reshaped = tl.reshape(bias_top, (BLOCK_SIZE_M // 2, 1))
                bias_bottom_reshaped = tl.reshape(bias_bottom, (BLOCK_SIZE_M // 2, 1))

                if QUANTIZED:
                    # For quantized mode: convert bias to float32 before adding
                    bias_top_float = bias_top_reshaped.to(tl.float32)
                    bias_bottom_float = bias_bottom_reshaped.to(tl.float32)
                    acc00 += bias_top_float
                    acc01 += bias_top_float
                    acc10 += bias_bottom_float
                    acc11 += bias_bottom_float
                else:
                    # For non-quantized mode: add bias directly
                    acc00 += bias_top_reshaped
                    acc01 += bias_top_reshaped
                    acc10 += bias_bottom_reshaped
                    acc11 += bias_bottom_reshaped

            # Convert to output dtype
            c00 = acc00.to(C.type.element_ty)
            c01 = acc01.to(C.type.element_ty)
            c10 = acc10.to(C.type.element_ty)
            c11 = acc11.to(C.type.element_ty)

            # Store all 4 quadrants - NOTE: Do NOT use modulo for correct masking
            rm_top = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)
            rm_bottom = pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)
            rn_left = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)
            rn_right = pid_n * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N // 2, BLOCK_SIZE_N)

            # Store quadrant 00 (top-left)
            mask00 = (rm_top < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C + rm_top[:, None] * stride_cm + rn_left[None, :] * stride_cn, c00, mask=mask00)

            # Store quadrant 01 (top-right)
            mask01 = (rm_top < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C + rm_top[:, None] * stride_cm + rn_right[None, :] * stride_cn, c01, mask=mask01)

            # Store quadrant 10 (bottom-left)
            mask10 = (rm_bottom < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C + rm_bottom[:, None] * stride_cm + rn_left[None, :] * stride_cn, c10, mask=mask10)

            # Store quadrant 11 (bottom-right)
            mask11 = (rm_bottom < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C + rm_bottom[:, None] * stride_cm + rn_right[None, :] * stride_cn, c11, mask=mask11)

        start_iter = end_iter

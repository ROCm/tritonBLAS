"""
Work-stealing persistent GEMM kernel with per-XCD atomic counters.

Instead of statically partitioning tiles across workgroups (for tile_id in
range(pid, total_tiles, NUM_SMS)), each WG dynamically grabs the next
available tile via an atomic counter that is local to its XCD.

PIDs are assigned round-robin across XCDs:
    pid 0 → XCD 0, pid 1 → XCD 1, …, pid 7 → XCD 7, pid 8 → XCD 0, …

The tile space is partitioned into contiguous per-XCD regions:
    XCD i owns tiles [i * tiles_per_xcd, min((i+1) * tiles_per_xcd, total_tiles))

To reduce atomic contention, each XCD uses COUNTERS_PER_XCD independent
counters (default 16).  The XCD's tiles are further sub-partitioned:
    counter slot j within XCD i owns tiles
        [xcd_base + j * tiles_per_slot, xcd_base + min((j+1) * tiles_per_slot, tiles_this_xcd))

Each WG picks its slot via:  slot = (pid // NUM_XCDS) % COUNTERS_PER_XCD

With 38 CUs per XCD and 16 slots, only ~2-3 CUs contend on each counter.
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
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    tile_counter,  # Per-XCD×slot atomic counters (int32[NUM_XCDS * COUNTERS_PER_XCD])
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
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    GLOBAL_ATOMIC: tl.constexpr = False,  # True: single device-wide counter
):
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tiles_per_xcd = tl.cdiv(total_tiles, NUM_XCDS)

    if GLOBAL_ATOMIC:
        # Single device-wide atomic — all CUs contend on one counter.
        counter_ptr = tile_counter
        bound = total_tiles
    else:
        # Per-XCD counters with multiple slots to reduce contention.
        local_wg_id = pid // NUM_XCDS
        slot = local_wg_id % COUNTERS_PER_XCD

        xcd_base = xcd_id * tiles_per_xcd
        xcd_end = tl.minimum(xcd_base + tiles_per_xcd, total_tiles)
        tiles_this_xcd = xcd_end - xcd_base

        tiles_per_slot = tl.cdiv(tiles_this_xcd, COUNTERS_PER_XCD)
        slot_base = slot * tiles_per_slot
        slot_end = tl.minimum(slot_base + tiles_per_slot, tiles_this_xcd)
        bound = slot_end - slot_base

        # Counters are padded to COUNTER_STRIDE int32 elements (256B) apart
        # to avoid false sharing across L2 cache lines.
        counter_ptr = tile_counter + (xcd_id * COUNTERS_PER_XCD + slot) * COUNTER_STRIDE

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")

    while raw_idx < bound:
        # Map raw counter value → global tile_id
        if GLOBAL_ATOMIC:
            # Chiplet swizzle: remap global sequential index into
            # per-XCD tile regions so data stays in the issuing XCD's L2.
            tile_id = chiplet_transform(raw_idx, total_tiles, NUM_XCDS)
        else:
            tile_id = xcd_base + slot_base + raw_idx

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

        if QUANTIZED:
            rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
            rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            A_scale = tl.load(A_scale_ptr + rm_A_scale)
            B_scale = tl.load(B_scale_ptr + rn_B_scale)
            acc *= A_scale[:, None] * B_scale[None, :]

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

        raw_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")

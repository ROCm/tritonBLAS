"""
Triton kernel implementations for mosaic shuffle GEMM scheduling.

Three strategies:
1. random: Lookup-table permutation of individual output tiles
2. l2_aware: Mosaic 2-level L2 tiling + lookup-table shuffle of L2 tile groups
3. llc_and_l2_aware: Mosaic 3-level deterministic hierarchy (LayoutRank2Depth3)
"""

import triton
import triton.language as tl
import torch

from ..random_grid.kernels import (
    remap_xcd_chunked,
    compute_level_index,
    _read_realtime,
    _get_xcc_id,
)


# ===================================================================
# Strategy 1: random -- Lookup-table permutation of individual tiles
# ===================================================================

@triton.jit
def random_transform(tile_id, perm_table_ptr):
    """Permute a single tile index via host-computed lookup table."""
    return tl.load(perm_table_ptr + tile_id)


@triton.jit()
def persistent_matmul_random(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    trace_start_ptr,
    trace_end_ptr,
    trace_pid_ptr,
    trace_xcd_ptr,
    perm_table_ptr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 4,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    TRACE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=CHUNK_SIZE)
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

    for tile_id in range(pid, total_tiles, NUM_SMS):
        shuffled_pid = random_transform(tile_id, perm_table_ptr)
        pid_m = shuffled_pid // num_pid_n
        pid_n = shuffled_pid % num_pid_n
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        if TRACE:
            flat_tile_id = pid_m * num_pid_n + pid_n
            tl.store(trace_start_ptr + flat_tile_id, _read_realtime())
            tl.store(trace_pid_ptr + flat_tile_id, tl.program_id(0))
            tl.store(trace_xcd_ptr + flat_tile_id, _get_xcc_id())

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
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
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
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)

        if TRACE:
            tl.store(trace_end_ptr + flat_tile_id, _read_realtime())


@triton.jit()
def persistent_matmul_debug_map_random(
    workgroup_map,
    M,
    N,
    perm_table_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    original_pid = pid
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        shuffled_pid = random_transform(tile_id, perm_table_ptr)
        tl.store(workgroup_map + shuffled_pid, original_pid)


# ===================================================================
# Strategy 2: l2_aware -- Mosaic L2 tiling + lookup-table shuffle
# ===================================================================

@triton.jit
def l2_aware_transform(
    index,
    grid_y,
    grid_x,
    perm_table_ptr,
    TILE_Y: tl.constexpr,
    TILE_X: tl.constexpr,
    INNER_ORDER: tl.constexpr,
):
    """
    Mosaic-style 2-level decomposition with lookup-table shuffle at the outer level.

    Tiles within each L2 tile group follow INNER_ORDER (preserving cache locality).
    L2 tile groups are randomly permuted via a host-computed permutation table.
    Fringe regions (right strip, bottom strip) fall back to row-major.
    """
    quantized_x = (grid_x // TILE_X) * TILE_X
    quantized_y = (grid_y // TILE_Y) * TILE_Y
    total_quantized = quantized_x * quantized_y
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    y_region_start = (total_quantized - 1) + (non_quantized_x * grid_y)

    if index > total_quantized - 1:
        if index > y_region_start:
            new_grid_x = (
                (index - total_quantized - (non_quantized_x * grid_y)) // non_quantized_y
            ) % grid_x
            new_grid_y = quantized_y + (index % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            new_grid_x = quantized_x + (index % non_quantized_x)
            new_grid_y = ((index - total_quantized) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

    inner_x_idx, inner_y_idx, _ = compute_level_index(
        index, TILE_X, TILE_Y, INNER_ORDER, 1
    )
    tile_linear_idx = index // (TILE_Y * TILE_X)

    shuffled_idx = tl.load(perm_table_ptr + tile_linear_idx)

    tiles_per_row = quantized_x // TILE_X
    outer_x = shuffled_idx % tiles_per_row
    outer_y = shuffled_idx // tiles_per_row

    new_grid_x = outer_x * TILE_X + inner_x_idx
    new_grid_y = outer_y * TILE_Y + inner_y_idx
    return new_grid_y * grid_x + new_grid_x


@triton.jit()
def persistent_matmul_l2_aware(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    trace_start_ptr,
    trace_end_ptr,
    trace_pid_ptr,
    trace_xcd_ptr,
    perm_table_ptr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_Y: tl.constexpr,
    TILE_X: tl.constexpr,
    INNER_ORDER: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 4,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    TRACE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=CHUNK_SIZE)
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

    for tile_id in range(pid, total_tiles, NUM_SMS):
        transformed = l2_aware_transform(
            tile_id, num_pid_m, num_pid_n, perm_table_ptr,
            TILE_Y=TILE_Y, TILE_X=TILE_X, INNER_ORDER=INNER_ORDER,
        )
        pid_m = transformed // num_pid_n
        pid_n = transformed % num_pid_n
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        if TRACE:
            flat_tile_id = pid_m * num_pid_n + pid_n
            tl.store(trace_start_ptr + flat_tile_id, _read_realtime())
            tl.store(trace_pid_ptr + flat_tile_id, tl.program_id(0))
            tl.store(trace_xcd_ptr + flat_tile_id, _get_xcc_id())

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
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
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
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)

        if TRACE:
            tl.store(trace_end_ptr + flat_tile_id, _read_realtime())


@triton.jit()
def persistent_matmul_debug_map_l2_aware(
    workgroup_map,
    M,
    N,
    perm_table_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TILE_Y: tl.constexpr,
    TILE_X: tl.constexpr,
    INNER_ORDER: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    original_pid = pid
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        transformed = l2_aware_transform(
            tile_id, num_pid_m, num_pid_n, perm_table_ptr,
            TILE_Y=TILE_Y, TILE_X=TILE_X, INNER_ORDER=INNER_ORDER,
        )
        tl.store(workgroup_map + transformed, original_pid)


# ===================================================================
# Strategy 3: llc_and_l2_aware -- 3-level deterministic hierarchy
# ===================================================================

@triton.jit
def llc_and_l2_aware_transform_quantized(
    index,
    grid_y,
    grid_x,
    ordering0: tl.constexpr,
    ordering1: tl.constexpr,
    ordering2: tl.constexpr,
    L3Y: tl.constexpr,
    L3X: tl.constexpr,
    L2Y: tl.constexpr,
    L2X: tl.constexpr,
):
    """Mosaic LayoutRank2Depth3 decomposition for the quantized region."""
    new_grid_x = 0
    new_grid_y = 0
    cumulative_denominator = 1
    cumulative_x = 1
    cumulative_y = 1

    # Inner level (L2)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        index, L2X, L2Y, ordering2, cumulative_denominator
    )
    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= L2X
    cumulative_y *= L2Y

    # Middle level (LLC/L3)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        index, L3X, L3Y, ordering1, cumulative_denominator
    )
    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= L3X
    cumulative_y *= L3Y

    # Outer level
    outer_x = grid_x // (L2X * L3X)
    outer_y = grid_y // (L2Y * L3Y)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        index, outer_x, outer_y, ordering0, cumulative_denominator
    )
    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y

    return new_grid_y * grid_x + new_grid_x


@triton.jit
def llc_and_l2_aware_transform(
    index,
    grid_y,
    grid_x,
    ordering0: tl.constexpr,
    ordering1: tl.constexpr,
    ordering2: tl.constexpr,
    L3Y: tl.constexpr,
    L3X: tl.constexpr,
    L2Y: tl.constexpr,
    L2X: tl.constexpr,
):
    """
    Mosaic LayoutRank2Depth3 with 3-region boundary handling.

    Region 1 (quantized): Full 3-level hierarchical decomposition
    Region 2 (right strip): Row-major fallback for remainder in X
    Region 3 (bottom strip): Row-major fallback for remainder in Y

    Follows the same if/return pattern as the working transform_depth3
    in random_grid/kernels.py.
    """
    timestep_x = L2X * L3X
    timestep_y = L2Y * L3Y
    quantized_x = (grid_x // timestep_x) * timestep_x
    quantized_y = (grid_y // timestep_y) * timestep_y
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    total_quantized = quantized_x * quantized_y
    y_region_start = (total_quantized - 1) + (non_quantized_x * grid_y)

    if index <= total_quantized - 1:
        local = llc_and_l2_aware_transform_quantized(
            index, quantized_y, quantized_x,
            ordering0, ordering1, ordering2,
            L3Y, L3X, L2Y, L2X,
        )
        new_grid_y = local // quantized_x
        new_grid_x = local % quantized_x
        return new_grid_y * grid_x + new_grid_x

    if index > y_region_start:
        new_grid_x = (
            (index - total_quantized - (non_quantized_x * grid_y)) // non_quantized_y
        ) % grid_x
        new_grid_y = quantized_y + (index % non_quantized_y)
        return new_grid_y * grid_x + new_grid_x

    new_grid_x = quantized_x + (index % non_quantized_x)
    new_grid_y = ((index - total_quantized) // non_quantized_x) % grid_y
    return (new_grid_y * grid_x) + new_grid_x


@triton.jit()
def persistent_matmul_llc_and_l2_aware(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    trace_start_ptr,
    trace_end_ptr,
    trace_pid_ptr,
    trace_xcd_ptr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ordering0: tl.constexpr,
    ordering1: tl.constexpr,
    ordering2: tl.constexpr,
    L3Y: tl.constexpr,
    L3X: tl.constexpr,
    L2Y: tl.constexpr,
    L2X: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    chunk_size: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    TRACE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=chunk_size)
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

    for tile_id in range(pid, total_tiles, NUM_SMS):
        transformed = llc_and_l2_aware_transform(
            tile_id, num_pid_m, num_pid_n,
            ordering0, ordering1, ordering2,
            L3Y, L3X, L2Y, L2X,
        )
        pid_m = transformed // num_pid_n
        pid_n = transformed % num_pid_n
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        if TRACE:
            flat_tile_id = pid_m * num_pid_n + pid_n
            tl.store(trace_start_ptr + flat_tile_id, _read_realtime())
            tl.store(trace_pid_ptr + flat_tile_id, tl.program_id(0))
            tl.store(trace_xcd_ptr + flat_tile_id, _get_xcc_id())

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
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
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
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)

        if TRACE:
            tl.store(trace_end_ptr + flat_tile_id, _read_realtime())


@triton.jit()
def persistent_matmul_debug_map_llc_and_l2_aware(
    workgroup_map,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ordering0: tl.constexpr,
    ordering1: tl.constexpr,
    ordering2: tl.constexpr,
    L3Y: tl.constexpr,
    L3X: tl.constexpr,
    L2Y: tl.constexpr,
    L2X: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    chunk_size: tl.constexpr,
):
    pid = tl.program_id(0)
    original_pid = pid
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(pid, NUM_SMS, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=chunk_size)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        transformed = llc_and_l2_aware_transform(
            tile_id, num_pid_m, num_pid_n,
            ordering0, ordering1, ordering2,
            L3Y, L3X, L2Y, L2X,
        )
        tl.store(workgroup_map + transformed, original_pid)

import triton
import triton.language as tl

from .pid_transforms import chiplet_transform_chunked


@triton.jit()
def grouped_persistent_matmul(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    gemm_offsets,
    g_lds,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MATMUL_DTYPE: tl.constexpr,
):
    """Persistent grouped GEMM kernel for heterogeneous groups.

    Flat iteration over all output tiles across all groups. Each tile
    finds its group via a scan of gemm_offsets. GROUP_COUNT is constexpr
    for compiler unrolling. Assumes row-major contiguous inputs.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    total_tiles = tl.load(gemm_offsets + GROUP_COUNT)

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Find group (GROUP_COUNT is constexpr → compiler can unroll)
        g = 0
        for g_idx in range(GROUP_COUNT):
            if tile_id >= tl.load(gemm_offsets + g_idx + 1):
                g = g_idx + 1

        g_start = tl.load(gemm_offsets + g)
        tile_in_group = tile_id - g_start

        M = tl.load(group_gemm_sizes + g * 3)
        N = tl.load(group_gemm_sizes + g * 3 + 1)
        K = tl.load(group_gemm_sizes + g * 3 + 2)

        A = tl.load(group_a_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))
        B = tl.load(group_b_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))
        C = tl.load(group_c_ptrs + g).to(tl.pointer_type(MATMUL_DTYPE))

        stride_am = tl.load(g_lds + g * 6)
        stride_bk = tl.load(g_lds + g * 6 + 2)
        stride_cm = tl.load(g_lds + g * 6 + 4)

        tl.assume(stride_am > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        # L2-friendly tile ordering
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_in_group // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_in_group % num_pid_in_group) % group_size_m)
        pid_n = (tile_in_group % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Modulo wrapping (no masking on loads)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :]
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :]

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Main K-loop: unmasked
        loop_k = K // BLOCK_SIZE_K
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K
            B_BASE += BLOCK_SIZE_K * stride_bk

        # Remainder K-block
        remainder_k = K % BLOCK_SIZE_K
        if remainder_k != 0:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + rm[:, None] * stride_am + rk_last[None, :]
            B_LAST = B + rk_last[:, None] * stride_bk + rn[None, :]
            k_mask = rk_last < K
            a = tl.load(tl.multiple_of(A_LAST, (1, 16)), mask=k_mask[None, :], other=0.0)
            b = tl.load(tl.multiple_of(B_LAST, (16, 1)), mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        # Store with boundary masking
        rm_store = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_store = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (rm_store[:, None] < M) & (rn_store[None, :] < N)
        rm_store = tl.max_contiguous(tl.multiple_of(rm_store % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_store = tl.max_contiguous(tl.multiple_of(rn_store % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        C_ = C + rm_store[:, None] * stride_cm + rn_store[None, :]
        tl.store(C_, c, mask=c_mask)

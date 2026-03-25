#!/usr/bin/env python3
"""
Autoresearch Phase 7: K-split Work-Stealing.

Each output tile is processed by K_SPLIT work-groups, each handling a
portion of the K dimension. Partial results are atomically accumulated
to the output tile.

Goal: reduce per-WG K-loop iterations → less overhead from barriers.
Tradeoff: atomic accumulation adds its own overhead.
"""
import json, os, statistics, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 50; N_ROT = 4


@triton.jit()
def ws_ksplit_matmul(
    A, B, C,
    locks,              # int32[total_tiles]: per-tile spin locks
    tile_counter,       # int32[NUM_XCDS * COUNTER_STRIDE]
    global_counter,     # int32[COUNTER_STRIDE]
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn,
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
    EVEN_K: tl.constexpr,
    K_SPLIT: tl.constexpr,
    mask_ptr=None,
):
    """K-split WS: each tile is split across K_SPLIT WGs along K dimension."""
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS

    mask = tl.load(mask_ptr + pid)
    if mask == 0:
        return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_local = LOCAL_TILES_PER_XCD * NUM_XCDS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    total_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    k_per_split = tl.cdiv(total_k_iters, K_SPLIT)

    # ================================================================
    # Level 1: Per-XCD stealing (L2-local tiles)
    # ================================================================
    xcd_base = xcd_id * LOCAL_TILES_PER_XCD
    local_counter = tile_counter + xcd_id * COUNTER_STRIDE

    raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")
    while raw_idx < LOCAL_TILES_PER_XCD * K_SPLIT:
        tile_id = xcd_base + (raw_idx // K_SPLIT)
        k_split_id = raw_idx % K_SPLIT

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

        k_start = k_split_id * k_per_split
        k_end = min(k_start + k_per_split, total_k_iters)

        rk = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(k_start, k_end):
            k_offset = k * BLOCK_SIZE_K
            rk_cur = k_offset + tl.arange(0, BLOCK_SIZE_K)
            A_ptr = A + rm[:, None] * stride_am + rk_cur[None, :] * stride_ak
            B_ptr = B + rk_cur[:, None] * stride_bk + rn[None, :] * stride_bn

            if k < total_k_iters - 1 or EVEN_K:
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_ptr, (1, 16)))
                else:
                    a = tl.load(tl.multiple_of(A_ptr, (16, 1)))
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_ptr, (16, 1)))
                else:
                    b = tl.load(tl.multiple_of(B_ptr, (1, 16)))
            else:
                if stride_ak == 1:
                    a = tl.load(A_ptr, mask=rk_cur[None, :] < K, other=0.0)
                else:
                    a = tl.load(A_ptr, mask=rk_cur[None, :] < K, other=0.0)
                if stride_bk == 1:
                    b = tl.load(B_ptr, mask=rk_cur[:, None] < K, other=0.0)
                else:
                    b = tl.load(B_ptr, mask=rk_cur[:, None] < K, other=0.0)

            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        next_raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")

        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        if K_SPLIT == 1:
            tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
        else:
            lock_ptr = locks + (raw_idx // K_SPLIT)
            while tl.atomic_cas(lock_ptr, 0, 1) == 1:
                pass
            tl.debug_barrier()
            if k_split_id == 0:
                tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
            else:
                existing = tl.load(C_, mask=(rm[:, None] < M) & (rn[None, :] < N))
                tl.store(C_, existing + c, mask=(rm[:, None] < M) & (rn[None, :] < N))
            tl.debug_barrier()
            tl.atomic_xchg(lock_ptr, 0)

        raw_idx = next_raw_idx

    # ================================================================
    # Level 2: Global fallback
    # ================================================================
    if GLOBAL_TILES == 0:
        return

    raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")
    while raw_idx < GLOBAL_TILES * K_SPLIT:
        tile_id = total_local + (raw_idx // K_SPLIT)
        k_split_id = raw_idx % K_SPLIT

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

        k_start = k_split_id * k_per_split
        k_end = min(k_start + k_per_split, total_k_iters)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(k_start, k_end):
            k_offset = k * BLOCK_SIZE_K
            rk_cur = k_offset + tl.arange(0, BLOCK_SIZE_K)
            A_ptr = A + rm[:, None] * stride_am + rk_cur[None, :] * stride_ak
            B_ptr = B + rk_cur[:, None] * stride_bk + rn[None, :] * stride_bn

            if k < total_k_iters - 1 or EVEN_K:
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_ptr, (1, 16)))
                else:
                    a = tl.load(tl.multiple_of(A_ptr, (16, 1)))
                if stride_bk == 1:
                    b = tl.load(tl.multiple_of(B_ptr, (16, 1)))
                else:
                    b = tl.load(tl.multiple_of(B_ptr, (1, 16)))
            else:
                a = tl.load(A_ptr, mask=rk_cur[None, :] < K, other=0.0)
                b = tl.load(B_ptr, mask=rk_cur[:, None] < K, other=0.0)

            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        next_raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")

        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        if K_SPLIT == 1:
            tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
        else:
            lock_ptr = locks + total_local + (raw_idx // K_SPLIT)
            while tl.atomic_cas(lock_ptr, 0, 1) == 1:
                pass
            tl.debug_barrier()
            if k_split_id == 0:
                tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
            else:
                existing = tl.load(C_, mask=(rm[:, None] < M) & (rn[None, :] < N))
                tl.store(C_, existing + c, mask=(rm[:, None] < M) & (rn[None, :] < N))
            tl.debug_barrier()
            tl.atomic_xchg(lock_ptr, 0)

        raw_idx = next_raw_idx


def bench_torch(sz):
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    for w in range(WARMUP): torch.matmul(As[w % N_ROT], Bs[w % N_ROT])
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record(); torch.matmul(As[i % N_ROT], Bs[i % N_ROT]); es[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(ss, es)]
    del As, Bs; torch.cuda.empty_cache()
    return statistics.median(times)


def bench_ksplit(sz, k_split):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gm = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0
        lp, gt = sel.hierarchical_split(num_xcds)

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        locks = torch.zeros(total_tiles, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset():
            tc.zero_(); gc.zero_(); locks.zero_()
            for c in Cs: c.zero_()

        def run(idx):
            reset()
            ws_ksplit_matmul[(n_cu,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                locks, tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1),
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
                COUNTER_STRIDE=COUNTER_STRIDE,
                EVEN_K=even_k, K_SPLIT=k_split,
                num_stages=2, num_warps=8, waves_per_eu=0,
                matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
            )

        for w in range(WARMUP): run(w)
        torch.cuda.synchronize()

        ref = torch.matmul(As[0], Bs[0])
        Cs[0].zero_(); reset(); run(0); torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            Cs[0].float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
        if cos < 0.999:
            del As, Bs, Cs; torch.cuda.empty_cache()
            return None, f"cos={cos:.6f}"

        ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            reset(); ss[i].record(); run(i); es[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(ss, es)]
        del As, Bs, Cs; torch.cuda.empty_cache()
        return statistics.median(times), None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, str(e)[:100]


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)
    print("=" * 80)
    print("  Phase 7: K-split WS — multiple WGs per tile")
    print("=" * 80)

    for sz in [12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")

        total_k_iters = triton.cdiv(sz, 64)
        print(f"  K-iters={total_k_iters}")
        print(f"  {'Config':<25s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}  {'note'}")
        print("  " + "-" * 65)

        for ks in [1, 2, 3, 4]:
            k_per = triton.cdiv(total_k_iters, ks)
            label = f"K_SPLIT={ks} ({k_per} iters/wg)"
            ms, err = bench_ksplit(sz, ks)
            if ms is None:
                print(f"  {label:<25s}  {'FAIL':>8s}  {err or ''}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<25s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")

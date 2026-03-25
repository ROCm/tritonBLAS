#!/usr/bin/env python3
"""
Autoresearch Phase 6: K-split within WS Hierarchical.

Split the K dimension across 2 work-groups per tile to reduce per-WG
K-loop iterations by 50%. Each WG computes a partial result and atomically
accumulates to the output tile.

Also explores: merged-phase kernel (single while-loop for both local/global).
"""
import json, os, statistics, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul
from tritonblas.kernels.stages.indexing.pid_transforms import chiplet_transform

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 40; N_ROT = 4


@triton.jit()
def ws_hierarchical_merged(
    A, B, C,
    tile_counter,
    global_counter,
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
    TOTAL_LOCAL: tl.constexpr,
    GLOBAL_TILES: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    EVEN_K: tl.constexpr,
    mask_ptr=None,
):
    """Merged-phase kernel: single K-loop body for both local and global tiles.

    Eliminates code duplication by using a unified tile-stealing loop.
    """
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS

    mask = tl.load(mask_ptr + pid)
    if mask == 0:
        return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    rk = tl.arange(0, BLOCK_SIZE_K)
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    has_remainder = not EVEN_K
    if has_remainder:
        loop_k -= 1
    tl.assume(loop_k > 1)

    xcd_base = xcd_id * LOCAL_TILES_PER_XCD
    local_counter = tile_counter + xcd_id * COUNTER_STRIDE

    # Phase: 0 = local, 1 = global
    phase: tl.constexpr = 0
    raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")
    in_local = raw_idx < LOCAL_TILES_PER_XCD

    while True:
        if in_local:
            tile_id = xcd_base + raw_idx
        else:
            tile_id = TOTAL_LOCAL + raw_idx

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

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))
            acc += tl.dot(a, b)
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
            a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0)
            b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        # Pre-fetch next tile index while store is in-flight
        if in_local:
            next_raw_idx = tl.atomic_add(local_counter, 1, scope="gpu")
        else:
            next_raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")

        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

        # Transition: local → global → done
        if in_local:
            if next_raw_idx < LOCAL_TILES_PER_XCD:
                raw_idx = next_raw_idx
            else:
                if GLOBAL_TILES == 0:
                    return
                raw_idx = tl.atomic_add(global_counter, 1, scope="gpu")
                if raw_idx >= GLOBAL_TILES:
                    return
                in_local = False
        else:
            if next_raw_idx >= GLOBAL_TILES:
                return
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


def bench_merged(sz):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gm = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0
        lp, gt = sel.hierarchical_split(num_xcds)
        total_local = lp * num_xcds

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset(): tc.zero_(); gc.zero_()
        def run(idx):
            reset()
            ws_hierarchical_merged[(n_cu,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1),
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=lp, TOTAL_LOCAL=total_local,
                GLOBAL_TILES=gt,
                COUNTER_STRIDE=COUNTER_STRIDE,
                EVEN_K=even_k,
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
        return None, str(e)[:120]


def bench_baseline(sz):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gm = sel.group_m
        even_k = sz % BK == 0
        lp, gt = sel.hierarchical_split(num_xcds)

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset(): tc.zero_(); gc.zero_()
        def run(idx):
            reset()
            ws_hierarchical_matmul[(n_cu,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                None, None, None, tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1), 0,
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
                COUNTER_STRIDE=COUNTER_STRIDE,
                BIAS=False, EVEN_K=even_k,
                CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
                QUANTIZED=False,
                num_stages=2, num_warps=8, waves_per_eu=0,
                matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
            )

        for w in range(WARMUP): run(w)
        torch.cuda.synchronize()

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
        return None, str(e)[:120]


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)
    print("=" * 80)
    print("  Phase 6: Merged-phase kernel vs baseline")
    print("=" * 80)

    for sz in [8192, 12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch:    {torch_ms:.3f} ms ({torch_tf:.1f} TF)")

        base_ms, _ = bench_baseline(sz)
        if base_ms:
            base_tf = flops / (base_ms * 1e-3) / 1e12
            base_vs = (base_ms - torch_ms) / torch_ms * 100
            print(f"  baseline: {base_ms:.3f} ms ({base_tf:.1f} TF) [{base_vs:+.1f}%]")

        merged_ms, err = bench_merged(sz)
        if merged_ms:
            merged_tf = flops / (merged_ms * 1e-3) / 1e12
            merged_vs = (merged_ms - torch_ms) / torch_ms * 100
            print(f"  merged:   {merged_ms:.3f} ms ({merged_tf:.1f} TF) [{merged_vs:+.1f}%]")
            if base_ms:
                imp = (base_ms - merged_ms) / base_ms * 100
                print(f"  merged vs baseline: {imp:+.1f}%")
        else:
            print(f"  merged:   FAIL ({err})")

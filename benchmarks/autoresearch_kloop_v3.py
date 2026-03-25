#!/usr/bin/env python3
"""
Autoresearch Phase 3: Per-tile overhead + scheduling tricks.

Variant A: Baseline (num_stages=2, standard)
Variant B: Best params from Phase 1 (wpe=2 for 4K, gm=8 for 8K/12K, etc.)
Variant C: Batch-steal (steal BATCH tiles at a time, reduce atomic overhead)
Variant D: Static + WS hybrid (precompute bulk assignment, WS only for remainder)
"""
import json, os, statistics, sys, time
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 10; ITERS = 30; N_ROT = 4


# ===========================================================================
# Variant C: Batch-steal hierarchical
# ===========================================================================
@triton.jit()
def ws_batch_steal(
    A, B, C, A_scale_ptr, B_scale_ptr, bias_ptr,
    tile_counter, global_counter,
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn, stride_bias,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
    LOCAL_TILES_PER_XCD: tl.constexpr, GLOBAL_TILES: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    BIAS: tl.constexpr, EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr, CACHE_MODIFIER_B: tl.constexpr,
    BATCH_SIZE: tl.constexpr = 4,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    USE_MASK: tl.constexpr = True, mask_ptr=None,
):
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS
    if USE_MASK:
        mask_val = tl.load(mask_ptr + pid)
        if mask_val == 0:
            return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_local = LOCAL_TILES_PER_XCD * NUM_XCDS
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0); tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    has_remainder = not EVEN_K
    if has_remainder:
        loop_k -= 1
    tl.assume(loop_k > 1)

    # Level 1: Per-XCD with batch stealing
    xcd_base = xcd_id * LOCAL_TILES_PER_XCD
    local_counter = tile_counter + xcd_id * COUNTER_STRIDE

    batch_start = tl.atomic_add(local_counter, BATCH_SIZE, scope="gpu")

    while batch_start < LOCAL_TILES_PER_XCD:
        batch_end = min(batch_start + BATCH_SIZE, LOCAL_TILES_PER_XCD)

        local_idx = batch_start
        while local_idx < batch_end:
            tile_id = xcd_base + local_idx

            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0); tl.assume(pid_n >= 0)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

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
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if has_remainder:
                k2 = loop_k
                rk_rem = k2 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                if stride_ak == 1: A_REM = tl.multiple_of(A_REM, (1, 16))
                else: A_REM = tl.multiple_of(A_REM, (16, 1))
                if stride_bk == 1: B_REM = tl.multiple_of(B_REM, (16, 1))
                else: B_REM = tl.multiple_of(B_REM, (1, 16))
                a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            c = acc.to(C.type.element_ty)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

            local_idx += 1

        batch_start = tl.atomic_add(local_counter, BATCH_SIZE, scope="gpu")

    # Level 2: global with batch stealing
    if GLOBAL_TILES == 0:
        return
    batch_start = tl.atomic_add(global_counter, BATCH_SIZE, scope="gpu")
    while batch_start < GLOBAL_TILES:
        batch_end = min(batch_start + BATCH_SIZE, GLOBAL_TILES)
        local_idx = batch_start
        while local_idx < batch_end:
            tile_id = total_local + local_idx

            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0); tl.assume(pid_n >= 0)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
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
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk
            if has_remainder:
                k2 = loop_k
                rk_rem = k2 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_REM = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
                B_REM = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
                if stride_ak == 1: A_REM = tl.multiple_of(A_REM, (1, 16))
                else: A_REM = tl.multiple_of(A_REM, (16, 1))
                if stride_bk == 1: B_REM = tl.multiple_of(B_REM, (16, 1))
                else: B_REM = tl.multiple_of(B_REM, (1, 16))
                a = tl.load(A_REM, mask=rk_rem[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_REM, mask=rk_rem[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            c = acc.to(C.type.element_ty)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
            local_idx += 1
        batch_start = tl.atomic_add(global_counter, BATCH_SIZE, scope="gpu")


# ===========================================================================
# Benchmarking
# ===========================================================================
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


def bench_kernel(sz, kernel_fn, extra_kwargs=None):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gsize_m = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0
        local_per_xcd = (total_tiles * 9) // (num_xcds * 10)
        local_per_xcd = max(local_per_xcd, 1)
        global_tiles = total_tiles - local_per_xcd * num_xcds

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        kw = dict(
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
            QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
        )
        if extra_kwargs:
            kw.update(extra_kwargs)

        def reset(): tc.zero_(); gc.zero_()
        def run(idx):
            reset()
            kernel_fn[(n_cu,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                None, None, None, tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1), 0,
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                **kw,
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
        return None, str(e)[:80]


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)

    # Per-size best params from Phase 1
    best_params = {
        4096:  dict(waves_per_eu=2),
        8192:  dict(GROUP_SIZE_M=8),
        12288: dict(),  # default is best
        16384: dict(),
    }

    all_results = {}
    for sz in [4096, 8192, 12288, 16384]:
        print(f"\n{'='*80}")
        print(f"  Phase 3 variants for {sz}x{sz}x{sz} BF16")
        print(f"{'='*80}")

        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch.matmul: {torch_ms:.3f} ms ({torch_tf:.1f} TF)")
        print()
        print(f"  {'Variant':<25s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}  {'note':>15s}")
        print("  " + "-" * 72)

        size_results = {"torch_ms": torch_ms}
        configs = [
            ("Baseline",        ws_hierarchical_matmul, {}),
            ("Tuned",           ws_hierarchical_matmul, best_params.get(sz, {})),
            ("Batch=2",         ws_batch_steal, dict(BATCH_SIZE=2)),
            ("Batch=4",         ws_batch_steal, dict(BATCH_SIZE=4)),
            ("Batch=8",         ws_batch_steal, dict(BATCH_SIZE=8)),
            ("Tuned+Batch=4",   ws_batch_steal, {**best_params.get(sz, {}), "BATCH_SIZE": 4}),
        ]

        for name, kfn, extra in configs:
            ws_ms, err = bench_kernel(sz, kfn, extra)
            if ws_ms is None:
                print(f"  {name:<25s}  {'FAIL':>8s}  {'':>7s}  {'':>9s}  {err or '':>15s}")
            else:
                ws_tf = flops / (ws_ms * 1e-3) / 1e12
                vs = (ws_ms - torch_ms) / torch_ms * 100
                print(f"  {name:<25s}  {ws_ms:>8.3f}  {ws_tf:>7.1f}  {vs:>+8.1f}%")
                size_results[name] = {"ms": ws_ms, "tflops": ws_tf, "vs_pct": vs}

        all_results[str(sz)] = size_results

    with open("results/autoresearch/phase3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("  SUMMARY (best per size)")
    print(f"{'='*80}")
    for k, r in all_results.items():
        best_name, best_pct = None, float('inf')
        for name, data in r.items():
            if name == "torch_ms" or not isinstance(data, dict):
                continue
            if data["vs_pct"] < best_pct:
                best_pct = data["vs_pct"]
                best_name = name
        print(f"  {k}: torch={r['torch_ms']:.3f}ms  best={best_name} ({best_pct:+.1f}%)")
    print(f"\n  Saved to results/autoresearch/phase3_results.json")

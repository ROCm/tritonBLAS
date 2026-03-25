#!/usr/bin/env python3
"""
Autoresearch Phase 6b: GROUP_SIZE_M sweep + tile ordering analysis.

Tests whether larger GROUP_SIZE_M values improve L2 hit rate at 12K.
Also tests whether tile ordering within XCD matters.
"""
import json, os, statistics, sys
import torch, triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 50; N_ROT = 4


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


def bench_ws(sz, gm, split_frac=None):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0

        if split_frac is not None:
            lp = int(total_tiles * split_frac) // num_xcds
            lp = max(lp, 1)
            gt = total_tiles - lp * num_xcds
        else:
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

    # Part 1: GROUP_SIZE_M sweep
    print("=" * 80)
    print("  Part 1: GROUP_SIZE_M sweep")
    print("=" * 80)

    for sz in [8192, 12288, 16384]:
        tiles_m = triton.cdiv(sz, 256)
        print(f"\n  --- {sz}x{sz}x{sz} BF16 (tiles_m={tiles_m}) ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        print(f"  torch: {torch_ms:.3f} ms\n")
        print(f"  {'Config':<28s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 60)

        for gm in [1, 2, 4, 8, 12, 16, 24, 32, tiles_m]:
            if gm > tiles_m:
                continue
            label = f"gm={gm}"
            ms, err = bench_ws(sz, gm=gm)
            if ms is None:
                print(f"  {label:<28s}  {'FAIL':>8s}  {err or ''}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                marker = " <--" if gm == 8 else ""
                print(f"  {label:<28s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%{marker}")

    # Part 2: Combined gm × split at 12K
    print(f"\n{'='*80}")
    print("  Part 2: gm × split at 12K (best combinations)")
    print(f"{'='*80}")
    sz = 12288
    torch_ms = bench_torch(sz)
    flops = 2.0 * sz ** 3
    print(f"\n  torch: {torch_ms:.3f} ms\n")
    print(f"  {'Config':<28s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
    print("  " + "-" * 60)

    combos = [
        (8, 0.8, "gm=8, 80/20"),
        (8, 0.7, "gm=8, 70/30"),
        (8, 0.6, "gm=8, 60/40"),
        (12, 0.8, "gm=12, 80/20"),
        (12, 0.7, "gm=12, 70/30"),
        (16, 0.8, "gm=16, 80/20"),
        (16, 0.7, "gm=16, 70/30"),
        (24, 0.8, "gm=24, 80/20"),
        (32, 0.8, "gm=32, 80/20"),
    ]
    for gm, frac, label in combos:
        ms, err = bench_ws(sz, gm=gm, split_frac=frac)
        if ms is None:
            print(f"  {label:<28s}  {'FAIL':>8s}  {err or ''}")
        else:
            tf = flops / (ms * 1e-3) / 1e12
            vs = (ms - torch_ms) / torch_ms * 100
            print(f"  {label:<28s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")

    # Part 3: Combined gm × split at 16K
    print(f"\n{'='*80}")
    print("  Part 3: gm × split at 16K (best combinations)")
    print(f"{'='*80}")
    sz = 16384
    torch_ms = bench_torch(sz)
    flops = 2.0 * sz ** 3
    print(f"\n  torch: {torch_ms:.3f} ms\n")
    print(f"  {'Config':<28s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
    print("  " + "-" * 60)

    combos_16k = [
        (8, 0.5, "gm=8, 50/50"),
        (12, 0.5, "gm=12, 50/50"),
        (16, 0.5, "gm=16, 50/50"),
        (24, 0.5, "gm=24, 50/50"),
        (32, 0.5, "gm=32, 50/50"),
        (12, 0.6, "gm=12, 60/40"),
        (16, 0.4, "gm=16, 40/60"),
    ]
    for gm, frac, label in combos_16k:
        ms, err = bench_ws(sz, gm=gm, split_frac=frac)
        if ms is None:
            print(f"  {label:<28s}  {'FAIL':>8s}  {err or ''}")
        else:
            tf = flops / (ms * 1e-3) / 1e12
            vs = (ms - torch_ms) / torch_ms * 100
            print(f"  {label:<28s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")

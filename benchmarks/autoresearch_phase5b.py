#!/usr/bin/env python3
"""
Autoresearch Phase 5b: Fine-grained optimization for 12K/16K.

Explores:
1. Fine-grained split ratios (60/40 through 100/0)
2. wpe × split combinations
3. BLOCK_K variants (32, 64, 128)
4. Grid size (n_cu vs total_tiles)
"""
import json, os, statistics, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 40; N_ROT = 4


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


def bench_ws(sz, local_frac=0.9, bk=64, wpe=0, grid=None):
    try:
        BM, BN, BK = 256, 256, bk
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gsize_m = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0

        if local_frac >= 1.0:
            local_per_xcd = total_tiles // num_xcds
            global_tiles = total_tiles - local_per_xcd * num_xcds
        else:
            local_per_xcd = int(total_tiles * local_frac) // num_xcds
            local_per_xcd = max(local_per_xcd, 1)
            global_tiles = total_tiles - local_per_xcd * num_xcds

        grid_size = grid if grid else n_cu

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset(): tc.zero_(); gc.zero_()
        def run(idx):
            reset()
            ws_hierarchical_matmul[(grid_size,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                None, None, None, tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1), 0,
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
                COUNTER_STRIDE=COUNTER_STRIDE,
                BIAS=False, EVEN_K=even_k,
                CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
                QUANTIZED=False,
                num_stages=2, num_warps=8, waves_per_eu=wpe,
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

    # Part 1: Split ratio sweep (12K and 16K)
    print("=" * 80)
    print("  Part 1: Split Ratio Sweep")
    print("=" * 80)

    all_results = {}
    for sz in [12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")
        print(f"  {'Config':<38s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 70)

        size_results = {"torch_ms": torch_ms}
        for frac in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.0]:
            label = f"split={int(frac*100)}/{int((1-frac)*100)}"
            ms, err = bench_ws(sz, local_frac=frac)
            if ms is None:
                print(f"  {label:<38s}  {'FAIL':>8s}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                marker = " <-- best" if frac in [0.8, 1.0] else ""
                print(f"  {label:<38s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%{marker}")
                size_results[label] = {"ms": ms, "tflops": tf, "vs_pct": vs}

        all_results[f"split_{sz}"] = size_results

    # Part 2: wpe × split combinations
    print(f"\n{'='*80}")
    print("  Part 2: wpe × split Combinations")
    print(f"{'='*80}")

    for sz in [12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = all_results.get(f"split_{sz}", {}).get("torch_ms")
        if torch_ms is None:
            torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        print(f"  torch: {torch_ms:.3f} ms\n")
        print(f"  {'Config':<38s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 70)

        combos = [
            ("wpe=1, split=80/20", 0.80, 1),
            ("wpe=1, split=100/0", 1.00, 1),
            ("wpe=1, split=75/25", 0.75, 1),
            ("wpe=2, split=80/20", 0.80, 2),
            ("wpe=2, split=75/25", 0.75, 2),
        ]
        for label, frac, wpe in combos:
            ms, err = bench_ws(sz, local_frac=frac, wpe=wpe)
            if ms is None:
                print(f"  {label:<38s}  {'FAIL':>8s}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<38s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")
                all_results.setdefault(f"combo_{sz}", {})[label] = {"ms": ms, "tflops": tf, "vs_pct": vs}

    # Part 3: BLOCK_K variants
    print(f"\n{'='*80}")
    print("  Part 3: BLOCK_K Variants (with optimal split)")
    print(f"{'='*80}")

    for sz in [12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = all_results.get(f"split_{sz}", {}).get("torch_ms")
        if torch_ms is None: torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        print(f"  torch: {torch_ms:.3f} ms\n")
        print(f"  {'Config':<38s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 70)

        for bk_label, bk in [("BK=32", 32), ("BK=64 (baseline)", 64), ("BK=128", 128)]:
            frac = 0.80  # use best split
            label = f"{bk_label}, split=80/20"
            ms, err = bench_ws(sz, local_frac=frac, bk=bk)
            if ms is None:
                print(f"  {label:<38s}  {'FAIL':>8s}  {err or ''}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<38s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")
                all_results.setdefault(f"bk_{sz}", {})[label] = {"ms": ms, "tflops": tf, "vs_pct": vs}

    # Part 4: Grid size (n_cu vs total_tiles)
    print(f"\n{'='*80}")
    print("  Part 4: Grid Size")
    print(f"{'='*80}")

    for sz in [8192, 12288]:
        BM = 256
        total_tiles = triton.cdiv(sz, BM) ** 2
        print(f"\n  --- {sz}x{sz} (tiles={total_tiles}) ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        print(f"  torch: {torch_ms:.3f} ms\n")
        print(f"  {'Config':<38s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 70)

        for label, grid, frac in [
            (f"grid=304, split=80/20", None, 0.80),
            (f"grid=304, split=100/0", None, 1.00),
            (f"grid={total_tiles}, split=100/0", total_tiles, 1.00),
            (f"grid={total_tiles}, split=80/20", total_tiles, 0.80),
        ]:
            ms, err = bench_ws(sz, local_frac=frac, grid=grid)
            if ms is None:
                print(f"  {label:<38s}  {'FAIL':>8s}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<38s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")

    with open("results/autoresearch/phase5b_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to results/autoresearch/phase5b_results.json")

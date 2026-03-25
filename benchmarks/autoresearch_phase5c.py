#!/usr/bin/env python3
"""
Autoresearch Phase 5c: Pipeline depth + accumulation precision.

1. num_stages sweep (1, 2, 3, 4)
2. num_warps sweep with best split
3. max_num_imprecise_acc (controls dot precision)
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


def bench_ws(sz, stages=2, warps=8, wpe=0, split_frac=None):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gsize_m = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0

        if split_frac is not None:
            local_per_xcd = int(total_tiles * split_frac) // num_xcds
            local_per_xcd = max(local_per_xcd, 1)
            global_tiles = total_tiles - local_per_xcd * num_xcds
        else:
            local_per_xcd, global_tiles = sel.hierarchical_split(num_xcds)

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
                GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
                COUNTER_STRIDE=COUNTER_STRIDE,
                BIAS=False, EVEN_K=even_k,
                CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
                QUANTIZED=False,
                num_stages=stages, num_warps=warps, waves_per_eu=wpe,
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

    # Part 1: num_stages sweep
    print("=" * 80)
    print("  Part 1: num_stages Sweep (with adaptive split)")
    print("=" * 80)

    all_results = {}
    for sz in [8192, 12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")
        print(f"  {'Config':<35s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 68)

        size_results = {"torch_ms": torch_ms}
        configs = [
            ("num_stages=1", 1, 8, 0),
            ("num_stages=2 (baseline)", 2, 8, 0),
            ("num_stages=3", 3, 8, 0),
            ("num_stages=4", 4, 8, 0),
            ("stages=2, warps=4", 2, 4, 0),
            ("stages=2, warps=16", 2, 16, 0),
            ("stages=3, warps=4", 3, 4, 0),
            ("stages=2, warps=8, wpe=1", 2, 8, 1),
            ("stages=3, warps=8, wpe=1", 3, 8, 1),
        ]

        for label, stages, warps, wpe in configs:
            ms, err = bench_ws(sz, stages=stages, warps=warps, wpe=wpe)
            if ms is None:
                print(f"  {label:<35s}  {'FAIL':>8s}  {err or ''}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<35s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")
                size_results[label] = {"ms": ms, "tflops": tf, "vs_pct": vs}

        all_results[str(sz)] = size_results

    with open("results/autoresearch/phase5c_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("  Best per size:")
    print(f"{'='*80}")
    for k, r in all_results.items():
        best_name, best_pct = None, float('inf')
        for name, data in r.items():
            if name == "torch_ms" or not isinstance(data, dict): continue
            if data["vs_pct"] < best_pct:
                best_pct = data["vs_pct"]; best_name = name
        if best_name:
            print(f"  {k}: best={best_name} ({best_pct:+.1f}%)")

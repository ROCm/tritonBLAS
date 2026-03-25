#!/usr/bin/env python3
"""
Autoresearch Phase 4: MFMA instruction variants + combined best config.

Tests matrix_instr_nonkdim=32 and kpack=2 (previously blocked by a bug).
Also tests the best combined configuration from all phases.
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
WARMUP = 10; ITERS = 30; N_ROT = 4


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


def bench_ws(sz, extra_kwargs=None):
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
            ws_hierarchical_matmul[(n_cu,)](
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
        msg = str(e)
        if "shared memory" in msg: return None, "OOM-LDS"
        if "resource" in msg: return None, msg[:40]
        return None, msg[:80]


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)

    all_results = {}
    for sz in [4096, 8192, 12288, 16384]:
        print(f"\n{'='*80}")
        print(f"  Phase 4 MFMA variants for {sz}x{sz}x{sz} BF16")
        print(f"{'='*80}")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch.matmul: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")
        print(f"  {'Variant':<30s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}  {'note':>15s}")
        print("  " + "-" * 78)

        configs = [
            ("Default (mfma16, kp1, w8)",  {}),
            ("mfma32, kp1, w8",            dict(matrix_instr_nonkdim=32)),
            ("mfma16, kp2, w8",            dict(kpack=2)),
            ("mfma32, kp2, w8",            dict(matrix_instr_nonkdim=32, kpack=2)),
            ("mfma32, kp1, w4",            dict(matrix_instr_nonkdim=32, num_warps=4)),
            ("mfma16, kp1, w8, wpe2",      dict(waves_per_eu=2)),
            ("mfma32, kp1, w8, wpe2",      dict(matrix_instr_nonkdim=32, waves_per_eu=2)),
            ("mfma16, kp2, w8, wpe2",      dict(kpack=2, waves_per_eu=2)),
            ("mfma32, kp2, w8, wpe1",      dict(matrix_instr_nonkdim=32, kpack=2, waves_per_eu=1)),
            ("mfma16, kp1, w8, gm8",       dict(GROUP_SIZE_M=8)),
            ("mfma32, kp1, w8, gm8",       dict(matrix_instr_nonkdim=32, GROUP_SIZE_M=8)),
        ]

        size_results = {"torch_ms": torch_ms}
        for name, extra in configs:
            ws_ms, err = bench_ws(sz, extra)
            if ws_ms is None:
                print(f"  {name:<30s}  {'FAIL':>8s}  {'':>7s}  {'':>9s}  {err or '':>15s}")
            else:
                ws_tf = flops / (ws_ms * 1e-3) / 1e12
                vs = (ws_ms - torch_ms) / torch_ms * 100
                print(f"  {name:<30s}  {ws_ms:>8.3f}  {ws_tf:>7.1f}  {vs:>+8.1f}%")
                size_results[name] = {"ms": ws_ms, "tflops": ws_tf, "vs_pct": vs}
        all_results[str(sz)] = size_results

    with open("results/autoresearch/phase4_mfma.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    for k, r in all_results.items():
        best_name, best_pct = None, float('inf')
        for name, data in r.items():
            if name == "torch_ms" or not isinstance(data, dict): continue
            if data["vs_pct"] < best_pct:
                best_pct = data["vs_pct"]; best_name = name
        if best_name:
            print(f"  {k}: torch={r['torch_ms']:.3f}ms  best={best_name} ({best_pct:+.1f}%)")
    print(f"\n  Saved to results/autoresearch/phase4_mfma.json")

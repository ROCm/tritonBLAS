#!/usr/bin/env python3
"""
Autoresearch: sweep kernel parameters to close the per-K-iteration gap
between WS Hierarchical and torch.matmul (hipBLASLt).

Sweeps tile shapes, num_warps, waves_per_eu, GROUP_SIZE_M.
Uses proven invocation with USE_MASK=True, all-ones mask.
"""
import itertools, json, os, statistics, sys, time, traceback
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 10
ITERS = 30
N_ROT = 4


def bench_torch(sz):
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    for w in range(WARMUP):
        torch.matmul(As[w % N_ROT], Bs[w % N_ROT])
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record()
        torch.matmul(As[i % N_ROT], Bs[i % N_ROT])
        es[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(ss, es)]
    del As, Bs; torch.cuda.empty_cache()
    return statistics.median(times)


def bench_ws(sz, BM, BN, BK, gsize_m, num_warps, waves_per_eu, num_stages=2):
    """Measure WS Hierarchical — proven invocation (USE_MASK=True, all-ones mask)."""
    try:
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU
        num_xcds = sel.num_sms
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0

        local_per_xcd = (total_tiles * 9) // (num_xcds * 10)
        local_per_xcd = max(local_per_xcd, 1)
        total_local = local_per_xcd * num_xcds
        global_tiles = total_tiles - total_local

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset():
            tc.zero_(); gc.zero_()

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
                num_stages=num_stages, num_warps=num_warps, waves_per_eu=waves_per_eu,
                matrix_instr_nonkdim=16, kpack=1,
                mask_ptr=mask,
            )

        for w in range(WARMUP):
            run(w)
        torch.cuda.synchronize()

        # Correctness: cosine similarity (robust for bf16 accum)
        ref = torch.matmul(As[0], Bs[0])
        Cs[0].zero_(); reset(); run(0); torch.cuda.synchronize()

        nz = (Cs[0].abs() > 0).sum().item()
        if nz < sz * sz * 0.99:
            del As, Bs, Cs; torch.cuda.empty_cache()
            return None, f"nz={nz}/{sz*sz}"

        c_flat = Cs[0].float().flatten()
        r_flat = ref.float().flatten()
        cos = torch.nn.functional.cosine_similarity(c_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()
        if cos < 0.999:
            del As, Bs, Cs; torch.cuda.empty_cache()
            return None, f"cos={cos:.6f}"

        ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            reset()
            ss[i].record(); run(i); es[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(ss, es)]
        del As, Bs, Cs; torch.cuda.empty_cache()
        return statistics.median(times), None

    except Exception as e:
        torch.cuda.empty_cache()
        msg = str(e)
        if "shared memory" in msg:
            return None, "OOM-LDS"
        return None, msg[:60]


def run_sweep(sz):
    print(f"\n{'='*80}")
    print(f"  Autoresearch sweep for {sz}x{sz}x{sz} BF16")
    print(f"{'='*80}")

    torch_ms = bench_torch(sz)
    flops = 2.0 * sz ** 3
    torch_tf = flops / (torch_ms * 1e-3) / 1e12
    print(f"  torch.matmul baseline: {torch_ms:.3f} ms ({torch_tf:.1f} TF)")
    print()

    configs = [
        # (BM, BN, BK, GROUP_SIZE_M, num_warps, waves_per_eu, num_stages)
        # --- 256x256 tile (3.4 tiles/CU at 8K, 7.6 at 12K, 13.5 at 16K) ---
        (256, 256, 64,  4, 8, 0, 2),   # current default
        (256, 256, 64,  4, 8, 1, 2),
        (256, 256, 64,  4, 8, 2, 2),
        (256, 256, 64,  2, 8, 0, 2),
        (256, 256, 64,  2, 8, 1, 2),
        (256, 256, 64,  8, 8, 0, 2),
        (256, 256, 64,  8, 8, 1, 2),
        (256, 256, 64, 16, 8, 0, 2),
        (256, 256, 64,  4, 4, 0, 2),
        (256, 256, 64,  4, 4, 1, 2),
        (256, 256, 64,  4, 4, 2, 2),
        # --- 256x128 tile (more tiles, potentially better ILP) ---
        (256, 128, 64,  4, 4, 0, 2),
        (256, 128, 64,  4, 4, 1, 2),
        (256, 128, 64,  4, 8, 0, 2),
        (256, 128, 64,  4, 8, 1, 2),
        (256, 128, 64,  8, 8, 0, 2),
        (256, 128, 64,  4, 4, 0, 3),  # 3 stages
        (256, 128, 64,  4, 8, 0, 3),
        # --- 128x256 tile ---
        (128, 256, 64,  4, 4, 0, 2),
        (128, 256, 64,  4, 8, 0, 2),
        (128, 256, 64,  4, 8, 1, 2),
        (128, 256, 64,  4, 4, 0, 3),
        (128, 256, 64,  4, 8, 0, 3),
        # --- 128x128 tile (4x more tiles, better load balancing) ---
        (128, 128, 64,  8, 4, 0, 2),
        (128, 128, 64,  8, 4, 1, 2),
        (128, 128, 64,  8, 4, 2, 2),
        (128, 128, 64,  8, 4, 0, 3),
        (128, 128, 64,  8, 4, 0, 4),
        (128, 128, 64,  8, 4, 1, 3),
        (128, 128, 64, 16, 4, 0, 2),
        (128, 128, 64, 16, 4, 1, 2),
        # --- 128x128 + BK=128 (halves K-iterations) ---
        (128, 128, 128, 8, 4, 0, 1),
        (128, 128, 128, 8, 4, 1, 1),
    ]

    print(f"  Configs to try: {len(configs)}")
    print(f"  {'#':>4s}  {'tile':>14s}  {'gm':>4s}  {'w':>2s}  {'wpe':>3s}  "
          f"{'st':>2s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}  {'note':>15s}")
    print("  " + "-" * 82)

    best_ms = 1e9
    best_cfg = None
    results = []

    for idx, (BM, BN, BK, gm, nw, wpe, ns) in enumerate(configs, 1):
        tile_str = f"{BM}x{BN}x{BK}"
        ws_ms, err = bench_ws(sz, BM, BN, BK, gm, nw, wpe, ns)

        if ws_ms is None:
            print(f"  {idx:>4d}  {tile_str:>14s}  {gm:>4d}  {nw:>2d}  {wpe:>3d}  "
                  f"{ns:>2d}  {'FAIL':>8s}  {'':>7s}  {'':>9s}  {err or '':>15s}")
            continue

        ws_tf = flops / (ws_ms * 1e-3) / 1e12
        vs = (ws_ms - torch_ms) / torch_ms * 100
        marker = " <<<" if ws_ms < best_ms else ""
        print(f"  {idx:>4d}  {tile_str:>14s}  {gm:>4d}  {nw:>2d}  {wpe:>3d}  "
              f"{ns:>2d}  {ws_ms:>8.3f}  {ws_tf:>7.1f}  {vs:>+8.1f}%{marker}")

        if ws_ms < best_ms:
            best_ms = ws_ms
            best_cfg = dict(BM=BM, BN=BN, BK=BK, gm=gm, warps=nw, wpe=wpe, stages=ns)

        results.append(dict(tile=tile_str, gm=gm, warps=nw, wpe=wpe, stages=ns,
                            ms=ws_ms, tflops=ws_tf, vs_torch_pct=vs))

    if best_cfg:
        best_tf = flops / (best_ms * 1e-3) / 1e12
        best_vs = (best_ms - torch_ms) / torch_ms * 100
        print()
        print(f"  BEST: {best_ms:.3f} ms ({best_tf:.1f} TF, {best_vs:+.1f}% vs torch)")
        print(f"  Config: {best_cfg}")
    else:
        best_vs = float('inf')
        print("\n  ALL CONFIGS FAILED")

    return dict(size=sz, torch_ms=torch_ms, torch_tf=torch_tf,
                best_ms=best_ms, best_tf=best_ms if best_cfg else None,
                best_vs_torch_pct=best_vs, best_config=best_cfg,
                all_results=results)


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)
    all_res = {}
    for sz in [4096, 8192, 12288, 16384]:
        r = run_sweep(sz)
        all_res[str(sz)] = r

    with open("results/autoresearch/kloop_sweep.json", "w") as f:
        json.dump(all_res, f, indent=2)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for k, r in all_res.items():
        if r.get("best_config"):
            print(f"  {k}: torch={r['torch_ms']:.3f}ms  best_WS={r['best_ms']:.3f}ms "
                  f"({r['best_vs_torch_pct']:+.1f}%)  cfg={r['best_config']}")
        else:
            print(f"  {k}: torch={r['torch_ms']:.3f}ms  ALL FAILED")
    print(f"\n  Saved to results/autoresearch/kloop_sweep.json")

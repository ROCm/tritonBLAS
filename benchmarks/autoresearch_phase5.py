#!/usr/bin/env python3
"""
Autoresearch Phase 5: ISA analysis + compiler flags + split ratios.

Part A: Dump compiled ISA for K-loop analysis
Part B: Test Triton LLVM flags that affect scheduling
Part C: Tune local/global split ratio
Part D: Cache modifier sweep (.cs, .cg, .ca)
Part E: max_num_imprecise_acc for tl.dot
"""
import json, os, statistics, sys, subprocess, tempfile
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


def bench_ws(sz, local_frac=0.9, cache_a=None, cache_b=None, extra_kw=None):
    try:
        BM, BN, BK = 256, 256, 64
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms; gsize_m = sel.group_m
        total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
        even_k = sz % BK == 0
        local_per_xcd = int(total_tiles * local_frac) // num_xcds
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
            CACHE_MODIFIER_A=cache_a, CACHE_MODIFIER_B=cache_b,
            QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
        )
        if extra_kw:
            kw.update(extra_kw)

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
        return None, str(e)[:80]


# ===========================================================================
# Part A: ISA dump
# ===========================================================================
def dump_isa(sz):
    """Compile the kernel and extract key ISA metrics."""
    print(f"\n  --- ISA Analysis for {sz}x{sz} ---")

    BM, BN, BK = 256, 256, 64
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    n_cu = sel._N_CU; num_xcds = sel.num_sms; gm = sel.group_m
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    even_k = sz % BK == 0
    lp = (total_tiles * 9) // (num_xcds * 10); lp = max(lp, 1)
    gt = total_tiles - lp * num_xcds

    A = torch.randn(sz, sz, dtype=dtype, device=device)
    B = torch.randn(sz, sz, dtype=dtype, device=device)
    C = torch.zeros(sz, sz, dtype=dtype, device=device)
    tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
    gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
    mask = torch.ones(n_cu, dtype=torch.int32, device=device)

    tc.zero_(); gc.zero_()
    ws_hierarchical_matmul[(n_cu,)](
        A, B, C, None, None, None, tc, gc,
        sz, sz, sz, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
        LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
        COUNTER_STRIDE=COUNTER_STRIDE,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
    )
    torch.cuda.synchronize()

    # Find the compiled kernel in Triton cache
    cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    asm_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".amdgcn"):
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                asm_files.append((mtime, path))

    if not asm_files:
        print("  No .amdgcn files found in Triton cache")
        del A, B, C; torch.cuda.empty_cache()
        return

    asm_files.sort(reverse=True)
    latest = asm_files[0][1]
    print(f"  Latest .amdgcn: {latest}")

    with open(latest, "r") as f:
        asm = f.read()

    # Count key instructions
    lines = asm.split("\n")
    mfma_count = sum(1 for l in lines if "v_mfma_" in l)
    buffer_load = sum(1 for l in lines if "buffer_load" in l)
    global_load = sum(1 for l in lines if "global_load" in l)
    ds_read = sum(1 for l in lines if "ds_read" in l or "ds_load" in l)
    ds_write = sum(1 for l in lines if "ds_write" in l or "ds_store" in l)
    s_waitcnt = sum(1 for l in lines if "s_waitcnt" in l)
    s_barrier = sum(1 for l in lines if "s_barrier" in l)
    total_instr = sum(1 for l in lines if l.strip() and not l.strip().startswith(";")
                      and not l.strip().startswith(".") and not l.strip().startswith("//"))

    # Find VGPR/SGPR usage
    vgpr_line = [l for l in lines if ".vgpr_count" in l]
    sgpr_line = [l for l in lines if ".sgpr_count" in l]
    lds_line = [l for l in lines if ".lds_size" in l or "group_segment_fixed_size" in l]

    print(f"  Total instructions: {total_instr}")
    print(f"  MFMA instructions: {mfma_count}")
    print(f"  Buffer loads:      {buffer_load}")
    print(f"  Global loads:      {global_load}")
    print(f"  DS reads:          {ds_read}")
    print(f"  DS writes:         {ds_write}")
    print(f"  s_waitcnt:         {s_waitcnt}")
    print(f"  s_barrier:         {s_barrier}")
    if vgpr_line: print(f"  VGPRs: {vgpr_line[0].strip()}")
    if sgpr_line: print(f"  SGPRs: {sgpr_line[0].strip()}")
    if lds_line: print(f"  LDS:   {lds_line[0].strip()}")

    # Estimate MFMA density: mfma / (mfma + waits + barriers + loads)
    overhead = s_waitcnt + s_barrier + buffer_load + global_load
    if mfma_count > 0:
        density = mfma_count / (mfma_count + overhead) * 100
        print(f"  MFMA density: {density:.1f}% (mfma / (mfma + waits + loads))")

    # Save full ISA for inspection
    os.makedirs("results/autoresearch", exist_ok=True)
    out_path = f"results/autoresearch/isa_{sz}.amdgcn"
    import shutil
    shutil.copy2(latest, out_path)
    print(f"  Saved to {out_path}")

    del A, B, C; torch.cuda.empty_cache()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)

    # Part A: ISA analysis
    print("=" * 80)
    print("  PART A: ISA Analysis")
    print("=" * 80)
    for sz in [8192, 16384]:
        dump_isa(sz)

    # Part B + C + D: Compiler flags, split ratios, cache modifiers
    print(f"\n{'='*80}")
    print("  PART B-D: Split Ratios, Cache Modifiers, Compiler Tuning")
    print(f"{'='*80}")

    configs = [
        # (label, local_frac, cache_a, cache_b, extra_kw)
        ("Baseline (90/10, no cache mod)", 0.9, None, None, None),
        ("85/15 split", 0.85, None, None, None),
        ("95/5 split", 0.95, None, None, None),
        ("80/20 split", 0.80, None, None, None),
        ("100/0 (all local)", 1.0, None, None, None),
        ("0/100 (all global)", 0.0, None, None, None),
        ("Cache .cg / .cg", 0.9, ".cg", ".cg", None),
        ("Cache .cs / .cs", 0.9, ".cs", ".cs", None),
        ("Cache .cg / .cs", 0.9, ".cg", ".cs", None),
        ("wpe=1", 0.9, None, None, dict(waves_per_eu=1)),
        ("wpe=2", 0.9, None, None, dict(waves_per_eu=2)),
        ("wpe=1 + 85/15", 0.85, None, None, dict(waves_per_eu=1)),
        ("wpe=2 + .cs/.cs", 0.9, ".cs", ".cs", dict(waves_per_eu=2)),
    ]

    all_results = {}
    for sz in [8192, 12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch.matmul: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")
        print(f"  {'Config':<32s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}  {'note':>10s}")
        print("  " + "-" * 75)

        size_results = {"torch_ms": torch_ms}
        for label, lf, ca, cb, ekw in configs:
            ws_ms, err = bench_ws(sz, local_frac=lf, cache_a=ca, cache_b=cb, extra_kw=ekw)
            if ws_ms is None:
                print(f"  {label:<32s}  {'FAIL':>8s}  {'':>7s}  {'':>9s}  {err or '':>10s}")
            else:
                ws_tf = flops / (ws_ms * 1e-3) / 1e12
                vs = (ws_ms - torch_ms) / torch_ms * 100
                print(f"  {label:<32s}  {ws_ms:>8.3f}  {ws_tf:>7.1f}  {vs:>+8.1f}%")
                size_results[label] = {"ms": ws_ms, "tflops": ws_tf, "vs_pct": vs}

        all_results[str(sz)] = size_results

    with open("results/autoresearch/phase5_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("  SUMMARY (best per size)")
    print(f"{'='*80}")
    for k, r in all_results.items():
        best_name, best_pct = None, float('inf')
        for name, data in r.items():
            if name == "torch_ms" or not isinstance(data, dict): continue
            if data["vs_pct"] < best_pct:
                best_pct = data["vs_pct"]; best_name = name
        if best_name:
            print(f"  {k}: torch={r['torch_ms']:.3f}ms  best={best_name} ({best_pct:+.1f}%)")
    print(f"\n  Saved to results/autoresearch/phase5_results.json")

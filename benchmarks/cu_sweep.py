#!/usr/bin/env python3
"""CU sweep: GFLOPs vs available CUs for ws, streamk, torch.

For WS/streamk: controls grid size via total_cus parameter.
For torch: uses ROC_GLOBAL_CU_MASK env var.

MI300X: 304 CUs = 8 XCDs * 38 CUs/XCD.
CU mask is per-SE (shader engine). 8 XCDs with ~5 SEs each = ~38 SEs.
ROC_GLOBAL_CU_MASK format: one 64-bit mask applied uniformly to all SEs.
Each SE has 10 CUs (2 WGPs * 4 CUs/WGP + 2 CUs from the last WGP).
Actually on MI300X: 304 CUs / 8 XCDs = 38 CUs/XCD.
Each XCD has 4 SEs (actually called "shader arrays"), each with ~10 CUs.
The mask bits correspond to CU lanes within each SE.

For CU masking we set ROC_GLOBAL_CU_MASK to limit active CUs.
For tritonBLAS WS, we pass total_cus to the selector which limits the grid.
"""
import torch
import tritonblas
import statistics
import json
import os
import sys
import ctypes

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
stream = torch.cuda.Stream(device=dev)

OUT_DIR = "results/plot_data"
os.makedirs(OUT_DIR, exist_ok=True)

GEMM_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
M = N = K = GEMM_SIZE
FLOPS = 2.0 * M * N * K

print(f"CU sweep for {M}x{N}x{K} bf16")
print(f"  Peak FP16 TFLOPS (MI300X): ~1300")
print()

CU_COUNTS = sorted(set(
    list(range(24, 305, 8)) + [304, 280, 272, 256, 240, 200, 160, 120, 80, 40]
))

A_base = torch.randn(M, K, dtype=dtype, device=dev)
B_base = torch.randn(K, N, dtype=dtype, device=dev)
C_base = torch.empty(M, N, dtype=dtype, device=dev)


def bench(fn, reset_fn, warmup=10, steps=30):
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            if reset_fn:
                reset_fn()
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        if reset_fn:
            with torch.cuda.stream(stream):
                reset_fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(stream)
        with torch.cuda.stream(stream):
            fn()
        en.record(stream)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


data = {"size": GEMM_SIZE, "cu_counts": CU_COUNTS}

# --- WS sweep ---
print("=== WS (grid-limited) ===")
ws_results = []
for n_cus in CU_COUNTS:
    try:
        A = A_base.clone()
        B = B_base.clone()
        C = C_base.clone()
        sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev,
                                                total_cus=n_cus)
        cfg = tritonblas.matmul_preamble(sel)
        tiles = (M // sel.block_m) * (N // sel.block_n)

        def fn(a=A, b=B, c=C, s_=sel, cf=cfg):
            tritonblas.matmul_lt(a, b, c, s_, cf, work_stealing=True)
        def rst(cf=cfg):
            cf.reset(work_stealing=True)

        times = bench(fn, rst)
        med = statistics.median(times)
        tflops = FLOPS / (med * 1e-3) / 1e12
        ws_results.append({"cus": n_cus, "median_ms": med, "tflops": tflops, "tiles": tiles})
        if n_cus % 40 == 0 or n_cus in [24, 304]:
            print(f"  CUs={n_cus:>3}: {med:.3f} ms  {tflops:.1f} TF  tiles={tiles}")
    except Exception as e:
        ws_results.append({"cus": n_cus, "median_ms": None, "tflops": None})
        print(f"  CUs={n_cus:>3}: FAILED ({str(e)[:60]})")
data["ws"] = ws_results

# --- StreamK+WS sweep ---
print("\n=== StreamK+WS (grid-limited) ===")
sk_results = []
for n_cus in CU_COUNTS:
    try:
        A = A_base.clone()
        B = B_base.clone()
        C = C_base.clone()
        sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev,
                                                streamk=True, total_cus=n_cus)
        cfg = tritonblas.matmul_preamble(sel)

        def fn(a=A, b=B, c=C, s_=sel, cf=cfg):
            tritonblas.matmul_lt(a, b, c, s_, cf, enable_streamk=True, work_stealing=True)
        def rst(cf=cfg):
            cf.reset(streamk=True, work_stealing=True)

        times = bench(fn, rst)
        med = statistics.median(times)
        tflops = FLOPS / (med * 1e-3) / 1e12
        sk_results.append({"cus": n_cus, "median_ms": med, "tflops": tflops})
        if n_cus % 40 == 0 or n_cus in [24, 304]:
            print(f"  CUs={n_cus:>3}: {med:.3f} ms  {tflops:.1f} TF")
    except Exception as e:
        sk_results.append({"cus": n_cus, "median_ms": None, "tflops": None})
        print(f"  CUs={n_cus:>3}: FAILED ({str(e)[:60]})")
data["streamk_ws"] = sk_results

# --- torch sweep (CU masking) ---
print("\n=== torch.matmul (ROC_GLOBAL_CU_MASK) ===")
torch_results = []

# MI300X CU mask: 304 CUs across 8 XCDs, 38 CUs per XCD
# ROC_GLOBAL_CU_MASK is a per-SE mask (10 CU lanes per SE, 4 SEs per XCD)
# To disable N CUs: we disable from the top SE lanes down
# Simpler approach: each SE has 10 CUs (bits 0-9). To get X CUs total out of 304:
#   active_per_se = ceil(X * 10 / 304) ≈ X / 30.4
# But this is approximate. For the benchmark, we just use the mask directly.
# Actually, ROC_GLOBAL_CU_MASK uses lower bits per SE.
# For N CUs out of 304 (38/XCD, ~10/SE, ~4 SEs/XCD, 8 XCDs = 32 SEs):
#   cus_per_se = 304/32 ≈ 9.5 → 10 CUs per SE
#   To get N CUs: enable ceil(N/32) CUs per SE = set that many bits

for n_cus in CU_COUNTS:
    try:
        cus_per_se = 10
        n_ses = 32  # approximate for MI300X
        target_per_se = max(1, round(n_cus * cus_per_se / 304))
        target_per_se = min(target_per_se, cus_per_se)
        mask_val = (1 << target_per_se) - 1
        actual_cus = target_per_se * n_ses
        os.environ["ROC_GLOBAL_CU_MASK"] = hex(mask_val)

        A = A_base.clone()
        B = B_base.clone()
        C = C_base.clone()
        def fn(a=A, b=B, c=C):
            torch.matmul(a, b, out=c)

        times = bench(fn, None)
        med = statistics.median(times)
        tflops = FLOPS / (med * 1e-3) / 1e12
        torch_results.append({
            "cus": n_cus, "actual_cus": actual_cus, "mask": hex(mask_val),
            "cus_per_se": target_per_se,
            "median_ms": med, "tflops": tflops
        })
        if n_cus % 40 == 0 or n_cus in [24, 304]:
            print(f"  CUs={n_cus:>3} (actual~{actual_cus:>3}, mask={hex(mask_val)}): "
                  f"{med:.3f} ms  {tflops:.1f} TF")
    except Exception as e:
        torch_results.append({"cus": n_cus, "median_ms": None, "tflops": None})
        print(f"  CUs={n_cus:>3}: FAILED ({str(e)[:60]})")
    finally:
        if "ROC_GLOBAL_CU_MASK" in os.environ:
            del os.environ["ROC_GLOBAL_CU_MASK"]

data["torch"] = torch_results

with open(f"{OUT_DIR}/cu_sweep_{GEMM_SIZE}.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved {OUT_DIR}/cu_sweep_{GEMM_SIZE}.json")

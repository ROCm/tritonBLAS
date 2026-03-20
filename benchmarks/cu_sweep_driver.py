#!/usr/bin/env python3
"""Driver: runs CU sweeps for WS (grid-limited) and torch (CU-masked via subprocess).

MI300X topology: 304 CUs = 8 XCDs * 38 CUs/XCD.
Each XCD has 4 Shader Arrays (SAs), each SA has ~10 CUs.
ROC_GLOBAL_CU_MASK: lower 10 bits correspond to CU lanes within each SA.
There are 32 SAs total (8 XCDs * 4 SAs).
Setting mask=0x1FF means 9 CUs per SA => 9*32=288 CUs active.
Setting mask=0x1 means 1 CU per SA => 32 CUs active.
"""
import torch
import tritonblas
import statistics
import json
import os
import subprocess
import sys

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
stream = torch.cuda.Stream(device=dev)

OUT_DIR = "results/plot_data"
os.makedirs(OUT_DIR, exist_ok=True)

SZ = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
M = N = K = SZ
FLOPS = 2.0 * M * N * K
CUS_PER_SA = 10
N_SAS = 32  # 8 XCDs * 4 SAs

# CU counts to sweep (we can only get multiples of 32 with uniform masking)
# mask bits 1-10 => 32, 64, 96, ..., 320 actual CUs (capped at 304)
MASK_CONFIGS = []
for bits in range(1, CUS_PER_SA + 1):
    mask_val = (1 << bits) - 1
    actual = min(bits * N_SAS, 304)
    MASK_CONFIGS.append((actual, mask_val))

# Also sweep WS with matching CU counts + finer granularity
WS_CU_COUNTS = sorted(set(
    [c for c, _ in MASK_CONFIGS] +
    list(range(24, 305, 8)) + [304]
))

print(f"CU sweep for {SZ}x{SZ}x{SZ} bf16")
print(f"  torch mask points: {[(c, hex(m)) for c, m in MASK_CONFIGS]}")
print()


def bench(fn, reset_fn, warmup=15, steps=30):
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


# --- WS sweep ---
print("=== WS (grid-limited) ===")
A = torch.randn(M, K, dtype=dtype, device=dev)
B = torch.randn(K, N, dtype=dtype, device=dev)
C = torch.empty(M, N, dtype=dtype, device=dev)

ws_results = []
for n_cus in WS_CU_COUNTS:
    try:
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
        if n_cus % 48 == 0 or n_cus in [24, 304]:
            print(f"  CUs={n_cus:>3}: {med:.3f} ms  {tflops:.1f} TF")
    except Exception as e:
        ws_results.append({"cus": n_cus, "median_ms": None, "tflops": None})

# --- torch sweep (subprocess per mask) ---
print("\n=== torch.matmul (ROC_GLOBAL_CU_MASK, subprocess) ===")
torch_results = []
for actual_cus, mask_val in MASK_CONFIGS:
    env = os.environ.copy()
    env["ROC_GLOBAL_CU_MASK"] = hex(mask_val)
    try:
        result = subprocess.run(
            ["python3", "benchmarks/cu_sweep_torch.py", str(M), str(N), str(K)],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            med = float(parts[0])
            tflops = float(parts[1])
            torch_results.append({
                "cus": actual_cus, "mask": hex(mask_val),
                "cus_per_sa": mask_val.bit_length() if isinstance(mask_val, int) else 0,
                "median_ms": med, "tflops": tflops,
            })
            print(f"  CUs={actual_cus:>3} (mask={hex(mask_val)}): {med:.3f} ms  {tflops:.1f} TF")
        else:
            print(f"  CUs={actual_cus:>3}: FAILED (rc={result.returncode})")
            torch_results.append({"cus": actual_cus, "median_ms": None, "tflops": None})
    except Exception as e:
        print(f"  CUs={actual_cus:>3}: FAILED ({str(e)[:60]})")
        torch_results.append({"cus": actual_cus, "median_ms": None, "tflops": None})

# Full CU torch baseline (no mask)
result = subprocess.run(
    ["python3", "benchmarks/cu_sweep_torch.py", str(M), str(N), str(K)],
    capture_output=True, text=True, timeout=120,
)
if result.returncode == 0:
    parts = result.stdout.strip().split()
    torch_results.append({"cus": 304, "mask": "none", "median_ms": float(parts[0]),
                           "tflops": float(parts[1])})
    print(f"  CUs=304 (no mask): {parts[0]} ms  {parts[1]} TF")

data = {"size": SZ, "ws": ws_results, "torch": torch_results,
        "mask_configs": [{"cus": c, "mask": hex(m)} for c, m in MASK_CONFIGS]}

with open(f"{OUT_DIR}/cu_sweep_{SZ}.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved {OUT_DIR}/cu_sweep_{SZ}.json")

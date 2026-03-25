#!/usr/bin/env python3
"""
Investigate the correct ROC_GLOBAL_CU_MASK format for MI300X (multi-XCD).

MI300X: 8 XCDs × 38 CUs/XCD = 304 CUs total
Each XCD has 4 Shader Engines (SEs), each SE has ~10 CUs.

Hypothesis: The mask format needs to specify per-XCD, not just a flat bitfield.
Possible formats:
  - Dot-separated per-SE: "0xFF.0xFF.0xFF.0xFF" (4 SEs per XCD)
  - Colon-separated per-XCD  
  - Extended 64-bit+ format for multi-XCD
"""
import torch
import statistics
import os
import subprocess
import sys
import time

def bench_torch(M=8192, warmup=10, steps=30):
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    A = torch.randn(M, M, dtype=torch.bfloat16, device=dev)
    B = torch.randn(M, M, dtype=torch.bfloat16, device=dev)
    C = torch.empty(M, M, dtype=torch.bfloat16, device=dev)
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    t = []
    for _ in range(steps):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        torch.matmul(A, B, out=C)
        en.record()
        torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    return statistics.median(t), min(t), max(t)


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    med, mn, mx = bench_torch()
    props = torch.cuda.get_device_properties(0)
    cus = props.multi_processor_count
    print(f"RESULT {med:.4f} {mn:.4f} {mx:.4f} {cus}")
    sys.exit(0)


def run_with_mask(mask_val, label):
    env = os.environ.copy()
    if mask_val is not None:
        env["ROC_GLOBAL_CU_MASK"] = mask_val
    elif "ROC_GLOBAL_CU_MASK" in env:
        del env["ROC_GLOBAL_CU_MASK"]
    try:
        r = subprocess.run(
            [sys.executable, __file__, "worker"],
            capture_output=True, text=True, timeout=90, env=env,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med, mn, mx, cus = float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
                    flops = 2.0 * 8192**3
                    tflops = flops / (med * 1e-3) / 1e12
                    print(f"  {label:<55} CUs={cus:>4} {med:>8.3f}ms {tflops:>6.0f}TF")
                    return cus, med, tflops
        else:
            err = (r.stderr or "")[-200:]
            print(f"  {label:<55} FAILED (rc={r.returncode}) {err[:80]}")
    except subprocess.TimeoutExpired:
        print(f"  {label:<55} TIMEOUT")
    return None, None, None


print("=" * 80)
print("  MI300X ROC_GLOBAL_CU_MASK Format Investigation")
print("=" * 80)
print()

# MI300X has 8 XCDs, each with 4 SEs, each SE has ~10 CUs
# SE mask format: 10 CUs per SE means bits 0-9 per SE
# XCD mask = SE0_mask | (SE1_mask << 10) | (SE2_mask << 20) | (SE3_mask << 30)
# But 4*10=40 bits > 32, so it might use a different mapping

print("--- Phase 1: Baseline ---")
run_with_mask(None, "no mask")
time.sleep(2)

print("\n--- Phase 2: Try dot-separated SE masks ---")
# Format: "SE0.SE1.SE2.SE3" per the ROCm documentation
for fmt in [
    "0x3ff.0x3ff.0x3ff.0x3ff",        # 10 bits per SE, 4 SEs = 40 CUs (1 XCD)
    "0xff.0xff.0xff.0xff",              # 8 bits per SE
    "0x1ff.0x1ff.0x1ff.0x1ff",          # 9 bits per SE  
    "0x1f.0x1f.0x1f.0x1f",              # 5 bits per SE = 20 CUs
    "0x3ff.0x3ff.0x3ff.0x3ff.0x3ff.0x3ff.0x3ff.0x3ff",  # 8 SEs?
]:
    run_with_mask(fmt, f"dot-sep: {fmt}")
    time.sleep(1)

print("\n--- Phase 3: Try colon-separated XCD masks ---")
# Format: "XCD0:XCD1:XCD2:...:XCD7"
for fmt in [
    ":".join(["0x3fffffffff"] * 8),     # all CUs on all 8 XCDs
    ":".join(["0xffffffff"] * 8),       # 32 bits per XCD
]:
    run_with_mask(fmt, f"colon-sep: {fmt[:50]}...")
    time.sleep(1)

print("\n--- Phase 4: Try comma-separated ---")
for fmt in [
    ",".join(["0x3fffffffff"] * 8),
]:
    run_with_mask(fmt, f"comma-sep: {fmt[:50]}...")
    time.sleep(1)

print("\n--- Phase 5: Try XCD:SE hierarchical format ---")
# Some ROCm versions use "XCD:SE0.SE1.SE2.SE3" format
for fmt in [
    "0xff:0x3ff.0x3ff.0x3ff.0x3ff",     
    "0:0x3ff.0x3ff.0x3ff.0x3ff",
]:
    run_with_mask(fmt, f"xcd:se: {fmt}")
    time.sleep(1)

print("\n--- Phase 6: Large hex (multi-XCD in single value) ---")
# If mask is 304 bits, we need 304/4 = 76 hex chars
# Or if it's 38 bits per XCD = 8 * 38 = 304 bits
all_cus_hex = hex((1 << 304) - 1)
half_cus_hex = hex((1 << 152) - 1)
for val, label in [
    (all_cus_hex, "304-bit all-ones"),
    (hex((1 << 76) - 1), "76-bit all-ones (2 XCDs)"),
    (hex((1 << 38) - 1), "38-bit (1 XCD)"),
]:
    run_with_mask(val, f"large-hex: {label}")
    time.sleep(1)

print("\n--- Phase 7: Check HSA_CU_MASK_SKIP_INIT and related vars ---")
for env_name, val in [
    ("HSA_CU_MASK_SKIP_INIT", "1"),
    ("HSA_OVERRIDE_GFX_VERSION", ""),
]:
    if env_name == "HSA_OVERRIDE_GFX_VERSION":
        continue
    env = os.environ.copy()
    env[env_name] = val
    env["ROC_GLOBAL_CU_MASK"] = "0x3fffffffff"
    try:
        r = subprocess.run(
            [sys.executable, __file__, "worker"],
            capture_output=True, text=True, timeout=90, env=env,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med, mn, mx, cus = float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
                    flops = 2.0 * 8192**3
                    tflops = flops / (med * 1e-3) / 1e12
                    print(f"  {env_name}={val} + 0x3fffffffff: CUs={cus:>4} "
                          f"{med:>8.3f}ms {tflops:>6.0f}TF")
        else:
            print(f"  {env_name}={val}: FAILED")
    except subprocess.TimeoutExpired:
        print(f"  {env_name}={val}: TIMEOUT")

print()
print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

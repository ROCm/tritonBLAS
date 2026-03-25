#!/usr/bin/env python3
"""Investigate why ROC_GLOBAL_CU_MASK causes torch.matmul to be catastrophically slow."""
import torch
import statistics
import os
import subprocess
import sys

def bench_torch(M=8192, warmup=10, steps=30):
    """Benchmark torch.matmul on current GPU."""
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


print("=" * 70)
print("  Investigating ROC_GLOBAL_CU_MASK + torch.matmul")
print("=" * 70)

# Test 1: Does the CU mask change the reported CU count?
print("\n--- Test 1: Reported CU count with different masks ---")
masks = [
    ("none", None),
    ("0x3fffffffff", "0x3fffffffff"),   # 38 bits → should be all CUs
    ("0x1fffffffff", "0x1fffffffff"),   # 37 bits
    ("0xffffffff",   "0xffffffff"),     # 32 bits
    ("0xffff",       "0xffff"),         # 16 bits
    ("0xff",         "0xff"),           # 8 bits
    ("0x1",          "0x1"),            # 1 bit
]

results = []
for label, mask_val in masks:
    env = os.environ.copy()
    if mask_val:
        env["ROC_GLOBAL_CU_MASK"] = mask_val
    elif "ROC_GLOBAL_CU_MASK" in env:
        del env["ROC_GLOBAL_CU_MASK"]

    try:
        r = subprocess.run(
            [sys.executable, __file__, "worker"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med, mn, mx, cus = float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
                    flops = 2.0 * 8192**3
                    tflops = flops / (med * 1e-3) / 1e12
                    results.append((label, mask_val, med, mn, cus, tflops))
                    print(f"  mask={label:>20}: {med:.3f}ms (min={mn:.3f}) "
                          f"CUs={cus} {tflops:.0f}TF")
                    break
        else:
            err = r.stderr[-200:] if r.stderr else "unknown"
            print(f"  mask={label:>20}: FAILED (rc={r.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  mask={label:>20}: TIMEOUT")

# Test 2: Is it the mask itself or the reduced CU count?
print("\n--- Test 2: Does setting mask to ALL bits still hurt? ---")
# If mask=0x3fffffffff (all 38 bits, should enable all 304 CUs) is slow,
# then the mask mechanism itself has overhead regardless of how many CUs are enabled.

no_mask = [r for r in results if r[1] is None]
all_mask = [r for r in results if r[1] == "0x3fffffffff"]
if no_mask and all_mask:
    nm = no_mask[0]
    am = all_mask[0]
    ratio = am[2] / nm[2]
    print(f"  No mask: {nm[2]:.3f}ms ({nm[5]:.0f}TF) CUs={nm[4]}")
    print(f"  All-bits mask: {am[2]:.3f}ms ({am[5]:.0f}TF) CUs={am[4]}")
    print(f"  Ratio: {ratio:.2f}x")
    if ratio > 2:
        print("  >>> CONFIRMED: The mask itself causes massive overhead,")
        print("      even when all CUs are enabled. This is NOT about reduced CU count.")
        print("      It's likely that hipBLASLt/Tensile selects a DIFFERENT (worse)")
        print("      kernel when ROC_GLOBAL_CU_MASK is set, or the mask mechanism")
        print("      adds overhead to every wavefront dispatch.")
    else:
        print("  >>> Mask with all bits does NOT cause overhead.")
        print("      The slowdown is from actual CU reduction.")

# Test 3: Does the HIP runtime report different CU count with mask?
print("\n--- Test 3: Reported CU counts ---")
for label, mask_val, med, mn, cus, tflops in results:
    if mask_val is None:
        baseline_cus = cus
        print(f"  Baseline CUs: {cus}")
        break
for label, mask_val, med, mn, cus, tflops in results:
    if mask_val:
        print(f"  mask={label:>20}: CUs={cus} (vs baseline {baseline_cus})")

print("\n--- Summary ---")
print(f"{'Mask':<22} {'CUs':>5} {'Median(ms)':>12} {'TFLOPS':>8} {'vs no-mask':>12}")
print("-" * 65)
for label, mask_val, med, mn, cus, tflops in results:
    nm_med = results[0][2]
    ratio = med / nm_med
    print(f"{label:<22} {cus:>5} {med:>12.3f} {tflops:>8.0f} {ratio:>11.2f}x")

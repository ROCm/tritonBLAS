#!/usr/bin/env python3
"""
Scalable autoresearch: validate StreamK+WS fix across CU counts in parallel.

Launches one subprocess per GPU, each testing a different CU mask value.
Tests the previously-broken CU counts (48, 80, 112, 144, 272, 304) plus
the always-good ones (8, 32, 64, 128, 256) for comparison.
"""
import json
import os
import subprocess
import sys
import time
import statistics

WORKER_SCRIPT = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas

M = int(sys.argv[1])
STEPS = int(sys.argv[2])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)

sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=True)
cfg = tritonblas.matmul_preamble(sel)

def bench():
    tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=True, work_stealing=True)
def reset():
    cfg.reset(streamk=True, work_stealing=True)

for _ in range(20):
    reset()
    bench()
torch.cuda.synchronize()

t = []
for _ in range(STEPS):
    reset()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); bench(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))

med = statistics.median(t)
cus = torch.cuda.get_device_properties(0).multi_processor_count
flops = 2.0 * M**3
tflops = flops / (med * 1e-3) / 1e12
print(f"RESULT {med:.4f} {min(t):.4f} {max(t):.4f} {cus} {tflops:.1f}")
'''


def make_multi_xcd_mask(cus_per_xcd, n_xcds=8, bits_per_xcd=38):
    per_xcd = (1 << cus_per_xcd) - 1
    full = 0
    for xcd in range(n_xcds):
        full |= per_xcd << (xcd * bits_per_xcd)
    return hex(full)


def main():
    script_path = "/tmp/_skws_fix_worker.py"
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT)

    # Test CU counts that were previously broken (odd divisors)
    # plus ones that were always good (powers of 2)
    test_cus_per_xcd = [1, 4, 6, 8, 10, 14, 16, 18, 32, 34, 38]
    size = 8192
    steps = 30
    n_gpus = 8

    print(f"Testing StreamK+WS fix: {size}x{size} BF16")
    print(f"{'CUs':>5} {'CUs/XCD':>8} {'TFLOPS':>8} {'ms':>8} {'was_broken':>12}")
    print("-" * 50)

    # Run in waves of n_gpus
    results = {}
    for wave_start in range(0, len(test_cus_per_xcd), n_gpus):
        wave = test_cus_per_xcd[wave_start:wave_start + n_gpus]
        procs = []
        for i, cpx in enumerate(wave):
            gpu = i
            total_cus = cpx * 8
            mask = make_multi_xcd_mask(cpx)
            env = os.environ.copy()
            env["HIP_VISIBLE_DEVICES"] = str(gpu)
            env["ROC_GLOBAL_CU_MASK"] = mask
            p = subprocess.Popen(
                [sys.executable, script_path, str(size), str(steps)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env,
            )
            procs.append((cpx, total_cus, p))

        for cpx, total_cus, p in procs:
            try:
                stdout, stderr = p.communicate(timeout=120)
                was_broken = total_cus not in {8, 16, 32, 64, 128, 256}
                if p.returncode == 0:
                    for line in stdout.strip().split("\n"):
                        if line.startswith("RESULT"):
                            parts = line.split()
                            med = float(parts[1])
                            tflops = float(parts[5])
                            results[total_cus] = tflops
                            tag = "FIXED?" if was_broken else "ok"
                            print(f"{total_cus:>5} {cpx:>8} {tflops:>8.0f} {med:>8.3f} {tag:>12}")
                else:
                    print(f"{total_cus:>5} {cpx:>8} {'FAIL':>8} {'---':>8} {'FAIL':>12}")
                    if stderr:
                        print(f"  stderr: {stderr[-200:]}")
            except subprocess.TimeoutExpired:
                p.kill()
                print(f"{total_cus:>5} {cpx:>8} {'TIMEOUT':>8}")

        time.sleep(3)

    # Summary
    print(f"\n{'='*50}")
    print("Summary: previously-broken CU counts")
    broken_cus = [48, 80, 112, 144, 272, 304]
    good_cus = [8, 32, 64, 128, 256]
    good_avg = statistics.mean([results[c] for c in good_cus if c in results]) if any(c in results for c in good_cus) else 0
    for c in sorted(results.keys()):
        tf = results[c]
        was_broken = c in [c2 * 8 for c2 in [6, 10, 14, 18, 34]]
        status = "WAS-BROKEN" if was_broken else "was-ok"
        print(f"  {c:>4} CUs: {tf:>6.0f} TF  ({status})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CU Sweep Benchmark for MI300X.

Sweeps CU count from 8 to 304 and measures GEMM performance for:
  - torch.matmul           (CU-masked via ROC_GLOBAL_CU_MASK)
  - WS grid-limited        (grid = cus, all 304 physical CUs available)
  - WS full-grid           (grid = num_tiles, CU-masked)
  - Persistent GEMM        (CU-masked)
  - Stream-K               (CU-masked)
  - Stream-K + WS          (CU-masked)

Usage:
    python3 benchmarks/cu_sweep.py --size 8192 --gpu 4
    python3 benchmarks/cu_sweep.py --size 4096 --gpu 4
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

DATA_DIR = "results/plot_data"


def make_multi_xcd_mask(cus_per_xcd, n_xcds=8, bits_per_xcd=38):
    """Build ROC_GLOBAL_CU_MASK hex for MI300X: cus_per_xcd bits set per XCD block."""
    per_xcd = (1 << cus_per_xcd) - 1
    full = 0
    for xcd in range(n_xcds):
        full |= per_xcd << (xcd * bits_per_xcd)
    return hex(full)


# The worker script runs in a subprocess for GPU isolation.
# MODE: torch | ws-grid | ws-full | persistent | streamk | streamk-ws
WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

MODE = sys.argv[1]
M = int(sys.argv[2])
STEPS = int(sys.argv[3])
TOTAL_CUS = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "0" else 0

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)

if MODE == "torch":
    def bench():
        torch.matmul(A, B)
    reset = lambda: None
else:
    import tritonblas
    C = torch.empty(M, M, dtype=dtype, device=dev)

    sk = MODE in ("streamk", "streamk-ws")
    ws = MODE in ("ws-grid", "ws-full", "streamk-ws")

    sel = tritonblas.OrigamiMatmulSelector(
        M, M, M, dtype, dtype, dtype, dev,
        streamk=sk,
        total_cus=TOTAL_CUS if TOTAL_CUS > 0 else None,
    )
    cfg = tritonblas.matmul_preamble(sel)
    def bench():
        tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=sk, work_stealing=ws)
    def reset():
        cfg.reset(streamk=sk, work_stealing=ws)

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


def run_worker(mode, size, steps, total_cus=0, mask=None, timeout=120):
    """Run the benchmark worker in a subprocess, return (med_ms, tflops, reported_cus) or None."""
    script = "/tmp/_cu_sweep_worker.py"
    if not os.path.exists(script):
        with open(script, "w") as f:
            f.write(WORKER)

    env = os.environ.copy()
    if mask:
        env["ROC_GLOBAL_CU_MASK"] = mask
    elif "ROC_GLOBAL_CU_MASK" in env:
        del env["ROC_GLOBAL_CU_MASK"]

    cmd = [sys.executable, script, mode, str(size), str(steps), str(total_cus)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med = float(parts[1])
                    reported = int(parts[4])
                    tflops = float(parts[5])
                    return med, tflops, reported
        else:
            err = (r.stderr or "")[-200:]
            print(f"      FAILED (rc={r.returncode}) {err[:80]}")
    except subprocess.TimeoutExpired:
        print(f"      TIMEOUT")
    return None


def sweep_one(label, mode, size, steps, cus_list, use_mask=False, cooldown=3):
    """Run a sweep for one backend. Returns list of {cus, tflops, med_ms, ...}."""
    print(f"\n  [{label}] ({len(cus_list)} points)")
    results = []
    for target_cus in cus_list:
        cpx = target_cus // 8
        mask = make_multi_xcd_mask(cpx) if use_mask else None
        total_cus_arg = target_cus if not use_mask else 0

        r = run_worker(mode, size, steps, total_cus=total_cus_arg, mask=mask)
        if r:
            med, tflops, reported = r
            results.append({
                "cus": target_cus, "tflops": tflops, "med_ms": med,
                "reported_cus": reported, "cus_per_xcd": cpx,
            })
            flag = f"mask={cpx}/xcd" if use_mask else f"grid={total_cus_arg}"
            print(f"    {target_cus:>4d} CUs ({flag}): {med:.3f}ms {tflops:.0f}TF")
        else:
            print(f"    {target_cus:>4d} CUs: FAILED")
        time.sleep(cooldown)
    return results


def main():
    parser = argparse.ArgumentParser(description="CU sweep benchmark")
    parser.add_argument("--size", type=int, required=True, help="GEMM size (M=N=K)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--steps", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--cooldown", type=int, default=3, help="Seconds between runs")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    sz = args.size
    os.makedirs(DATA_DIR, exist_ok=True)

    # CU values: every 2 CUs per XCD from 1 to 38
    cus_per_xcd_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    cus_list = [cpx * 8 for cpx in cus_per_xcd_list]
    # Add 304 explicitly (38*8=304, already there)

    print(f"{'='*70}")
    print(f"  CU Sweep: {sz}x{sz}x{sz} BF16 on MI300X")
    print(f"  CU points: {cus_list}")
    print(f"{'='*70}")

    data = {}

    # 1) torch (CU-masked)
    data["torch"] = sweep_one("torch (CU-masked)", "torch", sz, args.steps, cus_list,
                              use_mask=True, cooldown=args.cooldown)

    # 2) WS grid-limited (grid = cus, no mask)
    data["ws_grid"] = sweep_one("WS grid-limited", "ws-grid", sz, args.steps, cus_list,
                                use_mask=False, cooldown=args.cooldown)

    # 3) WS full-grid (grid = num_tiles, CU-masked)
    data["ws_full"] = sweep_one("WS full-grid (CU-masked)", "ws-full", sz, args.steps, cus_list,
                                use_mask=True, cooldown=args.cooldown)

    # 4) Persistent GEMM (CU-masked)
    data["persistent"] = sweep_one("Persistent GEMM (CU-masked)", "persistent", sz, args.steps, cus_list,
                                   use_mask=True, cooldown=args.cooldown)

    # 5) Stream-K (CU-masked)
    data["streamk"] = sweep_one("Stream-K (CU-masked)", "streamk", sz, args.steps, cus_list,
                                use_mask=True, cooldown=args.cooldown)

    # 6) Stream-K + WS (CU-masked)
    data["streamk_ws"] = sweep_one("Stream-K+WS (CU-masked)", "streamk-ws", sz, args.steps, cus_list,
                                   use_mask=True, cooldown=args.cooldown)

    out_path = f"{DATA_DIR}/cu_sweep_{sz}_v4.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

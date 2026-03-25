#!/usr/bin/env python3
"""
Driver script: collect all data for overlap analysis plots.

Runs everything on a SINGLE GPU (via HIP_VISIBLE_DEVICES) with proper
cooldown between tests. Uses overlap.py for all overlap/distribution data
and subprocess-based torch benchmarks for the CU sweep.

Usage (inside the Docker container):
    python3 benchmarks/collect_all_data.py --gpu 4

Or from the host:
    docker run ... -e HIP_VISIBLE_DEVICES=4 ... python3 benchmarks/collect_all_data.py
"""
import argparse
import json
import os
import subprocess
import statistics
import sys
import time

SIZES = [1024, 2048, 4096, 8192, 12288, 16384]
BACKENDS = ["ws", "torch"]
DATA_DIR = "results/plot_data"
STEPS = 200
WARMUP = 20


def gpu_cooldown(seconds=5):
    """Wait for GPU to idle and cool down between tests."""
    print(f"  [cooldown {seconds}s]", flush=True)
    time.sleep(seconds)


def run_cmd(cmd, env=None, timeout=600):
    """Run a command, print output, return (returncode, stdout)."""
    print(f"  $ {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode != 0:
        print(f"    FAILED (rc={result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"    stderr: {line}")
    return result.returncode, result.stdout


# ===========================================================================
# Phase 1: Overlap data (torchrun, 8 GPUs)
# ===========================================================================

def collect_overlap_data():
    """Collect overlap data for all sizes × backends using overlap.py standard."""
    print("\n" + "=" * 70)
    print("  Phase 1: Overlap Data (torchrun, all GPUs)")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    for backend in BACKENDS:
        for sz in SIZES:
            json_path = f"{DATA_DIR}/{backend}_{sz}.json"
            print(f"\n>>> {backend} {sz}x{sz}x{sz}")

            cmd = [
                "torchrun", "--nproc_per_node=8",
                "benchmarks/overlap.py", "standard",
                "--backend", backend,
                "--gemm-m", str(sz), "--gemm-n", str(sz), "--gemm-k", str(sz),
                "--comm-size", str(sz), str(sz),
                "--collective", "all_reduce",
                "--warmup", str(WARMUP), "--steps", str(STEPS),
                "--output-json", json_path,
            ]
            rc, stdout = run_cmd(cmd, timeout=300)
            if rc == 0 and stdout:
                for line in stdout.strip().split("\n"):
                    if any(kw in line for kw in ["median", "alone", "Overlap", "slowdown", "efficiency"]):
                        print(f"    {line.strip()}")
            gpu_cooldown(8)

    # Assemble into the format make_plots.py expects
    combined = {}
    for backend in BACKENDS:
        for sz in SIZES:
            json_path = f"{DATA_DIR}/{backend}_{sz}.json"
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                key = f"{backend}_{sz}"
                combined[key] = data

    out_path = f"{DATA_DIR}/rotating_overlap.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined overlap data → {out_path}")


# ===========================================================================
# Phase 2: CU Sweep (single GPU, subprocess per mask)
# ===========================================================================

def make_multi_xcd_mask(cus_per_xcd, n_xcds=8, bits_per_xcd=38):
    """Build a ROC_GLOBAL_CU_MASK hex string for MI300X.

    Each XCD occupies `bits_per_xcd` bits in a flat bitfield.
    Setting the first `cus_per_xcd` bits of each block enables
    that many CUs on every XCD.
    """
    per_xcd_mask = (1 << cus_per_xcd) - 1
    full_mask = 0
    for xcd in range(n_xcds):
        full_mask |= per_xcd_mask << (xcd * bits_per_xcd)
    return hex(full_mask)


CU_SWEEP_SCRIPT = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

MODE = sys.argv[1]       # "torch" or "ws"
M = int(sys.argv[2])
STEPS = int(sys.argv[3])
TOTAL_CUS = int(sys.argv[4]) if len(sys.argv) > 4 else 0

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
    sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=False,
                                           total_cus=TOTAL_CUS if TOTAL_CUS > 0 else None)
    cfg = tritonblas.matmul_preamble(sel)
    def bench():
        tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=False, work_stealing=True)
    def reset():
        cfg.reset(streamk=False, work_stealing=True)

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


def collect_cu_sweep(sz):
    """Collect CU sweep data for a given GEMM size."""
    print(f"\n--- CU sweep {sz}x{sz}x{sz} ---")

    sweep_data = {"ws": [], "torch": []}
    steps = 50

    # Write helper script
    script_path = "/tmp/_cu_sweep_worker.py"
    with open(script_path, "w") as f:
        f.write(CU_SWEEP_SCRIPT)

    # --- WS: grid-limited (vary total_cus) ---
    ws_cus_list = sorted(set([304, 280, 256, 240, 224, 200, 176, 152, 128, 96, 64, 32, 16]))
    print(f"  WS grid-limited sweep ({len(ws_cus_list)} points)")
    for total_cus in ws_cus_list:
        rc, stdout = run_cmd(
            [sys.executable, script_path, "ws", str(sz), str(steps), str(total_cus)],
            timeout=120,
        )
        if rc == 0:
            for line in stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med = float(parts[1])
                    cus = int(parts[4])
                    tflops = float(parts[5])
                    sweep_data["ws"].append({"cus": total_cus, "tflops": tflops, "med_ms": med})
                    print(f"    WS cus={total_cus:>4d}: {med:.3f}ms {tflops:.0f}TF")
        gpu_cooldown(3)

    # --- torch: ROC_GLOBAL_CU_MASK (multi-XCD) ---
    # Each XCD has 38 CUs. Sweep cus_per_xcd from 38 down to 1.
    torch_cus_per_xcd = sorted(set([38, 35, 32, 28, 24, 20, 16, 12, 8, 4, 2, 1]), reverse=True)
    print(f"\n  torch CU-masked sweep ({len(torch_cus_per_xcd)} + 1 points)")

    # First: no mask baseline
    rc, stdout = run_cmd(
        [sys.executable, script_path, "torch", str(sz), str(steps)],
        timeout=120,
    )
    if rc == 0:
        for line in stdout.strip().split("\n"):
            if line.startswith("RESULT"):
                parts = line.split()
                med, tflops = float(parts[1]), float(parts[5])
                sweep_data["torch"].append({"cus": 304, "tflops": tflops, "med_ms": med, "mask": "none"})
                print(f"    torch no-mask: {med:.3f}ms {tflops:.0f}TF")
    gpu_cooldown(3)

    for cpx in torch_cus_per_xcd:
        total_cus = cpx * 8
        mask = make_multi_xcd_mask(cpx)
        env = os.environ.copy()
        env["ROC_GLOBAL_CU_MASK"] = mask
        rc, stdout = run_cmd(
            [sys.executable, script_path, "torch", str(sz), str(steps)],
            env=env, timeout=120,
        )
        if rc == 0:
            for line in stdout.strip().split("\n"):
                if line.startswith("RESULT"):
                    parts = line.split()
                    med = float(parts[1])
                    reported_cus = int(parts[4])
                    tflops = float(parts[5])
                    sweep_data["torch"].append({
                        "cus": total_cus, "reported_cus": reported_cus,
                        "tflops": tflops, "med_ms": med,
                        "mask": mask[:20] + "...",
                        "cus_per_xcd": cpx,
                    })
                    print(f"    torch {cpx:>2d}/xcd ({total_cus:>3d} CUs, reported={reported_cus:>3d}): "
                          f"{med:.3f}ms {tflops:.0f}TF")
        gpu_cooldown(3)

    return sweep_data


def collect_cu_sweeps():
    """Collect CU sweep data for both 8K and 4K."""
    print("\n" + "=" * 70)
    print("  Phase 2: CU Sweep (single GPU)")
    print("=" * 70)

    for sz in [8192, 4096]:
        data = collect_cu_sweep(sz)
        out_path = f"{DATA_DIR}/cu_sweep_{sz}_v3.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved → {out_path}")
        gpu_cooldown(10)


# ===========================================================================
# Phase 3: Distribution data (already captured in Phase 1 JSON)
# ===========================================================================

def assemble_distribution_data():
    """Extract 8K distribution data from Phase 1 JSON files."""
    print("\n" + "=" * 70)
    print("  Phase 3: Assemble distribution data")
    print("=" * 70)

    dist_data = {}
    for backend in BACKENDS:
        json_path = f"{DATA_DIR}/{backend}_8192.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            dist_data[f"{backend}_alone"] = d.get("alone_all", [])
            dist_data[f"{backend}_rotating"] = d.get("rotating_all", [])
            dist_data[f"{backend}_overlap"] = d.get("overlap_all", [])
            dist_data[f"{backend}_rot_overlap"] = d.get("rot_overlap_mm_all", [])

    out_path = f"{DATA_DIR}/distribution_8k.json"
    with open(out_path, "w") as f:
        json.dump(dist_data, f, indent=2)
    print(f"  Saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect all plot data")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (sets HIP_VISIBLE_DEVICES)")
    parser.add_argument("--skip-overlap", action="store_true",
                        help="Skip Phase 1 (overlap data)")
    parser.add_argument("--skip-cusweep", action="store_true",
                        help="Skip Phase 2 (CU sweep)")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="Steps per benchmark")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu} (HIP_VISIBLE_DEVICES={args.gpu})")

    globals()["STEPS"] = args.steps

    os.makedirs(DATA_DIR, exist_ok=True)

    if not args.skip_overlap:
        collect_overlap_data()

    if not args.skip_cusweep:
        collect_cu_sweeps()

    assemble_distribution_data()

    print("\n" + "=" * 70)
    print("  All data collected! Run: python3 benchmarks/make_plots.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

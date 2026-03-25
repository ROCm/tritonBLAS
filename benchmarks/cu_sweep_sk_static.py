#!/usr/bin/env python3
"""
CU sweep for STATIC StreamK (non-WS, pid-based, knows NUM_SMS = grid).
This is the "cheating" roofline: SK knows exactly how many CUs are active.
Also runs a WS+static-SK hybrid for direct comparison.
"""
import json
import os
import subprocess
import sys
import time

DATA_DIR = "results/plot_data"

WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas

MODE = sys.argv[1]  # "streamk-static" or "ws-sk-static"
M = int(sys.argv[2])
STEPS = int(sys.argv[3])
GRID = int(sys.argv[4])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)

sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev,
                                        streamk=True, total_cus=GRID)
cfg = tritonblas.matmul_preamble(sel)

if MODE == "streamk-static":
    # Pure static StreamK (no WS): uses streamk_matmul kernel
    def bench():
        tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=True, work_stealing=False)
    def reset():
        cfg.reset(streamk=True, work_stealing=False)
else:
    # WS + static SK hybrid via matmul_lt (uses ws_streamk_matmul)
    def bench():
        tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=True, work_stealing=True)
    def reset():
        cfg.reset(streamk=True, work_stealing=True)

for _ in range(20):
    reset(); bench()
torch.cuda.synchronize()

t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); bench(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))

med = statistics.median(t)
flops = 2.0 * M**3
tflops = flops / (med * 1e-3) / 1e12

import triton
BLK_K = sel.block_k
total_tiles = (M // sel.block_m) * (M // sel.block_n)
grid_actual = min(GRID, total_tiles)
sk_tiles = total_tiles % grid_actual
print(f"RESULT {med:.4f} {tflops:.1f} {grid_actual} {sk_tiles}")
'''


def main():
    script = "/tmp/_sk_static_worker.py"
    with open(script, "w") as f:
        f.write(WORKER)

    cus_list = [c * 8 for c in [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]
    steps = 50
    n_gpus = 8

    for sz in [8192, 4096]:
        total_tiles = (sz // 256) ** 2

        print(f"\n{'='*70}")
        print(f"  Static StreamK roofline: {sz}x{sz} ({total_tiles} tiles)")
        print(f"{'='*70}")

        results_static = []
        for wave_start in range(0, len(cus_list), n_gpus):
            wave = cus_list[wave_start:wave_start + n_gpus]
            procs = []
            for i, cus in enumerate(wave):
                gpu = i
                grid = min(cus, total_tiles)
                env = os.environ.copy()
                env["HIP_VISIBLE_DEVICES"] = str(gpu)
                p = subprocess.Popen(
                    [sys.executable, script, "streamk-static", str(sz), str(steps), str(grid)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env=env,
                )
                procs.append((cus, grid, p))

            for cus, grid, p in procs:
                try:
                    stdout, stderr = p.communicate(timeout=120)
                    if p.returncode == 0:
                        for line in stdout.strip().split("\n"):
                            if line.startswith("RESULT"):
                                parts = line.split()
                                med, tflops = float(parts[1]), float(parts[2])
                                sk_tiles = int(parts[4])
                                results_static.append({"cus": cus, "tflops": tflops, "med_ms": med})
                                print(f"  {cus:>4d} CUs (grid={grid}, sk={sk_tiles}): "
                                      f"{med:.3f}ms {tflops:.0f}TF")
                    else:
                        print(f"  {cus:>4d} CUs: FAILED")
                except subprocess.TimeoutExpired:
                    p.kill()
                    print(f"  {cus:>4d} CUs: TIMEOUT")
            time.sleep(3)

        v4_path = f"{DATA_DIR}/cu_sweep_{sz}_v4.json"
        if os.path.exists(v4_path):
            with open(v4_path) as f:
                data = json.load(f)
        else:
            data = {}
        data["streamk_static"] = results_static
        with open(v4_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved 'streamk_static' -> {v4_path}")


if __name__ == "__main__":
    main()

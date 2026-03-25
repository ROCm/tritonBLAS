#!/usr/bin/env python3
"""Run StreamK+WS in grid-limited mode (total_cus parameter, no CU mask) and merge into v4 data."""
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

M = int(sys.argv[1])
STEPS = int(sys.argv[2])
TOTAL_CUS = int(sys.argv[3])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)

sel = tritonblas.OrigamiMatmulSelector(
    M, M, M, dtype, dtype, dtype, dev, streamk=True, total_cus=TOTAL_CUS)
cfg = tritonblas.matmul_preamble(sel)
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
cus = torch.cuda.get_device_properties(0).multi_processor_count
flops = 2.0 * M**3
tflops = flops / (med * 1e-3) / 1e12
print(f"RESULT {med:.4f} {min(t):.4f} {max(t):.4f} {cus} {tflops:.1f}")
'''


def main():
    script = "/tmp/_skws_gl_worker.py"
    with open(script, "w") as f:
        f.write(WORKER)

    cus_list = [c * 8 for c in [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]
    steps = 50
    n_gpus = 8

    for sz in [8192, 4096]:
        print(f"\n{'='*60}")
        print(f"  StreamK+WS grid-limited: {sz}x{sz}")
        print(f"{'='*60}")

        results = []
        for wave_start in range(0, len(cus_list), n_gpus):
            wave = cus_list[wave_start:wave_start + n_gpus]
            procs = []
            for i, cus in enumerate(wave):
                gpu = i
                env = os.environ.copy()
                env["HIP_VISIBLE_DEVICES"] = str(gpu)
                p = subprocess.Popen(
                    [sys.executable, script, str(sz), str(steps), str(cus)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env=env,
                )
                procs.append((cus, p))

            for cus, p in procs:
                try:
                    stdout, stderr = p.communicate(timeout=120)
                    if p.returncode == 0:
                        for line in stdout.strip().split("\n"):
                            if line.startswith("RESULT"):
                                parts = line.split()
                                med = float(parts[1])
                                tflops = float(parts[5])
                                results.append({"cus": cus, "tflops": tflops, "med_ms": med})
                                print(f"  {cus:>4d} CUs (grid={cus}): {med:.3f}ms {tflops:.0f}TF")
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
        data["streamk_ws_grid"] = results
        with open(v4_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Merged as 'streamk_ws_grid' → {v4_path}")


if __name__ == "__main__":
    main()

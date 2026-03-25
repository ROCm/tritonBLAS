#!/usr/bin/env python3
"""CU sweep: WS persistent with GLOBAL_ATOMIC=True (single counter, no slot starvation)."""
import json, os, subprocess, sys, time
DATA_DIR = "results/plot_data"
WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
M = int(sys.argv[1]); STEPS = int(sys.argv[2]); GRID = int(sys.argv[3])
torch.cuda.set_device(0); dev = torch.device("cuda", 0); dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)
sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=False, total_cus=GRID)
cfg = tritonblas.matmul_preamble(sel)
cfg.global_atomic = True
def bench(): tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=False, work_stealing=True)
def reset(): cfg.reset(streamk=False, work_stealing=True)
for _ in range(20): reset(); bench()
torch.cuda.synchronize()
t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); bench(); e.record(); torch.cuda.synchronize(); t.append(s.elapsed_time(e))
med = statistics.median(t); tflops = 2.0*M**3/(med*1e-3)/1e12
print(f"RESULT {med:.4f} {tflops:.1f}")
'''
def main():
    script = "/tmp/_ga_worker.py"
    with open(script, "w") as f: f.write(WORKER)
    cus_list = [c*8 for c in [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]
    for sz in [8192, 4096]:
        print(f"\n  WS GLOBAL_ATOMIC: {sz}x{sz}")
        results = []
        for wave_start in range(0, len(cus_list), 8):
            wave = cus_list[wave_start:wave_start+8]
            procs = []
            for i, cus in enumerate(wave):
                env = os.environ.copy(); env["HIP_VISIBLE_DEVICES"] = str(i)
                p = subprocess.Popen([sys.executable, script, str(sz), "50", str(cus)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                procs.append((cus, p))
            for cus, p in procs:
                try:
                    stdout, _ = p.communicate(timeout=120)
                    for line in stdout.strip().split("\n"):
                        if line.startswith("RESULT"):
                            med, tflops = float(line.split()[1]), float(line.split()[2])
                            results.append({"cus": cus, "tflops": tflops, "med_ms": med})
                            print(f"    {cus:>4d} CUs: {med:.3f}ms {tflops:.0f}TF")
                except: print(f"    {cus:>4d} CUs: FAILED")
            time.sleep(3)
        v4 = f"{DATA_DIR}/cu_sweep_{sz}_v4.json"
        if os.path.exists(v4):
            with open(v4) as f: data = json.load(f)
        else: data = {}
        data["ws_global_atomic"] = results
        with open(v4, "w") as f: json.dump(data, f, indent=2)
        print(f"  Saved 'ws_global_atomic' → {v4}")
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Collect torch (CU-masked) and Static SK (grid-limited) data for sizes missing from v4."""
import json, os, subprocess, sys, time

DATA_DIR = "results/plot_data"

def make_multi_xcd_mask(cus_per_xcd, n_xcds=8, bits_per_xcd=38):
    per_xcd = (1 << cus_per_xcd) - 1
    full = 0
    for xcd in range(n_xcds):
        full |= per_xcd << (xcd * bits_per_xcd)
    return hex(full)

TORCH_WORKER = r'''
import torch, statistics, sys
M = int(sys.argv[1]); STEPS = int(sys.argv[2])
torch.cuda.set_device(0)
A = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
B = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
for _ in range(20): torch.matmul(A, B)
torch.cuda.synchronize()
t = []
for _ in range(STEPS):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); torch.matmul(A, B); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))
med = statistics.median(t)
print(f"RESULT {med:.4f} {2.0*M**3/(med*1e-3)/1e12:.1f}")
'''

SK_WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
M = int(sys.argv[1]); STEPS = int(sys.argv[2]); GRID = int(sys.argv[3])
torch.cuda.set_device(0)
A = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
B = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
C = torch.empty(M, M, dtype=torch.bfloat16, device="cuda")
sel = tritonblas.OrigamiMatmulSelector(M, M, M, torch.bfloat16, torch.bfloat16, torch.bfloat16,
    torch.device("cuda", 0), streamk=True, total_cus=GRID)
cfg = tritonblas.matmul_preamble(sel)
def run(): tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=True, work_stealing=False)
def reset(): cfg.reset(streamk=True, work_stealing=False)
for _ in range(20): reset(); run()
torch.cuda.synchronize()
t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))
med = statistics.median(t)
print(f"RESULT {med:.4f} {2.0*M**3/(med*1e-3)/1e12:.1f}")
'''

def main():
    torch_script = "/tmp/_torch_w.py"
    sk_script = "/tmp/_sk_w.py"
    with open(torch_script, "w") as f: f.write(TORCH_WORKER)
    with open(sk_script, "w") as f: f.write(SK_WORKER)

    cus_per_xcd = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]
    cus_list = [c*8 for c in cus_per_xcd]
    steps = 30

    for M in [2048, 12288, 16384]:
        total_tiles = (M // (256 if M >= 4096 else 128)) ** 2
        print(f"\n===== {M}x{M} ({total_tiles} tiles) =====")

        torch_results = []
        sk_results = []

        for wave_start in range(0, len(cus_list), 8):
            wave = cus_list[wave_start:wave_start+8]
            procs = []
            for i, cus in enumerate(wave):
                gpu = i
                cpx = cus // 8
                mask = make_multi_xcd_mask(cpx)
                grid = min(cus, total_tiles)

                env_torch = os.environ.copy()
                env_torch["HIP_VISIBLE_DEVICES"] = str(gpu)
                env_torch["ROC_GLOBAL_CU_MASK"] = mask
                pt = subprocess.Popen([sys.executable, torch_script, str(M), str(steps)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_torch)

                env_sk = os.environ.copy()
                env_sk["HIP_VISIBLE_DEVICES"] = str(gpu)
                ps = subprocess.Popen([sys.executable, sk_script, str(M), str(steps), str(grid)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_sk)

                procs.append((cus, grid, pt, ps))

            for cus, grid, pt, ps in procs:
                for label, p, results in [("torch", pt, torch_results), ("SK", ps, sk_results)]:
                    try:
                        stdout, _ = p.communicate(timeout=120)
                        for line in stdout.strip().split("\n"):
                            if line.startswith("RESULT"):
                                med, tflops = float(line.split()[1]), float(line.split()[2])
                                results.append({"cus": cus, "tflops": tflops, "med_ms": med})
                    except:
                        pass
                cpx = cus // 8
                t_tf = torch_results[-1]["tflops"] if torch_results and torch_results[-1]["cus"] == cus else 0
                s_tf = sk_results[-1]["tflops"] if sk_results and sk_results[-1]["cus"] == cus else 0
                print(f"  {cus:>4d} CUs: torch={t_tf:.0f}TF  SK={s_tf:.0f}TF")
            time.sleep(3)

        # Merge into ga.json
        ga_path = f"{DATA_DIR}/cu_sweep_{M}_ga.json"
        if os.path.exists(ga_path):
            with open(ga_path) as f: data = json.load(f)
        else:
            data = {}
        data["torch"] = torch_results
        data["streamk_static"] = sk_results
        with open(ga_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved → {ga_path}")

if __name__ == "__main__":
    main()

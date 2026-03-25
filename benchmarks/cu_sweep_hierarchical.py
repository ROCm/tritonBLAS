#!/usr/bin/env python3
"""CU sweep: Hierarchical WS (per-XCD + global fallback)."""
import json, os, subprocess, sys, time

DATA_DIR = "results/plot_data"

WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul
import triton

M = int(sys.argv[1]); STEPS = int(sys.argv[2]); GRID = int(sys.argv[3])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)

sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev,
                                        streamk=False, total_cus=GRID)
BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
total_tiles = (M // BLK_M) * (M // BLK_N)
num_xcds = sel.num_sms
gsize_m = sel.group_m
n_cu = sel._N_CU
even_k = M % BLK_K == 0

# Tile split: 90% local, 10% global (clamped to tile boundaries)
local_per_xcd = (total_tiles * 9) // (num_xcds * 10)
local_per_xcd = max(local_per_xcd, 1)
total_local = local_per_xcd * num_xcds
global_tiles = total_tiles - total_local

# Buffers: per-XCD counters + global counter
tile_counter = torch.zeros(num_xcds * COUNTER_STRIDE, device=dev, dtype=torch.int32)
global_counter = torch.zeros(COUNTER_STRIDE, device=dev, dtype=torch.int32)
mask = torch.ones(n_cu, dtype=torch.int32, device=dev)
if GRID < n_cu:
    mask[GRID:] = 0

chunk_size = min(gsize_m * gsize_m, max(1, n_cu // num_xcds))

def run():
    ws_hierarchical_matmul[(n_cu,)](
        A, B, C, None, None, None,
        tile_counter, global_counter,
        M, M, M,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
        LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
        COUNTER_STRIDE=COUNTER_STRIDE,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
    )

def reset():
    tile_counter.zero_()
    global_counter.zero_()

# Verify correctness
C_ref = torch.matmul(A, B); torch.cuda.synchronize()
reset(); C.zero_(); run(); torch.cuda.synchronize()
nz = (C.abs() > 0).sum().item()
if nz < M*M * 0.99:
    print(f"FAIL nz={nz}/{M*M}")
    sys.exit(1)

for _ in range(15): reset(); run()
torch.cuda.synchronize()
t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))
med = statistics.median(t)
tflops = 2.0*M**3/(med*1e-3)/1e12
print(f"RESULT {med:.4f} {tflops:.1f}")
'''

def main():
    script = "/tmp/_hier_worker.py"
    with open(script, "w") as f: f.write(WORKER)

    cus_list = [c*8 for c in [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]

    for M in [2048, 4096, 8192, 12288, 16384]:
        print(f"\n  Hierarchical WS: {M}x{M}")
        results = []
        for wave_start in range(0, len(cus_list), 8):
            wave = cus_list[wave_start:wave_start+8]
            procs = []
            for i, cus in enumerate(wave):
                env = os.environ.copy(); env["HIP_VISIBLE_DEVICES"] = str(i)
                p = subprocess.Popen([sys.executable, script, str(M), "30", str(cus)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                procs.append((cus, p))
            for cus, p in procs:
                try:
                    stdout, stderr = p.communicate(timeout=120)
                    for line in stdout.strip().split("\n"):
                        if line.startswith("RESULT"):
                            med, tflops = float(line.split()[1]), float(line.split()[2])
                            results.append({"cus": cus, "tflops": tflops, "med_ms": med})
                            print(f"    {cus:>4d} CUs: {med:.3f}ms {tflops:.0f}TF")
                        elif line.startswith("FAIL"):
                            print(f"    {cus:>4d} CUs: {line}")
                except: print(f"    {cus:>4d} CUs: TIMEOUT/ERROR")
            time.sleep(3)

        ga_path = f"{DATA_DIR}/cu_sweep_{M}_ga.json"
        if os.path.exists(ga_path):
            with open(ga_path) as f: data = json.load(f)
        else: data = {}
        data["ws_hierarchical"] = results
        with open(ga_path, "w") as f: json.dump(data, f, indent=2)
        print(f"  Saved 'ws_hierarchical' → {ga_path}")

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""
CU sweep for SK+WS with StreamK ENABLED (STREAMK_TILES > 0).
Grid-limited: grid = total_cus, STREAMK_TILES = total_tiles % grid.
The SK phase uses dynamic chunk-K stealing to handle remainder tiles.
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
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels import ws_streamk_matmul
import triton

M = int(sys.argv[1])
STEPS = int(sys.argv[2])
GRID = int(sys.argv[3])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)
C = torch.empty(M, M, dtype=dtype, device=dev)

sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=True,
                                        total_cus=GRID)
cfg = tritonblas.matmul_preamble(sel)

BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
total_tiles = (M // BLK_M) * (M // BLK_N)
grid = min(GRID, total_tiles)
sk_tiles = total_tiles % grid
ipt = triton.cdiv(M, BLK_K)
sk_chunk = ipt  # whole-tile stealing for SK portion

gsize_m = sel.group_m
num_xcds = sel.num_sms
chunk_size = min(gsize_m * gsize_m, max(1, grid // num_xcds))
even_k = M % BLK_K == 0
n_cu = sel._ACTIVE_CU

def run():
    ws_streamk_matmul[(grid,)](
        A, B, C, None, None, None,
        cfg.tile_counter, cfg.sk_iter_counter, cfg.sk_P, cfg.sk_locks, cfg.sk_done,
        M, M, M, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
        COUNTERS_PER_XCD=sel.COUNTERS_PER_XCD, COUNTER_STRIDE=COUNTER_STRIDE,
        CHUNK_SIZE=chunk_size, STREAMK_TILES=sk_tiles,
        SK_CHUNK_ITERS=sk_chunk,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False, GLOBAL_ATOMIC=cfg.global_atomic,
        num_stages=2, num_warps=8, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1, mask_ptr=cfg.mask,
    )

def reset():
    cfg.reset(streamk=True, work_stealing=True)

for _ in range(20):
    reset(); run()
torch.cuda.synchronize()

t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))

med = statistics.median(t)
flops = 2.0 * M**3
tflops = flops / (med * 1e-3) / 1e12
print(f"RESULT {med:.4f} {tflops:.1f} {grid} {sk_tiles} {ipt}")
'''


def main():
    script = "/tmp/_skws_sken_worker.py"
    with open(script, "w") as f:
        f.write(WORKER)

    cus_list = [c * 8 for c in [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]
    steps = 50
    n_gpus = 8

    for sz in [8192, 4096]:
        print(f"\n{'='*60}")
        print(f"  SK+WS (SK ENABLED) grid-limited: {sz}x{sz}")
        print(f"{'='*60}")

        total_tiles = (sz // 256) ** 2
        results = []
        for wave_start in range(0, len(cus_list), n_gpus):
            wave = cus_list[wave_start:wave_start + n_gpus]
            procs = []
            for i, cus in enumerate(wave):
                gpu = i
                env = os.environ.copy()
                env["HIP_VISIBLE_DEVICES"] = str(gpu)
                grid = min(cus, total_tiles)
                p = subprocess.Popen(
                    [sys.executable, script, str(sz), str(steps), str(grid)],
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
                                med = float(parts[1])
                                tflops = float(parts[2])
                                actual_grid = int(parts[3])
                                sk_tiles = int(parts[4])
                                results.append({"cus": cus, "tflops": tflops, "med_ms": med})
                                print(f"  {cus:>4d} CUs (grid={actual_grid}, sk_tiles={sk_tiles}): "
                                      f"{med:.3f}ms {tflops:.0f}TF")
                    else:
                        print(f"  {cus:>4d} CUs: FAILED")
                        if stderr:
                            print(f"    {stderr[-200:]}")
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
        data["streamk_ws_sk_enabled"] = results
        with open(v4_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Merged as 'streamk_ws_sk_enabled' → {v4_path}")


if __name__ == "__main__":
    main()

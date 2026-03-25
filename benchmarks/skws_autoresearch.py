#!/usr/bin/env python3
"""
Scalable autoresearch: parallel SK+WS optimization across 8 GPUs.

Each GPU tests a different configuration variant simultaneously.
Reports correctness + timing for each.
"""
import json
import os
import subprocess
import sys
import statistics
import time

WORKER = r'''
import torch, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

EXPERIMENT = sys.argv[1]
M = int(sys.argv[2])
STEPS = int(sys.argv[3])

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16
A = torch.randn(M, M, dtype=dtype, device=dev)
B = torch.randn(M, M, dtype=dtype, device=dev)

C_ref = torch.matmul(A, B)
torch.cuda.synchronize()

import tritonblas
from tritonblas.config import COUNTER_STRIDE, MAX_SK_TILES
import triton

C = torch.empty(M, M, dtype=dtype, device=dev)
sel = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=True)
cfg = tritonblas.matmul_preamble(sel)

BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
total_tiles = (M // BLK_M) * (M // BLK_N)
n_cu = sel._ACTIVE_CU
grid = min(n_cu, total_tiles)
sk_tiles = total_tiles % grid
ipt = triton.cdiv(M, BLK_K)

if EXPERIMENT.startswith("chunk_"):
    divisor = int(EXPERIMENT.split("_")[1])
    sk_chunk = max(1, ipt // divisor)
elif EXPERIMENT == "whole_tile":
    sk_chunk = ipt
elif EXPERIMENT == "quarter":
    sk_chunk = max(1, ipt // 4)
elif EXPERIMENT == "eighth":
    sk_chunk = max(1, ipt // 8)
else:
    sk_chunk = max(1, ipt // 2)

from tritonblas.kernels import ws_streamk_matmul
from tritonblas.kernels.stages.indexing.pid_transforms import chiplet_transform_chunked

gsize_m = sel.group_m
num_xcds = sel.num_sms
chunk_size = min(gsize_m * gsize_m, grid // num_xcds)
even_k = M % BLK_K == 0

def run_kernel():
    ws_streamk_matmul[(grid,)](
        A, B, C,
        None, None, None,
        cfg.tile_counter,
        cfg.sk_iter_counter,
        cfg.sk_P,
        cfg.sk_locks,
        cfg.sk_done,
        M, M, M,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
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
    reset(); run_kernel()
torch.cuda.synchronize()

reset(); C.zero_(); run_kernel(); torch.cuda.synchronize()
nonzero = (C.abs() > 0).sum().item()
diff = (C.float() - C_ref.float()).abs()
mean_rel = (diff / (C_ref.float().abs() + 1e-8)).mean().item()

t = []
for _ in range(STEPS):
    reset(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); run_kernel(); e.record(); torch.cuda.synchronize()
    t.append(s.elapsed_time(e))

med = statistics.median(t)
flops = 2.0 * M**3
tflops = flops / (med * 1e-3) / 1e12
total = M * M
correct = nonzero == total and mean_rel < 0.01
print(f"RESULT {EXPERIMENT} {med:.4f} {tflops:.1f} {nonzero}/{total} {mean_rel:.6f} "
      f"{'PASS' if correct else 'FAIL'} sk_tiles={sk_tiles} sk_chunk={sk_chunk} ipt={ipt}")
'''


def main():
    script = "/tmp/_skws_ar_worker.py"
    with open(script, "w") as f:
        f.write(WORKER)

    M = 8192
    steps = 30
    n_gpus = 8

    experiments = [
        "whole_tile",   # SK_CHUNK = iters_per_tile (no K-split, no reduction)
        "chunk_2",      # SK_CHUNK = ipt // 2  (current default)
        "chunk_3",      # SK_CHUNK = ipt // 3
        "chunk_4",      # SK_CHUNK = ipt // 4
        "chunk_8",      # SK_CHUNK = ipt // 8
        "chunk_16",     # SK_CHUNK = ipt // 16
        "chunk_32",     # SK_CHUNK = ipt // 32
        "chunk_1",      # SK_CHUNK = ipt (= whole_tile, sanity check)
    ]

    print(f"{'='*80}")
    print(f"  SK+WS Autoresearch: {M}x{M} BF16, parallel across {n_gpus} GPUs")
    print(f"{'='*80}")
    print(f"{'Experiment':<16} {'ms':>8} {'TFLOPS':>8} {'Correct':>8} {'SK_tiles':>9} {'Chunk':>6}")
    print("-" * 65)

    procs = []
    for i, exp in enumerate(experiments):
        gpu = i % n_gpus
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.Popen(
            [sys.executable, script, exp, str(M), str(steps)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        procs.append((exp, p))

    results = []
    for exp, p in procs:
        try:
            stdout, stderr = p.communicate(timeout=120)
            if p.returncode == 0:
                for line in stdout.strip().split("\n"):
                    if line.startswith("RESULT"):
                        parts = line.split(maxsplit=6)
                        name = parts[1]
                        med = float(parts[2])
                        tflops = float(parts[3])
                        correct = parts[5]
                        extra = parts[6] if len(parts) > 6 else ""
                        sk_info = {kv.split("=")[0]: kv.split("=")[1]
                                   for kv in extra.split() if "=" in kv}
                        sk_tiles = sk_info.get("sk_tiles", "?")
                        sk_chunk = sk_info.get("sk_chunk", "?")
                        print(f"{name:<16} {med:>8.3f} {tflops:>8.0f} {correct:>8} {sk_tiles:>9} {sk_chunk:>6}")
                        results.append({"name": name, "ms": med, "tflops": tflops, "correct": correct})
            else:
                err = (stderr or "")[-200:]
                print(f"{exp:<16} {'FAIL':>8} {'---':>8} {'FAIL':>8}")
                if err:
                    print(f"  {err[:100]}")
        except subprocess.TimeoutExpired:
            p.kill()
            print(f"{exp:<16} {'TIMEOUT':>8}")

    print()
    if results:
        passing = [r for r in results if r["correct"] == "PASS"]
        if passing:
            best = max(passing, key=lambda r: r["tflops"])
            print(f"Best: {best['name']} at {best['tflops']:.0f} TF ({best['ms']:.3f}ms)")
        else:
            print("No passing experiments!")


if __name__ == "__main__":
    main()

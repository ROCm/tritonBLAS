#!/usr/bin/env python3
"""Single CU-mask point for torch.matmul. Called by cu_sweep_driver.py.

Usage: ROC_GLOBAL_CU_MASK=0x... python3 cu_sweep_torch.py <M> <N> <K>
"""
import torch
import statistics
import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
FLOPS = 2.0 * M * N * K

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

for _ in range(20):
    torch.matmul(A, B, out=C)
torch.cuda.synchronize()

times = []
for _ in range(30):
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    torch.matmul(A, B, out=C)
    en.record()
    torch.cuda.synchronize()
    times.append(st.elapsed_time(en))

med = statistics.median(times)
tflops = FLOPS / (med * 1e-3) / 1e12
print(f"{med:.4f} {tflops:.2f}")

#!/usr/bin/env python3
"""Autoresearch: sweep num_stages, waves_per_eu, num_warps for WS kernel.

Directly calls ws_persistent_matmul with varied launch parameters.
"""
import torch
import tritonblas
from tritonblas.kernels import ws_persistent_matmul
from tritonblas.config import COUNTER_STRIDE
import statistics
import itertools

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)

M = N = K = 8192
dtype = torch.bfloat16
FLOPS = 2.0 * M * N * K

A = torch.randn(M, K, dtype=dtype, device=dev)
B = torch.randn(K, N, dtype=dtype, device=dev)
C = torch.empty(M, N, dtype=dtype, device=dev)
ref = torch.matmul(A.float(), B.float()).bfloat16()

sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
cfg = tritonblas.matmul_preamble(sel)

BLK_M, BLK_N, BLK_K = sel.block_m, sel.block_n, sel.block_k
gsize_m = sel.group_m
num_xcds = sel.num_sms
even_k = K % BLK_K == 0
grids = sel._hardware.N_CU
chunk_size = min(gsize_m * gsize_m, max(1, 1024 // num_xcds))
mask = torch.ones(sel._N_CU, dtype=torch.int32, device=dev)

print(f"Tile config: {BLK_M}x{BLK_N}x{BLK_K}, group_m={gsize_m}, "
      f"grid={grids}, xcds={num_xcds}, counters/xcd={sel.COUNTERS_PER_XCD}")
print()

def launch_ws(ns, nw, wpe):
    ws_persistent_matmul[(grids,)](
        A, B, C, None, None, None, cfg.tile_counter,
        M, N, K,
        A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
        stride_ak=A.stride(1), stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m, NUM_SMS=grids, NUM_XCDS=num_xcds,
        COUNTERS_PER_XCD=sel.COUNTERS_PER_XCD,
        COUNTER_STRIDE=COUNTER_STRIDE,
        BIAS=False, EVEN_K=even_k,
        CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
        QUANTIZED=False, GLOBAL_ATOMIC=False,
        num_stages=ns, num_warps=nw, waves_per_eu=wpe,
        matrix_instr_nonkdim=16, kpack=1,
        mask_ptr=mask,
    )

s = torch.cuda.Stream(device=dev)

# torch baseline
for _ in range(20):
    torch.matmul(A, B, out=C)
torch.cuda.synchronize()
times = []
for _ in range(50):
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    torch.matmul(A, B, out=C)
    en.record()
    torch.cuda.synchronize()
    times.append(st.elapsed_time(en))
torch_ms = statistics.median(times)
torch_tflops = FLOPS / (torch_ms * 1e-3) / 1e12
print(f"torch.matmul: {torch_ms:.3f} ms  ({torch_tflops:.1f} TFLOPS)")

# WS default baseline
for _ in range(20):
    with torch.cuda.stream(s):
        cfg.reset(work_stealing=True)
        launch_ws(2, 8, 0)
torch.cuda.synchronize()
times = []
for _ in range(50):
    with torch.cuda.stream(s):
        cfg.reset(work_stealing=True)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record(s)
    with torch.cuda.stream(s):
        launch_ws(2, 8, 0)
    en.record(s)
    torch.cuda.synchronize()
    times.append(st.elapsed_time(en))
ws_default_ms = statistics.median(times)
ws_default_tflops = FLOPS / (ws_default_ms * 1e-3) / 1e12
print(f"WS default (stages=2 warps=8 waves=0): {ws_default_ms:.3f} ms  ({ws_default_tflops:.1f} TFLOPS)")
print()

STAGES = [1, 2, 3, 4]
WAVES = [0, 1, 2]
WARPS = [4, 8]

results = []
total = len(STAGES) * len(WAVES) * len(WARPS)
done = 0

for ns, wpe, nw in itertools.product(STAGES, WAVES, WARPS):
    done += 1
    label = f"stages={ns} waves={wpe} warps={nw}"

    try:
        for _ in range(5):
            with torch.cuda.stream(s):
                cfg.reset(work_stealing=True)
                launch_ws(ns, nw, wpe)
        torch.cuda.synchronize()

        cfg.reset(work_stealing=True)
        launch_ws(ns, nw, wpe)
        torch.cuda.synchronize()
        err = (C.float() - ref.float()).abs().max().item()
        if err > 10:
            print(f"  [{done}/{total}] {label}: INCORRECT (err={err:.1f})")
            results.append((label, None, None, "INCORRECT"))
            continue

        times = []
        for _ in range(30):
            with torch.cuda.stream(s):
                cfg.reset(work_stealing=True)
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(s)
            with torch.cuda.stream(s):
                launch_ws(ns, nw, wpe)
            en.record(s)
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en))

        med = statistics.median(times)
        tflops = FLOPS / (med * 1e-3) / 1e12
        vs_torch = med / torch_ms
        vs_default = med / ws_default_ms
        beat = " ***" if med < torch_ms else ""
        print(f"  [{done}/{total}] {label}: {med:.3f} ms ({tflops:.1f} TF) "
              f"vs torch {vs_torch:.3f}x  vs default {vs_default:.3f}x{beat}")
        results.append((label, med, tflops, "OK"))

    except Exception as e:
        short = str(e)[:100]
        print(f"  [{done}/{total}] {label}: FAILED ({short})")
        results.append((label, None, None, f"FAILED"))

print()
print("=" * 85)
print(f"  RESULTS — torch: {torch_ms:.3f}ms ({torch_tflops:.1f} TF) | "
      f"WS default: {ws_default_ms:.3f}ms ({ws_default_tflops:.1f} TF)")
print("=" * 85)
valid = [(l, m, t, s) for l, m, t, s in results if m is not None]
valid.sort(key=lambda x: x[1])
for i, (label, med, tflops, _) in enumerate(valid):
    vs_t = med / torch_ms
    vs_d = med / ws_default_ms
    beat = "  <<< BEATS TORCH" if med < torch_ms else ""
    best = "  <<< BEST WS" if i == 0 else ""
    print(f"  {i+1:>2}. {label:<35} {med:.3f} ms  {tflops:.1f} TF  "
          f"vs_torch={vs_t:.3f}x  vs_default={vs_d:.3f}x{beat}{best}")

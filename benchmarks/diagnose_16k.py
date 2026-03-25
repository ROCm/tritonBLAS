#!/usr/bin/env python3
"""Diagnose why 16K WS is slow: wave quantization, atomic contention,
per-tile overhead, and scheduler behavior."""
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16

WARMUP = 10
ITERS = 50
N_ROT = 4

sizes = [4096, 8192, 12288, 16384]

print("=" * 90)
print("Diagnostic: per-size WS breakdown")
print("=" * 90)
print()

# ---- Phase 1: tile geometry and wave analysis ----
print("--- Phase 1: Tile geometry ---")
print(f"{'Size':>6s}  {'BLK':>10s}  {'tiles':>6s}  {'tiles/CU':>9s}  "
      f"{'waves':>6s}  {'last_occ':>9s}  {'tiles/XCD':>10s}  {'atomics/WG':>11s}")
print("-" * 90)

for sz in sizes:
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    BM, BN, BK = sel.block_m, sel.block_n, sel.block_k
    n_cu = sel._N_CU
    num_xcds = sel.num_sms
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    tiles_per_cu = total_tiles / n_cu
    full_waves = total_tiles // n_cu
    last_wave = total_tiles - full_waves * n_cu
    last_occ = last_wave / n_cu * 100 if last_wave > 0 else 100.0
    cus_per_xcd = n_cu // num_xcds
    tiles_per_xcd = total_tiles // num_xcds
    atomics_per_wg = tiles_per_cu * 2  # one to grab, one prefetch

    print(f"{sz:>6d}  {BM:>3d}x{BN:>3d}x{BK:>2d}  {total_tiles:>6d}  {tiles_per_cu:>9.1f}  "
          f"{full_waves + (1 if last_wave else 0):>6d}  {last_occ:>8.1f}%  "
          f"{tiles_per_xcd:>10d}  {atomics_per_wg:>11.1f}")

print()

# ---- Phase 2: per-tile latency (isolate compute from atomic overhead) ----
print("--- Phase 2: Per-tile latency (grid=304, no-mask, rotating buffers) ---")
print(f"{'Size':>6s}  {'total ms':>9s}  {'per-tile us':>12s}  {'K-loop iters':>13s}  "
      f"{'expected ms':>12s}  {'overhead%':>10s}")
print("-" * 80)

for sz in sizes:
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    BM, BN, BK = sel.block_m, sel.block_n, sel.block_k
    n_cu = sel._N_CU
    num_xcds = sel.num_sms
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    even_k = sz % BK == 0
    local_per_xcd = total_tiles // num_xcds
    global_tiles = total_tiles - local_per_xcd * num_xcds

    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]

    tile_counter = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
    global_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)

    def run(idx):
        tile_counter.zero_()
        global_counter.zero_()
        ws_hierarchical_matmul[(n_cu,)](
            As[idx], Bs[idx], Cs[idx], None, None, None,
            tile_counter, global_counter,
            sz, sz, sz, As[idx].stride(0), Bs[idx].stride(1),
            Cs[idx].stride(0), Cs[idx].stride(1), 0,
            stride_ak=As[idx].stride(1), stride_bk=Bs[idx].stride(0),
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=sel.group_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, USE_MASK=False,
        )

    for w in range(WARMUP):
        run(w % N_ROT)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        run(i % N_ROT)
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    med = statistics.median(times)
    per_tile_us = med * 1000 / (total_tiles / n_cu)  # us per tile per WG
    k_iters = triton.cdiv(sz, BK)
    # Theoretical: each tile does k_iters MFMAs of BM*BN*BK FLOPs
    # At ~1300 TF peak BF16, per-tile compute = 2*BM*BN*sz / 1300e12 * 1e6 us
    flops_per_tile = 2.0 * BM * BN * sz
    compute_us = flops_per_tile / (1300e12) * 1e6  # theoretical at peak
    expected_ms = compute_us * (total_tiles / n_cu) / 1000
    overhead = (med - expected_ms) / expected_ms * 100

    print(f"{sz:>6d}  {med:>9.3f}  {per_tile_us:>12.1f}  {k_iters:>13d}  "
          f"{expected_ms:>12.3f}  {overhead:>9.1f}%")

    del As, Bs, Cs
    torch.cuda.empty_cache()

print()

# ---- Phase 3: grid=tiles wave quantization test ----
print("--- Phase 3: grid=tiles — wave quantization visible? ---")
print(f"{'Size':>6s}  {'grid':>6s}  {'med ms':>8s}  {'vs g=304':>9s}  {'last_occ':>9s}")
print("-" * 50)

for sz in sizes:
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    BM, BN, BK = sel.block_m, sel.block_n, sel.block_k
    n_cu = sel._N_CU
    num_xcds = sel.num_sms
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    even_k = sz % BK == 0
    local_per_xcd = total_tiles // num_xcds
    global_tiles = total_tiles - local_per_xcd * num_xcds

    A = torch.randn(sz, sz, dtype=dtype, device=device)
    B = torch.randn(sz, sz, dtype=dtype, device=device)
    C = torch.zeros(sz, sz, dtype=dtype, device=device)

    results_by_grid = {}
    for grid in [n_cu, total_tiles]:
        tile_counter = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        global_counter = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)

        def run_g(g=grid, tc=tile_counter, gc=global_counter):
            tc.zero_()
            gc.zero_()
            ws_hierarchical_matmul[(g,)](
                A, B, C, None, None, None, tc, gc,
                sz, sz, sz, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
                stride_ak=A.stride(1), stride_bk=B.stride(0),
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=sel.group_m, NUM_SMS=g, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
                COUNTER_STRIDE=COUNTER_STRIDE,
                BIAS=False, EVEN_K=even_k,
                CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
                num_stages=2, num_warps=8, waves_per_eu=0,
                matrix_instr_nonkdim=16, kpack=1, USE_MASK=False,
            )

        for _ in range(WARMUP):
            run_g()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            starts[i].record()
            run_g()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        results_by_grid[grid] = statistics.median(times)

    g304 = results_by_grid[n_cu]
    gtiles = results_by_grid[total_tiles]
    last_wave = total_tiles - (total_tiles // n_cu) * n_cu
    last_occ = last_wave / n_cu * 100 if last_wave > 0 else 100.0

    print(f"{sz:>6d}  {n_cu:>6d}  {g304:>8.3f}  {'baseline':>9s}  {'N/A':>9s}")
    print(f"{sz:>6d}  {total_tiles:>6d}  {gtiles:>8.3f}  "
          f"{(gtiles - g304) / g304 * 100:>+8.1f}%  {last_occ:>8.1f}%")

    del A, B, C
    torch.cuda.empty_cache()

print()

# ---- Phase 4: torch tile config comparison ----
print("--- Phase 4: torch.matmul (hipBLASLt) baseline ---")
print(f"{'Size':>6s}  {'med ms':>8s}  {'TF':>8s}")
print("-" * 30)

for sz in sizes:
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    for w in range(WARMUP):
        torch.matmul(As[w % N_ROT], Bs[w % N_ROT])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        torch.matmul(As[i % N_ROT], Bs[i % N_ROT])
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    med = statistics.median(times)
    flops = 2.0 * sz**3
    tf = flops / (med * 1e-3) / 1e12
    print(f"{sz:>6d}  {med:>8.3f}  {tf:>8.1f}")

    del As, Bs
    torch.cuda.empty_cache()

print()
print("--- Phase 5: WS raw gap vs torch (no overlap) ---")
print(f"{'Size':>6s}  {'torch ms':>9s}  {'WS ms':>8s}  {'gap%':>6s}  {'tiles/CU':>9s}")
print("-" * 50)

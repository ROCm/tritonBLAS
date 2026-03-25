#!/usr/bin/env python3
"""Verify StreamK+WS correctness and measurement accuracy."""
import torch
import statistics
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dtype = torch.bfloat16

SIZES = [4096, 8192]

for M in SIZES:
    print(f"\n{'='*70}")
    print(f"  Verifying {M}x{M}x{M} BF16")
    print(f"{'='*70}")

    A = torch.randn(M, M, dtype=dtype, device=dev)
    B = torch.randn(M, M, dtype=dtype, device=dev)

    # 1) Reference: torch.matmul
    C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()

    # 2) StreamK+WS
    C_skws = torch.empty(M, M, dtype=dtype, device=dev)
    sel_skws = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=True)
    cfg_skws = tritonblas.matmul_preamble(sel_skws)
    cfg_skws.reset(streamk=True, work_stealing=True)
    tritonblas.matmul_lt(A, B, C_skws, sel_skws, cfg_skws, enable_streamk=True, work_stealing=True)
    torch.cuda.synchronize()

    # 3) Regular WS (persistent)
    C_ws = torch.empty(M, M, dtype=dtype, device=dev)
    sel_ws = tritonblas.OrigamiMatmulSelector(M, M, M, dtype, dtype, dtype, dev, streamk=False)
    cfg_ws = tritonblas.matmul_preamble(sel_ws)
    cfg_ws.reset(streamk=False, work_stealing=True)
    tritonblas.matmul_lt(A, B, C_ws, sel_ws, cfg_ws, enable_streamk=False, work_stealing=True)
    torch.cuda.synchronize()

    # Correctness check
    def check(name, C_test):
        # BF16 GEMM has limited precision; use relative tolerance
        diff = (C_test.float() - C_ref.float()).abs()
        ref_abs = C_ref.float().abs()
        # Relative error where ref is nonzero
        mask = ref_abs > 1e-6
        rel_err = (diff[mask] / ref_abs[mask])
        max_rel = rel_err.max().item()
        mean_rel = rel_err.mean().item()
        max_abs = diff.max().item()
        # For BF16 8K GEMM, relative error should be < 1% typically
        ok = max_rel < 0.05  # 5% max relative error for BF16
        status = "PASS" if ok else "FAIL"
        print(f"  {name:20s}: max_rel={max_rel:.6f} mean_rel={mean_rel:.6f} "
              f"max_abs={max_abs:.2f} [{status}]")
        return ok

    print("\n--- Correctness ---")
    ok_skws = check("StreamK+WS", C_skws)
    ok_ws = check("WS (persistent)", C_ws)

    # 4) Timing comparison — all three on same inputs
    print("\n--- Timing (50 iterations, median) ---")

    def time_fn(fn, reset_fn, label, steps=50, warmup=20):
        for _ in range(warmup):
            reset_fn()
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(steps):
            reset_fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            t.append(s.elapsed_time(e))
        med = statistics.median(t)
        flops = 2.0 * M**3
        tflops = flops / (med * 1e-3) / 1e12
        print(f"  {label:20s}: {med:.4f}ms  ({tflops:.0f} TF)  "
              f"min={min(t):.4f} max={max(t):.4f}")
        return med, tflops

    # torch.matmul
    time_fn(lambda: torch.matmul(A, B), lambda: None, "torch.matmul")

    # WS persistent
    time_fn(
        lambda: tritonblas.matmul_lt(A, B, C_ws, sel_ws, cfg_ws,
                                      enable_streamk=False, work_stealing=True),
        lambda: cfg_ws.reset(streamk=False, work_stealing=True),
        "WS (persistent)",
    )

    # StreamK+WS
    time_fn(
        lambda: tritonblas.matmul_lt(A, B, C_skws, sel_skws, cfg_skws,
                                      enable_streamk=True, work_stealing=True),
        lambda: cfg_skws.reset(streamk=True, work_stealing=True),
        "StreamK+WS",
    )

    # 5) Check if StreamK+WS is just returning early / not computing
    print("\n--- Sanity: is SK+WS actually computing? ---")
    C_skws.zero_()
    torch.cuda.synchronize()
    cfg_skws.reset(streamk=True, work_stealing=True)
    tritonblas.matmul_lt(A, B, C_skws, sel_skws, cfg_skws,
                          enable_streamk=True, work_stealing=True)
    torch.cuda.synchronize()
    nonzero = (C_skws.abs() > 0).sum().item()
    total = M * M
    print(f"  After zero + compute: {nonzero}/{total} nonzero elements "
          f"({100*nonzero/total:.1f}%)")
    if nonzero < total * 0.99:
        print(f"  WARNING: Only {100*nonzero/total:.1f}% nonzero — kernel may be skipping tiles!")

    # 6) Check what grid/STREAMK_TILES are being used
    print(f"\n--- Grid parameters ---")
    n_cu = sel_skws._ACTIVE_CU
    total_tiles = (M // sel_skws.block_m) * (M // sel_skws.block_n)
    if total_tiles % n_cu == 0:
        grid = n_cu
    else:
        best = 1
        i = 1
        while i * i <= total_tiles:
            if total_tiles % i == 0:
                if i <= n_cu:
                    best = max(best, i)
                c = total_tiles // i
                if c <= n_cu:
                    best = max(best, c)
            i += 1
        grid = best
    sk_tiles = total_tiles % grid
    print(f"  N_CU={n_cu}, total_tiles={total_tiles}, grid={grid}, "
          f"STREAMK_TILES={sk_tiles}, tiles/WG={total_tiles//grid}")

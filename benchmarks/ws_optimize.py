#!/usr/bin/env python3
"""Continuous WS kernel optimization loop.

Runs experiments across all GPUs, keeps improvements, discards regressions.
Follows the scalable-autoresearch approach.
"""
import torch
import torch.multiprocessing as mp
import tritonblas
from tritonblas.kernels import ws_persistent_matmul
from tritonblas.config import COUNTER_STRIDE, matmul_preamble
import statistics
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results/ws_optimize")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def bench_ws(gpu_id, M=8192, warmup=15, steps=50):
    """Benchmark WS kernel on a specific GPU. Returns (ws_ms, torch_ms)."""
    torch.cuda.set_device(gpu_id)
    dev = torch.device("cuda", gpu_id)
    s = torch.cuda.Stream(device=dev)
    N = K = M
    FLOPS = 2.0 * M * N * K

    A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

    # torch
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    t = []
    for _ in range(steps):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        torch.matmul(A, B, out=C)
        en.record()
        torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    torch_ms = statistics.median(t)

    # WS
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
    cfg = matmul_preamble(sel)

    # Correctness
    ref = torch.matmul(A.float(), B.float()).bfloat16()
    cfg.reset(work_stealing=True)
    tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
    torch.cuda.synchronize()
    err = (C.float() - ref.float()).abs().max().item()
    if err > 10:
        return None, torch_ms, err

    for _ in range(warmup):
        with torch.cuda.stream(s):
            cfg.reset(work_stealing=True)
            tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
    torch.cuda.synchronize()

    t = []
    for _ in range(steps):
        with torch.cuda.stream(s):
            cfg.reset(work_stealing=True)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(s)
        with torch.cuda.stream(s):
            tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
        en.record(s)
        torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    ws_ms = statistics.median(t)

    return ws_ms, torch_ms, err


def bench_worker(gpu_id, size, result_dict):
    """Worker function for multiprocessing."""
    try:
        ws_ms, torch_ms, err = bench_ws(gpu_id, M=size)
        result_dict[f"{gpu_id}_{size}"] = {
            "gpu": gpu_id, "size": size,
            "ws_ms": ws_ms, "torch_ms": torch_ms, "err": err,
        }
    except Exception as e:
        result_dict[f"{gpu_id}_{size}"] = {
            "gpu": gpu_id, "size": size, "error": str(e)[:200],
        }


def run_benchmark_all_gpus(sizes=[8192]):
    """Run benchmarks on all GPUs in parallel."""
    n_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for gpu_id in range(n_gpus):
        for size in sizes:
            p = mp.Process(target=bench_worker, args=(gpu_id, size, result_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join(timeout=120)

    results = dict(result_dict)
    return results


def print_results(results, label=""):
    """Print benchmark results in a table."""
    print(f"\n  {label}")
    print(f"  {'GPU':>4} {'Size':>6} {'WS (ms)':>10} {'torch (ms)':>12} {'Gap':>8} {'Err':>6}")
    print(f"  {'-'*50}")

    ws_times = []
    torch_times = []
    for key in sorted(results.keys()):
        r = results[key]
        if "error" in r:
            print(f"  {r['gpu']:>4} {r['size']:>6} {'FAILED':>10}")
            continue
        ws = r.get("ws_ms")
        to = r.get("torch_ms")
        err = r.get("err", 0)
        if ws is None:
            print(f"  {r['gpu']:>4} {r['size']:>6} {'INCORRECT':>10} err={err:.0f}")
            continue
        gap = (ws / to - 1) * 100
        beat = " *" if ws < to else ""
        print(f"  {r['gpu']:>4} {r['size']:>6} {ws:>10.3f} {to:>12.3f} {gap:>+7.1f}%{beat}")
        if r["size"] == 8192:
            ws_times.append(ws)
            torch_times.append(to)

    if ws_times:
        ws_med = statistics.median(ws_times)
        to_med = statistics.median(torch_times)
        gap = (ws_med / to_med - 1) * 100
        print(f"\n  8K median across GPUs: WS={ws_med:.3f}ms  torch={to_med:.3f}ms  gap={gap:+.1f}%")
        return ws_med, to_med
    return None, None


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("  WS Kernel Optimization — Scalable Autoresearch")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print("=" * 60)

    # Baseline
    print("\n>>> Establishing baseline across all GPUs...")
    baseline = run_benchmark_all_gpus(sizes=[4096, 8192, 16384])
    ws_base, to_base = print_results(baseline, "BASELINE")

    print(f"\n  Target: beat torch at {to_base:.3f}ms")
    print(f"  Current best WS: {ws_base:.3f}ms ({(ws_base/to_base-1)*100:+.1f}%)")
    print(f"  Need to save: {ws_base - to_base:.3f}ms")

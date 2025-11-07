#!/usr/bin/env python3
"""
sweep_sk_grid_tritonblas.py

Sweep sk_grid (1..1024) for Stream-K matmul using tritonblas.matmul and print top-10 performers.

Usage:
    python sweep_sk_grid_tritonblas.py --M 128 --N 128 --K 256

Notes:
 - Requires CUDA, torch, and tritonblas to be installed.
 - The script will attempt to obtain block sizes from a selector factory exposed by tritonblas
   (several common attribute names are tried). If not found, it falls back to BLK_M=16, BLK_N=16.
"""

import argparse
import math
import time
import csv
import traceback
from statistics import mean

import torch

try:
    import tritonblas
except Exception as e:
    raise ImportError(
        "Failed to import tritonblas. Ensure tritonblas is installed."
    ) from e


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep sk_grid and show top performers (tritonblas.matmul)"
    )
    p.add_argument("--M", type=int, default=5248, help="M dimension")
    p.add_argument("--N", type=int, default=7936, help="N dimension")
    p.add_argument("--K", type=int, default=4096, help="K dimension")
    p.add_argument(
        "--dtype", choices=["fp16", "fp32"], default="fp16", help="Data type"
    )
    p.add_argument(
        "--min-grid", type=int, default=64, help="Minimum sk_grid (inclusive)"
    )
    p.add_argument(
        "--max-grid", type=int, default=1024, help="Maximum sk_grid (inclusive)"
    )
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations per grid")
    p.add_argument(
        "--repeats", type=int, default=100, help="Timed repeats per grid (avg used)"
    )
    p.add_argument("--topk", type=int, default=10, help="How many top results to show")
    p.add_argument(
        "--treat-rem-zero-full",
        action="store_true",
        help="If remainder == 0 treat last-wave occupancy as 100% (instead of 0%)",
    )
    p.add_argument(
        "--out-csv", type=str, default=None, help="Save full results to CSV filename"
    )
    return p.parse_args()


def dtype_from_flag(flag: str):
    if flag == "fp16":
        return torch.float16
    elif flag == "fp32":
        return torch.float32
    else:
        raise ValueError("unsupported dtype")


def compute_total_flops(M, N, K):
    # Dense matmul FLOPs = 2*M*N*K
    return 2 * M * N * K


def create_random_tensors(M, N, K, dtype):
    dev = torch.device("cuda")
    a = torch.randn((M, K), device=dev, dtype=dtype)
    b = torch.randn((K, N), device=dev, dtype=dtype)
    c = torch.empty((M, N), device=dev, dtype=dtype)
    return a, b, c


def try_get_selector_factory():
    """
    Try several likely names for a selector factory on tritonblas or related modules.
    Returns a callable factory or None.
    """
    candidates = [
        "_make_matmul_selector",
        "make_matmul_selector",
        "MatmulHeuristicResult",  # class constructor
        "matmul_selector",  # alternative
    ]
    # try on tritonblas
    for name in candidates:
        if hasattr(tritonblas, name):
            return getattr(tritonblas, name)
    # try in submodules (common place)
    for attr in dir(tritonblas):
        try:
            obj = getattr(tritonblas, attr)
            for name in candidates:
                if hasattr(obj, name):
                    return getattr(obj, name)
        except Exception:
            continue
    return None


def get_block_size_from_selector(factory, M, N, K, dtype):
    """
    If factory is available, call it with (M,N,K,a_dtype,b_dtype,c_dtype) to obtain a selector-like
    object and call selector.get_config() -> (BLK_M, BLK_N, BLK_K, gsize_m, ...)
    If anything fails, return fallback (16,16).
    """
    fallback = (16, 16)
    if factory is None:
        return fallback
    try:
        # Some factories are classes; some are functions. Try calling with dtype for a/b/c.
        selector = None
        try:
            selector = factory(M, N, K, dtype, dtype, dtype)
        except TypeError:
            # maybe factory expects different args or object constructor signature; try fewer args
            try:
                selector = factory(M, N, K)
            except Exception:
                selector = None
        if selector is None:
            return fallback
        if hasattr(selector, "get_config"):
            cfg = selector.get_config()
            # expect first two entries BLK_M, BLK_N
            if len(cfg) >= 2:
                return int(cfg[0]), int(cfg[1])
        # fallback if no get_config
        return fallback
    except Exception:
        return fallback


def compute_last_wave_occupancy_pct(
    M, N, BLK_M, BLK_N, sk_grid, treat_rem_zero_full=False
):
    total_blocks_M = math.ceil(M / BLK_M)
    total_blocks_N = math.ceil(N / BLK_N)
    tiles = total_blocks_M * total_blocks_N
    if sk_grid <= 0:
        return 0.0, tiles
    rem = tiles % sk_grid
    if rem == 0 and treat_rem_zero_full:
        occ_pct = 100.0
    else:
        occ_pct = (rem / sk_grid) * 100.0
    return occ_pct, tiles


def time_one_matmul(tritonblas_module, a, b, c, sk_grid, warmup=3, repeats=4):
    # Use tritonblas.matmul(..., enable_streamk=True, sk_grid=sk_grid)
    # Warmup
    for _ in range(warmup):
        try:
            tritonblas_module.matmul(a, b, c, enable_streamk=True, sk_grid=sk_grid)
        except Exception:
            # ignore errors in warmup
            pass
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            tritonblas_module.matmul(a, b, c, enable_streamk=True, sk_grid=sk_grid)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        except Exception:
            # failure for this sk_grid (e.g. invalid config / kernel error / OOM)
            return None
    if len(times) == 0:
        return None
    return mean(times)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script.")

    dtype = dtype_from_flag(args.dtype)
    M, N, K = args.M, args.N, args.K

    a, b, c = create_random_tensors(M, N, K, dtype)
    total_flops = compute_total_flops(M, N, K)

    selector_factory = try_get_selector_factory()
    BLK_M, BLK_N = get_block_size_from_selector(selector_factory, M, N, K, dtype)
    print(
        f"Using BLK_M={BLK_M}, BLK_N={BLK_N} for occupancy calculations (from selector or fallback)."
    )

    results = []

    print(
        f"Sweeping sk_grid from {args.min_grid} to {args.max_grid} for matmul {M}x{K} * {K}x{N}"
    )
    print(f"Using dtype={args.dtype}, warmup={args.warmup}, repeats={args.repeats}")
    print("Press Ctrl-C to stop early.\n")

    occ_pct, tiles = compute_last_wave_occupancy_pct(
        M, N, BLK_M, BLK_N, 304, treat_rem_zero_full=args.treat_rem_zero_full
    )

    for sk in [
        128,
        256,
        304,
        512,
        608,
        912,
        1024,
        1216,
        tiles // 2,
        tiles,
        tiles * 2,
        tiles * 3,
        tiles * 4,
        tiles * 5,
        tiles * 8,
    ]:
        try:
            occ_pct, tiles = compute_last_wave_occupancy_pct(
                M, N, BLK_M, BLK_N, sk, treat_rem_zero_full=args.treat_rem_zero_full
            )
            elapsed = time_one_matmul(
                tritonblas, a, b, c, sk, warmup=args.warmup, repeats=args.repeats
            )

            if elapsed is None:
                print(f"sk={sk:4d}: FAILED (error during run).")
                continue
            gflops = (total_flops / elapsed) / 1e9
            results.append(
                {
                    "sk_grid": sk,
                    "gflops": gflops,
                    "time_s": elapsed,
                    "last_wave_occupancy_pct": occ_pct,
                    "tiles": tiles,
                }
            )
            print(
                f"sk={sk:4d}  gflops={gflops:8.2f}  time={elapsed*1000:7.2f} ms  occ={occ_pct:6.2f}%  tiles={tiles}"
            )
        except KeyboardInterrupt:
            print("Interrupted by user. Stopping sweep.")
            break
        except Exception:
            print(f"sk={sk:4d} ERROR:\n{traceback.format_exc()}")
            continue

    if not results:
        print("No successful runs recorded.")
        return

    results_sorted = sorted(results, key=lambda r: r["gflops"], reverse=True)
    topk = results_sorted[: args.topk]

    # Print topk table
    print("\nTop performers:")
    header = f"{'rank':>4s} | {'sk_grid':>7s} | {'GFLOPS':>9s} | {'time(ms)':>9s} | {'occ(%)':>7s} | {'tiles':>6s}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(topk, start=1):
        print(
            f"{i:4d} | {r['sk_grid']:7d} | {r['gflops']:9.2f} | {r['time_s']*1000:9.2f} | {r['last_wave_occupancy_pct']:7.2f} | {r['tiles']:6d}"
        )

    # Optionally save all results
    out_csv = args.out_csv or f"sweep_tritonblas_M{M}_N{N}_K{K}_{args.dtype}.csv"
    try:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "sk_grid",
                    "gflops",
                    "time_s",
                    "last_wave_occupancy_pct",
                    "tiles",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nFull results written to {out_csv}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")


if __name__ == "__main__":
    main()

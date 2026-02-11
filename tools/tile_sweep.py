#!/usr/bin/env python3
"""
This script sweeps over possible macro tile sizes for a GEMM (matrix multiplication)
using the tritonBLAS API, with optional transposition of A and B. It manually overrides
the tile selection performed by tritonBLAS, runs benchmarks for each configuration, and
prints a table of results including performance relative to the best tile.
"""

import argparse
import itertools
import torch
import triton
import tritonblas
from tqdm import tqdm
import random


# Compute performance in TFLOPS given the elapsed time (in ms) and matrix dimensions.
def perf_ms(ms, m, n, k):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)


# Custom heuristic selector that forces TritonBLAS to use the provided macro tile.
class CustomHeuristic(tritonblas.MatmulHeuristicResult):
    def __init__(self, m, n, k, custom_tile, a_dtype, b_dtype, c_dtype):
        """
        custom_tile: A tuple (BLK_M, BLK_N, BLK_K)
        """
        self.custom_tile = custom_tile
        super().__init__(m, n, k, a_dtype, b_dtype, c_dtype)

    def _get_best_tile_size(self):
        return self.custom_tile

    def _get_gsize_m(self, BLK_M, BLK_N, BLK_K):
        return super()._get_gsize_m(BLK_M, BLK_N, BLK_K)


def run_tritonblas_matmul(
    m, n, k, a_dtype, b_dtype, c_dtype, BLK_M, BLK_N, BLK_K, transA, transB
):
    # Determine the actual shape to allocate, then transpose if needed
    # A_orig is always (M x K) if transA=='N', else (K x M)
    if transA == "N":
        A = torch.randn(m, k, device="cuda", dtype=a_dtype)
    else:  # transA == "T"
        A = torch.randn(k, m, device="cuda", dtype=a_dtype)

    # B_orig is always (K x N) if transB=='N', else (N x K)
    if transB == "N":
        B = torch.randn(k, n, device="cuda", dtype=b_dtype)
    else:  # transB == "T"
        B = torch.randn(n, k, device="cuda", dtype=b_dtype)

    # Now apply the logical transpose flags so that matmul sees A: (M x K), B: (K x N)
    if transA == "T":
        A = A.T  # shape -> (M x K)
    if transB == "T":
        B = B.T  # shape -> (K x N)

    # Allocate output
    C = torch.zeros((m, n), device="cuda", dtype=c_dtype)

    selector = CustomHeuristic(
        m,
        n,
        k,
        custom_tile=(BLK_M, BLK_N, BLK_K),
        a_dtype=A.dtype,
        b_dtype=B.dtype,
        c_dtype=C.dtype,
    )
    config = selector.get_config()  # (BLK_M, BLK_N, BLK_K, group_size)

    # Benchmark
    cfg = tritonblas.matmul_preamble(selector)
    matmul_fn = lambda: tritonblas.matmul_lt(A, B, C, selector, cfg, False)
    elapsed_ms = triton.testing.do_bench(matmul_fn, warmup=20, rep=200)
    tflops = perf_ms(elapsed_ms, m, n, k)
    return tflops, elapsed_ms, config


def sweep_macro_tiles_tritonblas(m, n, k, a_dtype, b_dtype, c_dtype, transA, transB):
    # Candidate tile sizes
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256, 512]
    valid_tiles = list(itertools.product(block_mn_range, block_mn_range, block_k_range))

    # Default heuristic (for comparison)
    default_selector = tritonblas.MatmulHeuristicResult(
        m, n, k, a_dtype, b_dtype, c_dtype
    )
    heur_config = default_selector.get_config()
    heur_tile = (heur_config[0], heur_config[1], heur_config[2])
    gsize_m = heur_config[3]
    print(f"Default heuristic selected tile and gsize_m: {heur_tile} {gsize_m}\n")

    results = []
    best_tflops = 0.0
    best_tile = None

    for tile in tqdm(valid_tiles, desc="Sweeping macro tiles"):
        BLK_M, BLK_N, BLK_K = tile
        try:
            tflops, ms, config = run_tritonblas_matmul(
                m, n, k, a_dtype, b_dtype, c_dtype, BLK_M, BLK_N, BLK_K, transA, transB
            )
            results.append((tile, tflops, ms))
            if tflops > best_tflops:
                best_tflops = tflops
                best_tile = tile
        except Exception as e:
            print(f"Error with tile {tile}: {e}")
            continue

    # Benchmark the heuristic tile
    try:
        heur_tflops, heur_ms, _ = run_tritonblas_matmul(
            m, n, k, a_dtype, b_dtype, c_dtype, *heur_tile, transA, transB
        )
        heur_ratio = heur_tflops / best_tflops if best_tflops > 0 else 0.0
    except Exception as e:
        print(f"Error running heuristic tile {heur_tile}: {e}")
        heur_tflops = heur_ms = heur_ratio = 0.0

    # Print results
    print("\n=== Sweep Results ===")
    header = (
        f"{'Tile':>15} | {'TFLOPS':>8} | {'Time (ms)':>9} | {'Ratio':>6} | {'Note':>20}"
    )
    print(header)
    print("-" * len(header))
    for tile, tflops, ms in sorted(results, key=lambda x: -x[1]):
        ratio = tflops / best_tflops if best_tflops > 0 else 0.0
        note = ""
        if tile == best_tile:
            note += " <--- best"
        if tile == heur_tile:
            note += " <--- heuristic"
        print(
            f"{tile[0]:3}x{tile[1]:3}x{tile[2]:3} | {tflops:8.3f} | {ms:9.3f} | {ratio:6.3f} | {note:20}"
        )

    print(f"\nProblem size: {m}x{n}x{k} (transA={transA}, transB={transB})")
    print(f"Best tile: {best_tile} → {best_tflops:.3f} TFLOPS")
    print(f"Heuristic tile: {heur_tile} → {heur_tflops:.3f} TFLOPS")
    print(f"Heuristic as % of best: {100 * heur_ratio:6.2f}%")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep macro tile sizes using TritonBLAS API for GEMM benchmarking with optional transpose."
    )
    parser.add_argument(
        "--m", type=int, default=random.randint(1024, 8192), help="Matrix M"
    )
    parser.add_argument(
        "--n", type=int, default=random.randint(1024, 8192), help="Matrix N"
    )
    parser.add_argument(
        "--k", type=int, default=random.randint(1024, 8192), help="Matrix K"
    )
    parser.add_argument(
        "--a_dtype",
        choices=[
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
        ],
        default=torch.float16,
        help="Input data type ('torch.float16', 'torch.bfloat16', 'torch.float8_e5m2', 'torch.float8_e4m3fnuz' default: 'torch.float16').",
    )
    parser.add_argument(
        "--b_dtype",
        choices=[
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
        ],
        default=torch.float16,
        help="Input data type ('torch.float16', 'torch.bfloat16', 'torch.float8_e5m2', 'torch.float8_e4m3fnuz' default: 'torch.float16').",
    )
    parser.add_argument(
        "--c_dtype",
        choices=[
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
        ],
        default=torch.float16,
        help="Input data type ('torch.float16', 'torch.bfloat16', 'torch.float8_e5m2', 'torch.float8_e4m3fnuz' default: 'torch.float16').",
    )
    parser.add_argument(
        "--transA",
        choices=["N", "T"],
        default="T",
        help="Transpose A? 'N' = no, 'T' = yes",
    )
    parser.add_argument(
        "--transB",
        choices=["N", "T"],
        default="N",
        help="Transpose B? 'N' = no, 'T' = yes",
    )
    args = parser.parse_args()

    print(
        f"Running problem {args.m}x{args.n}x{args.k} with transA={args.transA}, transB={args.transB}"
    )
    sweep_macro_tiles_tritonblas(
        args.m,
        args.n,
        args.k,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.transA,
        args.transB,
    )

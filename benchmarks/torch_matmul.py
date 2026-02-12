#!/usr/bin/env python3
"""
Benchmark torch.matmul performance.

Supports an optional --cu-sweep mode (MI300X) that re-invokes this script as
subprocesses with ROC_GLOBAL_CU_MASK set, producing results with an
``active_cus`` column.
"""
import json
import os
import subprocess
import sys
import yaml
import argparse
import torch
import csv

from common import get_cu_info, build_balanced_hex_mask


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a dtype to the corresponding torch.dtype.

    Args:
        dtype_str (str): The string representation of the dtype (e.g., "torch.float32").

    Returns:
        torch.dtype: The corresponding torch dtype.
    """
    # Remove the 'torch.' prefix if it exists
    dtype_str = dtype_str.replace("torch.", "")
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(
            f"Invalid dtype string: '{dtype_str}'. Available options are: {', '.join([attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)])}"
        )


def bench_matmul(input_yaml: str):
    # Load benchmark cases from the YAML file
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)

    benchmark_results = []
    # Convert the dataset cases into tuples: (m, n, k, in_dtype, out_dtype)
    dataset_tuples = [
        (
            case["m"],
            case["n"],
            case["k"],
            str_to_dtype(case["in_dtype"]),
            str_to_dtype(case["out_dtype"]),
            case["transA"],
            case["transB"],
        )
        for case in dataset
    ]

    # Iterate over all benchmark cases
    for m, n, k, in_dtype, out_dtype, transA, transB in dataset_tuples:
        # Initialize the matrices A and B with appropriate dimensions based on transA and transB
        if transA == "T":
            A_size = (m, k)  # A is MxK
        else:
            A_size = (k, m)  # A is KxM (we will later transpose it with .T)

        if transB == "T":
            B_size = (k, n)  # B is KxN
        else:
            B_size = (n, k)  # B is NxK (we will later transpose it with .T)

        # Initialize tensors with the appropriate dimensions
        A = torch.randn(*A_size, device="cuda", dtype=in_dtype)
        B = torch.randn(*B_size, device="cuda", dtype=in_dtype)

        # Apply transpose on A or B if necessary (only needed for "N" case)
        if transA == "N":
            A = A.T  # Apply transpose to A if transA is "N"

        if transB == "N":
            B = B.T  # Apply transpose to B if transB is "N"

        # Warm-up iterations
        for _ in range(20):
            _ = torch.matmul(A, B)

        # Benchmark the torch.matmul over 10 repetitions using CUDA events for timing.
        iterations = 10
        times = []
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = torch.matmul(A, B)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)  # time in milliseconds
            times.append(elapsed_ms)

        # Calculate mean execution time (ms) and derive performance.
        mean_ms = sum(times) / len(times)
        gflops = (2 * m * n * k) / (mean_ms * 1e-3) / 1e9

        print(
            f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype} perf={gflops:.1f} GFLOPS"
        )

        metrics = {
            "m": m,
            "n": n,
            "k": k,
            "gflops": gflops,
            "ms": mean_ms,
            "in_dtype": str(in_dtype),
            "out_dtype": str(out_dtype),
            "transA": transA,
            "transB": transB,
        }
        benchmark_results.append(metrics)

    return benchmark_results


def _build_child_cmd(args):
    """Build a subprocess command from parsed args (excludes --cu-sweep and --output-csv)."""
    cmd = [sys.executable, os.path.abspath(__file__)]
    cmd += ["--input-yaml", args.input_yaml]
    return cmd


def write_csv(filename: str, results):
    """Write the benchmark results to a CSV file."""
    fieldnames = ["m", "n", "k", "gflops", "ms", "in_dtype", "out_dtype", "transA", "transB"]
    if results and "active_cus" in results[0]:
        fieldnames.insert(0, "active_cus")
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Benchmark results saved to '{filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark torch.matmul performance and optionally output performance metrics to a CSV file."
    )
    parser.add_argument(
        "--input-yaml",
        type=str,
        default="matmul_random.yaml",
        help="Input YAML file containing benchmark cases (default: ./matmul_random.yaml).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Filename for CSV output (if not specified, CSV output is disabled).",
    )
    parser.add_argument(
        "--cu-sweep", action="store_true",
        help="Run a balanced CU-mask sweep (MI300X).  Re-invokes this script as "
             "subprocesses with ROC_GLOBAL_CU_MASK set.",
    )
    parser.add_argument(
        "--cu-sweep-max-remove", type=int, default=34,
        help="Max CUs to remove per XCD (default 34, minimum 4 CUs/XCD left).",
    )

    # Hidden: used by cu-sweep parent to tag subprocess results
    parser.add_argument("--_active-cus", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.cu_sweep:
        full_cus, num_xcds, cus_per_xcd = get_cu_info()
        all_results = []
        child_base = _build_child_cmd(args)
        max_remove = min(args.cu_sweep_max_remove, cus_per_xcd - 1)

        for r in range(max_remove + 1):
            active = full_cus - r * num_xcds
            mask = build_balanced_hex_mask(r, num_xcds, cus_per_xcd)

            child_cmd = child_base + ["--_active-cus", str(active)]

            env = os.environ.copy()
            if mask:
                env["ROC_GLOBAL_CU_MASK"] = mask

            proc = subprocess.run(
                child_cmd, capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                print(f"[active_cus={active}] subprocess failed:", file=sys.stderr)
                sys.stderr.write(proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr)
                continue

            step_results = json.loads(proc.stdout)
            all_results.extend(step_results)

        if args.output_csv:
            write_csv(args.output_csv, all_results)
        sys.exit(0)

    is_worker = args._active_cus is not None

    if is_worker:
        # Suppress prints when running as a subprocess
        import io
        sys.stdout = io.StringIO()

    benchmark_results = bench_matmul(args.input_yaml)

    if is_worker:
        sys.stdout = sys.__stdout__
        for row in benchmark_results:
            row["active_cus"] = args._active_cus
        print(json.dumps(benchmark_results))
        sys.exit(0)

    if args.output_csv:
        write_csv(args.output_csv, benchmark_results)

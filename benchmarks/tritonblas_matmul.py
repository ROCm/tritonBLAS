#!/usr/bin/env python3
import yaml
import argparse
import torch
import triton
import random
import tritonblas
import csv
from tqdm import tqdm


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a dtype to the corresponding torch.dtype.

    Args:
        dtype_str (str): The string representation of the dtype (e.g., "torch.float32").

    Returns:
        torch.dtype: The corresponding torch dtype.
    """
    dtype_str = dtype_str.replace("torch.", "")
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(
            f"Invalid dtype string: '{dtype_str}'. Available options are: "
            f"{', '.join([attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)])}"
        )


def init_by_size_and_type(size, dtype, init_type):
    """
    Initialize a tensor of the given size and type using the specified initialization method.

    Args:
        size (tuple): The size of the tensor to be initialized.
        dtype (torch.dtype): The data type of the tensor.
        init_type (str): The initialization method ('hpl', 'trig_float', 'zeros', 'randn').

    Returns:
        torch.Tensor: The initialized tensor.
    """
    if init_type == "hpl":
        return torch.empty(size, device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        M, N = size
        return (
            torch.reshape(
                torch.arange(0, M * N, device="cuda", dtype=torch.float32), (M, N)
            )
            .sin()
            .to(dtype=dtype)
        )
    elif init_type == "zeros":
        return torch.zeros(size, dtype=dtype, device="cuda")
    elif init_type == "randn":
        # Need to generate and then cast to support f8 (randn not supported for f8)
        A = torch.randn(size, dtype=torch.float32, device="cuda")
        return A.to(dtype)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")


def bench_matmul(
    input_yaml: str,
    init_type: str,
    print_verbose=False,
    shuffle_benchmark=True,
    output_csv=None,
    write_csv_freq=100,
    enable_streamk=False,
):
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)

    benchmark_results = []

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
    if shuffle_benchmark:
        random.shuffle(dataset_tuples)
    count = 0

    for m, n, k, in_dtype, out_dtype, transA, transB in (
        tqdm(dataset_tuples) if not print_verbose else dataset_tuples
    ):
        # Adjust dimensions for transposition and apply tensor.T if needed
        if transA == "T":
            A_size = (m, k)  # A is MxK
        else:
            A_size = (k, m)  # A is KxM (we will later transpose it with .T)

        if transB == "T":
            B_size = (k, n)  # B is KxN
        else:
            B_size = (n, k)  # B is NxK (we will later transpose it with .T)

        # Initialize tensors with the appropriate dimensions
        A = init_by_size_and_type(A_size, in_dtype, init_type)
        B = init_by_size_and_type(B_size, in_dtype, init_type)

        # Apply transpose on A or B if necessary (only needed for "N" case)
        if transA == "N":
            A = A.T  # Apply transpose to A if transA is "N"

        if transB == "N":
            B = B.T  # Apply transpose to B if transB is "N"

        C = torch.zeros((m, n), device="cuda", dtype=out_dtype)

        # Compute performance metrics
        flops = lambda: 2 * m * n * k * 1e-12
        gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
        bytes_fn = lambda: (A.element_size() * ((m * k) + (n * k))) + (
            (m * n) * C.element_size()
        )

        # Build a tritonBLAS selector config and launch matmul_lt
        selector = tritonblas.MatmulHeuristicResult(
            m, n, k, A.dtype, B.dtype, C.dtype
        )
        config = selector.get_config()
        matmul = lambda: tritonblas.matmul(A, B, C, enable_streamk=enable_streamk)
        ms = triton.testing.do_bench(matmul, warmup=20, rep=20)
        perf = gflops(ms)

        if print_verbose:
            print(
                f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype}, init={init_type}, perf={perf}(GFLOPs) selected_tile={selector.config[0]}x{selector.config[1]}x{selector.config[2]}"
            )

        metrics = {
            "m": m,
            "n": n,
            "k": k,
            "mnk": m * n * k,
            "macro_tile": f"{config[0]}x{config[1]}x{config[2]}",
            "bytes": bytes_fn(),
            "flops": flops(),
            "tritonblas_gflops": perf,
            "a_type": str(in_dtype),
            "b_type": str(in_dtype),
            "c_type": str(out_dtype),
            "d_type": str(out_dtype),
            "compute_type": str(out_dtype),
            "in_dtype": str(in_dtype),
            "out_dtype": str(out_dtype),
            "init_type": init_type,
            "transA": str(transA),
            "transB": str(transB),
            "us": ms / 1000,
            "alpha": 1,
            "beta": 0,
            "enable_streamk": enable_streamk,
        }
        benchmark_results.append(metrics)

        # Write every 100 entries
        if count % write_csv_freq == 0:
            if output_csv:
                write_csv(output_csv, benchmark_results)
        count = count + 1

    return benchmark_results


def write_csv(filename: str, results):
    fieldnames = [
        "m",
        "n",
        "k",
        "mnk",
        "macro_tile",
        "bytes",
        "flops",
        "tritonblas_gflops",
        "a_type",
        "b_type",
        "c_type",
        "d_type",
        "compute_type",
        "in_dtype",
        "out_dtype",
        "init_type",
        "transA",
        "transB",
        "us",
        "alpha",
        "beta",
        "enable_streamk",
    ]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Benchmark results saved to '{filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark matmul performance and optionally output performance metrics to a CSV file."
    )
    parser.add_argument(
        "--input-yaml",
        type=str,
        default="../datasets/matmul_random.yaml",
        help="Input YAML file containing benchmark cases (default: ./matmul_random.yaml).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Filename for CSV output (if not specified, CSV output is disabled).",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="randn",
        choices=["hpl", "trig_float", "zeros", "randn"],
        help="Tensor initialization type (default: randn).",
    )
    parser.add_argument(
        "--shuffle-bench",
        action="store_true",
        help="Randomly shuffle the order the benchmark runs",
    )
    parser.add_argument(
        "--csv-write-freq",
        type=int,
        default=1000,
        help="Number of problems to run before writing to csv",
    )
    parser.add_argument(
        "--print-verbose",
        action="store_true",
        help="Print detailed information for each benchmark.",
    )
    parser.add_argument(
        "--enable-streamk",
        action="store_true",
        help="Enable Stream-K mode for matrix multiplication (default: False for persistent mode).",
    )
    args = parser.parse_args()

    benchmark_results = bench_matmul(
        args.input_yaml,
        args.init_type,
        shuffle_benchmark=args.shuffle_bench,
        output_csv=args.output_csv,
        write_csv_freq=args.csv_write_freq,
        print_verbose=args.print_verbose,
        enable_streamk=args.enable_streamk,
    )

    if args.output_csv:
        write_csv(args.output_csv, benchmark_results)

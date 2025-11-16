#!/usr/bin/env python3
"""
TritonBLAS Unified Matrix Multiplication Benchmark

This benchmark script supports both standard (fp16/bf16/fp32) and quantized (fp8/int8) 
matrix multiplication. It automatically detects the dtype and uses the appropriate API.
"""
import argparse
import csv
import random

import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

from tritonblas.utils import MatmulInputs, generate_matmul_inputs, str_to_dtype, _is_float8_like  # type: ignore


def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk, init_type="randn"):
    """Test matmul with proper input generation - handles both quantized and non-quantized dtypes"""

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    selector = tritonblas.MatmulHeuristicResult(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype
    )

    if inputs.is_quantized:
        tritonblas.matmul_a8w8_lt(
            inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, enable_streamk
        )
    else:
        tritonblas.matmul_lt(inputs.A, inputs.B, inputs.C, selector, enable_streamk)

    if inputs.is_quantized:
        acc = torch.matmul(inputs.A.to(torch.float32), inputs.B.to(torch.float32))
        scale = inputs.scaleA[:, None] * inputs.scaleB[None, :]
        acc = acc * scale

        if out_dtype == torch.float8_e4m3fn:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            acc = torch.clamp(acc, -fp8_max, fp8_max)
        elif out_dtype == torch.float8_e5m2:
            fp8_max = torch.finfo(torch.float8_e5m2).max
            acc = torch.clamp(acc, -fp8_max, fp8_max)
        elif out_dtype == torch.int8:
            dtype_max = 127.0
            acc = torch.clamp(acc, -dtype_max, dtype_max)

        torch_c = acc.to(out_dtype)

        if out_dtype == torch.bfloat16:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=0.02, rtol=1e-2
            )
        elif _is_float8_like(out_dtype):
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2
            )
        elif out_dtype == torch.int8:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2
            )
        else:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=0.1, rtol=0.05
            )
    else:
        torch_c = torch.matmul(inputs.A, inputs.B)
        if in_dtype == torch.float16 or out_dtype == torch.float16:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05
            )
        elif in_dtype == torch.bfloat16 or out_dtype == torch.bfloat16:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05
            )
        elif in_dtype == torch.float32 and out_dtype == torch.float32:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=1e-2, rtol=1e-3
            )
        else:
            torch.testing.assert_close(
                inputs.C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05
            )

    size_str = f"SIZE M: {m}, N: {n}, K: {k}, trans: {transA}{transB}"
    print(f"{size_str} Correct✅")


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
        inputs = generate_matmul_inputs(
            m, n, k, in_dtype, out_dtype, transA, transB, init_type
        )

        # Compute performance metrics
        flops = lambda: 2 * m * n * k * 1e-12
        gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
        # Include scale tensors in byte count for quantized dtypes
        bytes_fn = lambda: (
            inputs.A.numel() * inputs.A.element_size()
            + inputs.B.numel() * inputs.B.element_size()
            + inputs.C.numel() * inputs.C.element_size()
            + (
                0
                if inputs.scaleA is None
                else inputs.scaleA.numel() * inputs.scaleA.element_size()
            )
            + (
                0
                if inputs.scaleB is None
                else inputs.scaleB.numel() * inputs.scaleB.element_size()
            )
        )

        # Build a tritonBLAS selector config and launch matmul
        selector = tritonblas.MatmulHeuristicResult(
            m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype
        )
        config = selector.get_config()

        # Use appropriate API based on quantization
        if inputs.is_quantized:
            matmul = lambda: tritonblas.matmul_a8w8_lt(
                inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, enable_streamk
            )
        else:
            matmul = lambda: tritonblas.matmul(
                inputs.A, inputs.B, inputs.C, enable_streamk=enable_streamk
            )

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
            "a_type": str(inputs.A.dtype),
            "b_type": str(inputs.B.dtype),
            "c_type": str(inputs.C.dtype),
            "d_type": str(inputs.C.dtype),
            "compute_type": str(inputs.C.dtype),
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

        # Write every N entries
        if count % write_csv_freq == 0:
            if output_csv:
                write_csv(output_csv, benchmark_results)
        count = count + 1

        if args.checkcorrectness:
            print("correctness: ", end=" ", flush=True)
            test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk, init_type)

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
        description="Benchmark matmul performance (supports both standard and quantized dtypes) and optionally output performance metrics to a CSV file."
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
        "--checkcorrectness",
        action="store_true",
        default=False,
        help="Check result correctness",
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

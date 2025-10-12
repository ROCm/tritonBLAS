#!/usr/bin/env python3
"""
TritonBLAS FP8/INT8 Matrix Multiplication Benchmark

This benchmark script uses the updated utils.py matmul_input_gen function
to properly generate quantized inputs and scales for FP8/INT8 matrix multiplication.
It automatically detects quantized dtypes and generates appropriate scales.
"""
import yaml
import argparse
import torch
import triton
import random
import tritonblas
import csv
import sys
import os
from typing import Optional, Tuple, Union
from tqdm import tqdm

# Add the tests directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from utils import matmul_input_gen, _is_quantized, _is_float8_like, _is_int8

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

def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk):
    """Test matmul with proper input generation using matmul_input_gen"""

    # Adjust dimensions for transposition and apply tensor.T if needed
    if transA == "T":
        A_size = (m, k)  # A is MxK
    else:
        A_size = (k, m)  # A is KxM (we will later transpose it with .T)

    if transB == "T":
        B_size = (k, n)  # B is KxN
    else:
        B_size = (n, k)  # B is NxK (we will later transpose it with .T)

    # Generate inputs using matmul_input_gen for quantized dtypes
    if _is_float8_like(in_dtype) or _is_int8(in_dtype):
        A_init = matmul_input_gen(A_size, in_dtype, "randn", quantize="auto")
        B_init = matmul_input_gen(B_size, in_dtype, "randn", quantize="auto")
        A, scaleA = _is_quantized(A_init)
        B, scaleB = _is_quantized(B_init)
    else:
        A = matmul_input_gen(A_size, in_dtype, "randn")
        B = matmul_input_gen(B_size, in_dtype, "randn")
        scaleA = scaleB = None

    # Apply transpose on A or B if necessary (only needed for "N" case)
    if transA == "N":
        A = A.T  # Apply transpose to A if transA is "N"
        if scaleA is not None:
            scaleA = scaleA.T.contiguous()

    if transB == "N":
        B = B.T  # Apply transpose to B if transB is "N"
        if scaleB is not None:
            scaleB = scaleB.T.contiguous()

    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    # Run TritonBLAS matmul
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    tritonblas.matmul_a8w8_lt(A, B, scaleA, scaleB, C, selector, enable_streamk)

    # Check correctness using the same approach as the working test
    # Use the same reference computation as in test_matmul_a8w8_lt.py
    # 1. Matrix multiplication in float32 (like kernel's tl.dot accumulation)
    acc = torch.matmul(A.to(torch.float32), B.to(torch.float32))
    
    if scaleA.shape[0] == 1:  # (1, M) -> (M, 1)
        scaleA = scaleA.T
    if scaleB.shape[0] == 1:  # (1, N) -> (N, 1) 
        scaleB = scaleB.T
        
    # 3. Apply scales to float32 accumulator (like kernel: acc *= A_scale[:, None] * B_scale[None, :])
    scale = torch.matmul(scaleA, scaleB.T)  # (M, 1) @ (1, N) -> (M, N)
    acc = acc * scale  # Keep in float32

    # 4. Convert to output dtype at the very end (like kernel: c = acc.to(C.type.element_ty))
    # The kernel does implicit clamping to FP8 range before conversion
    if out_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        acc = torch.clamp(acc, -fp8_max, fp8_max)
    elif out_dtype == torch.float8_e5m2:
        fp8_max = torch.finfo(torch.float8_e5m2).max
        acc = torch.clamp(acc, -fp8_max, fp8_max)
    elif out_dtype == torch.int8:
        # INT8 has range [-128, 127], but we use symmetric range [-127, 127] like the kernel
        dtype_max = 127.0
        acc = torch.clamp(acc, -dtype_max, dtype_max)
    
    torch_c = acc.to(out_dtype)

    # Use relaxed tolerance for quantized inputs
    if _is_float8_like(out_dtype):
        torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2)
    elif _is_int8(out_dtype):
        torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=5.0, rtol=0.5)
    size_str = f'SIZE M: {m}, N: {n}, K: {k}, trans: {transA}{transB}'
    print(f'{size_str} Correct✅')

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

        # Generate inputs using matmul_input_gen for quantized dtypes
        if _is_float8_like(in_dtype) or _is_int8(in_dtype):
            A_init = matmul_input_gen(A_size, in_dtype, init_type, quantize="auto")
            B_init = matmul_input_gen(B_size, in_dtype, init_type, quantize="auto")
            A, scaleA = _is_quantized(A_init)
            B, scaleB = _is_quantized(B_init)
        else:
            A = matmul_input_gen(A_size, in_dtype, init_type)
            B = matmul_input_gen(B_size, in_dtype, init_type)
            scaleA = scaleB = None

        # Apply transpose on A or B if necessary (only needed for "N" case)
        if transA == "N":
            A = A.T  # Apply transpose to A if transA is "N"
            if scaleA is not None:
                scaleA = scaleA.T.contiguous()

        if transB == "N":
            B = B.T  # Apply transpose to B if transB is "N"
            if scaleB is not None:
                scaleB = scaleB.T.contiguous()

        C = torch.zeros((m, n), device="cuda", dtype=out_dtype)

        # Compute performance metrics
        flops = lambda: 2 * m * n * k * 1e-12
        gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
#        bytes_fn = lambda: (A.element_size() * ((m * k) + (n * k))) + (
#            (m * n) * C.element_size()
#        )
        bytes_fn = lambda: (
            A.numel() * A.element_size()
            + B.numel() * B.element_size()
            + C.numel() * C.element_size()
            + (0 if scaleA is None else scaleA.numel() * scaleA.element_size())
            + (0 if scaleB is None else scaleB.numel() * scaleB.element_size())
        )

        # Build a tritonBLAS selector config and launch matmul_lt
        selector = tritonblas.MatmulHeuristicResult(
            m, n, k, A.dtype, B.dtype, C.dtype
        )
        config = selector.get_config()
        # fp8/int8 path (expects scales from matmul_input_gen)
        matmul = lambda: tritonblas.matmul_a8w8_lt(A, B, scaleA, scaleB, C, selector, enable_streamk)

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

        if args.checkcorrectness:
            print("correctness: ", end=" ", flush=True)
            test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk)

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
    # Quantization is automatically handled based on dtype

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

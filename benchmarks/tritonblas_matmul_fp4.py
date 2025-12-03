#!/usr/bin/env python3
"""
TritonBLAS FP4 Matrix Multiplication Benchmark

This benchmark script is specifically designed for FP4 matrix multiplication.
It uses the unified input generation infrastructure and provides YAML-driven
benchmarking with CSV output compatible with other TritonBLAS benchmarks.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

# Add include directory to path BEFORE importing tritonblas
sys.path.insert(0, str(Path(__file__).parent.parent / "include"))

import torch  # type: ignore
import triton  # type: ignore
import tritonblas  # type: ignore
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

from tritonblas.utils import (
    generate_matmul_inputs,
    str_to_dtype,
    mxfp4_to_f32,
    e8m0_to_f32,
)


def compute_fp4_reference(inputs, out_dtype):
    """
    Compute reference result using PyTorch with dequantized FP4 inputs.
    
    Args:
        inputs: MatmulInputs with FP4 data and scales
        out_dtype: Target output dtype
        
    Returns:
        Reference output tensor
    """
    m, k_packed = inputs.A.shape
    k_packed_b, n = inputs.B.shape
    k = k_packed * 2
    
    # Dequantize FP4 to FP32
    x_f32 = mxfp4_to_f32(inputs.A)  # (m, k)
    w_f32 = mxfp4_to_f32(inputs.B.T)  # (n, k) -> transpose to (k, n) for matmul
    
    # Convert e8m0 scales to FP32 and expand to match data shape
    x_scales_f32 = e8m0_to_f32(inputs.scaleA)  # (m, k//32)
    x_scales_f32 = x_scales_f32.repeat_interleave(32, dim=1)  # (m, k)
    
    w_scales_f32 = e8m0_to_f32(inputs.scaleB)  # (n, k//32)
    w_scales_f32 = w_scales_f32.repeat_interleave(32, dim=1)  # (n, k)
    
    # Apply scales
    x_f32 = x_f32 * x_scales_f32
    w_f32 = w_f32 * w_scales_f32
    
    # Compute matmul: (m, k) @ (k, n) = (m, n)
    return torch.mm(x_f32, w_f32.T).to(out_dtype)[:m, :n]


def test_matmul_fp4(m, n, k, out_dtype, transA, transB, init_type="randn"):
    """
    Test FP4 matmul correctness against PyTorch reference.
    
    Args:
        m, n, k: Matrix dimensions
        out_dtype: Output data type
        transA, transB: Transpose flags
        init_type: Initialization method
    """
    # Validate K divisibility
    if k % 32 != 0:
        raise ValueError(f"K must be divisible by 32 for FP4, got K={k}")
    
    # Generate FP4 inputs
    # Note: For FP4, we need B in shape (N, K//2) but generate_matmul_inputs
    # creates it as (K//2, N) when transB="N", so we transpose it back
    inputs = generate_matmul_inputs(
        m, n, k, in_dtype="fp4", out_dtype=out_dtype,
        transA=transA, transB=transB, init_type=init_type
    )
    
    # matmul_fp4 expects B as (N, K//2), but generate_matmul_inputs gives us (K//2, N)
    # So we need to transpose B and its scales
    B_for_fp4 = inputs.B.T  # (K//2, N) -> (N, K//2)
    
    # Run TritonBLAS FP4 matmul
    tritonblas.matmul_fp4(inputs.A, B_for_fp4, inputs.C, inputs.scaleA, inputs.scaleB)
    
    # Compute reference
    ref = compute_fp4_reference(inputs, out_dtype)
    
    # Validate correctness
    nan_mask = torch.isnan(inputs.C)
    inf_mask = torch.isinf(inputs.C)
    valid_mask = ~nan_mask & ~inf_mask
    
    ref_valid_mask = ~torch.isnan(ref) & ~torch.isinf(ref)
    both_valid = valid_mask & ref_valid_mask
    
    if both_valid.sum() == 0:
        print(f"SIZE M: {m}, N: {n}, K: {k}, trans: {transA}{transB} - WARNING: No valid values to compare")
        return
    
    # Compute error metrics
    out_valid = inputs.C[both_valid]
    ref_valid = ref[both_valid]
    
    abs_error = torch.abs(out_valid - ref_valid)
    mean_abs_err = abs_error.mean().item()
    max_abs_err = abs_error.max().item()
    
    # FP4 tolerance (similar to FP8)
    atol = 2.0
    rtol = 0.2
    
    try:
        torch.testing.assert_close(
            inputs.C.to(torch.float32),
            ref.to(torch.float32),
            atol=atol,
            rtol=rtol
        )
        size_str = f"SIZE M: {m}, N: {n}, K: {k}, trans: {transA}{transB}"
        print(f"{size_str} Correct✅ (mean_err={mean_abs_err:.6f}, max_err={max_abs_err:.6f})")
    except AssertionError as e:
        size_str = f"SIZE M: {m}, N: {n}, K: {k}, trans: {transA}{transB}"
        print(f"{size_str} FAILED❌ (mean_err={mean_abs_err:.6f}, max_err={max_abs_err:.6f})")
        raise


def bench_matmul_fp4(
    input_yaml: str,
    init_type: str,
    print_verbose=False,
    shuffle_benchmark=True,
    output_csv=None,
    write_csv_freq=100,
    check_correctness=False,
):
    """
    Benchmark FP4 matmul performance from YAML dataset.
    
    Args:
        input_yaml: Path to YAML file with benchmark cases
        init_type: Tensor initialization method
        print_verbose: Print detailed info for each benchmark
        shuffle_benchmark: Randomly shuffle benchmark order
        output_csv: CSV output filename
        write_csv_freq: Write CSV every N problems
        check_correctness: Validate correctness for each problem
        
    Returns:
        List of benchmark results
    """
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)
    
    benchmark_results = []
    
    # Parse dataset
    dataset_tuples = []
    for case in dataset:
        # Handle both "fp4" and "torch.fp4" formats
        in_dtype_str = case["in_dtype"]
        if in_dtype_str.lower() in ["fp4", "torch.fp4"]:
            in_dtype = "fp4"
        else:
            raise ValueError(f"Expected in_dtype='fp4', got '{in_dtype_str}'")
        
        out_dtype = str_to_dtype(case["out_dtype"])
        
        dataset_tuples.append((
            case["m"],
            case["n"],
            case["k"],
            in_dtype,
            out_dtype,
            case["transA"],
            case["transB"],
        ))
    
    if shuffle_benchmark:
        random.shuffle(dataset_tuples)
    
    count = 0
    
    for m, n, k, in_dtype, out_dtype, transA, transB in (
        tqdm(dataset_tuples) if not print_verbose else dataset_tuples
    ):
        # Validate K divisibility
        if k % 32 != 0:
            print(f"WARNING: Skipping M={m}, N={n}, K={k} - K must be divisible by 32 for FP4")
            continue
        
        # Generate FP4 inputs
        inputs = generate_matmul_inputs(
            m, n, k, in_dtype=in_dtype, out_dtype=out_dtype,
            transA=transA, transB=transB, init_type=init_type
        )
        
        # Compute performance metrics
        flops = lambda: 2 * m * n * k * 1e-12
        gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
        
        # Include scale tensors in byte count
        bytes_fn = lambda: (
            inputs.A.numel() * inputs.A.element_size()
            + inputs.B.numel() * inputs.B.element_size()
            + inputs.C.numel() * inputs.C.element_size()
            + inputs.scaleA.numel() * inputs.scaleA.element_size()
            + inputs.scaleB.numel() * inputs.scaleB.element_size()
        )
        
        # matmul_fp4 expects B as (N, K//2), but generate_matmul_inputs gives us (K//2, N)
        B_for_fp4 = inputs.B.T  # (K//2, N) -> (N, K//2)
        
        # Get tile configuration from matmul_fp4's selector
        # matmul_fp4 uses _make_matmul_selector internally with mx_block_size=32
        from tritonblas.matmul import _make_matmul_selector
        selector = _make_matmul_selector(m, n, k, "f4", "f4", out_dtype, mx_block_size=32)
        block_m, block_n, block_k, gsize_m = selector.get_config()
        
        # Benchmark FP4 matmul
        matmul = lambda: tritonblas.matmul_fp4(
            inputs.A, B_for_fp4, inputs.C, inputs.scaleA, inputs.scaleB
        )
        
        ms = triton.testing.do_bench(matmul, warmup=20, rep=20)
        perf = gflops(ms)
        
        if print_verbose:
            print(
                f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype}, "
                f"init={init_type}, perf={perf:.2f} GFLOPS"
            )
        
        metrics = {
            "m": m,
            "n": n,
            "k": k,
            "mnk": m * n * k,
            "macro_tile": f"{block_m}x{block_n}x{block_k}",
            "bytes": bytes_fn(),
            "flops": flops(),
            "tritonblas_gflops": perf,
            "a_type": str(inputs.A.dtype),
            "b_type": str(inputs.B.dtype),
            "c_type": str(inputs.C.dtype),
            "d_type": str(inputs.C.dtype),
            "compute_type": str(inputs.C.dtype),
            "in_dtype": in_dtype,
            "out_dtype": str(out_dtype),
            "init_type": init_type,
            "transA": str(transA),
            "transB": str(transB),
            "us": ms * 1000,
            "alpha": 1,
            "beta": 0,
            "enable_streamk": False,
        }
        benchmark_results.append(metrics)
        
        # Increment counter
        count = count + 1
        
        # Write every N entries
        if count % write_csv_freq == 0 and output_csv:
            write_csv(output_csv, benchmark_results)
        
        if check_correctness:
            print("correctness: ", end=" ", flush=True)
            test_matmul_fp4(m, n, k, out_dtype, transA, transB, init_type)
    
    return benchmark_results


def write_csv(filename: str, results):
    """Write benchmark results to CSV file."""
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
        description="Benchmark FP4 matmul performance and optionally output metrics to CSV."
    )
    parser.add_argument(
        "--input-yaml",
        type=str,
        required=True,
        help="Input YAML file containing FP4 benchmark cases.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Filename for CSV output (if not specified, CSV output is disabled).",
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="randn",
        choices=["hpl", "trig_float", "zeros", "randn"],
        help="Tensor initialization type (default: randn).",
    )
    parser.add_argument(
        "--shuffle-bench",
        action="store_true",
        help="Randomly shuffle the order the benchmark runs.",
    )
    parser.add_argument(
        "--csv-write-freq",
        type=int,
        default=1000,
        help="Number of problems to run before writing to CSV.",
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
        help="Check result correctness against PyTorch reference.",
    )
    args = parser.parse_args()
    
    benchmark_results = bench_matmul_fp4(
        args.input_yaml,
        args.init_type,
        shuffle_benchmark=args.shuffle_bench,
        output_csv=args.output_csv,
        write_csv_freq=args.csv_write_freq,
        print_verbose=args.print_verbose,
        check_correctness=args.checkcorrectness,
    )
    
    if args.output_csv:
        write_csv(args.output_csv, benchmark_results)

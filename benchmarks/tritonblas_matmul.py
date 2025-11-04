#!/usr/bin/env python3
"""
TritonBLAS Unified Matrix Multiplication Benchmark

This benchmark script supports both standard (fp16/bf16/fp32) and quantized (fp8/int8) 
matrix multiplication. It automatically detects the dtype and uses the appropriate API.
"""
import yaml
import argparse
import torch
import triton
import random
import tritonblas
import csv
from typing import Optional, Tuple, Union
from tqdm import tqdm

# Import utilities from the tritonblas package
from tritonblas.utils import (
    str_to_dtype,
    matmul_input_gen,
    _is_quantized,
    _is_float8_like,
    _is_int8,
)


def _is_quantized_dtype(dtype):
    """Check if dtype requires quantization (fp8/int8)"""
    return _is_float8_like(dtype) or _is_int8(dtype)


def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk):
    """Test matmul with proper input generation - handles both quantized and non-quantized dtypes"""

    # Adjust dimensions for transposition and apply tensor.T if needed
    if transA == "T":
        A_size = (m, k)  # A is MxK
    else:
        A_size = (k, m)  # A is KxM (we will later transpose it with .T)

    if transB == "T":
        B_size = (k, n)  # B is KxN
    else:
        B_size = (n, k)  # B is NxK (we will later transpose it with .T)

    # Generate inputs using aiter's approach for quantized dtypes
    if _is_quantized_dtype(in_dtype):
        # Generate base tensors in float32 first
        if transA == "T":
            A = torch.randn((m, k), dtype=torch.float32, device="cuda")
        else:
            A = torch.randn((k, m), dtype=torch.float32, device="cuda").T

        if transB == "T":
            B = torch.randn((k, n), dtype=torch.float32, device="cuda")
        else:
            B = torch.randn((n, k), dtype=torch.float32, device="cuda").T

        # Now A is (M, K) and B is (K, N) after transposes
        # Quantize using aiter's method
        dtype_max_val = torch.finfo(in_dtype).max if in_dtype.is_floating_point else 127.0

        # x_scale: per-row quantization for A → (M, 1)
        max_A = A.abs().float().amax(dim=1, keepdim=True)
        scaleA = max_A / dtype_max_val
        A = (A / scaleA).to(in_dtype)

        # w_scale: per-column quantization for B (along dim=1) → (1, N)
        # B is (K, N), we want one scale per output channel (N dimension)
        max_B = B.abs().float().amax(dim=0, keepdim=True)  # (1, N)
        scaleB = max_B / dtype_max_val
        B = (B / scaleB).to(in_dtype)
    else:
        # Non-quantized dtypes: generate tensors directly
        A = torch.randn(A_size, device="cuda", dtype=in_dtype)
        B = torch.randn(B_size, device="cuda", dtype=in_dtype)
        scaleA = scaleB = None

        # Apply transpose on A or B if necessary (only needed for "N" case)
        if transA == "N":
            A = A.T
        if transB == "N":
            B = B.T

    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    # Run TritonBLAS matmul - use appropriate API based on quantization
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    if _is_quantized_dtype(in_dtype) and scaleA is not None and scaleB is not None:
        # scaleA is (M, 1), scaleB is (1, N) - squeeze to 1D for kernel
        scaleA_1d = scaleA.squeeze(-1)  # (M, 1) → (M,)
        scaleB_1d = scaleB.squeeze(0)   # (1, N) → (N,)
        tritonblas.matmul_a8w8_lt(A, B, scaleA_1d, scaleB_1d, C, selector, enable_streamk)
    else:
        tritonblas.matmul_lt(A, B, C, selector, enable_streamk)

    # Check correctness
    if _is_quantized_dtype(in_dtype) and scaleA is not None and scaleB is not None:
        # Use aiter's reference computation approach
        # scaleA is (M, 1), scaleB is (1, N)

        # 1. Matrix multiplication in float32
        acc = torch.matmul(A.to(torch.float32), B.to(torch.float32))

        # 2. Compute scale matrix: (M, 1) @ (1, N) -> (M, N)
        scale = torch.matmul(scaleA, scaleB)

        # 3. Apply scale
        acc = acc * scale

        # 4. Convert to output dtype with clamping
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

        # Use tolerance settings based on output dtype
        # For quantized outputs (fp8/int8), errors accumulate more than bf16
        if out_dtype == torch.bfloat16:
            # Quantized input + bfloat16 output (matches aiter test_gemm_a8w8.py)
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=0.02, rtol=1e-2)
        elif _is_float8_like(out_dtype):
            # FP8 has very limited precision, need relaxed tolerance
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2)
        elif _is_int8(out_dtype):
            # INT8 quantization can have larger errors
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=2.0, rtol=0.2)
        else:
            # Default for other dtypes
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=0.1, rtol=0.05)
    else:
        # Non-quantized path: simple matmul
        torch_c = torch.matmul(A, B)
        # Convert both to float32 for comparison to avoid dtype mismatch issues
        # Use dtype-specific tolerances: fp16/bf16 inputs need relaxed tolerances due to limited precision
        # Check input dtype since errors accumulate from input precision, even if output is fp32
        if in_dtype == torch.float16 or out_dtype == torch.float16:
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05)
        elif in_dtype == torch.bfloat16 or out_dtype == torch.bfloat16:
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05)
        elif in_dtype == torch.float32 and out_dtype == torch.float32:
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=1e-2, rtol=1e-3)
        else:
            # Fallback for other dtypes or mixed precision
            torch.testing.assert_close(C.to(torch.float32), torch_c.to(torch.float32), atol=0.5, rtol=0.05)

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
        # Generate inputs using aiter's approach for quantized dtypes
        if _is_quantized_dtype(in_dtype):
            # Generate base tensors in float32 first
            if transA == "T":
                A = matmul_input_gen((m, k), torch.float32, init_type, quantize=None)
                A, _ = _is_quantized(A)  # Extract tensor if wrapped
            else:
                A = matmul_input_gen((k, m), torch.float32, init_type, quantize=None)
                A, _ = _is_quantized(A)
                A = A.T

            if transB == "T":
                B = matmul_input_gen((k, n), torch.float32, init_type, quantize=None)
                B, _ = _is_quantized(B)
            else:
                B = matmul_input_gen((n, k), torch.float32, init_type, quantize=None)
                B, _ = _is_quantized(B)
                B = B.T

            # Now A is (M, K) and B is (K, N) after transposes
            # Quantize using aiter's method
            dtype_max_val = torch.finfo(in_dtype).max if in_dtype.is_floating_point else 127.0

            # x_scale: per-row quantization for A → (M, 1)
            max_A = A.abs().float().amax(dim=1, keepdim=True)
            scaleA = max_A / dtype_max_val
            A = (A / scaleA).to(in_dtype)

            # w_scale: per-column quantization for B (along dim=1) → (1, N)
            # B is (K, N), we want one scale per output channel (N dimension)
            max_B = B.abs().float().amax(dim=0, keepdim=True)  # (1, N)
            scaleB = max_B / dtype_max_val
            B = (B / scaleB).to(in_dtype)
        else:
            # Non-quantized dtypes
            if transA == "T":
                A_size = (m, k)
            else:
                A_size = (k, m)

            if transB == "T":
                B_size = (k, n)
            else:
                B_size = (n, k)

            A_init = matmul_input_gen(A_size, in_dtype, init_type, quantize=None)
            B_init = matmul_input_gen(B_size, in_dtype, init_type, quantize=None)
            A, scaleA = _is_quantized(A_init)
            B, scaleB = _is_quantized(B_init)

            # Apply transpose if needed
            if transA == "N":
                A = A.T
            if transB == "N":
                B = B.T

        C = torch.zeros((m, n), device="cuda", dtype=out_dtype)

        # Compute performance metrics
        flops = lambda: 2 * m * n * k * 1e-12
        gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
        # Include scale tensors in byte count for quantized dtypes
        bytes_fn = lambda: (
            A.numel() * A.element_size()
            + B.numel() * B.element_size()
            + C.numel() * C.element_size()
            + (0 if scaleA is None else scaleA.numel() * scaleA.element_size())
            + (0 if scaleB is None else scaleB.numel() * scaleB.element_size())
        )

        # Build a tritonBLAS selector config and launch matmul
        selector = tritonblas.MatmulHeuristicResult(
            m, n, k, A.dtype, B.dtype, C.dtype
        )
        config = selector.get_config()

        # Use appropriate API based on quantization
        if _is_quantized_dtype(in_dtype) and scaleA is not None and scaleB is not None:
            # Handle scale shapes for quantized path
            scaleA_expanded = scaleA.squeeze(-1) if scaleA is not None else None
            if scaleB is not None:
                if scaleB.shape[1] == 1:  # (N, 1) case
                    scaleB_expanded = scaleB.squeeze(-1)
                else:  # (1, N) case
                    scaleB_expanded = scaleB.squeeze(0)
            else:
                scaleB_expanded = None
            matmul = lambda: tritonblas.matmul_a8w8_lt(A, B, scaleA_expanded, scaleB_expanded, C, selector, enable_streamk)
        else:
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

        # Write every N entries
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

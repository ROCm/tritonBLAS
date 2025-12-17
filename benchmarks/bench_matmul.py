#!/usr/bin/env python3
"""
TritonBLAS Matrix Multiplication Benchmark with Accuracy Validation

This benchmark compares different matrix multiplication implementations:
- torch.matmul (baseline reference)
- torch.mm with environment override
- torch.compile (Inductor Triton kernels)
- TritonBLAS (Stream-K implementation)

Features:
- Performance comparison (TFLOPS, latency)
- Numerical accuracy validation against torch.matmul reference
- Support for various data types (fp16, fp32, bf16, int8)
- CSV export for analysis
- Configurable accuracy tolerances

Usage Examples:
  # Basic performance benchmark
  python bench_matmul.py --input-yaml config.yaml --print-verbose

  # With accuracy checking
  python bench_matmul.py --input-yaml config.yaml --check-accuracy --print-verbose

  # Compare all implementations with accuracy
  python bench_matmul.py --input-yaml config.yaml --check-accuracy \\
    --enable-triton-sk --enable-streamk --torch-compile --enable-mm-env \\
    --output-csv results.csv --print-verbose

  # Custom accuracy tolerance for fp16
  python bench_matmul.py --input-yaml config.yaml --check-accuracy \\
    --accuracy-tolerance 1e-2 --enable-triton-sk
"""

import yaml
import argparse
import torch
import triton
import random
import tritonblas
import csv
import os
from tqdm import tqdm

# ============================================================
# Helper functions
# ============================================================

def setup_torch_compile_config():
    import torch._inductor.config
    import torch._dynamo.config
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.unique_user_kernel_names = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.freezing = True
    torch._inductor.config.max_autotune = True
    torch._dynamo.config.recompile_limit = 256


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.replace("torch.", "")
    return getattr(torch, dtype_str)


def init_by_size_and_type(size, dtype, init_type):
    if init_type == "hpl":
        return torch.empty(size, device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        M, N = size
        return torch.arange(0, M * N, device="cuda", dtype=torch.float32).reshape(M, N).sin().to(dtype=dtype)
    elif init_type == "zeros":
        return torch.zeros(size, dtype=dtype, device="cuda")
    elif init_type == "randn":
        return torch.randn(size, dtype=torch.float32, device="cuda").to(dtype)
    elif init_type == "increasing":
        # Create tensor with values increasing from 0 to M-1 along rows and 0 to N-1 along columns
        # Each element (i, j) = i + j (normalized to avoid overflow for large matrices)
        M, N = size
        row_indices = torch.arange(0, M, device="cuda", dtype=torch.float32).unsqueeze(1)  # Shape: (M, 1)
        col_indices = torch.arange(0, N, device="cuda", dtype=torch.float32).unsqueeze(0)  # Shape: (1, N)
        result = (row_indices + col_indices) / max(M, N)  # Normalize to avoid large values
        return result.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")


def print_tensor_debug(name, tensor, max_elements=16):
    """
    Print tensor info and sample values for debugging.

    Args:
        name: str - Name of the tensor for display
        tensor: torch.Tensor - Tensor to print
        max_elements: int - Maximum number of elements to show
    """
    print(f"\n{'='*60}")
    print(f"  {name}:")
    print(f"    Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"    Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.float().mean().item():.6f}")

    # Print corner values for visualization
    m, n = tensor.shape
    rows_to_show = min(4, m)
    cols_to_show = min(8, n)

    print(f"    Top-left {rows_to_show}x{cols_to_show} corner:")
    for i in range(rows_to_show):
        row_vals = [f"{tensor[i, j].item():10.4f}" for j in range(cols_to_show)]
        suffix = " ..." if n > cols_to_show else ""
        print(f"      [{', '.join(row_vals)}{suffix}]")
    if m > rows_to_show:
        print(f"      ... ({m - rows_to_show} more rows)")
    print(f"{'='*60}")


def print_differences(reference, test, max_diffs=20):
    """
    Print only the positions where reference and test differ.

    Args:
        reference: torch.Tensor - Reference result
        test: torch.Tensor - Test result to compare
        max_diffs: int - Maximum number of differences to print
    """
    diff = torch.abs(reference - test)
    # Find non-zero differences
    nonzero_mask = diff > 0
    nonzero_indices = torch.nonzero(nonzero_mask)

    num_diffs = nonzero_indices.shape[0]
    print(f"\n{'='*60}")
    print(f"  DIFFERENCES FOUND: {num_diffs} elements differ")
    print(f"{'='*60}")

    if num_diffs == 0:
        print("  No differences found!")
        return

    # Sort by error magnitude (largest first)
    diff_values = diff[nonzero_mask]
    sorted_indices = torch.argsort(diff_values, descending=True)

    print(f"  Showing top {min(max_diffs, num_diffs)} differences (sorted by magnitude):")
    print(f"  {'Index':<15} {'Reference':<15} {'Test':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
    print(f"  {'-'*75}")

    for i in range(min(max_diffs, num_diffs)):
        idx = sorted_indices[i]
        pos = nonzero_indices[idx]
        row, col = pos[0].item(), pos[1].item()
        ref_val = reference[row, col].item()
        test_val = test[row, col].item()
        abs_diff = diff[row, col].item()
        rel_diff = abs_diff / abs(ref_val) if abs(ref_val) > 1e-8 else abs_diff

        print(f"  [{row:4d},{col:4d}]     {ref_val:<15.4f} {test_val:<15.4f} {abs_diff:<15.4f} {rel_diff:<15.6f}")

    if num_diffs > max_diffs:
        print(f"  ... and {num_diffs - max_diffs} more differences")
    print(f"{'='*60}")


def check_accuracy(reference_result, test_result, impl_name, tolerance=None, relative_tolerance=None):
    """
    Check numerical accuracy between reference and test implementations.

    Uses torch.testing.assert_close() style comparison with dtype-appropriate tolerances.
    This matches the tritonblas_matmul.py reference implementation.

    Tolerance guidelines (from tritonblas_matmul.py):
    - bf16/fp16: atol=0.5, rtol=0.05
    - fp32: atol=1e-2, rtol=1e-3
    - fp8/int8 (quantized): atol=2.0, rtol=0.2

    Args:
        reference_result: torch.Tensor - Reference result from torch.matmul
        test_result: torch.Tensor - Result from implementation being tested
        impl_name: str - Name of implementation for logging
        tolerance: float - Absolute tolerance (atol). If None, uses dtype-based default.
        relative_tolerance: float - Relative tolerance (rtol). If None, uses dtype-based default.

    Returns:
        tuple: (is_accurate: bool, max_abs_error: float, max_rel_error: float)
    """
    if reference_result.shape != test_result.shape:
        print(f"❌ {impl_name}: Shape mismatch! Reference: {reference_result.shape}, Test: {test_result.shape}")
        return False, float('inf'), float('inf')

    # Determine dtype-appropriate tolerances if not specified
    # Based on tritonblas_matmul.py reference implementation
    dtype = reference_result.dtype
    if tolerance is None or relative_tolerance is None:
        if dtype in (torch.float16, torch.bfloat16):
            default_atol, default_rtol = 0.5, 0.05
        elif dtype == torch.float32:
            default_atol, default_rtol = 1e-2, 1e-3
        elif dtype in (torch.int8, torch.uint8):
            default_atol, default_rtol = 2.0, 0.2
        elif hasattr(torch, 'float8_e4m3fn') and dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            default_atol, default_rtol = 2.0, 0.2
        else:
            default_atol, default_rtol = 0.5, 0.05

        if tolerance is None:
            tolerance = default_atol
        if relative_tolerance is None:
            relative_tolerance = default_rtol

    # Convert to float32 for comparison (matches tritonblas_matmul.py)
    ref_f32 = reference_result.to(torch.float32)
    test_f32 = test_result.to(torch.float32)

    # Calculate absolute error
    abs_error = torch.abs(ref_f32 - test_f32)
    max_abs_error = torch.max(abs_error).item()

    # Calculate relative error only for elements with significant magnitude
    ref_abs = torch.abs(ref_f32)
    magnitude_threshold = 1.0
    significant_mask = ref_abs > magnitude_threshold

    if significant_mask.any():
        rel_error_significant = abs_error[significant_mask] / ref_abs[significant_mask]
        max_rel_error = torch.max(rel_error_significant).item()
    else:
        max_rel_error = 0.0

    # Use torch.testing.assert_close style check
    # This is what tritonblas_matmul.py uses for verification
    try:
        torch.testing.assert_close(
            test_f32, ref_f32,
            atol=tolerance, rtol=relative_tolerance
        )
        is_accurate = True
    except AssertionError:
        is_accurate = False

    return is_accurate, max_abs_error, max_rel_error


# ============================================================
# Benchmark core
# ============================================================

def bench_matmul(
    input_yaml: str,
    init_type: str,
    print_verbose=False,
    shuffle_benchmark=True,
    output_csv=None,
    write_csv_freq=100,
    enable_streamk=False,
    torch_compile=False,
    enable_triton_sk=False,
    dynamic=False,
    enable_mm_env=False,
    enable_accuracy_check=False,
    accuracy_tolerance=1e-3,
):
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)

    if torch_compile:
        setup_torch_compile_config()

    benchmark_results = []

    dataset_tuples = [
        (
            case["m"], case["n"], case["k"],
            str_to_dtype(case["in_dtype"]),
            str_to_dtype(case["out_dtype"]),
            case["transA"], case["transB"],
        )
        for case in dataset
    ]
    if shuffle_benchmark:
        random.shuffle(dataset_tuples)

    count = 0

    for m, n, k, in_dtype, out_dtype, transA, transB in (
        tqdm(dataset_tuples) if not print_verbose else dataset_tuples
    ):
        # === prepare shapes ===
        if transA == "T": A_size = (m, k)
        else: A_size = (k, m)
        if transB == "T": B_size = (k, n)
        else: B_size = (n, k)

        A = init_by_size_and_type(A_size, in_dtype, init_type)
        B = init_by_size_and_type(B_size, in_dtype, init_type)
        if transA == "N": A = A.T
        if transB == "N": B = B.T
        C = torch.zeros((m, n), device="cuda", dtype=out_dtype)

        # FLOPs & TFLOPs conversion
        flops = 2 * m * n * k
        tflops = lambda ms: (flops * 1e-12) / (ms * 1e-3)

        # ------------------------------------------------------------
        # 1️⃣ Torch.matmul (baseline)
        # ------------------------------------------------------------
        ms_torch = triton.testing.do_bench(lambda: torch.matmul(A, B), warmup=20, rep=20)
        perf_torch = tflops(ms_torch)

        # ------------------------------------------------------------
        # 2️⃣ Torch.mm with env override
        # ------------------------------------------------------------
        ms_mm_env, perf_mm_env = None, None
        if enable_mm_env:
            old_env = os.environ.get("TENSILE_SOLUTION_SELECTION_METHOD", None)
            os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = "2"
            ms_mm_env = triton.testing.do_bench(lambda: torch.mm(A, B), warmup=20, rep=20)
            perf_mm_env = tflops(ms_mm_env)
            # restore env
            if old_env is not None:
                os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = old_env
            else:
                del os.environ["TENSILE_SOLUTION_SELECTION_METHOD"]

        # ------------------------------------------------------------
        # 3️⃣ Torch.compile (Inductor Triton kernel)
        # ------------------------------------------------------------
        ms_compile, perf_compile = None, None
        if torch_compile:
            compiled_fn = torch.compile(torch.matmul, dynamic=dynamic)
            ms_compile = triton.testing.do_bench(lambda: compiled_fn(A, B), warmup=20, rep=20)
            perf_compile = tflops(ms_compile)

        # ------------------------------------------------------------
        # 4️⃣ TritonBLAS (Stream-K)
        # ------------------------------------------------------------
        ms_triton, perf_triton = None, None
        if enable_triton_sk:
            selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
            config = selector.get_config()
            matmul_triton = lambda: tritonblas.matmul(A, B, C, enable_streamk=enable_streamk)
            ms_triton = triton.testing.do_bench(matmul_triton, warmup=20, rep=20)
            perf_triton = tflops(ms_triton)

        # ------------------------------------------------------------
        # Accuracy Check (if enabled)
        # ------------------------------------------------------------
        accuracy_mm_env, accuracy_compile, accuracy_triton = None, None, None
        max_abs_error_mm_env, max_abs_error_compile, max_abs_error_triton = None, None, None
        max_rel_error_mm_env, max_rel_error_compile, max_rel_error_triton = None, None, None

        if enable_accuracy_check:
            # Compute reference result (torch.matmul)
            reference_result = torch.matmul(A, B)

            # Check torch.mm with env override
            if enable_mm_env:
                old_env = os.environ.get("TENSILE_SOLUTION_SELECTION_METHOD", None)
                os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = "2"
                mm_env_result = torch.mm(A, B)
                accuracy_mm_env, max_abs_error_mm_env, max_rel_error_mm_env = check_accuracy(
                    reference_result, mm_env_result, "torch.mm(env=2)", accuracy_tolerance, accuracy_tolerance
                )
                # restore env
                if old_env is not None:
                    os.environ["TENSILE_SOLUTION_SELECTION_METHOD"] = old_env
                else:
                    del os.environ["TENSILE_SOLUTION_SELECTION_METHOD"]

            # Check torch.compile
            if torch_compile:
                compiled_fn = torch.compile(torch.matmul, dynamic=dynamic)
                compile_result = compiled_fn(A, B)
                accuracy_compile, max_abs_error_compile, max_rel_error_compile = check_accuracy(
                    reference_result, compile_result, "torch.compile", accuracy_tolerance, accuracy_tolerance
                )

                # Print debug info when accuracy check fails
                if not accuracy_compile and print_verbose:
                    print(f"\n❌ torch.compile ACCURACY FAILURE for [M={m},N={n},K={k}]")
                    # Print only the differences (most useful for debugging)
                    print_differences(reference_result, compile_result, max_diffs=50)
                    # Also print summary stats
                    diff = torch.abs(reference_result - compile_result)
                    max_idx = torch.argmax(diff)
                    max_row = max_idx // n
                    max_col = max_idx % n
                    print(f"  Max error location: [{max_row}, {max_col}]")
                    print(f"  Reference value: {reference_result[max_row, max_col].item():.6f}")
                    print(f"  torch.compile value: {compile_result[max_row, max_col].item():.6f}")
                    print(f"  Difference: {diff[max_row, max_col].item():.6e}")

            # Check TritonBLAS
            if enable_triton_sk:
                C_triton = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                triton_result = tritonblas.matmul(A, B, C_triton, enable_streamk=enable_streamk)
                accuracy_triton, max_abs_error_triton, max_rel_error_triton = check_accuracy(
                    reference_result, triton_result, "TritonBLAS", accuracy_tolerance, accuracy_tolerance
                )

                # Print debug info when accuracy check fails
                if not accuracy_triton and print_verbose:
                    print(f"\n❌ TritonBLAS ACCURACY FAILURE for [M={m},N={n},K={k}]")
                    # Print only the differences (most useful for debugging)
                    print_differences(reference_result, triton_result, max_diffs=50)
                    # Also print summary stats
                    diff = torch.abs(reference_result - triton_result)
                    max_idx = torch.argmax(diff)
                    max_row = max_idx // n
                    max_col = max_idx % n
                    print(f"  Max error location: [{max_row}, {max_col}]")
                    print(f"  Reference value: {reference_result[max_row, max_col].item():.6f}")
                    print(f"  TritonBLAS value: {triton_result[max_row, max_col].item():.6f}")
                    print(f"  Difference: {diff[max_row, max_col].item():.6e}")

        # ------------------------------------------------------------
        # Speedup vs Torch baseline
        # ------------------------------------------------------------
        speedup_mm_env = perf_mm_env / perf_torch if perf_mm_env else None
        speedup_compile = perf_compile / perf_torch if perf_compile else None
        speedup_triton = perf_triton / perf_torch if perf_triton else None

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        if print_verbose:
            msg = (
                f"[M={m},N={n},K={k},trans={transA}{transB}] dtype={in_dtype} | "
                f"Torch={perf_torch:.3f} TF/s ({ms_torch:.2f} ms) "
            )
            if perf_mm_env:
                acc_str = ""
                if enable_accuracy_check and accuracy_mm_env is not None:
                    acc_str = f", {'✅' if accuracy_mm_env else '❌'}acc"
                msg += f"| mm(env=2)={perf_mm_env:.3f} ({ms_mm_env:.2f} ms, {speedup_mm_env:.2f}x{acc_str}) "
            if perf_compile:
                acc_str = ""
                if enable_accuracy_check and accuracy_compile is not None:
                    acc_str = f", {'✅' if accuracy_compile else '❌'}acc"
                msg += f"| compile={perf_compile:.3f} ({ms_compile:.2f} ms, {speedup_compile:.2f}x{acc_str}) "
            if perf_triton:
                acc_str = ""
                if enable_accuracy_check and accuracy_triton is not None:
                    acc_str = f", {'✅' if accuracy_triton else '❌'}acc"
                msg += f"| Triton={perf_triton:.3f} ({ms_triton:.2f} ms, {speedup_triton:.2f}x{acc_str})"
            print(msg)

            # Print detailed accuracy info if enabled and verbose
            if enable_accuracy_check and any([accuracy_mm_env is not None, accuracy_compile is not None, accuracy_triton is not None]):
                acc_details = "    Accuracy Details: "
                if accuracy_mm_env is not None:
                    status = "✅" if accuracy_mm_env else "❌"
                    acc_details += f"mm(env=2): {status} (abs_err={max_abs_error_mm_env:.2e}, rel_err={max_rel_error_mm_env:.2e}) "
                if accuracy_compile is not None:
                    status = "✅" if accuracy_compile else "❌"
                    acc_details += f"compile: {status} (abs_err={max_abs_error_compile:.2e}, rel_err={max_rel_error_compile:.2e}) "
                if accuracy_triton is not None:
                    status = "✅" if accuracy_triton else "❌"
                    acc_details += f"Triton: {status} (abs_err={max_abs_error_triton:.2e}, rel_err={max_rel_error_triton:.2e}) "
                print(acc_details)

        # ------------------------------------------------------------
        # Record
        # ------------------------------------------------------------
        metrics = {
            "m": m, "n": n, "k": k,
            "in_dtype": str(in_dtype), "out_dtype": str(out_dtype),
            "transA": transA, "transB": transB,
            "flops": flops,
            "torch_tflops": perf_torch,
            "torch_mm_env_tflops": perf_mm_env,
            "torch_compile_tflops": perf_compile,
            "tritonblas_tflops": perf_triton,
            "speedup_mm_env_vs_torch": speedup_mm_env,
            "speedup_compile_vs_torch": speedup_compile,
            "speedup_triton_vs_torch": speedup_triton,
            "ms_torch": ms_torch,
            "ms_mm_env": ms_mm_env,
            "ms_compile": ms_compile,
            "ms_triton": ms_triton,
            "enable_streamk": enable_streamk,
            "torch_compile_dynamic": dynamic,
            # Accuracy metrics
            "accuracy_mm_env": accuracy_mm_env,
            "accuracy_compile": accuracy_compile,
            "accuracy_triton": accuracy_triton,
            "max_abs_error_mm_env": max_abs_error_mm_env,
            "max_abs_error_compile": max_abs_error_compile,
            "max_abs_error_triton": max_abs_error_triton,
            "max_rel_error_mm_env": max_rel_error_mm_env,
            "max_rel_error_compile": max_rel_error_compile,
            "max_rel_error_triton": max_rel_error_triton,
            "accuracy_tolerance": accuracy_tolerance if enable_accuracy_check else None,
        }
        benchmark_results.append(metrics)

        if output_csv and count % write_csv_freq == 0:
            write_csv(output_csv, benchmark_results)
        count += 1

    return benchmark_results


def write_csv(filename: str, results):
    fieldnames = [
        "m","n","k","transA","transB","in_dtype","out_dtype",
        "flops",
        "torch_tflops","torch_mm_env_tflops","torch_compile_tflops","tritonblas_tflops",
        "speedup_mm_env_vs_torch","speedup_compile_vs_torch","speedup_triton_vs_torch",
        "ms_torch","ms_mm_env","ms_compile","ms_triton",
        "enable_streamk","torch_compile_dynamic",
        # Accuracy fields
        "accuracy_mm_env","accuracy_compile","accuracy_triton",
        "max_abs_error_mm_env","max_abs_error_compile","max_abs_error_triton",
        "max_rel_error_mm_env","max_rel_error_compile","max_rel_error_triton",
        "accuracy_tolerance"
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✅ Results saved to '{filename}'")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Torch, Torch(mm env=2), Torch.compile, TritonBLAS performance and accuracy (TFLOPS, ms, errors vs torch.matmul baseline)."
    )
    parser.add_argument("--input-yaml", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--init_type", type=str, default="randn",
                        choices=["hpl","trig_float","zeros","randn","increasing"],
                        help="Initialization type: randn (random normal), hpl (uniform -0.5 to 0.5), "
                             "trig_float (sin of indices), zeros, increasing (i+j normalized)")
    parser.add_argument("--shuffle-bench", action="store_true")
    parser.add_argument("--csv-write-freq", type=int, default=1000)
    parser.add_argument("--print-verbose", action="store_true")
    parser.add_argument("--enable-triton-sk", action="store_true")
    parser.add_argument("--enable-streamk", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--enable-mm-env", action="store_true",
                        help="Benchmark torch.mm with TENSILE_SOLUTION_SELECTION_METHOD=2")
    parser.add_argument("--check-accuracy", action="store_true",
                        help="Check numerical accuracy against torch.matmul reference")
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-2,
                        help="Tolerance for accuracy checks (default: 1e-2)")
    args = parser.parse_args()

    results = bench_matmul(
        args.input_yaml,
        args.init_type,
        shuffle_benchmark=args.shuffle_bench,
        output_csv=args.output_csv,
        write_csv_freq=args.csv_write_freq,
        print_verbose=args.print_verbose,
        enable_streamk=args.enable_streamk,
        torch_compile=args.torch_compile,
        enable_triton_sk=args.enable_triton_sk,
        dynamic=args.dynamic,
        enable_mm_env=args.enable_mm_env,
        enable_accuracy_check=args.check_accuracy,
        accuracy_tolerance=args.accuracy_tolerance,
    )

    if args.output_csv:
        write_csv(args.output_csv, results)


#!/usr/bin/env python3
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
    check_correctness=False,
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
        tflops = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
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
        perf = tflops(ms)

        if print_verbose:
            print(
                f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype}, init={init_type}, perf={perf}(TFLOPs) selected_tile={selector.config[0]}x{selector.config[1]}x{selector.config[2]}"
            )

        metrics = {
            "m": m,
            "n": n,
            "k": k,
            "mnk": m * n * k,
            "macro_tile": f"{config[0]}x{config[1]}x{config[2]}",
            "bytes": bytes_fn(),
            "flops": flops(),
            "tritonblas_tflops": perf,
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

        if check_correctness:
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
        "tritonblas_tflops",
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

def main():
    # Kernel A
    m = 8192
    n = 2048
    k = 256
    in_dtype = torch.float16
    out_dtype = torch.float16
    transA = "N"
    transB = "N"
    warmup = 20
    rep = 500

    kernel_a_inputs = generate_matmul_inputs(
        m, n, k, in_dtype, out_dtype, transA, transB, "randn"
    )

    # Compute performance metrics
    flops = lambda: 2 * m * n * k * 1e-12
    tflops = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
    # Include scale tensors in byte count for quantized dtypes
    kernel_a_bytes_fn = lambda: (
        kernel_a_inputs.A.numel() * kernel_a_inputs.A.element_size()
        + kernel_a_inputs.B.numel() * kernel_a_inputs.B.element_size()
        + kernel_a_inputs.C.numel() * kernel_a_inputs.C.element_size()
        + (
            0
            if kernel_a_inputs.scaleA is None
            else kernel_a_inputs.scaleA.numel() * kernel_a_inputs.scaleA.element_size()
        )
        + (
            0
            if kernel_a_inputs.scaleB is None
            else kernel_a_inputs.scaleB.numel() * kernel_a_inputs.scaleB.element_size()
        )
    )

    # Build a tritonBLAS selector config and launch matmul
    kernel_a_selector = tritonblas.MatmulHeuristicResult(
        m, n, k, kernel_a_inputs.A.dtype, kernel_a_inputs.B.dtype, kernel_a_inputs.C.dtype
    )
    kernel_a_config = kernel_a_selector.get_config()

    kernel_a_matmul = lambda: tritonblas.matmul(
        kernel_a_inputs.A, kernel_a_inputs.B, kernel_a_inputs.C, enable_streamk=False
    )

    kernel_a_ms = triton.testing.do_bench(kernel_a_matmul, warmup=warmup, rep=rep)
    kernel_a_perf = tflops(kernel_a_ms)

    print(f"\n\nKernel A GEMM: \nm={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype}, init=randn, \nperf={kernel_a_perf}(TFLOPs), ms={kernel_a_ms}\nselected_tile={kernel_a_config[0]}x{kernel_a_config[1]}x{kernel_a_config[2]}")

    # Kernel B
    p = 256
    kernel_b_inputs = generate_matmul_inputs(
        m, p, n, in_dtype, out_dtype, transA, transB, "randn"
    )

    kernel_b_bytes_fn = lambda: (
        kernel_b_inputs.A.numel() * kernel_b_inputs.A.element_size()
        + kernel_b_inputs.B.numel() * kernel_b_inputs.B.element_size()
        + kernel_b_inputs.C.numel() * kernel_b_inputs.C.element_size()
        + (
            0
            if kernel_b_inputs.scaleA is None
            else kernel_b_inputs.scaleA.numel() * kernel_b_inputs.scaleA.element_size()
        )
        + (
            0
            if kernel_b_inputs.scaleB is None
            else kernel_b_inputs.scaleB.numel() * kernel_b_inputs.scaleB.element_size()
        )
    )

    kernel_b_selector = tritonblas.MatmulHeuristicResult(
        m, p, n, kernel_b_inputs.A.dtype, kernel_b_inputs.B.dtype, kernel_b_inputs.C.dtype
    )
    kernel_b_config = kernel_b_selector.get_config()

    kernel_b_matmul = lambda: tritonblas.matmul(
        kernel_b_inputs.A, kernel_b_inputs.B, kernel_b_inputs.C, enable_streamk=False
    )

    kernel_b_ms = triton.testing.do_bench(kernel_b_matmul, warmup=warmup, rep=rep)
    kernel_b_perf = tflops(kernel_b_ms)

    print(f"\n\nKernel B GEMM: \nm={m}, p={p}, n={n}, in_dtype={in_dtype}, out_dtype={out_dtype}, init=randn, \nperf={kernel_b_perf}(TFLOPs), ms={kernel_b_ms}\nselected_tile={kernel_b_config[0]}x{kernel_b_config[1]}x{kernel_b_config[2]}")

    def sequential_matmul():
        C_0 = tritonblas.matmul(
            kernel_a_inputs.A, kernel_a_inputs.B, kernel_a_inputs.C, enable_streamk=False
        )
        C_1 = tritonblas.matmul(
            C_0, kernel_b_inputs.B, kernel_b_inputs.C, enable_streamk=False
        )

        return C_1

    sequential_ms = triton.testing.do_bench(sequential_matmul, warmup=warmup, rep=rep)
    sequential_perf = tflops(sequential_ms)

    print(f"\n\nSequential GEMM: \nperf={sequential_perf}(TFLOPs), ms={sequential_ms}")

    # Test correctness of fused_matmul
    print("\n\nTesting correctness of fused_matmul...")
    
    # Create fresh input tensors for correctness test (fused_matmul modifies c0 and c1 in-place)
    test_a = kernel_a_inputs.A.clone()
    test_b0 = kernel_a_inputs.B.clone()
    test_c0 = torch.zeros_like(kernel_a_inputs.C)
    test_b1 = kernel_b_inputs.B.clone()
    test_c1 = torch.zeros_like(kernel_b_inputs.C)
    
    # Run fused_matmul
    fused_c0, fused_c1 = tritonblas.fused_matmul(
        test_a, test_b0, test_c0, test_b1, test_c1, 
        kernel_a_selector, kernel_b_selector
    )
    
    # Compute PyTorch baseline
    torch_c0 = torch.matmul(test_a, test_b0)
    torch_c1 = torch.matmul(torch_c0, test_b1)
    
    # Compare results with appropriate tolerances based on dtype
    if in_dtype == torch.float16 or out_dtype == torch.float16:
        atol = 0.5
        rtol = 0.05
    elif in_dtype == torch.bfloat16 or out_dtype == torch.bfloat16:
        atol = 0.5
        rtol = 0.05
    elif in_dtype == torch.float32 and out_dtype == torch.float32:
        atol = 1e-2
        rtol = 1e-3
    else:
        atol = 0.5
        rtol = 0.05
    
    # Test C0 correctness
    try:
        torch.testing.assert_close(
            fused_c0.to(torch.float32), torch_c0.to(torch.float32), 
            atol=atol, rtol=rtol
        )
        print("✅ C0 = A @ B0: Correct")
    except AssertionError as e:
        print(f"❌ C0 = A @ B0: FAILED")
        print(f"   Error: {e}")
        max_diff = (fused_c0.to(torch.float32) - torch_c0.to(torch.float32)).abs().max()
        print(f"   Max difference: {max_diff}")
    
    # Test C1 correctness
    try:
        torch.testing.assert_close(
            fused_c1.to(torch.float32), torch_c1.to(torch.float32), 
            atol=atol, rtol=rtol
        )
        print("✅ C1 = C0 @ B1: Correct")
    except AssertionError as e:
        print(f"❌ C1 = C0 @ B1: FAILED")
        print(f"   Error: {e}")
        max_diff = (fused_c1.to(torch.float32) - torch_c1.to(torch.float32)).abs().max()
        print(f"   Max difference: {max_diff}")
    
    # Benchmark fused_matmul
    def fused_matmul_bench():
        test_c0.zero_()
        test_c1.zero_()
        tritonblas.fused_matmul(
            test_a, test_b0, test_c0, test_b1, test_c1,
            kernel_a_selector, kernel_b_selector
        )
        return test_c1
    
    fused_ms = triton.testing.do_bench(fused_matmul_bench, warmup=warmup, rep=rep)
    # Total FLOPS: 2*m*n*k (alpha) + 2*m*p*n (beta)
    total_flops = 2 * m * n * k + 2 * m * p * n
    fused_perf = total_flops * 1e-12 / (fused_ms * 1e-3)
    
    print(f"\n\nFused GEMM Performance: \nperf={fused_perf:.2f}(TFLOPs), ms={fused_ms:.3f}")
    print(f"Speedup over sequential: {sequential_ms / fused_ms:.2f}x")



if __name__ == "__main__":
    main()

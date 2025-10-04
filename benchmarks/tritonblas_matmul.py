#!/usr/bin/env python3
import yaml
import argparse
import torch
import triton
import random
import tritonblas
import csv
from typing import Optional, Tuple, Union
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

def _ensure_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
    raise TypeError(f"Unsupported dtype spec: {dtype}")

# supports torch.float8_* and, if present, torch.bfloat8
def _is_float8_like(dtype: torch.dtype) -> bool:
    s = str(dtype)
    return ("float8" in s) or ("bfloat8" in s) 

def _is_int8(dtype: torch.dtype) -> bool:
    return dtype == torch.int8

def init_by_size_and_type(
    size: Tuple[int, int],
    dtype: Union[torch.dtype, str],
    init_type: str,
    *,
    quantize: Optional[str] = None,   # None | "auto" | "fp8" | "int8"
    scale_axis: Optional[int] = None, # e.g., A(M,K) per-row over K -> 1; B(K,N) per-col over K -> 0
    min_scale: float = 1e-8,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize a tensor and (optionally) quantize to FP8/BF8 or INT8 with per-axis scale.

    IMPORTANT:
    - For FP16/BF16 (or any non-8-bit dtype), the default `quantize=None` (or `quantize="auto"`)
      simply returns the initialized tensor cast to that dtype — identical to our original behavior.
    - Only when dtype is float8-like or int8 (or you force `quantize="fp8"/"int8"`),  we return (q, scale).

    Returns:
      * If no quantization is performed: Tensor[dtype]
      * If quantization is performed: (q, scale) where scale is float32 and keeps the reduced dim
        (e.g., (M,1) if axis=1 or (1,N) if axis=0).
    """
    dtype = _ensure_dtype(dtype)
    device = "cuda"

    # 1) Initialize in fp32 for stable scale computation
    if init_type == "hpl":
        base = torch.empty(size, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        M, N = size
        base = torch.reshape(
            torch.arange(0, M * N, device=device, dtype=torch.float32), (M, N)
        ).sin()
    elif init_type == "zeros":
        base = torch.zeros(size, dtype=torch.float32, device=device)
    elif init_type == "randn":
        base = torch.randn(size, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")

    # 2) Decide quantization mode
    mode = quantize
    if mode is None:
        # Original behavior: just cast and return
        return base.to(dtype)
    if mode == "auto":
        if _is_float8_like(dtype):
            mode = "fp8"
        elif _is_int8(dtype):
            mode = "int8"
        else:
            # Not an 8-bit target → behave like original
            return base.to(dtype)

    # From here, we are quantizing; need scale_axis
    if scale_axis is None:
        raise ValueError("quantize requires scale_axis (0 or 1).")

    # 3) Per-axis amax → scale
    amax = torch.amax(base.abs(), dim=scale_axis, keepdim=True).to(torch.float32)

    if mode == "fp8":
        max_val = float(torch.finfo(dtype).max)   # e.g., e4m3/e5m2
        scale = torch.clamp(amax / max_val, min=min_scale)
        q = (base / scale).to(dtype)
        return q, scale

    if mode == "int8":
        qmax = 127.0  # symmetric int8 (zero_point = 0)
        scale = torch.clamp(amax / qmax, min=min_scale)
        q_fp = base / scale
        q = torch.round(q_fp).clamp_(-127, 127).to(torch.int8)
        return q, scale

    raise ValueError(f"Unsupported quantize mode: {mode}")

def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk):

    # Adjust dimensions for transposition and apply tensor.T if needed
    if transA == "T":
        A_size = (m, k)  # A is MxK
    else:
        A_size = (k, m)  # A is KxM (we will later transpose it with .T)

    if transB == "T":
        B_size = (k, n)  # B is KxN
    else:
        B_size = (n, k)  # B is NxK (we will later transpose it with .T)

    A = torch.randn(A_size, device="cuda", dtype=in_dtype)
    B = torch.randn(B_size, device="cuda", dtype=in_dtype)

    # Apply transpose on A or B if necessary (only needed for "N" case)
    if transA == "N":
        A = A.T  # Apply transpose to A if transA is "N"

    if transB == "N":
        B = B.T  # Apply transpose to B if transB is "N"

    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    # Run TritonBLAS matmul
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    tritonblas.matmul_lt(A, B, C, selector, enable_streamk)

    # Check correctness:
    torch_c = torch.matmul(A, B)
    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1e-2, rtol=1e-3)
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

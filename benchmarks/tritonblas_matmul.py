#!/usr/bin/env python3
"""
TritonBLAS Unified Matrix Multiplication Benchmark

This benchmark script supports both standard (fp16/bf16/fp32) and quantized (fp8/int8)
matrix multiplication. It automatically detects the dtype and uses the appropriate API.

Supports three kernel modes via CLI flags:
  (default)           Persistent GEMM
  --enable-streamk    Stream-K GEMM
  --work-stealing     Work-stealing persistent GEMM

Optional --cu-sweep runs a balanced CU-mask sweep (MI300X) by re-invoking this
script as subprocesses with ROC_GLOBAL_CU_MASK set.  Results flow into the same
CSV with an extra ``active_cus`` column.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys

import torch  # type: ignore
import triton  # type: ignore
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # type: ignore

from common import get_cu_info, build_balanced_hex_mask  # type: ignore
from tritonblas.utils import generate_matmul_inputs, str_to_dtype, _is_float8_like  # type: ignore


def test_matmul(m, n, k, in_dtype, out_dtype, transA, transB, enable_streamk,
                work_stealing=False, init_type="randn"):
    """Test matmul with proper input generation - handles both quantized and non-quantized dtypes"""

    inputs = generate_matmul_inputs(m, n, k, in_dtype, out_dtype, transA, transB, init_type)
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
        streamk=enable_streamk,
    )
    cfg = tritonblas.matmul_preamble(selector)

    if inputs.is_quantized:
        tritonblas.matmul_a8w8_lt(
            inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, cfg,
            enable_streamk, work_stealing=work_stealing,
        )
    else:
        tritonblas.matmul_lt(
            inputs.A, inputs.B, inputs.C, selector, cfg,
            enable_streamk, work_stealing=work_stealing,
        )

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
    work_stealing=False,
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

        selector = tritonblas.OrigamiMatmulSelector(
            m, n, k, inputs.A.dtype, inputs.B.dtype, inputs.C.dtype, inputs.A.device,
            streamk=enable_streamk
        )
        cfg = tritonblas.matmul_preamble(selector)
        config = (selector.block_m, selector.block_n, selector.block_k)

        if inputs.is_quantized:
            matmul = lambda: tritonblas.matmul_a8w8_lt(
                inputs.A, inputs.B, inputs.scaleA, inputs.scaleB, inputs.C, selector, cfg,
                enable_streamk, work_stealing=work_stealing,
            )
        else:
            matmul = lambda: tritonblas.matmul_lt(
                inputs.A, inputs.B, inputs.C, selector, cfg,
                enable_streamk, work_stealing=work_stealing,
            )

        reset = lambda: cfg.reset(streamk=enable_streamk, work_stealing=work_stealing)
        ms = tritonblas.do_bench(matmul, reset_fn=reset, n_warmup=20, n_repeat=20)
        perf = gflops(ms)

        # Determine mode string for output
        if work_stealing:
            mode_str = "work_stealing"
        elif enable_streamk:
            mode_str = "streamk"
        else:
            mode_str = "persistent"

        if print_verbose:
            print(
                f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype}, "
                f"init={init_type}, mode={mode_str}, perf={perf}(GFLOPs) "
                f"selected_tile={selector.block_m}x{selector.block_n}x{selector.block_k}"
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
            "mode": mode_str,
        }
        benchmark_results.append(metrics)

        # Write every N entries
        if count % write_csv_freq == 0:
            if output_csv:
                write_csv(output_csv, benchmark_results)
        count = count + 1

        if check_correctness:
            print("correctness: ", end=" ", flush=True)
            test_matmul(m, n, k, in_dtype, out_dtype, transA, transB,
                        enable_streamk, work_stealing=work_stealing, init_type=init_type)

    return benchmark_results


def _build_child_cmd(args):
    """Build a subprocess command from parsed args (excludes --cu-sweep and --output-csv)."""
    cmd = [sys.executable, os.path.abspath(__file__)]
    cmd += ["--input-yaml", args.input_yaml]
    cmd += ["--init_type", args.init_type]
    cmd += ["--csv-write-freq", str(args.csv_write_freq)]
    if args.shuffle_bench:
        cmd.append("--shuffle-bench")
    if args.checkcorrectness:
        cmd.append("--checkcorrectness")
    if args.enable_streamk:
        cmd.append("--enable-streamk")
    elif args.work_stealing:
        cmd.append("--work-stealing")
    return cmd


def write_csv(filename: str, results):
    fieldnames = [
        "m", "n", "k", "mnk", "macro_tile", "bytes", "flops",
        "tritonblas_gflops",
        "a_type", "b_type", "c_type", "d_type", "compute_type",
        "in_dtype", "out_dtype", "init_type",
        "transA", "transB", "us", "alpha", "beta", "mode",
    ]
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
        description="Benchmark matmul performance (supports both standard and quantized dtypes) "
                    "and optionally output performance metrics to a CSV file."
    )
    parser.add_argument(
        "--input-yaml", type=str, default="../datasets/matmul_random.yaml",
        help="Input YAML file containing benchmark cases (default: ./matmul_random.yaml).",
    )
    parser.add_argument(
        "--output-csv", type=str, default="",
        help="Filename for CSV output (if not specified, CSV output is disabled).",
    )
    parser.add_argument(
        "--init_type", type=str, default="randn",
        choices=["hpl", "trig_float", "zeros", "randn"],
        help="Tensor initialization type (default: randn).",
    )
    parser.add_argument(
        "--shuffle-bench", action="store_true",
        help="Randomly shuffle the order the benchmark runs",
    )
    parser.add_argument(
        "--csv-write-freq", type=int, default=1000,
        help="Number of problems to run before writing to csv",
    )
    parser.add_argument(
        "--print-verbose", action="store_true",
        help="Print detailed information for each benchmark.",
    )
    parser.add_argument(
        "--checkcorrectness", action="store_true", default=False,
        help="Check result correctness",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--enable-streamk", action="store_true",
        help="Enable Stream-K mode for matrix multiplication.",
    )
    mode_group.add_argument(
        "--work-stealing", action="store_true",
        help="Enable work-stealing persistent GEMM with per-XCD atomic counters.",
    )

    parser.add_argument(
        "--cu-sweep", action="store_true",
        help="Run a balanced CU-mask sweep (MI300X).  Uses the same --input-yaml "
             "shapes and kernel mode.  Re-invokes this script as subprocesses with "
             "ROC_GLOBAL_CU_MASK set; results include an active_cus column.",
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
    benchmark_results = bench_matmul(
        args.input_yaml,
        args.init_type,
        shuffle_benchmark=args.shuffle_bench,
        output_csv=args.output_csv if not is_worker else None,
        write_csv_freq=args.csv_write_freq,
        print_verbose=args.print_verbose if not is_worker else False,
        enable_streamk=args.enable_streamk,
        work_stealing=args.work_stealing,
        check_correctness=args.checkcorrectness,
    )

    if is_worker:
        # Tag each result with CU count and dump JSON to stdout for parent
        for row in benchmark_results:
            row["active_cus"] = args._active_cus
        print(json.dumps(benchmark_results))
        sys.exit(0)

    if args.output_csv:
        write_csv(args.output_csv, benchmark_results)

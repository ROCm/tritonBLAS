# SPDX-License-Identifier: MIT
# Comprehensive FP4 matmul test suite for tritonblas
# Based on aiter's test_gemm_a4w4.py

import torch
import tritonblas
import time
import argparse
import aiter
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_rms_mxfp4_quant,
)
from fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32
from hadamard import hadamard_blocked_transform, generate_hadamard_matrix
torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
import math
from test_hadamard import full_hadamard_triton, fwht_matmul, fwht_torch_reference

def run_torch_reference(x_fp4, w_fp4, x_scales, w_scales, dtype):
    """
    Compute reference result using PyTorch with dequantized FP4 inputs.
    
    This provides the ground truth for correctness validation.
    """
    m, k_packed = x_fp4.shape
    n, k_packed = w_fp4.shape
    k = k_packed * 2
    
    # Dequantize FP4 to FP32
    x_f32 = mxfp4_to_f32(x_fp4)
    w_f32 = mxfp4_to_f32(w_fp4)
    
    # Convert e8m0 scales to FP32 and expand to match data shape
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_scales_f32 = x_scales_f32.repeat_interleave(32, dim=1)
    
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_scales_f32 = w_scales_f32.repeat_interleave(32, dim=1)
    
    # Apply scales
    x_f32 = x_f32 * x_scales_f32
    w_f32 = w_f32 * w_scales_f32
    
    # Compute matmul
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


def benchmark_kernel(func, *args, num_iters=10, warmup=3):
    """Benchmark a kernel with warmup iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iters):
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_us = (end_time - start_time) / num_iters * 1e6
    return avg_time_us


def test_gemm_fp4(dtype, M, N, K, verbose=True):
    """
    Test FP4 GEMM with given dimensions and dtype.
    
    Returns dictionary with performance metrics and error statistics.
    """
    ret = {}
    
    # Generate random input data
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    
    # Quantize to FP4
    x_fp4, x_scales = dynamic_mxfp4_quant(x)
    w_fp4, w_scales = dynamic_mxfp4_quant(w)
    
    # Allocate output
    out = torch.empty((M, N), dtype=dtype)
    
    # Compute reference
    ref = run_torch_reference(x_fp4, w_fp4, x_scales, w_scales, dtype)
    
    # Run tritonblas FP4 matmul
    def run_tritonblas():
        tritonblas.matmul_fp4(x_fp4, w_fp4, out, x_scales, w_scales)
    
    us = benchmark_kernel(run_tritonblas, num_iters=10, warmup=3)
    
    # Compute performance metrics
    total_ops = 2 * M * N * K
    ret["M"] = M
    ret["N"] = N
    ret["K"] = K
    ret["dtype"] = str(dtype)
    ret["us"] = us
    ret["TFLOPS"] = total_ops / us / 1e6
    ret["TB/s"] = (x_fp4.nbytes + w_fp4.nbytes) / us / 1e6
    
    # Compute error metrics
    nan_mask = torch.isnan(out)
    inf_mask = torch.isinf(out)
    valid_mask = ~nan_mask & ~inf_mask
    
    num_valid = valid_mask.sum().item()
    num_nan = nan_mask.sum().item()
    num_inf = inf_mask.sum().item()
    total = M * N
    
    ret["valid_%"] = 100 * num_valid / total
    ret["nan_%"] = 100 * num_nan / total
    ret["inf_%"] = 100 * num_inf / total
    
    # Compute error against reference
    ref_valid_mask = ~torch.isnan(ref) & ~torch.isinf(ref)
    both_valid = valid_mask & ref_valid_mask
    
    if both_valid.sum() > 0:
        out_valid = out[both_valid]
        ref_valid = ref[both_valid]
        
        abs_error = torch.abs(out_valid - ref_valid)
        ret["mean_abs_err"] = abs_error.mean().item()
        ret["max_abs_err"] = abs_error.max().item()
        
        rel_error = abs_error / (torch.abs(ref_valid) + 1e-8)
        ret["mean_rel_err"] = rel_error.mean().item()
    else:
        ret["mean_abs_err"] = float('nan')
        ret["max_abs_err"] = float('nan')
        ret["mean_rel_err"] = float('nan')
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FP4 GEMM Test: M={M}, N={N}, K={K}, dtype={dtype}")
        print(f"{'='*80}")
        print(f"Performance:")
        print(f"  Time: {us:.2f} us")
        print(f"  Throughput: {ret['TFLOPS']:.2f} TFLOPS")
        print(f"  Bandwidth: {ret['TB/s']:.2f} TB/s")
        print(f"Correctness:")
        print(f"  Valid values: {num_valid}/{total} ({ret['valid_%']:.1f}%)")
        print(f"  NaN values: {num_nan}/{total} ({ret['nan_%']:.1f}%)")
        print(f"  Inf values: {num_inf}/{total} ({ret['inf_%']:.1f}%)")
        if both_valid.sum() > 0:
            print(f"Error vs Reference:")
            print(f"  Mean absolute error: {ret['mean_abs_err']:.6f}")
            print(f"  Max absolute error: {ret['max_abs_err']:.6f}")
            print(f"  Mean relative error: {ret['mean_rel_err']:.6f}")
        print(f"{'='*80}\n")
    
    return ret

def benchmark_block_sizes():
    """Sweep block sizes to find optimal configuration."""
    print("\n" + "="*80)
    print("FP4 GEMM Block Size Sweep (8192x8192x8192)")
    print("="*80)
    
    # Use smaller size to avoid OOM in Docker
    M, N, K =  16384, 16384, 16384
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate test data once
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    w_fp4, w_scales = dynamic_mxfp4_quant(w)
    out = torch.empty((M, N), dtype=dtype)

    # for RMSNorm
    weight = torch.randn(K, dtype=dtype)
    eps = 1e-5
    use_model_sensitive_rmsnorm = 0
    quant_dtype = torch.float8_e4m3fnuz
    xq_fused = torch.empty(x.shape, dtype=quant_dtype, device=device)
    xscale_fused = torch.empty(x.shape[0], 1, dtype=torch.float32, device="cuda")
    # x_norm = tritonblas.rms_norm(x, weight, eps, use_model_sensitive_rmsnorm)

    quant_dtype = torch.float8_e4m3fn
    w_fp8 = torch.empty(w.shape, dtype=quant_dtype, device=device)
    w_fp8_scales = torch.empty(w.shape[0], 1, dtype=torch.float32, device=device)
    # w_fp8, w_fp8_scales =  quantize_tensor_per_channel(w.clone(), quant_dtype, axis=1
    x_fp8 = torch.empty(x.shape, dtype=quant_dtype, device=device)
    x_fp8_scales = torch.empty(x.shape[0], 1, dtype=torch.float32, device=device)
    aiter.ops.triton.quant.dynamic_per_tensor_quant_fp8_i8(w_fp8, w, w_fp8_scales)
    # for Hadamard
    had_size = 32

    # Block size configurations to test
    block_m_sizes = [64, 128, 256]
    block_n_sizes = [64, 128, 256]
    block_k_sizes = [128, 256, 512]

    results = {}
    best_tflops = 0
    best_config = None
    print(x.shape)
    for block_k in block_k_sizes:
        results[block_k] = {}
        for block_m in block_m_sizes:
            for block_n in block_n_sizes:
                try:
                    def run_kernel():
                        x_norm = tritonblas.rms_norm(x.clone(), weight, eps, use_model_sensitive_rmsnorm)
                        # x_fp82, x_fp8_scales2 =  quantize_tensor_per_channel(x_norm.clone(), quant_dtype, axis=1)
                        aiter.ops.triton.quant.dynamic_per_tensor_quant_fp8_i8(x_fp8, x_norm, x_fp8_scales)
                        selector = tritonblas.MatmulHeuristicResult(
                            M, N, K, x_fp8.dtype, w_fp8.dtype, out.dtype
                        )
                        tritonblas.matmul_a8w8_lt(x_fp8, w_fp8, x_fp8_scales, w_fp8_scales, out, selector)

                        # x_norm = tritonblas.rms_norm(x, weight, eps, use_model_sensitive_rmsnorm)
                        # (x_fp4, x_scales), _, _ = tritonblas.fused_rms_hadamard_mxfp4_quant(x, weight, eps)
                        # x_norm = tritonblas.rmsnorm2d_fwd_with_dynamicquant(xq_fused, x, xscale_fused, weight, eps)
                        # x_fp4, x_scales = dynamic_mxfp4_quant(x)
                        
                        # tritonblas.matmul_fp4(
                        #     x_fp4, w_fp4, out, x_scales, w_scales,
                        #     block_m=block_m, block_n=block_n, block_k=block_k
                        # )

                        # triton_result = full_hadamard_triton(x) / math.sqrt(had_size)

                        # triton_result = full_hadamard_triton(x.reshape(-1, 32)) / math.sqrt(had_size)

                        # ref_result = fwht_torch_reference(x.clone(), had_size) 

                        # H = generate_hadamard_matrix(had_size).cuda() / math.sqrt(had_size)
                        # blocked_result = hadamard_blocked_transform(x, H) 
                        # fast_blocked_result = tritonblas.hadamard_blocked_fast(x.reshape(-1,had_size)) 
                        # fast_blocked_result = tritonblas.hadamard_blocked_fast(x) 
                        
                    us = benchmark_kernel(run_kernel, num_iters=5, warmup=2)
                    total_ops = 2 * M * N * K
                    tflops = total_ops / us / 1e6
                    
                    key = f"M{block_m}_N{block_n}"
                    results[block_k][key] = us #tflops
                    
                    if tflops > best_tflops:
                        best_tflops = tflops
                        best_config = (block_m, block_n, block_k)
                
                except Exception as e:
                    key = f"M{block_m}_N{block_n}"
                    results[block_k][key] = 0.0
                    print(f"BLK_M={block_m}, BLK_N={block_n}, BLK_K={block_k}: FAILED - {str(e)}")
    
    # Print results table
    print("\nThroughput Table (TFLOPS):")
    print("-" * 80)
    
    # Header
    header = "BLK_K  |"
    for block_m in block_m_sizes:
        for block_n in block_n_sizes:
            header += f" M{block_m:3d}xN{block_n:3d} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for block_k in block_k_sizes:
        row = f"  {block_k:3d}  |"
        for block_m in block_m_sizes:
            for block_n in block_n_sizes:
                key = f"M{block_m}_N{block_n}"
                tflops = results[block_k].get(key, 0.0)
                
                # Highlight best configuration
                if best_config and (block_m, block_n, block_k) == best_config:
                    row += f" *{tflops:6.2f}* |"
                else:
                    row += f"  {tflops:7.2f}  |"
        print(row)
    
    print("-" * 80)
    
    if best_config:
        print(f"\nBest Configuration:")
        print(f"  BLK_M={best_config[0]}, BLK_N={best_config[1]}, BLK_K={best_config[2]}")
        print(f"  Performance: {best_tflops:.2f} TFLOPS")
    
    print("="*80 + "\n")
    return results


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="TritonBLAS FP4 GEMM Test Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for output (default: bf16)"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["all", "correctness", "production", "blocksweep", "single"],
        default="all",
        help="Test mode (default: all)"
    )
    parser.add_argument(
        "--mnk",
        type=str,
        default=None,
        help="Single test size as M,N,K (e.g., --mnk 1024,1024,1024)"
    )
    
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[args.dtype]
    
    print("\n" + "="*80)
    print("TritonBLAS FP4 GEMM Test Suite")
    print("="*80)
    print(f"Output dtype: {dtype}")
    print(f"Test mode: {args.mode}")
    print("="*80)
    
    if args.mode == "blocksweep" or args.mode == "all":
        # Block size sweep
        benchmark_block_sizes()
    
    print("\n" + "="*80)
    print("Test suite completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

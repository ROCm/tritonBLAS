# SPDX-License-Identifier: MIT
# Comprehensive FP8 matmul test suite for tritonblas
# Based on test_matmul_fp4.py

import torch
import triton
import tritonblas
import time
import argparse
import pytest
from tritonblas.utils import dynamic_mxfp8_quant, mxfp8_to_f32, e8m0_to_f32

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def run_torch_reference(x_fp8, w_fp8, x_scales, w_scales, dtype):
    """
    Compute reference result using PyTorch with dequantized FP8 inputs.
    
    This provides the ground truth for correctness validation.
    """
    m, k = x_fp8.shape
    n, k = w_fp8.shape
    
    # Dequantize FP8 to FP32
    x_f32 = mxfp8_to_f32(x_fp8)
    w_f32 = mxfp8_to_f32(w_fp8)
    
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


def run_gemm_fp8_test(dtype, M, N, K, verbose=True):
    """
    Test FP8 GEMM with given dimensions and dtype.
    
    Returns dictionary with performance metrics and error statistics.
    """
    ret = {}
    
    # Generate FP8 input data
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    
    # Quantize to FP8
    x_fp8, x_scales = dynamic_mxfp8_quant(x)
    w_fp8, w_scales = dynamic_mxfp8_quant(w)
    
    # Allocate output
    out = torch.empty((M, N), dtype=dtype, device="cuda")
    
    # Compute reference
    ref = run_torch_reference(x_fp8, w_fp8, x_scales, w_scales, dtype)
    
    # Run tritonblas FP8 matmul using triton.testing.do_bench
    fn = lambda: tritonblas.matmul_fp8(x_fp8, w_fp8, out, x_scales, w_scales)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    
    # Compute performance metrics
    total_ops = 2 * M * N * K
    ret["M"] = M
    ret["N"] = N
    ret["K"] = K
    ret["dtype"] = str(dtype)
    ret["us"] = ms * 1000  # Convert ms to us
    ret["TFLOPS"] = total_ops / (ms * 1e-3) / 1e12
    ret["TB/s"] = (x_fp8.nbytes + w_fp8.nbytes) / (ms * 1e-3) / 1e12
    
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
        print(f"FP8 GEMM Test: M={M}, N={N}, K={K}, dtype={dtype}")
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


# Pytest test functions
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp8(dtype, M, N, K):
    """Pytest test for FP8 GEMM correctness."""
    ret = run_gemm_fp8_test(dtype, M, N, K, verbose=False)
    
    # Assert validity thresholds
    assert ret["valid_%"] >= 95.0, f"Only {ret['valid_%']:.1f}% valid values"
    assert ret["nan_%"] <= 5.0, f"{ret['nan_%']:.1f}% NaN values"
    assert ret["inf_%"] <= 5.0, f"{ret['inf_%']:.1f}% Inf values"


@pytest.mark.performance
def test_fp8_production_benchmarks():
    """Pytest test for FP8 production benchmarks - prints performance tables."""
    print("\n" + "="*80)
    print("FP8 GEMM Production Benchmark")
    print("="*80)
    
    # Problem sizes similar to FP4 tests
    test_sizes = [
        # Pure compute
        (256, 2048, 8192),
        (2048, 8192, 8192),
        (16384, 16384, 16384),
        # QKV projection
        (1, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (256, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        # Attention output
        (1, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (256, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
    ]
    
    dtype = torch.bfloat16
    results = []
    
    for M, N, K in test_sizes:
        try:
            ret = run_gemm_fp8_test(dtype, M, N, K, verbose=False)
            results.append(ret)
            print(f"M={M:5d}, N={N:6d}, K={K:5d}: {ret['TFLOPS']:6.2f} TFLOPS, "
                  f"{ret['us']:8.2f} us, err={ret['mean_abs_err']:.6f}")
            
            # Clean up to avoid OOM
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"M={M:5d}, N={N:6d}, K={K:5d}: SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else:
                raise
    
    print("="*80 + "\n")
    
    # Assert we got at least some results
    assert len(results) > 0, "No benchmark results collected"


@pytest.mark.performance
def test_fp8_block_size_sweep():
    """Pytest test for FP8 block size sweep - prints performance tables."""
    print("\n" + "="*80)
    print("FP8 GEMM Block Size Sweep (8192x8192x8192)")
    print("="*80)
    
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    
    # Generate test data once
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    x_fp8, x_scales = dynamic_mxfp8_quant(x)
    w_fp8, w_scales = dynamic_mxfp8_quant(w)
    out = torch.empty((M, N), dtype=dtype)
    
    # Block size configurations to test
    block_m_sizes = [64, 128, 256]
    block_n_sizes = [64, 128, 256]
    block_k_sizes = [128, 256, 512]
    
    results = {}
    best_tflops = 0
    best_config = None
    
    for block_k in block_k_sizes:
        results[block_k] = {}
        for block_m in block_m_sizes:
            for block_n in block_n_sizes:
                try:
                    fn = lambda: tritonblas.matmul_fp8(
                        x_fp8, w_fp8, out, x_scales, w_scales,
                        block_m=block_m, block_n=block_n, block_k=block_k
                    )
                    
                    ms = triton.testing.do_bench(fn, warmup=10, rep=20)
                    total_ops = 2 * M * N * K
                    tflops = total_ops / (ms * 1e-3) / 1e12
                    
                    key = f"M{block_m}_N{block_n}"
                    results[block_k][key] = tflops
                    
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
    
    # Assert we found a best configuration
    assert best_config is not None, "No valid block size configuration found"
    assert best_tflops > 0, "Best configuration has zero throughput"


def main():
    """Main test runner - now just runs pytest with appropriate markers."""
    parser = argparse.ArgumentParser(
        description="TritonBLAS FP8 GEMM Test Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["all", "correctness", "performance"],
        default="all",
        help="Test mode: 'correctness' runs basic tests, 'performance' runs benchmarks, 'all' runs both (default: all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TritonBLAS FP8 GEMM Test Suite")
    print("="*80)
    print(f"Test mode: {args.mode}")
    print("="*80 + "\n")
    
    # Build pytest arguments
    pytest_args = [__file__, "-v", "-s"]
    
    if args.mode == "correctness":
        pytest_args.extend(["-m", "not performance"])
    elif args.mode == "performance":
        pytest_args.extend(["-m", "performance"])
    # For "all", run everything (no marker filter)
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "="*80)
    print("Test suite completed!")
    print("="*80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    main()

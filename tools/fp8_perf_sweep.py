#!/usr/bin/env python3
"""
FP8 GEMM Performance Sweep Tool

Comprehensive parameter sweep to find peak performance for FP8 matrix multiplication.
Sweeps over block sizes, num_warps, num_stages, and group_size_m to maximize throughput.

Usage:
    python tools/fp8_perf_sweep.py --size 8192 8192 8192
    python tools/fp8_perf_sweep.py --size 8192 8192 8192 --output results.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Add include directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "include"))

import torch
import triton
import tritonblas
from tritonblas.utils import dynamic_mxfp8_quant

torch.set_default_device("cuda")


def benchmark_config(x_fp8, w_fp8, x_scales, w_scales, out, M, N, K,
                     block_m, block_n, block_k, num_warps, num_stages, group_size_m,
                     warmup=25, rep=100):
    """Benchmark a specific configuration using triton.testing.do_bench."""
    try:
        # Create lambda for do_bench
        fn = lambda: tritonblas.matmul_fp8(
            x_fp8, w_fp8, out, x_scales, w_scales,
            block_m=block_m, block_n=block_n, block_k=block_k,
            num_warps=num_warps, num_stages=num_stages, group_size_m=group_size_m
        )
        
        # Use triton's do_bench for accurate benchmarking
        time_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        
        # Calculate metrics
        total_ops = 2 * M * N * K
        tflops = total_ops / (time_ms * 1e-3) / 1e12
        bandwidth_gbs = (x_fp8.nbytes + w_fp8.nbytes + out.nbytes) / (time_ms * 1e-3) / 1e9
        
        return {
            'time_ms': time_ms,
            'tflops': tflops,
            'bandwidth_gbs': bandwidth_gbs,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'time_ms': 0,
            'tflops': 0,
            'bandwidth_gbs': 0,
            'success': False,
            'error': str(e)
        }


def run_sweep(M, N, K, output_csv=None):
    """Run comprehensive parameter sweep."""
    print(f"\n{'='*80}")
    print(f"FP8 GEMM Performance Sweep: M={M}, N={N}, K={K}")
    print(f"{'='*80}\n")
    
    # Generate test data once
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    x_fp8, x_scales = dynamic_mxfp8_quant(x)
    w_fp8, w_scales = dynamic_mxfp8_quant(w)
    out = torch.empty((M, N), dtype=dtype)
    
    print(f"Input shapes:")
    print(f"  x_fp8: {x_fp8.shape}, scales: {x_scales.shape}")
    print(f"  w_fp8: {w_fp8.shape}, scales: {w_scales.shape}")
    print(f"  out: {out.shape}\n")
    
    # Parameter ranges to sweep - more exhaustive
    block_m_range = [256]
    block_n_range = [256]
    block_k_range = [128, 256]  # Must be multiple of 32
    num_warps_range = [8, 16]
    num_stages_range = [2]
    group_size_m_range = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16]
    
    results = []
    best_config = None
    best_tflops = 0
    
    total_configs = (len(block_m_range) * len(block_n_range) * len(block_k_range) * 
                    len(num_warps_range) * len(num_stages_range) * len(group_size_m_range))
    
    print(f"Testing {total_configs} configurations...\n")
    
    config_num = 0
    for block_m in block_m_range:
        for block_n in block_n_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for num_stages in num_stages_range:
                        for group_size_m in group_size_m_range:
                            config_num += 1
                            
                            result = benchmark_config(
                                x_fp8, w_fp8, x_scales, w_scales, out, M, N, K,
                                block_m, block_n, block_k, num_warps, num_stages, group_size_m
                            )
                            
                            config = {
                                'M': M, 'N': N, 'K': K,
                                'block_m': block_m,
                                'block_n': block_n,
                                'block_k': block_k,
                                'num_warps': num_warps,
                                'num_stages': num_stages,
                                'group_size_m': group_size_m,
                                **result
                            }
                            results.append(config)
                            
                            if result['success'] and result['tflops'] > best_tflops:
                                best_tflops = result['tflops']
                                best_config = config
                            
                            # Print progress every 50 configs
                            if config_num % 50 == 0 or result['tflops'] > best_tflops * 0.95:
                                status = "✓" if result['success'] else "✗"
                                print(f"[{config_num:4d}/{total_configs}] {status} "
                                      f"BLK={block_m:3d}x{block_n:3d}x{block_k:3d} "
                                      f"W={num_warps:2d} S={num_stages} G={group_size_m:2d}: "
                                      f"{result['tflops']:6.2f} TFLOPS")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}\n")
    
    if best_config:
        print("Best Configuration:")
        print(f"  Block sizes: {best_config['block_m']}x{best_config['block_n']}x{best_config['block_k']}")
        print(f"  Num warps: {best_config['num_warps']}")
        print(f"  Num stages: {best_config['num_stages']}")
        print(f"  Group size M: {best_config['group_size_m']}")
        print(f"  Performance: {best_config['tflops']:.2f} TFLOPS")
        print(f"  Bandwidth: {best_config['bandwidth_gbs']:.2f} GB/s")
        print(f"  Time: {best_config['time_ms']:.3f} ms")
    
    # Print top 10 configurations
    print(f"\nTop 10 Configurations:")
    print(f"{'-'*80}")
    successful_results = [r for r in results if r['success']]
    top_10 = sorted(successful_results, key=lambda x: x['tflops'], reverse=True)[:10]
    
    for i, config in enumerate(top_10, 1):
        print(f"{i:2d}. {config['tflops']:6.2f} TFLOPS | "
              f"BLK={config['block_m']:3d}x{config['block_n']:3d}x{config['block_k']:3d} | "
              f"W={config['num_warps']:2d} S={config['num_stages']} G={config['group_size_m']:2d}")
    
    # Save to CSV if requested
    if output_csv:
        fieldnames = ['M', 'N', 'K', 'block_m', 'block_n', 'block_k', 
                     'num_warps', 'num_stages', 'group_size_m',
                     'tflops', 'bandwidth_gbs', 'time_ms', 'success', 'error']
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_csv}")
    
    return best_config, results


def run_focused_sweep(M, N, K, output_csv=None):
    """Run a more focused sweep around likely good configurations."""
    print(f"\n{'='*80}")
    print(f"FP8 GEMM Focused Performance Sweep: M={M}, N={N}, K={K}")
    print(f"{'='*80}\n")
    
    # Generate test data once
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    x_fp8, x_scales = dynamic_mxfp8_quant(x)
    w_fp8, w_scales = dynamic_mxfp8_quant(w)
    out = torch.empty((M, N), dtype=dtype)
    
    # Focused parameter ranges (based on typical good configs for large GEMMs)
    configs_to_test = [
        # (block_m, block_n, block_k, num_warps, num_stages, group_size_m)
        (128, 128, 128, 8, 2, 8),
        (128, 128, 256, 8, 2, 8),
        (128, 256, 128, 8, 2, 8),
        (256, 128, 128, 8, 2, 8),
        (256, 256, 128, 8, 2, 8),
        (256, 256, 256, 8, 2, 8),
        (128, 128, 128, 8, 3, 8),
        (256, 256, 128, 8, 3, 8),
        (128, 128, 256, 8, 3, 8),
        (256, 256, 256, 8, 3, 8),
        (128, 128, 128, 16, 2, 8),
        (256, 256, 128, 16, 2, 8),
        (128, 128, 128, 8, 4, 8),
        (256, 256, 256, 8, 4, 8),
        (128, 128, 128, 8, 2, 4),
        (128, 128, 128, 8, 2, 16),
    ]
    
    results = []
    best_config = None
    best_tflops = 0
    
    print(f"Testing {len(configs_to_test)} focused configurations...\n")
    
    for i, (block_m, block_n, block_k, num_warps, num_stages, group_size_m) in enumerate(configs_to_test, 1):
        result = benchmark_config(
            x_fp8, w_fp8, x_scales, w_scales, out, M, N, K,
            block_m, block_n, block_k, num_warps, num_stages, group_size_m,
            warmup=25, rep=100
        )
        
        config = {
            'M': M, 'N': N, 'K': K,
            'block_m': block_m,
            'block_n': block_n,
            'block_k': block_k,
            'num_warps': num_warps,
            'num_stages': num_stages,
            'group_size_m': group_size_m,
            **result
        }
        results.append(config)
        
        if result['success'] and result['tflops'] > best_tflops:
            best_tflops = result['tflops']
            best_config = config
        
        status = "✓" if result['success'] else "✗"
        print(f"[{i:2d}/{len(configs_to_test)}] {status} "
              f"BLK={block_m:3d}x{block_n:3d}x{block_k:3d} "
              f"W={num_warps:2d} S={num_stages} G={group_size_m:2d}: "
              f"{result['tflops']:6.2f} TFLOPS")
    
    # Print summary
    print(f"\n{'='*80}")
    print("FOCUSED SWEEP COMPLETE")
    print(f"{'='*80}\n")
    
    if best_config:
        print("Best Configuration:")
        print(f"  Block sizes: {best_config['block_m']}x{best_config['block_n']}x{best_config['block_k']}")
        print(f"  Num warps: {best_config['num_warps']}")
        print(f"  Num stages: {best_config['num_stages']}")
        print(f"  Group size M: {best_config['group_size_m']}")
        print(f"  Performance: {best_config['tflops']:.2f} TFLOPS")
        print(f"  Bandwidth: {best_config['bandwidth_gbs']:.2f} GB/s")
        print(f"  Time: {best_config['time_ms']:.3f} ms")
    
    # Save to CSV if requested
    if output_csv:
        fieldnames = ['M', 'N', 'K', 'block_m', 'block_n', 'block_k', 
                     'num_warps', 'num_stages', 'group_size_m',
                     'tflops', 'bandwidth_gbs', 'time_ms', 'success', 'error']
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_csv}")
    
    return best_config, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FP8 GEMM Performance Sweep Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        default=[8192, 8192, 8192],
        metavar=('M', 'N', 'K'),
        help="Matrix dimensions M N K (default: 8192 8192 8192)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "focused"],
        default="focused",
        help="Sweep mode: 'full' tests all combinations, 'focused' tests likely good configs (default: focused)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results (optional)"
    )
    
    args = parser.parse_args()
    M, N, K = args.size
    
    if args.mode == "full":
        best_config, results = run_sweep(M, N, K, args.output)
    else:
        best_config, results = run_focused_sweep(M, N, K, args.output)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR PEAK PERFORMANCE:")
    print(f"{'='*80}")
    
    if best_config:
        print(f"\nUse these parameters in your code:")
        print(f"```python")
        print(f"tritonblas.matmul_fp8(")
        print(f"    x_fp8, w_fp8, out, x_scales, w_scales,")
        print(f"    block_m={best_config['block_m']},")
        print(f"    block_n={best_config['block_n']},")
        print(f"    block_k={best_config['block_k']},")
        print(f"    num_warps={best_config['num_warps']},")
        print(f"    num_stages={best_config['num_stages']},")
        print(f"    group_size_m={best_config['group_size_m']}")
        print(f")")
        print(f"```")
        print(f"\nExpected performance: {best_config['tflops']:.2f} TFLOPS")
    
    print(f"\n{'='*80}\n")

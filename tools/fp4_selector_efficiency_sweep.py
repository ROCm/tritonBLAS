#!/usr/bin/env python3
"""
FP4 GEMM Selector Efficiency Sweep

This script compares the efficiency of the Origami selector versus exhaustive search
for FP4 GEMM operations. It measures:
1. Selector overhead (time to select configuration)
2. Performance of selector-chosen configuration
3. Performance of all possible configurations (exhaustive search)
4. Efficiency ratio: selector performance / best exhaustive performance

Usage:
    python tools/fp4_selector_efficiency_sweep.py
    python tools/fp4_selector_efficiency_sweep.py --sizes small
    python tools/fp4_selector_efficiency_sweep.py --output results.csv
"""

import torch
import tritonblas
import triton
import time
import argparse
import csv
import sys
from pathlib import Path

# Add tests directory to path for fp4_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from fp4_utils import dynamic_mxfp4_quant

torch.set_default_device("cuda")


def benchmark_config(a_fp4, b_fp4, a_scales, b_scales, M, N, 
                     block_m, block_n, block_k, num_warps=8, num_stages=2):
    """Benchmark a specific block configuration using Triton's do_bench."""
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    
    # Define the function to benchmark
    def run_kernel():
        tritonblas.matmul_fp4(
            a_fp4, b_fp4, out, a_scales, b_scales,
            block_m=block_m, block_n=block_n, block_k=block_k,
            num_warps=num_warps, num_stages=num_stages
        )
    
    # Test if kernel works first
    try:
        run_kernel()
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)
    
    # Use Triton's do_bench for accurate benchmarking
    # Returns time in milliseconds
    try:
        time_ms = triton.testing.do_bench(run_kernel, warmup=25, rep=100)
        time_us = time_ms * 1000  # Convert to microseconds
        return time_us, None
    except Exception as e:
        return None, str(e)


def exhaustive_search(a_fp4, b_fp4, a_scales, b_scales, M, N, K):
    """
    Perform exhaustive search over all valid block configurations.
    
    Returns:
        best_config: (block_m, block_n, block_k, num_warps, num_stages, time_us, tflops)
        all_results: list of all tested configurations
    """
    # Define search space
    block_mn_range = [64, 128, 256]
    block_k_range = [128, 256, 512]  # Must be multiple of 64 for FP4
    num_warps_range = [4, 8]
    num_stages_range = [2, 4]
    
    all_results = []
    best_config = None
    best_tflops = 0
    
    total_configs = (len(block_mn_range) * len(block_mn_range) * len(block_k_range) * 
                     len(num_warps_range) * len(num_stages_range))
    tested = 0
    
    print(f"  Exhaustive search: testing {total_configs} configurations...")
    print(f"  (Block sizes: {len(block_mn_range)}×{len(block_mn_range)}×{len(block_k_range)}, "
          f"Warps: {num_warps_range}, Stages: {num_stages_range})")
    
    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for num_stages in num_stages_range:
                        tested += 1
                        
                        time_us, error = benchmark_config(
                            a_fp4, b_fp4, a_scales, b_scales, M, N,
                            block_m, block_n, block_k, num_warps, num_stages
                        )
                        
                        if time_us is not None:
                            total_ops = 2 * M * N * K
                            tflops = total_ops / time_us / 1e6
                            
                            result = {
                                'block_m': block_m,
                                'block_n': block_n,
                                'block_k': block_k,
                                'num_warps': num_warps,
                                'num_stages': num_stages,
                                'time_us': time_us,
                                'tflops': tflops,
                                'error': None
                            }
                            all_results.append(result)
                            
                            if tflops > best_tflops:
                                best_tflops = tflops
                                best_config = (block_m, block_n, block_k, num_warps, num_stages, time_us, tflops)
                            
                            status = "✓" if tflops > best_tflops * 0.95 else " "
                            print(f"    [{tested:3d}/{total_configs}] {status} "
                                  f"M{block_m:3d}xN{block_n:3d}xK{block_k:3d} "
                                  f"W{num_warps}S{num_stages}: "
                                  f"{tflops:7.2f} TFLOPS ({time_us:7.2f} us)")
                        else:
                            all_results.append({
                                'block_m': block_m,
                                'block_n': block_n,
                                'block_k': block_k,
                                'num_warps': num_warps,
                                'num_stages': num_stages,
                                'time_us': None,
                                'tflops': None,
                                'error': error
                            })
                            # Only print failures occasionally to reduce clutter
                            if tested % 10 == 1:
                                print(f"    [{tested:3d}/{total_configs}]   "
                                      f"M{block_m:3d}xN{block_n:3d}xK{block_k:3d} "
                                      f"W{num_warps}S{num_stages}: FAILED")
    
    return best_config, all_results


def test_selector_vs_exhaustive(M, N, K, dtype=torch.bfloat16):
    """
    Compare selector performance against exhaustive search for a given problem size.
    
    Returns:
        results: dict with comparison metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing M={M}, N={N}, K={K}")
    print(f"{'='*80}")
    
    # Generate test data
    print("Generating test data...")
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    
    # Quantize to FP4
    x_fp4, x_scales = dynamic_mxfp4_quant(x)
    w_fp4, w_scales = dynamic_mxfp4_quant(w)
    
    # Test selector
    print("\nTesting Origami selector...")
    out = torch.empty((M, N), dtype=dtype, device="cuda")
    
    # Measure selector overhead
    selector_start = time.time()
    # Call with None to trigger selector
    tritonblas.matmul_fp4(x_fp4, w_fp4, out, x_scales, w_scales,
                          block_m=None, block_n=None, block_k=None)
    selector_time = (time.time() - selector_start) * 1e6
    
    # Get selector's choice by calling again (cached)
    from tritonblas.matmul import _make_matmul_selector
    selector = _make_matmul_selector(M, N, K, "f4", "f4", dtype, mx_block_size=32)
    sel_m, sel_n, sel_k, sel_gsize = selector.get_config()
    
    print(f"  Selector chose: M{sel_m}xN{sel_n}xK{sel_k} (gsize_m={sel_gsize})")
    print(f"  Selector overhead: {selector_time:.2f} us")
    
    # Benchmark selector's choice
    sel_time_us, sel_error = benchmark_config(
        x_fp4, w_fp4, x_scales, w_scales, M, N,
        sel_m, sel_n, sel_k
    )
    
    if sel_time_us is not None:
        total_ops = 2 * M * N * K
        sel_tflops = total_ops / sel_time_us / 1e6
        print(f"  Selector performance: {sel_tflops:.2f} TFLOPS ({sel_time_us:.2f} us)")
    else:
        print(f"  Selector FAILED: {sel_error}")
        return None
    
    # Exhaustive search
    print("\nPerforming exhaustive search...")
    best_config, all_results = exhaustive_search(
        x_fp4, w_fp4, x_scales, w_scales, M, N, K
    )
    
    if best_config is None:
        print("  Exhaustive search found no valid configurations!")
        return None
    
    best_m, best_n, best_k, best_warps, best_stages, best_time_us, best_tflops = best_config
    print(f"\n  Best configuration: M{best_m}xN{best_n}xK{best_k} W{best_warps}S{best_stages}")
    print(f"  Best performance: {best_tflops:.2f} TFLOPS ({best_time_us:.2f} us)")
    
    # Compute efficiency metrics
    efficiency = (sel_tflops / best_tflops) * 100
    speedup = best_time_us / sel_time_us
    # Selector uses default warps=8, stages=2
    is_optimal = (sel_m == best_m and sel_n == best_n and sel_k == best_k and 
                  best_warps == 8 and best_stages == 2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Selector efficiency: {efficiency:.2f}% of best")
    print(f"  Speedup vs best: {speedup:.3f}x")
    print(f"  Selector is optimal: {'YES ✓' if is_optimal else 'NO'}")
    print(f"  Selector overhead: {selector_time:.2f} us")
    if not is_optimal:
        print(f"  Note: Selector uses W8S2, best is W{best_warps}S{best_stages}")
    print(f"{'='*80}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return {
        'M': M,
        'N': N,
        'K': K,
        'selector_config': f"{sel_m}x{sel_n}x{sel_k}_W8S2",
        'selector_tflops': sel_tflops,
        'selector_time_us': sel_time_us,
        'selector_overhead_us': selector_time,
        'best_config': f"{best_m}x{best_n}x{best_k}_W{best_warps}S{best_stages}",
        'best_tflops': best_tflops,
        'best_time_us': best_time_us,
        'efficiency_pct': efficiency,
        'speedup': speedup,
        'is_optimal': is_optimal,
        'all_results': all_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="FP4 GEMM Selector Efficiency Sweep",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--sizes',
        type=str,
        choices=['small', 'medium', 'large', 'production', 'all'],
        default='medium',
        help='Problem size set to test (default: medium)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for results (optional)'
    )
    parser.add_argument(
        '--custom',
        type=str,
        default=None,
        help='Custom size as M,N,K (e.g., --custom 1024,1024,1024)'
    )
    
    args = parser.parse_args()
    
    # Define problem size sets
    size_sets = {
        'small': [
            # Square matrices - small
            (128, 128, 128),
            (256, 256, 256),
            (384, 384, 384),
            (512, 512, 512),
            (640, 640, 640),
            (768, 768, 768),
            (896, 896, 896),
            (1024, 1024, 1024),
            # Non-square - small
            (256, 512, 1024),
            (512, 1024, 512),
        ],
        'medium': [
            # Square matrices - medium
            (1024, 1024, 1024),
            (1536, 1536, 1536),
            (2048, 2048, 2048),
            (3072, 3072, 3072),
            (4096, 4096, 4096),
            # Non-square - medium
            (1024, 2048, 4096),
            (2048, 4096, 2048),
            (1024, 4096, 2048),
            (2048, 2048, 4096),
            (4096, 2048, 2048),
        ],
        'large': [
            # Square matrices - large
            (4096, 4096, 4096),
            (6144, 6144, 6144),
            (8192, 8192, 8192),
            (12288, 12288, 12288),
            (16384, 16384, 16384),
            # Non-square - large
            (4096, 8192, 8192),
            (8192, 4096, 8192),
            (8192, 8192, 4096),
            (8192, 16384, 8192),
            (16384, 8192, 8192),
        ],
        'production': [
            # QKV projection - various batch sizes
            (1, 1280, 8192),
            (32, 1280, 8192),
            (64, 1280, 8192),
            (128, 1280, 8192),
            (256, 1280, 8192),
            (512, 1280, 8192),
            (1024, 1280, 8192),
            (2048, 1280, 8192),
            # Attention output - various batch sizes
            (64, 8192, 1024),
            (256, 8192, 1024),
            (512, 8192, 1024),
            (1024, 8192, 1024),
            (2048, 8192, 1024),
            # Pure compute workloads
            (2048, 8192, 8192),
            (4096, 8192, 8192),
            (8192, 8192, 8192),
            # MLP layers
            (1024, 4096, 1024),
            (2048, 4096, 2048),
            (4096, 1024, 4096),
            (2048, 11008, 4096),
        ],
        'all': [
            # Comprehensive coverage
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
            # Non-square
            (256, 512, 1024),
            (1024, 2048, 4096),
            (2048, 8192, 8192),
            (8192, 16384, 8192),
        ]
    }
    
    # Select problem sizes
    if args.custom:
        M, N, K = map(int, args.custom.split(','))
        test_sizes = [(M, N, K)]
    else:
        test_sizes = size_sets[args.sizes]
    
    print("\n" + "="*80)
    print("FP4 GEMM Selector Efficiency Sweep")
    print("="*80)
    print(f"Testing {len(test_sizes)} problem sizes")
    print(f"Size set: {args.sizes if not args.custom else 'custom'}")
    if args.output:
        print(f"Output file: {args.output}")
    print("="*80)
    
    # Run tests
    all_results = []
    for M, N, K in test_sizes:
        result = test_selector_vs_exhaustive(M, N, K)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not all_results:
        print("No successful tests!")
        return
    
    # Summary table
    print(f"\n{'M':>6} {'N':>6} {'K':>6} | {'Selector':>17} | {'Sel TFLOPS':>10} | {'Best':>17} | {'Best TFLOPS':>11} | {'Eff%':>6} | {'Opt':>3}")
    print("-" * 110)
    
    for r in all_results:
        optimal_mark = "✓" if r['is_optimal'] else ""
        print(f"{r['M']:6d} {r['N']:6d} {r['K']:6d} | "
              f"{r['selector_config']:>17} | "
              f"{r['selector_tflops']:10.2f} | "
              f"{r['best_config']:>17} | "
              f"{r['best_tflops']:11.2f} | "
              f"{r['efficiency_pct']:6.1f} | "
              f"{optimal_mark:>3}")
    
    # Statistics
    efficiencies = [r['efficiency_pct'] for r in all_results]
    optimal_count = sum(1 for r in all_results if r['is_optimal'])
    
    print("-" * 80)
    print(f"Average efficiency: {sum(efficiencies) / len(efficiencies):.2f}%")
    print(f"Min efficiency: {min(efficiencies):.2f}%")
    print(f"Max efficiency: {max(efficiencies):.2f}%")
    print(f"Optimal selections: {optimal_count}/{len(all_results)} "
          f"({100 * optimal_count / len(all_results):.1f}%)")
    
    avg_overhead = sum(r['selector_overhead_us'] for r in all_results) / len(all_results)
    print(f"Average selector overhead: {avg_overhead:.2f} us")
    
    # Save to CSV if requested
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'M', 'N', 'K',
                'selector_config', 'selector_tflops', 'selector_time_us', 'selector_overhead_us',
                'best_config', 'best_tflops', 'best_time_us',
                'efficiency_pct', 'speedup', 'is_optimal'
            ])
            writer.writeheader()
            for r in all_results:
                # Remove all_results field for CSV
                csv_row = {k: v for k, v in r.items() if k != 'all_results'}
                writer.writerow(csv_row)
        print(f"Results saved!")
        
        # Also save detailed results with all configurations
        detailed_output = args.output.replace('.csv', '_detailed.csv')
        print(f"Saving detailed results to {detailed_output}...")
        with open(detailed_output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'M', 'N', 'K', 'block_m', 'block_n', 'block_k', 
                'num_warps', 'num_stages', 'time_us', 'tflops', 'error'
            ])
            writer.writeheader()
            for r in all_results:
                for config in r['all_results']:
                    row = {
                        'M': r['M'],
                        'N': r['N'],
                        'K': r['K'],
                        **config
                    }
                    writer.writerow(row)
        print(f"Detailed results saved!")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

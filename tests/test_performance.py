import pytest
import torch
import tritonblas
import time
import sys
import json
from datetime import datetime
from tritonblas.utils import matmul_input_gen


def get_gpu_name():
    """Get the GPU device name."""
    if not torch.cuda.is_available():
        return None
    
    try:
        # Try to get device name
        device_name = torch.cuda.get_device_name(0)
        if device_name:
            return device_name
    except Exception:
        pass
    
    try:
        # Try ROCm-specific device properties
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, 'name') and props.name:
            return props.name
        if hasattr(props, 'gcnArchName') and props.gcnArchName:
            return props.gcnArchName
    except Exception:
        pass
    
    return "CUDA Device"


def benchmark_matmul(m, n, k, dtype, enable_streamk=False, warmup=5, iterations=100):
    """Benchmark matmul performance and return TFLOPs."""
    # Create input tensors using matmul_input_gen
    # Use "auto" quantization mode to handle FP8 dtypes properly
    A_result = matmul_input_gen((m, k), dtype, "randn", quantize="auto")
    B_result = matmul_input_gen((k, n), dtype, "randn", quantize="auto")
    
    # Handle quantized vs non-quantized results
    if isinstance(A_result, tuple):
        A, scaleA = A_result
    else:
        A = A_result
        scaleA = None
    
    if isinstance(B_result, tuple):
        B, scaleB = B_result
    else:
        B = B_result
        scaleB = None
    
    C = torch.zeros((m, n), device="cuda", dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        tritonblas.matmul(A, B, C, enable_streamk=enable_streamk)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        tritonblas.matmul(A, B, C, enable_streamk=enable_streamk)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Calculate performance
    elapsed_time = (end_time - start_time) / iterations
    flops = 2 * m * n * k  # FLOPs for matrix multiplication
    tflops = (flops / elapsed_time) / 1e12
    
    return tflops, elapsed_time


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires CUDA GPU"
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ],
)
@pytest.mark.parametrize(
    "enable_streamk",
    [
        False,
        True,
    ],
)
def test_mi350_performance(m, n, k, dtype, enable_streamk):
    """Test matmul performance for various sizes and configurations."""
    tflops, elapsed_time = benchmark_matmul(m, n, k, dtype, enable_streamk)
    
    print(f"\nPerformance Results:")
    print(f"  Size: {m}x{n}x{k}")
    print(f"  Dtype: {dtype}")
    print(f"  StreamK: {enable_streamk}")
    print(f"  Performance: {tflops:.2f} TFLOPs")
    print(f"  Time per iteration: {elapsed_time*1000:.3f} ms")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires CUDA GPU"
)
def test_mi350_performance_report(capsys):
    """Generate a performance report for various sizes."""
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]
    dtypes = [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
    
    # Collect results
    results = []
    
    for m, n, k in sizes:
        for dtype in dtypes:
            for enable_streamk in [False, True]:
                # Determine dtype string first
                if dtype == torch.float16:
                    dtype_str = "fp16"
                elif dtype == torch.bfloat16:
                    dtype_str = "bf16"
                elif dtype == torch.float8_e4m3fn:
                    dtype_str = "fp8"
                else:
                    dtype_str = str(dtype)
                
                mode = "StreamK" if enable_streamk else "Persistent"
                
                try:
                    tflops, elapsed_time = benchmark_matmul(
                        m, n, k, dtype, enable_streamk, warmup=3, iterations=50
                    )
                    results.append({
                        'size': f"{m}x{n}x{k}",
                        'dtype': dtype_str,
                        'mode': mode,
                        'tflops': tflops,
                        'time_ms': elapsed_time * 1000
                    })
                except Exception as e:
                    results.append({
                        'size': f"{m}x{n}x{k}",
                        'dtype': dtype_str,
                        'mode': mode,
                        'tflops': 0.0,
                        'time_ms': 0.0,
                        'error': str(e)
                    })
    
    # Build the table as a string
    gpu_name = get_gpu_name() or "Unknown GPU"
    table_lines = []
    table_lines.append("\n" + "="*100)
    table_lines.append(f"Performance Report - {gpu_name}")
    table_lines.append("="*100)
    table_lines.append(f"{'Size':<20} {'Dtype':<8} {'Mode':<12} {'TFLOPs':>12} {'Time (ms)':>12} {'Status':<20}")
    table_lines.append("-"*100)
    
    for result in results:
        if 'error' in result:
            table_lines.append(f"{result['size']:<20} {result['dtype']:<8} {result['mode']:<12} "
                  f"{'N/A':>12} {'N/A':>12} {'FAILED':<20}")
        else:
            status = "✓ PASS" if result['tflops'] >= 1000.0 and result['size'] == "8192x8192x8192" else "✓"
            table_lines.append(f"{result['size']:<20} {result['dtype']:<8} {result['mode']:<12} "
                  f"{result['tflops']:>12.2f} {result['time_ms']:>12.3f} {status:<20}")
    
    table_lines.append("="*100)
    
    # Add summary statistics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        max_tflops = max(r['tflops'] for r in valid_results)
        avg_tflops = sum(r['tflops'] for r in valid_results) / len(valid_results)
        table_lines.append(f"\nSummary:")
        table_lines.append(f"  Max Performance: {max_tflops:.2f} TFLOPs")
        table_lines.append(f"  Avg Performance: {avg_tflops:.2f} TFLOPs")
        table_lines.append(f"  Total Tests: {len(results)}")
        table_lines.append(f"  Passed: {len(valid_results)}")
        table_lines.append(f"  Failed: {len(results) - len(valid_results)}")
    table_lines.append("="*100 + "\n")
    
    # Print to stderr and also disable capture temporarily to ensure visibility
    table_output = "\n".join(table_lines)
    print(table_output, file=sys.stderr)
    
    # Also use capsys to ensure output is shown
    with capsys.disabled():
        print(table_output)
    
    # Export results to JSON file for CI artifacts
    json_output = {
        'timestamp': datetime.now().isoformat(),
        'gpu_name': gpu_name,
        'results': results,
        'summary': {
            'max_tflops': max_tflops if valid_results else 0.0,
            'avg_tflops': avg_tflops if valid_results else 0.0,
            'total_tests': len(results),
            'passed': len(valid_results),
            'failed': len(results) - len(valid_results)
        } if valid_results else None
    }
    
    try:
        with open('performance-results.json', 'w') as f:
            json.dump(json_output, f, indent=2)
        print("\nPerformance results saved to performance-results.json")
    except Exception as e:
        print(f"\nWarning: Could not save JSON results: {e}", file=sys.stderr)
    
    # Export table to text file for CI artifacts
    try:
        with open('performance-report.txt', 'w') as f:
            f.write(table_output)
        print("Performance report saved to performance-report.txt")
    except Exception as e:
        print(f"\nWarning: Could not save text report: {e}", file=sys.stderr)

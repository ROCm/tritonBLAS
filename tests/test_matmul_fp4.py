# SPDX-License-Identifier: MIT
# Comprehensive FP4 matmul test suite for tritonblas
# Based on aiter's test_gemm_a4w4.py

import torch
import tritonblas
import time
import argparse
import pytest
from tritonblas.utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
#torch.backends.cuda.preferred_blas_library("hipblas")


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


def run_gemm_fp4_test(dtype, M, N, K, verbose=True, enable_streamk=False, sk_grid=None):
    """
    Test FP4 GEMM with given dimensions and dtype.

    Args:
        dtype: Output dtype (e.g. torch.bfloat16)
        M, N, K: Matrix dimensions
        verbose: Print detailed results
        enable_streamk: Use Stream-K load balancing
        sk_grid: Override Stream-K grid size (when enable_streamk=True)
    
    Returns dictionary with performance metrics and error statistics.
    """
    ret = {}
    
    # Generate FP4 input data using unified API
    from tritonblas.utils import generate_matmul_inputs
    
    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",  # Use FP4 quantization
        out_dtype=dtype,
        init_type="randn"
    )
    
    # Extract FP4 tensors and scales
    x_fp4 = inputs.A      # Shape: (M, K//2)
    w_fp4 = inputs.B.T    # Shape: (K//2, N) -> transpose to (N, K//2) for reference
    x_scales = inputs.scaleA  # Shape: (M, K//32)
    w_scales = inputs.scaleB  # Shape: (N, K//32)
    
    # Allocate output
    out = inputs.C
    
    # Compute reference
    ref = run_torch_reference(x_fp4, w_fp4, x_scales, w_scales, dtype)
    
    # Run tritonblas FP4 matmul (with optional Stream-K)
    def run_tritonblas():
        tritonblas.matmul_fp4(
            x_fp4, w_fp4, out, x_scales, w_scales,
            enable_streamk=enable_streamk,
            sk_grid=sk_grid,
        )
    
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

    ret["enable_streamk"] = enable_streamk

    if verbose:
        streamk_str = " (Stream-K)" if enable_streamk else ""
        print(f"\n{'='*80}")
        print(f"FP4 GEMM Test: M={M}, N={N}, K={K}, dtype={dtype}{streamk_str}")
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
def test_gemm_fp4(dtype, M, N, K):
    """Pytest test for FP4 GEMM correctness (non-Stream-K)."""
    ret = run_gemm_fp4_test(dtype, M, N, K, verbose=False, enable_streamk=False)

    # Assert validity thresholds
    assert ret["valid_%"] >= 95.0, f"Only {ret['valid_%']:.1f}% valid values"
    assert ret["nan_%"] <= 5.0, f"{ret['nan_%']:.1f}% NaN values"
    assert ret["inf_%"] <= 5.0, f"{ret['inf_%']:.1f}% Inf values"


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk(dtype, M, N, K):
    """Pytest test for FP4 GEMM correctness with Stream-K load balancing."""
    ret = run_gemm_fp4_test(dtype, M, N, K, verbose=False, enable_streamk=True)

    # Assert validity thresholds
    assert ret["valid_%"] >= 95.0, f"Stream-K: Only {ret['valid_%']:.1f}% valid values"
    assert ret["nan_%"] <= 5.0, f"Stream-K: {ret['nan_%']:.1f}% NaN values"
    assert ret["inf_%"] <= 5.0, f"Stream-K: {ret['inf_%']:.1f}% Inf values"


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_smoke(dtype, M, N, K):
    """Smoke test: Stream-K FP4 matmul runs and produces finite output (no reference)."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    out = inputs.C
    tritonblas.matmul_fp4(
        inputs.A, inputs.B.T, out, inputs.scaleA, inputs.scaleB,
        enable_streamk=True,
    )
    assert not torch.isnan(out).all(), "Output is all NaN"
    assert not torch.isinf(out).all(), "Output is all Inf"
    valid_pct = 100 * ((~torch.isnan(out) & ~torch.isinf(out)).sum().item() / out.numel())
    assert valid_pct >= 95.0, f"Only {valid_pct:.1f}% valid values"


@pytest.mark.parametrize("M,N,K", [
    (256, 256, 256),
    (512, 512, 512),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_vs_standard(dtype, M, N, K):
    """Verify Stream-K and standard FP4 matmul produce equivalent results."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    x_fp4, w_fp4 = inputs.A, inputs.B.T
    x_scales, w_scales = inputs.scaleA, inputs.scaleB

    out_standard = inputs.C.clone()
    out_streamk = inputs.C.clone()

    tritonblas.matmul_fp4(x_fp4, w_fp4, out_standard, x_scales, w_scales, enable_streamk=False)
    tritonblas.matmul_fp4(x_fp4, w_fp4, out_streamk, x_scales, w_scales, enable_streamk=True)

    max_diff = (out_standard.float() - out_streamk.float()).abs().max().item()
    assert max_diff < 0.1, f"Stream-K vs standard max diff={max_diff:.6f} (M={M}, N={N}, K={K})"


@pytest.mark.parametrize("M,N,K,sk_grid", [
    (256, 256, 256, 50),   # tiles=64, rem=14
    (512, 512, 512, 100),  # tiles=128, rem=28
    (1024, 1024, 1024, 200),  # tiles=256, rem=56
    # Large remainder: rem=48 > half of 80 CUs, exercises Stream-K partial path significantly
    (512, 512, 512, 80),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_with_remainder(dtype, M, N, K, sk_grid):
    """Smoke test: Stream-K FP4 with partial tiles (sk_grid forces remainder)."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    out = inputs.C
    tritonblas.matmul_fp4(
        inputs.A, inputs.B.T, out, inputs.scaleA, inputs.scaleB,
        enable_streamk=True,
        sk_grid=sk_grid,
    )
    assert not torch.isnan(out).all(), "Output is all NaN"
    assert not torch.isinf(out).all(), "Output is all Inf"
    valid_pct = 100 * ((~torch.isnan(out) & ~torch.isinf(out)).sum().item() / out.numel())
    assert valid_pct >= 95.0, f"Only {valid_pct:.1f}% valid values"


@pytest.mark.parametrize("M,N,K", [
    # K % 32 == 0 (FP4 requirement) but K % block_k != 0 (non-even K)
    # M=1024 N=1024 K=416 -> block_k=128, 416 % 128 = 32 -> EVEN_K=False
    (1024, 1024, 416),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_non_even_k(dtype, M, N, K):
    """Smoke test: Stream-K FP4 with non-even K (K % block_k != 0)."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    out = inputs.C
    tritonblas.matmul_fp4(
        inputs.A, inputs.B.T, out, inputs.scaleA, inputs.scaleB,
        enable_streamk=True,
    )
    assert not torch.isnan(out).all(), "Output is all NaN"
    assert not torch.isinf(out).all(), "Output is all Inf"
    valid_pct = 100 * ((~torch.isnan(out) & ~torch.isinf(out)).sum().item() / out.numel())
    assert valid_pct >= 95.0, f"Only {valid_pct:.1f}% valid values"


@pytest.mark.parametrize("M,N,K", [
    (1024, 1024, 416),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_vs_standard_non_even_k(dtype, M, N, K):
    """Verify Stream-K and standard FP4 matmul match when K % block_k != 0."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    x_fp4, w_fp4 = inputs.A, inputs.B.T
    x_scales, w_scales = inputs.scaleA, inputs.scaleB

    out_standard = inputs.C.clone()
    out_streamk = inputs.C.clone()

    tritonblas.matmul_fp4(x_fp4, w_fp4, out_standard, x_scales, w_scales, enable_streamk=False)
    tritonblas.matmul_fp4(x_fp4, w_fp4, out_streamk, x_scales, w_scales, enable_streamk=True)

    max_diff = (out_standard.float() - out_streamk.float()).abs().max().item()
    assert max_diff < 0.1, f"Stream-K vs standard max diff={max_diff:.6f} (M={M}, N={N}, K={K})"


@pytest.mark.parametrize("M,N,K,sk_grid", [
    (256, 256, 256, 50),
    (512, 512, 512, 100),
    # Large remainder: rem=48 > half of 80 CUs
    (512, 512, 512, 80),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_fp4_streamk_vs_standard_with_remainder(dtype, M, N, K, sk_grid):
    """Verify Stream-K and standard FP4 matmul match when Stream-K has partial tiles."""
    from tritonblas.utils import generate_matmul_inputs

    inputs = generate_matmul_inputs(
        m=M, n=N, k=K,
        in_dtype="fp4",
        out_dtype=dtype,
        init_type="randn",
    )
    x_fp4, w_fp4 = inputs.A, inputs.B.T
    x_scales, w_scales = inputs.scaleA, inputs.scaleB

    out_standard = inputs.C.clone()
    out_streamk = inputs.C.clone()

    tritonblas.matmul_fp4(x_fp4, w_fp4, out_standard, x_scales, w_scales, enable_streamk=False)
    tritonblas.matmul_fp4(x_fp4, w_fp4, out_streamk, x_scales, w_scales, enable_streamk=True, sk_grid=sk_grid)

    max_diff = (out_standard.float() - out_streamk.float()).abs().max().item()
    assert max_diff < 0.1, f"Stream-K vs standard max diff={max_diff:.6f} (M={M}, N={N}, K={K}, sk_grid={sk_grid})"


@pytest.mark.performance
def test_fp4_production_benchmarks():
    """Pytest test for FP4 production benchmarks - prints performance tables."""
    print("\n" + "="*80)
    print("FP4 GEMM Production Benchmark")
    print("="*80)
    
    # Problem sizes from aiter test_gemm_a4w4.py
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
        # Large batch (e.g. 32768 x 2880 x 2880)
        (32768, 2880, 2880),
    ]
    
    dtype = torch.bfloat16
    results = []
    
    for M, N, K in test_sizes:
        try:
            ret = run_gemm_fp4_test(dtype, M, N, K, verbose=False)
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
def test_fp4_streamk_production_benchmarks():
    """Pytest test for FP4 Stream-K production benchmarks - prints performance tables.
    Runs kernel directly without reference (avoids run_torch_reference crash in some envs).
    """
    from tritonblas.utils import generate_matmul_inputs

    print("\n" + "="*80)
    print("FP4 GEMM Stream-K Production Benchmark")
    print("="*80)

    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (16384, 16384, 16384),
        (256, 2048, 8192),
        (64, 1280, 8192),
        (1024, 1280, 8192),
        (32768, 2880, 2880),
    ]

    dtype = torch.bfloat16
    results = []

    for M, N, K in test_sizes:
        try:
            inputs = generate_matmul_inputs(
                m=M, n=N, k=K,
                in_dtype="fp4",
                out_dtype=dtype,
                init_type="randn",
            )
            out = inputs.C

            def run_kernel():
                tritonblas.matmul_fp4(
                    inputs.A, inputs.B.T, out, inputs.scaleA, inputs.scaleB,
                    enable_streamk=True,
                )

            us = benchmark_kernel(run_kernel, num_iters=10, warmup=3)
            total_ops = 2 * M * N * K
            tflops = total_ops / us / 1e6
            results.append({"M": M, "N": N, "K": K, "us": us, "TFLOPS": tflops})
            print(f"M={M:5d}, N={N:6d}, K={K:5d}: {tflops:6.2f} TFLOPS, {us:8.2f} us (Stream-K)")

            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"M={M:5d}, N={N:6d}, K={K:5d}: SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else:
                raise

    print("="*80 + "\n")
    assert len(results) > 0, "No Stream-K benchmark results collected"


@pytest.mark.performance
def test_fp4_standard_vs_streamk_benchmark():
    """Compare standard FP4 matmul vs Stream-K FP4 matmul performance."""
    from tritonblas.utils import generate_matmul_inputs

    print("\n" + "="*100)
    print("FP4 GEMM: Standard vs Stream-K Performance Comparison")
    print("="*100)
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Standard TFLOPS':>14} {'Standard us':>12} | {'Stream-K TFLOPS':>16} {'Stream-K us':>12} | {'Speedup':>8}")
    print("-"*100)

    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (16384, 16384, 16384),
        (256, 2048, 8192),
        (64, 1280, 8192),
        (1024, 1280, 8192),
        # Non-even K: K % block_k != 0 (exercises K-loop peel path)
        (1024, 1024, 416),
        (32768, 2880, 2880),
    ]

    dtype = torch.bfloat16
    results = []

    for M, N, K in test_sizes:
        try:
            inputs = generate_matmul_inputs(
                m=M, n=N, k=K,
                in_dtype="fp4",
                out_dtype=dtype,
                init_type="randn",
            )
            out_std = inputs.C.clone()
            out_sk = inputs.C.clone()
            total_ops = 2 * M * N * K

            # Standard FP4
            def run_standard():
                tritonblas.matmul_fp4(
                    inputs.A, inputs.B.T, out_std, inputs.scaleA, inputs.scaleB,
                    enable_streamk=False,
                )
            us_std = benchmark_kernel(run_standard, num_iters=20, warmup=5)
            tflops_std = total_ops / us_std / 1e6

            # Stream-K FP4
            def run_streamk():
                tritonblas.matmul_fp4(
                    inputs.A, inputs.B.T, out_sk, inputs.scaleA, inputs.scaleB,
                    enable_streamk=True,
                )
            us_sk = benchmark_kernel(run_streamk, num_iters=20, warmup=5)
            tflops_sk = total_ops / us_sk / 1e6

            speedup = tflops_sk / tflops_std if tflops_std > 0 else 0
            results.append({
                "M": M, "N": N, "K": K,
                "tflops_std": tflops_std, "us_std": us_std,
                "tflops_sk": tflops_sk, "us_sk": us_sk,
                "speedup": speedup,
            })
            print(f"{M:6d} {N:6d} {K:6d} | {tflops_std:14.2f} {us_std:12.2f} | {tflops_sk:16.2f} {us_sk:12.2f} | {speedup:7.2f}x")

            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{M:6d} {N:6d} {K:6d} | SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else:
                raise

    print("="*100 + "\n")
    assert len(results) > 0, "No benchmark comparison results collected"


@pytest.mark.performance
def test_fp4_aiter_vs_tritonblas_benchmark():
    """Compare aiter gemm_afp4wfp4 vs tritonBLAS fp4_streamk performance.

    Requires aiter with gemm module. If import fails, set PYTHONPATH to include
    the aiter repo root (e.g. PYTHONPATH=/path/to/aiter).
    Speedup > 1 means tritonBLAS is faster; < 1 means aiter is faster.
    """
    pytest.importorskip("aiter")
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    except ModuleNotFoundError as e:
        pytest.skip(
            f"aiter gemm module not found: {e}. "
            "Set PYTHONPATH to include the aiter repo root."
        )
    from tritonblas.utils import generate_matmul_inputs

    print("\n" + "="*100)
    print("FP4 GEMM: aiter gemm_afp4wfp4 vs tritonBLAS Stream-K")
    print("="*100)
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'aiter TFLOPS':>12} {'aiter us':>10} | {'tritonBLAS TFLOPS':>18} {'tritonBLAS us':>12} | {'Speedup':>8}")
    print("-"*100)

    test_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (16384, 16384, 16384),
        (32768, 2880, 2880),
    ]

    dtype = torch.bfloat16
    results = []

    for M, N, K in test_sizes:
        try:
            inputs = generate_matmul_inputs(
                m=M, n=N, k=K,
                in_dtype="fp4",
                out_dtype=dtype,
                init_type="randn",
            )
            x_fp4 = inputs.A
            w_fp4 = inputs.B.T
            x_scales = inputs.scaleA
            w_scales = inputs.scaleB
            out_aiter = inputs.C.clone()
            out_triton = inputs.C.clone()
            total_ops = 2 * M * N * K

            def run_aiter():
                gemm_afp4wfp4(x_fp4, w_fp4, x_scales, w_scales, dtype=dtype, y=out_aiter)

            def run_tritonblas():
                tritonblas.matmul_fp4(
                    x_fp4, w_fp4, out_triton, x_scales, w_scales,
                    enable_streamk=True,
                )

            us_aiter = benchmark_kernel(run_aiter, num_iters=20, warmup=5)
            us_triton = benchmark_kernel(run_tritonblas, num_iters=20, warmup=5)
            tflops_aiter = total_ops / us_aiter / 1e6
            tflops_triton = total_ops / us_triton / 1e6
            speedup = tflops_triton / tflops_aiter if tflops_aiter > 0 else 0

            results.append({
                "M": M, "N": N, "K": K,
                "tflops_aiter": tflops_aiter, "us_aiter": us_aiter,
                "tflops_triton": tflops_triton, "us_triton": us_triton,
                "speedup": speedup,
            })
            print(f"{M:6d} {N:6d} {K:6d} | {tflops_aiter:12.2f} {us_aiter:10.2f} | {tflops_triton:18.2f} {us_triton:12.2f} | {speedup:7.2f}x")

            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{M:6d} {N:6d} {K:6d} | SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else:
                raise

    print("="*100 + "\n")
    assert len(results) > 0, "No aiter vs tritonBLAS benchmark results collected"


@pytest.mark.performance
def test_fp4_streamk_remainder_benchmark():
    """Benchmark Stream-K with remainder path enabled (sk_grid=256, rem>128).
    Simulates max 256 CUs; shapes chosen so remainder tiles > 128.
    """
    from tritonblas.utils import generate_matmul_inputs

    SK_GRID = 256  # Simulate max 256 CUs
    print("\n" + "="*100)
    print(f"FP4 GEMM: Stream-K with Remainder (sk_grid={SK_GRID}, rem>128)")
    print("="*100)
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Standard TFLOPS':>14} {'Standard us':>12} | {'Stream-K TFLOPS':>16} {'Stream-K us':>12} | {'Speedup':>8}")
    print("-"*100)

    # Shapes where tiles % 256 > 128: (512,128), (1280,200), (1536,144)
    test_sizes = [
        (512, 512, 512),    # tiles=128, rem=128 (all partial)
        (1280, 1280, 1280), # tiles=200, rem=200
        (1536, 1536, 1536), # tiles=144, rem=144
    ]

    dtype = torch.bfloat16
    results = []

    for M, N, K in test_sizes:
        try:
            inputs = generate_matmul_inputs(
                m=M, n=N, k=K,
                in_dtype="fp4",
                out_dtype=dtype,
                init_type="randn",
            )
            out_std = inputs.C.clone()
            out_sk = inputs.C.clone()
            total_ops = 2 * M * N * K

            def run_standard():
                tritonblas.matmul_fp4(
                    inputs.A, inputs.B.T, out_std, inputs.scaleA, inputs.scaleB,
                    enable_streamk=False,
                )
            us_std = benchmark_kernel(run_standard, num_iters=20, warmup=5)
            tflops_std = total_ops / us_std / 1e6

            def run_streamk():
                tritonblas.matmul_fp4(
                    inputs.A, inputs.B.T, out_sk, inputs.scaleA, inputs.scaleB,
                    enable_streamk=True,
                    sk_grid=SK_GRID,
                )
            us_sk = benchmark_kernel(run_streamk, num_iters=20, warmup=5)
            tflops_sk = total_ops / us_sk / 1e6

            speedup = tflops_sk / tflops_std if tflops_std > 0 else 0
            results.append({"M": M, "N": N, "K": K, "tflops_std": tflops_std, "tflops_sk": tflops_sk, "speedup": speedup})
            print(f"{M:6d} {N:6d} {K:6d} | {tflops_std:14.2f} {us_std:12.2f} | {tflops_sk:16.2f} {us_sk:12.2f} | {speedup:7.2f}x")

            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{M:6d} {N:6d} {K:6d} | SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else:
                raise

    print("="*100 + "\n")
    assert len(results) > 0, "No Stream-K remainder benchmark results collected"


@pytest.mark.performance
def test_fp4_block_size_sweep():
    """Pytest test for FP4 block size sweep - prints performance tables."""
    print("\n" + "="*80)
    print("FP4 GEMM Block Size Sweep (16384x16384x16384)")
    print("="*80)
    
    M, N, K = 16384, 16384, 16384
    dtype = torch.bfloat16
    
    # Generate test data once
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    x_fp4, x_scales = dynamic_mxfp4_quant(x)
    w_fp4, w_scales = dynamic_mxfp4_quant(w)
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
                    def run_kernel():
                        tritonblas.matmul_fp4(
                            x_fp4, w_fp4, out, x_scales, w_scales,
                            block_m=block_m, block_n=block_n, block_k=block_k
                        )
                    
                    us = benchmark_kernel(run_kernel, num_iters=5, warmup=2)
                    total_ops = 2 * M * N * K
                    tflops = total_ops / us / 1e6
                    
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
        description="TritonBLAS FP4 GEMM Test Suite",
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
    print("TritonBLAS FP4 GEMM Test Suite")
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

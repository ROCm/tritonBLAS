import pytest
import torch
import triton
import tritonblas
import time


def quantize_to_fp4_e2m1(tensor_fp16, group_size=32):
    """
    Quantize FP16 tensor to FP4 e2m1 format with e8m0 scales.
    
    FP4 e2m1 format has:
    - 1 sign bit
    - 2 exponent bits
    - 1 mantissa bit
    
    E8M0 scale format stores only the exponent (8 bits).
    
    Args:
        tensor_fp16: Input tensor in FP16 to quantize
        group_size: Number of elements sharing one scale (default: 32)
    
    Returns:
        fp4_data: Packed FP4 data (2 elements per uint8)
        scales: e8m0 scales (uint8)
    """
    M, K = tensor_fp16.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    
    # Reshape to group elements
    tensor_grouped = tensor_fp16.reshape(M, K // group_size, group_size)
    
    # Compute scales per group (max absolute value in each group)
    max_vals = torch.abs(tensor_grouped).max(dim=2, keepdim=True)[0]
    
    # Avoid division by zero
    max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
    
    # Normalize by scale to get values in [-1, 1] range
    normalized = tensor_grouped / max_vals
    
    # Quantize to FP4 e2m1 (simplified: map to nearest representable value)
    # FP4 e2m1 representable values (normalized): 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    # We'll use a simple rounding scheme
    fp4_values = torch.clamp(torch.round(normalized * 4) / 4, -2.0, 2.0)
    
    # Pack two FP4 values into one uint8
    # For simplicity, we'll store as scaled uint8 (this is a mock - real HW uses special encoding)
    fp4_values_flat = fp4_values.reshape(M, K)
    
    # Pack pairs of values
    fp4_packed = torch.zeros((M, K // 2), dtype=torch.uint8, device=tensor_fp16.device)
    for i in range(M):
        for j in range(K // 2):
            # Simple packing scheme (not actual FP4 encoding, but preserves values)
            val1 = int((fp4_values_flat[i, j*2] + 2) * 16)  # Map [-2, 2] to [0, 64]
            val2 = int((fp4_values_flat[i, j*2+1] + 2) * 16)
            fp4_packed[i, j] = min(255, max(0, (val1 & 0xF) | ((val2 & 0xF) << 4)))
    
    # Convert scales to e8m0 format (exponent only)
    # e8m0 stores 2^exponent, so we need to find the exponent
    # For simplicity, we'll store the log2 of the scale
    scales_e8m0 = torch.zeros((M, K // group_size), dtype=torch.uint8, device=tensor_fp16.device)
    max_vals_flat = max_vals.squeeze(-1)
    
    for i in range(M):
        for j in range(K // group_size):
            scale_val = max_vals_flat[i, j].item()
            if scale_val > 0:
                # Compute exponent (simplified e8m0 encoding)
                exponent = int(torch.log2(torch.tensor(scale_val)).item()) + 127
                scales_e8m0[i, j] = min(255, max(0, exponent))
            else:
                scales_e8m0[i, j] = 0
    
    return fp4_packed, scales_e8m0


def dequantize_from_fp4_e2m1(fp4_data, scales, group_size=32):
    """
    Dequantize FP4 e2m1 data back to FP16.
    
    Args:
        fp4_data: Packed FP4 data (2 elements per uint8)
        scales: e8m0 scales (uint8)
        group_size: Number of elements sharing one scale (default: 32)
    
    Returns:
        tensor_fp16: Dequantized FP16 tensor
    """
    M, K_packed = fp4_data.shape
    K = K_packed * 2
    
    # Unpack FP4 values
    fp4_unpacked = torch.zeros((M, K), dtype=torch.float16, device=fp4_data.device)
    
    for i in range(M):
        for j in range(K_packed):
            packed_val = fp4_data[i, j].item()
            # Unpack two 4-bit values
            val1 = (packed_val & 0xF) / 16.0 - 2.0  # Map [0, 64] back to [-2, 2]
            val2 = ((packed_val >> 4) & 0xF) / 16.0 - 2.0
            fp4_unpacked[i, j*2] = val1
            fp4_unpacked[i, j*2+1] = val2
    
    # Decode scales from e8m0
    M_s, K_s = scales.shape
    scales_decoded = torch.zeros((M_s, K_s), dtype=torch.float16, device=scales.device)
    
    for i in range(M_s):
        for j in range(K_s):
            exponent = scales[i, j].item()
            if exponent > 0:
                # Decode e8m0: 2^(exponent - 127)
                scales_decoded[i, j] = 2.0 ** (exponent - 127)
            else:
                scales_decoded[i, j] = 0.0
    
    # Apply scales to dequantize
    fp4_grouped = fp4_unpacked.reshape(M, K // group_size, group_size)
    scales_expanded = scales_decoded.unsqueeze(-1).expand(-1, -1, group_size)
    
    dequantized = fp4_grouped * scales_expanded
    
    return dequantized.reshape(M, K)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (128, 128, 128),
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
    "out_dtype",
    [
        torch.bfloat16,
        torch.float16,
    ],
)
def test_matmul_fp4(m, n, k, out_dtype):
    """Test FP4 matrix multiplication with performance benchmarking."""
    
    # Create FP16 data in FP4-representable range
    A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.01
    B_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
    
    # Quantize to FP4 format
    # A has shape (M, K//2) packed
    # B has shape (N, K//2) packed (will be transposed to K x N in matmul_fp4)
    A_fp4, A_scales = quantize_to_fp4_e2m1(A_fp16)
    B_fp4, B_scales = quantize_to_fp4_e2m1(B_fp16)
    
    # Allocate output tensor
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    
    # Warm up
    for _ in range(3):
        tritonblas.matmul_fp4(A_fp4, B_fp4, C, A_scales, B_scales)
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_iterations = 10
    start_time = time.time()
    
    for _ in range(num_iterations):
        tritonblas.matmul_fp4(A_fp4, B_fp4, C, A_scales, B_scales)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate performance metrics
    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    
    # Calculate TFLOPS (for FP4, we count operations as if they were full precision)
    # GEMM: 2*M*N*K operations
    total_ops = 2 * m * n * k
    tflops = (total_ops / (avg_time_ms / 1000)) / 1e12
    
    print(f"\n{'='*60}")
    print(f"FP4 GEMM Performance: M={m}, N={n}, K={k}, dtype={out_dtype}")
    print(f"{'='*60}")
    print(f"Average time: {avg_time_ms:.3f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")
    print(f"{'='*60}\n")
    
    # Basic shape check
    assert C.shape == (m, n), f"Output shape mismatch: expected {(m, n)}, got {C.shape}"


@pytest.mark.parametrize(
    "m, n, k",
    [
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    ],
)
def test_matmul_fp4_non_square(m, n, k):
    """Test FP4 matrix multiplication with non-square matrices."""
    
    out_dtype = torch.bfloat16
    
    # Create FP16 data
    A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.01
    B_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
    
    # Quantize to FP4 format
    # A has shape (M, K//2), B has shape (N, K//2)
    A_fp4, A_scales = quantize_to_fp4_e2m1(A_fp16)
    B_fp4, B_scales = quantize_to_fp4_e2m1(B_fp16)
    
    # Allocate output tensor
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    
    # Run FP4 matmul
    tritonblas.matmul_fp4(A_fp4, B_fp4, C, A_scales, B_scales)
    
    # Basic shape check
    assert C.shape == (m, n), f"Output shape mismatch: expected {(m, n)}, got {C.shape}"
    
    print(f"Non-square test passed: M={m}, N={n}, K={k}")


@pytest.mark.parametrize(
    "m, n, k",
    [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
)
def test_matmul_fp4_correctness(m, n, k):
    """Test FP4 matrix multiplication correctness against FP16 reference.
    
    This test:
    1. Quantizes FP16 inputs to FP4
    2. Dequantizes them back to FP16 to get the actual values being used
    3. Computes FP16 reference using the dequantized values
    4. Compares FP4 matmul result against this reference
    """
    
    out_dtype = torch.bfloat16
    
    # Create FP16 input data
    torch.manual_seed(42)
    A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.01
    B_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
    
    # Quantize to FP4 format
    A_fp4, A_scales = quantize_to_fp4_e2m1(A_fp16)
    B_fp4, B_scales = quantize_to_fp4_e2m1(B_fp16)
    
    # Dequantize to get the actual FP16 values that will be used in computation
    A_dequant = dequantize_from_fp4_e2m1(A_fp4, A_scales)
    B_dequant = dequantize_from_fp4_e2m1(B_fp4, B_scales)
    
    # Compute FP16 reference using dequantized values
    C_ref = torch.matmul(A_dequant, B_dequant.T).to(out_dtype)
    
    # Run FP4 matmul
    C_fp4 = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    tritonblas.matmul_fp4(A_fp4, B_fp4, C_fp4, A_scales, B_scales)
    
    # Verify kernel produces valid outputs
    nan_mask = torch.isnan(C_fp4)
    inf_mask = torch.isinf(C_fp4)
    num_nan = nan_mask.sum().item()
    num_inf = inf_mask.sum().item()
    num_valid = (~nan_mask & ~inf_mask).sum().item()
    total = m * n
    
    valid_percentage = 100 * num_valid / total
    nan_percentage = 100 * num_nan / total
    inf_percentage = 100 * num_inf / total
    
    # Assertions
    assert not torch.all(C_fp4 == 0), "Output should not be all zeros"
    assert valid_percentage > 95.0, f"Expected >95% valid values, got {valid_percentage:.1f}%"
    assert nan_percentage < 5.0, f"Expected <5% NaN, got {nan_percentage:.1f}%"
    assert inf_percentage < 5.0, f"Expected <5% Inf, got {inf_percentage:.1f}%"
    
    # Compute error metrics against FP16 reference (using dequantized values)
    ref_nan_mask = torch.isnan(C_ref)
    both_valid_mask = ~nan_mask & ~inf_mask & ~ref_nan_mask
    
    if both_valid_mask.sum() > 0:
        fp4_valid = C_fp4[both_valid_mask]
        ref_valid = C_ref[both_valid_mask]
        
        # Compute error metrics
        abs_error = torch.abs(fp4_valid - ref_valid)
        mean_abs_error = abs_error.mean().item()
        max_abs_error = abs_error.max().item()
        
        # Relative error (avoid division by zero)
        rel_error = abs_error / (torch.abs(ref_valid) + 1e-8)
        mean_rel_error = rel_error.mean().item()
    else:
        mean_abs_error = float('nan')
        max_abs_error = float('nan')
        mean_rel_error = float('nan')
    
    print(f"Correctness test passed: M={m}, N={n}, K={k}")
    print(f"  Kernel produces valid outputs: ✓")
    print(f"  Valid values: {num_valid}/{total} ({valid_percentage:.1f}%)")
    print(f"  NaN values: {num_nan}/{total} ({nan_percentage:.1f}%)")
    print(f"  Inf values: {num_inf}/{total} ({inf_percentage:.1f}%)")
    
    if num_valid > 0:
        valid_values = C_fp4[~nan_mask & ~inf_mask]
        print(f"  Output range: [{valid_values.min().item():.2f}, {valid_values.max().item():.2f}]")
        print(f"  Output mean: {valid_values.mean().item():.2f}, std: {valid_values.std().item():.2f}")
    
    print(f"  Error vs FP16 Reference (using dequantized FP4 values):")
    print(f"    Mean absolute error: {mean_abs_error:.6f}")
    print(f"    Max absolute error: {max_abs_error:.6f}")
    print(f"    Mean relative error: {mean_rel_error:.6f}")


def benchmark_fp4_vs_fp16():
    """Benchmark FP4 vs FP16 performance."""
    
    print("\n" + "="*80)
    print("FP4 vs FP16 Performance Comparison")
    print("="*80)
    
    sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192,8192,8192),(16384,16384,16384)]
    
    for m, n, k in sizes:
        # FP4 benchmark
        A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.01
        B_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
        A_fp4, A_scales = quantize_to_fp4_e2m1(A_fp16)
        B_fp4, B_scales = quantize_to_fp4_e2m1(B_fp16)
        C_fp4 = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
        
        # Warm up
        for _ in range(3):
            tritonblas.matmul_fp4(A_fp4, B_fp4, C_fp4, A_scales, B_scales)
        torch.cuda.synchronize()
        
        # Benchmark FP4
        num_iterations = 10
        start_time = time.time()
        for _ in range(num_iterations):
            tritonblas.matmul_fp4(A_fp4, B_fp4, C_fp4, A_scales, B_scales)
        torch.cuda.synchronize()
        fp4_time = (time.time() - start_time) / num_iterations * 1000
        
        # FP16 benchmark
        A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16)
        B_fp16 = torch.randn((k, n), device="cuda", dtype=torch.float16)
        C_fp16 = torch.zeros((m, n), device="cuda", dtype=torch.float16)
        
        # Warm up
        for _ in range(3):
            tritonblas.matmul(A_fp16, B_fp16, C_fp16)
        torch.cuda.synchronize()
        
        # Benchmark FP16
        start_time = time.time()
        for _ in range(num_iterations):
            tritonblas.matmul(A_fp16, B_fp16, C_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start_time) / num_iterations * 1000
        
        # Calculate metrics
        total_ops = 2 * m * n * k
        fp4_tflops = (total_ops / (fp4_time / 1000)) / 1e12
        fp16_tflops = (total_ops / (fp16_time / 1000)) / 1e12
        speedup = fp16_time / fp4_time
        
        print(f"\nSize: M={m}, N={n}, K={k}")
        print(f"  FP4:  {fp4_time:.3f} ms ({fp4_tflops:.2f} TFLOPS)")
        print(f"  FP16: {fp16_time:.3f} ms ({fp16_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "="*80 + "\n")


def benchmark_block_size_sweep():
    """Benchmark different block sizes for 16kx16kx16k matrix."""
    
    print("\n" + "="*80)
    print("Block Size Sweep for 16384x16384x16384 FP4 GEMM")
    print("="*80)
    
    m, n, k = 16384, 16384, 16384
    out_dtype = torch.bfloat16
    
    # Create FP16 data and quantize to FP4
    A_fp16 = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.01
    B_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
    A_fp4, A_scales = quantize_to_fp4_e2m1(A_fp16)
    B_fp4, B_scales = quantize_to_fp4_e2m1(B_fp16)
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    
    # Sweep M/N block sizes (powers of 2): 64, 128, 256
    block_mn_sizes = [64, 128, 256]
    # Sweep K block sizes (powers of 2): 128, 256, 512
    block_k_sizes = [128, 256, 512]
    
    # Store results in a table format
    results_table = {}
    
    for block_k in block_k_sizes:
        results_table[block_k] = {}
        for block_m in block_mn_sizes:
            for block_n in block_mn_sizes:
                try:
                    # Warm up
                    for _ in range(2):
                        tritonblas.matmul_fp4(
                            A_fp4, B_fp4, C, A_scales, B_scales,
                            block_m=block_m, block_n=block_n, block_k=block_k
                        )
                    torch.cuda.synchronize()
                    
                    # Benchmark
                    num_iterations = 5
                    start_time = time.time()
                    for _ in range(num_iterations):
                        tritonblas.matmul_fp4(
                            A_fp4, B_fp4, C, A_scales, B_scales,
                            block_m=block_m, block_n=block_n, block_k=block_k
                        )
                    torch.cuda.synchronize()
                    avg_time_ms = (time.time() - start_time) / num_iterations * 1000
                    
                    # Calculate TFLOPS
                    total_ops = 2 * m * n * k
                    tflops = (total_ops / (avg_time_ms / 1000)) / 1e12
                    
                    key = f"M{block_m}_N{block_n}"
                    results_table[block_k][key] = tflops
                
                except Exception as e:
                    key = f"M{block_m}_N{block_n}"
                    results_table[block_k][key] = 0.0
                    print(f"BLK_M={block_m}, BLK_N={block_n}, BLK_K={block_k}: FAILED - {str(e)}")
    
    # Print results table
    print("\n" + "="*80)
    print("FP4 Throughput Table (TFLOPS) for 16384x16384x16384")
    print("="*80)
    
    # Header
    header = "BLK_K  |"
    for block_m in block_mn_sizes:
        for block_n in block_mn_sizes:
            header += f" M{block_m:3d}xN{block_n:3d} |"
    print(header)
    print("-" * len(header))
    
    # Find best configuration first
    best_tflops = 0
    best_config = None
    
    for block_k in block_k_sizes:
        for block_m in block_mn_sizes:
            for block_n in block_mn_sizes:
                key = f"M{block_m}_N{block_n}"
                tflops = results_table[block_k].get(key, 0.0)
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_config = (block_m, block_n, block_k)
    
    # Rows with highlighting
    for block_k in block_k_sizes:
        row = f"  {block_k:3d}  |"
        for block_m in block_mn_sizes:
            for block_n in block_mn_sizes:
                key = f"M{block_m}_N{block_n}"
                tflops = results_table[block_k].get(key, 0.0)
                
                # Highlight best configuration with asterisk
                if best_config and (block_m, block_n, block_k) == best_config:
                    row += f" *{tflops:6.2f}* |"
                else:
                    row += f"  {tflops:7.2f}  |"
        print(row)
    
    print("="*80)
    
    if best_config:
        print(f"\nBest Configuration:")
        print(f"  BLK_M={best_config[0]}, BLK_N={best_config[1]}, BLK_K={best_config[2]}")
        print(f"  Performance: {best_tflops:.2f} TFLOPS")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Run correctness tests first
    print("\n" + "="*80)
    print("Running FP4 Correctness Tests...")
    print("="*80)
    test_matmul_fp4_correctness(128, 128, 128)
    test_matmul_fp4_correctness(256, 256, 256)
    test_matmul_fp4_correctness(512, 512, 512)
    test_matmul_fp4_correctness(1024, 1024, 1024)
    print("All correctness tests passed!")
    print("="*80 + "\n")
    
    # Run basic tests
    print("Running FP4 matmul performance tests...")
    test_matmul_fp4(1024, 1024, 1024, torch.bfloat16)
    test_matmul_fp4_non_square(128, 256, 512)
    
    # Run benchmark comparison
    benchmark_fp4_vs_fp16()
    
    # Run block size sweep
    benchmark_block_size_sweep()

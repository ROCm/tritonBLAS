# Quick Start Guide

Get up and running with tritonBLAS in minutes!

## Your First Matrix Multiplication

Here's a simple example to get you started:

```python
import torch
import tritonblas

# Create input matrices
A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')

# Perform matrix multiplication
tritonblas.matmul(A, B, C)

print(f"Result shape: {C.shape}")
```

## Using the Peak Performance API

For optimal performance, use the two-step API that separates configuration from execution:

```python
import torch
import tritonblas

# Step 1: Get optimal configuration for your matrix dimensions
m, n, k = 4096, 4096, 4096
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Step 2: Create input matrices
A = torch.randn(m, k, dtype=torch.float16, device='cuda')
B = torch.randn(k, n, dtype=torch.float16, device='cuda')
C = torch.zeros(m, n, dtype=torch.float16, device='cuda')

# Step 3: Perform matrix multiplication with optimal config
tritonblas.matmul_lt(A, B, C, selector)

print(f"Result shape: {C.shape}")
```

## Running the Examples

tritonBLAS comes with several examples in the `examples/` directory:

```bash
cd examples

# Basic matrix multiplication
python3 example_matmul.py

# Matrix multiplication with left transpose
python3 example_matmul_lt.py
```

## Common Use Cases

### 1. Different Data Types

tritonBLAS supports multiple data types:

```python
import torch
import tritonblas

# FP32
A_fp32 = torch.randn(2048, 2048, dtype=torch.float32, device='cuda')
B_fp32 = torch.randn(2048, 2048, dtype=torch.float32, device='cuda')
C_fp32 = torch.zeros(2048, 2048, dtype=torch.float32, device='cuda')
tritonblas.matmul(A_fp32, B_fp32, C_fp32)

# FP16
A_fp16 = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
B_fp16 = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
C_fp16 = torch.zeros(2048, 2048, dtype=torch.float16, device='cuda')
tritonblas.matmul(A_fp16, B_fp16, C_fp16)

# BF16
A_bf16 = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
B_bf16 = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
C_bf16 = torch.zeros(2048, 2048, dtype=torch.bfloat16, device='cuda')
tritonblas.matmul(A_bf16, B_bf16, C_bf16)
```

### 2. Using Stream-K Algorithm

Enable the Stream-K algorithm for improved performance on certain workloads:

```python
import torch
import tritonblas

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')

# Enable Stream-K
tritonblas.matmul(A, B, C, enable_streamk=True)
```

### 4. Reusing Configuration

When performing multiple operations with the same dimensions, reuse the configuration:

```python
import torch
import tritonblas

# Get configuration once
m, n, k = 2048, 2048, 2048
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Reuse for multiple operations
C = torch.zeros(m, n, dtype=torch.float16, device='cuda')
for i in range(10):
    A = torch.randn(m, k, dtype=torch.float16, device='cuda')
    B = torch.randn(k, n, dtype=torch.float16, device='cuda')
    tritonblas.matmul_lt(A, B, C, selector)
```

## Performance Tips

1. **Use the Peak Performance API**: The `matmul_lt` function with pre-computed selector is faster than `matmul`
2. **Reuse Configurations**: When dimensions don't change, reuse the `MatmulHeuristicResult`
3. **Choose Appropriate Data Types**: FP16 and BF16 are typically faster than FP32
4. **Experiment with Stream-K**: For certain workloads, Stream-K can provide better performance

## Benchmarking

To benchmark tritonBLAS against other implementations:

```bash
cd benchmarks

# Compare with torch.matmul
python3 torch_matmul.py

# Benchmark tritonBLAS
python3 tritonblas_matmul.py
```

## Next Steps

Now that you're familiar with the basics:

1. Explore more [Examples](examples.md)
2. Learn about the [Analytical Model](../conceptual/analytical-model.md)
3. Read the [API Reference](../reference/api.md) for detailed documentation
4. Check out [Performance](../conceptual/performance.md) for optimization techniques

## Getting Help

If you run into issues:

- Check the [Installation Guide](installation.md) for setup problems
- Review [Examples](examples.md) for working code
- Open an issue on [GitHub](https://github.com/ROCm/tritonBLAS/issues)

# Examples

This page provides working examples demonstrating various features of tritonBLAS.

## Basic Examples

### Simple Matrix Multiplication

The most basic usage of tritonBLAS:

```python
import torch
import tritonblas

# Create matrices
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
C = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda')

# Multiply
tritonblas.matmul(A, B, C)
```

### Using Peak Performance API

For best performance, use the configuration-based API:

```python
import torch
import tritonblas

# Define dimensions
m, n, k = 4096, 4096, 4096

# Get optimal configuration
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Create matrices
A = torch.randn(m, k, dtype=torch.float16, device='cuda')
B = torch.randn(k, n, dtype=torch.float16, device='cuda')
C = torch.zeros(m, n, dtype=torch.float16, device='cuda')

# Multiply with configuration
tritonblas.matmul_lt(A, B, C, selector)
```

## Data Type Examples

### FP16 (Half Precision)

```python
import torch
import tritonblas

A = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
B = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
C = torch.zeros(2048, 2048, dtype=torch.float16, device='cuda')
tritonblas.matmul(A, B, C)
```

### BF16 (Brain Float 16)

```python
import torch
import tritonblas

A = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
B = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
C = torch.zeros(2048, 2048, dtype=torch.bfloat16, device='cuda')
tritonblas.matmul(A, B, C)
```

### FP32 (Single Precision)

```python
import torch
import tritonblas

A = torch.randn(2048, 2048, dtype=torch.float32, device='cuda')
B = torch.randn(2048, 2048, dtype=torch.float32, device='cuda')
C = torch.zeros(2048, 2048, dtype=torch.float32, device='cuda')
tritonblas.matmul(A, B, C)
```

### FP8 (8-bit Floating Point)

```python
import torch
import tritonblas

# Note: FP8 support requires appropriate hardware
A = torch.randn(2048, 2048, dtype=torch.float8_e4m3fn, device='cuda')
B = torch.randn(2048, 2048, dtype=torch.float8_e4m3fn, device='cuda')
C = torch.zeros(2048, 2048, dtype=torch.float8_e4m3fn, device='cuda')
tritonblas.matmul(A, B, C)
```

## Advanced Examples

### Stream-K Algorithm

```python
import torch
import tritonblas

A = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')
B = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')
C = torch.zeros(8192, 8192, dtype=torch.float16, device='cuda')

# Enable Stream-K for better load balancing
tritonblas.matmul(A, B, C, enable_streamk=True)
```

### Configuration Reuse

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

# Reuse configuration for multiple operations
C = torch.zeros(m, n, dtype=torch.float16, device='cuda')
results = []
for i in range(100):
    A = torch.randn(m, k, dtype=torch.float16, device='cuda')
    B = torch.randn(k, n, dtype=torch.float16, device='cuda')
    tritonblas.matmul_lt(A, B, C, selector)
    results.append(C.clone())
```

### Output Tensor Specification

```python
import torch
import tritonblas

A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Pre-allocate output tensor
C = torch.empty(1024, 1024, dtype=torch.float16, device='cuda')

# Use pre-allocated output
tritonblas.matmul(A, B, C)
```

## Benchmarking Examples

### Simple Benchmark

```python
import torch
import tritonblas
import time

def benchmark_matmul(m, n, k, dtype, num_iterations=100):
    # Create matrices
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    C = torch.zeros(m, n, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(10):
        tritonblas.matmul(A, B, C)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        tritonblas.matmul(A, B, C)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    tflops = (2 * m * n * k) / (avg_time * 1e12)
    
    print(f"Shape: ({m}, {n}, {k})")
    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")

# Run benchmark
benchmark_matmul(4096, 4096, 4096, torch.float16)
```

### Comparison with PyTorch

```python
import torch
import tritonblas
import time

def compare_implementations(m, n, k, dtype):
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    C_torch = torch.zeros(m, n, dtype=dtype, device='cuda')
    C_triton = torch.zeros(m, n, dtype=dtype, device='cuda')
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        torch.matmul(A, B, out=C_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    
    # Benchmark tritonBLAS
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        tritonblas.matmul(A, B, C_triton)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    
    print(f"PyTorch: {torch_time*1000:.2f} ms")
    print(f"tritonBLAS: {triton_time*1000:.2f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

compare_implementations(4096, 4096, 4096, torch.float16)
```

## Running the Built-in Examples

tritonBLAS includes several example scripts in the `examples/` directory:

```bash
cd examples

# Basic matrix multiplication
python3 example_matmul.py

# Matrix multiplication with left transpose
python3 example_matmul_lt.py
```

## Next Steps

- Learn about the [Analytical Model](../conceptual/analytical-model.md)
- Explore [Performance Optimization](../conceptual/performance.md)
- Read the [API Reference](../reference/api.md)

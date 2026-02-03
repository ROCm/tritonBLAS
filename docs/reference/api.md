# Core API Guide

This guide introduces the tritonBLAS matrix multiplication APIs and their usage patterns. For complete API signatures and parameters, see the [Core API Reference](api-autodoc.rst).

## Overview

tritonBLAS provides high-performance matrix multiplication with two main interfaces:

- **`matmul`** - Simple API with automatic configuration
- **`matmul_lt`** - Peak performance API with explicit configuration management

Both APIs follow a similar pattern: provide input matrices A and B, a pre-allocated output matrix C, and tritonBLAS handles the rest.

## Basic Usage

The simplest way to use tritonBLAS:

```python
import torch
import tritonblas

# Create input matrices
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
C = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda')

# Compute C = A @ B
tritonblas.matmul(A, B, C)
```

## Peak Performance with matmul_lt

For maximum performance in production workloads, use `matmul_lt` with a pre-configured selector. This avoids configuration overhead on each call:

```python
import torch
import tritonblas

# Define problem size
m, n, k = 4096, 4096, 4096
dtype = torch.float16

# Create selector once (can be reused)
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=dtype, b_dtype=dtype, c_dtype=dtype
)

# Allocate tensors
A = torch.randn(m, k, dtype=dtype, device='cuda')
B = torch.randn(k, n, dtype=dtype, device='cuda')
C = torch.zeros(m, n, dtype=dtype, device='cuda')

# Reuse selector for multiple operations
for _ in range(1000):
    tritonblas.matmul_lt(A, B, C, selector)
```

## Stream-K for Better Load Balancing

Enable Stream-K when you have workloads with uneven tile distributions:

```python
# Enable Stream-K GEMM
tritonblas.matmul(A, B, C, enable_streamk=True)

# Or with matmul_lt
tritonblas.matmul_lt(A, B, C, selector, enable_streamk=True)
```

## Quantized Operations

For INT8 quantized inference:

```python
# A8W8 quantized GEMM
tritonblas.matmul_a8w8(A_int8, B_int8, C, A_scale, B_scale)

# With explicit configuration
tritonblas.matmul_a8w8_lt(A_int8, B_int8, C, A_scale, B_scale, selector)
```

For FP4 quantized inference:

```python
# FP4 quantized GEMM
tritonblas.matmul_fp4(A_fp4, B_fp4, C, A_scale, B_scale)
```

## Supported Data Types

| Data Type | PyTorch Type | Description |
|-----------|--------------|-------------|
| FP32 | `torch.float32` | 32-bit floating point |
| FP16 | `torch.float16` | 16-bit floating point |
| BF16 | `torch.bfloat16` | Brain float 16 |
| TF32 | `torch.float32` (with TF32 mode) | TensorFloat-32 |
| FP8 | `torch.float8_e4m3fn` | 8-bit floating point |

## Best Practices

### 1. Reuse Configuration Objects

```python
# Good: Create selector once
selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
for batch in dataloader:
    tritonblas.matmul_lt(batch.A, batch.B, batch.C, selector)
```

### 2. Ensure Contiguous Tensors

```python
# Ensure inputs are contiguous
if not A.is_contiguous():
    A = A.contiguous()
```

### 3. Pre-allocate Output

```python
# Pre-allocate output tensor
C = torch.zeros(m, n, dtype=dtype, device='cuda')
```

## See Also

- [Core API Reference](api-autodoc.rst): Complete API signatures and parameters
- [Configuration Guide](configuration.md): Detailed configuration options
- [Advanced Features](advanced.md): Stream-K and other optimizations
- [Examples](../getting-started/examples.md): Working code examples

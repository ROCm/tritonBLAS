# API Reference

Complete API documentation for tritonBLAS.

## Core Functions

### matmul

```python
tritonblas.matmul(input, other, out, *, enable_streamk=False, sk_grid=None) → Tensor
```

Matrix multiplication with automatic configuration selection.

**Parameters:**

- **input** (*Tensor*) – The first tensor to be multiplied (M x K)
- **other** (*Tensor*) – The second tensor to be multiplied (K x N)
- **out** (*Tensor*) – The output tensor (M x N). Must be pre-allocated.

**Keyword Arguments:**

- **enable_streamk** (*bool*, optional) – Enable Stream-K GEMM algorithm for better load balancing. Default: `False`
- **sk_grid** (*int*, optional) – Stream-K grid size override. Only used when `enable_streamk=True`. Default: `None`

**Returns:**

- *Tensor* – The output tensor (same as `out` parameter)

**Example:**

```python
import torch
import tritonblas

A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
C = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda')

# Basic usage
tritonblas.matmul(A, B, C)

# With Stream-K
tritonblas.matmul(A, B, C, enable_streamk=True)
```

---

### matmul_lt

```python
tritonblas.matmul_lt(input, other, out, selector, *, enable_streamk=False) → Tensor
```

Peak performance API with explicit configuration management.

**Parameters:**

- **input** (*Tensor*) – The first tensor to be multiplied (M x K)
- **other** (*Tensor*) – The second tensor to be multiplied (K x N)
- **out** (*Tensor*) – The output tensor (M x N). Must be pre-allocated.
- **selector** (*MatmulHeuristicResult*) – Configuration object containing optimal kernel parameters

**Keyword Arguments:**

- **enable_streamk** (*bool*, optional) – Enable Stream-K GEMM algorithm. Default: `False`

**Returns:**

- *Tensor* – The output tensor (same as `out` parameter)

**Example:**

```python
import torch
import tritonblas

# Get optimal configuration
m, n, k = 4096, 4096, 4096
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Use configuration
A = torch.randn(m, k, dtype=torch.float16, device='cuda')
B = torch.randn(k, n, dtype=torch.float16, device='cuda')
C = torch.zeros(m, n, dtype=torch.float16, device='cuda')
tritonblas.matmul_lt(A, B, C, selector)
```

---

### MatmulHeuristicResult

```python
tritonblas.MatmulHeuristicResult(m, n, k, a_dtype, b_dtype, c_dtype) → MatmulHeuristicResult
```

Creates a configuration object with optimal kernel parameters for the given matrix dimensions and data types.

**Parameters:**

- **m** (*int*) – Number of rows of the left-hand matrix
- **n** (*int*) – Number of columns of the right-hand matrix
- **k** (*int*) – Shared dimension (columns of left matrix, rows of right matrix)
- **a_dtype** (*torch.dtype*) – Data type of the left-hand matrix
- **b_dtype** (*torch.dtype*) – Data type of the right-hand matrix
- **c_dtype** (*torch.dtype*) – Data type of the output matrix

**Returns:**

- *MatmulHeuristicResult* – Configuration object containing optimal kernel parameters

**Example:**

```python
import torch
import tritonblas

# Create configuration for FP16 matrices
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Reuse configuration for multiple operations
C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')
for i in range(100):
    A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    tritonblas.matmul_lt(A, B, C, selector)
```

## Supported Data Types

tritonBLAS supports the following PyTorch data types:

| Data Type | PyTorch Type | Description |
|-----------|--------------|-------------|
| FP32 | `torch.float32` | 32-bit floating point |
| FP16 | `torch.float16` | 16-bit floating point |
| BF16 | `torch.bfloat16` | Brain float 16 |
| TF32 | `torch.float32` (with TF32 mode) | TensorFloat-32 |
| FP8 | `torch.float8_e4m3fn` | 8-bit floating point |

## Supported Operations

### Matrix Transpose Modes

tritonBLAS supports all combinations of transpose modes:

| Mode | Description | Example |
|------|-------------|---------|
| N/N | No transpose | `C = A @ B` |
| T/N | Left transpose | `C = A.T @ B` |
| N/T | Right transpose | `C = A @ B.T` |
| T/T | Both transpose | `C = A.T @ B.T` |

## Configuration Details

### MatmulHeuristicResult Attributes

The `MatmulHeuristicResult` object contains:

- **Block sizes**: Optimal tile dimensions
- **Thread configuration**: Warp and thread counts
- **Memory layout**: Data arrangement strategy
- **Pipeline stages**: Computation/memory overlap configuration

These are determined automatically by the analytical model and should not be modified directly.

## Error Handling

tritonBLAS raises standard PyTorch exceptions:

```python
# Dimension mismatch
A = torch.randn(100, 200, device='cuda')
B = torch.randn(300, 400, device='cuda')
C = torch.zeros(100, 400, device='cuda')
tritonblas.matmul(A, B, C)  # RuntimeError: incompatible dimensions

# Type mismatch
A = torch.randn(100, 100, dtype=torch.float16, device='cuda')
B = torch.randn(100, 100, dtype=torch.float32, device='cuda')
C = torch.zeros(100, 100, dtype=torch.float16, device='cuda')
tritonblas.matmul(A, B, C)  # RuntimeError: dtype mismatch

# Device mismatch
A = torch.randn(100, 100, device='cuda')
B = torch.randn(100, 100, device='cpu')
C = torch.zeros(100, 100, device='cuda')
tritonblas.matmul(A, B, C)  # RuntimeError: device mismatch
```

## Performance Considerations

### Configuration Reuse

For best performance, reuse `MatmulHeuristicResult` objects:

```python
# Good: Reuse configuration
selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
C = torch.zeros(m, n, dtype=dtype, device='cuda')
for _ in range(1000):
    tritonblas.matmul_lt(A, B, C, selector)

# Bad: Recreate configuration each time
C = torch.zeros(m, n, dtype=dtype, device='cuda')
for _ in range(1000):
    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    tritonblas.matmul_lt(A, B, C, selector)
```

### Memory Layout

Ensure tensors are contiguous for optimal performance:

```python
if not A.is_contiguous():
    A = A.contiguous()
if not B.is_contiguous():
    B = B.contiguous()

C = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device='cuda')
tritonblas.matmul(A, B, C)
```

## See Also

- [Configuration Guide](configuration.md): Detailed configuration options
- [Advanced Features](advanced.md): Stream-K and other optimizations
- [Examples](../getting-started/examples.md): Working code examples

# Advanced Features

This page covers advanced features and optimization techniques in tritonBLAS.

## Stream-K Algorithm

Stream-K is an alternative GEMM algorithm that provides better load balancing for certain workloads.

### What is Stream-K?

Stream-K (Streaming-K) is a matrix multiplication algorithm that:
- Splits work more evenly across GPU compute units
- Reduces tail effects in irregular workloads
- Can improve performance for specific matrix shapes

### When to Use Stream-K

Stream-K is beneficial for:
- **Irregular matrix shapes**: Non-square or unusual dimensions
- **Load imbalance**: When standard tiling creates uneven work distribution
- **Specific workloads**: Certain M, N, K combinations

### Enabling Stream-K

```python
import torch
import tritonblas

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')

# Enable Stream-K
tritonblas.matmul(A, B, C, enable_streamk=True)
```

With configuration API:

```python
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')
tritonblas.matmul_lt(A, B, C, selector, enable_streamk=True)
```

### Stream-K Performance Characteristics

**Advantages:**
- Better load balancing
- Reduced tail effects
- Can improve performance for irregular shapes

**Trade-offs:**
- Slightly higher overhead
- May not benefit all workloads
- Best determined through benchmarking

### Benchmarking Stream-K

```python
import torch
import tritonblas
import time

def compare_algorithms(m, n, k, dtype):
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    C = torch.zeros(m, n, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(10):
        tritonblas.matmul(A, B, C)
        tritonblas.matmul(A, B, C, enable_streamk=True)
    
    # Benchmark standard
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        tritonblas.matmul(A, B, C)
    torch.cuda.synchronize()
    standard_time = (time.time() - start) / 100
    
    # Benchmark Stream-K
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        tritonblas.matmul(A, B, C, enable_streamk=True)
    torch.cuda.synchronize()
    streamk_time = (time.time() - start) / 100
    
    print(f"Standard: {standard_time*1000:.2f} ms")
    print(f"Stream-K: {streamk_time*1000:.2f} ms")
    print(f"Speedup: {standard_time/streamk_time:.2f}x")

compare_algorithms(4096, 4096, 4096, torch.float16)
```

## Output Tensor Management

### Pre-allocated Output

For memory-sensitive applications, pre-allocate the output tensor:

```python
import torch
import tritonblas

A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Pre-allocate output
C = torch.empty(1024, 1024, dtype=torch.float16, device='cuda')

# Use pre-allocated tensor
tritonblas.matmul(A, B, C)
```

### In-place Operations

While tritonBLAS doesn't support true in-place operations (C = C @ B), you can reuse output tensors:

```python
# Allocate once
C = torch.empty(1024, 1024, dtype=torch.float16, device='cuda')

# Reuse for multiple operations
for i in range(100):
    A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    tritonblas.matmul(A, B, C)
    # Process C...
```

## Mixed Precision Operations

### Different Input/Output Types

```python
import torch
import tritonblas

# FP16 inputs, FP32 output
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float32  # Higher precision output
)

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.zeros(4096, 4096, dtype=torch.float32, device='cuda')
tritonblas.matmul_lt(A, B, C, selector)

print(C.dtype)  # torch.float32
```

### Accumulation Precision

The internal accumulation precision is determined by the output data type:

```python
# FP32 accumulation (more accurate)
selector_fp32 = tritonblas.MatmulHeuristicResult(
    m, n, k,
    torch.float16, torch.float16, torch.float32
)

# FP16 accumulation (faster)
selector_fp16 = tritonblas.MatmulHeuristicResult(
    m, n, k,
    torch.float16, torch.float16, torch.float16
)
```

## Transpose Operations

### Efficient Transpose Handling

tritonBLAS efficiently handles all transpose combinations:

```python
import torch
import tritonblas

A = torch.randn(1024, 2048, dtype=torch.float16, device='cuda')
B = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')

# No transpose (N/N)
C = torch.zeros(1024, 4096, dtype=torch.float16, device='cuda')
tritonblas.matmul(A, B, C)  # [1024, 4096]

# Left transpose (T/N)
C = torch.zeros(2048, 4096, dtype=torch.float16, device='cuda')
tritonblas.matmul(A.T, B, C)  # [2048, 4096]

# Right transpose (N/T)
C = torch.zeros(1024, 2048, dtype=torch.float16, device='cuda')
tritonblas.matmul(A, B.T, C)  # [1024, 2048]

# Both transpose (T/T)
C = torch.zeros(2048, 2048, dtype=torch.float16, device='cuda')
tritonblas.matmul(A.T, B.T, C)  # [2048, 2048]
```

### Transpose Performance

Transpose operations are handled efficiently without explicit data movement:

```python
# These have similar performance
C1 = torch.zeros(A.shape[0], B.shape[1], dtype=A.dtype, device='cuda')
C2 = torch.zeros(A.T.shape[0], B.T.shape[1], dtype=A.dtype, device='cuda')
tritonblas.matmul(A, B, C1)
tritonblas.matmul(A.T, B.T, C2)
```

## Memory Optimization

### Contiguous Tensors

Ensure tensors are contiguous for optimal performance:

```python
import torch
import tritonblas

A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Make contiguous if needed
if not A.is_contiguous():
    A = A.contiguous()
if not B.is_contiguous():
    B = B.contiguous()

C = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda')
tritonblas.matmul(A, B, C)
```

### Memory Layout

For best performance, use row-major (C-contiguous) layout:

```python
# Good: Row-major layout
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Less optimal: Column-major layout
A_col = A.T.contiguous().T  # Creates column-major tensor
```

## Integration with PyTorch

### Autograd Support

tritonBLAS operations support PyTorch autograd:

```python
import torch
import tritonblas

A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda', requires_grad=True)
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda', requires_grad=True)
C = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda')

# Forward pass
tritonblas.matmul(A, B, C)

# Backward pass
loss = C.sum()
loss.backward()

# Gradients are computed
print(A.grad.shape)  # [1024, 1024]
print(B.grad.shape)  # [1024, 1024]
```

### JIT Compilation

tritonBLAS uses JIT compilation. First call includes compilation overhead:

```python
import torch
import tritonblas
import time

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

C = torch.zeros(4096, 4096, dtype=torch.float16, device='cuda')

# First call: includes compilation
start = time.time()
tritonblas.matmul(A, B, C)
torch.cuda.synchronize()
first_call = time.time() - start

# Subsequent calls: no compilation
start = time.time()
tritonblas.matmul(A, B, C)
torch.cuda.synchronize()
subsequent_call = time.time() - start

print(f"First call: {first_call*1000:.2f} ms")
print(f"Subsequent: {subsequent_call*1000:.2f} ms")
```

## Performance Tuning

### Warmup Strategy

```python
def warmup_tritonblas(shapes, dtypes, iterations=10):
    """Warmup tritonBLAS for common shapes and types."""
    for (m, n, k), dtype in zip(shapes, dtypes):
        A = torch.randn(m, k, dtype=dtype, device='cuda')
        B = torch.randn(k, n, dtype=dtype, device='cuda')
        C = torch.zeros(m, n, dtype=dtype, device='cuda')
        for _ in range(iterations):
            tritonblas.matmul(A, B, C)

# Warmup common configurations
shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
dtypes = [torch.float16, torch.float16, torch.float16]
warmup_tritonblas(shapes, dtypes)
```

### Configuration Caching

```python
class ConfigCache:
    def __init__(self):
        self.cache = {}
    
    def get_config(self, m, n, k, a_dtype, b_dtype, c_dtype):
        key = (m, n, k, a_dtype, b_dtype, c_dtype)
        if key not in self.cache:
            self.cache[key] = tritonblas.MatmulHeuristicResult(
                m, n, k, a_dtype, b_dtype, c_dtype
            )
        return self.cache[key]

# Use cached configurations
cache = ConfigCache()
selector = cache.get_config(4096, 4096, 4096, 
                           torch.float16, torch.float16, torch.float16)
```

## See Also

- [API Reference](api.md): Complete API documentation
- [Configuration](configuration.md): Configuration options
- [Performance](../conceptual/performance.md): Performance optimization

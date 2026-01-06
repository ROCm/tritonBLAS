# Performance Recommendations

This page provides performance optimization recommendations and best practices for tritonBLAS.

## Performance Characteristics

tritonBLAS achieves competitive performance with vendor libraries while offering:
- **No autotuning overhead**: Instant configuration selection via Origami analytical model
- **Smaller binary size**: Fewer kernel variants compiled
- **Predictable behavior**: Deterministic performance
- **Easy debugging**: Explainable configuration choices

## Key Performance Advantages

### Compilation Efficiency

**Traditional Autotuned Approach:**
- First call: 10-30 seconds (evaluating many configurations)
- Generates 50-100+ kernel variants
- Unpredictable compilation time

**tritonBLAS with Origami:**
- First call: <1 second (analytical model selection)
- Generates 1-5 kernel variants
- Predictable, minimal compilation overhead

## Optimization Recommendations

### 1. Configuration Reuse

**Problem**: Repeated configuration selection adds overhead

**Solution**: Reuse `MatmulHeuristicResult` for same dimensions

```python
import torch
import tritonblas

# Get configuration once
m, n, k = 4096, 4096, 4096
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# Reuse for multiple operations
for i in range(1000):
    A = torch.randn(m, k, dtype=torch.float16, device='cuda')
    B = torch.randn(k, n, dtype=torch.float16, device='cuda')
    C = tritonblas.matmul_lt(A, B, selector=selector)
```

**Impact**: Eliminates configuration overhead for repeated operations

### 2. Data Type Selection

**FP16/BF16 vs FP32**
- FP16 and BF16 are typically 2x faster than FP32
- Use lower precision when accuracy permits
- Consider mixed precision for best balance

```python
# Faster with FP16
A_fp16 = A.to(torch.float16)
B_fp16 = B.to(torch.float16)
C = tritonblas.matmul(A_fp16, B_fp16)

# Convert back if needed
C_fp32 = C.to(torch.float32)
```

### 3. Stream-K Algorithm

**When to Use**:
- Irregular matrix shapes
- Load balancing issues
- Specific workload patterns

```python
# Enable Stream-K for better load balancing
C = tritonblas.matmul(A, B, enable_streamk=True)
```

**Trade-offs**:
- Better load balancing
- Slightly higher overhead
- Best for specific workloads

### 4. Memory Layout

**Contiguous Tensors**:
Ensure tensors are contiguous for optimal performance

```python
# Check and make contiguous
if not A.is_contiguous():
    A = A.contiguous()
if not B.is_contiguous():
    B = B.contiguous()

C = tritonblas.matmul(A, B)
```

## Performance Profiling

### Basic Timing

```python
import torch
import tritonblas
import time

def benchmark(m, n, k, dtype, iterations=100):
    A = torch.randn(m, k, dtype=dtype, device='cuda')
    B = torch.randn(k, n, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(10):
        C = tritonblas.matmul(A, B)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        C = tritonblas.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    tflops = (2 * m * n * k) / (avg_time * 1e12)
    
    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")

benchmark(4096, 4096, 4096, torch.float16)
```

### Using PyTorch Profiler

```python
import torch
import tritonblas
from torch.profiler import profile, ProfilerActivity

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    C = tritonblas.matmul(A, B)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

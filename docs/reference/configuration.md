# Configuration

This guide covers configuration options and the analytical model's decision-making process in tritonBLAS.

## MatmulHeuristicResult

The `MatmulHeuristicResult` class encapsulates the optimal kernel configuration for a given matrix multiplication operation.

### Creating a Configuration

```python
import torch
import tritonblas

selector = tritonblas.MatmulHeuristicResult(
    m=4096,           # Rows of left matrix
    n=4096,           # Columns of right matrix
    k=4096,           # Shared dimension
    a_dtype=torch.float16,  # Left matrix data type
    b_dtype=torch.float16,  # Right matrix data type
    c_dtype=torch.float16   # Output matrix data type
)
```

### Configuration Parameters

The analytical model determines optimal values for:

#### Block Sizes
- **BLOCK_M**: Tile size in M dimension
- **BLOCK_N**: Tile size in N dimension
- **BLOCK_K**: Tile size in K dimension

These determine how the matrix is divided into tiles for processing.

#### Thread Configuration
- **num_warps**: Number of warps per thread block
- **num_stages**: Pipeline stages for computation/memory overlap

#### Memory Layout
- Data arrangement in shared memory
- Access patterns for optimal cache utilization

## How Configuration is Selected

### 1. Problem Analysis

The analytical model analyzes:

```python
# Matrix dimensions
m, n, k = 4096, 4096, 4096

# Data types
a_dtype = torch.float16  # 2 bytes per element
b_dtype = torch.float16  # 2 bytes per element
c_dtype = torch.float16  # 2 bytes per element

# Compute requirements
total_ops = 2 * m * n * k  # FLOPs
memory_a = m * k * 2       # bytes
memory_b = k * n * 2       # bytes
memory_c = m * n * 2       # bytes
```

### 2. Hardware Characteristics

Considers GPU properties:
- Compute units and FLOP/s capacity
- Memory bandwidth
- Cache sizes (L1, L2)
- Shared memory per SM

### 3. Optimal Configuration Selection

The model selects configuration to:
- Maximize compute utilization
- Minimize memory bottlenecks
- Optimize cache hit rates
- Balance thread occupancy

## Configuration Reuse

### When to Reuse

Reuse configurations when:
- Matrix dimensions are identical
- Data types are the same
- Running multiple iterations

```python
# Create once
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

# Reuse many times
for i in range(1000):
    A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    C = tritonblas.matmul_lt(A, B, selector=selector)
```

### When to Create New

Create new configurations when:
- Matrix dimensions change
- Data types change
- Switching between different operations

```python
# Different sizes need different configurations
selector_small = tritonblas.MatmulHeuristicResult(
    1024, 1024, 1024,
    torch.float16, torch.float16, torch.float16
)

selector_large = tritonblas.MatmulHeuristicResult(
    8192, 8192, 8192,
    torch.float16, torch.float16, torch.float16
)
```

## Configuration for Different Scenarios

### Small Matrices

```python
# For small matrices (< 1024)
selector = tritonblas.MatmulHeuristicResult(
    512, 512, 512,
    torch.float16, torch.float16, torch.float16
)
```

The model typically selects:
- Smaller block sizes
- Fewer pipeline stages
- Configuration optimized for low latency

### Large Matrices

```python
# For large matrices (> 4096)
selector = tritonblas.MatmulHeuristicResult(
    8192, 8192, 8192,
    torch.float16, torch.float16, torch.float16
)
```

The model typically selects:
- Larger block sizes
- More pipeline stages
- Configuration optimized for throughput

### Rectangular Matrices

```python
# For rectangular matrices
selector = tritonblas.MatmulHeuristicResult(
    2048, 8192, 4096,  # M != N != K
    torch.float16, torch.float16, torch.float16
)
```

The model adapts block sizes to matrix shape.

### Mixed Precision

```python
# Different input/output types
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    a_dtype=torch.float16,   # Input A
    b_dtype=torch.float16,   # Input B
    c_dtype=torch.float32    # Output C (higher precision)
)
```

## Configuration Best Practices

### 1. Cache Configurations

```python
# Cache configurations for common sizes
configs = {}

def get_or_create_config(m, n, k, dtype):
    key = (m, n, k, dtype)
    if key not in configs:
        configs[key] = tritonblas.MatmulHeuristicResult(
            m, n, k, dtype, dtype, dtype
        )
    return configs[key]

# Use cached configuration
selector = get_or_create_config(4096, 4096, 4096, torch.float16)
```

### 2. Batch Processing

```python
# Create configuration once for batch
batch_size = 16
m, n, k = 1024, 1024, 1024

selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    torch.float16, torch.float16, torch.float16
)

# Process entire batch
A = torch.randn(batch_size, m, k, dtype=torch.float16, device='cuda')
B = torch.randn(batch_size, k, n, dtype=torch.float16, device='cuda')
C = tritonblas.matmul_lt(A, B, selector=selector)
```

### 3. Dynamic Shapes

```python
# For varying shapes, create configurations as needed
def matmul_dynamic(A, B):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Dimension mismatch"
    
    # Create configuration for this shape
    selector = tritonblas.MatmulHeuristicResult(
        m, n, k,
        A.dtype, B.dtype, A.dtype
    )
    
    return tritonblas.matmul_lt(A, B, selector=selector)
```

## Understanding Configuration Decisions

### Inspecting Configuration

While configuration internals are not directly exposed, you can understand decisions through:

```python
import torch
import tritonblas

# Create configuration
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

# The configuration is optimized for:
# - Matrix size: 4096x4096x4096
# - Data type: FP16 (2 bytes per element)
# - Hardware: Current GPU architecture
```

### Configuration Determinism

The analytical model is deterministic:

```python
# Same inputs always produce same configuration
config1 = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

config2 = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

# config1 and config2 will select identical kernel parameters
```

## Advanced Configuration Topics

### Configuration Overhead

Configuration creation is fast (<1ms) but should still be minimized:

```python
import time

start = time.time()
selector = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)
elapsed = time.time() - start

print(f"Configuration time: {elapsed*1000:.2f} ms")  # Typically < 1ms
```

### Configuration Lifetime

Configurations can be reused indefinitely:

```python
# Create once at initialization
GLOBAL_CONFIG = tritonblas.MatmulHeuristicResult(
    4096, 4096, 4096,
    torch.float16, torch.float16, torch.float16
)

# Use throughout application lifetime
def process_batch(A, B):
    return tritonblas.matmul_lt(A, B, selector=GLOBAL_CONFIG)
```

## See Also

- [API Reference](api.md): Complete API documentation
- [Advanced Features](advanced.md): Stream-K and optimizations
- [Analytical Model](../conceptual/analytical-model.md): How configurations are predicted

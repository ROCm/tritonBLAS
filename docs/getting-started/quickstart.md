# Quick Start Guide

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

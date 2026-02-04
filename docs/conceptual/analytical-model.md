# Analytical Model

tritonBLAS uses an analytical model called **Origami** to predict optimal kernel configurations without autotuning.

## Overview

Traditional Triton kernels rely on `@triton.autotune` which performs a greedy search through configuration spaces. This approach:
- Requires expensive runtime evaluation of many configurations
- Produces numerous JIT-compiled kernel variants
- Lacks predictability and explainability

**tritonBLAS eliminates autotuning** by using Origami, an analytical model that predicts the optimal configuration based on:
- Matrix dimensions (M, N, K)
- Data types
- Hardware characteristics
- Memory hierarchy

## Origami

Origami is AMD's analytical performance model for GEMM kernels. Instead of searching through configuration spaces via autotuning, Origami analytically determines the optimal kernel configuration.


### Learn More

For detailed information about Origami:
- **Source Code**: [ROCm/rocm-libraries/shared/origami](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami)
- **Implementation**: See `include/tritonblas/origami.py` in the tritonBLAS repository

## How Origami Works in tritonBLAS

### Configuration Selection

Given matrix dimensions and data types, Origami:

1. Analyzes the problem characteristics (M, N, K, data types)
2. Evaluates configuration options analytically based on hardware model
3. Selects the configuration with predicted best performance
4. Returns a `MatmulHeuristicResult` object with optimal parameters

## Example Usage

```python
import torch
import tritonblas

# The analytical model works behind the scenes
m, n, k = 4096, 4096, 4096
selector = tritonblas.MatmulHeuristicResult(
    m, n, k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    c_dtype=torch.float16
)

# The selector contains the optimal configuration
# predicted by the analytical model
A = torch.randn(m, k, dtype=torch.float16, device='cuda')
B = torch.randn(k, n, dtype=torch.float16, device='cuda')
C = tritonblas.matmul_lt(A, B, selector=selector)
```

## Model Accuracy

The analytical model has been validated against:
- Autotuned configurations
- Vendor BLAS libraries (hipBLAS, rocBLAS)
- Empirical performance measurements

Results show that the model consistently selects near-optimal configurations while eliminating autotuning overhead.

## Learn More

- [Architecture](architecture.md): Library design and components
- [API Reference](../reference/api-autodoc.rst): API documentation

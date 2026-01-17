---
myst:
  html_meta:
    "description": "tritonBLAS: A Lightweight Triton-based General Matrix Multiplication (GEMM) Library"
    "keywords": "tritonBLAS, AMD, GPU, GEMM, Matrix Multiplication, Triton, BLAS, MI300X"
---

# tritonBLAS

<div style="text-align: center; margin: 2rem 0;">
  <p style="font-size: 1.5rem; font-weight: bold; color: #333;">A Lightweight Triton-based General Matrix Multiplication (GEMM) Library</p>
  <p style="font-size: 1.1rem; color: #666; margin: 1rem 0;">Analytical Model-Driven Performance Without Autotuning</p>
</div>

> **Important**: This project is intended for research purposes only. Use it at your own risk and discretion.

## What is tritonBLAS?

tritonBLAS is a high-performance GEMM library built on Triton that **eliminates the need for autotuning and heuristics**. Instead of relying on greedy search through configuration spaces, tritonBLAS uses an **analytical model** to predict the optimal configuration for matrix multiplication operations.

### Key Features

- **No Autotuning Required**: Analytical model predicts optimal configurations without expensive search
- **Smaller Footprint**: Only JIT-compiles kernels precisely needed for given matrix shapes
- **Predictable & Deterministic**: Model-driven decisions are explainable and reproducible
- **Scalable Engineering**: Easier to maintain and extend without complex heuristics
- **Peak Performance**: Achieves peak performance without greedy search overhead
- **Familiar APIs**: Drop-in replacement for `torch.matmul` and BLAS-like interfaces

## Quick Start

### Installation

tritonBLAS requires hipBLASLt dependencies which are automatically fetched. Set up using Docker:

```bash
# Clone the repository
git clone https://github.com/ROCm/tritonBLAS.git
cd tritonBLAS

# Start the development container
docker compose up --build -d
docker attach tritonBLAS-dev

# Install tritonBLAS
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

### Run Your First Example

```bash
cd examples
python3 example_matmul.py
```

## API Overview

tritonBLAS provides two API styles to suit different use cases:

### 1. Peak Performance API (Recommended)

Inspired by `hipBLASLt` and `cuBLASLt`, this API separates configuration selection from execution for optimal performance:

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

# Step 3: Perform matrix multiplication with optimal config
C = tritonblas.matmul_lt(A, B, selector=selector)
```

**Key Functions:**

- `tritonblas.MatmulHeuristicResult(m, n, k, a_dtype, b_dtype, c_dtype)` - Returns optimal configuration
- `tritonblas.matmul_lt(input, other, *, out=None, selector, enable_streamk=False)` - Executes GEMM with config

### 2. Drop-in Replacement API

Familiar PyTorch-style API for seamless integration:

```python
import torch
import tritonblas

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

# Direct replacement for torch.matmul
C = tritonblas.matmul(A, B)

# With Stream-K algorithm
C = tritonblas.matmul(A, B, enable_streamk=True)
```

## Support Matrix

### GEMM on AMD MI300X

| Transpose (A/B) | TF32 | FP32 | FP16 | BF16 | FP8 | FP4 |
|-----------------|------|------|------|------|-----|-----|
| T/N             | ✓    | ✓    | ✓    | ✓    | ✓   | ✗   |
| N/T             | ✓    | ✓    | ✓    | ✓    | ✓   | ✗   |
| T/T             | ✓    | ✓    | ✓    | ✓    | ✓   | ✗   |
| N/N             | ✓    | ✓    | ✓    | ✓    | ✓   | ✗   |

### Supported GPUs

tritonBLAS supports the following AMD GPUs:

| GPU Model | Support Status |
|-----------|----------------|
| MI300X | ✅ Supported |
| MI300A | ✅ Supported |
| MI308X | ✅ Supported |
| MI350X | ✅ Supported |
| MI355X | ✅ Supported |

> **Note**: tritonBLAS is optimized for AMD Instinct MI300 and MI350 series GPUs.

## Documentation Structure

### 📚 **Getting Started**
- **[Installation](getting-started/installation.md)**: Detailed setup instructions
- **[Quick Start Guide](getting-started/quickstart.md)**: Get running in minutes
- **[Examples](getting-started/examples.md)**: Working code examples

### 🧠 **Conceptual**
- **[Analytical Model](conceptual/analytical-model.md)**: How tritonBLAS predicts optimal configurations
- **[Architecture](conceptual/architecture.md)**: Library design and components
- **[Performance](conceptual/performance.md)**: Benchmarks and optimization techniques

### 📖 **API Reference**
- **[Core API](reference/api.md)**: Complete API documentation
- **[Configuration](reference/configuration.md)**: MatmulHeuristicResult and tuning options
- **[Advanced Features](reference/advanced.md)**: Stream-K and other optimizations

### 🔧 **Developer Guide**
- **[Contributing](CONTRIBUTING.md)**: How to contribute to tritonBLAS
- **[Building from Source](developer/building.md)**: Development setup
- **[Testing](developer/testing.md)**: Running tests and benchmarks

## Why tritonBLAS?

Traditional Triton kernels rely on `@triton.autotune` which:
- Requires expensive greedy search through configuration spaces
- Produces many JIT-compiled kernel variants
- Uses complex heuristics that are hard to maintain
- Lacks predictability and explainability

**tritonBLAS solves these problems** by using an analytical model that:
- Predicts optimal configurations without search
- Compiles only necessary kernels for given shapes
- Provides deterministic, explainable decisions
- Simplifies maintenance and extension

## Performance

tritonBLAS achieves competitive performance with vendor libraries while offering:
- **Faster compilation**: No autotuning overhead
- **Smaller binary size**: Fewer kernel variants
- **Predictable behavior**: Model-driven configuration selection
- **Easy debugging**: Explainable configuration choices

See our [Performance Guide](conceptual/performance.md) for detailed benchmarks.

## Community & Support

### GitHub Issues
Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/tritonBLAS/issues/new/choose) on GitHub.

### Contributing
Want to contribute? Check out our [Contributing Guide](CONTRIBUTING.md) to learn how you can help improve tritonBLAS.

### Contact
Need direct support? Contact our development team through GitHub or the ROCm community channels.

## Contributors

See our [Contributors List](CONTRIBUTORS.md) for the full list of developers and contributors.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE.md) file for details.

---

**Ready to get started? Begin with the [Installation Guide](getting-started/installation.md)!**

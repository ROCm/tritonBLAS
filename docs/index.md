---
myst:
  html_meta:
    "description": "tritonBLAS: A Lightweight Triton-based General Matrix Multiplication (GEMM) Library"
    "keywords": "tritonBLAS, AMD, GPU, GEMM, Matrix Multiplication, Triton, BLAS, MI300X"
---

# tritonBLAS

A lightweight Triton-based GEMM library that uses an analytical model to predict optimal configurations without autotuning.

> **Important**: This project is intended for research purposes only.

## Quick Start

```bash
git clone https://github.com/ROCm/tritonBLAS.git
cd tritonBLAS
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

```python
import torch
import tritonblas

A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = tritonblas.matmul(A, B)
```

## Supported GPUs

| GPU Model | Support Status |
|-----------|----------------|
| MI300X | ✅ Supported |
| MI300A | ✅ Supported |
| MI308X | ✅ Supported |
| MI350X | ✅ Supported |
| MI355X | ✅ Supported |

## License

This project is licensed under the MIT License.

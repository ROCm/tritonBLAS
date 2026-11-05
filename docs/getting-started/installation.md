# Installation

This guide will help you set up tritonBLAS on your system.

## Prerequisites

Before installing tritonBLAS, ensure you have:

- **Python 3.10+**
- **PyTorch 2.0+** (ROCm version)
- **ROCm 6.3.1+** HIP runtime
- **Triton** (compatible version)
- **ROCm development files** (HIP headers and CMake package, e.g. a full ROCm install under `/opt/rocm`) and a **C++17 compiler**, to build the [rocm-origami](https://pypi.org/project/rocm-origami/) dependency, which is published as source only
- **Network access to PyPI and GitHub** during installation (the rocm-origami build downloads nanobind from GitHub)

## Supported Hardware

tritonBLAS supports the following AMD GPUs:

| GPU Model | Support Status |
|-----------|----------------|
| MI300X | Supported |
| MI300A | Supported |
| MI308X | Supported |
| MI350X | Supported |
| MI355X | Supported |

> **Note**: tritonBLAS is optimized for AMD Instinct MI300 and MI350 series GPUs.

## Installation

Install tritonBLAS from source:

```bash
# Clone the repository
git clone https://github.com/ROCm/tritonBLAS.git
cd tritonBLAS

# Install tritonBLAS in editable mode
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

> **Note**: `pip3 install -e .` also builds `rocm-origami` at the version pinned in `pyproject.toml`.
> If its CMake step can't find HIP, point it at your ROCm install, e.g.
> `CMAKE_PREFIX_PATH=/opt/rocm pip3 install -e .`. For ROCm installed as Python packages
> (`rocm[devel]`), run `rocm-sdk init` once and use `CMAKE_PREFIX_PATH="$(rocm-sdk path --root)"`.

## Verifying Installation

After installation, verify that tritonBLAS is working correctly:

```python
import torch
import tritonblas

# Create test matrices
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Perform matrix multiplication
C = tritonblas.matmul(A, B)

print("tritonBLAS is working correctly!")
print(f"Result shape: {C.shape}")
```

## Next Steps

Now that you have tritonBLAS installed, you can:

1. Follow the [Quick Start Guide](quickstart.md) to run your first example
2. Explore [Examples](examples.rst) for common use cases
3. Read the [API Reference](../reference/api-autodoc.rst) for detailed documentation

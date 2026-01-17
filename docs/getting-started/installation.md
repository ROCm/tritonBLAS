# Installation

This guide will help you set up tritonBLAS on your system.

## Prerequisites

Before installing tritonBLAS, ensure you have:

- **Python 3.10+**
- **PyTorch 2.0+** (ROCm version)
- **ROCm 6.3.1+** HIP runtime
- **Triton** (compatible version)
- **Docker** (recommended for development)

## Supported Hardware

tritonBLAS supports the following AMD GPUs:

| GPU Model | Support Status |
|-----------|----------------|
| MI300X | ✅ Supported |
| MI300A | ✅ Supported |
| MI308X | ✅ Supported |
| MI350X | ✅ Supported |
| MI355X | ✅ Supported |

> **Note**: tritonBLAS is optimized for AMD Instinct MI300 and MI350 series GPUs.

## Installation Methods

### Method 1: From Source

Install tritonBLAS from source:

```bash
# Clone the repository
git clone https://github.com/ROCm/tritonBLAS.git
cd tritonBLAS

# Install tritonBLAS in editable mode
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

### Method 2: Docker Compose

Use Docker for a containerized development environment:

```bash
# Clone the repository
git clone https://github.com/ROCm/tritonBLAS.git
cd tritonBLAS

# Start the development container
docker compose up --build -d

# Attach to the running container
docker attach tritonBLAS-dev

# Install tritonBLAS in development mode
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

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

print("✅ tritonBLAS is working correctly!")
print(f"Result shape: {C.shape}")
```

## Dependencies

tritonBLAS automatically fetches required dependencies from hipBLASLt. The main dependencies are:

- **PyTorch**: For tensor operations and CUDA/ROCm integration
- **Triton**: For kernel compilation and execution
- **hipBLASLt**: For certain C++ utilities (automatically fetched)

## Troubleshooting

### Common Issues

**Issue: Import Error**
```
ImportError: No module named 'tritonblas'
```
**Solution**: Ensure you've set the PYTHONPATH correctly:
```bash
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

**Issue: CUDA/ROCm Not Found**
```
RuntimeError: No CUDA GPUs are available
```
**Solution**: Verify ROCm installation and GPU visibility:
```bash
rocm-smi
```

**Issue: Triton Compilation Error**
```
TritonError: Compilation failed
```
**Solution**: Ensure you have a compatible Triton version installed. Try reinstalling:
```bash
pip3 install --upgrade triton
```

## Next Steps

Now that you have tritonBLAS installed, you can:

1. Follow the [Quick Start Guide](quickstart.md) to run your first example
2. Explore [Examples](examples.md) for common use cases
3. Read the [API Reference](../reference/api.md) for detailed documentation

## Getting Help

If you encounter issues during installation:

1. Check the [GitHub Issues](https://github.com/ROCm/tritonBLAS/issues) for similar problems
2. Open a new issue with details about your environment and error messages
3. Contact the development team through GitHub

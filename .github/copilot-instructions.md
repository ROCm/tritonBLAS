# tritonBLAS Development Instructions

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

tritonBLAS is a lightweight Triton-based BLAS library for general matrix multiplication (GEMM) that uses analytical models instead of autotuning. It requires ROCm/CUDA GPU environments and has complex dependency requirements.

## Working Effectively

### Prerequisites and Environment Setup

**CRITICAL: tritonBLAS requires a ROCm or CUDA GPU environment. CPU-only environments will fail during installation when trying to install triton.**

#### Docker Setup (RECOMMENDED)
```bash
docker compose up --build -d
docker attach tritonBLAS-dev
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

**TIMING: Docker build takes 7-10 minutes. NEVER CANCEL. Set timeout to 15+ minutes.**

#### Local Development (Advanced Users Only)
**WARNING: Local installation often fails due to triton/CUDA dependencies. Use Docker unless you have a specific need.**

```bash
# Only works in ROCm/CUDA environments
pip install -e ".[dev]"
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

**TIMING: pip install takes 2-5 minutes depending on network. NEVER CANCEL. Set timeout to 10+ minutes.**

### Build and Installation Process

The setup.py performs these steps automatically:
1. **Dependency Fetching** (~4 seconds total):
   - Clones ROCm libraries from GitHub (~2.5 seconds)
   - Sparse checkout of specific paths (~1 second)
   - Builds origami utility from hipBLASLt

**NEVER CANCEL the build process. The dependency fetching is critical and must complete.**

### Code Quality and Linting

**ALWAYS run these before committing changes:**

```bash
# Fix linting issues (takes <1 second)
ruff check .
ruff format .
```

**TIMING: Both ruff commands complete in <1 second each.**

### Testing

**CRITICAL: Tests require CUDA GPU and cannot run in CPU-only environments.**

```bash
# Run tests (requires CUDA GPU)
pytest
```

**TIMING: Test collection takes ~2 seconds. Full test execution timing varies by GPU.**

**WARNING: Tests will fail with "ModuleNotFoundError: No module named 'triton'" if run without CUDA/ROCm environment.**

### Running Examples

```bash
# Basic matrix multiplication example
cd examples
python3 example_matmul.py --m 8192 --n 8192 --k 8192

# Performance-focused example
python3 example_matmul_lt.py
```

**TIMING: Examples require GPU and triton installation to run.**

## Validation and Development Workflow

### After Making Changes

1. **ALWAYS run linting first:**
   ```bash
   ruff check .
   ruff format .
   ```

2. **Test your changes in Docker environment:**
   ```bash
   docker compose up --build -d  # Starts build in background (7-10 minutes). Command returns immediately due to -d (detached mode); wait for build to finish before proceeding. NEVER CANCEL.
   docker attach tritonBLAS-dev
   pip3 install -e .  # 2-5 minutes in container
   ```

3. **Run examples to validate functionality:**
   ```bash
   cd examples
   python3 example_matmul.py
   ```

4. **Run tests if you have GPU access:**
   ```bash
   pytest  # Requires CUDA GPU
   ```

### Common Build Issues and Solutions

1. **"No space left on device"**: Clean up Docker images and system cache
2. **"ModuleNotFoundError: No module named 'triton'"**: Must use ROCm/CUDA environment
3. **Timeout during Docker build**: Increase timeout to 15+ minutes, NEVER CANCEL
4. **pip install network timeouts**: Retry with longer timeout (10+ minutes)

## Project Structure and Navigation

### Key Directories

- **`include/tritonblas/`**: Main package source code
  - `__init__.py`: Package entry point
  - `matmul.py`: Core matrix multiplication functions
  - `origami.py`: Heuristic selection logic
  - `internal/`: Internal implementation details

- **`examples/`**: Working examples demonstrating usage
  - `example_matmul.py`: Basic usage example
  - `example_matmul_lt.py`: Performance-focused API example

- **`tests/`**: Test suite (requires GPU)
  - `test_matmul.py`: Basic matmul tests
  - `test_matmul_lt.py`: Performance API tests

- **`benchmarks/`**: Performance benchmarking tools
  - `benchmark_autotuning.py`: Autotuning overhead comparison
  - `heuristic_benchmark.py`: Heuristic selection timing

### Important Files

- **`setup.py`**: Custom build script with ROCm dependency fetching
- **`pyproject.toml`**: Project configuration and dependencies
- **`docker-compose.yml`**: Docker environment setup
- **`Dockerfile`**: ROCm/PyTorch base image setup

## API Usage Patterns

### Peak Performance API (Recommended)
```python
import tritonblas

# Create heuristic selector (one-time setup)
selector = tritonblas.MatmulHeuristicResult(m, n, k, a_dtype, b_dtype, c_dtype)

# Use for actual computation
result = tritonblas.matmul_lt(A, B, selector=selector, enable_streamk=False)
```

### Drop-in Replacement API
```python
import tritonblas

# Direct usage (performs heuristic selection internally)
result = tritonblas.matmul(A, B, enable_streamk=False)
```

## Development Notes

### Code Style Requirements
- **Line length**: 120 characters (configured in pyproject.toml)
- **Auto-formatting**: Use `ruff format` before committing
- **Linting**: Fix all `ruff check` issues before committing

### Testing Requirements
- All tests require CUDA GPU environment
- Tests use parametrized inputs for different data types and matrix sizes
- Tests include stream-k algorithm validation

### Performance Considerations
- The library eliminates autotuning overhead through analytical models
- Heuristic selection is cached for previously seen problem sizes
- Docker environment includes optimized ROCm/PyTorch stack

## Troubleshooting

### "Cannot import triton"
- **Cause**: CPU-only environment or missing CUDA
- **Solution**: Use Docker with ROCm/CUDA support

### "No space left on device"
- **Cause**: Docker build exhausts disk space
- **Solution**: `docker system prune -af && sudo apt-get clean`

### Docker build timeout
- **Cause**: Large base image download (ROCm/PyTorch)
- **Solution**: Increase timeout to 15+ minutes, be patient

### pip install fails with network errors
- **Cause**: PyPI timeouts or network issues
- **Solution**: Retry with increased timeout, check network connectivity

**Remember: This is a research project intended for ROCm/CUDA environments. Always validate your changes in the proper GPU environment before submitting.**
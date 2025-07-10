# tritonBLAS: A Lightweight Triton-based General Matrix Multiplication (GEMM) Library

> [!IMPORTANT]  
> This project is intended for research purposes only. Use it at your own risk and discretion.

Triton is a language and compiler for writing highly efficient ML primitives, one of the most common primitive is matrix-multiplication. Triton typically builds these primitives using just-in-time (JIT) compilation, and relies on functionality such as [`@triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html) to create efficient variants of the primitives. Autotune evaluates all the possible configurations defined by the user to produce a kernel perfect for a given inputs.

**Our work, tritonBLAS, removes the need for autotune and heuristics, and instead uses an analytical model to predict the correct configuration for common algorithms such as Matrix Multiplication. We believe this technique is also extensible to other dense, static, well-defined primitives in the Deep-learning applications.**

Because there is now no need for autotuning or heuristcis, we now produce a library that is;

1. **Smaller**: Number of kernels that are JIT'ed are few and precisely whats needed for the m,n,k shapes,
2. **Predictable and Deterministic**: No need for complex heuristics, we can use the model to explain all the decisions it took to pick a given configuration for a problem shape/size,
3. **Scalable Software Engineering**: Managing and upkeeping the code becomes easier, and
4. **Peak Performance**: Achives peak performance without the need for a greedy-search.

## Getting Started

tritonBLAS currently requires a dependency on a few C++ files from hipBLASLt, which it will automatically fetch. Run the following to setup a docker container with `rocm/pytorch:latest-release` and a fresh `triton` install:

```bash
docker compose up --build -d
docker attach tritonBLAS-dev
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

Run a simple example:

```bash
cd examples
python3 example_matmul.py
```

## API

### Peak Performance API

Borrows from performant variants of BLAS interfaces such as `hipBLASLt` and `cuBLASLt`, where the user initiates an initial call to set up some arguments and learn from the matrix descriptors before calling the actual `matmul`.

```python
tritonblas.MatmulHeuristicResult(m, n, k) → MatmulHeuristicResult
```

**Parameters:**

- **m** (*int*): Number of rows of the left-hand matrix.
- **n** (*int*): Number of columns of the right-hand matrix.
- **k** (*int*): Shared dimension between the two matrices (columns of the left-hand matrix and rows of the right-hand matrix).

**Returns:**

- `MatmulHeuristicResult`: An object containing a precomputed kernel configuration optimized for the provided matrix dimensions.

```python
tritonblas.matmul_lt(input,other,*,out=None,selector) → Tensor
```

#### Parameters

- **input** (*Tensor*) – the first tensor to be multiplied
- **other** (*Tensor*) – the second tensor to be multiplied

#### Keyword Arguments

- **out** (*Tensor*, optional) – the output tensor.
- **selector** (*MatmulHeuristicResult*): Configuration object returned by `MatmulHeuristicResult`, providing optimal tiling and launch parameters.

### Drop-in Replacement for `torch.matmul` (work-in-progress)

Borrows from familiar pytorch API (`torch.matmul`) making integration within larger models and applications seamless.

```python
tritonblas.matmul(input,other,*,out=None) → Tensor
```

**Parameters**

- **input** (*Tensor*) – the first tensor to be multiplied
- **other** (*Tensor*) – the second tensor to be multiplied

**Keyword Arguments**

- **out** (*Tensor*, optional) – the output tensor.

## Support Matrix

As we work on supporting other BLAS and ML primitives and data types, we will update this document to reflect that.

### GEMM, Platform ![AMD_HIP](https://img.shields.io/badge/MI300X-%23000000.svg?style=for-the-badge&logo=amd&logoColor=white&logoSize=auto)

| Transpose (A/B) | TF32 | FP32               | FP16               | BF16               | FP8                | FP4 |
|------------|------|--------------------|--------------------|--------------------|--------------------|-----|
| T/N        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |
| N/T        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |
| T/T        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |
| N/N        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |

## Contributors

The official list of developers and contributors is available here: [CONTRIBUTORS](docs/CONTRIBUTORS.md). We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.

## Support

Need help? We're here to support you! Here are a few ways to get in touch:

1. **Open an Issue**: Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/tritonBLAS/issues/new/choose) on GitHub,
2. **Contact the Team**: If GitHub issues aren't working for you or you need to reach us directly, feel free to contact our development team.

We welcome your feedback and contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

# tritonBLAS: A Lightweight Triton-based General Matrix Multiplication (GEMM) Library

> [!IMPORTANT]
> This project is intended for research purposes only. Use it at your own risk and discretion.

## Latest
- **[30/03/2026]:** Work-Stealing Persistent Kernel Released — enables linear performance scaling with active CU count, avoiding nonlinear decay when partitioning CUs across concurrent kernels
- **[05/12/2025]:** FP4 Support Released — native FP4 matrix multiplication for low-precision matmul
- **[22/07/2025]:** Stream-K GEMM Released — fine-grained tile-based work partitioning for improved occupancy on small/skinny GEMMs
- **[08/07/2025]:** tritonBLAS Released — analytical model-driven GEMM library eliminating the need for autotuning

## About

Triton is a language and compiler for writing highly efficient ML primitives, one of the most common primitive is matrix-multiplication. Triton typically builds these primitives using just-in-time (JIT) compilation, and relies on functionality such as [`@triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html) to create efficient variants of the primitives. Autotune evaluates all the possible configurations defined by the user to produce a kernel perfect for a given inputs.

**Our work, tritonBLAS, removes the need for autotune and heuristics, and instead uses an analytical model to predict the correct configuration for common algorithms such as Matrix Multiplication. We believe this technique is also extensible to other dense, static, well-defined primitives in the Deep-learning applications.**

Because there is now no need for autotuning or heuristcis, we now produce a library that is;

1. **Smaller**: Number of kernels that are JIT'ed are few and precisely whats needed for the m,n,k shapes,
2. **Predictable and Deterministic**: No need for complex heuristics, we can use the model to explain all the decisions it took to pick a given configuration for a problem shape/size,
3. **Scalable Software Engineering**: Managing and upkeeping the code becomes easier, and
4. **Peak Performance**: Achives peak performance without the need for a greedy-search.

## Getting Started

tritonBLAS depends on [rocm-origami](https://pypi.org/project/rocm-origami/),
AMD's analytical GEMM model, which `pip` installs automatically. rocm-origami
is published as source only, so installing it compiles a small C++ extension
against your ROCm install; see the [installation
guide](docs/getting-started/installation.md) for the build requirements. Run
the following to setup a docker container with `rocm/pytorch:latest-release`
and a fresh `triton` install:

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

Run a GEMM that asynchronously wakes a waiting Triton kernel when its first output tile is ready:

```bash
pip install -e ".[examples]"
python3 examples/example_async_tile_trigger.py
python3 examples/example_async_tile_trigger.py --block-m 128 --block-n 128 --block-k 64
python3 examples/example_async_tile_trigger.py --num-observers 8
python3 examples/example_async_tile_trigger.py --signal-tiles-m 3 --signal-tiles-n 5
python3 examples/example_async_tile_trigger.py --signal-layout dispatch --signal-tiles-m 3 --signal-tiles-n 5
python3 examples/example_async_tile_trigger.py --signal-layout random --signal-tiles-m 3 --signal-tiles-n 5 --signal-seed 17
```

The example checks the complete GEMM against a float32 reference, verifies every value
read by the asynchronous consumer, logs the GPU timeline, writes one CSV row per tile,
and saves `async_tile_trigger_heatmap.png`, comparing measured gate-trigger order with
the order published by `tritonblas.schedule`. It prints Origami's selected tile, the
tile actually used, and the resulting output-tile grid; any omitted block dimension
keeps Origami's selection. The plot also separates producer-side gate-release order
from consumer-observed order; `--num-observers` controls the bounded parallel poller
pool without risking one waiting workgroup per tile. Signal layouts may be rectangular
blocks (including ragged edge groups), row-major chunks, chunks of the published dispatch
order, modulo-scattered groups, or seeded random groups. Non-block layouts use
`--signal-tiles-m × --signal-tiles-n` as their target producer count. The signal-level CSV
compares the predicted readiness order—each signal's final producer in the published
schedule—with its measured opening order.

## API

### Tile Schedule API

`tritonblas.schedule(a, b)` returns the complete static producer plan needed by a
triggered consumer:

```python
plan = tritonblas.schedule(a, b)
plan = plan.with_signal_layout(signal_of_tile)

plan.tile_of_wg          # dispatch position -> linear output tile
plan.tile_order          # (tile_m, tile_n) coordinates in dispatch order
plan.signal_plan.signal_of_tile # output tile -> signal read by the GEMM epilogue
plan.signal_plan.arms    # grouped SignalArm(signal_id, tiles, producers) entries
plan.grid_m              # output-tile grid height
plan.grid_n              # output-tile grid width
```

The default is one signal per output tile, so the returned plan is immediately usable.
`with_signal_layout()` accepts any dense tile-to-signal table and derives both thresholds
and firing order. `arming_order` and `signal_arming_order` remain convenience aliases for
`tile_order` and `signal_plan.arms`. This is the API boundary between tritonBLAS and a triggered
consumer: the consumer should use the plan rather than reconstructing the GEMM's permutation.
Dynamic work-stealing and Stream-K calls return `None` because they have no static tile ownership
to publish.

### Peak Performance API

Borrows from performant variants of BLAS interfaces such as `hipBLASLt` and `cuBLASLt`, where the user initiates an initial call to set up some arguments and learn from the matrix descriptors before calling the actual `matmul`.

```python
tritonblas.OrigamiMatmulSelector(m, n, k, a_dtype, b_dtype, c_dtype, device) → OrigamiMatmulSelector
```

**Parameters:**

- **m** (*int*): Number of rows of the left-hand matrix.
- **n** (*int*): Number of columns of the right-hand matrix.
- **k** (*int*): Shared dimension between the two matrices (columns of the left-hand matrix and rows of the right-hand matrix).
- **a_dtype** (*torch.dtype*): Data type of left-hand matrix.
- **b_dtype** (*torch.dtype*): Data type of right-hand matrix.
- **c_dtype** (*torch.dtype*): Data type of output matrix.
- **device** (*torch.device*): Torch device object for the GPU which the tensors reside on.

**Returns:**

- `OrigamiMatmulSelector`: An object containing a precomputed kernel configuration optimized for the provided matrix dimensions.

```python
tritonblas.matmul_lt(input,other,*,out=None,selector,enable_streamk=False) → Tensor
```

#### Parameters

- **input** (*Tensor*) – the first tensor to be multiplied
- **other** (*Tensor*) – the second tensor to be multiplied

#### Keyword Arguments

- **out** (*Tensor*, optional) – the output tensor.
- **selector** (*OrigamiMatmulSelector*): Configuration object returned by `OrigamiMatmulSelector`, providing optimal tiling and launch parameters.
- **enable_streamk** (*bool*, optional) – enable [Stream-K](https://arxiv.org/abs/2301.03598) GEMM algorithm. Default: `False`.

### Drop-in Replacement for `torch.matmul` (work-in-progress)

Borrows from familiar pytorch API (`torch.matmul`) making integration within larger models and applications seamless.

```python
tritonblas.matmul(input,other,*,out=None,enable_streamk=False) → Tensor
```

**Parameters**

- **input** (*Tensor*) – the first tensor to be multiplied
- **other** (*Tensor*) – the second tensor to be multiplied

**Keyword Arguments**

- **out** (*Tensor*, optional) – the output tensor.
- **enable_streamk** (*bool*, optional) – enable [Stream-K](https://arxiv.org/abs/2301.03598) GEMM algorithm. Default: `False`.

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

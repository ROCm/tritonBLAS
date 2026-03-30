# Architecture

tritonBLAS delivers high-performance matrix multiplication through an elegant, layered architecture that balances simplicity with power.

## At a Glance

```{admonition} Design Philosophy
:class: tip

**No autotuning required.** tritonBLAS uses an analytical model to predict optimal configurations instantly, eliminating the overhead and unpredictability of traditional autotuning approaches.
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Application                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐       ┌─────────────────────────────┐     │
│   │      matmul()       │       │        matmul_lt()          │     │
│   │   Simple & Quick    │       │   Peak Performance API      │     │
│   └──────────┬──────────┘       └──────────────┬──────────────┘     │
│              │                                  │                   │
│              └──────────────┬───────────────────┘                   │
│                             ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   Analytical Model                          │   │
│   │          Instant configuration prediction                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│              ┌──────────────┼──────────────┐                        │
│              ▼              ▼              ▼                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│   │  Persistent  │  │   Stream-K   │  │ Specialized  │              │
│   │    GEMM      │  │    GEMM      │  │   Kernels    │              │
│   └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Triton + ROCm  │
                    │    GPU Runtime   │
                    └──────────────────┘
```

---

## The Two APIs

tritonBLAS offers two paths to high performance:

| | **matmul()** | **matmul_lt()** |
|---|---|---|
| **Purpose** | Drop-in replacement | Peak performance |
| **Config** | Automatic | Reusable selector |
| **Best for** | Prototyping, varied workloads | Production, repeated operations |

### matmul() — Simple & Quick

Perfect for quick integration. Just swap in `tritonblas.matmul()` and get automatic optimization:

```python
import tritonblas

tritonblas.matmul(A, B, C)
```

### matmul_lt() — Peak Performance

Maximum throughput with reusable configurations. Inspired by hipBLASLt/cuBLASLt:

```python
import tritonblas

# Create configuration once
selector = tritonblas.MatmulHeuristicResult(m, n, k, a_dtype, b_dtype, c_dtype)

# Reuse for maximum performance
tritonblas.matmul_lt(A, B, C, selector)
```

---

## How It Works

### Standard Path

When you call `matmul()`, tritonBLAS automatically:

```
   Your Code          Analytical Model         Optimal Kernel
       │                    │                       │
       │   matmul(A, B)     │                       │
       │ ──────────────────►│                       │
       │                    │  Analyze M, N, K      │
       │                    │  + dtypes + hardware  │
       │                    │                       │
       │                    │  Select config        │
       │                    │ ─────────────────────►│
       │                    │                       │  Execute
       │◄───────────────────────────────────────────│
       │         Result C                           │
```

### Optimized Path

With `matmul_lt()`, you control when configuration happens:

```
                         ┌──────────────────────────────────────┐
                         │  MatmulHeuristicResult(m, n, k, ...) │
                         │                                      │
                         │  ► Analyzed once                     │
                         │  ► Stored in selector                │
                         │  ► Reused for all calls              │
                         └───────────────┬──────────────────────┘
                                         │
        ┌──────────────────┬─────────────┼─────────────┬──────────────────┐
        ▼                  ▼             ▼             ▼                  ▼
   matmul_lt()        matmul_lt()   matmul_lt()   matmul_lt()        matmul_lt()
        │                  │             │             │                  │
        ▼                  ▼             ▼             ▼                  ▼
   ┌─────────┐        ┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐
   │ Result  │        │ Result  │   │ Result  │   │ Result  │        │ Result  │
   └─────────┘        └─────────┘   └─────────┘   └─────────┘        └─────────┘
   
   No recompilation. No reconfiguration. Maximum throughput.
```

---

## Kernel Architecture

tritonBLAS includes several kernel implementations:

### Persistent GEMM

The workhorse kernel for most workloads:

- **Persistent threads**: Workgroups stay alive to process multiple tiles
- **Tiled computation**: Optimized block sizes for GPU cache hierarchy  
- **Multi-XCD aware**: Chiplet-optimized scheduling for MI300X

### Stream-K GEMM

For better load balancing on irregular shapes:

- **Fine-grained work distribution**: Splits work at K-iteration level
- **Automatic tail handling**: No wasted compute on partial tiles
- **Enable with**: `enable_streamk=True`

### Specialized Kernels

- **FP4 GEMM**: 4-bit floating point for extreme compression
- **A8W8 GEMM**: INT8 quantized inference with scale factors
- **FP8 GEMM**: 8-bit floating point for efficient training

---

## The Stages Module

For kernel developers, tritonBLAS provides composable building blocks:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Custom Kernel                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │ GemmContext │   │ScheduleCtx  │   │ InputView/OutputView│    │
│  │             │   │             │   │                     │    │
│  │ Block sizes │   │ Tile loop   │   │ Matrix access       │    │
│  │ K-loop      │   │ Work dist.  │   │ Pointer math        │    │
│  │ Accumulator │   │ Stream-K    │   │ Bounds checking     │    │
│  └─────────────┘   └─────────────┘   └─────────────────────┘    │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐                              │
│  │  ScaleView  │   │  BiasView   │                              │
│  │             │   │             │                              │
│  │ Quantization│   │ Bias add    │                              │
│  │ scales      │   │ epilogue    │                              │
│  └─────────────┘   └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

See the [Stages Reference](../reference/stages-autodoc.rst) for details.

---


## Learn More

- **[Analytical Model](analytical-model.md)**: How we predict optimal configurations
- **[Stages Reference](../reference/stages-autodoc.rst)**: Building custom kernels
- **[Core API Reference](../reference/api-autodoc.rst)**: API documentation

# Stages Guide

This guide introduces the stages module for building custom GEMM kernels in Triton. For complete API signatures and parameters, see the [Stages Reference](stages-autodoc.rst).

```{note}
All stages APIs are **device-side only** — they execute within `@triton.jit` kernels on the GPU. They cannot be called from host Python code.
```

## Overview

The stages module provides composable abstractions that simplify writing high-performance GEMM kernels:

| Component | Purpose |
|-----------|---------|
| `GemmContext` | Manages GEMM configuration and the K-loop |
| `ScheduleContext` | Handles tile scheduling for persistent/Stream-K |
| `InputView` / `OutputView` | Encapsulates matrix memory access |
| `Tile` | Represents a 2D tile with coordinates and shape |

## The Basic Pattern

A tritonBLAS stages kernel follows this pattern:

```python
@triton.jit
def gemm_kernel(A, B, C, M, N, K, ...):
    # 1. Create matrix views - describe your data layout
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # 2. Create context - configure the GEMM
    ctx = GemmContext(
        block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
        num_sms=NUM_SMS,
    )
    
    # 3. Create scheduler - handle work distribution
    sched = ScheduleContext(M, N, K, ctx)
    
    # 4. Persistent tile loop
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # 5. Compute and store
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        tensorC.store(acc, out_tile)
```

## Matrix Views

Matrix views encapsulate pointer arithmetic and memory access. You describe your matrix once, and the view handles the rest:

```python
# Create input views for A [M, K] and B [K, N]
tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)

# Create output view for C [M, N]
tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
```

The key benefit: you don't need to worry about row-major vs column-major layouts, transpose flags, or pointer arithmetic. Just provide the dimensions and strides.

## The GEMM Context

`GemmContext` bundles all GEMM configuration and provides the core computation:

```python
ctx = GemmContext(
    block_m=128, block_n=256, block_k=64,
    num_sms=NUM_SMS,
)

# Full K-loop in one call
acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
```

For fine-grained control, use `reduce_tile` to process one K-step at a time:

```python
acc = ctx.init_accumulator()
for k_idx in range(num_k_tiles):
    acc = ctx.reduce_tile(tensorA, tensorB, out_tile, k_idx, acc)
```

## Tile Scheduling

`ScheduleContext` handles work distribution across GPU workgroups:

```python
sched = ScheduleContext(M, N, K, ctx)

# Get this workgroup's tile range
start, total, stride = sched.persistent_tile_range()

for tile_id in range(start, total, stride):
    # Get tile coordinates
    out_tile = sched.get_tile_from_idx(tile_id)
    # Process tile...
```

For Stream-K scheduling (finer-grained work distribution):

```python
sched = ScheduleContext(M, N, K, ctx, streamk_tiles=STREAMK_TILES)

start, end = sched.iter_range()
for iter_id in range(start, end):
    pid_m, pid_n, k_iter = sched.get_iter(iter_id)
    # Process single K iteration...
```

## Epilogue Operations

For quantized GEMM, add scale and bias views:

```python
# Create epilogue views
scale_view = make_scale_view(A_scale, B_scale, M, N)
bias_view = make_bias_view(bias, M)

# Store applies: scale -> bias -> type convert -> store
tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
```

## Complete Example

Here's a complete quantized GEMM kernel:

```python
from tritonblas.kernels.stages import (
    GemmContext, ScheduleContext,
    make_tensor_view, make_output_view,
    make_scale_view, make_bias_view,
)

@triton.jit
def quantized_gemm(
    A, B, C, A_scale, B_scale, bias,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    # Matrix views
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # Epilogue views
    scale_view = make_scale_view(A_scale, B_scale, M, N)
    bias_view = make_bias_view(bias, M) if HAS_BIAS else None
    
    # Context and scheduler
    ctx = GemmContext(
        block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
        num_sms=NUM_SMS, quantized=True,
    )
    sched = ScheduleContext(M, N, K, ctx)
    
    # Persistent loop
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
```

## Key Concepts

### Device-Side Execution

All stages objects exist only on the GPU. They're created inside `@triton.jit` kernels and live in GPU registers:

```python
@triton.jit
def kernel(...):
    # All of this runs on the GPU
    ctx = GemmContext(...)       # GPU registers
    sched = ScheduleContext(...) # GPU registers
    tile = sched.get_tile(...)   # GPU registers
```

### Compile-Time vs Runtime

- **Compile-time (`tl.constexpr`)**: Block sizes, flags — baked into the kernel
- **Runtime**: Dimensions M, N, K — passed as kernel arguments

### Layout Independence

Views handle any memory layout. Just provide the correct strides:

```python
# Row-major A: stride_am = K, stride_ak = 1
# Column-major A: stride_am = 1, stride_ak = M
tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
```

## See Also

- [Stages Reference](stages-autodoc.rst): Complete API signatures and parameters
- [Architecture](../conceptual/architecture.md): How stages fit into tritonBLAS
- [Examples](../getting-started/examples.md): Working code examples

# Stages API

The `stages` module provides composable **device-side abstractions** for building high-performance GEMM kernels in Triton. These abstractions run entirely on the GPU and encapsulate common patterns for tile scheduling, memory access, and epilogue operations.

```{note}
All stages APIs are **device-side only** — they execute within `@triton.jit` kernels on the GPU. They cannot be called from host Python code.
```

## Overview

The stages module provides four core device-side abstractions:

| Component | Purpose |
|-----------|---------|
| `Tile` | 2D tile with coordinates and shape |
| `GemmContext` | GEMM configuration and K-loop execution |
| `ScheduleContext` | Persistent/Stream-K tile scheduling |
| `InputView`/`OutputView` | Matrix memory access patterns |

## Quick Example

```python
from tritonblas.kernels.stages import (
    GemmContext, ScheduleContext,
    make_tensor_view, make_output_view,
    make_scale_view, make_bias_view,
)

@triton.jit
def gemm_kernel(A, B, C, M, N, K, 
                stride_am, stride_ak, stride_bk, stride_bn, 
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                BLOCK_K: tl.constexpr, NUM_SMS: tl.constexpr):
    
    # Create matrix views - describe your matrices
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # Create GEMM context with configuration
    ctx = GemmContext(
        block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
        num_sms=NUM_SMS, num_xcds=1, group_size_m=8,
    )
    
    # Create scheduler
    sched = ScheduleContext(M, N, K, ctx)
    
    # Persistent tile loop
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # Compute GEMM
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        
        # Store result
        tensorC.store(acc, out_tile)
```

---

## Device-Side Execution Model

The stages module uses Triton's `@aggregate` decorator to create structured types that exist entirely on the GPU:

- **Aggregates are device-side objects**: `Tile`, `GemmContext`, `ScheduleContext`, and all view types are instantiated and used within `@triton.jit` kernels
- **No host-device transfers**: All state is stored in GPU registers or compile-time constants
- **Compile-time parameters**: Parameters marked `tl.constexpr` are baked into the kernel at compile time
- **Runtime parameters**: Dimensions like M, N, K are passed as kernel arguments and live in GPU registers

```python
@triton.jit
def my_kernel(...):
    # All of this runs on the GPU:
    ctx = GemmContext(...)      # Device-side aggregate
    sched = ScheduleContext(...) # Device-side aggregate
    tile = sched.get_tile(...)   # Device-side Tile object
    acc = ctx.reduce_axis(...)   # GPU tensor operations
```

---

## Tile

A device-side 2D tile with runtime coordinates and compile-time block sizes.

### Constructor

```python
Tile(pid_m, pid_n, block_m, block_n)
```

**Parameters:**
- `pid_m` - Tile coordinate in M dimension
- `pid_n` - Tile coordinate in N dimension  
- `block_m` - Block size in M dimension (constexpr)
- `block_n` - Block size in N dimension (constexpr)

### Methods

#### `indices()`

Compute row and column indices for this tile.

```python
rm, rn = tile.indices()
# rm: [BLOCK_M] row indices
# rn: [BLOCK_N] column indices
```

#### `layout(M, N)`

Compute memory layout with bounds checking.

```python
rm, rn, mask = tile.layout(M, N)
# rm, rn: Wrapped indices for pointer computation
# mask: Boolean mask for boundary handling
```

#### `scale(acc, A_scale_ptr, B_scale_ptr, M, N)`

Apply quantization scales to accumulator.

```python
acc = tile.scale(acc, A_scale_ptr, B_scale_ptr, M, N)
```

#### `bias(acc, bias_ptr, M)`

Add bias vector to accumulator.

```python
acc = tile.bias(acc, bias_ptr, M)
```

---

## GemmContext

GEMM execution context that manages configuration and the K-loop.

### Constructor

```python
GemmContext(
    block_m, block_n, block_k,
    num_sms, num_xcds=1,
    group_size_m=8, chunk_size=1,
    cache_modifier_a=".cg", cache_modifier_b=".cg",
    acc_dtype=tl.float32, allow_tf32=True,
    even_k=True, quantized=False,
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_m` | - | Block size M (constexpr) |
| `block_n` | - | Block size N (constexpr) |
| `block_k` | - | Block size K (constexpr) |
| `num_sms` | - | Number of SMs/CUs (constexpr) |
| `num_xcds` | 1 | Number of XCDs for chiplet transform |
| `group_size_m` | 8 | Group size for tile scheduling |
| `chunk_size` | 1 | Chunk size for chiplet scheduling |
| `cache_modifier_a` | ".cg" | Cache modifier for A loads |
| `cache_modifier_b` | ".cg" | Cache modifier for B loads |
| `acc_dtype` | tl.float32 | Accumulator dtype |
| `allow_tf32` | True | Allow TF32 for matmul |
| `even_k` | True | Whether K is evenly divisible by BLOCK_K |
| `quantized` | False | Use int32 accumulation for quantized inputs |

### Methods

#### `init_accumulator()`

Initialize a zero accumulator.

```python
acc = ctx.init_accumulator()
# Returns: Tensor [BLOCK_M, BLOCK_N] initialized to zeros
```

#### `reduce_tile(A, B, out_tile, k_idx, acc, boundary=False)`

Execute a single K step (one BLOCK_K iteration).

```python
acc = ctx.init_accumulator()
for k_idx in range(num_k_tiles):
    acc = ctx.reduce_tile(A, B, out_tile, k_idx, acc)
```

**Parameters:**
- `A` - InputView for matrix A [M, K]
- `B` - InputView for matrix B [K, N]
- `out_tile` - Output Tile with (pid_m, pid_n)
- `k_idx` - Current K tile index
- `acc` - Accumulator to update
- `boundary` - Whether this is a boundary iteration needing masking

#### `reduce_axis(A, B, out_tile)`

Execute the full GEMM K loop.

```python
acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
```

This method handles:
- Accumulator initialization
- Main K loop iteration
- Boundary handling (when `even_k=False`)

---

## ScheduleContext

Unified scheduling for persistent GEMM and Stream-K patterns.

### Constructor

```python
ScheduleContext(M, N, K, ctx, streamk_tiles=0)
```

**Parameters:**
- `M, N, K` - Problem dimensions
- `ctx` - GemmContext with block sizes and scheduling parameters
- `streamk_tiles` - Number of tiles for Stream-K (0 = persistent only)

### Persistent GEMM Methods

#### `persistent_tile_range()`

Get tile iteration range for persistent GEMM.

```python
start, total, stride = sched.persistent_tile_range()
for tile_id in range(start, total, stride):
    out_tile = sched.get_tile_from_idx(tile_id)
    # Process tile...
```

#### `get_tile_from_idx(tile_id)`

Get a Tile for a given linear tile ID.

```python
out_tile = sched.get_tile_from_idx(tile_id)
# Returns: Tile with computed (pid_m, pid_n)
```

#### `get_tile_from_coord(pid_m, pid_n)`

Get a Tile from 2D coordinates.

```python
out_tile = sched.get_tile_from_coord(pid_m, pid_n)
```

### Stream-K Methods

#### `iter_range()`

Get iteration range for Stream-K mode.

```python
start_iter, end_iter = sched.iter_range()
for iter_id in range(start_iter, end_iter):
    pid_m, pid_n, k_iter = sched.get_iter(iter_id)
    # Process single K iteration...
```

#### `get_iter(global_iter)`

Get coordinates for a given global iteration.

```python
pid_m, pid_n, k_iter = sched.get_iter(global_iter)
```

---

## Matrix Views

### InputView

Input matrix view for GEMM. Stores pointer, dimensions, and strides.

```python
tensorA = make_input_view(A, M, K, stride_am, stride_ak)
tensorB = make_input_view(B, K, N, stride_bk, stride_bn)

# Or use the alias:
tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
```

#### Methods

**`tile_ptrs(tile)`** - Compute pointer array and bounds mask.

```python
ptrs, mask = tensorA.tile_ptrs(a_tile)
```

**`load(tile, boundary=False, cache_modifier=".cg")`** - Load a tile.

```python
a_data = tensorA.load(a_tile, boundary=is_boundary)
```

### OutputView

Output matrix view with epilogue support.

```python
tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
```

#### Methods

**`store(data, tile, mask=None, scale=None, bias=None)`** - Store with optional epilogue.

```python
# Simple store
tensorC.store(acc, out_tile)

# With epilogue: scale -> bias -> type convert -> store
tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
```

**`load(tile, boundary=False)`** - Load for read-modify-write patterns.

### ScaleView

Scale vectors for quantized GEMM epilogue.

```python
scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N)
```

#### Methods

**`apply(acc, tile)`** - Apply quantization scales.

```python
acc = scale_view.apply(acc, out_tile)
```

### BiasView

Bias vector for GEMM epilogue.

```python
bias_view = make_bias_view(bias_ptr, M, stride_bias)
```

#### Methods

**`apply(acc, tile)`** - Add bias to accumulator.

```python
acc = bias_view.apply(acc, out_tile)
```

---

## Factory Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `make_input_view(ptr, rows, cols, stride_row, stride_col)` | `InputView` | Create input matrix view |
| `make_tensor_view(...)` | `InputView` | Alias for `make_input_view` |
| `make_output_view(ptr, rows, cols, stride_row, stride_col)` | `OutputView` | Create output matrix view |
| `make_scale_view(a_scale_ptr, b_scale_ptr, M, N)` | `ScaleView` | Create scale view for quantization |
| `make_bias_view(bias_ptr, M, stride=1)` | `BiasView` | Create bias view |

---

## Grid Utilities

### `chiplet_transform(pid, num_sms, num_xcds)`

Transform program ID for multi-XCD scheduling.

### `chiplet_transform_chunked(pid, num_sms, num_xcds, chunk_size)`

Chunked version of chiplet transform for better locality.

---

## Complete Example: Quantized GEMM

```python
from tritonblas.kernels.stages import (
    GemmContext, ScheduleContext,
    make_tensor_view, make_output_view,
    make_scale_view, make_bias_view,
)

@triton.jit
def quantized_gemm_kernel(
    A, B, C, A_scale, B_scale, bias,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    # Create matrix views
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # Create epilogue views
    scale_view = make_scale_view(A_scale, B_scale, M, N)
    bias_view = make_bias_view(bias, M) if HAS_BIAS else None
    
    # Create context and scheduler
    ctx = GemmContext(
        block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
        num_sms=NUM_SMS, num_xcds=NUM_XCDS,
        even_k=EVEN_K, quantized=True,
    )
    sched = ScheduleContext(M, N, K, ctx)
    
    # Persistent tile loop
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        
        # Compute GEMM with int32 accumulation
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        
        # Store with full epilogue: scale -> bias -> convert -> store
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)

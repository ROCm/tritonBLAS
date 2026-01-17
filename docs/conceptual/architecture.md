# Architecture

This page describes the architecture and design of tritonBLAS.

## Overview

tritonBLAS is designed as a lightweight, modular library that provides high-performance matrix multiplication through analytical model-driven kernel selection.

## Component Architecture

```
tritonBLAS
├── Core API Layer
│   ├── matmul() - Drop-in replacement API
│   └── matmul_lt() - Peak performance API
├── Configuration Layer
│   ├── MatmulHeuristicResult - Configuration selector
│   └── Analytical Model - Configuration predictor
├── Kernel Layer
│   ├── Persistent GEMM kernels
│   ├── Stream-K GEMM kernels
│   └── Specialized kernels (FP4, etc.)
└── Utilities
    ├── Origami - Tensor manipulation
    └── Helper functions
```

## Core Components

### 1. API Layer

The API layer provides two interfaces:

**Drop-in Replacement API (`matmul`)**
- PyTorch-compatible interface
- Automatic configuration selection
- Simplified usage for quick integration

**Peak Performance API (`matmul_lt`)**
- Explicit configuration management
- Optimal for repeated operations
- Inspired by hipBLASLt/cuBLASLt

### 2. Configuration Layer

**MatmulHeuristicResult**
- Encapsulates optimal kernel configuration
- Contains block sizes, thread counts, memory layout
- Reusable across multiple operations

**Analytical Model**
- Predicts optimal configuration without autotuning
- Considers matrix dimensions, data types, hardware
- Provides deterministic, explainable decisions

### 3. Kernel Layer

**Persistent GEMM Kernels**
- Main matrix multiplication implementation
- Optimized for various data types (FP16, BF16, FP32, FP8)
- Support for different transpose modes

**Stream-K GEMM Kernels**
- Alternative algorithm for better load balancing
- Useful for irregular matrix shapes
- Optional feature enabled via parameter

**Specialized Kernels**
- FP4 support (experimental)
- Custom optimizations for specific use cases

### 4. Utilities

**Origami**
- Tensor manipulation and layout transformations
- Memory-efficient operations

**Helper Functions**
- Data type conversions
- Validation and error checking
- Performance utilities

## Design Principles

### 1. Simplicity

- Minimal API surface
- Clear separation of concerns
- Easy to understand and use

### 2. Performance

- Analytical model eliminates autotuning overhead
- Efficient kernel implementations
- Optimal memory access patterns

### 3. Maintainability

- Modular architecture
- Well-defined interfaces
- Extensible design

### 4. Compatibility

- PyTorch integration
- ROCm/HIP backend
- Triton-based kernels

## Data Flow

### Standard Workflow

```
User Code
    ↓
matmul(A, B)
    ↓
Analytical Model
    ↓
Configuration Selection
    ↓
Kernel Compilation (JIT)
    ↓
Kernel Execution
    ↓
Result (C)
```

### Optimized Workflow (with Configuration Reuse)

```
User Code
    ↓
MatmulHeuristicResult(m, n, k, dtypes)
    ↓
Analytical Model (once)
    ↓
Configuration (reusable)
    ↓
matmul_lt(A, B, selector) ← Multiple calls
    ↓
Kernel Execution (no recompilation)
    ↓
Results
```

## Memory Management

### Tensor Allocation

- Input tensors: User-managed
- Output tensors: Auto-allocated or user-provided
- Intermediate buffers: Managed by kernels

### Memory Layout

- Row-major and column-major support
- Efficient transpose handling
- Optimized for cache locality

## Kernel Selection

The analytical model selects kernels based on:

1. **Matrix Dimensions**: M, N, K sizes
2. **Data Types**: Input and output precision
3. **Hardware**: GPU architecture and capabilities
4. **Workload**: Regular vs irregular shapes

## Extension Points

tritonBLAS is designed to be extensible:

### Adding New Data Types

1. Implement kernel variants
2. Update analytical model
3. Add API support

### Adding New Algorithms

1. Implement kernel (e.g., new GEMM variant)
2. Integrate with configuration layer
3. Expose via API if needed

### Custom Optimizations

1. Extend analytical model
2. Add specialized kernels
3. Update configuration selection

## Dependencies

### Required

- **PyTorch**: Tensor operations and GPU management
- **Triton**: Kernel compilation and execution
- **ROCm/HIP**: GPU runtime

### Optional

- **hipBLASLt**: C++ utilities (auto-fetched)

## Performance Considerations

### Compilation

- JIT compilation on first use
- Configuration reuse eliminates recompilation
- Smaller binary size than autotuned approaches

### Execution

- Optimized memory access patterns
- Efficient use of GPU resources
- Minimal overhead from configuration selection

### Scalability

- Handles various matrix sizes efficiently
- Adapts to different GPU architectures
- Supports batched operations

## Future Architecture

Planned enhancements:

- Multi-GPU support
- Additional BLAS operations
- Enhanced analytical models
- Custom kernel generators

## Learn More

- [Analytical Model](analytical-model.md): Configuration prediction
- [Performance](performance.md): Benchmarks and optimization
- [API Reference](../reference/api.md): Detailed API documentation

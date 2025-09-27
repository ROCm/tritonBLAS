"""!
@file matmul.py
@brief High-level matrix multiplication interface with persistent and Stream-K execution modes.

This module provides the main user-facing API for matrix multiplication operations
using Triton kernels optimized for AMD GPU hardware. It supports both persistent
and Stream-K execution strategies with automatic heuristic-based optimization.

Key features:
- Automatic tile size and grid optimization via MatmulHeuristicResult
- Support for persistent and Stream-K execution modes
- LRU caching of heuristic results for repeated problem sizes
- Pre-allocated global buffers for Stream-K synchronization
- Support for various data types (FP32, FP16, BF16, FP8, etc.)

@author TritonBLAS Development Team  
@date 2024
"""

import torch
import triton
import random
import functools
import time
from .internal.persistent_matmul import persistent_matmul
from .internal.streamk_matmul import streamk_matmul
from .origami import MatmulHeuristicResult
from typing import Dict, Tuple, Optional

#! Cache for tensor storage to avoid repeated allocations
_tensor_cache = {}

#! Current CUDA device index for hardware queries
current_device_index = torch.cuda.current_device()

#! CUDA device properties for the current device
current_device = torch.cuda.get_device_properties(current_device_index)

#! Maximum number of streaming multiprocessors on current device
MAX_SMS = current_device.multi_processor_count

#! Maximum block size for pre-allocated buffers (256x256 for fp16/bf16)
#! TODO: Adjust for fp8/fp4 data types
MAX_BLOCK_SIZE = 65536

#! Pre-allocated global synchronization locks for Stream-K execution
#! Used to coordinate partial tile accumulation across workgroups
_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)

#! Pre-allocated global partial results buffer for Stream-K execution  
#! Stores intermediate accumulation results during multi-workgroup tiles
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


#! LRU cache decorator for heuristic result caching
#! Saves several microseconds for previously seen problems by not rerunning optimization
@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
):
    """!
    @brief Create and cache a matrix multiplication heuristic selector.
    
    Factory function that creates MatmulHeuristicResult instances with LRU caching
    to avoid recomputing optimization parameters for repeated problem configurations.
    This provides significant performance benefits for applications with recurring
    matrix sizes and data types.
    
    @param M (int): Number of rows in matrix A and output matrix C
    @param N (int): Number of columns in matrix B and output matrix C
    @param K (int): Number of columns in matrix A and rows in matrix B
    @param a_dtype (torch.dtype): Data type of input matrix A
    @param b_dtype (torch.dtype): Data type of input matrix B  
    @param c_dtype (torch.dtype): Data type of output matrix C
    
    @return MatmulHeuristicResult: Optimized configuration selector for the problem
    
    @details
    The LRU cache stores up to 1024 unique problem configurations, keyed by
    the combination of matrix dimensions and data types. Cache hits avoid:
    - Hardware detection and analysis
    - Matrix instruction dimension inference
    - Tile size optimization via Origami performance modeling
    - Grid size computation
    
    Cache effectiveness depends on application patterns:
    - Training workloads: High hit rates due to repeated layer sizes
    - Inference workloads: High hit rates for batch processing
    - Benchmarking: Perfect hit rates for repeated measurements
    
    @note The cache key includes all optimization-relevant parameters.
    Changes to any parameter will result in cache misses and recomputation.
    
    @see MatmulHeuristicResult for optimization algorithm details
    """
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, a_dtype, b_dtype, c_dtype)


def persistent_matmul_lt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector):
    """!
    @brief Execute matrix multiplication using persistent (data-parallel) execution mode.
    
    Performs C = A @ B using a persistent execution strategy where each workgroup
    processes exactly one tile of the output matrix. This mode provides predictable
    performance and is optimal for problems with good load balance.
    
    @param a (torch.Tensor): Input matrix A with shape (M, K)
    @param b (torch.Tensor): Input matrix B with shape (K, N) 
    @param c (torch.Tensor): Output matrix C with shape (M, N) - modified in-place
    @param selector (MatmulHeuristicResult): Pre-computed optimization configuration
    
    @return torch.Tensor: Reference to the modified output tensor c
    
    @details
    Persistent execution characteristics:
    - One-to-one mapping between workgroups and output tiles
    - Grid size equals total number of tiles (M/BLK_M * N/BLK_N)
    - Optimal for well-balanced problems with uniform tile work
    - Lower synchronization overhead compared to Stream-K
    - Predictable execution time and resource usage
    
    Kernel configuration:
    - 2 pipeline stages for memory latency hiding
    - 8 warps per workgroup for occupancy optimization
    - Matrix instruction size of 16x16 for AMD CDNA architectures
    - K-dimension packing factor of 1
    
    @pre Matrix A inner dimension must equal matrix B outer dimension (a.shape[1] == b.shape[0])
    @pre All tensors must be on the same CUDA device
    @pre Selector must be configured for the same problem dimensions
    
    @throws AssertionError: If matrix dimensions are incompatible
    
    @see streamk_matmul_lt() for Stream-K execution alternative
    @see MatmulHeuristicResult for optimization configuration details
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % BLK_K == 0

    # Kernel execution parameters optimized for most compute-bound workloads
    # TODO: Separate these configs for different problem characteristics
    num_stages = 2        # Pipeline stages for memory/compute overlap
    num_warps = 8         # Warps per workgroup for occupancy
    waves_per_eu = 0      # Let hardware scheduler decide
    mfmaInstrSize = 16    # Matrix instruction size (16x16)
    kpack = 1             # K-dimension packing factor

    # Configure for data-parallel execution
    grids = total_tiles

    # Execute persistent matrix multiplication kernel
    # TODO: Support bias addition and other GEMM variants
    kk = persistent_matmul[(grids,)](
        a,
        b,
        c,
        None,  # TODO: Enable bias tensor
        M,
        N,
        K,
        a.stride(0),     # Stride for A rows
        b.stride(1),     # Stride for B columns  
        c.stride(0),     # Stride for C rows
        c.stride(1),     # Stride for C columns
        0,               # TODO: Enable bias stride
        stride_ak=a.stride(1),    # Stride for A columns (K dimension)
        stride_bk=b.stride(0),    # Stride for B rows (K dimension)
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_programs,
        NUM_XCDS=8,      # Number of shader arrays (hardware-specific)
        BIAS=False,      # TODO: Enable bias support
        EVEN_K=even_k,   # K dimension divisibility optimization
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def streamk_matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector, sk_grid: Optional[int] = None
):
    """!
    @brief Execute matrix multiplication using Stream-K execution mode for load balancing.
    
    Performs C = A @ B using Stream-K execution strategy where workgroups can process
    multiple tiles or fractions of tiles to achieve better load balancing. This mode
    is particularly effective for problems with irregular tile boundaries or when
    the number of tiles doesn't evenly divide the number of compute units.
    
    @param a (torch.Tensor): Input matrix A with shape (M, K)
    @param b (torch.Tensor): Input matrix B with shape (K, N)
    @param c (torch.Tensor): Output matrix C with shape (M, N) - modified in-place  
    @param selector (MatmulHeuristicResult): Pre-computed optimization configuration
    @param sk_grid (Optional[int]): Override grid size for Stream-K execution.
                                   If None, uses selector's optimized grid size.
    
    @return torch.Tensor: Reference to the modified output tensor c
    
    @details
    Stream-K execution characteristics:
    - Workgroups can process partial tiles for load balancing
    - Grid size optimized via dynamic algorithms in compute_sk_grid()
    - Requires synchronization buffers for partial result accumulation
    - Better utilization when tiles don't evenly distribute across CUs
    - Higher overhead but improved performance for irregular problems
    
    Synchronization mechanism:
    - Uses pre-allocated global buffers when possible for performance
    - Falls back to dynamic allocation for large grids/blocks
    - Atomic operations coordinate partial tile accumulation
    - Lock-based synchronization ensures correctness
    
    Buffer management:
    - Reuses global buffers (_global_locks, _global_P) when size permits
    - Allocates temporary buffers for oversized configurations
    - Optimized buffer zeroing and initialization
    
    @pre Matrix A inner dimension must equal matrix B outer dimension (a.shape[1] == b.shape[0])
    @pre All tensors must be on the same CUDA device
    @pre Selector must be configured for the same problem dimensions
    
    @throws AssertionError: If matrix dimensions are incompatible
    
    @see persistent_matmul_lt() for data-parallel execution alternative
    @see MatmulHeuristicResult.compute_sk_grid() for grid optimization algorithm
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    ##
    # Grid Size Configuration
    ##
    total_programs_streamk = selector.get_grid()

    if total_programs_streamk > 0:  # Stream-K mode enabled
        total_tiles_streamk = total_tiles % total_programs_streamk
    else:  # Fallback to classical blocking (persistent mode)
        total_tiles_streamk = 0

    # Kernel execution parameters
    num_stages = 2        # Pipeline stages for memory/compute overlap
    num_warps = 8         # Warps per workgroup for occupancy
    waves_per_eu = 0      # Let hardware scheduler decide
    mfmaInstrSize = 16    # Matrix instruction size (16x16)
    kpack = 1             # K-dimension packing factor

    # Override grid size if explicitly specified
    if sk_grid is not None:
        total_programs_streamk = sk_grid

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    # Efficient buffer management with pre-allocated globals when possible
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]           # Synchronization locks  
        P = _global_P[:grids, :block_size]      # Partial results buffer
    else:
        # Dynamic allocation for oversized configurations
        locks = torch.empty(grids, device="cuda", dtype=torch.uint8)
        P = torch.empty(grids, block_size, device="cuda", dtype=torch.float32)

    # Execute Stream-K matrix multiplication kernel
    kk = streamk_matmul[(grids,)](
        a,
        b,
        c,
        None,  # TODO: Enable bias tensor
        P,     # Partial results accumulation buffer
        locks, # Synchronization locks for coordination
        M,
        N,
        K,
        a.stride(0),     # Stride for A rows
        b.stride(1),     # Stride for B columns
        c.stride(0),     # Stride for C rows  
        c.stride(1),     # Stride for C columns
        0,               # TODO: Enable bias stride
        stride_ak=a.stride(1),        # Stride for A columns (K dimension)
        stride_bk=b.stride(0),        # Stride for B rows (K dimension)
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=grids,
        NUM_XCDS=8,      # Number of shader arrays (hardware-specific)
        STREAMK_TILES=total_tiles_streamk,  # Number of Stream-K tiles
        BIAS=False,      # TODO: Enable bias support
        EVEN_K=even_k,   # K dimension divisibility optimization
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector, enable_streamk=False
):
    """!
    @brief Low-level matrix multiplication interface with pre-computed selector.
    
    Performs C = A @ B using a pre-computed MatmulHeuristicResult selector to avoid
    optimization overhead. Provides choice between persistent and Stream-K execution
    modes for different performance characteristics.
    
    @param a (torch.Tensor): Input matrix A with shape (M, K)
    @param b (torch.Tensor): Input matrix B with shape (K, N)
    @param c (torch.Tensor): Output matrix C with shape (M, N) - modified in-place
    @param selector (MatmulHeuristicResult): Pre-computed optimization configuration
    @param enable_streamk (bool, optional): Enable Stream-K execution mode for load balancing.
                                          Default is False (uses persistent mode).
    
    @return torch.Tensor: Reference to the modified output tensor c
    
    @details
    This is a low-level interface that requires manual selector management but provides
    maximum control over execution strategy. Useful for scenarios where:
    - Multiple operations share the same matrix dimensions and data types
    - Custom grid size or tile selection is required
    - Benchmark or profiling requires consistent configurations
    
    Execution mode selection:
    - Persistent mode (enable_streamk=False): Optimal for well-balanced problems
    - Stream-K mode (enable_streamk=True): Better for irregular or small problems
    
    @pre Matrix A inner dimension must equal matrix B outer dimension (a.shape[1] == b.shape[0])
    @pre All tensors must be on the same CUDA device
    @pre Selector must be configured for compatible problem dimensions
    
    @throws AssertionError: If matrix dimensions are incompatible
    
    @see matmul() for high-level interface with automatic selector creation
    @see persistent_matmul_lt() for persistent execution details
    @see streamk_matmul_lt() for Stream-K execution details
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector)
    else:
        return persistent_matmul_lt(a, b, c, selector)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    sk_grid=None,
):
    """!
    @brief High-level matrix multiplication interface with automatic optimization.
    
    Performs C = A @ B with automatic heuristic-based optimization for tile sizes,
    grid configuration, and execution strategy. This is the primary user-facing API
    for matrix multiplication operations in TritonBLAS.
    
    @param a (torch.Tensor): Input matrix A with shape (M, K)
    @param b (torch.Tensor): Input matrix B with shape (K, N)  
    @param c (torch.Tensor): Output matrix C with shape (M, N) - modified in-place
    @param enable_streamk (bool, optional): Enable Stream-K execution mode for load balancing.
                                          Default is False (uses persistent mode).
    @param sk_grid (Optional[int]): Override Stream-K grid size. Only used when
                                   enable_streamk=True. If None, uses optimized grid size.
    
    @return torch.Tensor: Reference to the modified output tensor c
    
    @details
    This function provides a complete automated matrix multiplication pipeline:
    1. Extract matrix dimensions and data types
    2. Create/retrieve cached MatmulHeuristicResult selector  
    3. Dispatch to appropriate execution kernel (persistent or Stream-K)
    4. Return modified output tensor
    
    Automatic optimizations include:
    - Hardware-aware tile size selection via Origami performance modeling
    - Matrix instruction dimension inference based on data types
    - Grid size optimization for load balancing
    - LRU caching of optimization results for repeated problem sizes
    
    Execution modes:
    - Persistent mode (default): One workgroup per output tile, optimal for balanced problems
    - Stream-K mode: Dynamic load balancing with partial tile processing
    
    Performance considerations:
    - First call for new (M,N,K,dtype) combination incurs optimization overhead (~microseconds)
    - Subsequent calls with same parameters use cached results (near-zero overhead)
    - Memory allocation is handled internally with pre-allocated buffers when possible
    
    @pre Matrix A inner dimension must equal matrix B outer dimension (a.shape[1] == b.shape[0])
    @pre All tensors must be on the same CUDA device and contiguous in memory
    @pre Tensors must have compatible data types supported by the hardware
    
    @throws AssertionError: If matrix dimensions are incompatible
    
    @see matmul_lt() for low-level interface with manual selector management
    @see MatmulHeuristicResult for optimization algorithm details
    
    @par Example Usage:
    @code{.py}
    import torch
    import tritonblas
    
    # Create input matrices
    A = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
    B = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    C = torch.zeros(1024, 1024, device="cuda", dtype=torch.float16)
    
    # Perform matrix multiplication
    result = tritonblas.matmul(A, B, C)
    
    # Enable Stream-K for better load balancing
    result = tritonblas.matmul(A, B, C, enable_streamk=True)
    @endcode
    """
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Create or retrieve cached optimization selector
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
    
    # Dispatch to appropriate execution mode
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, sk_grid=sk_grid)
    else:
        return persistent_matmul_lt(a, b, c, selector)

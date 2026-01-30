import triton
import triton.language as tl
import torch

from tritonblas.shards import (
    ScheduleContext, 
    Tile, 
    GemmContext, 
    InputView, 
    OutputView,
    GemmConfig,
)


@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    # Performance parameters (used to construct GemmConfig on device)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """
    Persistent GEMM kernel using GemmConfig aggregate.
    
    GemmConfig is constructed on-device from the constexpr parameters,
    then passed to ScheduleContext and GemmContext constructors.
    
    GemmConfig bundles all GEMM parameters:
    - block_m, block_n, block_k
    - num_sms, num_xcds
    - group_size_m, chunk_size
    - cache_modifier_a, cache_modifier_b
    - acc_dtype, allow_tf32, even_k, quantized
    """
    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    
    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    
    # ============================================================
    # Create matrix views
    # InputView takes: (ptr, stride_reduction_dim, stride_free_dim, rows, cols)
    # For A [M, K]: K is reduction, M is free
    # For B [K, N]: K is reduction, N is free
    # ============================================================
    tensorA = InputView(A, stride_ak, stride_am, M, K)
    tensorB = InputView(B, stride_bk, stride_bn, K, N)
    tensorC = OutputView(C, stride_cm, stride_cn, M, N)
    
    # ============================================================
    # Construct GemmConfig aggregate on device with ALL parameters
    # ============================================================
    config = GemmConfig(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )
    
    # ============================================================
    # Create schedule context to give user control over outer loop scheduling
    # ============================================================
    sched = ScheduleContext(M, N, K, config)
    # ============================================================
    # Create GEMM context to handle the K-loop iteration and accumulation
    # ============================================================
    ctx = GemmContext(config)
    
    # ============================================================
    # Persistent loop: process multiple tiles per workgroup
    # ============================================================
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        pid_m, pid_n = sched.get_tile(tile_id)
        out_tile = Tile(pid_m, pid_n, config.block_m, config.block_n)
        
        # Compute GEMM for this tile
        acc = ctx.k_complete(tensorA, tensorB, out_tile)
        
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            acc = out_tile.scale(acc, A_scale_ptr, B_scale_ptr, M, N)
        
        # Add bias if provided
        if BIAS:
            acc = out_tile.bias(acc, bias_ptr, M, stride_bias)
        
        # Convert to output dtype and store
        result = acc.to(C.type.element_ty)
        c_ptrs, c_mask = tensorC.tile_ptrs(out_tile)
        tl.store(c_ptrs, result, mask=c_mask)

import triton
import triton.language as tl
import torch

from tritonblas.shards import ScheduleContext, Tile, GemmContext, tile_ptr, InputView


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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """
    Persistent GEMM kernel using the shards API.
    
    Shards used:
    - ScheduleContext: Unified scheduling with tile_range()/get_tile()
    - MatrixView: Bundle ptr, strides, dims into a single object
    - Tile: 2D tile coordinates and shape
    - GemmContext: GEMM accumulation context with gemm()/k_complete()
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
    # Create matrix view aggregates - bundle ptr, strides, dims
    # ============================================================
    tensorA = InputView(A, stride_ak, stride_am, M, K)
    tensorB = InputView(B, stride_bk, stride_bn, K, N)
    
    # ============================================================
    # Create schedule context - hides all loop index complexity
    # ============================================================
    sched = ScheduleContext(
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M, NUM_SMS,
        num_xcds=NUM_XCDS,
        chunk_size=CHUNK_SIZE,
    )
    
    # GEMM Context (created once, reused for all tiles)
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        acc_dtype=acc_dtype,
        allow_tf32=ALLOW_TF32,
        even_k=EVEN_K,
        quantized=QUANTIZED,
        cache_modifier_a=CACHE_MODIFIER_A,
        cache_modifier_b=CACHE_MODIFIER_B,
    )
    
    # ============================================================
    # Persistent loop: process multiple tiles per workgroup
    # ============================================================
    start_tile, total_tiles, stride = sched.tile_range()
    for tile_id in range(start_tile, total_tiles, stride):
        #Get output tile coordinate from scheduler
        pid_m, pid_n = sched.get_tile(tile_id)
        
        # Output tile descriptor
        out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        
        # ============================================================
        # Compute matrix multiplication using k_complete() API
        # Takes A_View and B_View aggregates
        # ============================================================
        acc = ctx.k_complete(tensorA, tensorB, out_tile)
        
        # ============================================================
        # Apply quantization scales and bias using tile methods
        # ============================================================
        
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            acc = out_tile.scale(acc, A_scale_ptr, B_scale_ptr, M, N)
        
        # Add bias if provided
        if BIAS:
            acc = out_tile.bias(acc, bias_ptr, M, stride_bias)
        
        # Convert to output dtype
        result = acc.to(C.type.element_ty)
        
        # ============================================================
        # Store result using tile_ptr device function
        # ============================================================
        c_ptrs, c_mask = tile_ptr(C, stride_cm, stride_cn, M, N, out_tile)
        tl.store(c_ptrs, result, mask=c_mask)

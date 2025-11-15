import triton
import triton.language as tl
import torch

from .indexing import grid_index, preamble, compute_scale_indices
from .memory import load, store
from .algorithms import multiply_accumulate
from .algorithms.binary import apply_scales, add_vector
from .algorithms.unary import convert_dtype

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
    load_func: tl.constexpr = load,  # Custom load function (default: tritonblas load)
    store_func: tl.constexpr = store,  # Custom store function (default: tritonblas store)
):
    """
    Persistent matmul kernel using 5-phase composable shards.
    
    This kernel demonstrates the 5-phase model:
    1. Preamble: Indexing only
    2. Load: Address math + global → CU movement
    3. Compute: Math only
    4. Postprocess: Scales, bias, activation, type conversion
    5. Store: Register → global movement
    
    This separation allows users to customize data movement (phases 2 & 5)
    while reusing compute logic (phase 3).
    
    Customization:
    Users can override the default load, postprocess, and store functions by
    passing custom @triton.jit functions via the load_func, postprocess_func,
    and store_func parameters (all marked as tl.constexpr).
    
    Example with custom postprocess:
        @triton.jit
        def my_custom_postprocess(acc, A_scale_ptr, B_scale_ptr, bias_ptr,
                                   pid_m, pid_n, rm, M, N, stride_bias,
                                   BLOCK_SIZE_M, BLOCK_SIZE_N, QUANTIZED, output_dtype):
            # Custom postprocess logic (e.g., different activation)
            result = acc * 2.0  # Example: scale by 2
            return result.to(output_dtype)
        
        # Use default load/store, custom postprocess
        persistent_matmul[grid](..., postprocess_func=my_custom_postprocess, ...)
    """
    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Determine output dtype for accumulator
    OUTPUT_IS_INT8 = C.type.element_ty == tl.int8
    
    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # Get 2D grid info
    pid, num_pid_m, num_pid_n, total_tiles = grid_index(
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        NUM_SMS, NUM_XCDS, CHUNK_SIZE,
        USE_CHIPLET_PID
    )

    # Persistent loop: process multiple tiles per thread block
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # ============================================================
        # Phase 1: Preamble - Get specific tile index and offsets
        # ============================================================
        pid_m, pid_n, row_indices, col_indices, loop_k, acc = preamble(
            tile_id, num_pid_m, num_pid_n,
            M, N, K,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M,
            OUTPUT_IS_INT8,
            EVEN_K,
        )
        
        # ============================================================
        # Phases 2 & 3: Load + Compute pipeline iteratively over K dimension
        # ============================================================
        for k_iter in range(loop_k):
            k0 = k_iter * BLOCK_SIZE_K
            
            # Phase 2: Load - Address math + global → CU load
            a = load_func(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=False, is_row_major=True)
            b = load_func(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=False, is_row_major=False)
            
            # Phase 3: Compute - Math only
            acc = multiply_accumulate(acc, a, b, QUANTIZED, ALLOW_TF32)
        
        # Handle K tail if needed
        if not EVEN_K:
            k0 = loop_k * BLOCK_SIZE_K
            
            # Phase 2: Load with masking
            a = load_func(A, row_indices, k0, stride_am, stride_ak, BLOCK_SIZE_K, K, CACHE_MODIFIER_A, mask_k=True, is_row_major=True)
            b = load_func(B, col_indices, k0, stride_bn, stride_bk, BLOCK_SIZE_K, K, CACHE_MODIFIER_B, mask_k=True, is_row_major=False)
            
            # Phase 3: Compute
            acc = multiply_accumulate(acc, a, b, QUANTIZED, ALLOW_TF32)
        
        # ============================================================
        # Phase 4: Postprocess - Scales, bias, activation
        # ============================================================
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            row_scale_indices, col_scale_indices = compute_scale_indices(pid_m, pid_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
            a_scales = tl.load(A_scale_ptr + row_scale_indices)
            b_scales = tl.load(B_scale_ptr + col_scale_indices)
            acc = apply_scales(acc, a_scales, b_scales)
        
        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0)
            # Check if we're using quantized mode based on whether scales were applied
            acc = add_vector(acc, bias_vector, QUANTIZED=(A_scale_ptr is not None))
        
        # Convert to output dtype
        result = convert_dtype(acc, C.type.element_ty)
        
        # ============================================================
        # Phase 5: Store - CU → global movement
        # ============================================================
        store_func(
            C, result,
            row_indices, col_indices,
            M, N,
            stride_cm, stride_cn,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

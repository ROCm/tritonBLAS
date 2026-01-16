import triton
import triton.language as tl
import torch

from .stages.indexing import grid_setup, idx2coord, compute_scale_indices
from .stages.algorithms import gemm_loop
from .stages.algorithms.binary import apply_scales, add_vector
from .stages.algorithms.unary import convert_dtype
from .stages.memory import store

@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    OUT,  # Output pointer (same as C unless C is broadcast)
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
    stride_outm,
    stride_outn,
    stride_bias,
    alpha,  # Scalar multiplier for A@B
    beta,   # Scalar multiplier for initial C
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
    C_ROW_BROADCAST: tl.constexpr,  # True if C is a row vector (M,) to broadcast across columns
    C_COL_BROADCAST: tl.constexpr,  # True if C is a column vector (N,) to broadcast across rows
    C_SCALAR: tl.constexpr,  # True if C is a scalar to broadcast to all elements
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    QUANTIZED: tl.constexpr = False,  # True for int8/fp8, False for fp16/bf16
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    # Stride guards
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Determine accumulator dtype based on output type
    # Use int32 for int8 output, float32 for all other types (fp16/bf16/fp32)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    
    # Use chiplet-aware PID mapping if NUM_XCDS > 1
    USE_CHIPLET_PID = NUM_XCDS != 1

    # Compute Global Grid information once.
    pid, num_pid_m, num_pid_n, total_tiles = grid_setup( #WG Index and number of tiles in M/N/total
        M, N, K, #Problem Dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N, #Tile Dimensions
        NUM_SMS, NUM_XCDS, CHUNK_SIZE, #Hardware Info
        USE_CHIPLET_PID #Enable chiplet swizzle (hw dependent)
    )

    # Persistent loop: process multiple tiles per workgroup
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # ============================================================
        # Compute tile coordinates and initialize accumulator (tile_id is 1D index row major)
        # ============================================================
        output_coord_m, output_coord_n, row_indices, col_indices, acc = idx2coord(
            tile_id, num_pid_m, num_pid_n,
            M, N,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
            GROUP_SIZE_M,
            acc_dtype,
        )
        
        # ============================================================
        # Compute matrix multiplication over full K dimension
        # ============================================================
        acc = gemm_loop(
            A, B, #Pointers to A and B tensors
            row_indices, col_indices, #The row and column indices to process
            acc, K, #Accumulator and problem K dimension
            stride_am, stride_ak, #A tensor strides
            stride_bn, stride_bk, #B tensor strides
            BLOCK_SIZE_K, #Block Size in K dimension
            CACHE_MODIFIER_A, CACHE_MODIFIER_B, #Cache modifiers to control locality
            QUANTIZED, ALLOW_TF32, EVEN_K, #Extra compile time constants
        )
        
        # ============================================================
        # Apply quantization scales, bias, and convert to output dtype
        # ============================================================
        # Apply quantization scales if provided
        if A_scale_ptr is not None:
            row_scale_indices, col_scale_indices = compute_scale_indices(output_coord_m, output_coord_n, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
            a_scales = tl.load(A_scale_ptr + row_scale_indices) #Load A scale tensor
            b_scales = tl.load(B_scale_ptr + col_scale_indices) #Load B scale tensor
            acc = apply_scales(acc, a_scales, b_scales) #Multiply A * B * acc
        
        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(bias_ptr + row_indices * stride_bias, mask=row_indices < M, other=0.0) #Load Bias vector
            # Check if we're using quantized mode based on whether scales were applied
            acc = add_vector(acc, bias_vector, QUANTIZED=(A_scale_ptr is not None)) #Add bias vector to output accumulator
        
        # Apply addmm formula: result = beta*C + alpha*acc
        # Load initial C values if beta != 0
        if beta != 0.0:
            if C_SCALAR:
                # C is a scalar - load single value and broadcast to all elements
                c_scalar = tl.load(C).to(tl.float32)
                acc = beta * c_scalar + alpha * acc
            elif C_ROW_BROADCAST:
                # C is a row vector (M,) - load and broadcast across columns
                c_vector = tl.load(C + row_indices * stride_cm, mask=row_indices < M, other=0.0).to(tl.float32)
                acc = beta * c_vector[:, None] + alpha * acc
            elif C_COL_BROADCAST:
                # C is a column vector (N,) - load and broadcast across rows
                c_vector = tl.load(C + col_indices * stride_cn, mask=col_indices < N, other=0.0).to(tl.float32)
                acc = beta * c_vector[None, :] + alpha * acc
            else:
                # C is a full matrix (M, N) - load tile normally
                c_offsets = row_indices[:, None] * stride_cm + col_indices[None, :] * stride_cn
                c_mask = (row_indices[:, None] < M) & (col_indices[None, :] < N)
                c_tile = tl.load(C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
                acc = beta * c_tile + alpha * acc
        elif alpha != 1.0:
            acc = acc * alpha
        
        # Convert to output dtype
        result = convert_dtype(acc, C.type.element_ty) #Quantize output accumulator to output datatype
        
        # ============================================================
        # Store result to output matrix
        # ============================================================
        store(
            OUT, result, #Output tensor pointer and output accumulator
            row_indices, col_indices, #Precomputed offsets
            M, N, #M and N dimension for masking OOB writes
            stride_outm, stride_outn, #Stride of output dimensions.
        )

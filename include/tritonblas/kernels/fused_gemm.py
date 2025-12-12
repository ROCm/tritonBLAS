import triton
import triton.language as tl
import torch

from .stages.indexing.pid_transforms import chiplet_transform_chunked

@triton.jit()
def fused_persistent_matmul(
    A,
    B0,
    C0,
    B1,
    C1,
    locks,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    M,
    N,
    K,
    P,
    stride_am,
    stride_b0n,
    stride_c0m,
    stride_c0n,
    stride_b1n,
    stride_c1m,
    stride_c1n,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_b0k: tl.constexpr,
    stride_b1k: tl.constexpr,
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

    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)
    alpha_num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    alpha_num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    alpha_total_tiles = alpha_num_pid_m * alpha_num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_b0n > 0)
    tl.assume(stride_b0k > 0)
    tl.assume(stride_c0m > 0)
    tl.assume(stride_c0n > 0)
    tl.assume(stride_b1n > 0)
    tl.assume(stride_b1k > 0)
    tl.assume(stride_c1m > 0)
    tl.assume(stride_c1n > 0)

    acc_dtype = tl.float32 if C0.type.element_ty != tl.int8 else tl.int32


    # GEMM ALPHA
    if pid < alpha_total_tiles:
        for tile_id in range(pid, alpha_total_tiles, NUM_SMS):
            num_pid_in_group = GROUP_SIZE_M * alpha_num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(alpha_num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rk = tl.arange(0, BLOCK_SIZE_K)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B0 + rk[:, None] * stride_b0k + rn[None, :] * stride_b0n

            if BIAS:
                bias_ = bias_ptr + rm * stride_bias
                bias = tl.load(bias_, mask=rm < M, other=0.0)

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1
            tl.assume(loop_k > 1)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

                if stride_b0k == 1:
                    b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
                else:
                    b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

                # Conditional dot product precision based on quantization mode
                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_b0k

            if not EVEN_K:
                k = loop_k
                rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                B_BASE = B0 + rk[:, None] * stride_b0k + rn[None, :] * stride_b0n
                if stride_ak == 1:
                    A_BASE = tl.multiple_of(A_BASE, (1, 16))
                else:
                    A_BASE = tl.multiple_of(A_BASE, (16, 1))

                if stride_b0k == 1:
                    B_BASE = tl.multiple_of(B_BASE, (16, 1))
                else:
                    B_BASE = tl.multiple_of(B_BASE, (1, 16))
                a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)

                if QUANTIZED:
                    acc += tl.dot(a, b, input_precision="ieee")
                else:
                    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            # Conditional scaling for quantized mode
            if QUANTIZED:
                # Create pointers for the scale tensors and load them
                rm_A_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
                rn_B_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
                A_scale = tl.load(A_scale_ptr + rm_A_scale)
                B_scale = tl.load(B_scale_ptr + rn_B_scale)
                acc *= A_scale[:, None] * B_scale[None, :]

            # Unified bias handling
            if BIAS:
                if QUANTIZED:
                    # For quantized mode: convert bias to float32, add to acc, then convert to output dtype
                    bias_float = bias.to(tl.float32)
                    c = acc + bias_float[:, None]
                    c = c.to(C0.type.element_ty)
                else:
                    # For non-quantized mode: convert acc to output dtype, then add bias
                    c = acc.to(C0.type.element_ty)
                    c += bias[:, None]
            else:
                c = acc.to(C0.type.element_ty)

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            C_ = C0 + rm[:, None] * stride_c0m + rn[None, :] * stride_c0n
            tl.store(C_, c, c_mask)
            
            # Update lock to signal this tile is ready
            tl.store(locks + tile_id, 1, cache_modifier=".wt")
    else:
        # BETA GEMM
        # Calculate beta workgroup ID and which tiles to wait on
        beta_pid = pid - alpha_total_tiles
        beta_num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        beta_num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
        beta_total_tiles = beta_num_pid_m * beta_num_pid_n
        
        # Wait for the 4 ALPHA tiles this BETA workgroup depends on (4:1 ratio)
        first_tile = beta_pid * 4
        for dep_tile in range(first_tile, first_tile + 4):
            if dep_tile < alpha_total_tiles:
                while tl.load(locks + dep_tile, cache_modifier=".cv", volatile=True) != 1:
                    pass
        
        # Process tiles for this BETA workgroup (persistent pattern)
        for tile_id in range(beta_pid, beta_total_tiles, NUM_SMS):
            num_pid_in_group = GROUP_SIZE_M * beta_num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(beta_num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)
            
            # C1 = C0 @ B1
            # C0 is (M, N), B1 is (N, P), C1 is (M, P)
            # So we're computing a tile of C1: (M_tile, P_tile)
            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rp = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % P
            rk = tl.arange(0, BLOCK_SIZE_K)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rp = tl.max_contiguous(tl.multiple_of(rp, BLOCK_SIZE_N), BLOCK_SIZE_N)
            
            # C0_BASE: read from C0, shape (M, N)
            # B1_BASE: read from B1, shape (N, P)
            C0_BASE = C0 + rm[:, None] * stride_c0m + rk[None, :] * stride_c0n
            B1_BASE = B1 + rk[:, None] * stride_b1k + rp[None, :] * stride_b1n
            
            loop_k = tl.cdiv(N, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1
            tl.assume(loop_k > 1)
            
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                # Load from C0 (M, N) - check stride
                if stride_c0n == 1:
                    c0 = tl.load(tl.multiple_of(C0_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
                else:
                    c0 = tl.load(tl.multiple_of(C0_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)
                
                # Load from B1 (N, P) - check stride
                if stride_b1k == 1:
                    b1 = tl.load(tl.multiple_of(B1_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
                else:
                    b1 = tl.load(tl.multiple_of(B1_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)
                
                # Conditional dot product precision based on quantization mode
                if QUANTIZED:
                    acc += tl.dot(c0, b1, input_precision="ieee")
                else:
                    acc += tl.dot(c0, b1, allow_tf32=ALLOW_TF32)
                C0_BASE += BLOCK_SIZE_K * stride_c0n
                B1_BASE += BLOCK_SIZE_K * stride_b1k
            
            if not EVEN_K:
                k = loop_k
                rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                C0_BASE = C0 + rm[:, None] * stride_c0m + rk[None, :] * stride_c0n
                B1_BASE = B1 + rk[:, None] * stride_b1k + rp[None, :] * stride_b1n
                if stride_c0n == 1:
                    C0_BASE = tl.multiple_of(C0_BASE, (1, 16))
                else:
                    C0_BASE = tl.multiple_of(C0_BASE, (16, 1))
                
                if stride_b1k == 1:
                    B1_BASE = tl.multiple_of(B1_BASE, (16, 1))
                else:
                    B1_BASE = tl.multiple_of(B1_BASE, (1, 16))
                c0 = tl.load(C0_BASE, mask=rk[None, :] < N, other=0.0, cache_modifier=CACHE_MODIFIER_A)
                b1 = tl.load(B1_BASE, mask=rk[:, None] < N, other=0.0, cache_modifier=CACHE_MODIFIER_B)
                
                if QUANTIZED:
                    acc += tl.dot(c0, b1, input_precision="ieee")
                else:
                    acc += tl.dot(c0, b1, allow_tf32=ALLOW_TF32)
            
            # Conditional scaling for quantized mode
            if QUANTIZED:
                # Create pointers for the scale tensors and load them
                rm_C0_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
                rp_B1_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % P
                C0_scale = tl.load(A_scale_ptr + rm_C0_scale)  # Reuse A_scale_ptr for C0 scales
                B1_scale = tl.load(B_scale_ptr + rp_B1_scale)  # Reuse B_scale_ptr for B1 scales
                acc *= C0_scale[:, None] * B1_scale[None, :]
            
            # Unified bias handling
            if BIAS:
                bias_ = bias_ptr + rm * stride_bias
                bias = tl.load(bias_, mask=rm < M, other=0.0)
                if QUANTIZED:
                    # For quantized mode: convert bias to float32, add to acc, then convert to output dtype
                    bias_float = bias.to(tl.float32)
                    c = acc + bias_float[:, None]
                    c = c.to(C1.type.element_ty)
                else:
                    # For non-quantized mode: convert acc to output dtype, then add bias
                    c = acc.to(C1.type.element_ty)
                    c += bias[:, None]
            else:
                c = acc.to(C1.type.element_ty)
            
            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rp = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % P
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rp = tl.max_contiguous(tl.multiple_of(rp, BLOCK_SIZE_N), BLOCK_SIZE_N)
            c_mask = (rm[:, None] < M) & (rp[None, :] < P)
            C_ = C1 + rm[:, None] * stride_c1m + rp[None, :] * stride_c1n
            tl.store(C_, c, c_mask)


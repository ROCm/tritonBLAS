import triton
import triton.language as tl

@triton.jit
def build_H(SIZE: tl.constexpr):
    r"""
    Construct Hadamard matrix using H_{i,j} = (-1)^{popcount(i & j)}.
    
    This computes the bitwise dot product of row index i and column index j:
    - popcount(i & j) counts the number of matching 1-bits
    - If count is even: H_{i,j} = 1
    - If count is odd: H_{i,j} = -1
    
    Args:
        SIZE: Matrix dimension (must be power of 2, max 64)
        dtype: Output data type
        
    Returns:
        SIZE x SIZE Hadamard matrix
    """
    tl.static_assert(0 < SIZE)
    tl.static_assert(SIZE <= 64) # extend to 128 ?
    
    # Create row and column indices
    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)
    
    # Compute bitwise AND for all (i, j) pairs
    matching_bits = i[:, None] & j[None, :]
    
    # Count set bits (popcount) - simple iterative approach
    bit_sum = tl.zeros_like(matching_bits)
    temp = matching_bits
    for _ in tl.static_range(6):  # 6 iterations for up to 64 bits
        bit_sum += temp & 1
        temp >>= 1
    
    # Map: even popcount -> +1, odd popcount -> -1
    H = 1 - 2 * (bit_sum & 1)
    
    # normalize by sqrt(d)
    H = H / tl.math.sqrt(float(SIZE))
    
    return H

@triton.jit
def _hadamard_mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M* NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # to print: tl.device_print to print values @ runtime maybe....
    
    # add hadamard here
    h_block = build_H(MXFP4_QUANT_BLOCK_SIZE).to(x.dtype)
    x = tl.dot(x, h_block)
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # 1 x 32  block size 
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True) # 32 x 1 scales, 32 threads doing 1 x 32 parallel across threads
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign, exponents and mantissa fields from FP32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # to print: tl.device_print to print values @ runtime maybe....
    
    # add hadamard here
    # tl.dot (x, hadamard)

    # 1 x 32  block size 
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True) # 32 x 1 scales, 32 threads doing 1 x 32 parallel across threads
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign, exponents and mantissa fields from FP32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor[:, None] * weight
    return rms_norm

@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,
        "EVEN_M_N2": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N2"] % (args["BLOCK_SIZE_N2"]) == 0,
    }
)
@triton.jit
def _fused_rms_hadamard_mxfp4_quant_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    res1_ptr,
    out1_fp4_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    eps1,
    eps2,
    M,
    N1,
    N2,
    x1_stride_m,
    x2_stride_m,
    res1_stride_m,
    out1_fp4_stride_m,
    out1_bs_stride_m,
    out1_bs_stride_n,
    out2_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    HAS_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_M_PAD: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
    SHUFFLE: tl.constexpr,
    SHUFFLE_PAD: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    EVEN_M_N2: tl.constexpr,
):
    # TODO: XCD remapping where every 32-token block should share the same XCD
    # TODO: debug for large M
    # TODO: investigate cache_modifier='.cg' on tl.store
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    if pid >= num_pid_m:
        if HAS_SECOND_INPUT:
            pid -= num_pid_m
            x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            x_offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
            mask2 = None
            other2 = None
            if not EVEN_M_N2:
                mask2 = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]
                other2 = 0.0

            x2 = tl.load(
                x2_ptr + x_offs_m[:, None] * x2_stride_m + x_offs_n2[None, :],
                mask=mask2,
                other=other2,
                cache_modifier=".cg",
            ).to(tl.float32)

            w_mask2 = None
            w_other2 = None
            if not EVEN_M_N2:
                w_mask2 = x_offs_n2 < N2
                w_other2 = 0.0

            w2 = tl.load(w2_ptr + x_offs_n2, mask=w_mask2, other=w_other2).to(
                tl.float32
            )

            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            tl.store(
                out2_ptr + x_offs_m[:, None] * out2_stride_m + x_offs_n2[None, :],
                norm2.to(out2_ptr.type.element_ty),
                mask=mask2,
                cache_modifier=".cg",
            )
        return

    x_offs_n = tl.arange(0, BLOCK_SIZE_N)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask1 = None
    other1 = None
    if not EVEN_M_N:
        mask1 = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]
        other1 = 0.0

    x1 = tl.load(
        x1_ptr + x_offs_m[:, None] * x1_stride_m + x_offs_n[None, :],
        mask=mask1,
        other=other1,
        cache_modifier=".cg",
    ).to(tl.float32)

    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + x_offs_m[:, None] * res1_stride_m + x_offs_n[None, :],
            mask=mask1,
            other=other1,
            cache_modifier=".cg",
        ).to(tl.float32)
        x1 = x1 + res1

    w_mask1 = None
    w_other1 = None
    if not EVEN_M_N:
        w_mask1 = x_offs_n < N1
        w_other1 = 0.0

    w1 = tl.load(w1_ptr + x_offs_n, mask=w_mask1, other=w_other1).to(tl.float32)

    norm1 = _rmsmorm_op(x1, w1, N1, eps1)
    out1_fp4, bs_e8m0 = _hadamard_mxfp4_quant_op(
        norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # store the results
    half_x_offs_n = tl.arange(0, BLOCK_SIZE_N // 2)
    out_mask1 = None
    if not EVEN_M_N:
        out_mask1 = (x_offs_m < M)[:, None] & (half_x_offs_n < (N1 // 2))[None, :]

    tl.store(
        out1_fp4_ptr + x_offs_m[:, None] * out1_fp4_stride_m + half_x_offs_n[None, :],
        out1_fp4,
        mask=out_mask1,
        cache_modifier=".cg",
    )

    bs_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )
        bs_mask_127 = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
        bs_e8m0 = tl.where(bs_mask_127, bs_e8m0, 127)
    else:
        bs_offs = (
            bs_offs_m[:, None] * out1_bs_stride_m
            + bs_offs_n[None, :] * out1_bs_stride_n
        )

    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
                None, :
            ]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    tl.store(
        out1_bs_ptr + bs_offs,
        bs_e8m0.to(out1_bs_ptr.type.element_ty),
        mask=bs_mask,
        cache_modifier=".cg",
    )

    if FIRST_INPUT_RES:
        tl.store(
            out_res1_ptr + x_offs_m[:, None] * out_res1_stride_m + x_offs_n[None, :],
            x1.to(out_res1_ptr.dtype.element_ty),
            mask=mask1,
            cache_modifier=".cg",
        )


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,
        "EVEN_M_N2": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N2"] % (args["BLOCK_SIZE_N2"]) == 0,
    }
)
@triton.jit
def _fused_rms_mxfp4_quant_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    res1_ptr,
    out1_fp4_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    eps1,
    eps2,
    M,
    N1,
    N2,
    x1_stride_m,
    x2_stride_m,
    res1_stride_m,
    out1_fp4_stride_m,
    out1_bs_stride_m,
    out1_bs_stride_n,
    out2_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    HAS_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_M_PAD: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
    SHUFFLE: tl.constexpr,
    SHUFFLE_PAD: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    EVEN_M_N2: tl.constexpr,
):
    # TODO: XCD remapping where every 32-token block should share the same XCD
    # TODO: debug for large M
    # TODO: investigate cache_modifier='.cg' on tl.store
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    if pid >= num_pid_m:
        if HAS_SECOND_INPUT:
            pid -= num_pid_m
            x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            x_offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
            mask2 = None
            other2 = None
            if not EVEN_M_N2:
                mask2 = (x_offs_m < M)[:, None] & (x_offs_n2 < N2)[None, :]
                other2 = 0.0

            x2 = tl.load(
                x2_ptr + x_offs_m[:, None] * x2_stride_m + x_offs_n2[None, :],
                mask=mask2,
                other=other2,
                cache_modifier=".cg",
            ).to(tl.float32)

            w_mask2 = None
            w_other2 = None
            if not EVEN_M_N2:
                w_mask2 = x_offs_n2 < N2
                w_other2 = 0.0

            w2 = tl.load(w2_ptr + x_offs_n2, mask=w_mask2, other=w_other2).to(
                tl.float32
            )

            norm2 = _rmsmorm_op(x2, w2, N2, eps2)

            tl.store(
                out2_ptr + x_offs_m[:, None] * out2_stride_m + x_offs_n2[None, :],
                norm2.to(out2_ptr.type.element_ty),
                mask=mask2,
                cache_modifier=".cg",
            )
        return

    x_offs_n = tl.arange(0, BLOCK_SIZE_N)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask1 = None
    other1 = None
    if not EVEN_M_N:
        mask1 = (x_offs_m < M)[:, None] & (x_offs_n < N1)[None, :]
        other1 = 0.0

    x1 = tl.load(
        x1_ptr + x_offs_m[:, None] * x1_stride_m + x_offs_n[None, :],
        mask=mask1,
        other=other1,
        cache_modifier=".cg",
    ).to(tl.float32)

    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + x_offs_m[:, None] * res1_stride_m + x_offs_n[None, :],
            mask=mask1,
            other=other1,
            cache_modifier=".cg",
        ).to(tl.float32)
        x1 = x1 + res1

    w_mask1 = None
    w_other1 = None
    if not EVEN_M_N:
        w_mask1 = x_offs_n < N1
        w_other1 = 0.0

    w1 = tl.load(w1_ptr + x_offs_n, mask=w_mask1, other=w_other1).to(tl.float32)

    norm1 = _rmsmorm_op(x1, w1, N1, eps1)
    out1_fp4, bs_e8m0 = _mxfp4_quant_op(
        norm1, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # store the results
    half_x_offs_n = tl.arange(0, BLOCK_SIZE_N // 2)
    out_mask1 = None
    if not EVEN_M_N:
        out_mask1 = (x_offs_m < M)[:, None] & (half_x_offs_n < (N1 // 2))[None, :]

    tl.store(
        out1_fp4_ptr + x_offs_m[:, None] * out1_fp4_stride_m + half_x_offs_n[None, :],
        out1_fp4,
        mask=out_mask1,
        cache_modifier=".cg",
    )

    bs_offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )
        bs_mask_127 = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
        bs_e8m0 = tl.where(bs_mask_127, bs_e8m0, 127)
    else:
        bs_offs = (
            bs_offs_m[:, None] * out1_bs_stride_m
            + bs_offs_n[None, :] * out1_bs_stride_n
        )

    bs_mask = None
    if not EVEN_M_N:
        if SHUFFLE_PAD:
            bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
                None, :
            ]
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < SCALE_N)[None, :]

    tl.store(
        out1_bs_ptr + bs_offs,
        bs_e8m0.to(out1_bs_ptr.type.element_ty),
        mask=bs_mask,
        cache_modifier=".cg",
    )

    if FIRST_INPUT_RES:
        tl.store(
            out_res1_ptr + x_offs_m[:, None] * out_res1_stride_m + x_offs_n[None, :],
            x1.to(out_res1_ptr.dtype.element_ty),
            mask=mask1,
            cache_modifier=".cg",
        )


@triton.jit
def _fused_flatten_mxfp4_quant(
    x_ptr,
    out_ptr,
    out_scales_ptr,
    x_stride_m,
    x_stride_n1,
    x_stride_n2,
    out_stride_m,
    out_stride_n,
    out_scales_stride_m,
    out_scales_stride_n,
    N2,
    BLOCK_SIZE_N2: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // MXFP4_QUANT_BLOCK_SIZE
    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=n2_offs < N2)

    out, out_block_scales = _mxfp4_quant_op(x, BLOCK_SIZE_N2, 1, MXFP4_QUANT_BLOCK_SIZE)
    out = tl.ravel(out)
    out_block_scales = tl.ravel(out_block_scales)

    half_block_offs = tl.arange(0, BLOCK_SIZE_N2 // 2)
    tl.store(
        out_ptr
        + m * out_stride_m
        + (n1 * (BLOCK_SIZE_N2 // 2) + half_block_offs) * out_stride_n,
        out,
        mask=half_block_offs < (N2 // 2),
    )
    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * NUM_QUANT_BLOCKS + block_scale_offs) * out_scales_stride_n,
        out_block_scales,
        mask=block_scale_offs < tl.cdiv(N2, MXFP4_QUANT_BLOCK_SIZE),
    )

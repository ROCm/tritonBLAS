# SPDX-License-Identifier: MIT
# FP4 utilities for tritonblas testing
# Based on aiter's fp4_utils implementation

import torch
import triton
import triton.language as tl


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """
    Convert packed FP4 e2m1 data to FP32.
    
    FP4 e2m1 format (4 bits per value):
    - 1 sign bit
    - 2 exponent bits  
    - 1 mantissa bit
    
    Representable values:
    - 0x0 (0000): +0.0
    - 0x1 (0001): +0.5
    - 0x2 (0010): +1.0
    - 0x3 (0011): +1.5
    - 0x4 (0100): +2.0
    - 0x5 (0101): +3.0
    - 0x6 (0110): +4.0
    - 0x7 (0111): +6.0
    - 0x8 (1000): -0.0
    - 0x9 (1001): -0.5
    - 0xA (1010): -1.0
    - 0xB (1011): -1.5
    - 0xC (1100): -2.0
    - 0xD (1101): -3.0
    - 0xE (1110): -4.0
    - 0xF (1111): -6.0
    
    Args:
        x: Packed FP4 tensor (2 values per uint8)
        
    Returns:
        Unpacked FP32 tensor
    """
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    
    # Unpack: 2 FP4 values per uint8
    # Shape: (..., N) -> (..., N*2)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF  # Lower 4 bits
    x[..., 1::2] = x[..., 1::2] >> 4  # Upper 4 bits
    
    # Lookup table for FP4 e2m1 values
    mxfp4_list = [
        0.0,   # 0x0
        0.5,   # 0x1
        1.0,   # 0x2
        1.5,   # 0x3
        2.0,   # 0x4
        3.0,   # 0x5
        4.0,   # 0x6
        6.0,   # 0x7
        -0.0,  # 0x8
        -0.5,  # 0x9
        -1.0,  # 0xA
        -1.5,  # 0xB
        -2.0,  # 0xC
        -3.0,  # 0xD
        -4.0,  # 0xE
        -6.0,  # 0xF
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """
    Convert e8m0 scales to FP32.
    
    E8M0 format stores only the exponent (8 bits, biased by 127).
    The value is 2^(exponent - 127).
    
    Special cases:
    - 0x00: Represents 2^(-126) (minimum normal)
    - 0xFF: Represents NaN/Inf
    
    Args:
        scale_e8m0: E8M0 scale tensor
        
    Returns:
        FP32 scale tensor
    """
    scale_e8m0_biased = scale_e8m0.view(torch.uint8)
    
    # Special cases
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    
    # Convert to FP32 by placing exponent in correct position
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    
    # Handle special cases
    scale_f32[zero_case] = 0x00400000  # 2^(-126)
    scale_f32[nan_case] = 0x7F800001   # NaN
    
    scale_f32 = scale_f32.view(torch.float32)
    return scale_f32


@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for quantizing FP32/FP16/BF16 to FP4 e2m1 format.
    
    Each row is divided into blocks of MXFP4_QUANT_BLOCK_SIZE elements.
    Each block gets one e8m0 scale computed from the max absolute value.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load input block
    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Calculate scale per row (max absolute value)
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    
    # Convert to e8m0 format
    # Round up to nearest power of 2 for better numerical stability
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    
    # Compute unbiased exponent: log2(amax) - 2 (because max FP4 value is 6.0 = 2^2 * 1.5)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    
    # Quantization scale
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    
    # Quantize to FP4 range
    qx = x * quant_scale
    
    # Store e8m0 scale (add bias of 127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Convert quantized FP32 to FP4 e2m1
    qx = qx.to(tl.uint32, bitcast=True)
    
    # Extract sign, exponent, mantissa
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Handle denormal numbers
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

    # Adjust exponent bias
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine and saturate to 4 bits
    # Round nearest with tie breaking up
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

    # Pack two FP4 values into one uint8
    e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    # Store packed FP4 output
    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE // 2)
    out_offs = out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    # Store e8m0 scales
    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n
    bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
    bs_mask = (bs_offs_m < M)[:, None]
    tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def dynamic_mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 e2m1 format with e8m0 scales.
    
    The input is divided into blocks of 32 elements along the last dimension.
    Each block gets one e8m0 scale computed from the max absolute value.
    
    Args:
        x: Input tensor (FP32, FP16, or BF16), shape (..., N) where N % 32 == 0
        
    Returns:
        Tuple of:
        - x_fp4: Packed FP4 data, shape (..., N//2)
        - blockscale_e8m0: E8M0 scales, shape (..., N//32)
    """
    assert x.ndim == 2, "Input must be 2D tensor"
    M, N = x.shape
    assert N % 32 == 0, f"N ({N}) must be divisible by 32"

    # Fixed by MXFP4 spec
    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE = 128

    # Allocate output tensors
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    blockscale_e8m0 = torch.empty((M, scaleN), dtype=torch.uint8, device=x.device)

    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE), scaleN)
    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        x.stride(0),
        x.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        blockscale_e8m0.stride(0),
        blockscale_e8m0.stride(1),
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
    )

    return x_fp4, blockscale_e8m0

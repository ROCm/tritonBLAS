#!/usr/bin/env python3
"""
Utility functions for TritonBLAS

This module provides helper functions for dtype handling, quantization, and input generation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch  # type: ignore
import triton
import triton.language as tl

import os

# FP8 support flags - check for both variants
TORCH_HAS_FP8E5B16_FNUZ = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8_FNUZ = hasattr(torch, 'float8_e4m3fnuz')
TORCH_HAS_FP8E5B16_STD = hasattr(torch, 'float8_e5m2')
TORCH_HAS_FP8E4B8_STD = hasattr(torch, 'float8_e4m3fn')

# Base mapping from Triton language types to torch types (non-FP8)
_tl_to_torch_types_base = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}


def get_arch() -> str:
    """
    Get the current GPU architecture string (e.g., 'gfx950', 'gfx942').

    Returns:
        str: Architecture string, or 'unknown' if unable to determine.
    """
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch
    except Exception:
        return 'unknown'


def get_fp8_dtypes() -> Tuple[torch.dtype, torch.dtype]:
    """
    Get architecture-specific FP8 dtypes.

    For gfx950 (MI300): uses standard FP8 dtypes (float8_e5m2, float8_e4m3fn)
    For other architectures: uses FNUZ variants (float8_e5m2fnuz, float8_e4m3fnuz)

    Returns:
        Tuple[torch.dtype, torch.dtype]: (e5m2_dtype, e4m3_dtype)

    Raises:
        RuntimeError: If required FP8 dtypes are not available in PyTorch.
    """
    arch = get_arch()

    if arch == "gfx950":
        if not TORCH_HAS_FP8E5B16_STD:
            raise RuntimeError(f"Architecture {arch} requires torch.float8_e5m2 but it's not available")
        if not TORCH_HAS_FP8E4B8_STD:
            raise RuntimeError(f"Architecture {arch} requires torch.float8_e4m3fn but it's not available")
        e5m2_dtype = torch.float8_e5m2
        e4m3_dtype = torch.float8_e4m3fn
    else:
        if not TORCH_HAS_FP8E5B16_FNUZ:
            raise RuntimeError(f"Architecture {arch} requires torch.float8_e5m2fnuz but it's not available")
        if not TORCH_HAS_FP8E4B8_FNUZ:
            raise RuntimeError(f"Architecture {arch} requires torch.float8_e4m3fnuz but it's not available")
        e5m2_dtype = torch.float8_e5m2fnuz
        e4m3_dtype = torch.float8_e4m3fnuz

    return e5m2_dtype, e4m3_dtype


def get_tl_to_torch_types() -> dict:
    """
    Get the complete mapping from Triton language types to torch types,
    including architecture-specific FP8 dtypes.

    Returns:
        dict: Mapping from tl types to torch dtypes.
    """
    mapping = _tl_to_torch_types_base.copy()

    # Add FP8 dtypes based on architecture
    try:
        e5m2_dtype, e4m3_dtype = get_fp8_dtypes()
        mapping[tl.float8e5b16] = e5m2_dtype
        mapping[tl.float8e4b8] = e4m3_dtype
    except RuntimeError:
        # FP8 not available, skip
        pass

    return mapping


# For backward compatibility, create a lazy mapping that gets updated on first access
_tl_to_torch_types_cache = None


def _get_tl_to_torch_types_cached() -> dict:
    """Get cached version of tl_to_torch_types mapping."""
    global _tl_to_torch_types_cache
    if _tl_to_torch_types_cache is None:
        _tl_to_torch_types_cache = get_tl_to_torch_types()
    return _tl_to_torch_types_cache


# Backward compatibility: provide tl_to_torch_types as a property-like access
# This allows existing code using tl_to_torch_types to continue working
class _TlToTorchTypesProxy:
    """Proxy object that provides dict-like access to tl_to_torch_types."""
    def __getitem__(self, key):
        return _get_tl_to_torch_types_cached()[key]

    def get(self, key, default=None):
        return _get_tl_to_torch_types_cached().get(key, default)

    def __contains__(self, key):
        return key in _get_tl_to_torch_types_cached()

    def keys(self):
        return _get_tl_to_torch_types_cached().keys()

    def values(self):
        return _get_tl_to_torch_types_cached().values()

    def items(self):
        return _get_tl_to_torch_types_cached().items()


# Create proxy instance for backward compatibility
tl_to_torch_types = _TlToTorchTypesProxy()

# Backward compatibility: provide old flag names for code that still uses them
# These check if ANY variant is available (not architecture-specific)
TORCH_HAS_FP8E5B16 = TORCH_HAS_FP8E5B16_FNUZ or TORCH_HAS_FP8E5B16_STD
TORCH_HAS_FP8E4B8 = TORCH_HAS_FP8E4B8_FNUZ or TORCH_HAS_FP8E4B8_STD

# Mapping from shorthand string names to Triton language types
name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a dtype to the corresponding torch.dtype.

    Args:
        dtype_str (str): The string representation of the dtype (e.g., "torch.float32").

    Returns:
        torch.dtype: The corresponding torch dtype.
    """
    dtype_str = dtype_str.replace("torch.", "")
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(
            f"Invalid dtype string: '{dtype_str}'. Available options are: "
            f"{', '.join([attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)])}"
        )


def _ensure_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    """
    Convert a dtype string or torch.dtype to torch.dtype.

    Supports both shorthand names (fp16, fp8, int8, etc.) via name_to_tl_types/tl_to_torch_types
    and torch dtype names (float16, float8_e4m3fnuz, etc.).
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        # Use local mappings (now merged into this module)
        if dtype in name_to_tl_types:
            tl_type = name_to_tl_types[dtype]
            if tl_type in tl_to_torch_types:
                return tl_to_torch_types[tl_type]

        # Remove "torch." prefix if present
        dtype_clean = dtype.replace("torch.", "")

        # Try to get the torch dtype directly
        if hasattr(torch, dtype_clean):
            return getattr(torch, dtype_clean)
        else:
            shorthand_names = ['fp16', 'bf16', 'fp32', 'int8', 'int32', 'fp8', 'bf8']
            if dtype in shorthand_names:
                raise ValueError(
                    f"Unsupported dtype string: '{dtype}'. "
                    f"This should have been handled by name_to_tl_types mapping."
                )
            else:
                raise ValueError(
                    f"Unsupported dtype string: '{dtype}'. "
                    f"Expected a torch dtype name (e.g., 'float16') or shorthand name (e.g., 'fp16')."
                )
    raise TypeError(f"Unsupported dtype spec: {dtype}")


def _is_float8_like(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _is_int8(dtype: torch.dtype) -> bool:
    return dtype == torch.int8


def _is_quantized(init_result):
    """Normalize return from matmul_input_gen: either Tensor or (Tensor, scale)."""
    if isinstance(init_result, tuple) and len(init_result) == 2:
        return init_result[0], init_result[1]
    return init_result, None


@dataclass
class MatmulInputs:
    """
    Container for matmul tensors plus optional scale metadata.

    Fields:
        A (torch.Tensor): The left-hand matrix.
        B (torch.Tensor): The right-hand matrix.
        C (torch.Tensor): The output matrix.
        bias (torch.Tensor): Optional bias tensor, reserved for future support of bias-enabled matmul operations.
        scaleA (Optional[torch.Tensor]): Optional scale for quantized A. For FP8/INT8: 1D per-channel scales.
                                         For FP4: 2D block scales (M, K//32).
        scaleB (Optional[torch.Tensor]): Optional scale for quantized B. For FP8/INT8: 1D per-channel scales.
                                         For FP4: 2D block scales (N, K//32).
    """

    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    bias: torch.Tensor  # Reserved for future support of bias-enabled matmul operations.
    scaleA: Optional[torch.Tensor] = None
    scaleB: Optional[torch.Tensor] = None

    @property
    def is_quantized(self) -> bool:
        return self.scaleA is not None and self.scaleB is not None
    
    @property
    def is_fp4(self) -> bool:
        """Check if this is FP4 quantized data (2D block scales)."""
        if not self.is_quantized:
            return False
        # FP4 has 2D scales, FP8/INT8 have 1D scales
        return self.scaleA.ndim == 2 and self.scaleB.ndim == 2


def matmul_input_gen(
    size: Tuple[int, int],
    dtype: Union[torch.dtype, str],
    init_type: str,
    *,
    quantize: Optional[str] = None,  # None | "auto" | "fp8" | "int8"
    min_scale: float = 1e-8,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize a tensor and (optionally) quantize to FP8 or INT8 using per-channel scaling.

    Args:
        size: Shape of the tensor to create (M, N)
        dtype: Target dtype 
        init_type: Initialization method ("hpl", "trig_float", "zeros", "randn")
        quantize: Quantization mode - None (no quant), "auto" (based on dtype), "fp8", or "int8"
        min_scale: Minimum scale value to prevent division by very small numbers

    Returns:
        - If no quantization: Tensor[dtype]
        - If quantized: Tuple of (quantized_tensor, scale) where scale has shape (M, 1)
          and represents per-row scaling along dimension 1

    Quantization method:
        - For "fp8": The tensor is scaled per row so that the maximum absolute value in each row
          maps to the maximum representable value of the FP8 dtype. The scale tensor contains one
          value per row (shape (M, 1)), and quantization is performed as q = (base / scale).to(dtype).
        - For "int8": The tensor is scaled per row so that the maximum absolute value in each row
          maps to 127. The scale tensor contains one value per row (shape (M, 1)), and quantization
          is performed as q = round(base / scale).clamp(-127, 127).to(torch.int8).
        - The scale tensor is always float32.
    """
    dtype = _ensure_dtype(dtype)
    device = "cuda"
    M, N = size

    if init_type == "hpl":
        base = torch.empty(size, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        base = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(M, N).sin()
    elif init_type == "zeros":
        base = torch.zeros(size, dtype=torch.float32, device=device)
    elif init_type == "randn":
        base = torch.randn(size, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")

    mode = quantize
    if mode is None:
        return base.to(dtype)
    if mode == "auto":
        if _is_float8_like(dtype):
            mode = "fp8"
        elif _is_int8(dtype):
            mode = "int8"
        else:
            return base.to(dtype)

    base = base.to(torch.float32)

    if mode == "fp8":
        dtypeMax = torch.finfo(dtype).max
        max_base = torch.clamp(base.abs().amax(dim=1, keepdim=True), min=min_scale)
        scale = max_base / dtypeMax
        q = (base / scale).to(dtype)
    elif mode == "int8":
        dtypeMax = 127.0
        max_base = torch.clamp(base.abs().amax(dim=1, keepdim=True), min=min_scale)
        scale = max_base / dtypeMax
        q = torch.round(base / scale).clamp_(-dtypeMax, dtypeMax).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quantize mode: {mode}")

    return q, scale.to(torch.float32)


def quantize_tensor_per_channel(
    tensor: torch.Tensor,
    target_dtype: Union[torch.dtype, str],
    axis: int,
    *,
    min_scale: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize `tensor` along `axis` using per-channel scaling suitable for FP8/INT8 kernels.

    Args:
        tensor: Input tensor to quantize.
        target_dtype: Target quantized dtype (FP8 or INT8).
        axis: Dimension along which to compute per-channel scales.
        min_scale: Minimum scale value to prevent division by zero.

    Returns:
        Tuple of (quantized_tensor, scale_vector) where scale_vector is 1D
        (after squeezing along `axis`). The quantization formula is
        `q = base * (1 / scale)` or equivalently `q = base / scale`.
        The original tensor can be approximately reconstructed as
        `original ≈ quantized_tensor * scale_vector` (broadcasted along `axis`).
    """
    dtype = _ensure_dtype(target_dtype)
    if not (_is_float8_like(dtype) or _is_int8(dtype)):
        raise ValueError(f"Unsupported quantized dtype: {dtype}")

    base = tensor.to(torch.float32)
    max_abs = torch.clamp(base.abs().amax(dim=axis, keepdim=True), min=min_scale)
    dtype_max = torch.finfo(dtype).max if _is_float8_like(dtype) else 127.0
    scale = max_abs / dtype_max
    inv_scale = 1.0 / scale
    q = base * inv_scale
    if _is_int8(dtype):
        q = torch.round(q).clamp_(-dtype_max, dtype_max)
    q = q.to(dtype)
    squeezed_scale = scale.squeeze(axis).to(torch.float32)
    return q, squeezed_scale


def _init_matrix(shape: Tuple[int, int], init_type: str, device: str = "cuda", seed: Optional[int] = None) -> torch.Tensor:
    """Initialize a float32 matrix with the specified shape and initialization type."""
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    tensor = matmul_input_gen(shape, torch.float32, init_type, quantize=None)
    return tensor.to(device)


def generate_matmul_inputs(
    m: int,
    n: int,
    k: int,
    in_dtype: Union[torch.dtype, str, None] = None,
    out_dtype: Union[torch.dtype, str] = torch.float16,
    transA: str = "T",
    transB: str = "T",
    init_type: str = "randn",
    *,
    device: str = "cuda",
    quantize_mode: str = "auto",
    dtype_a: Union[torch.dtype, str, None] = None,
    dtype_b: Union[torch.dtype, str, None] = None,
    seed: Optional[int] = None,
) -> MatmulInputs:
    """
    Produce tensors (and optional scale metadata) for matmul benchmarks/tests.

    Generates A (m×k), B (k×n), C (m×n), and bias (m,) tensors. For quantized dtypes
    (fp4/fp8/int8), also produces scale tensors for A and B.

    Args:
        m (int): Number of rows of output matrix C (and A if not transposed).
        n (int): Number of columns of output matrix C (and B if not transposed).
        k (int): Shared dimension for A and B.
        in_dtype (torch.dtype or str, optional): Data type for input matrices A and B (if dtype_a/dtype_b not provided).
            For backward compatibility. If both in_dtype and dtype_a/dtype_b are provided, dtype_a/dtype_b take precedence.
            Special value "fp4" enables FP4 quantization.
        out_dtype (torch.dtype or str): Data type for output matrix C and bias. Default: torch.float16.
        transA (str): "T" if A is stored as m×k, "N" if stored as k×m and needs transpose. Default: "T".
        transB (str): "T" if B is stored as k×n, "N" if stored as n×k and needs transpose. Default: "T".
        init_type (str): Initialization method for A and B ("randn", "hpl", "trig_float", "zeros"). Default: "randn".
        device (str, optional): Device to allocate tensors on. Default: "cuda".
        quantize_mode (str, optional): "auto" (quantize if dtype is fp4/fp8/int8), "fp4", "fp8", "int8", or None. Default: "auto".
        dtype_a (torch.dtype or str, optional): Data type for matrix A. If None, uses in_dtype. Default: None.
        dtype_b (torch.dtype or str, optional): Data type for matrix B. If None, uses in_dtype. Default: None.
        seed (int, optional): Random seed for reproducibility. If None, uses current random state. Default: None.

    Returns:
        MatmulInputs: Dataclass with fields:
            - A (Tensor): Input matrix A. Shape (m, k) for non-FP4. For FP4: shape (m, k//2) physically packed.
            - B (Tensor): Input matrix B. Shape (k, n) for non-FP4. For FP4: shape (k//2, n) physically packed.
            - C (Tensor): Output matrix, shape (m, n).
            - bias (Tensor): Bias vector, shape (m,).
            - scaleA (Tensor or None): Scale for A. For FP4: shape (m, k//32). For FP8/INT8: shape (m,). None if not quantized.
            - scaleB (Tensor or None): Scale for B. For FP4: shape (n, k//32). For FP8/INT8: shape (n,). None if not quantized.

    Notes:
        - For FP4: K (the parameter) represents the unpacked logical dimension. Physical tensors are packed (2 FP4 values per uint8).
          K must be divisible by 32. Produces 2D block scales with one scale per 32 elements.
        - For FP8/INT8: Produces 1D per-channel scales.
        - The shapes of A and B depend on transA/transB: if "T", shape is (m, k)/(k, n); if "N", input is transposed.
        - Output C and bias are always allocated as (m, n) and (m,) respectively.
    """
    # Determine dtypes: dtype_a/dtype_b take precedence over in_dtype
    # Handle special "fp4" string before dtype conversion
    is_fp4_a = False
    is_fp4_b = False
    
    if dtype_a is None:
        if in_dtype is None:
            raise ValueError("Either in_dtype or dtype_a must be provided")
        dtype_a = in_dtype
    if dtype_b is None:
        if in_dtype is None:
            raise ValueError("Either in_dtype or dtype_b must be provided")
        dtype_b = in_dtype
    
    # Check for FP4 before dtype conversion
    if isinstance(dtype_a, str) and dtype_a.lower() == "fp4":
        is_fp4_a = True
        dtype_a = torch.uint8  # FP4 is stored as packed uint8
    if isinstance(dtype_b, str) and dtype_b.lower() == "fp4":
        is_fp4_b = True
        dtype_b = torch.uint8  # FP4 is stored as packed uint8

    dtype_a = _ensure_dtype(dtype_a)
    dtype_b = _ensure_dtype(dtype_b)
    out_dtype = _ensure_dtype(out_dtype)

    if transA not in {"T", "N"}:
        raise ValueError(f"transA must be 'T' or 'N', got: {transA}")
    if transB not in {"T", "N"}:
        raise ValueError(f"transB must be 'T' or 'N', got: {transB}")

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    # Determine quantization needs for A and B separately
    needs_quant_a = False
    needs_quant_b = False
    if quantize_mode == "auto":
        needs_quant_a = _is_float8_like(dtype_a) or _is_int8(dtype_a)
        needs_quant_b = _is_float8_like(dtype_b) or _is_int8(dtype_b)
    elif quantize_mode in {"fp8", "int8"}:
        needs_quant_a = True
        needs_quant_b = True
    elif quantize_mode not in {None, "none"}:
        raise ValueError(f"Unsupported quantize_mode: {quantize_mode}")

    # Generate A matrix
    if transA == "T":
        A = _init_matrix((m, k), init_type, device=device, seed=seed)
    else:
        A = _init_matrix((k, m), init_type, device=device, seed=seed).T

    # Generate B matrix (use different seed offset if seed provided)
    b_seed = seed + 10000 if seed is not None else None
    if transB == "T":
        B = _init_matrix((k, n), init_type, device=device, seed=b_seed)
    else:
        B = _init_matrix((n, k), init_type, device=device, seed=b_seed).T

    # Quantize A and B separately with their respective dtypes
    scaleA = scaleB = None
    
    # Handle FP4 quantization
    if is_fp4_a:
        if k % 32 != 0:
            raise ValueError(f"For FP4 quantization, K must be divisible by 32, got K={k}")
        A_fp4, scaleA = dynamic_mxfp4_quant(A)  # Returns (m, k//2) and (m, k//32)
        A = A_fp4
    elif needs_quant_a:
        qA, scaleA = quantize_tensor_per_channel(A, dtype_a, axis=1)  # Per-row scaling → (m,)
        A = qA
    else:
        A = A.to(dtype_a)

    if is_fp4_b:
        if k % 32 != 0:
            raise ValueError(f"For FP4 quantization, K must be divisible by 32, got K={k}")
        # B is (k, n), need to transpose for quantization, then transpose back
        B_T = B.T  # (n, k)
        B_fp4, scaleB = dynamic_mxfp4_quant(B_T)  # Returns (n, k//2) and (n, k//32)
        B = B_fp4.T  # (k//2, n)
    elif needs_quant_b:
        qB, scaleB = quantize_tensor_per_channel(B, dtype_b, axis=0)  # Per-column scaling → (n,)
        B = qB
    else:
        B = B.to(dtype_b)

    C = torch.zeros((m, n), device=device, dtype=out_dtype)
    bias = torch.zeros((m,), device=device, dtype=out_dtype)

    return MatmulInputs(A=A, B=B, C=C, bias=bias, scaleA=scaleA, scaleB=scaleB)


# ============================================================================
# FP4 Utilities
# ============================================================================
# Based on aiter's fp4_utils implementation


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


__all__ = [
    'get_arch',
    'get_fp8_dtypes',
    'get_tl_to_torch_types',
    'str_to_dtype',
    'MatmulInputs',
    'matmul_input_gen',
    'quantize_tensor_per_channel',
    'generate_matmul_inputs',
    'mxfp4_to_f32',
    'e8m0_to_f32',
    'dynamic_mxfp4_quant',
    'mx8_to_f32',
    'dynamic_mx8_quant',
]


# ============================================================================
# FP8 Utilities (MXFP8)
# ============================================================================
# Based on FP4 utilities, adapted for FP8 e5m2 format


def mx8_to_f32(x: torch.Tensor) -> torch.Tensor:
    """
    Convert FP8 e5m2 data to FP32.
    
    FP8 e5m2 format (8 bits per value):
    - 1 sign bit
    - 5 exponent bits (biased by 15)
    - 2 mantissa bits
    
    This function handles the standard FP8 e5m2 format by converting
    uint8 representation to the appropriate FP8 dtype, then to FP32.
    
    Args:
        x: FP8 tensor stored as uint8 (1 value per byte)
        
    Returns:
        FP32 tensor
    """
    # Get the appropriate FP8 dtype for this architecture
    e5m2_dtype, _ = get_fp8_dtypes()
    
    # If input is already in FP8 format, convert to float32
    if x.dtype == e5m2_dtype:
        return x.to(torch.float32)
    
    # Convert uint8 representation to FP8 dtype, then to float32
    # This uses PyTorch's native FP8 conversion which handles the format correctly
    if x.dtype == torch.uint8:
        x_fp8 = x.view(e5m2_dtype)
        return x_fp8.to(torch.float32)
    
    raise ValueError(f"Unsupported input dtype for mx8_to_f32: {x.dtype}")


@triton.jit
def _dynamic_mx8_quant_kernel(
    x_ptr,
    x_fp8_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp8_m,
    stride_x_fp8_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for quantizing FP32/FP16/BF16 to FP8 e5m2 format.
    
    Each row is divided into blocks of MXFP8_QUANT_BLOCK_SIZE elements.
    Each block gets one e8m0 scale computed from the max absolute value.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load input block
    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP8_QUANT_BLOCK_SIZE + tl.arange(0, MXFP8_QUANT_BLOCK_SIZE)
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
    
    # Compute unbiased exponent for FP8 e5m2
    # Max FP8 e5m2 value is 57344.0 = 2^15 * 1.75
    scale_e8m0_unbiased = tl.log2(amax).floor() - 15
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    
    # Quantization scale
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    
    # Quantize to FP8 range
    qx = x * quant_scale
    
    # Store e8m0 scale (add bias of 127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Convert quantized FP32 to FP8 e5m2
    qx = qx.to(tl.uint32, bitcast=True)
    
    # Extract sign, exponent, mantissa
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS: tl.constexpr = 127
    E5_BIAS: tl.constexpr = 15

    # Handle denormal numbers
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

    # Adjust exponent bias from FP32 (127) to FP8 e5m2 (15)
    e = tl.maximum(e, E8_BIAS - E5_BIAS) - (E8_BIAS - E5_BIAS)

    # Combine and saturate to 8 bits (1 sign + 5 exp + 2 mantissa)
    # Round nearest with tie breaking up
    e5m2_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7F)
    e5m2_value = ((s >> 24) | e5m2_tmp).to(tl.uint8)

    # Store FP8 output (no packing needed, 1 value per byte)
    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP8_QUANT_BLOCK_SIZE + tl.arange(0, MXFP8_QUANT_BLOCK_SIZE)
    out_offs = out_offs_m[:, None] * stride_x_fp8_m + out_offs_n[None, :] * stride_x_fp8_n
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < N)[None, :]
    tl.store(x_fp8_ptr + out_offs, e5m2_value, mask=out_mask)

    # Store e8m0 scales
    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n
    bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
    bs_mask = (bs_offs_m < M)[:, None]
    tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def dynamic_mx8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP8 e5m2 format with e8m0 scales.
    
    The input is divided into blocks of 32 elements along the last dimension.
    Each block gets one e8m0 scale computed from the max absolute value.
    
    Args:
        x: Input tensor (FP32, FP16, or BF16), shape (..., N) where N % 32 == 0
        
    Returns:
        Tuple of:
        - x_fp8: FP8 data, shape (..., N)
        - blockscale_e8m0: E8M0 scales, shape (..., N//32)
    """
    assert x.ndim == 2, "Input must be 2D tensor"
    M, N = x.shape
    assert N % 32 == 0, f"N ({N}) must be divisible by 32"

    # Fixed by MXFP8 spec (same as FP4)
    MXFP8_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE = 128

    # Allocate output tensors
    x_fp8 = torch.empty((M, N), dtype=torch.uint8, device=x.device)
    scaleN = triton.cdiv(N, MXFP8_QUANT_BLOCK_SIZE)
    blockscale_e8m0 = torch.empty((M, scaleN), dtype=torch.uint8, device=x.device)

    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE), scaleN)
    _dynamic_mx8_quant_kernel[grid](
        x,
        x_fp8,
        blockscale_e8m0,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        blockscale_e8m0.stride(0),
        blockscale_e8m0.stride(1),
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP8_QUANT_BLOCK_SIZE=MXFP8_QUANT_BLOCK_SIZE,
    )

    return x_fp8, blockscale_e8m0

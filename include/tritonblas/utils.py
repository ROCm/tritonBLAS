#!/usr/bin/env python3
"""
Utility functions for TritonBLAS

This module provides helper functions for dtype handling, quantization, and input generation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch  # type: ignore


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
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
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
        scaleA (Optional[torch.Tensor]): Optional scale for quantized A.
        scaleB (Optional[torch.Tensor]): Optional scale for quantized B.
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


def _init_matrix(shape: Tuple[int, int], init_type: str, device: str = "cuda") -> torch.Tensor:
    """Initialize a float32 matrix with the specified shape and initialization type."""
    tensor = matmul_input_gen(shape, torch.float32, init_type, quantize=None)
    return tensor.to(device)


def generate_matmul_inputs(
    m: int,
    n: int,
    k: int,
    in_dtype: Union[torch.dtype, str],
    out_dtype: Union[torch.dtype, str],
    transA: str,
    transB: str,
    init_type: str,
    *,
    device: str = "cuda",
    quantize_mode: str = "auto",
) -> MatmulInputs:
    """
    Produce tensors (and optional scale metadata) for matmul benchmarks/tests.

    Generates A (m×k), B (k×n), C (m×n), and bias (m,) tensors. For quantized dtypes
    (fp8/int8), also produces per-channel scale vectors for A and B.

    Args:
        m (int): Number of rows of output matrix C (and A if not transposed).
        n (int): Number of columns of output matrix C (and B if not transposed).
        k (int): Shared dimension for A and B.
        in_dtype (torch.dtype or str): Data type for input matrices A and B.
        out_dtype (torch.dtype or str): Data type for output matrix C and bias.
        transA (str): "T" if A is stored as m×k, "N" if stored as k×m and needs transpose.
        transB (str): "T" if B is stored as k×n, "N" if stored as n×k and needs transpose.
        init_type (str): Initialization method for A and B ("randn", "hpl", "trig_float", "zeros").
        device (str, optional): Device to allocate tensors on. Default: "cuda".
        quantize_mode (str, optional): "auto" (quantize if dtype is fp8/int8), "fp8", "int8", or None.

    Returns:
        MatmulInputs: Dataclass with fields:
            - A (Tensor): Input matrix A, shape (m, k) after any transpose.
            - B (Tensor): Input matrix B, shape (k, n) after any transpose.
            - C (Tensor): Output matrix, shape (m, n).
            - bias (Tensor): Bias vector, shape (m,).
            - scaleA (Tensor or None): Per-channel scale for A (if quantized), shape (k,) or None.
            - scaleB (Tensor or None): Per-channel scale for B (if quantized), shape (k,) or None.

    Notes:
        - If quantization is enabled (via quantize_mode or dtype), A and B are quantized and scaleA/scaleB are populated.
        - The shapes of A and B depend on transA/transB: if "T", shape is (m, k)/(k, n); if "N", input is transposed.
        - Output C and bias are always allocated as (m, n) and (m,) respectively.
    """
    in_dtype = _ensure_dtype(in_dtype)
    out_dtype = _ensure_dtype(out_dtype)
    if transA not in {"T", "N"}:
        raise ValueError(f"transA must be 'T' or 'N', got: {transA}")
    if transB not in {"T", "N"}:
        raise ValueError(f"transB must be 'T' or 'N', got: {transB}")
    needs_quant = False
    if quantize_mode == "auto":
        needs_quant = _is_float8_like(in_dtype) or _is_int8(in_dtype)
    elif quantize_mode in {"fp8", "int8"}:
        needs_quant = True
    elif quantize_mode not in {None, "none"}:
        raise ValueError(f"Unsupported quantize_mode: {quantize_mode}")

    if transA == "T":
        A = _init_matrix((m, k), init_type, device=device)
    else:
        A = _init_matrix((k, m), init_type, device=device).T

    if transB == "T":
        B = _init_matrix((k, n), init_type, device=device)
    else:
        B = _init_matrix((n, k), init_type, device=device).T

    scaleA = scaleB = None
    if needs_quant:
        qA, scaleA = quantize_tensor_per_channel(A, in_dtype, axis=1)
        qB, scaleB = quantize_tensor_per_channel(B, in_dtype, axis=0)
        A = qA
        B = qB
    else:
        A = A.to(in_dtype)
        B = B.to(in_dtype)

    C = torch.zeros((m, n), device=device, dtype=out_dtype)
    bias = torch.zeros((m,), device=device, dtype=out_dtype)

    return MatmulInputs(A=A, B=B, C=C, bias=bias, scaleA=scaleA, scaleB=scaleB)

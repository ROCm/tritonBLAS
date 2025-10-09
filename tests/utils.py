#!/usr/bin/env python3
import yaml
import torch
import random
import csv
from typing import Optional, Tuple, Union

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
    s = str(dtype)
    return ("float8" in s)

def _is_int8(dtype: torch.dtype) -> bool:
    return dtype == torch.int8

def _is_quantized(init_result):
    """Normalize return from init_by_size_and_type: either Tensor or (Tensor, scale)."""
    if isinstance(init_result, tuple) and len(init_result) == 2:
        return init_result[0], init_result[1]
    return init_result, None

##def matmul_input_gen(
##    size: Tuple[int, int],
##    dtype: Union[torch.dtype, str],
##    init_type: str,
##    *,
##    # quantization control
##    quantize: Optional[str] = None,      # None | "auto" | "fp8" | "int8"
##    scale_mode: str = "per_axis",         # "per_axis" | "per_tensor"
##    scale_axis: Optional[int] = None,     # required when scale_mode="per_axis": 0 or 1
##    min_scale: float = 1e-8,
##) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
##    """
##    Initialize a tensor and (optionally) quantize to FP8/BF8 or INT8.
##
##    Behavior:
##      - For FP16/BF16 (or any non-8-bit dtype) and quantize in {None,"auto"}:
##          returns Tensor[dtype] (backward compatible).
##      - For 8-bit quant (dtype is float8/bfloat8/int8 OR quantize="fp8"/"int8"):
##          returns (q, scale).
##        * scale_mode="per_tensor": scale is a single-element tensor shaped (1,1).
##        * scale_mode="per_axis":    scale reduces along 'scale_axis' (kept dim).
##
##    scale shapes:
##      - per_tensor: (1, 1)  (broadcastable to any (M,K) or (K,N))
##      - per_axis:   (M,1) if scale_axis=1; (1,N) if scale_axis=0
##    """
##    dtype = _ensure_dtype(dtype)
##    device = "cuda"
##
##    # ---- 1) Initialize base in fp32 ----
##    if init_type == "hpl":
##        base = torch.empty(size, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
##    elif init_type == "trig_float":
##        M, N = size
##        base = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(M, N).sin()
##    elif init_type == "zeros":
##        base = torch.zeros(size, dtype=torch.float32, device=device)
##    elif init_type == "randn":
##        base = torch.randn(size, dtype=torch.float32, device=device)
##    else:
##        raise ValueError(f"Unsupported init_type: {init_type}")
##
##    # ---- 2) Decide quant mode ----
##    mode = quantize
##    if mode is None:
##        return base.to(dtype)
##    if mode == "auto":
##        if _is_float8_like(dtype):
##            mode = "fp8"
##        elif _is_int8(dtype):
##            mode = "int8"
##        else:
##            # Non-8-bit target: keep original behavior
##            return base.to(dtype)
##
##    # ---- 3) Compute scale ----
##    if scale_mode not in ("per_axis", "per_tensor"):
##        raise ValueError("scale_mode must be 'per_axis' or 'per_tensor'.")
##
##    if scale_mode == "per_axis":
##        if scale_axis is None:
##            raise ValueError("scale_axis (0 or 1) is required when scale_mode='per_axis'.")
##        amax = torch.amax(base.abs(), dim=scale_axis, keepdim=True).to(torch.float32)
##    else:
##        # per_tensor: reduce over all dims and keep a broadcastable shape (1,1)
##        amax = torch.amax(base.abs()).to(torch.float32).reshape(1, 1)
##
##    if mode == "fp8":
##        max_val = float(torch.finfo(dtype).max)     # e.g., e4m3/e5m2
##        scale = torch.clamp(amax / max_val, min=min_scale)
##        q = (base / scale).to(dtype)
##        return q, scale
##
##    if mode == "int8":
##        qmax = 127.0                                # symmetric int8
##        scale = torch.clamp(amax / qmax, min=min_scale)
##        q = torch.round(base / scale).clamp_(-127, 127).to(torch.int8)
##        return q, scale
##
##    # Always return inverse scale for kernel consumption
##    inv_scale = (1.0 / scale).to(torch.float32)
##
##    raise ValueError(f"Unsupported quantize mode: {mode}")

def matmul_input_gen(
    size: Tuple[int, int],
    dtype: Union[torch.dtype, str],
    init_type: str,
    *,
    # quantization control
    quantize: Optional[str] = None,      # None | "auto" | "fp8" | "int8"
    scale_mode: str = "per_axis",        # "per_axis" | "per_tensor"
    scale_axis: Optional[int] = None,    # required when scale_mode="per_axis": 0 or 1
    min_scale: float = 1e-8,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize a tensor and (optionally) quantize to FP8 or INT8.

    Returns:
      - If no quantization: Tensor[dtype]
      - If quantized: (q, inv_scale)
          where q = base / scale, and inv_scale = 1/scale (float32),
          shaped per_axis ((M,1) or (1,N)) or per_tensor ((1,1)).
    """
    dtype = _ensure_dtype(dtype)
    device = "cuda"

    # ---- 1) Initialize base in fp32 ----
    if init_type == "hpl":
        base = torch.empty(size, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        M, N = size
        base = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(M, N).sin()
    elif init_type == "zeros":
        base = torch.zeros(size, dtype=torch.float32, device=device)
    elif init_type == "randn":
        base = torch.randn(size, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")

    # ---- 2) Decide quant mode ----
    mode = quantize
    if mode is None:
        return base.to(dtype)
    if mode == "auto":
        if _is_float8_like(dtype):
            mode = "fp8"
        elif _is_int8(dtype):
            mode = "int8"
        else:
            # Non-8-bit target: keep original behavior
            return base.to(dtype)

    # ---- 3) Compute scale (amax) ----
    if scale_mode not in ("per_axis", "per_tensor"):
        raise ValueError("scale_mode must be 'per_axis' or 'per_tensor'.")

    if scale_mode == "per_axis":
        if scale_axis is None:
            raise ValueError("scale_axis (0 or 1) is required when scale_mode='per_axis'.")
        amax = torch.amax(base.abs(), dim=scale_axis, keepdim=True).to(torch.float32)
    else:
        amax = torch.amax(base.abs()).to(torch.float32).reshape(1, 1)

    # Base scale (used for quantization)
    if mode == "fp8":
        max_val = float(torch.finfo(dtype).max)       # e.g., e4m3/e5m2
        scale = torch.clamp(amax / max_val, min=min_scale)
        q = (base / scale).to(dtype)
    elif mode == "int8":
        qmax = 127.0                                  # symmetric int8
        scale = torch.clamp(amax / qmax, min=min_scale)
        q = torch.round(base / scale).clamp_(-127, 127).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quantize mode: {mode}")

    # Always return inverse scale for kernel consumption
    inv_scale = (1.0 / scale).to(torch.float32)
    return q, inv_scale

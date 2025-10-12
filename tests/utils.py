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

def matmul_input_gen(
    size: Tuple[int, int],
    dtype: Union[torch.dtype, str],
    init_type: str,
    *,
    # quantization control
    quantize: Optional[str] = None,      # None | "auto" | "fp8" | "int8"
    min_scale: float = 1e-8,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize a tensor and (optionally) quantize to FP8 or INT8 using per-channel/per-token scaling.
    This function uses the exact same approach as AITer's generate_gemm_a8w8_inputs function.
    
    Returns:
      - If no quantization: Tensor[dtype]
      - If quantized: (q, scale)
          where q = base / scale, and scale is regular scale (not inverse) in float32,
          shaped for per-channel scaling: (M, 1) for input tensors, (1, N) for weight tensors.
    """
    dtype = _ensure_dtype(dtype)
    device = "cuda"
    M, N = size

    # ---- 1) Initialize base in fp32 ----
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

    base = base.to(torch.float32)
    
    if mode == "fp8":
        # Per-channel scaling: max along dimension 1 (columns), keepdim=True
        dtypeMax = torch.finfo(dtype).max
        max_base = base.abs().float().amax(dim=1, keepdim=True)  # (M, 1)
        scale = max_base / dtypeMax  # (M, 1)
        q = (base / scale).to(dtype)
    elif mode == "int8":
        # Per-channel scaling for INT8
        dtypeMax = 127.0  # symmetric int8
        max_base = base.abs().float().amax(dim=1, keepdim=True)  # (M, 1)
        scale = max_base / dtypeMax  # (M, 1)
        q = torch.round(base / scale).clamp_(-127, 127).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quantize mode: {mode}")

    return q, scale.to(torch.float32)

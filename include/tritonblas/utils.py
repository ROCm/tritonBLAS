#!/usr/bin/env python3
"""
Utility functions for TritonBLAS

This module provides helper functions for dtype handling, quantization, input generation,
and various utility functions for tuning and benchmarking.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch  # type: ignore
import triton
import triton.language as tl

import os
import subprocess
from datetime import datetime
from pathlib import Path

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
    (fp8/int8), also produces per-channel scale vectors for A and B.

    Args:
        m (int): Number of rows of output matrix C (and A if not transposed).
        n (int): Number of columns of output matrix C (and B if not transposed).
        k (int): Shared dimension for A and B.
        in_dtype (torch.dtype or str, optional): Data type for input matrices A and B (if dtype_a/dtype_b not provided).
            For backward compatibility. If both in_dtype and dtype_a/dtype_b are provided, dtype_a/dtype_b take precedence.
        out_dtype (torch.dtype or str): Data type for output matrix C and bias. Default: torch.float16.
        transA (str): "T" if A is stored as m×k, "N" if stored as k×m and needs transpose. Default: "T".
        transB (str): "T" if B is stored as k×n, "N" if stored as n×k and needs transpose. Default: "T".
        init_type (str): Initialization method for A and B ("randn", "hpl", "trig_float", "zeros"). Default: "randn".
        device (str, optional): Device to allocate tensors on. Default: "cuda".
        quantize_mode (str, optional): "auto" (quantize if dtype is fp8/int8), "fp8", "int8", or None. Default: "auto".
        dtype_a (torch.dtype or str, optional): Data type for matrix A. If None, uses in_dtype. Default: None.
        dtype_b (torch.dtype or str, optional): Data type for matrix B. If None, uses in_dtype. Default: None.
        seed (int, optional): Random seed for reproducibility. If None, uses current random state. Default: None.

    Returns:
        MatmulInputs: Dataclass with fields:
            - A (Tensor): Input matrix A, shape (m, k) after any transpose.
            - B (Tensor): Input matrix B, shape (k, n) after any transpose.
            - C (Tensor): Output matrix, shape (m, n).
            - bias (Tensor): Bias vector, shape (m,).
            - scaleA (Tensor or None): Per-channel scale for A (if quantized), shape (m,) or None.
            - scaleB (Tensor or None): Per-channel scale for B (if quantized), shape (n,) or None.

    Notes:
        - If quantization is enabled (via quantize_mode or dtype), A and B are quantized and scaleA/scaleB are populated.
        - The shapes of A and B depend on transA/transB: if "T", shape is (m, k)/(k, n); if "N", input is transposed.
        - Output C and bias are always allocated as (m, n) and (m,) respectively.
        - scaleA has shape (m,) for per-row scaling of A.
        - scaleB has shape (n,) for per-column scaling of B.
    """
    # Determine dtypes: dtype_a/dtype_b take precedence over in_dtype
    if dtype_a is None:
        if in_dtype is None:
            raise ValueError("Either in_dtype or dtype_a must be provided")
        dtype_a = in_dtype
    if dtype_b is None:
        if in_dtype is None:
            raise ValueError("Either in_dtype or dtype_b must be provided")
        dtype_b = in_dtype

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
    if needs_quant_a:
        qA, scaleA = quantize_tensor_per_channel(A, dtype_a, axis=1)  # Per-row scaling → (m,)
        A = qA
    else:
        A = A.to(dtype_a)

    if needs_quant_b:
        qB, scaleB = quantize_tensor_per_channel(B, dtype_b, axis=0)  # Per-column scaling → (n,)
        B = qB
    else:
        B = B.to(dtype_b)

    C = torch.zeros((m, n), device=device, dtype=out_dtype)
    bias = torch.zeros((m,), device=device, dtype=out_dtype)

    return MatmulInputs(A=A, B=B, C=C, bias=bias, scaleA=scaleA, scaleB=scaleB)


# ============================================================================
# Utility functions for tuning and benchmarking (merged from tools/utils/utils.py)
# ============================================================================

def get_num_sms():
    """Returns the Compute Unit count of the current device."""
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_output_dir():
    """Get the output directory for profiling results, creating it if needed."""
    # Try to determine the tools directory relative to this file
    # If we're in include/tritonblas/utils.py, tools is at ../../tools
    current_file = Path(__file__)
    tools_dir = current_file.parent.parent.parent / "tools"
    if not tools_dir.exists():
        # Fallback: use current directory
        tools_dir = Path.cwd()
    output_dir = tools_dir / "output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def run_bash_command(commandstring, capture=True):
    """Run a bash command and optionally capture output."""
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None


def run_bash_command_wrapper(commandstring, capture=True):
    """Run a bash command with retry logic."""
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError:
        if not capture:
            print(f"running {commandstring} one more time")
        try:
            run_bash_command(commandstring, capture)
        except subprocess.CalledProcessError:
            print("failed again!!!!")


def get_filename_myKernels():
    """Get the path to myKernels.py file."""
    # Try to determine the tools directory relative to this file
    current_file = Path(__file__)
    tools_dir = current_file.parent.parent.parent / "tools"
    if not tools_dir.exists():
        # Fallback: use current directory
        tools_dir = Path.cwd()
    return str(tools_dir / "myKernels.py")


def get_filename_without_extension(file_path):
    """Get filename without extension."""
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def get_filename_compile_driver():
    """Get the path to compile_driver.py file."""
    # Try to determine the tools directory relative to this file
    current_file = Path(__file__)
    tools_dir = current_file.parent.parent.parent / "tools"
    if not tools_dir.exists():
        # Fallback: use current directory
        tools_dir = Path.cwd()
    return str(tools_dir / "compile_driver.py")


def get_filename_profile_driver(M, N, K, job_id):
    """Get the path to profile_driver_{M}x{N}x{K}_{job_id}.py file."""
    # Try to determine the tools directory relative to this file
    current_file = Path(__file__)
    tools_dir = current_file.parent.parent.parent / "tools"
    if not tools_dir.exists():
        # Fallback: use current directory
        tools_dir = Path.cwd()
    return str(tools_dir / f"profile_driver_{M}x{N}x{K}_{job_id}.py")


def get_default_tuning_result_filename(kernel_name):
    """Generate default filename for tuning results."""
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    # handle branch name of "xxx/xxx" format
    git_branch_name = git_branch_name.replace('/', '_')
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    # Try to determine the tools directory relative to this file
    current_file = Path(__file__)
    tools_dir = current_file.parent.parent.parent / "tools"
    if not tools_dir.exists():
        # Fallback: use current directory
        tools_dir = Path.cwd()
    defaultName = str(tools_dir / f"tuning_results@{kernel_name}@{git_branch_name}@{git_commit_hash}_{dt_string}.yaml")
    return defaultName


def patch_triton_compiler():
    """Patch triton compiler to avoid backend queries (hacky workaround)."""
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    target = triton.runtime.driver.active.get_current_target()

    triton_location_str = run_bash_command("pip show triton | grep Editable")
    if not triton_location_str:
        print("triton source not found from pip show triton")
        return

    triton_dir = triton_location_str[0].split()[-1].decode('utf-8')

    jit_filename = os.path.join(triton_dir, "triton/runtime", "jit.py")

    run_bash_command(f"sed -i 's/driver.active.get_current_device()/{device}/g' {jit_filename}")
    run_bash_command(f"sed -i 's/driver.active.get_current_stream(device)/{stream}/g' {jit_filename}")

    hip_driver_filename = os.path.join(triton_dir, "../third_party/amd/backend/", "driver.py")
    cuda_driver_filename = os.path.join(triton_dir, "../third_party/nvidia/backend/", "driver.py")

    run_bash_command(f"sed -i 's/import torch/return True/g' {hip_driver_filename}")
    run_bash_command(
        f"sed -i 's/device = self.get_current_device()/return GPUTarget(\"hip\", \"{target.arch}\", 64)/g' {hip_driver_filename}"
    )
    run_bash_command(f"sed -i 's/import torch/return False/g' {cuda_driver_filename}")

import torch
import triton
import triton.language as tl

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

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


def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_output_dir():
    """Get the output directory for profiling results, creating it if needed."""
    # tools/utils/utils.py -> tools/output
    output_dir = Path(__file__).parent.parent / "output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def run_bash_command_wrapper(commandstring, capture=True):
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError:
        if not capture:
            print(f"running {commandstring} one more time")
        try:
            run_bash_command(commandstring, capture)
        except subprocess.CalledProcessError:
            print("failed again!!!!")


def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None


def get_filename_myKernels():
    """Get the path to myKernels.py file."""
    # tools/utils/utils.py -> tools/myKernels.py
    path = Path(__file__).parent.parent
    return str(path / "myKernels.py")


def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def get_filename_compile_driver():
    """Get the path to compile_driver.py file."""
    # tools/utils/utils.py -> tools/compile_driver.py
    path = Path(__file__).parent.parent
    return str(path / "compile_driver.py")


def get_filename_profile_driver(M, N, K, job_id):
    """Get the path to profile_driver_{M}x{N}x{K}_{job_id}.py file."""
    # tools/utils/utils.py -> tools/profile_driver_*.py
    path = Path(__file__).parent.parent
    return str(path / f"profile_driver_{M}x{N}x{K}_{job_id}.py")


def get_default_tuning_result_filename(kernel_name):
    """Generate default filename for tuning results."""
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    # handle branch name of "xxx/xxx" format
    git_branch_name = git_branch_name.replace('/', '_')
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    # tools/utils/utils.py -> tools/tuning_results@*.yaml
    path = Path(__file__).parent.parent
    defaultName = str(path / f"tuning_results@{kernel_name}@{git_branch_name}@{git_commit_hash}_{dt_string}.yaml")
    return defaultName


def patch_triton_compiler():
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    target = triton.runtime.driver.active.get_current_target()

    triton_location_str = run_bash_command("pip show triton | grep Editable")
    if not triton_location_str:
        print("triton source not found from pip show triton")

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

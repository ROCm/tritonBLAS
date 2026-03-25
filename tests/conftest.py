"""
Shared test configuration for tritonblas correctness tests.

This module defines common dimension lists, dtype lists, and test parameters
used across multiple correctness test files.  Import from here to keep all
test shapes in one place.
"""

import pytest
import torch


# Torch compile / dynamo settings required for compile-enabled tests.
# Applied here so they take effect before any test file is imported.
# If we don't increase this, torch will complain about too many recompilations.
torch._dynamo.config.cache_size_limit = 100000
# Also disable caches so every compile is fresh and new issues are caught.
# Note this causes a single UserWarning that notes caches are disabled.
torch._inductor.config.force_disable_caches = True
# FIXME: Inductor seems to be initializing multiple CUDA runtimes somehow in
# relation to some of triton's new features which is causing errors unrelated to
# tritonBLAS.  The error tells you to change the multiprocessing strategy to
# 'spawn' but that actually doesn't fix the issue - you have to force
# single-threaded compilation.  This needs to be fixed upstream in torch/triton.
torch._inductor.config.compile_threads = 1

# Hardware capability detection via torch (works for both CUDA and ROCm)
_gcn_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
GPU_ARCH = _gcn_arch.split(":")[0] if _gcn_arch else "unknown"
_FP4_ARCHS = {"gfx950"}

requires_fp4 = pytest.mark.skipif(
    GPU_ARCH not in _FP4_ARCHS,
    reason=f"FP4 not supported on {GPU_ARCH}",
)

# Standard test dimensions
STANDARD_DIMS = [
    (128, 256, 512),      # Medium sizes
    (256, 256, 256),      # Square
    (512, 1024, 768),     # Larger
    (768, 1024, 512),     # Backward-sensitive (StreamK aggregation, see STREAMK_BUG.md)
    (2048, 1024, 512),    # Wide output
    (1024, 2048, 512),    # Tall output
]

# Edge case dimensions (small dimensions, N < 16 cases)
EDGE_CASE_DIMS = [
    (32, 32, 32),         # Small square
    (64, 16, 128),        # Small N
    (16, 64, 128),        # Small M
    (128, 8, 256),        # N < 16
    (8, 128, 256),        # M < 16
    (12, 12, 512),        # Small M and N
    (15, 17, 512),        # Weird and small M and N
    (19, 13, 512),        # Weird and small M and N
    (128, 64, 12),        # Small K
]

# Skinny matrix dimensions (stress tests)
SKINNY_DIMS = [
    (16, 16, 4096),       # Very large K
    (32, 32, 8192),       # Large K
]

# Data types to test
DTYPES = [torch.bfloat16, torch.float16]

# Whether to test with torch.compile
USE_COMPILE = [False, True]

# Whether to enable StreamK (vs. Persistent path)
ENABLE_STREAMK = [False, True]

# Whether to enable work stealing
ENABLE_WORK_STEALING = [False, True]

# Number of trials for select tests to catch intermittent problems
MULTITRIAL_NUM_TRIALS = 1


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch.compile state between tests to prevent
    accumulated recompilation limits across parametrized cases."""
    yield
    torch._dynamo.reset()

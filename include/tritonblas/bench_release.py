import ctypes
from hip import hip, hiprtc
import math
import numpy as np
import statistics
import torch
import triton
import triton.language as tl
import sys

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


def _get_empty_cache_for_benchmark():
    cache_size = 512 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")


@triton.jit
def _clear_cache_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Force read and write to evict cache lines
    data = tl.load(ptr + offsets, mask=mask)
    tl.store(ptr + offsets, data + 1, mask=mask)


def _clear_cache(cache):
    n_elements = cache.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _clear_cache_kernel[grid](cache, n_elements, BLOCK_SIZE=1024)


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def do_bench_release(
    fn,
    numCUs_hptr,
    cus_to_release,
    reset_fn=lambda: None,
    preamble_fn=lambda: None,
    n_warmup=25,
    n_repeat=100,
    quantiles=None,
    return_mode="mean",
):
    """
    Benchmark a function by timing its execution using CUDA events.

    ``reset_fn`` is called before every invocation (warmup and timed) and is
    **not** included in the measured time.  Use it to zero mutable kernel state
    such as ``MatmulConfig.reset()``.

    ``preamble_fn`` is called once before each invocation (after reset) and is
    also **not** timed.  Use it for any one-time setup that should not be
    measured (e.g. ``matmul_preamble``).

    Args:
        fn:           Function to benchmark.
        reset_fn:     Called before each invocation to reset kernel state.
        preamble_fn:  Called before each invocation for setup.
        n_warmup:     Number of warmup iterations.
        n_repeat:     Number of timed iterations.
        quantiles:    Quantiles to return instead of a summary statistic.
        return_mode:  ``"mean"``, ``"min"``, ``"max"``, ``"median"``, or ``"all"``.

    Returns:
        float or list: Timing result(s) in milliseconds.
    """
    # Load libatomic
    libatomic = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libatomic.so.1.2.0")
    # __atomic_fetch_add_4(int *ptr, int val, int memorder)
    libatomic.__atomic_fetch_add_4.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    libatomic.__atomic_fetch_add_4.restype = ctypes.c_int

    MEMORDER_RELAXED = 0
    ptr = ctypes.cast(ctypes.byref(numCUs_hptr.contents, 0 * ctypes.sizeof(ctypes.c_int)), ctypes.POINTER(ctypes.c_int))

    # Initial sync + single run to compile / warm caches
    torch.cuda.synchronize()
    preamble_fn()
    reset_fn()
    fn()
    torch.cuda.synchronize()

    cache = _get_empty_cache_for_benchmark()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    # Warmup
    for _ in range(n_warmup):
        torch.cuda.synchronize()
        reset_fn()
        preamble_fn()
        _clear_cache(cache)
        torch.cuda.synchronize()
        fn()

    # Timed runs
    for i in range(n_repeat):
        torch.cuda.synchronize()
        reset_fn()
        preamble_fn()
        _clear_cache(cache)
        torch.cuda.synchronize()
        start_event[i].record()
        # release CUs 
        prev = libatomic.__atomic_fetch_add_4(ptr, -cus_to_release, MEMORDER_RELAXED)
        fn()
        end_event[i].record()
#        print(f"atomic done", file=sys.stderr)

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)

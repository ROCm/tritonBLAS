"""
Concurrent Triton kernels on AMD/ROCm without deprecated 'stream='.

Key points:
  * Use torch.cuda.Stream() and 'with torch.cuda.stream(s): ...' to choose streams.
  * Triton launches are async; measure with explicit synchronization.
  * AMD guidance: prefer num_stages >= 2 with the current stream pipeliner,
    and size BLOCK_SIZE/num_warps to leave headroom for overlap.

References:
  - PyTorch HIP semantics reuse torch.cuda API on AMD (streams, etc.).
  - torch.cuda.StreamContext enqueues ops on the chosen stream.
  - Triton kernels run asynchronously; torch.cuda.synchronize() is appropriate.
"""

import ctypes
import math
import sys
import time

import array
import numpy as np

import torch
from hip import hip, hiprtc

import triton
import triton.language as tl

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


# -------------------------
# Triton kernels
# -------------------------

@triton.jit
def sch(live_flags_ptr, x_ptr, n_elements,
        ITERS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    live = 1
    done = 0

    while tl.atomic_add(live_flags_ptr + pid, 0) == live:

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        for _ in range(ITERS):
            x = x * 1.000000119 + 0.000000137

        tl.store(x_ptr + offsets, x, mask=mask)

        ret = tl.inline_asm_elementwise(
            asm="""s_sleep 128""",
            constraints=("=s"),
            args=[],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )

@triton.jit
def gemm(x_ptr, n_elements,
         ITERS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 1D program index
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # simple compute loop
    for _ in range(ITERS):
        x = x * 1.000000119 + 0.000000137
    tl.store(x_ptr + offsets, x, mask=mask)


@triton.jit
def comm(x_ptr, n_elements,
         ITERS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    for _ in range(ITERS):
        x = tl.sin(x) + 1.000000119 * x + 0.000000071
    tl.store(x_ptr + offsets, x, mask=mask)


def main():
    # -------------------------
    # Tunables for AMD GPUs
    # -------------------------
    N = 32 * 1024 * 1024        # elements
    ITERS_A = 200
    ITERS_B = 200
    BLOCK_SIZE = 256            # elements per Triton "program"/CTA
    NUM_WARPS = 4               # try 4~8; AMD wavefront is 64-wide
    NUM_STAGES = 2              # AMD's current stream pipeliner favors >=2

    # Load libatomic
    libatomic = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libatomic.so.1.2.0")  # Adjust path as needed
    # Define the function signature
    # __atomic_fetch_add_4(int *ptr, int val, int memorder)
    libatomic.__atomic_fetch_add_4.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    libatomic.__atomic_fetch_add_4.restype = ctypes.c_int

    assert torch.cuda.is_available(), "Need a ROCm-enabled PyTorch build with an AMD GPU"
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    backend = "HIP" if getattr(torch.version, "hip", None) else "CUDA"
    print(f"Device: {prop.name} | backend: {backend} | total_mem: {prop.total_memory/1e9:.1f} GB")
    if backend != "HIP":
        print("Warning: This script targets AMD/ROCm, but a non-HIP backend was detected.")

    # Data
    a = torch.linspace(0, 1, N, device="cuda", dtype=torch.float32).contiguous()
    b = torch.linspace(1, 2, N, device="cuda", dtype=torch.float32).contiguous()

    sch_stream = torch.cuda.Stream()

    # Two independent streams (ROCm uses torch.cuda.* too)
    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    num_xcds = 8
    sch_grid = num_xcds * BLOCK_SIZE

    done = 0
    live = 1

    flags = array.array("I", [live for i in range(0, sch_grid)])
    # allocate a Pointer class to be passed to the GPU kernel, flags_h is a void*
    flags_h = hip_check(hip.hipHostMalloc(sch_grid * sys.getsizeof(live), 1))
    flags_h.fromObj(flags) # initialize the storage pointed by flags_h
    # casting flags_h to a typed pointer to access the pointed contents
    flags_typed_ptr = ctypes.cast(flags_h.as_c_void_p(), ctypes.POINTER(ctypes.c_int * sch_grid))
    print(f'Flags (init):')
    for i in range(0, sch_grid):
        print(f'{flags_typed_ptr.contents[i]}')

    flags_h_np_array = np.ctypeslib.as_array(flags_typed_ptr, shape=(sch_grid,))
    flags_h_tensor = torch.from_numpy(flags_h_np_array)

    print(f'Scheduler kernel started')
    with torch.cuda.stream(sch_stream):
        sch[(sch_grid, 1, 1)](flags_h_tensor, a, N, ITERS=ITERS_A, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)

    # Memory order: 0 = relaxed, 1 = consume, 2 = acquire, 3 = release, 4 = acq_rel, 5 = seq_cst
    MEMORDER_RELAXED = 0
    # Stop the scheduler kernel
    print(f'Flags (__atomic_fetch_add flags_h to signal the GPU kernel to proceed):')
    for i in range(0, sch_grid):
        ptr = ctypes.cast(ctypes.byref(flags_typed_ptr.contents, i * ctypes.sizeof(ctypes.c_int)), ctypes.POINTER(ctypes.c_int))
        prev = libatomic.__atomic_fetch_add_4(ptr, done, MEMORDER_RELAXED)
        print(f'{prev} {flags_typed_ptr.contents[i]}')

    sch_stream.synchronize()
    print(f'Scheduler kernel done')

    # Grid: one program per BLOCK_SIZE chunk
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # -------------------------
    # Warm-up (JIT & cache)
    # -------------------------
    with torch.cuda.stream(gemm_stream):
        gemm[grid](a, N, ITERS=ITERS_A, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    with torch.cuda.stream(comm_stream):
        comm[grid](b, N, ITERS=ITERS_B, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    # Triton is async; explicit sync for clean timing
    gemm_stream.synchronize(); comm_stream.synchronize()
    print("Warm-up complete.\n")

    # -------------------------
    # Sequential timing (A then B)
    # -------------------------
    t0 = time.perf_counter()
    with torch.cuda.stream(gemm_stream):
        gemm[grid](a, N, ITERS=ITERS_A, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    gemm_stream.synchronize()
    with torch.cuda.stream(comm_stream):
        comm[grid](b, N, ITERS=ITERS_B, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    comm_stream.synchronize()
    t_seq = time.perf_counter() - t0
    print(f"Sequential total time: {t_seq:.3f} s")

    # -------------------------
    # Concurrent timing (A || B)
    # -------------------------
    t0 = time.perf_counter()
    with torch.cuda.stream(gemm_stream):
        gemm[grid](a, N, ITERS=ITERS_A, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    with torch.cuda.stream(comm_stream):
        comm[grid](b, N, ITERS=ITERS_B, BLOCK_SIZE=BLOCK_SIZE,
                       num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    # wait for both
    gemm_stream.synchronize(); comm_stream.synchronize()
    t_conc = time.perf_counter() - t0
    print(f"Concurrent total time: {t_conc:.3f} s")

    # Check a couple of results
    print("\nResults (samples):")
    print(f"A[123456] = {a[123_456].item():.6f}")
    print(f"B[234567] = {b[234_567].item():.6f}")

    print("\nTip: If there's little or no overlap, reduce NUM_WARPS or BLOCK_SIZE "
          "to leave headroom for both kernels to co-reside on the GPU. "
          "On AMD, keep num_stages>=2 for the current stream pipeliner.")


if __name__ == "__main__":
    main()

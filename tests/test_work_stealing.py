"""
Standalone smoke-test for the work-stealing persistent GEMM kernel.

Directly imports the work-stealing kernel to avoid the stages/streamk import
chain that requires a newer Triton with `constexpr_function`.
"""

import os
import sys
import time
import types
import importlib.util
import torch
import triton

# ---------------------------------------------------------------------------
# Bootstrap: load only the pieces the work-stealing kernel needs, bypassing
# the full tritonblas package init (which pulls in stages/__init__.py that
# requires triton.constexpr_function not available in this Triton build).
# ---------------------------------------------------------------------------
_kernels_dir = os.path.join(
    os.path.dirname(__file__), "..", "include", "tritonblas", "kernels",
)
_stages_dir = os.path.join(_kernels_dir, "stages")
_indexing_dir = os.path.join(_stages_dir, "indexing")


def _load_module(fqn, filepath, package_path=None):
    """Load a single .py file and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    if package_path is not None:
        mod.__path__ = [package_path]
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_stub_package(fqn, path):
    """Register a stub package so relative imports can traverse it."""
    pkg = types.ModuleType(fqn)
    pkg.__path__ = [path]
    pkg.__package__ = fqn
    sys.modules[fqn] = pkg
    return pkg


# Stub packages (just enough for the relative import chain)
_make_stub_package("tritonblas", os.path.join(_kernels_dir, ".."))
_make_stub_package("tritonblas.kernels", _kernels_dir)
_make_stub_package("tritonblas.kernels.stages", _stages_dir)
_make_stub_package("tritonblas.kernels.stages.indexing", _indexing_dir)

# Load pid_transforms (pure @triton.jit, no constexpr_function dependency)
_load_module(
    "tritonblas.kernels.stages.indexing.pid_transforms",
    os.path.join(_indexing_dir, "pid_transforms.py"),
)

# Now load the work-stealing kernel — its relative import will resolve
_ws_mod = _load_module(
    "tritonblas.kernels.persistent_gemm_work_stealing",
    os.path.join(_kernels_dir, "persistent_gemm_work_stealing.py"),
)
ws_persistent_matmul = _ws_mod.ws_persistent_matmul


def make_tile_counter(device="cuda"):
    """Allocate a fresh work-stealing tile counter."""
    return torch.zeros(1, device=device, dtype=torch.int32)


def run_ws_persistent_matmul(A, B, C, tile_counter, BLK_M=128, BLK_N=128, BLK_K=64, GROUP_M=8):
    """Launch the work-stealing persistent kernel."""
    M, K = A.shape
    _, N = B.shape

    props = torch.cuda.get_device_properties(A.device)
    NUM_SMS = props.multi_processor_count

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    NUM_XCDS = 8
    chunk_size = GROUP_M * GROUP_M
    chunk_size = min(chunk_size, max(1, total_tiles // NUM_XCDS))

    # Grid = number of CUs (work-stealing)
    grids = NUM_SMS

    # Reset counter
    tile_counter.zero_()

    ws_persistent_matmul[(grids,)](
        A, B, C,
        None,  # A_scale_ptr
        None,  # B_scale_ptr
        None,  # bias_ptr
        tile_counter,
        M, N, K,
        A.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        0,     # bias stride
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=GROUP_M,
        NUM_SMS=grids,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=None,
        CACHE_MODIFIER_B=None,
        QUANTIZED=False,
        num_stages=2,
        num_warps=8,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )


def test_correctness(m, n, k, dtype=torch.float16):
    """Run a single test and compare against torch.matmul."""
    A = torch.randn(m, k, device="cuda", dtype=dtype)
    B = torch.randn(n, k, device="cuda", dtype=dtype).T
    C = torch.zeros(m, n, device="cuda", dtype=dtype)

    ref = torch.matmul(A, B)

    tile_counter = make_tile_counter()
    run_ws_persistent_matmul(A, B, C, tile_counter)
    torch.cuda.synchronize()

    max_diff = (C - ref).abs().max().item()
    mean_diff = (C - ref).abs().mean().item()
    passed = torch.allclose(C, ref, atol=1e-1, rtol=1e-2)

    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {m:>5}x{n:<5}x{k:<5}  "
        f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}"
    )
    return passed


def bench_throughput(m, n, k, dtype=torch.float16, warmup=5, iters=20):
    """Quick throughput benchmark."""
    A = torch.randn(m, k, device="cuda", dtype=dtype)
    B = torch.randn(n, k, device="cuda", dtype=dtype).T
    C = torch.zeros(m, n, device="cuda", dtype=dtype)
    tile_counter = make_tile_counter()

    # Warmup
    for _ in range(warmup):
        run_ws_persistent_matmul(A, B, C, tile_counter)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        run_ws_persistent_matmul(A, B, C, tile_counter)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000
    flops = 2.0 * m * n * k
    tflops = (flops / (avg_ms / 1000)) / 1e12
    print(f"  {m:>5}x{n:<5}x{k:<5}  avg={avg_ms:7.3f} ms  {tflops:6.2f} TFLOP/s")
    return avg_ms


def main():
    torch.manual_seed(42)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}  (CUs: {props.multi_processor_count})")
    print(f"HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', '<not set>')}")
    print()

    # ── Correctness ───────────────────────────────────────────────────
    print("=" * 68)
    print("Correctness (work-stealing kernel vs torch.matmul)")
    print("=" * 68)
    all_pass = True
    for m, n, k in [
        (256,  256,  256),
        (512,  512,  512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        try:
            ok = test_correctness(m, n, k)
            all_pass &= ok
        except Exception as e:
            print(f"  [ERROR] {m}x{n}x{k}: {e}")
            import traceback; traceback.print_exc()
            all_pass = False

    # ── Throughput ────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("Throughput (work-stealing kernel)")
    print("=" * 68)
    for m, n, k in [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        try:
            bench_throughput(m, n, k)
        except Exception as e:
            print(f"  [ERROR] {m}x{n}x{k}: {e}")
            import traceback; traceback.print_exc()

    print()
    if all_pass:
        print("All correctness tests PASSED.")
    else:
        print("Some correctness tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

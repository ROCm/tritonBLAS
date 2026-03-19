#!/usr/bin/env python3
"""
Sweep WS GEMM over different grid sizes (number of WGs) on a single GPU.
This isolates the CU-count effect from any RCCL interference.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


def main():
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16

    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    C = torch.empty(M, N, dtype=dtype, device=dev)

    selector = tritonblas.OrigamiMatmulSelector(M, N, K, dtype, dtype, dtype, dev, streamk=False)
    cfg = tritonblas.matmul_preamble(selector)

    N_CU = selector._hardware.N_CU  # 304
    print(f"Hardware CUs: {N_CU}")
    BLK_M, BLK_N, BLK_K = selector.block_m, selector.block_n, selector.block_k
    print(f"Tile size: {BLK_M}x{BLK_N}x{BLK_K}")

    num_pid_m = (M + BLK_M - 1) // BLK_M
    num_pid_n = (N + BLK_N - 1) // BLK_N
    total_tiles = num_pid_m * num_pid_n
    print(f"Total tiles: {total_tiles}")
    print()

    # Warmup at default grid size
    for _ in range(10):
        cfg.reset(streamk=False, work_stealing=True)
        tritonblas.matmul_lt(A, B, C, selector, cfg, enable_streamk=False, work_stealing=True)
    torch.cuda.synchronize()

    print(f"{'Grid (WGs)':<12s} {'Median (ms)':<12s} {'Slowdown':<10s}")
    print("-" * 36)

    # Baseline: full CU count
    base = None
    for grid_size in [304, 296, 288, 280, 272, 264, 256, 240, 224, 200, 176, 152, 128]:
        # Temporarily override the grid size in matmul.py
        # We need to monkey-patch the selector or pass it differently
        # The simplest approach: modify grids in the persistent_matmul_lt function
        # Let's just re-call the kernel directly
        from tritonblas.matmul import persistent_matmul_lt

        # Save original N_CU and temporarily override
        orig_n_cu = selector._hardware.N_CU
        selector._hardware.N_CU = grid_size

        n_steps = 50
        times = []
        for _ in range(n_steps):
            cfg.reset(streamk=False, work_stealing=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            tritonblas.matmul_lt(A, B, C, selector, cfg, enable_streamk=False, work_stealing=True)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

        selector._hardware.N_CU = orig_n_cu

        import statistics
        med = statistics.median(times)
        if base is None:
            base = med
        slow = med / base
        print(f"{grid_size:<12d} {med:<12.3f} {slow:<10.2f}x")


if __name__ == "__main__":
    main()

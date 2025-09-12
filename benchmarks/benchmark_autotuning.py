#!/usr/bin/env python3
import os

# Ensure Triton prints autotuning info
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import time
import torch
import triton
import triton.language as tl
from triton import Config, autotune
import tritonblas

# ------------- Kernel Definition -------------


@autotune(
    configs=[],  # will overwrite per-size below
    key=["M", "N", "K"],
    warmup=10,
    rep=50,
)
@triton.jit
def persistent_matmul(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    Adesc: tl.constexpr,
    Bdesc: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        xcd = pid % NUM_XCDS
        pos_in_xcd = pid // NUM_XCDS
        min_per_xcd = NUM_SMS // NUM_XCDS
        extra_sms = NUM_SMS % NUM_XCDS
        offset = xcd * min_per_xcd + min(xcd, extra_sms)
        pid = offset + pos_in_xcd

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # compute rm, rn, rk, load A/B, dot-product, accumulate
        rm = ((tile_id % num_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = ((tile_id // num_pid_m) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        loop_k = tl.cdiv(K, BLOCK_SIZE_K) - (0 if EVEN_K else 1)
        for _ in range(loop_k):
            a = tl.load(
                tl.multiple_of(A_BASE, (1, 16))
                if Adesc
                else tl.multiple_of(A_BASE, (16, 1))
            )
            b = tl.load(
                tl.multiple_of(B_BASE, (1, 16))
                if Bdesc
                else tl.multiple_of(B_BASE, (16, 1))
            )
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk
        if not EVEN_K:
            rk2 = (loop_k * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
            A2 = tl.multiple_of(
                A + rm[:, None] * stride_am + rk2[None, :] * stride_ak,
                (1, 16) if Adesc else (16, 1),
            )
            B2 = tl.multiple_of(
                B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn,
                (1, 16) if Bdesc else (16, 1),
            )
            acc += tl.dot(
                tl.load(A2, mask=rk2[None, :] < K, other=0.0),
                tl.load(B2, mask=rk2[:, None] < K, other=0.0),
            )
        c = acc.to(C.type.element_ty)
        rm2 = ((tile_id % num_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn2 = ((tile_id // num_pid_m) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        mask = (rm2[:, None] < M) & (rn2[None, :] < N)
        C_BASE = C + rm2[:, None] * stride_cm + rn2[None, :] * stride_cn
        tl.store(C_BASE, c, mask)


# ------------- Benchmark Driver -------------

if __name__ == "__main__":
    # Problem sizes to test
    SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # Pre-allocate max-size tensors and reuse slices
    max_S = max(SIZES)
    A_full = torch.randn((max_S, max_S), device="cuda", dtype=torch.float16)
    B_full = torch.randn((max_S, max_S), device="cuda", dtype=torch.float16)
    C_full = torch.zeros((max_S, max_S), device="cuda", dtype=torch.float16)

    # 0) Initial JIT run for 1024 to exclude compile time from measurements
    S0 = 64
    A0 = A_full[:S0, :S0].clone()
    B0 = B_full[:S0, :S0].clone()
    C0 = C_full[:S0, :S0].clone()
    strides0 = (
        A0.stride(0),
        A0.stride(1),
        B0.stride(0),
        B0.stride(1),
        C0.stride(0),
        C0.stride(1),
        0,
    )
    # Build configs for S0
    full_list0 = tritonblas.MatmulHeuristicResult(S0, S0, S0)._get_valid_tiles()
    unique0 = {(m, n, k) for (m, n, k, *_) in full_list0}
    tiles0 = sorted(unique0)[::-1]
    configs0 = [
        Config(
            {
                "BLOCK_SIZE_M": m,
                "BLOCK_SIZE_N": n,
                "BLOCK_SIZE_K": k,
                "GROUP_SIZE_M": 1,
                "NUM_SMS": 304,
                "NUM_XCDS": 8,
                "BIAS": False,
                "EVEN_K": True,
            },
            num_warps=8,
            num_stages=2,
        )
        for m, n, k in tiles0
    ]
    persistent_matmul.configs = configs0
    grid0 = (triton.cdiv(S0, tiles0[0][0]) * triton.cdiv(S0, tiles0[0][1]),)

    torch.cuda.synchronize()
    t_jit0 = time.perf_counter()
    persistent_matmul[grid0](A0, B0, C0, None, S0, S0, S0, *strides0, True, False)
    torch.cuda.synchronize()
    t_jit1 = time.perf_counter()
    jit_time = t_jit1 - t_jit0

    # 1) Now measure autotune overhead for each size
    results = []
    for S in SIZES:
        print("====================")
        print(f"{S}")
        print("====================")

        A = A_full[:S, :S].clone()
        B = B_full[:S, :S].clone()
        C = C_full[:S, :S].clone()
        strides = (
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            0,
        )

        # get tiles & configs
        fl = tritonblas.MatmulHeuristicResult(S, S, S)._get_valid_tiles()
        uniq = {(m, n, k) for (m, n, k, *_) in fl}
        tiles = sorted(uniq)[::-1]
        num_tiles = len(tiles)
        configs = [
            Config(
                {
                    "BLOCK_SIZE_M": m,
                    "BLOCK_SIZE_N": n,
                    "BLOCK_SIZE_K": k,
                    "GROUP_SIZE_M": 1,
                    "NUM_SMS": 304,
                    "NUM_XCDS": 8,
                    "BIAS": False,
                    "EVEN_K": True,
                },
                num_warps=8,
                num_stages=2,
            )
            for m, n, k in tiles
        ]
        persistent_matmul.configs = configs
        grid = (triton.cdiv(S, tiles[0][0]) * triton.cdiv(S, tiles[0][1]),)

        # first launch: autotune + run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        persistent_matmul[grid](A, B, C, None, S, S, S, *strides, True, False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total = t1 - t0

        # second launch: run-only
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        persistent_matmul[grid](A, B, C, None, S, S, S, *strides, True, False)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        run_only = t3 - t2

        overhead = total - run_only
        results.append((S, num_tiles, overhead))

    # 2) Print results
    print(f"Initial JIT + first-run (1024³): {jit_time:.3f} s\n")
    print("| Problem Size (M×N×K) | # of Tile Sizes | Triton Autotuning (s) |")
    print("|-----------------------|-----------------|-----------------------|")
    for S, num_tiles, ov in results:
        print(f"| {S}×{S}×{S}           | {num_tiles:<15} | {ov:>7.3f}              |")

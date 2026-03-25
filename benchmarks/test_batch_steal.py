#!/usr/bin/env python3
"""Test batch tile stealing: grab N tiles per atomic instead of 1."""
import torch
import triton
import triton.language as tl
import tritonblas
from tritonblas.config import COUNTER_STRIDE, matmul_preamble
import statistics

torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
s = torch.cuda.Stream(device=dev)
M = N = K = 8192
FLOPS = 2.0 * M * N * K

A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
ref = torch.matmul(A.float(), B.float()).bfloat16()

for _ in range(20):
    torch.matmul(A, B, out=C)
torch.cuda.synchronize()
t = []
for _ in range(100):
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    torch.matmul(A, B, out=C)
    en.record()
    torch.cuda.synchronize()
    t.append(st.elapsed_time(en))
torch_ms = statistics.median(t)
print(f"torch: {torch_ms:.3f}ms ({FLOPS/(torch_ms*1e-3)/1e12:.0f}TF)")


@triton.jit()
def ws_batch_matmul(
    A, B, C, tile_counter, M, N, K,
    stride_am, stride_bn, stride_cm, stride_cn,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
    COUNTERS_PER_XCD: tl.constexpr, COUNTER_STRIDE: tl.constexpr,
    EVEN_K: tl.constexpr, BATCH_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    xcd_id = pid % NUM_XCDS
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tiles_per_xcd = tl.cdiv(total_tiles, NUM_XCDS)

    local_wg_id = pid // NUM_XCDS
    slot = local_wg_id % COUNTERS_PER_XCD
    xcd_base = xcd_id * tiles_per_xcd
    xcd_end = tl.minimum(xcd_base + tiles_per_xcd, total_tiles)
    tiles_this_xcd = xcd_end - xcd_base
    tiles_per_slot = tl.cdiv(tiles_this_xcd, COUNTERS_PER_XCD)
    slot_base = slot * tiles_per_slot
    slot_end = tl.minimum(slot_base + tiles_per_slot, tiles_this_xcd)
    bound = slot_end - slot_base
    counter_ptr = tile_counter + (xcd_id * COUNTERS_PER_XCD + slot) * COUNTER_STRIDE

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)

    raw_idx_base = tl.atomic_add(counter_ptr, BATCH_SIZE, scope="gpu")

    while raw_idx_base < bound:
        for b in tl.static_range(BATCH_SIZE):
            raw_idx = raw_idx_base + b
            if raw_idx >= bound:
                break

            tile_id = xcd_base + slot_base + raw_idx
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            rk = tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, loop_k):
                if stride_ak == 1:
                    a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                else:
                    a = tl.load(tl.multiple_of(A_BASE, (16, 1)))
                if stride_bk == 1:
                    b_tile = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                else:
                    b_tile = tl.load(tl.multiple_of(B_BASE, (1, 16)))
                acc += tl.dot(a, b_tile)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            c = acc.to(C.type.element_ty)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(C_, c, mask=(rm[:, None] < M) & (rn[None, :] < N))

        raw_idx_base = tl.atomic_add(counter_ptr, BATCH_SIZE, scope="gpu")


sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
cfg = matmul_preamble(sel)
grids = sel._hardware.N_CU

for batch_size in [1, 2, 3, 4]:
    try:
        for _ in range(10):
            with torch.cuda.stream(s):
                cfg.reset(work_stealing=True)
                ws_batch_matmul[(grids,)](
                    A, B, C, cfg.tile_counter, M, N, K,
                    A.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    stride_ak=A.stride(1), stride_bk=B.stride(0),
                    BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=4, NUM_SMS=grids, NUM_XCDS=sel.num_sms,
                    COUNTERS_PER_XCD=4, COUNTER_STRIDE=COUNTER_STRIDE,
                    EVEN_K=True, BATCH_SIZE=batch_size,
                    num_stages=2, num_warps=8, waves_per_eu=0,
                    matrix_instr_nonkdim=16, kpack=1,
                )
        torch.cuda.synchronize()

        cfg.reset(work_stealing=True)
        ws_batch_matmul[(grids,)](
            A, B, C, cfg.tile_counter, M, N, K,
            A.stride(0), B.stride(1), C.stride(0), C.stride(1),
            stride_ak=A.stride(1), stride_bk=B.stride(0),
            BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=4, NUM_SMS=grids, NUM_XCDS=sel.num_sms,
            COUNTERS_PER_XCD=4, COUNTER_STRIDE=COUNTER_STRIDE,
            EVEN_K=True, BATCH_SIZE=batch_size,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1,
        )
        torch.cuda.synchronize()
        err = (C.float() - ref.float()).abs().max().item()

        t = []
        for _ in range(50):
            with torch.cuda.stream(s):
                cfg.reset(work_stealing=True)
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(s)
            with torch.cuda.stream(s):
                ws_batch_matmul[(grids,)](
                    A, B, C, cfg.tile_counter, M, N, K,
                    A.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    stride_ak=A.stride(1), stride_bk=B.stride(0),
                    BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=4, NUM_SMS=grids, NUM_XCDS=sel.num_sms,
                    COUNTERS_PER_XCD=4, COUNTER_STRIDE=COUNTER_STRIDE,
                    EVEN_K=True, BATCH_SIZE=batch_size,
                    num_stages=2, num_warps=8, waves_per_eu=0,
                    matrix_instr_nonkdim=16, kpack=1,
                )
            en.record(s)
            torch.cuda.synchronize()
            t.append(st.elapsed_time(en))

        med = statistics.median(t)
        pct = (med / torch_ms - 1) * 100
        ok = "OK" if err < 10 else "BAD"
        beat = " ***" if med < torch_ms else ""
        print(f"  batch={batch_size}: {med:.3f}ms ({FLOPS/(med*1e-3)/1e12:.0f}TF) "
              f"{pct:+.1f}% err={ok}{beat}")
    except Exception as e:
        print(f"  batch={batch_size}: FAIL {str(e)[:80]}")

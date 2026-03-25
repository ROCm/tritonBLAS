#!/usr/bin/env python3
"""
RCCL CU sweep: measure GEMM+RCCL overlap as RCCL claims more CUs.

Worker script — launched via torchrun for each NCCL_MAX_NCHANNELS value.
Outputs a single JSON line to stdout (rank 0 only).

Usage (called by wrapper):
  HSA_NO_SCRATCH_RECLAIM=1 NCCL_MAX_NCHANNELS=X torchrun --nproc_per_node=8 \
      benchmarks/rccl_cu_sweep.py --size 8192 --comm_size 16384 --channels X
"""
import argparse, json, math, os, statistics, sys
import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

N_CU = 304
NUM_XCDS = 8
BLK_M = 256
BLK_N = 256
BLK_K = 64
GROUP_SIZE_M = 8
COUNTER_STRIDE = 64
WARMUP = 15
STEPS = 60
dtype = torch.bfloat16


def hierarchical_split(M, N):
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    tiles_per_cu = total_tiles / max(N_CU, 1)
    local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
    local_per_xcd = int(total_tiles * local_frac) // NUM_XCDS
    local_per_xcd = max(local_per_xcd, 1)
    global_tiles = total_tiles - local_per_xcd * NUM_XCDS
    return local_per_xcd, global_tiles


def make_ws(A, B, C):
    M, K = A.shape; _, N = B.shape; dev = A.device
    even_k = K % BLK_K == 0
    lp, gt = hierarchical_split(M, N)
    tc = torch.zeros(NUM_XCDS * COUNTER_STRIDE, device=dev, dtype=torch.int32)
    gc = torch.zeros(COUNTER_STRIDE, device=dev, dtype=torch.int32)
    mask = torch.ones(N_CU, dtype=torch.int32, device=dev)

    def fn():
        ws_hierarchical_matmul[(N_CU,)](
            A, B, C, None, None, None, tc, gc,
            M, N, K, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
            stride_ak=A.stride(1), stride_bk=B.stride(0),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=N_CU, NUM_XCDS=NUM_XCDS,
            LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1,
            USE_MASK=True, mask_ptr=mask,
        )
    def reset(): tc.zero_(); gc.zero_()
    return fn, reset


def bench_alone(fn, rfn, stream):
    for _ in range(WARMUP):
        if rfn: rfn()
        with torch.cuda.stream(stream): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(STEPS):
        if rfn: rfn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(stream)
        with torch.cuda.stream(stream): fn()
        e.record(stream)
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench_overlap(fn, rfn, comm_fn, mm_stream, co_stream):
    for _ in range(WARMUP):
        if rfn: rfn()
        co_stream.wait_stream(mm_stream)
        with torch.cuda.stream(co_stream): comm_fn()
        mm_stream.wait_stream(co_stream)
        with torch.cuda.stream(mm_stream): fn()
    torch.cuda.synchronize()

    walls, gemm_times, comm_times = [], [], []
    for _ in range(STEPS):
        if rfn: rfn()
        torch.cuda.synchronize()
        sw = torch.cuda.Event(enable_timing=True)
        ew = torch.cuda.Event(enable_timing=True)
        sm = torch.cuda.Event(enable_timing=True)
        em = torch.cuda.Event(enable_timing=True)
        sc = torch.cuda.Event(enable_timing=True)
        ec = torch.cuda.Event(enable_timing=True)
        sw.record()
        sc.record(co_stream)
        with torch.cuda.stream(co_stream): comm_fn()
        ec.record(co_stream)
        sm.record(mm_stream)
        with torch.cuda.stream(mm_stream): fn()
        em.record(mm_stream)
        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)
        ew.record()
        torch.cuda.synchronize()
        walls.append(sw.elapsed_time(ew))
        gemm_times.append(sm.elapsed_time(em))
        comm_times.append(sc.elapsed_time(ec))
    return statistics.median(walls), statistics.median(gemm_times), statistics.median(comm_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--comm_size", type=int, default=16384)
    parser.add_argument("--channels", type=int, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)

    M = N = K = args.size
    comm_buf = torch.randn(args.comm_size, args.comm_size, dtype=dtype, device=dev)
    def comm_fn(): dist.all_reduce(comm_buf)

    # 1. Comm alone
    comm_alone = bench_alone(comm_fn, None, co_stream)

    # 2. torch.matmul
    tA = torch.randn(M, K, dtype=dtype, device=dev)
    tB = torch.randn(K, N, dtype=dtype, device=dev)
    torch_fn = lambda: torch.matmul(tA, tB)
    torch_alone = bench_alone(torch_fn, None, mm_stream)
    torch_wall, torch_gem_ov, torch_com_ov = bench_overlap(
        torch_fn, None, comm_fn, mm_stream, co_stream)

    # 3. WS Hierarchical
    wA = torch.randn(M, K, dtype=dtype, device=dev)
    wB = torch.randn(K, N, dtype=dtype, device=dev)
    wC = torch.zeros(M, N, dtype=dtype, device=dev)
    ws_fn, ws_rfn = make_ws(wA, wB, wC)
    ws_alone = bench_alone(ws_fn, ws_rfn, mm_stream)
    ws_wall, ws_gem_ov, ws_com_ov = bench_overlap(
        ws_fn, ws_rfn, comm_fn, mm_stream, co_stream)

    if rank == 0:
        result = {
            "channels": args.channels,
            "size": args.size,
            "comm_alone_ms": round(comm_alone, 4),
            "torch_alone_ms": round(torch_alone, 4),
            "torch_wall_ms": round(torch_wall, 4),
            "torch_gemm_overlap_ms": round(torch_gem_ov, 4),
            "torch_comm_overlap_ms": round(torch_com_ov, 4),
            "ws_alone_ms": round(ws_alone, 4),
            "ws_wall_ms": round(ws_wall, 4),
            "ws_gemm_overlap_ms": round(ws_gem_ov, 4),
            "ws_comm_overlap_ms": round(ws_com_ov, 4),
        }
        print("RESULT:" + json.dumps(result), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

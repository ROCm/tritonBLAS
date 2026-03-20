#!/usr/bin/env python3
"""Collect data for overlap analysis plots. Run with torchrun --nproc_per_node=8.

Usage:
    torchrun --nproc_per_node=8 benchmarks/collect_all_plot_data.py
"""
import torch
import torch.distributed as dist
import tritonblas
import statistics
import json
import os
import sys

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
dtype = torch.bfloat16
s = torch.cuda.Stream(device=dev)
cs = torch.cuda.Stream(device=dev)

OUT_DIR = "results/plot_data"
if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)


def bench_alone(fn, reset_fn, warmup=20, steps=100):
    for _ in range(warmup):
        with torch.cuda.stream(s):
            if reset_fn:
                reset_fn()
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        if reset_fn:
            with torch.cuda.stream(s):
                reset_fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(s)
        with torch.cuda.stream(s):
            fn()
        en.record(s)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


def bench_overlap(fn, reset_fn, comm_fn, warmup=20, steps=100):
    for _ in range(warmup):
        with torch.cuda.stream(s):
            if reset_fn:
                reset_fn()
        with torch.cuda.stream(cs):
            comm_fn()
        with torch.cuda.stream(s):
            torch.cuda._sleep(100_000)
            fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        if reset_fn:
            with torch.cuda.stream(s):
                reset_fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(cs):
            comm_fn()
        st.record(s)
        with torch.cuda.stream(s):
            torch.cuda._sleep(100_000)
            fn()
        en.record(s)
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return times


def make_backend(M, N, K, backend):
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    C = torch.empty(M, N, dtype=dtype, device=dev)
    info = {}
    if backend == "ws":
        sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
        cfg = tritonblas.matmul_preamble(sel)
        info = {"grid": sel._hardware.N_CU,
                "tiles": (M // sel.block_m) * (N // sel.block_n),
                "block": f"{sel.block_m}x{sel.block_n}x{sel.block_k}"}
        def fn():
            tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
        def rst():
            cfg.reset(work_stealing=True)
        return fn, rst, info, [A, B, C, sel, cfg]
    else:
        def fn():
            torch.matmul(A, B, out=C)
        return fn, None, info, [A, B, C]


# ===================================================================
# 1. Multi-size: alone + overlap for ws and torch
# ===================================================================
sizes = [1024, 2048, 4096, 8192, 12288, 16384]
all_data = {}

for sz in sizes:
    M = N = K = sz
    flops = 2.0 * M * N * K
    comm_t = torch.randn(sz, sz, dtype=dtype, device=dev)
    def comm_fn(c=comm_t):
        dist.all_reduce(c, op=dist.ReduceOp.SUM)

    for bk in ["ws", "torch"]:
        fn, rst, info, refs = make_backend(M, N, K, bk)

        alone_times = bench_alone(fn, rst, warmup=20, steps=100)
        ovl_times = bench_overlap(fn, rst, comm_fn, warmup=20, steps=100)

        # Rotating
        N_BUFS = 4
        rot_refs = []
        rot_fns = []
        for _ in range(N_BUFS):
            rfn, rrst, _, rr = make_backend(M, N, K, bk)
            rot_fns.append((rfn, rrst))
            rot_refs.append(rr)

        for j in range(max(20, N_BUFS)):
            idx = j % N_BUFS
            with torch.cuda.stream(s):
                if rot_fns[idx][1]:
                    rot_fns[idx][1]()
                rot_fns[idx][0]()
        torch.cuda.synchronize()

        rot_alone_times = []
        for i in range(100):
            idx = i % N_BUFS
            if rot_fns[idx][1]:
                with torch.cuda.stream(s):
                    rot_fns[idx][1]()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(s)
            with torch.cuda.stream(s):
                rot_fns[idx][0]()
            en.record(s)
            torch.cuda.synchronize()
            rot_alone_times.append(st.elapsed_time(en))

        rot_ovl_times = []
        for i in range(100):
            idx = i % N_BUFS
            if rot_fns[idx][1]:
                with torch.cuda.stream(s):
                    rot_fns[idx][1]()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(cs):
                comm_fn()
            st.record(s)
            with torch.cuda.stream(s):
                torch.cuda._sleep(100_000)
                rot_fns[idx][0]()
            en.record(s)
            torch.cuda.synchronize()
            rot_ovl_times.append(st.elapsed_time(en))

        key = f"{bk}_{sz}"
        entry = {
            "size": sz, "backend": bk, "flops": flops, **info,
            "alone_warm": alone_times,
            "alone_rotating": rot_alone_times,
            "overlap_warm": ovl_times,
            "overlap_rotating": rot_ovl_times,
        }
        all_data[key] = entry

        if rank == 0:
            aw = statistics.median(alone_times)
            ar = statistics.median(rot_alone_times)
            ow = statistics.median(ovl_times)
            orr = statistics.median(rot_ovl_times)
            tf_a = flops/(aw*1e-3)/1e12
            tf_o = flops/(ow*1e-3)/1e12
            penalty = (orr/ar - 1)*100
            print(f"  {bk:>5} {sz:>6}: alone={aw:.3f}ms({tf_a:.0f}TF)  "
                  f"rot={ar:.3f}  ovl={ow:.3f}({tf_o:.0f}TF)  "
                  f"ovl_rot={orr:.3f}  penalty={penalty:+.1f}%")

        del refs, rot_refs, rot_fns, fn, rst
        torch.cuda.empty_cache()

    del comm_t
    torch.cuda.empty_cache()

if rank == 0:
    with open(f"{OUT_DIR}/full_multisize.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved {OUT_DIR}/full_multisize.json")

dist.destroy_process_group()

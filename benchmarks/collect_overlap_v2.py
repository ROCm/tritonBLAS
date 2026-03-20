#!/usr/bin/env python3
"""Collect overlap data with per-size optimal COUNTERS_PER_XCD.
Run with: torchrun --nproc_per_node=8 benchmarks/collect_overlap_v2.py
"""
import torch
import torch.distributed as dist
import tritonblas
import statistics
import json
import os

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
s = torch.cuda.Stream(device=dev)
cs = torch.cuda.Stream(device=dev)

OPT_CPX = {1024: 16, 2048: 8, 4096: 16, 8192: 4, 12288: 1, 16384: 2}
SIZES = [1024, 2048, 4096, 8192, 12288, 16384]
results = {}

for sz in SIZES:
    M = N = K = sz
    FLOPS = 2.0 * M * N * K
    comm = torch.randn(sz, sz, dtype=torch.bfloat16, device=dev)

    def comm_fn(c=comm):
        dist.all_reduce(c)

    for bk in ["ws", "torch"]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

        if bk == "ws":
            sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
            sel.COUNTERS_PER_XCD = OPT_CPX[sz]
            cfg = tritonblas.matmul_preamble(sel)
            tiles = (M // sel.block_m) * (N // sel.block_n)
            grid = sel._hardware.N_CU

            def fn(a=A, b=B, c=C, s_=sel, cf=cfg):
                tritonblas.matmul_lt(a, b, c, s_, cf, work_stealing=True)

            def rst(cf=cfg):
                cf.reset(work_stealing=True)
        else:
            def fn(a=A, b=B, c=C):
                torch.matmul(a, b, out=c)
            rst = None
            tiles = grid = None

        for _ in range(15):
            with torch.cuda.stream(s):
                if rst:
                    rst()
                fn()
        torch.cuda.synchronize()

        alone = []
        for _ in range(100):
            if rst:
                with torch.cuda.stream(s):
                    rst()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record(s)
            with torch.cuda.stream(s):
                fn()
            en.record(s)
            torch.cuda.synchronize()
            alone.append(st.elapsed_time(en))

        for _ in range(15):
            if rst:
                with torch.cuda.stream(s):
                    rst()
            with torch.cuda.stream(cs):
                comm_fn()
            with torch.cuda.stream(s):
                torch.cuda._sleep(100_000)
                fn()
            torch.cuda.synchronize()

        ovl = []
        for _ in range(100):
            if rst:
                with torch.cuda.stream(s):
                    rst()
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
            ovl.append(st.elapsed_time(en))

        key = f"{bk}_{sz}"
        results[key] = {
            "size": sz, "backend": bk,
            "alone_median": statistics.median(alone),
            "overlap_median": statistics.median(ovl),
            "alone_all": alone, "overlap_all": ovl,
            "grid": grid, "tiles": tiles,
            "cpx": OPT_CPX.get(sz),
        }
        if rank == 0:
            am = statistics.median(alone)
            om = statistics.median(ovl)
            penalty = (om / am - 1) * 100
            tf_a = FLOPS / (am * 1e-3) / 1e12
            tf_o = FLOPS / (om * 1e-3) / 1e12
            print(f"  {bk:>5} {sz:>6}: alone={am:.3f}ms "
                  f"({tf_a:.0f}TF) ovl={om:.3f}ms ({tf_o:.0f}TF) "
                  f"penalty={penalty:+.1f}%")

        del A, B, C
        torch.cuda.empty_cache()
    del comm
    torch.cuda.empty_cache()

if rank == 0:
    with open("results/plot_data/overlap_optimal_cpx.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved overlap_optimal_cpx.json")

dist.destroy_process_group()

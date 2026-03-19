#!/usr/bin/env python3
"""
SE Oversubscription Benchmark (v2)

Demonstrates the Shader Engine (SE) oversubscription problem on MI300X when
concurrent dispatches from different HW pipes have non-SE-aligned workgroup
counts.

MI300X has 8 XCCs x 4 SEs = 32 SEs (304 CUs, asymmetric harvesting).
When GEMM and RCCL run concurrently on different streams, each pipe maintains
its own SE rotation pointer.  If either dispatch's WG count is not a multiple
of 32 (or 8 with CP FW fix), SE rotation can desynchronize, causing
oversubscription and up to 2x execution time.

Work-stealing is expected to be immune: it over-launches WGs and dynamically
grabs tiles via atomics, so delayed WGs don't block forward progress.

This script measures every (shape, backend) combination under BOTH warm-cache
and rotating-buffer baselines, reports mean AND max (excluding the first
measured iteration), and includes Tensile macro-tile information for torch.

Usage:
    torchrun --nproc_per_node=8 benchmarks/se_oversubscription.py \
        --nccl-max-nchannels 32 --steps 100
"""
import argparse
import math
import os
import statistics
import sys

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas  # noqa: E402


# -- Shapes to sweep -----------------------------------------------------------
# (M, N, K) -- tritonblas uses 256x256 tiles (origami heuristic for large GEMMs)
SHAPES = [
    # -- Small shapes --
    (3584, 3584, 4096),   # tritonblas: 14x14 = 196 tiles (%32=4, %8=4) trigger
    (3840, 3840, 4096),   # tritonblas: 15x15 = 225 tiles (%32=1, %8=1) trigger
    (4096, 4096, 4096),   # tritonblas: 16x16 = 256 tiles -- control
    (4352, 4352, 4096),   # tritonblas: 17x17 = 289 tiles (%32=1, %8=1) trigger

    # -- Large shapes --
    (8192, 8192, 8192),   # tritonblas: 32x32 = 1024 tiles -- control
    (8448, 8448, 8192),   # tritonblas: 33x33 = 1089 tiles (%32=1, %8=1) trigger
    (8704, 8704, 8192),   # tritonblas: 34x34 = 1156 tiles (%32=4, %8=4) trigger

    (12288, 12288, 8192), # tritonblas: 48x48 = 2304 tiles -- control
    (12544, 12544, 8192), # tritonblas: 49x49 = 2401 tiles (%32=1, %8=1) trigger

    (16384, 16384, 8192), # tritonblas: 64x64 = 4096 tiles -- control
    (16640, 16640, 8192), # tritonblas: 65x65 = 4225 tiles (%32=1, %8=1) trigger
]

# Tensile (hipBLASLt) macro-tile sizes obtained via TENSILE_DB=0x8040.
# torch.matmul selects non-square tiles -- we store (MT_M, MT_N) per shape
# so we can compute the *actual* tile count for torch and check SE alignment.
TENSILE_MT = {
    (3584, 3584, 4096): (256, 176),
    (3840, 3840, 4096): (256, 192),
    (4096, 4096, 4096): (256, 224),
    (4352, 4352, 4096): (256, 128),
    (8192, 8192, 8192): (256, 224),
    (8448, 8448, 8192): (256, 160),
    (8704, 8704, 8192): (512, 128),
    (12288, 12288, 8192): (512, 128),
    (12544, 12544, 8192): (256, 304),
    (16384, 16384, 8192): (512, 160),
    (16640, 16640, 8192): (256, 192),
}

N_ROTATING = 4


# -- Matmul helpers ------------------------------------------------------------

def _make_tritonblas_matmul(A, B, C, backend):
    M, K = A.shape
    _, N = B.shape
    work_stealing = backend == "ws"

    selector = tritonblas.OrigamiMatmulSelector(
        M, N, K, A.dtype, B.dtype, C.dtype, A.device, streamk=False)
    cfg = tritonblas.matmul_preamble(selector)

    def matmul_fn():
        tritonblas.matmul_lt(A, B, C, selector, cfg,
                             enable_streamk=False,
                             work_stealing=work_stealing)

    def reset_fn():
        cfg.reset(streamk=False, work_stealing=work_stealing)

    tile_m = selector.block_m
    tile_n = selector.block_n
    total_tiles = triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n)
    return matmul_fn, reset_fn, tile_m, tile_n, total_tiles


def _make_torch_matmul(A, B):
    def matmul_fn():
        torch.matmul(A, B)
    def reset_fn():
        pass
    return matmul_fn, reset_fn


def _tensile_tiles(M, N, K):
    """Compute Tensile tile count from the known macro-tile sizes."""
    key = (M, N, K)
    if key in TENSILE_MT:
        tm, tn = TENSILE_MT[key]
        tiles = math.ceil(M / tm) * math.ceil(N / tn)
        return tm, tn, tiles
    return 0, 0, 0


def _make_warm(M, N, K, dtype, dev, backend):
    """Single (A, B, C) set for warm-cache measurements."""
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    if backend == "torch":
        mfn, rfn = _make_torch_matmul(A, B)
        tile_m, tile_n, total_tiles = _tensile_tiles(M, N, K)
    else:
        C = torch.empty(M, N, dtype=dtype, device=dev)
        mfn, rfn, tile_m, tile_n, total_tiles = _make_tritonblas_matmul(
            A, B, C, backend)
    return mfn, rfn, tile_m, tile_n, total_tiles


def _make_rotating(M, N, K, dtype, dev, backend, n_bufs=N_ROTATING):
    """N rotating (A, B, C) sets for cold-L2 measurements."""
    fns, rfns = [], []
    tile_m = tile_n = total_tiles = 0
    for _ in range(n_bufs):
        rA = torch.randn(M, K, dtype=dtype, device=dev)
        rB = torch.randn(K, N, dtype=dtype, device=dev)
        if backend == "torch":
            mfn, rfn = _make_torch_matmul(rA, rB)
            tile_m, tile_n, total_tiles = _tensile_tiles(M, N, K)
        else:
            rC = torch.empty(M, N, dtype=dtype, device=dev)
            mfn, rfn, tile_m, tile_n, total_tiles = _make_tritonblas_matmul(
                rA, rB, rC, backend)
        fns.append(mfn)
        rfns.append(rfn)
    return fns, rfns, tile_m, tile_n, total_tiles


# -- Timing helpers ------------------------------------------------------------

def _ext_stats(times):
    """mean, median, max, and max excluding the first measured iteration."""
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "max": max(times),
        "max1": max(times[1:]) if len(times) > 1 else max(times),
    }


def _time_warm(matmul_fn, reset_fn, stream, n_warmup, n_steps):
    """GEMM alone with warm cache (same buffers every iteration)."""
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(stream):
            matmul_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        torch.cuda.synchronize()
        starts[i].record(stream)
        with torch.cuda.stream(stream):
            matmul_fn()
        ends[i].record(stream)
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_rotating(matmul_fns, reset_fns, stream, n_warmup, n_steps):
    """GEMM alone with rotating buffers (cold L2 each iteration)."""
    n_bufs = len(matmul_fns)
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        with torch.cuda.stream(stream):
            matmul_fns[idx]()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    for i in range(n_steps):
        idx = i % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        torch.cuda.synchronize()
        starts[i].record(stream)
        with torch.cuda.stream(stream):
            matmul_fns[idx]()
        ends[i].record(stream)
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_overlap_warm(matmul_fn, reset_fn, comm_fn,
                       mm_stream, co_stream, n_warmup, n_steps):
    """Overlapped GEMM + comm with warm cache."""
    for _ in range(n_warmup):
        if reset_fn:
            reset_fn()
        with torch.cuda.stream(co_stream):
            comm_fn()
        with torch.cuda.stream(mm_stream):
            matmul_fn()
    torch.cuda.synchronize()

    mm_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        if reset_fn:
            reset_fn()
        torch.cuda.synchronize()

        mm_stream.wait_stream(torch.cuda.current_stream())
        co_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(co_stream):
            comm_fn()

        mm_s[i].record(mm_stream)
        with torch.cuda.stream(mm_stream):
            torch.cuda._sleep(100_000)
            matmul_fn()
        mm_e[i].record(mm_stream)

        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)

    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(mm_s, mm_e)]


def _time_overlap_rotating(matmul_fns, reset_fns, comm_fn,
                           mm_stream, co_stream, n_warmup, n_steps):
    """Overlapped GEMM + comm with rotating buffers (cold L2)."""
    n_bufs = len(matmul_fns)
    for j in range(max(n_warmup, n_bufs)):
        idx = j % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        with torch.cuda.stream(co_stream):
            comm_fn()
        with torch.cuda.stream(mm_stream):
            matmul_fns[idx]()
    torch.cuda.synchronize()

    mm_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    mm_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]

    for i in range(n_steps):
        idx = i % n_bufs
        if reset_fns[idx]:
            reset_fns[idx]()
        torch.cuda.synchronize()

        mm_stream.wait_stream(torch.cuda.current_stream())
        co_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(co_stream):
            comm_fn()

        mm_s[i].record(mm_stream)
        with torch.cuda.stream(mm_stream):
            torch.cuda._sleep(100_000)
            matmul_fns[idx]()
        mm_e[i].record(mm_stream)

        torch.cuda.current_stream().wait_stream(mm_stream)
        torch.cuda.current_stream().wait_stream(co_stream)

    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(mm_s, mm_e)]


# -- Formatting helpers --------------------------------------------------------

def _risk_tag(tiles):
    if tiles <= 0:
        return "?"
    if tiles % 8 != 0:
        return "TRIGGER"
    elif tiles % 32 != 0:
        return "border"
    return "ok"


def _print_shape_block(rank, shape_results, scenario_label, alone_key, ovlp_key):
    """Print a compact sub-table for one shape under one scenario."""
    if rank != 0:
        return

    p = shape_results["persistent"]
    w = shape_results["ws"]
    t = shape_results["torch"]

    print(f"    --- {scenario_label} ---")
    print(f"    {'Backend':>10s}  "
          f"{'Alone(mean)':>11s}  {'Alone(max*)':>11s}  "
          f"{'Ovlp(mean)':>10s}  {'Ovlp(max*)':>10s}  "
          f"{'Slowdown':>8s}")

    for bname, r in [("persistent", p), ("ws", w), ("torch", t)]:
        alone = r[alone_key]
        ovlp = r[ovlp_key]
        slow = ovlp["mean"] / alone["mean"] if alone["mean"] > 0 else 0

        marker = ""
        if bname == "ws":
            ws_o = ovlp["mean"]
            p_o = p[ovlp_key]["mean"]
            t_o = t[ovlp_key]["mean"]
            if ws_o < p_o and ws_o < t_o:
                marker = "  << ABS WINNER"
            elif ws_o < p_o:
                marker = "  < beats P"
            elif ws_o < t_o:
                marker = "  < beats T"

        print(f"    {bname:>10s}  "
              f"{alone['mean']:>11.3f}  {alone['max1']:>11.3f}  "
              f"{ovlp['mean']:>10.3f}  {ovlp['max1']:>10.3f}  "
              f"{slow:>7.2f}x{marker}")


# -- Main ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SE Oversubscription Benchmark (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--comm-size", type=int, nargs="+", default=[8192, 8192],
                   help="Collective tensor shape for overlap phase")
    p.add_argument("--nccl-max-nchannels", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    if args.nccl_max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.nccl_max_nchannels)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    dtype = torch.bfloat16
    dev = torch.device("cuda", local_rank)

    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)

    comm_tensor = torch.randn(*args.comm_size, dtype=dtype, device=dev)
    comm_fn = lambda: dist.all_reduce(comm_tensor)

    backends = ["persistent", "ws", "torch"]

    # results[shape_idx] = { "persistent": {...}, "ws": {...}, "torch": {...} }
    all_results = []

    for shape_idx, (M, N, K) in enumerate(SHAPES):
        shape_results = {}

        # Tile info header
        if rank == 0:
            sel = tritonblas.OrigamiMatmulSelector(
                M, N, K, dtype, dtype, dtype, dev, streamk=False)
            tb_tm, tb_tn = sel.block_m, sel.block_n
            tb_tiles = triton.cdiv(M, tb_tm) * triton.cdiv(N, tb_tn)
            tb_risk = _risk_tag(tb_tiles)

            t_tm, t_tn, t_tiles = _tensile_tiles(M, N, K)
            t_risk = _risk_tag(t_tiles)

            print(f"\n{'='*90}")
            print(f"  {M}x{N}x{K}")
            print(f"  tritonblas tile: {tb_tm}x{tb_tn}  ->  {tb_tiles} tiles  "
                  f"(%32={tb_tiles%32}, %8={tb_tiles%8})  [{tb_risk}]")
            if t_tm > 0:
                print(f"  Tensile tile:    {t_tm}x{t_tn}  ->  {t_tiles} tiles  "
                      f"(%32={t_tiles%32}, %8={t_tiles%8})  [{t_risk}]")
            print(f"{'='*90}")
            del sel

        for backend in backends:
            torch.cuda.empty_cache()

            # Build warm set (single A, B, C)
            warm_fn, warm_rfn, tile_m, tile_n, total_tiles = _make_warm(
                M, N, K, dtype, dev, backend)

            # Build rotating sets (N_ROTATING independent A, B, C)
            rot_fns, rot_rfns, _, _, _ = _make_rotating(
                M, N, K, dtype, dev, backend)

            # 1. Warm alone
            wa_times = _time_warm(
                warm_fn, warm_rfn, mm_stream, args.warmup, args.steps)

            # 2. Rotating alone
            ra_times = _time_rotating(
                rot_fns, rot_rfns, mm_stream, args.warmup, args.steps)

            # 3. Warm overlap
            wo_times = _time_overlap_warm(
                warm_fn, warm_rfn, comm_fn,
                mm_stream, co_stream, args.warmup, args.steps)

            # 4. Rotating overlap
            ro_times = _time_overlap_rotating(
                rot_fns, rot_rfns, comm_fn,
                mm_stream, co_stream, args.warmup, args.steps)

            wa = _ext_stats(wa_times)
            ra = _ext_stats(ra_times)
            wo = _ext_stats(wo_times)
            ro = _ext_stats(ro_times)

            shape_results[backend] = {
                "M": M, "N": N, "K": K,
                "tile_m": tile_m, "tile_n": tile_n,
                "total_tiles": total_tiles,
                "warm_alone": wa,
                "rot_alone": ra,
                "warm_overlap": wo,
                "rot_overlap": ro,
            }

            if rank == 0:
                risk = _risk_tag(total_tiles)
                tile_str = f"{tile_m}x{tile_n}" if tile_m > 0 else "N/A"
                w_slow = wo["mean"] / wa["mean"] if wa["mean"] > 0 else 0
                r_slow = ro["mean"] / ra["mean"] if ra["mean"] > 0 else 0
                print(f"  {backend:>10s}  tile={tile_str:>9s}  "
                      f"tiles={total_tiles:>5d}  [{risk:>7s}]  "
                      f"W: {wa['mean']:.3f}->{wo['mean']:.3f} ({w_slow:.2f}x)  "
                      f"R: {ra['mean']:.3f}->{ro['mean']:.3f} ({r_slow:.2f}x)")

        all_results.append(shape_results)

        # Per-shape detailed block
        _print_shape_block(rank, shape_results,
                           "Warm Cache", "warm_alone", "warm_overlap")
        _print_shape_block(rank, shape_results,
                           "Rotating Buffer", "rot_alone", "rot_overlap")

    # -- Final compact summary tables ------------------------------------------
    if rank == 0:
        nccl_ch = os.environ.get("NCCL_MAX_NCHANNELS", "unset")

        for scenario_label, alone_key, ovlp_key in [
            ("WARM CACHE", "warm_alone", "warm_overlap"),
            ("ROTATING BUFFER", "rot_alone", "rot_overlap"),
        ]:
            W = 178
            print(f"\n{'='*W}")
            print(f"  {scenario_label} SUMMARY -- NCCL_MAX_NCHANNELS={nccl_ch}, "
                  f"comm={args.comm_size}, steps={args.steps}")
            print(f"{'='*W}")
            print(f"  {'Shape':>20s}  "
                  f"{'P.tile':>9s}  {'P.#':>5s}  "
                  f"{'T.tile':>9s}  {'T.#':>5s}  "
                  f"{'P.alone':>7s}  {'P.ovlp':>7s}  {'P.max*':>7s}  {'P.slow':>6s}  "
                  f"{'WS.alone':>8s}  {'WS.ovlp':>8s}  {'WS.max*':>8s}  {'WS.slow':>7s}  "
                  f"{'T.alone':>7s}  {'T.ovlp':>7s}  {'T.max*':>7s}  {'T.slow':>6s}  "
                  f"{'WS<P':>4s}  {'WS<T':>4s}")
            print(f"  {'-'*(W-2)}")

            for i, (M, N, K) in enumerate(SHAPES):
                p = all_results[i]["persistent"]
                w = all_results[i]["ws"]
                t = all_results[i]["torch"]

                shape_str = f"{M}x{N}x{K}"

                p_tile_str = f"{p['tile_m']}x{p['tile_n']}"
                p_tiles = p["total_tiles"]

                t_tm, t_tn, t_tiles = _tensile_tiles(M, N, K)
                t_tile_str = f"{t_tm}x{t_tn}" if t_tm > 0 else "N/A"

                p_al = p[alone_key]["mean"]
                p_ov = p[ovlp_key]["mean"]
                p_mx = p[ovlp_key]["max1"]
                p_sl = p_ov / p_al if p_al > 0 else 0

                w_al = w[alone_key]["mean"]
                w_ov = w[ovlp_key]["mean"]
                w_mx = w[ovlp_key]["max1"]
                w_sl = w_ov / w_al if w_al > 0 else 0

                t_al = t[alone_key]["mean"]
                t_ov = t[ovlp_key]["mean"]
                t_mx = t[ovlp_key]["max1"]
                t_sl = t_ov / t_al if t_al > 0 else 0

                ws_lt_p = "YES" if w_ov < p_ov else "no"
                ws_lt_t = "YES" if w_ov < t_ov else "no"

                print(f"  {shape_str:>20s}  "
                      f"{p_tile_str:>9s}  {p_tiles:>5d}  "
                      f"{t_tile_str:>9s}  {t_tiles:>5d}  "
                      f"{p_al:>7.3f}  {p_ov:>7.3f}  {p_mx:>7.3f}  "
                      f"{p_sl:>5.2f}x  "
                      f"{w_al:>8.3f}  {w_ov:>8.3f}  {w_mx:>8.3f}  "
                      f"{w_sl:>6.2f}x  "
                      f"{t_al:>7.3f}  {t_ov:>7.3f}  {t_mx:>7.3f}  "
                      f"{t_sl:>5.2f}x  "
                      f"{ws_lt_p:>4s}  {ws_lt_t:>4s}")

            print(f"  {'-'*(W-2)}")
            print(f"  P = persistent | WS = work-stealing | T = torch.matmul (Tensile)")
            print(f"  max* = max excluding first measured iteration")
            print(f"  WS<P / WS<T = WS overlap(mean) < P/T overlap(mean)")
            print()

        # -- Tensile reference table -------------------------------------------
        print(f"\n{'='*80}")
        print(f"  Tensile (hipBLASLt) Macro-Tile Reference  (TENSILE_DB=0x8040)")
        print(f"{'='*80}")
        print(f"  {'Shape':>20s}  {'MT (MxN)':>12s}  {'Tiles':>6s}  "
              f"{'%32':>4s}  {'%8':>3s}  {'Risk':>7s}")
        print(f"  {'-'*60}")
        for M, N, K in SHAPES:
            tm, tn, tiles = _tensile_tiles(M, N, K)
            if tm > 0:
                shape_str = f"{M}x{N}x{K}"
                mt_str = f"{tm}x{tn}"
                risk = _risk_tag(tiles)
                print(f"  {shape_str:>20s}  {mt_str:>12s}  {tiles:>6d}  "
                      f"{tiles%32:>4d}  {tiles%8:>3d}  {risk:>7s}")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

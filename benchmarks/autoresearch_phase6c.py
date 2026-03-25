#!/usr/bin/env python3
"""
Autoresearch Phase 6c: Tile shape exploration.

Tests smaller tiles (128x128, 128x256) that could achieve higher occupancy
by reducing VGPR pressure. Also tests gm=4 as a potential improvement over gm=8.
"""
import json, os, statistics, sys
import torch, triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 50; N_ROT = 4


def bench_torch(sz):
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    for w in range(WARMUP): torch.matmul(As[w % N_ROT], Bs[w % N_ROT])
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record(); torch.matmul(As[i % N_ROT], Bs[i % N_ROT]); es[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(ss, es)]
    del As, Bs; torch.cuda.empty_cache()
    return statistics.median(times)


def bench_ws(sz, bm, bn, bk, warps, gm, split_frac=None, wpe=0):
    try:
        sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
        n_cu = sel._N_CU; num_xcds = sel.num_sms
        total_tiles = triton.cdiv(sz, bm) * triton.cdiv(sz, bn)
        even_k = sz % bk == 0

        if split_frac is not None:
            lp = int(total_tiles * split_frac) // num_xcds
        else:
            tiles_per_cu = total_tiles / max(n_cu, 1)
            local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
            lp = int(total_tiles * local_frac) // num_xcds
        lp = max(lp, 1)
        gt = total_tiles - lp * num_xcds

        As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
        tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
        gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
        mask = torch.ones(n_cu, dtype=torch.int32, device=device)

        def reset(): tc.zero_(); gc.zero_()
        def run(idx):
            reset()
            ws_hierarchical_matmul[(n_cu,)](
                As[idx % N_ROT], Bs[idx % N_ROT], Cs[idx % N_ROT],
                None, None, None, tc, gc,
                sz, sz, sz,
                As[0].stride(0), Bs[0].stride(1),
                Cs[0].stride(0), Cs[0].stride(1), 0,
                stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
                LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
                COUNTER_STRIDE=COUNTER_STRIDE,
                BIAS=False, EVEN_K=even_k,
                CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
                QUANTIZED=False,
                num_stages=2, num_warps=warps, waves_per_eu=wpe,
                matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
            )

        for w in range(WARMUP): run(w)
        torch.cuda.synchronize()

        ref = torch.matmul(As[0], Bs[0])
        Cs[0].zero_(); reset(); run(0); torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            Cs[0].float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
        if cos < 0.999:
            del As, Bs, Cs; torch.cuda.empty_cache()
            return None, f"cos={cos:.6f}"

        ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            reset(); ss[i].record(); run(i); es[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(ss, es)]
        del As, Bs, Cs; torch.cuda.empty_cache()
        return statistics.median(times), None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, str(e)[:100]


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)

    print("=" * 80)
    print("  Phase 6c: Tile shape × warps × occupancy exploration")
    print("=" * 80)

    configs = [
        # (label, BM, BN, BK, warps, gm, wpe)
        ("256x256x64 w=8 gm=8 (baseline)", 256, 256, 64, 8, 8, 0),
        ("256x256x64 w=8 gm=4", 256, 256, 64, 8, 4, 0),
        ("128x128x64 w=4 gm=8", 128, 128, 64, 4, 8, 0),
        ("128x128x64 w=4 gm=4", 128, 128, 64, 4, 4, 0),
        ("128x128x64 w=8 gm=8", 128, 128, 64, 8, 8, 0),
        ("128x256x64 w=8 gm=8", 128, 256, 64, 8, 8, 0),
        ("256x128x64 w=8 gm=8", 256, 128, 64, 8, 8, 0),
        ("128x128x64 w=4 gm=8 wpe=2", 128, 128, 64, 4, 8, 2),
        ("128x128x32 w=4 gm=8", 128, 128, 32, 4, 8, 0),
    ]

    for sz in [12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12
        print(f"  torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)\n")
        print(f"  {'Config':<38s}  {'ms':>8s}  {'TF':>7s}  {'vs torch':>9s}")
        print("  " + "-" * 65)

        for label, bm, bn, bk, warps, gm, wpe in configs:
            tiles = triton.cdiv(sz, bm) * triton.cdiv(sz, bn)
            ms, err = bench_ws(sz, bm, bn, bk, warps, gm, wpe=wpe)
            if ms is None:
                print(f"  {label:<38s}  {'FAIL':>8s}  {err or ''}")
            else:
                tf = flops / (ms * 1e-3) / 1e12
                vs = (ms - torch_ms) / torch_ms * 100
                print(f"  {label:<38s}  {ms:>8.3f}  {tf:>7.1f}  {vs:>+8.1f}%")

#!/usr/bin/env python3
"""Deeper diagnosis: CP dispatch waves, L2 pressure, atomic contention."""
import statistics, sys, os, torch, triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP, ITERS, N_ROT = 10, 50, 4

def bench_ws(sz, grid_size):
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    BM, BN, BK = sel.block_m, sel.block_n, sel.block_k
    n_cu = sel._N_CU
    num_xcds = sel.num_sms
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    even_k = sz % BK == 0
    lp = total_tiles // num_xcds
    gt = total_tiles - lp * num_xcds
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Cs = [torch.zeros(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    tc = torch.zeros(num_xcds * COUNTER_STRIDE, device=device, dtype=torch.int32)
    gc = torch.zeros(COUNTER_STRIDE, device=device, dtype=torch.int32)
    def run(i):
        tc.zero_(); gc.zero_()
        ws_hierarchical_matmul[(grid_size,)](
            As[i%N_ROT], Bs[i%N_ROT], Cs[i%N_ROT], None, None, None, tc, gc,
            sz, sz, sz, As[0].stride(0), Bs[0].stride(1),
            Cs[0].stride(0), Cs[0].stride(1), 0,
            stride_ak=As[0].stride(1), stride_bk=Bs[0].stride(0),
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=sel.group_m, NUM_SMS=grid_size, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, USE_MASK=False)
    for w in range(WARMUP): run(w)
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record(); run(i); es[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(ss, es)]
    del As, Bs, Cs; torch.cuda.empty_cache()
    return statistics.median(times), total_tiles

def bench_torch(sz):
    As = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    Bs = [torch.randn(sz, sz, dtype=dtype, device=device) for _ in range(N_ROT)]
    for w in range(WARMUP): torch.matmul(As[w%N_ROT], Bs[w%N_ROT])
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record(); torch.matmul(As[i%N_ROT], Bs[i%N_ROT]); es[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(ss, es)]
    del As, Bs; torch.cuda.empty_cache()
    return statistics.median(times)

sizes = [4096, 8192, 12288, 16384]

print("=" * 85)
print("Test 1: CP waves — grid=tiles vs grid=304")
print("=" * 85)
hdr = f"{'Size':>6s}  {'tiles':>6s}  {'grid':>6s}  {'waves':>6s}  {'last%':>6s}  {'ms':>8s}  {'vs 304':>8s}"
print(hdr); print("-" * len(hdr))
for sz in sizes:
    g304, tt = bench_ws(sz, 304)
    gt, _ = bench_ws(sz, tt)
    fw = tt // 304; lw = tt - fw * 304
    nw = fw + (1 if lw else 0)
    lo = lw / 304 * 100 if lw else 100.0
    d = (gt - g304) / g304 * 100
    print(f"{sz:>6d}  {tt:>6d}  {304:>6d}  {'1':>6s}  {'100':>5s}%  {g304:>8.3f}  {'base':>8s}")
    print(f"{'':>6s}  {'':>6s}  {tt:>6d}  {nw:>6d}  {lo:>5.1f}%  {gt:>8.3f}  {d:>+7.1f}%")
    print()

print("=" * 85)
print("Test 2: L2 live footprint per XCD")
print("  MI300X: 32 MB L2 per XCD")
print("=" * 85)
for sz in sizes:
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    BM, BN, BK = sel.block_m, sel.block_n, sel.block_k
    tt = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    nxcd = sel.num_sms; cpx = sel._N_CU // nxcd; tpx = tt // nxcd
    ns = 2  # num_stages
    live_cu = (BM * BK + BK * BN) * 2 * ns + BM * BN * 2
    live_xcd = live_cu * cpx
    print(f"  {sz:>5d}: {tpx} tiles/XCD, {cpx} CUs/XCD, "
          f"live/CU={live_cu/1024:.0f}KB, live/XCD={live_xcd/1024/1024:.1f}MB "
          f"({live_xcd/1024/1024/32*100:.0f}% of 32MB)")

print()
print("=" * 85)
print("Test 3: Raw gap breakdown — torch vs WS (grid=304, no-mask)")
print("=" * 85)
hdr2 = f"{'Size':>6s}  {'torch ms':>9s}  {'torch TF':>9s}  {'WS ms':>8s}  {'WS TF':>7s}  {'gap%':>6s}  {'gap ms':>7s}  {'tiles/CU':>9s}  {'oh/tile us':>11s}"
print(hdr2); print("-" * len(hdr2))
for sz in sizes:
    tm = bench_torch(sz)
    wm, tt = bench_ws(sz, 304)
    flops = 2.0 * sz**3
    ttf = flops / (tm * 1e-3) / 1e12
    wtf = flops / (wm * 1e-3) / 1e12
    gap = wm - tm; gp = gap / tm * 100
    tpc = tt / 304
    oh = gap * 1000 / tpc if tpc > 0 else 0
    print(f"{sz:>6d}  {tm:>9.3f}  {ttf:>9.1f}  {wm:>8.3f}  {wtf:>7.1f}  {gp:>+5.1f}%  {gap:>7.3f}  {tpc:>9.1f}  {oh:>11.1f}")

print()
print("=" * 85)
print("Test 4: Varying grid size for 16K (isolate scheduler effects)")
print("=" * 85)
sz = 16384
sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
tt = triton.cdiv(sz, sel.block_m) * triton.cdiv(sz, sel.block_n)
grids = [152, 200, 256, 304, 608, 1024, 2048, 4096]
hdr3 = f"{'grid':>6s}  {'ms':>8s}  {'waves':>6s}  {'last%':>6s}  {'vs 304':>8s}"
print(hdr3); print("-" * len(hdr3))
g304, _ = bench_ws(sz, 304)
for g in grids:
    gm, _ = bench_ws(sz, g)
    fw = tt // g; lw = tt - fw * g
    nw = fw + (1 if lw else 0)
    lo = lw / g * 100 if lw else 100.0
    d = (gm - g304) / g304 * 100
    print(f"{g:>6d}  {gm:>8.3f}  {nw:>6d}  {lo:>5.1f}%  {d:>+7.1f}%")

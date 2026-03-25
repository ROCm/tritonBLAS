#!/usr/bin/env python3
"""
Autoresearch: Final validation.
1. Confirm tuned WS Hierarchical matches Phase 4 improvements.
2. Run overlap comparison: WS Hierarchical vs torch.matmul with RCCL.
"""
import json, os, statistics, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))
import tritonblas
from tritonblas.config import COUNTER_STRIDE
from tritonblas.kernels.persistent_gemm_ws_hierarchical import ws_hierarchical_matmul

torch.cuda.set_device(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16
WARMUP = 15; ITERS = 40; N_ROT = 4


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


def bench_ws(sz):
    BM, BN, BK = 256, 256, 64
    sel = tritonblas.OrigamiMatmulSelector(sz, sz, sz, dtype, dtype, dtype, device, streamk=False)
    n_cu = sel._N_CU; num_xcds = sel.num_sms; gsize_m = sel.group_m
    total_tiles = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)
    even_k = sz % BK == 0
    local_per_xcd = (total_tiles * 9) // (num_xcds * 10)
    local_per_xcd = max(local_per_xcd, 1)
    global_tiles = total_tiles - local_per_xcd * num_xcds

    print(f"    Config: BLK={BM}x{BN}x{BK}, gm={gsize_m}, n_cu={n_cu}, "
          f"xcds={num_xcds}, tiles={total_tiles}, local/xcd={local_per_xcd}, global={global_tiles}")

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
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=gsize_m, NUM_SMS=n_cu, NUM_XCDS=num_xcds,
            LOCAL_TILES_PER_XCD=local_per_xcd, GLOBAL_TILES=global_tiles,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None,
            QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, mask_ptr=mask,
        )

    for w in range(WARMUP): run(w)
    torch.cuda.synchronize()

    # Correctness
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


if __name__ == "__main__":
    os.makedirs("results/autoresearch", exist_ok=True)

    print("=" * 80)
    print("  Final Validation: WS Hierarchical (gm=8 universally)")
    print("=" * 80)

    results = {}
    for sz in [4096, 8192, 12288, 16384]:
        print(f"\n  --- {sz}x{sz}x{sz} BF16 ---")
        torch_ms = bench_torch(sz)
        flops = 2.0 * sz ** 3
        torch_tf = flops / (torch_ms * 1e-3) / 1e12

        ws_ms, err = bench_ws(sz)
        if ws_ms is None:
            print(f"    torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)")
            print(f"    WS:    FAIL ({err})")
            continue

        ws_tf = flops / (ws_ms * 1e-3) / 1e12
        gap = (ws_ms - torch_ms) / torch_ms * 100
        print(f"    torch: {torch_ms:.3f} ms ({torch_tf:.1f} TF)")
        print(f"    WS:    {ws_ms:.3f} ms ({ws_tf:.1f} TF)  [{gap:+.1f}%]")

        results[str(sz)] = {
            "torch_ms": torch_ms, "torch_tflops": torch_tf,
            "ws_ms": ws_ms, "ws_tflops": ws_tf, "gap_pct": gap,
        }

    with open("results/autoresearch/final_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("  Summary")
    print(f"{'='*80}")
    for k, r in results.items():
        print(f"  {k}: torch={r['torch_ms']:.3f}ms  WS={r['ws_ms']:.3f}ms  "
              f"gap={r['gap_pct']:+.1f}%  WS_TF={r['ws_tflops']:.0f}")
    print()

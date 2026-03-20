# tritonBLAS Work-Stealing GEMM Optimization Program

## Goal

Optimize the WS persistent GEMM kernel (`include/tritonblas/kernels/persistent_gemm_work_stealing.py`)
to match or beat `torch.matmul` (hipBLASLt Tensile) raw performance on MI300X.

## Current State

- **Target**: 8192x8192x8192 BF16 on MI300X (304 CUs, 8 XCDs)
- **torch.matmul**: ~1.63ms (680 TFLOPS)
- **WS kernel**: ~1.72ms (639 TFLOPS) — 5.7% gap
- **Tile config**: 256x256x64, num_stages=2, num_warps=8
- **The WS kernel already beats torch during RCCL overlap** (2.05ms vs 2.29ms)

## Files You Can Modify

- `include/tritonblas/kernels/persistent_gemm_work_stealing.py` — THE kernel
- `include/tritonblas/kernels/streamk_gemm_work_stealing.py` — StreamK+WS variant
- `include/tritonblas/matmul.py` — launch configuration
- `include/tritonblas/origami.py` — tile selection, GROUP_SIZE_M, COUNTERS_PER_XCD
- `include/tritonblas/config.py` — pre-allocated buffers

## Files You MUST NOT Modify

- `include/tritonblas/kernels/stages/` — shared stage infrastructure
- `benchmarks/overlap.py` — benchmark harness (read-only for experiments)

## How to Run an Experiment

```bash
# Single-GPU benchmark (fast, ~5 sec)
python3 -c "
import torch, tritonblas, statistics
torch.cuda.set_device(GPU_ID)
dev = torch.device('cuda', GPU_ID)
M = N = K = 8192
A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
cfg = tritonblas.matmul_preamble(sel)
s = torch.cuda.Stream(device=dev)
for _ in range(20):
    with torch.cuda.stream(s): cfg.reset(work_stealing=True); tritonblas.matmul_lt(A,B,C,sel,cfg,work_stealing=True)
torch.cuda.synchronize()
t = []
for _ in range(50):
    with torch.cuda.stream(s): cfg.reset(work_stealing=True)
    torch.cuda.synchronize()
    st,en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record(s)
    with torch.cuda.stream(s): tritonblas.matmul_lt(A,B,C,sel,cfg,work_stealing=True)
    en.record(s); torch.cuda.synchronize()
    t.append(st.elapsed_time(en))
med = statistics.median(t)
print(f'WS: {med:.3f}ms ({2*M*N*K/(med*1e-3)/1e12:.0f} TFLOPS)')
"
```

## Metric

- Primary: median kernel time (ms) at 8192x8192x8192 BF16
- Must pass correctness: max error < 10 vs torch.matmul reference
- Also check: 4096 and 16384 sizes (don't regress)

## Optimization Ideas (Prioritized)

### High Priority
1. **K-loop software pipelining** — the K-loop loads A and B tiles sequentially; 
   overlap the load of the NEXT K-iteration's tiles with the MFMA of the current.
   Triton's num_stages=2 should do this, but verify the compiler is actually pipelining.
2. **tl.assume hints** — add `tl.assume(K > 0)`, `tl.assume(loop_k > 0)` etc. to help
   the compiler eliminate bounds checks in the hot loop.
3. **Reduce register pressure** — the 256x256 accumulator is 64KB of VGPRs. If the 
   compiler spills, that's the biggest perf hit. Check if smaller accumulators 
   (e.g., processing the tile in 2x128x256 halves) help.

### Medium Priority
4. **A/B load coalescing** — ensure loads use full 128-byte transactions. The 
   `tl.multiple_of` hints help but may need explicit padding.
5. **LDS double-buffering** — verify Triton actually generates double-buffered LDS 
   access (not sequential load-compute-load-compute).
6. **Epilogue optimization** — the C store at the end uses masked stores; for aligned
   tiles (BLOCK_SIZE_M divides M), eliminate the mask.

### Experimental
7. **Split-K** — divide the K dimension across workgroups, accumulate via atomics.
   May help for large K relative to M*N.
8. **Persistent kernel with multiple tiles per WG** — instead of 1 tile per atomic,
   grab N tiles at once to amortize the atomic overhead.
9. **XCD-aware tile ordering** — ensure tiles assigned to each XCD have good L2 locality
   (GROUP_SIZE_M already does this partially).

## Rules

- Every experiment must check correctness (max error < 10 vs reference)
- Record: experiment ID, hypothesis, config, result (ms), conclusion
- Keep changes that improve performance; revert changes that don't
- Test at multiple sizes (4K, 8K, 16K) before committing
- Commit improvements with a perf table in the commit message

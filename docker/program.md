# tritonBLAS Overlap Research Program

## Goal

Systematically investigate the **4 causes of GEMM performance degradation during
compute-communication overlap** on MI300X.

The 4 causes (from Slide 3):
1. Tail Latency Effect (wave quantization)        — partially understood
2. Work Distribution Across XCDs and SEs           — partially understood
3. **L2 and MALL Thrashing**                       — L2 ruled out, MALL needs df-counters
4. **HBM and Network Contention**                  — partially decomposed via hog isolation

## Completed Findings

### Finding 1: L2 Cache is NOT Affected by RCCL Overlap (exp_002)
GEMM L2 hit rates are IDENTICAL with or without RCCL (delta < 0.3% across all sizes).
MI300X's per-XCD non-coherent L2 provides complete isolation.

### Finding 2: Overlap Slowdown Decomposition (exp_004)
For 8K GEMM + all_reduce with work-stealing:
- **CU contention: +1.2%** — work-stealing handles it almost perfectly
- **Memory BW contention: +7.5%** — streaming 32MB from 32 WGs
- **Network/other: +4.3%** — likely Infinity Fabric overhead
- **Total: ~13%** (matching 1.13x measured in exp_003)

### Finding 3: NCCL Channel Count Does Not Affect GEMM (exp_005)
GEMM slowdown is CONSTANT at ~1.03x whether NCCL uses 4 or 32 channels.
The bottleneck is not NCCL's memory traffic volume — it's the fixed overhead
of sharing CUs with NCCL workgroups.

## Next Experiments

### With df-counters rocprofv3 (Priority 1)
1. **MALL/LLC baseline vs overlap**: Does MALL show the thrashing that L2 doesn't?
2. **HBM bandwidth isolation**: Are GEMM's HBM reads/writes degraded during overlap?
3. **Per-XCD MALL analysis**: Which XCDs show most MALL pressure during overlap?

### Without df-counters (can run now)
4. **Backend comparison**: WS vs persistent vs torch overlap efficiency
5. **Size sweep with timing**: Full GEMM size × NCCL channel matrix
6. **NCCL collective type sweep**: all_reduce vs all_gather vs reduce_scatter
7. **Multi-size GEMM shapes**: Non-square shapes common in transformers

## Environment

- Docker: `tritonblas-research:latest`
- ROCm 7.2.0, PyTorch 2.12.0, Triton 3.6.0
- GPU: MI300X (304 CUs, 8 XCDs)
- rocprofv3 1.1.0 (L2/TCC counters)
- df-counters for MALL/HBM (needs internal repo)

## How to Run

```bash
# Build Docker
./docker/build.sh

# Launch container
./docker/run.sh

# Inside container:
python3 benchmarks/autoresearch.py list          # Show experiments
python3 benchmarks/autoresearch.py run -e <name>  # Run experiment
python3 benchmarks/autoresearch.py log            # View results

# Direct overlap benchmarks:
torchrun --nproc_per_node=8 benchmarks/overlap.py standard \
    --backend ws --gemm-m 8192 --gemm-n 8192 --gemm-k 8192 \
    --comm-size 8192 8192 --steps 50

# L2 profiling with rocprofv3:
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum -o out -d results/ --output-format csv \
    -- python3 benchmarks/overlap.py l2-profile --profile-mode gemm-alone --backend ws

# MALL/HBM profiling (requires df-counters):
rocprofv3 --pmc MALL_BANDWIDTH_ALL HBM_READ_BYTES HBM_WRITE_BYTES \
    -o out -d results/ --output-format csv \
    -- python3 benchmarks/overlap.py l2-profile --profile-mode gemm-alone --backend ws
```

## Experiment Log
See `results/experiment_log.json` for structured results from all experiments.

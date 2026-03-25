#!/bin/bash
# Collect MALL, L2, HBM counters for 8K GEMM alone vs GEMM+RCCL overlap.
# Run INSIDE the mall-experiments Docker container.
set -e

OUTDIR="/workspace/tritonBLAS/results/counters_8k"
SCRIPT="/workspace/tritonBLAS/benchmarks/collect_counters.py"
PYTHONPATH="/workspace/tritonBLAS/include"
export PYTHONPATH HSA_NO_SCRATCH_RECLAIM=1

mkdir -p "$OUTDIR"

# Counter groups (rocprofv3 may limit per-pass count)
PASS1="TCC_HIT_sum,TCC_MISS_sum,TCC_EA0_RDREQ_DRAM_sum,TCC_EA0_WRREQ_DRAM_sum"
PASS2="MALL_HIT_RATE_1,MALL_MISS_RATE_1,MALL_BANDWIDTH_ALL"
PASS3="HBM_READ_BYTES,HBM_WRITE_BYTES"

run_alone() {
    local scenario=$1
    local pass_name=$2
    local counters=$3
    local tag="${scenario}_${pass_name}"
    echo ">>> $tag: $counters"
    HIP_VISIBLE_DEVICES=4 rocprofv3 --pmc $counters \
        -o "$OUTDIR/$tag" \
        -- python3 "$SCRIPT" --scenario "$scenario" 2>&1 | grep -v "^W2026"
    echo "  done: $tag"
}

run_rccl() {
    local scenario=$1
    local pass_name=$2
    local counters=$3
    local tag="${scenario}_${pass_name}"
    echo ">>> $tag: $counters"
    # Use -d with per-PID subdirs and CSV format to avoid SQLite concurrent write issues
    rm -rf "$OUTDIR/${tag}_dir"
    NCCL_MAX_NCHANNELS=32 rocprofv3 --pmc $counters \
        -d "$OUTDIR/${tag}_dir" -f csv \
        -- torchrun --nproc_per_node=8 "$SCRIPT" --scenario "$scenario" 2>&1 | grep -v "^W2026"
    echo "  done: $tag"
}

echo "========== ALONE scenarios =========="
for SCENARIO in torch_alone ws_alone; do
    run_alone "$SCENARIO" l2    "$PASS1"
    run_alone "$SCENARIO" mall  "$PASS2"
    run_alone "$SCENARIO" hbm   "$PASS3"
done

echo "========== RCCL overlap scenarios =========="
for SCENARIO in torch_rccl ws_rccl; do
    run_rccl "$SCENARIO" l2    "$PASS1"
    run_rccl "$SCENARIO" mall  "$PASS2"
    run_rccl "$SCENARIO" hbm   "$PASS3"
done

echo "========== ALL DONE =========="
ls -la "$OUTDIR"/*.db 2>/dev/null

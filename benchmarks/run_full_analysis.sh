#!/usr/bin/env bash
# Full overlap analysis: 3 approaches × warm/rotating × L2+MALL counters
# Must run INSIDE the tritonblas-research Docker container
set -e

RESULTS=/workspace/tritonBLAS/results/full_analysis
mkdir -p "$RESULTS"

M=8192; N=8192; K=8192
WARMUP=5; STEPS=20
COUNTERS="TCC_HIT_sum TCC_MISS_sum TCC_WRITEBACK_sum"

run_profiled() {
    local label="$1"; shift
    local dir="$RESULTS/$label"
    mkdir -p "$dir"
    echo ">>> [$label] $@"
    rocprofv3 --pmc $COUNTERS \
        -o out -d "$dir" \
        --output-format csv \
        -- "$@" 2>&1 | grep -E "GEMM|Overlap|alone|rotating|efficiency|slowdown|Error" || true
    echo ""
}

echo "================================================================"
echo " FULL OVERLAP ANALYSIS — $(date)"
echo " GEMM: ${M}×${N}×${K} bf16 | GPU: MI300X | Steps: $STEPS"
echo "================================================================"

# =================================================================
# PART 1: Timing benchmarks (no counters, full accuracy)
# =================================================================
echo ""
echo "=== PART 1: Timing Benchmarks ==="

echo "--- 1a. WS GEMM + all_reduce (default channels) ---"
torchrun --nproc_per_node=8 \
    benchmarks/overlap.py standard \
        --backend ws \
        --gemm-m $M --gemm-n $N --gemm-k $K \
        --comm-size $M $N --collective all_reduce \
        --steps $STEPS --warmup $WARMUP \
        --output-csv "$RESULTS/timing_ws_default.csv" 2>&1 | tail -15

echo ""
echo "--- 1b. torch.matmul + all_reduce (default channels) ---"
torchrun --nproc_per_node=8 \
    benchmarks/overlap.py standard \
        --backend torch \
        --gemm-m $M --gemm-n $N --gemm-k $K \
        --comm-size $M $N --collective all_reduce \
        --steps $STEPS --warmup $WARMUP \
        --output-csv "$RESULTS/timing_torch_default.csv" 2>&1 | tail -15

echo ""
echo "--- 1c. CU-masked WS GEMM alone (272 CUs, simulating 32 lost) ---"
ROC_GLOBAL_CU_MASK=0x0000ffffffffffff \
python3 benchmarks/overlap.py trace \
    --backend ws \
    --m $M --n $N --k $K \
    --no-overlap \
    --warmup $WARMUP --steps $STEPS 2>&1 | tail -5

echo ""
echo "--- 1d. CU-masked WS GEMM alone (240 CUs, simulating 64 lost) ---"
ROC_GLOBAL_CU_MASK=0x00003fffffffffff \
python3 benchmarks/overlap.py trace \
    --backend ws \
    --m $M --n $N --k $K \
    --no-overlap \
    --warmup $WARMUP --steps $STEPS 2>&1 | tail -5

echo ""
echo "--- 1e. Full 304 CUs WS GEMM alone ---"
python3 benchmarks/overlap.py trace \
    --backend ws \
    --m $M --n $N --k $K \
    --no-overlap \
    --warmup $WARMUP --steps $STEPS 2>&1 | tail -5

# =================================================================
# PART 2: L2 + WRITEBACK counters — GEMM alone
# =================================================================
echo ""
echo "=== PART 2: Counter Collection — GEMM Alone ==="

run_profiled "ws_alone_warm" \
    python3 benchmarks/overlap.py l2-profile \
        --profile-mode gemm-alone --backend ws \
        --m $M --n $N --k $K --warmup $WARMUP --steps $STEPS

run_profiled "ws_alone_polluted_256MB" \
    python3 benchmarks/overlap.py l2-profile \
        --profile-mode gemm-polluted --backend ws \
        --m $M --n $N --k $K --pollution-mb 256 --warmup $WARMUP --steps $STEPS

# =================================================================
# PART 3: L2 + WRITEBACK counters — GEMM + RCCL
# =================================================================
echo ""
echo "=== PART 3: Counter Collection — GEMM + RCCL ==="

run_profiled "ws_rccl" \
    torchrun --nproc_per_node=8 \
    benchmarks/overlap.py l2-profile \
        --profile-mode gemm-rccl --backend ws \
        --gemm-m $M --gemm-n $N --gemm-k $K \
        --warmup $WARMUP --steps $STEPS

run_profiled "torch_rccl" \
    torchrun --nproc_per_node=8 \
    benchmarks/overlap.py l2-profile \
        --profile-mode gemm-rccl --backend torch \
        --gemm-m $M --gemm-n $N --gemm-k $K \
        --warmup $WARMUP --steps $STEPS

# =================================================================
# PART 4: CU-masked counters
# =================================================================
echo ""
echo "=== PART 4: Counter Collection — CU Masking ==="

ROC_GLOBAL_CU_MASK=0x0000ffffffffffff \
run_profiled "ws_cu272_alone" \
    python3 benchmarks/overlap.py l2-profile \
        --profile-mode gemm-alone --backend ws \
        --m $M --n $N --k $K --warmup $WARMUP --steps $STEPS

echo ""
echo "================================================================"
echo " ANALYSIS COMPLETE — Results in $RESULTS"
echo "================================================================"
ls -la "$RESULTS"/*/out_counter_collection.csv 2>/dev/null || echo "No CSV files found"

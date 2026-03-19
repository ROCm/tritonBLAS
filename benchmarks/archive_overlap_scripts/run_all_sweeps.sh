#!/bin/bash
set -e
cd /home/muhosama/tritonBLAS/benchmarks
export HIP_VISIBLE_DEVICES=6

SIZES=("4096x4096x4096" "8192x4096x16384" "16384x4096x4096" "4096x16384x4096" "256x131072x4096")
RESULTS=/home/muhosama/tritonBLAS/results/multi_size
DATASETS=/home/muhosama/tritonBLAS/datasets

mkdir -p "$RESULTS"

for SIZE in "${SIZES[@]}"; do
    YAML="$DATASETS/bench_${SIZE}.yaml"
    echo "=== $SIZE: Persistent ==="
    python3 tritonblas_matmul.py --input-yaml "$YAML" --cu-sweep --cu-sweep-max-remove 34 --output-csv "$RESULTS/${SIZE}_persistent.csv"

    echo "=== $SIZE: Work-Stealing ==="
    python3 tritonblas_matmul.py --input-yaml "$YAML" --work-stealing --counters-per-xcd 1 --cu-sweep --cu-sweep-max-remove 34 --output-csv "$RESULTS/${SIZE}_ws.csv"

    echo "=== $SIZE: torch.mm ==="
    python3 torch_matmul.py --input-yaml "$YAML" --cu-sweep --cu-sweep-max-remove 34 --output-csv "$RESULTS/${SIZE}_torch.csv"
done

echo "All sweeps complete."

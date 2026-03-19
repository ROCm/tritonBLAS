#!/bin/bash
set -e
cd /home/muhosama/tritonBLAS/benchmarks
export HIP_VISIBLE_DEVICES=6
R=/home/muhosama/tritonBLAS/results/multi_size
D=/home/muhosama/tritonBLAS/datasets

echo "=== 4096x16384x4096: WS ==="
python3 tritonblas_matmul.py --input-yaml $D/bench_4096x16384x4096.yaml --work-stealing --counters-per-xcd 1 --cu-sweep --cu-sweep-max-remove 34 --output-csv $R/4096x16384x4096_ws.csv

echo "=== 4096x16384x4096: torch ==="
python3 torch_matmul.py --input-yaml $D/bench_4096x16384x4096.yaml --cu-sweep --cu-sweep-max-remove 34 --output-csv $R/4096x16384x4096_torch.csv

echo "=== 256x131072x4096: Persistent ==="
python3 tritonblas_matmul.py --input-yaml $D/bench_256x131072x4096.yaml --cu-sweep --cu-sweep-max-remove 34 --output-csv $R/256x131072x4096_persistent.csv

echo "=== 256x131072x4096: WS ==="
python3 tritonblas_matmul.py --input-yaml $D/bench_256x131072x4096.yaml --work-stealing --counters-per-xcd 1 --cu-sweep --cu-sweep-max-remove 34 --output-csv $R/256x131072x4096_ws.csv

echo "=== 256x131072x4096: torch ==="
python3 torch_matmul.py --input-yaml $D/bench_256x131072x4096.yaml --cu-sweep --cu-sweep-max-remove 34 --output-csv $R/256x131072x4096_torch.csv

echo "All remaining sweeps complete."

#!/bin/bash
# RCCL CU sweep: vary NCCL_MAX_NCHANNELS and collect overlap results.
# Run inside Docker container.
set -e

SIZE=${1:-8192}
COMM=${2:-16384}
OUTFILE="results/rccl_cu_sweep_${SIZE}.json"
mkdir -p results

echo "[]" > "$OUTFILE"

for CH in 2 4 8 16 24 32 40 48 56; do
    echo "===== NCCL_MAX_NCHANNELS=$CH ====="
    NCCL_MAX_NCHANNELS=$CH torchrun --nproc_per_node=8 \
        benchmarks/rccl_cu_sweep.py --size $SIZE --comm_size $COMM --channels $CH \
        2>/dev/null | grep "^RESULT:" | sed 's/^RESULT://' >> "${OUTFILE}.tmp" || true
    echo "  done"
done

# Convert newline-delimited JSON to array
python3 -c "
import json, sys
results = []
with open('${OUTFILE}.tmp') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
with open('${OUTFILE}', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Wrote {len(results)} results to ${OUTFILE}')
"
rm -f "${OUTFILE}.tmp"

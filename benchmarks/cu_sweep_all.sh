#!/usr/bin/env bash
# CU sweep: launches separate processes for each CU count.
# For torch: ROC_GLOBAL_CU_MASK
# For WS/StreamK: grid-limited via total_cus parameter
# For WS-cumask: ROC_GLOBAL_CU_MASK + full grid
set -e

SZ=${1:-8192}
STEPS=30
WARMUP=15
OUT="results/plot_data/cu_sweep_${SZ}_v2.json"

echo "CU sweep ${SZ}x${SZ}x${SZ} bf16"
echo "[]" > "$OUT"

# Helper: run one point and append to JSON
run_point() {
    local BACKEND="$1"
    local CUS="$2"
    local MASK="$3"
    local EXTRA_ENV="$4"

    local RESULT
    RESULT=$(env $EXTRA_ENV python3 -c "
import torch, tritonblas, statistics, sys

torch.cuda.set_device(0)
dev = torch.device('cuda', 0)
M = N = K = $SZ
dtype = torch.bfloat16
FLOPS = 2.0 * M * N * K
s = torch.cuda.Stream(device=dev)

backend = '$BACKEND'
total_cus = $CUS if $CUS > 0 else None

A = torch.randn(M, K, dtype=dtype, device=dev)
B = torch.randn(K, N, dtype=dtype, device=dev)
C = torch.empty(M, N, dtype=dtype, device=dev)

if backend == 'torch':
    fn = lambda: torch.matmul(A, B, out=C)
    rst = None
elif backend in ('ws', 'ws-cumask'):
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev,
                                           total_cus=total_cus)
    cfg = tritonblas.matmul_preamble(sel)
    fn = lambda: tritonblas.matmul_lt(A, B, C, sel, cfg, work_stealing=True)
    rst = lambda: cfg.reset(work_stealing=True)
elif backend == 'streamk-ws':
    sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev,
                                           streamk=True, total_cus=total_cus)
    cfg = tritonblas.matmul_preamble(sel)
    fn = lambda: tritonblas.matmul_lt(A, B, C, sel, cfg, enable_streamk=True, work_stealing=True)
    rst = lambda: cfg.reset(streamk=True, work_stealing=True)

for _ in range($WARMUP):
    with torch.cuda.stream(s):
        if rst: rst()
        fn()
torch.cuda.synchronize()

times = []
for _ in range($STEPS):
    if rst:
        with torch.cuda.stream(s): rst()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record(s)
    with torch.cuda.stream(s): fn()
    en.record(s)
    torch.cuda.synchronize()
    times.append(st.elapsed_time(en))

med = statistics.median(times)
tflops = FLOPS / (med * 1e-3) / 1e12
print(f'{med:.4f} {tflops:.2f}')
" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$RESULT" ]; then
        local MS=$(echo "$RESULT" | awk '{print $1}')
        local TF=$(echo "$RESULT" | awk '{print $2}')
        python3 -c "
import json
with open('$OUT') as f: d = json.load(f)
d.append({'backend': '$BACKEND', 'cus': $CUS, 'mask': '$MASK', 'ms': $MS, 'tflops': $TF})
with open('$OUT', 'w') as f: json.dump(d, f)
"
        echo "  $BACKEND CUs=$CUS mask=$MASK: ${MS}ms ${TF}TF"
    else
        echo "  $BACKEND CUs=$CUS: FAILED"
    fi
}

# WS grid-limited sweep
echo "=== WS (grid-limited) ==="
for CUS in 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 264 272 280 288 296 304; do
    run_point "ws" "$CUS" "none" ""
done

# StreamK+WS grid-limited sweep
echo "=== StreamK+WS (grid-limited) ==="
for CUS in 24 48 72 96 120 144 168 192 216 240 264 288 304; do
    run_point "streamk-ws" "$CUS" "none" ""
done

# Torch (no mask baseline)
echo "=== torch (no mask) ==="
run_point "torch" "304" "none" ""

# Torch (CU-masked) - each subprocess gets mask before init
echo "=== torch (CU-masked) ==="
for BITS in $(seq 1 38); do
    MASK=$(python3 -c "print(hex((1<<$BITS)-1))")
    run_point "torch" "$BITS" "$MASK" "ROC_GLOBAL_CU_MASK=$MASK"
done

# WS CU-masked (full grid but hardware CUs limited)
echo "=== WS (CU-masked, full grid) ==="
for BITS in $(seq 1 38); do
    MASK=$(python3 -c "print(hex((1<<$BITS)-1))")
    run_point "ws-cumask" "0" "$MASK" "ROC_GLOBAL_CU_MASK=$MASK"
done

# Pretty-print result
python3 -c "
import json
with open('$OUT') as f: d = json.load(f)
# Reformat
out = {'size': $SZ}
for bk in ['ws', 'streamk-ws', 'torch', 'ws-cumask']:
    pts = [p for p in d if p['backend'] == bk]
    pts.sort(key=lambda x: x['cus'])
    out[bk] = pts
with open('$OUT', 'w') as f: json.dump(out, f, indent=2)
print(f'Saved $OUT ({len(d)} total points)')
"

#!/usr/bin/env bash
# run-tier.sh <smoke|pr|nightly> [out-dir] [extra pytest args...]
#
# Run a tritonBLAS test tier on a simulated AMD GPU (rocjitsu). Call it under the simulator, e.g.
# in the docker/rocjitsu image:   rjsim gfx950 -- tests/rocjitsu/run-tier.sh smoke
# Functional only: rocjitsu has no timing model.
#
# Tiers:
#   smoke    bit-exact tests at small shapes + LDS estimation        ~2.5 min at 16 CPUs, ~5 min at 4
#   pr       all bit-exact tests + the matmul/addmm correctness suites without torch.compile,
#            shapes <= 0.6 GFLOP                                     ~5-12 min at 16 CPUs
#   nightly  every upstream test except performance tests and shapes over 2.2 GFLOP,
#            including torch.compile; split it with RJ_SHARD=I/N
# Writes <out-dir>/junit.xml (default out-dir: rocjitsu-results).
# Env: RJ_JOBS (xdist workers, each its own simulated GPU; default 1), RJ_SHARD (I/N),
#      RJ_MAX_GFLOP (override the tier's per-test budget), RJ_TEST_TIMEOUT (s per test, default 1800).
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
tier=$1; shift
out=rocjitsu-results
[[ $# -ge 1 && $1 != -* ]] && { out=$1; shift; }
rjdir=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$rjdir/../.." && pwd)
up=$src/tests
mkdir -p "$out"; out=$(cd "$out" && pwd)
case $tier in
  smoke)   gflop=${RJ_MAX_GFLOP:-0.6}
           sel=("$rjdir/test_rj_exact.py" -m "not rj_full" "$up/test_lds_estimation.py") ;;
  pr)      gflop=${RJ_MAX_GFLOP:-0.6}
           sel=("$rjdir/test_rj_exact.py" "$up/test_lds_estimation.py" "$up/test_matmul_correctness.py"
                "$up/test_addmm_correctness.py" --rj-no-compile) ;;
  nightly) gflop=${RJ_MAX_GFLOP:-2.2}
           # a8w8 is skipped upstream (known bug); collecting it only adds noise.
           sel=("$up" --ignore="$up/test_matmul_a8w8_lt.py") ;;
  *) echo "unknown tier $tier" >&2; exit 2 ;;
esac
[[ -n ${RJ_SHARD:-} ]] && sel+=(--rj-shard "$RJ_SHARD")
arch=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2>/dev/null || echo unknown)
cd "$src"
# Faulting tests (e.g. test_k1_known_bug) would otherwise leave ROCr gpucore.*.gpu dumps in the repo.
ulimit -c 0
export PYTHONPATH="$src/include:$rjdir${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
# Thread-method timeouts (as in Triton's rocjitsu CI): the signal method needs the main thread
# back, which a wedged simulated queue may never give. Always run in xdist workers: a GPU fault
# aborts its process, and xdist reports that as one failed test and restarts the worker instead
# of losing the whole session.
exec python3 -m pytest -p pytest_rj -p no:cacheprovider --rj-max-gflop "$gflop" \
  --timeout "${RJ_TEST_TIMEOUT:-1800}" --timeout-method thread -rfEs --durations=25 \
  -n "${RJ_JOBS:-1}" --max-worker-restart 50 \
  --junit-xml="$out/junit.xml" -o junit_suite_name="tritonblas-rocjitsu-$arch-$tier${RJ_SHARD:+-${RJ_SHARD/\//of}}" \
  "${sel[@]}" "$@"

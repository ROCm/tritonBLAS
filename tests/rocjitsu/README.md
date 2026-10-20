# Testing tritonBLAS without a GPU (rocjitsu)

[rocjitsu](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu) runs the unmodified
ROCm user stack (HIP, torch, Triton) against a simulated AMD GPU: an `LD_PRELOAD` interposer emulates
the KFD/DRM ioctls and the kernels' ISA is executed on the CPU. tritonBLAS's tests run unchanged on
a simulated MI300X (gfx942) or MI355X (gfx950) on any x86 machine with docker.

**Functional only.** rocjitsu has no timing model: it tells you whether the bytes are right, never
how fast. Performance tests are dropped, and a time measured here is not kernel performance. A pass
is a correctness filter; real hardware stays the authority.

## Run it

```bash
docker build -t tritonblas-rocjitsu docker/rocjitsu
docker run --rm --init --cpus 16 --memory 24g --memory-swap 24g --user $(id -u):$(id -g) -e HOME=/tmp/home \
  -e RJ_JOBS=4 -e RJ_THREADS=4 -v $PWD:/work -w /work tritonblas-rocjitsu \
  rjsim gfx950 -- tests/rocjitsu/run-tier.sh smoke
```

`rjsim <gfx942|gfx950> -- <cmd>` runs a command on a simulated GPU (torch sees it as `cuda:0`).
`RJ_JOBS` is the number of pytest-xdist workers (each its own simulated GPU) and `RJ_THREADS` the
emulator threads per worker; keep `RJ_JOBS * RJ_THREADS` near the CPUs you give docker.
**Memory: each worker holds its own torch + Triton + emulator, about 5 GB (gfx950).** Size `RJ_JOBS`
to the RAM you can spare and give docker a hard cap (`--memory` = `RJ_JOBS` x 5 GB + 4 GB), so an
oversized run is OOM-killed instead of taking down a shared machine.
Results land in `rocjitsu-results/junit.xml`.

| tier | what | 16 CPUs | 4 CPUs |
|---|---|---|---|
| `smoke` | `test_rj_exact.py` small shapes + `test_lds_estimation.py` | ~2.5 min | ~5 min |
| `pr` | all of `test_rj_exact.py` + matmul/addmm correctness suites without `torch.compile`, shapes <= 0.6 GFLOP | 5-12 min | shard it |
| `nightly` | every test except performance tests and shapes > 2.2 GFLOP, incl. `torch.compile` | long | shard it |

`RJ_SHARD=I/N` splits a tier across jobs; `RJ_MAX_GFLOP` overrides the per-test size budget.
gfx942 is slower than gfx950 (304 vs 256 simulated CUs).

## What's here

- `pytest_rj.py`: pytest plugin loaded by `run-tier.sh`. Emulation runs at about 1-1.5 GFLOP/s,
  so it deselects parametrizations whose `2*m*n*k` exceeds the budget (the 4096^3+ shapes in
  `test_matmul.py` / `test_matmul_lt.py` would take minutes to hours each), drops performance
  tests, and marks known failures (below). The other test files are not modified.
- `test_rj_exact.py`: bit-exact tests sized for emulation, valid on hardware too. Inputs are
  integers in [-2, 2] so fp32 accumulation is exact and the reference is exact; no tolerances.
  Covers what the other suites only reach at large shapes: operand layouts (transposes), fp32,
  `matmul_lt`, `addmm`, `out=` into NaN-poisoned buffers (an unwritten tile fails), M/N/K tails,
  and repeated Stream-K / work-stealing launches.
- `run-tier.sh`: the tier definitions. `docker/rocjitsu/` holds the image and `rjsim`.

Each test runs in an xdist worker so a GPU fault fails that one test instead of the session, and
per-test timeouts use the thread method so a hung simulated queue still names its test.

## Known failures

Recorded as xfail so the tiers stay green and new failures stand out. `--rj-run-known-failures`
(pass it after the tier name) runs the plugin-marked ones as plain tests.

| test | status |
|---|---|
| `test_rj_exact.py::test_k1_known_bug` | K=1: Triton specializes `K == 1` to a constant, so `GemmContext.reduce_axis` computes `num_k_tiles = cdiv(1, BLOCK_K) - 1 = 0` and `tl.assume(num_k_tiles > 0)` becomes `assume(false)`; the kernel body is dropped. gfx950 memory fault, gfx942 wrong result. |
| `test_rj_exact.py::test_addmm_exact[*-streamk]` | Stream-K's non-quantized epilogue adds the bias after rounding to the output dtype, so it can be 1 ulp off `torch.addmm` at a tie. |
| every `enable_streamk=True, work_stealing=True` case in the matmul/addmm correctness suites | Wrong results under rocjitsu, up to ~all of the output. Which cases fail depends on earlier launches in the same process (a shape passes fresh and fails mid-session), so all are non-strict xfail. Not yet reproduced on hardware. |
| `*_skinny_matrices` on gfx942 (skipped) | The tests' `torch.mm` reference (hipBLASLt) never completes under rocjitsu on gfx942 for these shapes; tritonBLAS itself finishes in ~1 s. Emulator/reference issue, not tritonBLAS. |

## Arch coverage

gfx942 and gfx950: the origami version `setup.py` pins only knows those two. The image's rocjitsu
also has gfx1250, gfx1201 and gfx1100 configs for when origami supports them.

## Notes

- The image pins `triton==3.6.0`: PyPI triton 3.8.0 hangs loading `libtriton` under the emulator.
- `docker/rocjitsu/Dockerfile` pins origami to the commit in `setup.py`; keep them in sync.

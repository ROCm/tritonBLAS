# Image for running tritonBLAS tests on a CPU-emulated AMD GPU (rocjitsu). No GPU needed.
#
# Layered on Triton's own AMD CI image, which ships TheRock ROCm (hipcc, torch) with rocjitsu
# packaged as `mirage` + configs under $ROCM_ROOT/share/rocjitsu/configs (the image used by
# triton-lang/triton's integration-tests-amd-rocjitsu job), so rocjitsu is not built here.
# Adds triton, test deps, and rocm-origami from PyPI at the version pyproject.toml pins.
#
#   docker build -t tritonblas-rocjitsu docker/rocjitsu
#   docker run --rm -v $PWD:/work -w /work tritonblas-rocjitsu rjsim gfx950 -- tests/rocjitsu/run-tier.sh smoke
ARG BASE=tritonci/triton-amd-ci:rocm10.2.0a20261012-pytorch2.16.0a0-gfx942-gfx950-gfx1250-r1
FROM ${BASE}

# Keep in sync with the rocm-origami pin in pyproject.toml.
ARG ROCM_ORIGAMI_VERSION=0.0.3
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# triton pinned: PyPI triton 3.8.0 hangs loading libtriton under mirage/rocjitsu 0.3.0 (2026-10-20);
# 3.6.0 is the version validated on the shared rocjitsu image.
RUN pip3 install --no-cache-dir triton==3.6.0 numpy pandas pytest pytest-xdist pytest-timeout

# rocm-origami is published as source only, so pip compiles it here. CMake needs TheRock's ROCm on
# CMAKE_PREFIX_PATH, and g++ because the base image's CXX (system clang) lacks the compiler-rt
# builtins TheRock's hip::host links. The extension has no rpath to the HIP runtime, so the check
# imports torch first (as tritonBLAS does) to load it.
RUN rocm-sdk init >/dev/null \
    && export ROCM_PATH="$(rocm-sdk path --root)" \
    && CMAKE_PREFIX_PATH="${ROCM_PATH}" CXX=g++ \
       pip3 install --no-cache-dir "rocm-origami==${ROCM_ORIGAMI_VERSION}" \
    && python3 -c "import torch, origami"

COPY rjsim /usr/local/bin/rjsim
WORKDIR /work

# Image for running tritonBLAS tests on a CPU-emulated AMD GPU (rocjitsu). No GPU needed.
#
# Layered on Triton's own AMD CI image, which ships TheRock ROCm (hipcc, torch) with rocjitsu
# packaged as `mirage` + configs under $ROCM_ROOT/share/rocjitsu/configs (the image used by
# triton-lang/triton's integration-tests-amd-rocjitsu job), so rocjitsu is not built here.
# Adds triton, test deps, and origami at the commit setup.py pins, built with TheRock's hipcc.
#
#   docker build -t tritonblas-rocjitsu docker/rocjitsu
#   docker run --rm -v $PWD:/work -w /work tritonblas-rocjitsu rjsim gfx950 -- tests/rocjitsu/run-tier.sh smoke
ARG BASE=tritonci/triton-amd-ci:rocm10.2.0a20261012-pytorch2.16.0a0-gfx942-gfx950-gfx1250-r1
FROM ${BASE}

# Keep in sync with ORIGAMI_COMMIT in setup.py.
ARG ORIGAMI_COMMIT=66af6cd8ceb5c4540f857460dc130c9e9207de3e
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# triton pinned: PyPI triton 3.8.0 hangs loading libtriton under mirage/rocjitsu 0.3.0 (2026-10-20);
# 3.6.0 is the version validated on the shared rocjitsu image.
RUN pip3 install --no-cache-dir triton==3.6.0 numpy pandas pytest pytest-xdist pytest-timeout nanobind pybind11

RUN rocm-sdk init >/dev/null \
    && export ROCM_PATH="$(rocm-sdk path --root)" \
    && git clone -q --depth 1 --filter=blob:none --sparse --branch develop \
         https://github.com/ROCm/rocm-libraries.git /opt/origami-src \
    && cd /opt/origami-src && git fetch -q --depth 1 origin "${ORIGAMI_COMMIT}" \
    && git checkout -q "${ORIGAMI_COMMIT}" && git sparse-checkout set shared/origami \
    && pip3 install --no-cache-dir --no-build-isolation shared/origami/python \
    && python3 -c "import origami" \
    && echo "${ORIGAMI_COMMIT}" > /opt/ORIGAMI_REF

COPY rjsim /usr/local/bin/rjsim
WORKDIR /work

ARG BASE_IMAGE=rocm/pytorch-nightly:2026-03-10-rocm7.2

# ==============================================================================
# Stage 1: Base ROCm + PyTorch environment
# ==============================================================================
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        sudo nano git git-lfs wget curl \
        build-essential g++ cmake ninja-build \
        python3-pip python-is-python3 \
        rccl-dev libdw-dev inotify-tools \
        bash-completion jq \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Install tritonBLAS and dependencies
# ==============================================================================
FROM base AS tritonblas

WORKDIR /workspace

RUN pip install --no-cache-dir \
        pyyaml tqdm matplotlib pandas \
        python-pptx

COPY . /workspace/tritonBLAS

# Install tritonBLAS (editable)
WORKDIR /workspace/tritonBLAS
RUN pip install -e . 2>/dev/null || echo "tritonBLAS install deferred to runtime"

# ==============================================================================
# Stage 3: df-counters rocprofv3 overlay (optional, requires AMD internal repos)
#
# To use the custom rocprofv3 with MALL/LLC counters:
#   1. Build the df-counters Docker separately from df-counters/Dockerfile
#   2. Copy the built artifacts into this image:
#      docker build -t df-counters -f df-counters/Dockerfile df-counters/
#      docker create --name df-tmp df-counters
#      docker cp df-tmp:/opt/rocm/lib/librocprofiler-sdk.so /tmp/
#      docker cp df-tmp:/opt/rocm/lib/rocprofiler-sdk/ /tmp/
#      docker rm df-tmp
#   3. Mount or COPY these into this container at /opt/rocm/
#
# For now, we provide a mount point. At runtime:
#   docker run -v /path/to/df-rocprofv3:/opt/rocm/lib/df-override ...
# ==============================================================================

# Environment
ENV TRITONBLAS_DIR=/workspace/tritonBLAS
ENV PYTHONPATH=/workspace/tritonBLAS/include
WORKDIR /workspace/tritonBLAS

CMD ["/bin/bash"]

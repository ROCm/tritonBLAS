#!/usr/bin/env bash
# Launch the tritonBLAS overlap research Docker container
#
# Usage:
#   ./docker/run.sh                   # Interactive shell
#   ./docker/run.sh <command>          # Run a command
#
# Options (env vars):
#   NPROC=8                            # Number of GPUs (default: all)
#   DF_ROCPROF=/path/to/df-libs        # Path to df-counters rocprofv3 libs
#   IMAGE=tritonblas-research:latest    # Docker image to use

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="${IMAGE:-tritonblas-research:latest}"
NPROC="${NPROC:-all}"

# GPU access
DOCKER_ARGS=(
    --rm -it
    --network host
    --ipc host
    --shm-size 64g
    --device /dev/kfd
    --device /dev/dri
    --group-add video
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v "$REPO_DIR":/workspace/tritonBLAS
    -w /workspace/tritonBLAS
    -e HSA_NO_SCRATCH_RECLAIM=1
)

# Mount df-counters rocprofv3 if available
if [[ -n "$DF_ROCPROF" && -d "$DF_ROCPROF" ]]; then
    echo "Mounting df-counters rocprofv3 from: $DF_ROCPROF"
    DOCKER_ARGS+=(-v "$DF_ROCPROF":/opt/rocm/lib/df-override)
    DOCKER_ARGS+=(-e LD_LIBRARY_PATH=/opt/rocm/lib/df-override:/opt/rocm/lib)
fi

if [[ $# -eq 0 ]]; then
    echo "Starting interactive shell in $IMAGE"
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" /bin/bash
else
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
fi

#!/usr/bin/env bash
# Build the tritonBLAS research Docker image
#
# Usage:
#   ./docker/build.sh                    # Use latest nightly if available, else fallback
#   ./docker/build.sh rocm7.2-nightly    # Force nightly
#   ./docker/build.sh local              # Use existing rocm/pytorch:latest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

TARGET="${1:-auto}"

# Determine base image
if [[ "$TARGET" == "rocm7.2-nightly" ]]; then
    BASE="rocm/pytorch-nightly:2026-03-10-rocm7.2"
elif [[ "$TARGET" == "local" ]]; then
    BASE="rocm/pytorch:latest"
elif [[ "$TARGET" == "auto" ]]; then
    if docker image inspect rocm/pytorch-nightly:2026-03-10-rocm7.2 &>/dev/null; then
        BASE="rocm/pytorch-nightly:2026-03-10-rocm7.2"
        echo "Using nightly: $BASE"
    else
        BASE="rocm/pytorch:latest"
        echo "Nightly not available, using: $BASE"
    fi
else
    BASE="$TARGET"
fi

echo "Building tritonblas-research with base: $BASE"

docker build \
    --build-arg BASE_IMAGE="$BASE" \
    -t tritonblas-research:latest \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$REPO_DIR"

echo ""
echo "Build complete!"
echo "Run with: ./docker/run.sh"

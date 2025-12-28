#!/bin/bash
# Build the RLBench container for MSI
# Usage: ./build_container.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="rlbench.sif"

echo "==================================="
echo "Building RLBench container"
echo "This will take ~20-30 minutes"
echo "==================================="

# Load CUDA module if available (MSI)
if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck source=/etc/profile.d/modules.sh
    . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda/12.1.1 || true
fi

# Ensure cache directories exist (prefer scratch to avoid home quota)
DEFAULT_SCRATCH="${SCRATCH:-/scratch.global/$USER}"
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$DEFAULT_SCRATCH/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$DEFAULT_SCRATCH/apptainer_tmp}
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Build the container
apptainer build --fakeroot $SCRIPT_DIR/$CONTAINER_NAME $SCRIPT_DIR/rlbench.def

echo "==================================="
echo "Build complete: $SCRIPT_DIR/$CONTAINER_NAME"
echo "==================================="

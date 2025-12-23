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

# Ensure cache directories exist
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$HOME/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$HOME/apptainer_tmp}
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Build the container
apptainer build --fakeroot $SCRIPT_DIR/$CONTAINER_NAME $SCRIPT_DIR/rlbench.def

echo "==================================="
echo "Build complete: $SCRIPT_DIR/$CONTAINER_NAME"
echo "==================================="

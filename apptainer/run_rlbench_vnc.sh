#!/bin/bash
# Run RLBench evaluation with VNC GUI on MSI
# Usage: ./run_rlbench_vnc.sh [command]
# Example: ./run_rlbench_vnc.sh python -m src.evaluation.eval --task_name=lift_tray

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER="$SCRIPT_DIR/rlbench.sif"

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Run build_container.sh first"
    exit 1
fi

# ============================================
# MSI-specific: Library bindings for Rocky Linux -> Ubuntu container
# Same approach as Isaac Gym setup
# ============================================

# Find NVIDIA driver libraries on host (Rocky Linux paths)
NVIDIA_LIB_DIR="/usr/lib64"
if [ ! -d "$NVIDIA_LIB_DIR" ]; then
    NVIDIA_LIB_DIR="/usr/lib/x86_64-linux-gnu"
fi

# Collect library bindings
BIND_LIBS=""

# Core NVIDIA libraries needed for OpenGL rendering
NVIDIA_LIBS=(
    "libGLX_nvidia.so"
    "libEGL_nvidia.so"
    "libnvidia-glcore.so"
    "libnvidia-tls.so"
    "libnvidia-glsi.so"
    "libGLdispatch.so"
    "libOpenGL.so"
    "libGLX.so"
    "libEGL.so"
)

for lib in "${NVIDIA_LIBS[@]}"; do
    # Find the library (may have version suffix)
    found=$(find $NVIDIA_LIB_DIR -name "${lib}*" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        # Bind to Ubuntu-expected path inside container
        BIND_LIBS="$BIND_LIBS --bind $found:/usr/lib/x86_64-linux-gnu/$(basename $found)"
    fi
done

# Bind Vulkan ICD files if present (CoppeliaSim may use them)
if [ -d "/usr/share/vulkan/icd.d" ]; then
    BIND_LIBS="$BIND_LIBS --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d"
fi

# ============================================
# Project bindings
# ============================================

# Bind the project directory
PROJECT_BINDS="--bind $PROJECT_DIR:/workspace"

# Bind PyRep and RLBench for installation
PYREP_BIND="--bind $PROJECT_DIR/PyRep:/pyrep_src"
RLBENCH_BIND="--bind $PROJECT_DIR/RLBench:/rlbench_src"

# ============================================
# Run container
# ============================================

echo "==================================="
echo "Starting RLBench container with VNC"
echo "Project: $PROJECT_DIR"
echo "==================================="

# If no command specified, just start shell
if [ $# -eq 0 ]; then
    CMD="bash"
else
    CMD="$@"
fi

# Run with GPU and all bindings
apptainer run --nv \
    $BIND_LIBS \
    $PROJECT_BINDS \
    $PYREP_BIND \
    $RLBENCH_BIND \
    --pwd /workspace \
    $CONTAINER \
    bash -c "
        # Install PyRep and RLBench on first run
        if [ ! -f /tmp/.pyrep_installed ]; then
            echo 'Installing PyRep...'
            cd /pyrep_src && pip install -r requirements.txt && pip install . && touch /tmp/.pyrep_installed
        fi
        if [ ! -f /tmp/.rlbench_installed ]; then
            echo 'Installing RLBench...'
            cd /rlbench_src && pip install -r requirements.txt && pip install . && touch /tmp/.rlbench_installed
        fi
        
        # Install bimanual project
        cd /workspace && pip install -e . 2>/dev/null || true
        
        # Run user command
        $CMD
    "

echo "==================================="
echo "Container exited"
echo "==================================="

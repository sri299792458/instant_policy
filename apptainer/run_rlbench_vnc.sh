#!/bin/bash
# Run RLBench evaluation with VNC GUI on MSI HPC
# Usage: ./run_rlbench_vnc.sh [command]
# Example: ./run_rlbench_vnc.sh python -m src.evaluation.eval --task_name=lift_tray

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER="$SCRIPT_DIR/rlbench.sif"
DISPLAY_NUM="${RLBENCH_DISPLAY:-1}"
VNC_PORT="${RLBENCH_VNC_PORT:-5900}"

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Run build_container.sh first"
    exit 1
fi

# ============================================
# Library bindings for Rocky Linux -> Ubuntu container
# ============================================
NVIDIA_LIB_DIR="/usr/lib64"
[ ! -d "$NVIDIA_LIB_DIR" ] && NVIDIA_LIB_DIR="/usr/lib/x86_64-linux-gnu"

BIND_LIBS=""
for lib in libGLX_nvidia.so libEGL_nvidia.so libnvidia-glcore.so libnvidia-tls.so \
           libnvidia-glsi.so libGLdispatch.so libOpenGL.so libGLX.so libEGL.so; do
    found=$(find $NVIDIA_LIB_DIR -name "${lib}*" 2>/dev/null | head -1)
    [ -n "$found" ] && BIND_LIBS="$BIND_LIBS --bind $found:/usr/lib/x86_64-linux-gnu/$(basename $found)"
done

[ -d "/usr/share/vulkan/icd.d" ] && BIND_LIBS="$BIND_LIBS --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d"

# ============================================
# Project bindings
# ============================================
PROJECT_BINDS="--bind $PROJECT_DIR:/workspace"
PYREP_BIND="--bind $PROJECT_DIR/external/PyRep:/pyrep_src"
RLBENCH_BIND="--bind $PROJECT_DIR/external/RLBench:/rlbench_src"
SCRATCH_BIND=""
[ -d "/scratch.global/$USER" ] && SCRATCH_BIND="--bind /scratch.global/$USER:/scratch.global/$USER"

# ============================================
# Cleanup stale processes
# ============================================
cleanup_stale_processes() {
    if command -v ps >/dev/null 2>&1; then
        for pattern in "Xvfb :$DISPLAY_NUM" "x11vnc.*-rfbport $VNC_PORT" "fluxbox"; do
            pid=$(ps -u "$USER" -o pid= -o args= | awk "/$pattern/{print \$1; exit}")
            [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
        done
    fi
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"
}

echo "==================================="
echo "Starting RLBench container with VNC"
echo "Project: $PROJECT_DIR"
echo "==================================="

cleanup_stale_processes

# Default command
CMD="${@:-bash}"

# Run container
apptainer exec --nv \
    $BIND_LIBS \
    $PROJECT_BINDS \
    $PYREP_BIND \
    $RLBENCH_BIND \
    $SCRATCH_BIND \
    --pwd /workspace \
    $CONTAINER \
    bash -c "
        # Start Xvfb + VNC
        export DISPLAY=:$DISPLAY_NUM
        Xvfb :$DISPLAY_NUM -screen 0 1280x1024x24 &
        sleep 2
        x11vnc -display :$DISPLAY_NUM -forever -nopw -rfbport $VNC_PORT -noxdamage -nowf &
        sleep 1
        fluxbox &
        sleep 1

        echo '==================================='
        echo 'VNC server running on port $VNC_PORT'
        echo 'Connect via: ssh -L $VNC_PORT:<node>:$VNC_PORT user@agate.msi.umn.edu'
        echo 'Then open VNC viewer to localhost:$VNC_PORT'
        echo '==================================='

        # Environment setup
        export PYTHONPATH=/workspace/external:\$PYTHONPATH
        
        # CoppeliaSim setup (copy to writable path)
        if [ ! -d /workspace/.coppeliasim ]; then
            echo 'Copying CoppeliaSim to /workspace/.coppeliasim...'
            cp -r /opt/CoppeliaSim /workspace/.coppeliasim
        fi
        export COPPELIASIM_ROOT=/workspace/.coppeliasim
        export LD_LIBRARY_PATH=\$COPPELIASIM_ROOT:\$LD_LIBRARY_PATH
        export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT

        # Persistent install markers (not /tmp which doesn't persist across nodes)
        mkdir -p /workspace/.installed

        # Install PyRep (PerAct2 bimanual fork)
        if [ ! -f /workspace/.installed/.pyrep_ok ]; then
            pip uninstall -y pyrep rlbench 2>/dev/null || true
            pip install --no-cache-dir 'cffi>=1.14.0' setuptools wheel numpy
            
            echo 'Installing PyRep from /pyrep_src...'
            cd /pyrep_src
            [ -z \"\$COPPELIASIM_ROOT\" ] || [ ! -d \"\$COPPELIASIM_ROOT\" ] && { echo 'ERROR: COPPELIASIM_ROOT invalid!'; exit 1; }
            pip install --no-cache-dir -r requirements.txt
            pip install --no-cache-dir --no-build-isolation --force-reinstall . || {
                python setup.py build_ext --inplace && pip install --no-cache-dir --no-deps .
            }
            touch /workspace/.installed/.pyrep_ok
        fi
        
        # Install RLBench (PerAct2 bimanual fork)
        if [ ! -f /workspace/.installed/.rlbench_ok ]; then
            echo 'Installing RLBench from /rlbench_src...'
            cd /rlbench_src
            pip install --no-cache-dir -r requirements.txt
            pip install --no-cache-dir --force-reinstall .
            touch /workspace/.installed/.rlbench_ok
        fi
        
        # Verify bimanual support
        python -c 'from pyrep.robots.arms.dual_panda import PandaLeft; print(\"PyRep bimanual support: OK\")' || {
            echo 'ERROR: PyRep missing bimanual support! Check external/PyRep is PerAct2 fork.'
            exit 1
        }
        
        # Install project
        cd /workspace && pip install -e . 2>/dev/null || true
        
        # Run user command
        $CMD
    "

echo "==================================="
echo "Container exited"
echo "==================================="

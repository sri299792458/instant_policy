# RLBench Evaluation on UMN MSI

Setup for running bimanual Instant Policy evaluation with CoppeliaSim/PyRep/RLBench on UMN MSI cluster.

## Prerequisites

- Access to UMN MSI cluster (agate)
- GPU partition access (v100 or a100)
- VNC viewer on your local machine ([RealVNC](https://www.realvnc.com/en/connect/download/viewer/))

## Quick Start

### Step 1: Build Container (one-time, ~20 min)

```bash
# Request GPU node
srun --time=02:00:00 -p v100 --gres=gpu:v100:1 --mem=64gb --cpus-per-task=8 --pty bash

# Setup build environment
export APPTAINER_CACHEDIR=$HOME/apptainer_cache
export APPTAINER_TMPDIR=$HOME/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Load CUDA and build
module load cuda/12.1.1
cd apptainer/
chmod +x build_container.sh
./build_container.sh
```

Note: `run_rlbench_vnc.sh` binds `external/PyRep` and `external/RLBench` into the container.

### Step 2: Run Evaluation

```bash
chmod +x run_rlbench_vnc.sh

# Interactive shell
./run_rlbench_vnc.sh

# Or run evaluation directly
./run_rlbench_vnc.sh python -m src.evaluation.eval --task_name=lift_tray --num_rollouts=5
```

Note the compute node name (e.g., `cn2105`).

### Step 3: Connect via VNC

On your local machine:
```bash
# SSH tunnel (replace cn2105 with your node)
ssh -L 5900:cn2105:5900 your_username@agate.msi.umn.edu
```

Open VNC viewer → connect to `localhost:5900`

## Container Contents

| Component   | Version         | Purpose                         |
| ----------- | --------------- | ------------------------------- |
| CoppeliaSim | 4.1.0           | Robot simulation                |
| PyRep       | latest          | Python interface to CoppeliaSim |
| RLBench     | PerAct2 fork    | Bimanual task benchmark         |
| PyTorch     | 2.2 + CUDA 12.1 | Model inference                 |
| PyG         | latest          | Graph neural networks           |

## Troubleshooting

**Container build fails:**
- Ensure you're on a GPU node with enough memory (64GB recommended)
- Check APPTAINER_TMPDIR has enough space

**VNC shows black screen:**
- Wait a few seconds for Xvfb to initialize
- Try refreshing VNC connection

**OpenGL errors:**
- The library bindings handle Rocky Linux → Ubuntu mismatch
- If issues persist, check `nvidia-smi` works on the node

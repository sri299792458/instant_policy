#!/bin/bash
#SBATCH --job-name=rlbench_build
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --account=kdesingh
#SBATCH -o logs/rlbench_build-%j.out
#SBATCH -e logs/rlbench_build-%j.err

set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "RLBench Container Build"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "=================================================="

# Load modules (safe even if build script also tries to load CUDA)
module load cuda/12.1.1

# Prefer scratch for Apptainer temp/cache to avoid home quota
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/scratch.global/$USER/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/scratch.global/$USER/apptainer_tmp}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

REPO_DIR="${REPO_DIR:-$HOME/bimanual_instant_policy}"
APPTAINER_DIR="$REPO_DIR/apptainer"

cd "$APPTAINER_DIR"
rm -f rlbench.sif
./build_container.sh

echo "--------------------"
echo "Build finished at $(date)"

#!/bin/bash
#SBATCH --job-name=bimanual_ip_full
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=kdesingh
#SBATCH -o logs/bimanual_ip_full-%j.out
#SBATCH -e logs/bimanual_ip_full-%j.err

set -euo pipefail

# Create logs directory
mkdir -p logs

echo "=================================================="
echo "Bimanual Instant Policy Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "=================================================="

# Load modules and activate environment
module load cuda/12.1.1
module load conda
source activate ip

# Print GPU information to the log
echo "NVIDIA GPU Info:"
nvidia-smi
echo "--------------------"

# Change to repo directory
cd ~/bimanual_instant_policy

# Training configuration
RUN_NAME="bimanual_ip_full"
SAVE_ROOT="/scratch.global/kanth042/ip/checkpoints"
WANDB_DIR="/scratch.global/kanth042/ip/wandb"

python -u -m scripts.train \
  --run_name "${RUN_NAME}" \
  --use_pseudo_demos 1 \
  --online_pseudo_demos 1 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_steps 2000000 \
  --num_workers 8 \
  --record 1 \
  --use_wandb 1 \
  --save_root "${SAVE_ROOT}" \
  --wandb_dir "${WANDB_DIR}" \
  --log_every_n_steps 100 \
  --val_check_interval 10000 \
  --save_every_steps 50000 \
  --save_top_k 3 \
  --grad_norm_log_every 100 \
  --throughput_log_every 50

echo "--------------------"
echo "Job finished at $(date)"

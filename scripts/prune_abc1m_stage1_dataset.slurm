#!/bin/bash
#SBATCH --job-name=abc1m_prune_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=CPU
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
DATASET_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

echo "[slurm] start $(date '+%F %T')"
echo "[slurm] repo=$REPO_DIR"
echo "[slurm] dataset_root=$DATASET_ROOT"

python scripts/prune_abc1m_stage1_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --quality-resolution 64 \
    --max-voxel-components 32 \
    --max-small-component-ratio 0.05 \
    --low-ratio-min 0.9 \
    --low-ratio-max 0.95 \
    --low-ratio-component-count-threshold 16 \
    --progress-interval 2000 \
    --delete-progress-interval 2000

echo "[slurm] end $(date '+%F %T')"
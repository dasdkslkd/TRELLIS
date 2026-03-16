#!/bin/bash
#SBATCH --job-name=abc1m_ss80_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_A800_8
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
ABC_ROOT=/public/home/pb22000140/xjn/datasets/ABC-1M
OUTPUT_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_trellis_split

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

python dataset_toolkits/build_abc1m_sparse_structure.py \
    --abc-root "$ABC_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --resolution 64 \
    --n-sample-points 24 \
    --target-count 100000 \
    --val-count 2000 \
    --batch-size 8
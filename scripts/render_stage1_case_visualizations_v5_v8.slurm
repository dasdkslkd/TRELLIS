#!/bin/bash
#SBATCH --job-name=trellis_stage1_caseviz_v5v8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_H800_8
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
OUTPUT_DIR=$REPO_DIR/outputs/vae/stage1_case_visualizations_plain_v5_v8

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

python scripts/visualize_stage1_vae_cases.py \
    --device cuda \
    --checkpoint-source plain \
    --experiments abc1m_v5 abc1m_v8 \
    --k-best 10 \
    --k-middle 10 \
    --k-worst 10 \
    --output-root "$OUTPUT_DIR"
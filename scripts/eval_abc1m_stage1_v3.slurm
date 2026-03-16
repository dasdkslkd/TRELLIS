#!/bin/bash
#SBATCH --job-name=abc1m_s1_v3_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_H800_8
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
DATA_DIR=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_trellis_split/train
OUTPUT_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/eval/abc1m_stage1_v3
ENCODER_CKPT=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_stage1_bce_v3/ckpts/encoder_ema0.9999_step0001500.pt
DECODER_CKPT=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_stage1_bce_v3/ckpts/decoder_stage1_ema0.9999_step0001500.pt

mkdir -p "$REPO_DIR/logs"
mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

python evaluate_stage1.py \
  --config "$REPO_DIR/configs/vae/ss_stage1_fusion360_80ch_bce_v3.json" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --val_data_dir "$DATA_DIR" \
  --encoder_ckpt "$ENCODER_CKPT" \
  --decoder_stage1_ckpt "$DECODER_CKPT" \
  --num_gpus 1
#!/bin/bash
#SBATCH --job-name=trellis_f360_s1bce_v3
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
CONFIG_PATH=$REPO_DIR/configs/vae/ss_stage1_fusion360_80ch_bce_v3.json
TRAIN_DATA=/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/train
VAL_DATA=/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val
OUTPUT_DIR=$REPO_DIR/outputs/vae/fusion360_stage1_bce_v3

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

EXTRA_ARGS=""
if [[ "${TRYRUN:-0}" == "1" ]]; then
    EXTRA_ARGS="--tryrun"
fi

python train.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$TRAIN_DATA" \
    --val_data_dir "$VAL_DATA" \
    --num_gpus 1 \
    $EXTRA_ARGS
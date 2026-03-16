#!/bin/bash
#SBATCH --job-name=trellis_mix_s1_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_A800_8
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
CONFIG_PATH=$REPO_DIR/configs/vae/ss_stage1_mix_fusion360_abc1m_80ch_v2.json
F360_TRAIN=/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/train
F360_VAL=/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val
ABC1M_TRAIN=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/train
ABC1M_VAL=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/val
TRAIN_DATA=${F360_TRAIN},${ABC1M_TRAIN}
VAL_DATA=${F360_VAL},${ABC1M_VAL}
OUTPUT_DIR=$REPO_DIR/outputs/vae/mix_fusion360_abc1m_stage1_v2

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
#!/bin/bash
#SBATCH --job-name=trellis_abc1m_s1_v5p
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
CONFIG_PATH=$REPO_DIR/configs/vae/ss_stage1_abc1m_80ch_v5.json
TRAIN_DATA=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/train
VAL_DATA=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/val
OUTPUT_DIR=$REPO_DIR/outputs/vae/abc1m_stage1_v5_pruned

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

echo "[slurm] start $(date '+%F %T')"
echo "[slurm] config=$CONFIG_PATH"
echo "[slurm] train_data=$TRAIN_DATA"
echo "[slurm] val_data=$VAL_DATA"
echo "[slurm] output_dir=$OUTPUT_DIR"

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

echo "[slurm] end $(date '+%F %T')"
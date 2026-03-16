#!/bin/bash
#SBATCH --job-name=f360_joint_cases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_H800_8
#SBATCH --nodelist=gnode1
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
cd "$REPO_DIR"

python scripts/visualize_joint_vae_cases.py \
    --config /public/home/pb22000140/TRELLIS-new/TRELLIS-main/configs/vae/ss_two_stage_joint_fusion360_80ch_stage1warm_v1.json \
    --output-dir /public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_ss_two_stage_joint_stage1warm_v1 \
    --val-data-dir /public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val \
    --title "Fusion360 Joint Stage1Warm v1" \
    --case-name fusion360_stage1warm_v1 \
    --checkpoint-source plain \
    --rank-by miou_occ \
    --k-best 3 \
    --k-middle 3 \
    --k-worst 3
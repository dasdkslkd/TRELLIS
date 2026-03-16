#!/bin/bash
#SBATCH --job-name=abc1m_ss80_v2_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=CPU
#SBATCH --array=0-2
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%A_%a.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
ABC_ROOT=/public/home/pb22000140/xjn/datasets/ABC-1M
OUTPUT_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2
PIPELINE_ROOT=$OUTPUT_ROOT/_pipeline

mkdir -p "$REPO_DIR/logs" "$PIPELINE_ROOT/scan"
cd "$REPO_DIR"

python dataset_toolkits/build_abc1m_sparse_structure_v2.py scan \
    --abc-root "$ABC_ROOT" \
    --scan-root "$PIPELINE_ROOT/scan" \
    --num-shards 3 \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --batch-size 16 \
    --bit 10 \
    --max-edge 1000
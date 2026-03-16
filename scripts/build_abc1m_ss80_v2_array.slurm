#!/bin/bash
#SBATCH --job-name=abc1m_ss80_v2_build
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
OUTPUT_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2
PIPELINE_ROOT=$OUTPUT_ROOT/_pipeline
SELECTED_MANIFEST=$PIPELINE_ROOT/selected_manifest.jsonl

mkdir -p "$REPO_DIR/logs" "$PIPELINE_ROOT/cache"
cd "$REPO_DIR"

python dataset_toolkits/build_abc1m_sparse_structure_v2.py build \
    --selected-manifest "$SELECTED_MANIFEST" \
    --cache-root "$PIPELINE_ROOT/cache" \
    --num-shards 3 \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --batch-size 8 \
    --resolution 64 \
    --n-sample-points 24 \
    --bit 10 \
    --max-edge 1000
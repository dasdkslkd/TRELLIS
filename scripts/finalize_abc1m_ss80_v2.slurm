#!/bin/bash
#SBATCH --job-name=abc1m_ss80_v2_finalize
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
OUTPUT_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2
PIPELINE_ROOT=$OUTPUT_ROOT/_pipeline
SELECTED_MANIFEST=$PIPELINE_ROOT/selected_manifest.jsonl

mkdir -p "$REPO_DIR/logs" "$OUTPUT_ROOT"
cd "$REPO_DIR"

python dataset_toolkits/build_abc1m_sparse_structure_v2.py finalize-dataset \
    --selected-manifest "$SELECTED_MANIFEST" \
    --cache-root "$PIPELINE_ROOT/cache" \
    --output-root "$OUTPUT_ROOT" \
    --target-count 100000 \
    --val-count 2000 \
    --resolution 64 \
    --n-sample-points 24
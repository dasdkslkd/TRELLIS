#!/bin/bash
#SBATCH --job-name=abc1m_ss80_v2_select
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

mkdir -p "$REPO_DIR/logs" "$PIPELINE_ROOT"
cd "$REPO_DIR"

python dataset_toolkits/build_abc1m_sparse_structure_v2.py finalize-selection \
    --scan-root "$PIPELINE_ROOT/scan" \
    --selected-manifest "$SELECTED_MANIFEST" \
    --target-count 100000 \
    --reserve-count 10000
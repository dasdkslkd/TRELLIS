#!/bin/bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prune_job_id>"
    exit 1
fi

PRUNE_JOB_ID="$1"
REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
TRAIN_SCRIPT=$REPO_DIR/scripts/train_abc1m_stage1_v5_pruned.slurm

cd "$REPO_DIR"

echo "[submit] waiting on prune job: $PRUNE_JOB_ID"
sbatch --dependency=afterok:"$PRUNE_JOB_ID" "$TRAIN_SCRIPT"
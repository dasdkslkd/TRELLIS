#!/bin/bash

source ~/.bashrc
conda init >/dev/null 2>&1
conda activate trellis
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_JIT=0

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
OUTPUT_ROOT=/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2
PIPELINE_ROOT=$OUTPUT_ROOT/_pipeline
SELECTED_MANIFEST=$PIPELINE_ROOT/selected_manifest.jsonl
SELECTED_SUMMARY=$PIPELINE_ROOT/selected_manifest.summary.json
SCAN_SUMMARY=$PIPELINE_ROOT/scan/summary_shard00.json
LOG_DIR=$REPO_DIR/logs

mkdir -p "$LOG_DIR" "$PIPELINE_ROOT" "$PIPELINE_ROOT/cache"
cd "$REPO_DIR"

CANDIDATE_COUNT=$(python - <<'PY'
import json
from pathlib import Path
path = Path('/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/_pipeline/selected_manifest.summary.json')
summary = json.loads(path.read_text())
print(summary['candidate_count'])
PY
)

if [[ ! -f "$PIPELINE_ROOT/selected_manifest_100k_backup.jsonl" && -f "$SELECTED_MANIFEST" ]]; then
    cp "$SELECTED_MANIFEST" "$PIPELINE_ROOT/selected_manifest_100k_backup.jsonl"
fi
if [[ ! -f "$PIPELINE_ROOT/selected_manifest_100k_backup.summary.json" && -f "$SELECTED_SUMMARY" ]]; then
    cp "$SELECTED_SUMMARY" "$PIPELINE_ROOT/selected_manifest_100k_backup.summary.json"
fi

python dataset_toolkits/build_abc1m_sparse_structure_v2.py finalize-selection \
    --scan-root "$PIPELINE_ROOT/scan" \
    --selected-manifest "$SELECTED_MANIFEST" \
    --target-count "$CANDIDATE_COUNT" \
    --reserve-count 0

run_build_shard() {
    local shard_id="$1"
    local log_path="$LOG_DIR/abc1m_ss80_v2_full_build_shard${shard_id}.log"
    python dataset_toolkits/build_abc1m_sparse_structure_v2.py build \
        --selected-manifest "$SELECTED_MANIFEST" \
        --cache-root "$PIPELINE_ROOT/cache" \
        --num-shards 3 \
        --shard-id "$shard_id" \
        --batch-size 8 \
        --resolution 64 \
        --n-sample-points 24 \
        --bit 10 \
        --max-edge 1000 \
        >"$log_path" 2>&1
}

run_build_shard 0 &
pid0=$!
run_build_shard 1 &
pid1=$!
run_build_shard 2 &
pid2=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"

SUCCESS_COUNT=$(python - <<'PY'
import json
from pathlib import Path
cache_root = Path('/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/_pipeline/cache')
success = 0
for path in sorted(cache_root.glob('build_records_shard*.jsonl')):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('status') in {'success', 'cached'}:
                success += 1
print(success)
PY
)

if [[ "$SUCCESS_COUNT" -le 2000 ]]; then
    echo "Insufficient successful tensors: $SUCCESS_COUNT" >&2
    exit 1
fi

python dataset_toolkits/build_abc1m_sparse_structure_v2.py finalize-dataset \
    --selected-manifest "$SELECTED_MANIFEST" \
    --cache-root "$PIPELINE_ROOT/cache" \
    --output-root "$OUTPUT_ROOT" \
    --target-count "$SUCCESS_COUNT" \
    --val-count 2000 \
    --resolution 64 \
    --n-sample-points 24

echo "FULL_REBUILD_DONE success_count=$SUCCESS_COUNT"

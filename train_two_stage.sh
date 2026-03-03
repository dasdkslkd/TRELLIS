#!/bin/bash
# ============================================================
# Two-Stage Sparse Structure VAE Training Script
# ============================================================
#
# 数据布局: 83通道体素 = 8 (占有场+分类) + 75 (25采样点×3坐标)
#   Stage 1: ch0 占有场(dice loss) + ch1-7 one-hot分类(cross-entropy)
#   Stage 2: ch8-82 采样点坐标(MSE loss, 按占有场mask)
#
# 三种训练模式:
#   1. joint  - 联合训练 (encoder + decoder_stage1 + decoder_stage2)
#   2. stage1 - 仅阶段一 (encoder + decoder_stage1)
#   3. stage2 - 仅阶段二 (冻结encoder + decoder_stage1, 训练decoder_stage2)
#
# 用法:
#   bash train_two_stage.sh joint    # 联合训练
#   bash train_two_stage.sh stage1   # 仅阶段一
#   bash train_two_stage.sh stage2   # 仅阶段二 (需先完成阶段一)
# ============================================================

set -e

MODE=${1:-joint}  # 默认联合训练
DATA_DIR=${2:-./data/}
VAL_DATA_DIR=${3:-}

echo "==========================================="
echo " Two-Stage SS VAE Training"
echo " Mode: ${MODE}"
echo " Data: ${DATA_DIR}"
echo "==========================================="

case $MODE in
    joint)
        CONFIG="configs/vae/ss_two_stage_joint.json"
        OUTPUT_DIR="output/two_stage_joint"
        ;;
    stage1)
        CONFIG="configs/vae/ss_two_stage_phase1.json"
        OUTPUT_DIR="output/two_stage_phase1"
        ;;
    stage2)
        CONFIG="configs/vae/ss_two_stage_phase2.json"
        OUTPUT_DIR="output/two_stage_phase2"
        
        # Check if phase1 checkpoint exists
        PHASE1_DIR="output/two_stage_phase1"
        if [ ! -d "${PHASE1_DIR}/ckpts" ]; then
            echo "ERROR: Phase 1 checkpoint not found at ${PHASE1_DIR}/ckpts"
            echo "Please run stage1 first: bash train_two_stage.sh stage1"
            exit 1
        fi
        
        # Find latest phase1 checkpoint step
        LATEST_STEP=$(ls ${PHASE1_DIR}/ckpts/misc_step*.pt 2>/dev/null | \
            sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
        
        if [ -z "$LATEST_STEP" ]; then
            echo "ERROR: No checkpoint found in ${PHASE1_DIR}/ckpts"
            exit 1
        fi
        
        echo "Using Phase 1 checkpoint at step ${LATEST_STEP}"
        echo "  encoder:        ${PHASE1_DIR}/ckpts/encoder_ema0.9999_step${LATEST_STEP}.pt"
        echo "  decoder_stage1: ${PHASE1_DIR}/ckpts/decoder_stage1_ema0.9999_step${LATEST_STEP}.pt"
        
        # Update finetune_ckpt paths in config using a temp file
        TEMP_CONFIG=$(mktemp)
        python3 -c "
import json, sys
with open('${CONFIG}') as f:
    cfg = json.load(f)
cfg['trainer']['args']['finetune_ckpt'] = {
    'encoder': '${PHASE1_DIR}/ckpts/encoder_ema0.9999_step${LATEST_STEP}.pt',
    'decoder_stage1': '${PHASE1_DIR}/ckpts/decoder_stage1_ema0.9999_step${LATEST_STEP}.pt'
}
with open('${TEMP_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=4)
"
        CONFIG="${TEMP_CONFIG}"
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: bash train_two_stage.sh [joint|stage1|stage2] [data_dir] [val_data_dir]"
        exit 1
        ;;
esac

# Build command
CMD="python train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR}"

if [ -n "${VAL_DATA_DIR}" ]; then
    CMD="${CMD} --val_data_dir ${VAL_DATA_DIR}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

eval ${CMD}

# Cleanup temp config if used
if [ -n "${TEMP_CONFIG:-}" ] && [ -f "${TEMP_CONFIG}" ]; then
    rm -f "${TEMP_CONFIG}"
fi

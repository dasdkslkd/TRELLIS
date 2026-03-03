#!/bin/bash
# Custom Sparse Structure Flow Matching 训练示例脚本
# 
# 使用方法：
#   1. 修改 DATA_DIR 为你的数据集路径
#   2. 选择训练模式（取消注释对应的配置）
#   3. 运行: bash train_custom_ss_flow.sh

# ==================== 配置参数 ====================

# 数据集路径（必须修改）
DATA_DIR="/path/to/your/custom_dataset"

# 输出目录
OUTPUT_DIR="outputs/custom_ss_flow"

# GPU数量
NUM_GPUS=8

# 是否从检查点恢复（如果有的话）
RESUME=false
CHECKPOINT="latest"  # 或者指定具体的step数，如 "10000"

# ==================== 选择训练配置 ====================

# 1. 无条件生成（推荐首次训练）
CONFIG="configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json"
MODE="uncond"

# 2. 文本条件生成 - Base模型
# CONFIG="configs/generation/ss_flow_custom_txt_dit_B_16l8_fp16.json"
# MODE="text_base"

# 3. 文本条件生成 - Large模型
# CONFIG="configs/generation/ss_flow_custom_txt_dit_L_16l8_fp16.json"
# MODE="text_large"

# 4. 图像条件生成 - Large模型
# CONFIG="configs/generation/ss_flow_custom_img_dit_L_16l8_fp16.json"
# MODE="image_large"

# ==================== 验证数据集（可选但推荐） ====================

echo "=================================================="
echo "Custom Sparse Structure Flow Matching 训练"
echo "=================================================="
echo "配置文件: $CONFIG"
echo "数据集: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "训练模式: $MODE"
echo ""

# 询问是否验证数据集
read -p "是否先验证数据集？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在验证数据集..."
    
    # 根据模式设置验证参数
    if [[ $MODE == "uncond" ]]; then
        VALIDATE_MODE="uncond"
    elif [[ $MODE == "text_"* ]]; then
        VALIDATE_MODE="text"
    elif [[ $MODE == "image_"* ]]; then
        VALIDATE_MODE="image"
    fi
    
    python validate_custom_ss_dataset.py "$DATA_DIR" --mode "$VALIDATE_MODE"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ 数据集验证失败！请修复数据集后再训练。"
        exit 1
    fi
    
    echo ""
    read -p "数据集验证通过！按回车继续训练..." 
fi

# ==================== 构建训练命令 ====================

TRAIN_CMD="python train.py \
    --config $CONFIG \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --num_gpus $NUM_GPUS"

# 如果需要恢复训练
if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD \
    --load_dir $OUTPUT_DIR \
    --ckpt $CHECKPOINT"
fi

# ==================== 开始训练 ====================

echo ""
echo "=================================================="
echo "开始训练"
echo "=================================================="
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""
echo "训练日志将保存到: $OUTPUT_DIR/logs"
echo "检查点将保存到: $OUTPUT_DIR/ckpts"
echo ""
echo "监控训练进度（在另一个终端运行）:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "=================================================="
echo ""

# 询问确认
read -p "确认开始训练？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 记录训练开始时间
    echo "训练开始时间: $(date)" > "$OUTPUT_DIR/training_log.txt"
    echo "配置: $CONFIG" >> "$OUTPUT_DIR/training_log.txt"
    echo "数据集: $DATA_DIR" >> "$OUTPUT_DIR/training_log.txt"
    echo "" >> "$OUTPUT_DIR/training_log.txt"
    
    # 执行训练
    eval $TRAIN_CMD 2>&1 | tee -a "$OUTPUT_DIR/training_log.txt"
    
    # 记录训练结束时间
    echo "" >> "$OUTPUT_DIR/training_log.txt"
    echo "训练结束时间: $(date)" >> "$OUTPUT_DIR/training_log.txt"
    
    echo ""
    echo "=================================================="
    echo "训练完成或中断"
    echo "=================================================="
    echo "完整日志已保存到: $OUTPUT_DIR/training_log.txt"
else
    echo "训练已取消"
    exit 0
fi

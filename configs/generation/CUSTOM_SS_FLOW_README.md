# Custom Sparse Structure Flow Matching 训练配置

本目录包含用于训练custom_sparse_structure数据集的flow matching模型配置。这些配置完全复用原有的训练框架，只需修改数据集和模型配置即可。

## 📋 配置文件说明

### 1. **无条件生成** (Unconditional Generation)
- **配置文件**: `ss_flow_custom_uncond_dit_B_16l8_fp16.json`
- **模型**: DiT-B (Base) - 768维，12层
- **Trainer**: `SparseFlowMatchingTrainer`
- **数据集**: `CustomSparseStructureLatent`
- **用途**: 无需任何条件，直接生成稀疏结构
- **批量大小**: 16/GPU

### 2. **文本条件生成** (Text-Conditioned Generation)

#### 小模型 (Base)
- **配置文件**: `ss_flow_custom_txt_dit_B_16l8_fp16.json`
- **模型**: DiT-B - 768维，12层
- **Trainer**: `TextConditionedSparseFlowMatchingCFGTrainer`
- **数据集**: `TextConditionedCustomSparseStructureLatent`
- **条件编码器**: CLIP ViT-L/14
- **批量大小**: 16/GPU
- **CFG**: p_uncond=0.1 (10%无条件训练)

#### 大模型 (Large)
- **配置文件**: `ss_flow_custom_txt_dit_L_16l8_fp16.json`
- **模型**: DiT-L - 1024维，24层
- **Trainer**: `TextConditionedSparseFlowMatchingCFGTrainer`
- **数据集**: `TextConditionedCustomSparseStructureLatent`
- **条件编码器**: CLIP ViT-L/14
- **批量大小**: 8/GPU
- **CFG**: p_uncond=0.1

### 3. **图像条件生成** (Image-Conditioned Generation)
- **配置文件**: `ss_flow_custom_img_dit_L_16l8_fp16.json`
- **模型**: DiT-L - 1024维，24层
- **Trainer**: `ImageConditionedSparseFlowMatchingCFGTrainer`
- **数据集**: `ImageConditionedCustomSparseStructureLatent`
- **条件编码器**: DINOv2 ViT-L/14
- **批量大小**: 8/GPU
- **图像尺寸**: 518x518

## 🚀 使用方法

### 前置要求

1. **准备数据集**: 数据集需要符合以下目录结构：
```
your_dataset_root/
├── metadata.csv              # 必须包含 'sha256' 列
└── ss_latents/
    └── ss_enc_conv3d_16l8_fp16/
        ├── {sha256_1}.npz    # 包含 'mean' 键的latent编码
        ├── {sha256_2}.npz
        └── ...
```

2. **文本条件训练**需要额外在metadata.csv中添加：
   - `captions` 列：JSON格式的文本描述列表

3. **图像条件训练**需要额外：
   - `cond_rendered` 列：布尔值，标记是否已渲染条件图像
   - 渲染好的条件图像文件

### 训练命令

#### 基础训练命令模板
```bash
python train.py \
    --config configs/generation/ss_flow_custom_[TYPE]_dit_[SIZE]_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir /path/to/output \
    --num_gpus [GPU数量]
```

#### 具体示例

**1. 无条件生成训练**
```bash
python train.py \
    --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/custom_dataset \
    --output_dir outputs/custom_uncond_B \
    --num_gpus 8
```

**2. 文本条件生成训练 (Base模型)**
```bash
python train.py \
    --config configs/generation/ss_flow_custom_txt_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/custom_dataset \
    --output_dir outputs/custom_txt_B \
    --num_gpus 8
```

**3. 文本条件生成训练 (Large模型)**
```bash
python train.py \
    --config configs/generation/ss_flow_custom_txt_dit_L_16l8_fp16.json \
    --data_dir /path/to/your/custom_dataset \
    --output_dir outputs/custom_txt_L \
    --num_gpus 8
```

**4. 图像条件生成训练**
```bash
python train.py \
    --config configs/generation/ss_flow_custom_img_dit_L_16l8_fp16.json \
    --data_dir /path/to/your/custom_dataset \
    --output_dir outputs/custom_img_L \
    --num_gpus 8
```

### 从检查点恢复训练
```bash
python train.py \
    --config configs/generation/ss_flow_custom_txt_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir outputs/custom_txt_B \
    --load_dir outputs/custom_txt_B \
    --ckpt latest \
    --num_gpus 8
```

### 多数据集混合训练
```bash
python train.py \
    --config configs/generation/ss_flow_custom_txt_dit_B_16l8_fp16.json \
    --data_dir /path/to/dataset1,/path/to/dataset2,/path/to/dataset3 \
    --output_dir outputs/custom_txt_B_mixed \
    --num_gpus 8
```

## ⚙️ 配置参数说明

### 通用训练参数
- `max_steps`: 1000000 - 最大训练步数
- `batch_size_per_gpu`: 8-16 - 每GPU批量大小
- `lr`: 0.0001 - 学习率
- `ema_rate`: [0.9999] - EMA衰减率
- `i_log`: 500 - 日志记录间隔
- `i_sample`: 10000 - 采样间隔
- `i_save`: 10000 - 保存检查点间隔

### Flow Matching特定参数
- `t_schedule`: logitNormal分布，mean=1.0, std=1.0
- `sigma_min`: 1e-5 - 最小噪声标准差
- `p_uncond`: 0.1 - 无条件训练概率（仅CFG训练）

### 模型架构参数
- `resolution`: 16 - Latent空间分辨率
- `in_channels/out_channels`: 8 - 输入/输出通道数
- `model_channels`: 768(B) / 1024(L) - 模型隐藏维度
- `num_blocks`: 12(B) / 24(L) - Transformer层数
- `num_heads`: 12(B) / 16(L) - 注意力头数

## 📊 数据准备工作流

如果你还没有latent编码，需要先运行VAE编码：

```bash
# 假设你已经有原始的sparse structure数据
python dataset_toolkits/encode_ss_latent.py \
    --data_dir /path/to/your/custom_dataset \
    --encoder ss_enc_conv3d_16l8_fp16 \
    --output_dir /path/to/your/custom_dataset/ss_latents/ss_enc_conv3d_16l8_fp16
```

## 🔧 自定义配置

如需修改训练参数，直接编辑JSON配置文件：

```json
{
    "trainer": {
        "args": {
            "batch_size_per_gpu": 16,  // 修改批量大小
            "lr": 0.0001,               // 修改学习率
            "max_steps": 500000,        // 修改最大步数
            ...
        }
    }
}
```

## 📈 训练监控

训练过程中会生成：
- **日志**: `{output_dir}/logs/` - TensorBoard日志
- **检查点**: `{output_dir}/ckpts/` - 模型权重
- **采样**: `{output_dir}/samples/` - 定期生成的样本

使用TensorBoard查看训练进度：
```bash
tensorboard --logdir outputs/custom_txt_B/logs
```

## 🎯 推荐训练策略

1. **快速原型验证**: 使用无条件生成配置，快速验证数据集质量
2. **小模型起步**: 先用DiT-B验证效果，再扩展到DiT-L
3. **渐进式训练**: 可以先训练较少步数(10-20万步)观察效果，再决定是否继续
4. **数据增强**: 多个数据集混合训练可以提升模型泛化能力

## ⚠️ 注意事项

1. **内存要求**:
   - DiT-B: 至少12GB显存/GPU
   - DiT-L: 至少16GB显存/GPU
   - 如显存不足，可减小`batch_size_per_gpu`或启用梯度累积

2. **数据集要求**:
   - 无条件生成：只需latent编码
   - 文本条件：需要captions字段
   - 图像条件：需要渲染的条件图像

3. **Latent编码**:
   - 必须使用与VAE训练时相同的encoder
   - 默认使用`ss_enc_conv3d_16l8_fp16`
   - 如使用不同encoder，需修改配置中的`latent_model`参数

4. **兼容性**:
   - 所有配置完全兼容原有`train.py`入口
   - 使用原有的trainer和model代码，无需修改
   - 支持所有原有的训练功能（EMA、梯度裁剪、混合精度等）

## 📚 更多资源

- **原始Flow Matching配置**: `configs/generation/ss_flow_txt_dit_*.json`
- **数据集工具**: `dataset_toolkits/`
- **训练器代码**: `trellis/trainers/flow_matching/sparse_flow_matching.py`
- **数据集代码**: `trellis/datasets/custom_sparse_structure.py`

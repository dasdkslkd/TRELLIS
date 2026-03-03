# Custom Sparse Structure 伪颜色图像条件生成

本文档说明如何使用 DINOv2 特征编码器对 custom_sparse_structure 进行图像条件生成训练。

## 概述

每个活跃体素的伪颜色由数据的第1到8通道的 one-hot 编码决定：
- 通道 0: occupancy field (占用场)
- 通道 1-7: one-hot 标签 (共7-8个类别)

伪颜色映射表：
| 类别 | 颜色 | RGB |
|------|------|-----|
| 0 | 蓝色 | (0.12, 0.47, 0.71) |
| 1 | 橙色 | (1.00, 0.50, 0.05) |
| 2 | 绿色 | (0.17, 0.63, 0.17) |
| 3 | 红色 | (0.84, 0.15, 0.16) |
| 4 | 紫色 | (0.58, 0.40, 0.74) |
| 5 | 棕色 | (0.55, 0.34, 0.29) |
| 6 | 粉色 | (0.89, 0.47, 0.76) |
| 7 | 灰色 | (0.50, 0.50, 0.50) |

## 数据准备流程

### 1. 准备数据集结构

确保你的数据集目录结构如下：
```
dataset/
├── metadata.csv          # 必须包含 sha256 列
├── data/                 # 原始数据文件
│   ├── {sha256}.pt
│   └── ...
├── ss_latents/           # VAE编码后的潜码 (用于flow matching训练)
│   └── {latent_model}/
│       ├── {sha256}.npz
│       └── ...
├── renders_cond/         # 条件渲染图像 (步骤2生成)
│   └── {sha256}/
│       ├── 0000.png
│       ├── 0001.png
│       ├── ...
│       └── transforms.json
└── features/             # DINOv2特征 (可选，步骤3生成)
    └── {model_name}/
        ├── {sha256}.npz
        └── ...
```

### 2. 渲染条件图像

使用伪颜色渲染稀疏结构：

```bash
cd dataset_toolkits

# 基本用法
python render_cond_ss.py --output_dir /path/to/dataset

# 完整参数
python render_cond_ss.py \
    --output_dir /path/to/dataset \
    --data_dir_name data \
    --file_ext .pt \
    --resolution 64 \
    --num_views 24 \
    --max_workers 1

# 分布式处理
python render_cond_ss.py \
    --output_dir /path/to/dataset \
    --rank 0 \
    --world_size 4
```

这将在 `renders_cond/{sha256}/` 目录下生成：
- 多视角渲染图像 (PNG格式，RGBA)
- `transforms.json` 文件包含相机参数

### 3. (可选) 预提取 DINOv2 特征

如果想加速训练，可以预先提取 DINOv2 特征：

```bash
python extract_feature_ss.py \
    --output_dir /path/to/dataset \
    --model dinov2_vitl14_reg \
    --num_views 24
```

## 训练配置

### 配置文件

使用 `configs/generation/ss_flow_custom_pseudocolor_img_dit_B_16l8_fp16.json`：

```json
{
    "models": {
        "denoiser": {
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": 16,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 768,
                "cond_channels": 1024,    // DINOv2 vitl14 特征维度
                "num_blocks": 12,
                "num_heads": 12,
                ...
            }
        }
    },
    "dataset": {
        "name": "PseudoColorImageConditionedCustomSparseStructureLatent",
        "args": {
            "latent_model": "your_latent_model_name",
            "image_size": 518
        }
    },
    "trainer": {
        "name": "ImageConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            ...
            "image_cond_model": "dinov2_vitl14_reg"
        }
    }
}
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `cond_channels` | 条件特征维度，DINOv2 vitl14 使用 1024 |
| `image_size` | 图像大小，DINOv2 推荐 518 |
| `image_cond_model` | DINOv2 模型名称 |
| `p_uncond` | Classifier-free guidance 的无条件概率 |

### 启动训练

```bash
python train.py \
    --config configs/generation/ss_flow_custom_pseudocolor_img_dit_B_16l8_fp16.json \
    --data /path/to/train_dataset \
    --val_data /path/to/val_dataset \
    --output_dir results/my_experiment
```

## 数据集类说明

### PseudoColorImageConditionedMixin

这个 Mixin 类提供了：
- 从 `renders_cond/` 目录加载渲染图像
- 自动裁剪和缩放图像以适应 DINOv2 输入
- 随机选择视角进行训练

### PseudoColorImageConditionedCustomSparseStructureLatent

完整的数据集类，组合了：
- `PseudoColorImageConditionedMixin`: 图像条件处理
- `CustomSparseStructureLatent`: 潜码加载

## API 参考

### 伪颜色映射函数

```python
from trellis.datasets.custom_sparse_structure import PseudoColorImageConditionedMixin

# 获取伪颜色映射表
colors = PseudoColorImageConditionedMixin.PSEUDO_COLORS  # [8, 3] tensor
```

### 渲染工具

```python
from dataset_toolkits.render_cond_ss import (
    get_pseudo_colors_from_onehot,
    render_sparse_structure_with_pseudo_colors,
)

# 将 one-hot 标签转换为伪颜色
colors = get_pseudo_colors_from_onehot(onehot_labels)  # [N, 3]

# 渲染带伪颜色的稀疏结构
renders = render_sparse_structure_with_pseudo_colors(
    ss,                     # [C, H, W, D] tensor
    resolution=64,
    image_size=512,
    num_views=24,
    device='cuda'
)
```

## 常见问题

### Q: metadata.csv 需要哪些列？

最少需要：
- `sha256`: 实例唯一标识符
- `cond_rendered`: 布尔值，指示是否已渲染条件图像
- `ss_latent_{model}`: 布尔值，指示是否已编码潜码

### Q: 如何添加新的伪颜色类别？

修改 `PSEUDO_COLORS` 张量，添加新的 RGB 值。确保索引与 one-hot 编码对应。

### Q: DINOv2 模型选择？

| 模型 | 特征维度 | 参数量 | 推荐场景 |
|------|----------|--------|----------|
| dinov2_vits14_reg | 384 | 22M | 轻量级 |
| dinov2_vitb14_reg | 768 | 86M | 平衡 |
| dinov2_vitl14_reg | 1024 | 300M | 高质量 |
| dinov2_vitg14_reg | 1536 | 1.1B | 最高质量 |

相应地调整配置中的 `cond_channels` 参数。

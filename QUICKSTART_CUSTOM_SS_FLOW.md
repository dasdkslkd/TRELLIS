# Custom Sparse Structure Flow Matching - 快速开始

## 🎯 最简单的训练示例

### 1. 准备数据集

确保你的数据集结构如下：
```
/path/to/your/dataset/
├── metadata.csv                          # 必须有 sha256 列
└── ss_latents/
    └── ss_enc_conv3d_16l8_fp16/
        ├── abc123...xyz.npz              # latent文件
        ├── def456...uvw.npz
        └── ...
```

### 2. 开始训练

#### 🔥 最推荐：无条件生成训练（最简单）

```bash
# 单GPU训练
python train.py \
    --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir outputs/my_first_training \
    --num_gpus 1

# 多GPU训练（推荐）
python train.py \
    --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir outputs/my_first_training \
    --num_gpus 8
```

#### 📝 文本条件生成（需要captions）

```bash
python train.py \
    --config configs/generation/ss_flow_custom_txt_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir outputs/my_text_training \
    --num_gpus 8
```

## 📋 数据集要求检查清单

### 无条件生成
- [x] `metadata.csv`文件存在，包含`sha256`列
- [x] `ss_latents/ss_enc_conv3d_16l8_fp16/`目录存在
- [x] 每个sha256对应的`.npz`文件存在

### 文本条件生成（额外要求）
- [x] `metadata.csv`中有`captions`列
- [x] `captions`列是JSON格式的文本列表，如`["a red car", "car"]`

### 图像条件生成（额外要求）
- [x] `metadata.csv`中有`cond_rendered`列（True/False）
- [x] 条件图像已经渲染并保存

## 🔍 训练监控

```bash
# 在另一个终端运行
tensorboard --logdir outputs/my_first_training/logs
```

然后打开浏览器访问 http://localhost:6006

## ⏸️ 暂停和恢复训练

训练会自动保存检查点，恢复训练：

```bash
python train.py \
    --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
    --data_dir /path/to/your/dataset \
    --output_dir outputs/my_first_training \
    --load_dir outputs/my_first_training \
    --ckpt latest \
    --num_gpus 8
```

## 💾 检查点位置

训练过程中的文件：
```
outputs/my_first_training/
├── ckpts/
│   ├── denoiser_ema0.9999_step010000.pt    # EMA模型
│   ├── denoiser_step010000.pt              # 原始模型
│   └── misc_step010000.pt                  # 优化器状态
├── logs/                                    # TensorBoard日志
├── samples/                                 # 训练期间的采样
└── config.json                             # 训练配置备份
```

## 🚨 常见问题

**Q: 显存不足怎么办？**
```bash
# 编辑配置文件，减小batch_size_per_gpu
# 从16改为8或4
```

**Q: 我没有latent编码文件？**
```bash
# 需要先使用VAE编码原始数据
# 参考CUSTOM_SS_FLOW_README.md中的"数据准备工作流"部分
```

**Q: 训练速度太慢？**
```bash
# 1. 增加GPU数量
# 2. 检查数据加载是否是瓶颈（增加num_workers）
# 3. 确保使用了--num_gpus参数进行多GPU训练
```

## 📖 详细文档

更多高级用法请参考：
- [configs/generation/CUSTOM_SS_FLOW_README.md](CUSTOM_SS_FLOW_README.md) - 完整文档

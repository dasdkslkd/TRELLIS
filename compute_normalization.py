#!/usr/bin/env python
"""
计算数据集的归一化参数（mean 和 std）

用法:
    python compute_normalization.py --dataset_root ../TRELLIS/dataset/sparse_vae_dataset/train/ --latent_model ss_enc_conv3d_16l8_fp16

输出将显示可以直接复制到配置文件中的 normalization 配置
"""
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm


def compute_normalization_stats(dataset_root: str, latent_model: str, num_samples: int = None):
    """
    计算数据集的归一化统计量
    
    Args:
        dataset_root: 数据集根目录
        latent_model: latent 模型名称
        num_samples: 用于计算的样本数量，None 表示使用全部
        
    Returns:
        dict: 包含 mean 和 std 的字典
    """
    latent_dir = os.path.join(dataset_root, 'ss_latents', latent_model)
    
    if not os.path.exists(latent_dir):
        raise FileNotFoundError(f"Latent 目录不存在: {latent_dir}")
    
    # 获取所有 npz 文件
    npz_files = [f for f in os.listdir(latent_dir) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        raise ValueError(f"在 {latent_dir} 中没有找到 npz 文件")
    
    print(f"找到 {len(npz_files)} 个 latent 文件")
    
    if num_samples is not None:
        npz_files = npz_files[:num_samples]
        print(f"使用前 {num_samples} 个样本进行计算")
    
    # 收集统计信息
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    channel_means = []
    channel_stds = []
    
    first_shape = None
    
    for npz_file in tqdm(npz_files, desc="计算统计量"):
        latent_path = os.path.join(latent_dir, npz_file)
        try:
            data = np.load(latent_path)
            z = data['mean']  # [C, H, W, D]
            
            if first_shape is None:
                first_shape = z.shape
                print(f"Latent shape: {first_shape}")
                # 初始化 per-channel 统计
                channel_means = [[] for _ in range(z.shape[0])]
                channel_stds = [[] for _ in range(z.shape[0])]
            
            all_means.append(z.mean())
            all_stds.append(z.std())
            all_mins.append(z.min())
            all_maxs.append(z.max())
            
            # Per-channel statistics
            for c in range(z.shape[0]):
                channel_means[c].append(z[c].mean())
                channel_stds[c].append(z[c].std())
                
        except Exception as e:
            print(f"跳过文件 {npz_file}: {e}")
            continue
    
    # 计算全局统计量
    global_mean = np.mean(all_means)
    global_std = np.mean(all_stds)
    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    
    # 计算 per-channel 统计量
    per_channel_mean = [np.mean(cm) for cm in channel_means]
    per_channel_std = [np.mean(cs) for cs in channel_stds]
    
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    print(f"样本数量: {len(all_means)}")
    print(f"Latent shape: {first_shape}")
    print(f"\n全局统计:")
    print(f"  Mean: {global_mean:.6f}")
    print(f"  Std:  {global_std:.6f}")
    print(f"  Min:  {global_min:.6f}")
    print(f"  Max:  {global_max:.6f}")
    
    print(f"\nPer-channel Mean: {[f'{m:.4f}' for m in per_channel_mean]}")
    print(f"Per-channel Std:  {[f'{s:.4f}' for s in per_channel_std]}")
    
    # 推荐的归一化配置
    print("\n" + "="*60)
    print("推荐的配置文件设置 (使用全局统计)")
    print("="*60)
    print(f'''
"dataset": {{
    "name": "CustomSparseStructureLatent",
    "args": {{
        "latent_model": "{latent_model}",
        "normalization": {{
            "mean": [{global_mean:.6f}],
            "std": [{global_std:.6f}]
        }}
    }}
}}
''')
    
    print("="*60)
    print("推荐的配置文件设置 (使用 per-channel 统计)")
    print("="*60)
    mean_str = ", ".join([f"{m:.6f}" for m in per_channel_mean])
    std_str = ", ".join([f"{s:.6f}" for s in per_channel_std])
    print(f'''
"dataset": {{
    "name": "CustomSparseStructureLatent",
    "args": {{
        "latent_model": "{latent_model}",
        "normalization": {{
            "mean": [{mean_str}],
            "std": [{std_str}]
        }}
    }}
}}
''')
    
    return {
        'global_mean': global_mean,
        'global_std': global_std,
        'per_channel_mean': per_channel_mean,
        'per_channel_std': per_channel_std,
        'min': global_min,
        'max': global_max,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算数据集归一化参数')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--latent_model', type=str, default='ss_enc_conv3d_16l8_fp16',
                        help='Latent 模型名称')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='用于计算的样本数量（默认使用全部）')
    
    args = parser.parse_args()
    
    stats = compute_normalization_stats(
        dataset_root=args.dataset_root,
        latent_model=args.latent_model,
        num_samples=args.num_samples,
    )

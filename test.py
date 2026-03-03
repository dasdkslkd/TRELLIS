"""诊断脚本：检查数据分布"""
import torch
from trellis.datasets import CustomSparseStructureLatent

dataset = CustomSparseStructureLatent(roots='../TRELLIS/dataset/sparse_vae_dataset/train/',latent_model="")
sample = dataset[0]

print("=== 数据诊断 ===")
print(f"Features shape: {sample['x_0'].shape}")
print(f"Features range: [{sample['x_0'].min():.4f}, {sample['x_0'].max():.4f}]")
print(f"Features mean: {sample['x_0'].mean():.4f}")
print(f"Features std: {sample['x_0'].std():.4f}")

# 检查是否有异常值
outliers = (sample['x_0'].abs() > 10).sum()
print(f"Outliers (|x| > 10): {outliers}")
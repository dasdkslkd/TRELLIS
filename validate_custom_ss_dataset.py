#!/usr/bin/env python3
"""
验证custom sparse structure数据集是否符合flow matching训练要求

使用方法:
    python validate_custom_ss_dataset.py /path/to/your/dataset [--mode uncond|text|image]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path


def check_metadata(dataset_root):
    """检查metadata.csv文件"""
    print("=" * 80)
    print("检查 metadata.csv...")
    print("=" * 80)
    
    metadata_path = os.path.join(dataset_root, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"❌ 错误: metadata.csv 不存在于 {dataset_root}")
        return None
    
    print(f"✓ metadata.csv 存在")
    
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"✓ CSV文件可以正常读取")
        print(f"  - 总行数: {len(metadata)}")
    except Exception as e:
        print(f"❌ 错误: 无法读取metadata.csv: {e}")
        return None
    
    # 检查sha256列
    if 'sha256' not in metadata.columns:
        print(f"❌ 错误: metadata.csv 缺少 'sha256' 列")
        print(f"   当前列: {list(metadata.columns)}")
        return None
    
    print(f"✓ 包含 'sha256' 列")
    print(f"  - 唯一sha256数量: {metadata['sha256'].nunique()}")
    
    return metadata


def check_latents(dataset_root, metadata, latent_model='ss_enc_conv3d_16l8_fp16'):
    """检查latent编码文件"""
    print("\n" + "=" * 80)
    print(f"检查 latent 编码文件 ({latent_model})...")
    print("=" * 80)
    
    latent_dir = os.path.join(dataset_root, 'ss_latents', latent_model)
    
    if not os.path.exists(latent_dir):
        print(f"❌ 错误: latent目录不存在: {latent_dir}")
        return False
    
    print(f"✓ latent目录存在: {latent_dir}")
    
    # 检查metadata中是否有latent标记列
    latent_col = f'ss_latent_{latent_model}'
    if latent_col in metadata.columns:
        print(f"✓ metadata中有 '{latent_col}' 标记列")
        has_latent = metadata[latent_col].sum()
        print(f"  - 标记为有latent的样本数: {has_latent}/{len(metadata)}")
    else:
        print(f"⚠ 警告: metadata中没有 '{latent_col}' 列，将检查所有样本")
    
    # 随机检查几个latent文件
    sample_sha256s = metadata['sha256'].sample(min(5, len(metadata))).tolist()
    
    all_ok = True
    for sha256 in sample_sha256s:
        latent_path = os.path.join(latent_dir, f'{sha256}.npz')
        if not os.path.exists(latent_path):
            print(f"❌ 样本 {sha256}: latent文件不存在")
            all_ok = False
            continue
        
        try:
            latent = np.load(latent_path)
            if 'mean' not in latent:
                print(f"❌ 样本 {sha256}: latent文件缺少 'mean' 键")
                all_ok = False
                continue
            
            mean = latent['mean']
            print(f"✓ 样本 {sha256[:8]}...: 形状 {mean.shape}")
            
        except Exception as e:
            print(f"❌ 样本 {sha256}: 无法读取latent文件: {e}")
            all_ok = False
    
    return all_ok


def check_text_condition(metadata):
    """检查文本条件数据"""
    print("\n" + "=" * 80)
    print("检查文本条件数据...")
    print("=" * 80)
    
    if 'captions' not in metadata.columns:
        print(f"❌ 错误: metadata.csv 缺少 'captions' 列")
        return False
    
    print(f"✓ 包含 'captions' 列")
    
    # 检查有caption的样本数
    has_captions = metadata['captions'].notna().sum()
    print(f"  - 有caption的样本数: {has_captions}/{len(metadata)}")
    
    # 随机检查几个caption
    sample_captions = metadata[metadata['captions'].notna()]['captions'].sample(min(3, has_captions))
    
    all_ok = True
    for idx, caption in sample_captions.items():
        try:
            caption_list = json.loads(caption)
            if not isinstance(caption_list, list):
                print(f"❌ 行 {idx}: caption不是列表格式")
                all_ok = False
                continue
            print(f"✓ 行 {idx}: {len(caption_list)} 条caption")
            print(f"  示例: {caption_list[0][:50]}...")
        except json.JSONDecodeError as e:
            print(f"❌ 行 {idx}: caption不是有效的JSON: {e}")
            all_ok = False
    
    return all_ok


def check_image_condition(metadata):
    """检查图像条件数据"""
    print("\n" + "=" * 80)
    print("检查图像条件数据...")
    print("=" * 80)
    
    if 'cond_rendered' not in metadata.columns:
        print(f"❌ 错误: metadata.csv 缺少 'cond_rendered' 列")
        return False
    
    print(f"✓ 包含 'cond_rendered' 列")
    
    # 检查有rendered条件图像的样本数
    has_rendered = metadata['cond_rendered'].sum()
    print(f"  - 已渲染条件图像的样本数: {has_rendered}/{len(metadata)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='验证custom sparse structure数据集')
    parser.add_argument('dataset_root', type=str, help='数据集根目录路径')
    parser.add_argument('--mode', type=str, choices=['uncond', 'text', 'image'], 
                       default='uncond', help='训练模式 (默认: uncond)')
    parser.add_argument('--latent-model', type=str, default='ss_enc_conv3d_16l8_fp16',
                       help='Latent模型名称 (默认: ss_enc_conv3d_16l8_fp16)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Custom Sparse Structure 数据集验证工具")
    print("=" * 80)
    print(f"数据集路径: {args.dataset_root}")
    print(f"训练模式: {args.mode}")
    print(f"Latent模型: {args.latent_model}")
    
    # 检查数据集目录存在
    if not os.path.exists(args.dataset_root):
        print(f"\n❌ 错误: 数据集目录不存在: {args.dataset_root}")
        sys.exit(1)
    
    # 1. 检查metadata.csv
    metadata = check_metadata(args.dataset_root)
    if metadata is None:
        print("\n❌ metadata.csv检查失败")
        sys.exit(1)
    
    # 2. 检查latent文件
    latent_ok = check_latents(args.dataset_root, metadata, args.latent_model)
    if not latent_ok:
        print("\n❌ latent文件检查失败")
        sys.exit(1)
    
    # 3. 根据模式检查额外要求
    if args.mode == 'text':
        text_ok = check_text_condition(metadata)
        if not text_ok:
            print("\n❌ 文本条件数据检查失败")
            sys.exit(1)
    elif args.mode == 'image':
        image_ok = check_image_condition(metadata)
        if not image_ok:
            print("\n❌ 图像条件数据检查失败")
            sys.exit(1)
    
    # 总结
    print("\n" + "=" * 80)
    print("验证结果")
    print("=" * 80)
    print("✓ 所有检查通过！")
    print(f"\n数据集已准备好用于 {args.mode} 模式训练")
    
    # 推荐的训练命令
    print("\n" + "=" * 80)
    print("推荐的训练命令:")
    print("=" * 80)
    
    if args.mode == 'uncond':
        config = 'ss_flow_custom_uncond_dit_B_16l8_fp16.json'
    elif args.mode == 'text':
        config = 'ss_flow_custom_txt_dit_B_16l8_fp16.json'
    else:  # image
        config = 'ss_flow_custom_img_dit_L_16l8_fp16.json'
    
    print(f"""
python train.py \\
    --config configs/generation/{config} \\
    --data_dir {args.dataset_root} \\
    --output_dir outputs/my_training \\
    --num_gpus 8
""")
    
    print("=" * 80)
    sys.exit(0)


if __name__ == '__main__':
    main()

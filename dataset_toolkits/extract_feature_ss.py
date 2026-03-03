"""
为custom_sparse_structure预提取DINOv2特征。

此脚本对使用伪颜色渲染的条件图像预提取DINOv2 patch token特征，
保存到 features/{model_name}/{sha256}.npz。

注意:
  - 当前训练流水线 (ImageConditionedFlowMatchingCFGTrainer) 在训练时
    会自动使用 DINOv2 编码输入图像，因此本脚本为**可选**的预计算工具。
  - 预提取的特征可用于:
    1. 离线分析和可视化
    2. 自定义推理脚本直接加载特征（跳过 DINOv2 前向）
    3. 未来实现特征缓存加速训练

用法:
    python extract_feature_ss.py --output_dir /path/to/dataset --model dinov2_vitl14_reg
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor


torch.set_grad_enabled(False)


def load_and_preprocess_image(image_path: str, image_size: int = 518):
    """
    加载并预处理图像用于DINOv2特征提取
    
    Args:
        image_path: 图像路径
        image_size: 目标图像大小
        
    Returns:
        预处理后的图像张量 [C, H, W]
    """
    image = Image.open(image_path)
    
    # 处理alpha通道进行裁剪
    if image.mode == 'RGBA':
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        if len(bbox[0]) > 0:
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_size_ratio = 1.2
            aug_hsize = hsize * aug_size_ratio
            aug_bbox = (
                int(center[0] - aug_hsize), 
                int(center[1] - aug_hsize), 
                int(center[0] + aug_hsize), 
                int(center[1] + aug_hsize)
            )
            image = image.crop(aug_bbox)
    
    # 调整大小
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # 提取alpha和RGB
    if image.mode == 'RGBA':
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
    else:
        image = image.convert('RGB')
        alpha = torch.ones(image_size, image_size)
    
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    # 应用alpha遮罩 (白色背景)
    image = image * alpha.unsqueeze(0) + (1 - alpha.unsqueeze(0))
    
    return image


@torch.no_grad()
def extract_features(
    model, 
    transform, 
    image_root: str, 
    num_views_to_use: Optional[int] = None
) -> torch.Tensor:
    """
    提取所有视角的DINOv2特征
    
    Args:
        model: DINOv2模型
        transform: 图像预处理变换
        image_root: 包含渲染图像的目录
        num_views_to_use: 使用的视角数量 (None表示全部)
        
    Returns:
        特征张量 [num_views, num_patches, feature_dim]
    """
    with open(os.path.join(image_root, 'transforms.json')) as f:
        metadata = json.load(f)
    
    frames = metadata['frames']
    if num_views_to_use is not None:
        frames = frames[:num_views_to_use]
    
    features_list = []
    
    for frame in frames:
        image_path = os.path.join(image_root, frame['file_path'])
        image = load_and_preprocess_image(image_path)
        image = image.unsqueeze(0).cuda()
        
        # 应用DINOv2变换
        image = transform(image)
        
        # 提取特征
        feat = model(image, is_training=True)['x_prenorm']
        feat = F.layer_norm(feat, feat.shape[-1:])
        
        features_list.append(feat.cpu())
    
    return torch.cat(features_list, dim=0)  # [num_views, num_patches, feature_dim]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='数据集目录 (包含metadata.csv和renders_cond/)')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='DINOv2模型名称')
    parser.add_argument('--num_views', type=int, default=None,
                        help='每个实例使用的视角数量 (默认全部)')
    parser.add_argument('--instances', type=str, default=None,
                        help='指定要处理的实例列表文件')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)
    
    # 加载DINOv2模型
    print(f'Loading DINOv2 model: {opt.model}')
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model, pretrained=True)
    dinov2_model.eval().cuda()
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载元数据
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # 过滤实例
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        # 只处理已渲染条件图像的实例
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[metadata['cond_rendered'] == True]
        # 过滤已提取特征的
        feature_col = f'feature_{feature_name}'
        if feature_col in metadata.columns:
            metadata = metadata[metadata[feature_col] != True]
    
    # 分片处理
    metadata = metadata.iloc[opt.rank::opt.world_size]
    
    print(f'Processing {len(metadata)} instances...')
    
    results = []
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        sha256 = row['sha256']
        image_root = os.path.join(opt.output_dir, 'renders_cond', sha256)
        
        if not os.path.exists(os.path.join(image_root, 'transforms.json')):
            results.append({'sha256': sha256, 'success': False, 'error': 'No transforms.json'})
            continue
        
        try:
            # 提取特征
            features = extract_features(
                dinov2_model, 
                transform, 
                image_root, 
                opt.num_views
            )
            
            # 保存特征
            output_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
            np.savez_compressed(
                output_path,
                features=features.numpy(),  # [num_views, num_patches, feature_dim]
            )
            
            results.append({'sha256': sha256, 'success': True})
            
        except Exception as e:
            results.append({'sha256': sha256, 'success': False, 'error': str(e)})
            print(f"Error processing {sha256}: {e}")
    
    # 更新metadata
    print('Updating metadata...')
    metadata_full = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    feature_col = f'feature_{feature_name}'
    if feature_col not in metadata_full.columns:
        metadata_full[feature_col] = False
    
    for result in results:
        sha256 = result['sha256']
        idx = metadata_full[metadata_full['sha256'] == sha256].index
        if len(idx) > 0:
            metadata_full.loc[idx, feature_col] = result['success']
    
    metadata_full.to_csv(os.path.join(opt.output_dir, 'metadata.csv'), index=False)
    
    print(f'Done! Processed {len(results)} instances.')
    print(f'Successful: {sum(1 for r in results if r["success"])}')
    print(f'Failed: {sum(1 for r in results if not r["success"])}')

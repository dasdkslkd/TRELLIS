"""
为custom_sparse_structure生成条件渲染图像。

每个活跃体素的伪颜色由数据的第1到8通道的onehot编码决定。
使用DINOv2提取特征用于图像条件生成。

用法:
    python render_cond_ss.py --output_dir /path/to/dataset --num_views 24
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor

import utils3d
from trellis.representations.octree import DfsOctree as Octree
from trellis.renderers import OctreeRenderer

# 8类的伪颜色映射 (对应channel 1-8的one-hot编码)
# 使用matplotlib tab10颜色方案
PSEUDO_COLORS = torch.tensor([
    [0.12, 0.47, 0.71],  # 类别0: 蓝色
    [1.00, 0.50, 0.05],  # 类别1: 橙色
    [0.17, 0.63, 0.17],  # 类别2: 绿色
    [0.84, 0.15, 0.16],  # 类别3: 红色
    [0.58, 0.40, 0.74],  # 类别4: 紫色
    [0.55, 0.34, 0.29],  # 类别5: 棕色
    [0.89, 0.47, 0.76],  # 类别6: 粉色
    [0.50, 0.50, 0.50],  # 类别7: 灰色
], dtype=torch.float32)


def get_pseudo_colors_from_onehot(onehot_labels: torch.Tensor) -> torch.Tensor:
    """
    根据one-hot标签获取伪颜色
    
    Args:
        onehot_labels: 形状为 [N, 8] 或 [N, 7] 的one-hot编码张量
                      如果是7维，假设第一个类别缺失
    
    Returns:
        形状为 [N, 3] 的RGB颜色张量
    """
    if onehot_labels.shape[-1] == 7:
        # 补齐第一个通道
        onehot_labels = F.pad(onehot_labels, (1, 0), value=0)
    
    # 获取类别索引
    class_indices = torch.argmax(onehot_labels, dim=-1)  # [N]
    
    # 映射到颜色
    colors = PSEUDO_COLORS.to(onehot_labels.device)[class_indices]  # [N, 3]
    
    return colors


def load_sparse_structure(file_path: str, resolution: int = 64) -> torch.Tensor:
    """
    从pt文件加载sparse structure并转换为dense tensor
    
    Args:
        file_path: pt文件路径
        resolution: 期望的分辨率
        
    Returns:
        形状为 [C, H, W, D] 的dense tensor
    """
    data = torch.load(file_path, map_location='cpu', weights_only=True)
    
    if isinstance(data, dict):
        if 'coords' in data and 'feats' in data:
            coords = data['coords']
            feats = data['feats']
            if coords.dim() == 2 and feats.dim() == 2:
                dense = torch.zeros(
                    feats.shape[1], 
                    resolution, resolution, resolution,
                    dtype=feats.dtype
                )
                coords_int = coords.long()
                valid_mask = (
                    (coords_int >= 0).all(dim=1) & 
                    (coords_int < resolution).all(dim=1)
                )
                coords_int = coords_int[valid_mask]
                feats = feats[valid_mask]
                if coords_int.shape[0] > 0:
                    dense[:, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = feats.t()
                return dense
        if 'dense' in data or 'tensor' in data:
            tensor = data.get('dense', data.get('tensor'))
            if isinstance(tensor, torch.Tensor):
                return tensor
    elif hasattr(data, 'layout') and data.layout == torch.sparse_coo:
        return data.to_dense().permute(3, 0, 1, 2)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 4:
            return data
        elif data.dim() == 3:
            return data.unsqueeze(0)
    
    raise ValueError(f"无法识别的数据格式: {type(data)}")


def render_sparse_structure_with_pseudo_colors(
    ss: torch.Tensor,
    resolution: int,
    image_size: int = 512,
    num_views: int = 4,
    device: str = 'cuda'
) -> list:
    """
    使用伪颜色渲染sparse structure
    
    Args:
        ss: 形状为 [C, H, W, D] 的dense tensor
            第0通道: occupancy field
            第1-7通道 (或1-8): one-hot标签
        resolution: 体素分辨率
        image_size: 输出图像大小
        num_views: 渲染视角数量
        device: 计算设备
        
    Returns:
        渲染图像列表，每个元素为 dict(image, alpha, extrinsics, intrinsics, ...)
    """
    from utils import sphere_hammersley_sequence
    
    ss = ss.to(device)
    
    # 获取occupancy和one-hot标签
    occupancy = ss[0]  # [H, W, D]
    if ss.shape[0] >= 8:
        # 通道1-7是one-hot标签
        onehot_labels = ss[1:8]  # [7, H, W, D]
    else:
        # 假设所有通道都是one-hot
        onehot_labels = ss[1:]
    
    # 获取活跃体素坐标
    coords = torch.nonzero(occupancy > 0.5, as_tuple=False)  # [N, 3]
    
    if coords.shape[0] == 0:
        return []
    
    # 获取每个体素的one-hot标签
    voxel_labels = onehot_labels[:, coords[:, 0], coords[:, 1], coords[:, 2]]  # [7, N]
    voxel_labels = voxel_labels.permute(1, 0)  # [N, 7]
    
    # 转换为伪颜色
    pseudo_colors = get_pseudo_colors_from_onehot(voxel_labels)  # [N, 3]
    
    # ---------- 计算包围球 ----------
    # 位置在 AABB 归一化空间 [0, 1]
    pos_normalized = (coords.float() + 0.5) / resolution
    # 世界坐标: AABB origin(-0.5) + pos * size(1.0) = pos - 0.5
    pos_world = pos_normalized - 0.5
    
    bbox_min = pos_world.min(dim=0).values
    bbox_max = pos_world.max(dim=0).values
    center = (bbox_min + bbox_max) / 2  # 包围盒中心
    r_sphere = ((bbox_max - bbox_min) / 2).norm().item()  # 包围球半径（半对角线）
    r_sphere = max(r_sphere, 0.05)  # 防止退化
    
    # ---------- 自适应相机参数 ----------
    fov_deg = 40.0
    fov_rad = np.radians(fov_deg)
    margin = 1.3  # 30% 边距确保物体完整可见
    cam_dist = r_sphere * margin / np.tan(fov_rad / 2)
    cam_dist = max(cam_dist, 1.0)  # 最小相机距离
    
    near = max(0.1, cam_dist - r_sphere * 2.5)
    far = cam_dist + r_sphere * 2.5
    
    # 创建渲染器
    renderer = OctreeRenderer()
    renderer.rendering_options.resolution = image_size
    renderer.rendering_options.near = near
    renderer.rendering_options.far = far
    renderer.rendering_options.bg_color = (1, 1, 1)  # 白色背景
    renderer.rendering_options.ssaa = 4
    
    # 创建octree表示
    representation = Octree(
        depth=10,
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
        device=device,
        primitive='voxel',
        sh_degree=0,
        primitive_config={'solid': True},
    )
    representation.position = pos_normalized
    representation.depth = torch.full(
        (coords.shape[0], 1), 
        int(np.log2(resolution)), 
        dtype=torch.uint8, 
        device=device
    )
    
    # ---------- 生成相机位姿 (Hammersley 序列) ----------
    results = []
    offset = (np.random.rand(), np.random.rand())
    
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        # 相机位于包围盒中心的球壳上
        cam_pos = center.to(device) + torch.tensor([
            np.cos(pitch) * np.sin(yaw),
            np.cos(pitch) * np.cos(yaw),
            np.sin(pitch),
        ], dtype=torch.float32, device=device) * cam_dist
        
        fov_t = torch.deg2rad(torch.tensor(fov_deg, dtype=torch.float32)).to(device)
        extrinsics = utils3d.torch.extrinsics_look_at(
            cam_pos, 
            center.to(device), 
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        )
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)
        
        # 渲染
        res = renderer.render(
            representation, 
            extrinsics, 
            intrinsics, 
            colors_overwrite=pseudo_colors
        )
        
        image = res['color'].cpu()  # [3, H, W]
        alpha = res['alpha'].cpu()  # [H, W] or [1, H, W]
        
        results.append({
            'image': image,
            'alpha': alpha.squeeze(0) if alpha.dim() == 3 else alpha,
            'extrinsics': extrinsics.cpu(),
            'intrinsics': intrinsics.cpu(),
            'yaw': float(yaw),
            'pitch': float(pitch),
            'radius': float(cam_dist),
            'fov': float(torch.rad2deg(fov_t).cpu()),
        })
    
    return results


def save_renders(renders: list, output_folder: str, sha256: str):
    """
    保存渲染结果
    
    Args:
        renders: 渲染结果列表
        output_folder: 输出目录
        sha256: 实例ID
    """
    instance_folder = os.path.join(output_folder, sha256)
    os.makedirs(instance_folder, exist_ok=True)
    
    transforms = {
        'frames': []
    }
    
    for i, render in enumerate(renders):
        image = render['image']
        alpha = render['alpha']
        
        # 合成RGBA图像
        rgba = torch.cat([image, alpha.unsqueeze(0)], dim=0)  # [4, H, W]
        rgba = (rgba.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # 保存图像
        file_name = f'{i:04d}.png'
        Image.fromarray(rgba).save(os.path.join(instance_folder, file_name))
        
        # 记录元数据
        transforms['frames'].append({
            'file_path': file_name,
            'yaw': render['yaw'],
            'pitch': render['pitch'],
            'radius': render['radius'],
            'fov': render['fov'],
            'extrinsics': render['extrinsics'].tolist(),
            'intrinsics': render['intrinsics'].tolist(),
        })
    
    # 保存transforms.json
    with open(os.path.join(instance_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)


def process_instance(args):
    """处理单个实例"""
    sha256, data_path, output_dir, num_views, resolution = args
    
    try:
        # 加载sparse structure
        ss = load_sparse_structure(data_path, resolution)
        
        # 渲染
        renders = render_sparse_structure_with_pseudo_colors(
            ss, 
            resolution=resolution,
            image_size=512,
            num_views=num_views,
            device='cuda'
        )
        
        if len(renders) == 0:
            return {'sha256': sha256, 'cond_rendered': False, 'error': 'No active voxels'}
        
        # 保存
        save_renders(renders, os.path.join(output_dir, 'renders_cond'), sha256)
        
        return {'sha256': sha256, 'cond_rendered': True}
    
    except Exception as e:
        return {'sha256': sha256, 'cond_rendered': False, 'error': str(e)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='数据集目录 (包含metadata.csv和data/)')
    parser.add_argument('--data_dir_name', type=str, default='data',
                        help='数据子目录名称')
    parser.add_argument('--file_ext', type=str, default='.pt',
                        help='数据文件扩展名')
    parser.add_argument('--resolution', type=int, default=64,
                        help='体素分辨率')
    parser.add_argument('--num_views', type=int, default=24,
                        help='每个实例渲染的视角数量')
    parser.add_argument('--instances', type=str, default=None,
                        help='指定要处理的实例列表文件')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1,
                        help='并行处理数量 (GPU渲染建议设为1)')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    # 创建输出目录
    os.makedirs(os.path.join(opt.output_dir, 'renders_cond'), exist_ok=True)
    
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
        # 过滤已渲染的
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[metadata['cond_rendered'] != True]
    
    # 分片处理
    metadata = metadata.iloc[opt.rank::opt.world_size]
    
    print(f'Processing {len(metadata)} instances...')
    
    # 准备任务列表
    tasks = []
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        data_path = os.path.join(opt.output_dir, opt.data_dir_name, f'{sha256}{opt.file_ext}')
        if os.path.exists(data_path):
            tasks.append((sha256, data_path, opt.output_dir, opt.num_views, opt.resolution))
    
    # 处理
    results = []
    for task in tqdm(tasks):
        result = process_instance(task)
        results.append(result)
        if not result['cond_rendered']:
            print(f"Warning: {result['sha256']} - {result.get('error', 'unknown error')}")
    
    # 更新metadata
    print('Updating metadata...')
    metadata_full = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    for result in results:
        sha256 = result['sha256']
        idx = metadata_full[metadata_full['sha256'] == sha256].index
        if len(idx) > 0:
            metadata_full.loc[idx, 'cond_rendered'] = result['cond_rendered']
    
    metadata_full.to_csv(os.path.join(opt.output_dir, 'metadata.csv'), index=False)
    
    print(f'Done! Processed {len(results)} instances.')
    print(f'Successful: {sum(1 for r in results if r["cond_rendered"])}')
    print(f'Failed: {sum(1 for r in results if not r["cond_rendered"])}')

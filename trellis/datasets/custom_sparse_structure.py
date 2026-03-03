import os
import json
from typing import Union, Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
from PIL import Image
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
import utils3d


class CustomSparseStructure(StandardDatasetBase):
    """
    自定义稀疏结构数据集，从pt文件加载sparse tensor
    
    数据集目录结构：
    root/
        metadata.csv (包含sha256列，对应pt文件名)
        data/
            {sha256}.pt (存储torch sparse tensor或字典)
    
    Args:
        roots (str): 数据集根目录路径（可以是多个目录，用逗号分隔）
        resolution (int): 体素网格分辨率，默认64
        data_dir_name (str): 数据子目录名称，默认'data'
        file_ext (str): 文件扩展名，默认'.pt'
    """
    
    def __init__(
        self,
        roots: str,
        resolution: int = 64,
        data_dir_name: str = 'data',
        file_ext: str = '.pt',
    ):
        self.resolution = resolution
        self.data_dir_name = data_dir_name
        self.file_ext = file_ext
        self.value_range = (0, 1)
        
        super().__init__(roots)
    
    def filter_metadata(self, metadata: pd.DataFrame):
        """
        过滤元数据。如果没有特殊需求，可以接受所有数据。
        如果需要过滤，可以在这里添加条件。
        """
        stats = {}
        stats['Total'] = len(metadata)
        # 这里可以添加过滤条件，例如：
        # metadata = metadata[metadata['some_column'] == True]
        return metadata, stats
    
    def _load_sparse_tensor(self, file_path: str) -> torch.Tensor:
        """
        从pt文件加载sparse tensor并转换为dense tensor
        
        Args:
            file_path: pt文件路径
            
        Returns:
            形状为 [C, H, W, D] 的dense tensor
        """
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        # 处理不同的存储格式
        if isinstance(data, dict):
            # 如果存储的是字典，可能包含coords和feats
            if 'coords' in data and 'feats' in data:
                coords = data['coords']
                feats = data['feats']
                # coords可能是 [N, 3] 格式，feats是 [N, C] 格式
                if coords.dim() == 2 and feats.dim() == 2:
                    dense = torch.zeros(
                        feats.shape[1], 
                        self.resolution, 
                        self.resolution, 
                        self.resolution,
                        dtype=feats.dtype
                    )
                    # 将sparse坐标转换为dense
                    coords_int = coords.long()
                    # 确保坐标在有效范围内
                    valid_mask = (
                        (coords_int >= 0).all(dim=1) & 
                        (coords_int < self.resolution).all(dim=1)
                    )
                    coords_int = coords_int[valid_mask]
                    feats = feats[valid_mask]
                    if coords_int.shape[0] > 0:
                        dense[:, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = feats.t()
                    return dense
            # 如果字典中直接有dense tensor
            if 'dense' in data or 'tensor' in data:
                tensor = data.get('dense', data.get('tensor'))
                if isinstance(tensor, torch.Tensor):
                    return tensor
        elif isinstance(data, torch.sparse.FloatTensor) or isinstance(data, torch.sparse_coo_tensor):
            data = data.coalesce()
            # 如果是torch的sparse tensor，转换为dense
            if data.layout == torch.sparse_coo:
                # indices = data.indices()
                # values = data.values()
                # # indices是 [ndim, nnz] 格式
                # dense_shape = list(data.shape)
                # dense = torch.zeros(dense_shape, dtype=values.dtype)
                # if indices.shape[0] == 3:  # 3D
                #     dense[0, indices[0], indices[1], indices[2]] = values
                #     # 如果不是13通道，需要扩展到13通道
                #     if dense.shape[0] == 1:
                #         dense = dense.repeat(13, 1, 1, 1)
                return data.to_dense().permute(3,0,1,2)
            else:
                # 其他sparse格式，尝试to_dense()
                return data.to_dense().permute(3,0,1,2)
        elif isinstance(data, torch.Tensor):
            # 如果直接是dense tensor
            if data.dim() == 4:  # [C, H, W, D]
                return data
            elif data.dim() == 3:  # [H, W, D]，需要添加通道维度
                return data.unsqueeze(0).repeat(13, 1, 1, 1)
        
        # 如果都不匹配，抛出错误
        raise ValueError(f"无法识别的数据格式: {type(data)}")
    
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        加载单个实例
        
        Args:
            root: 数据集根目录
            instance: 实例ID（sha256或文件名，不含扩展名）
            
        Returns:
            包含'ss'键的字典，值为 [C, H, W, D] 格式的tensor
        """
        data_dir = os.path.join(root, self.data_dir_name)
        file_path = os.path.join(data_dir, f"{instance}{self.file_ext}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 加载sparse tensor并转换为dense
        dense_tensor = self._load_sparse_tensor(file_path)
        # dense_tensor = dense_tensor[0:1,:, :, :]
        # 确保形状正确 [C, H, W, D]
        if dense_tensor.dim() != 4:
            raise ValueError(f"期望4维tensor [C, H, W, D]，但得到形状: {dense_tensor.shape}")
        
        # 确保分辨率匹配
        if dense_tensor.shape[1:] != (self.resolution, self.resolution, self.resolution):
            # 如果分辨率不匹配，可以调整大小
            dense_tensor = torch.nn.functional.interpolate(
                dense_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution, self.resolution),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        # 添加batch维度，转换为 [1, C, H, W, D] 格式（训练时需要）
        # 但这里先保持 [C, H, W, D]，在collate时再处理
        return {'ss': dense_tensor}
    
    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        """
        可视化样本（可选实现）
        """
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        ss = ss[:,0:1]
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)


class SimpleSparseStructureDataset(torch.utils.data.Dataset):
    """
    简化的数据集类，不需要metadata.csv文件
    直接从目录中读取所有pt文件
    
    Args:
        data_dir (str): 包含pt文件的目录
        resolution (int): 体素网格分辨率，默认64
        file_ext (str): 文件扩展名，默认'.pt'
    """
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 64,
        file_ext: str = '.pt',
    ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.file_ext = file_ext
        
        # 获取所有pt文件
        self.file_list = [
            f[:-len(file_ext)] if f.endswith(file_ext) else f
            for f in os.listdir(data_dir)
            if f.endswith(file_ext)
        ]
        self.file_list.sort()
        
        print(f"找到 {len(self.file_list)} 个数据文件")
    
    def _load_sparse_tensor(self, file_path: str) -> torch.Tensor:
        """从pt文件加载sparse tensor并转换为dense tensor"""
        data = torch.load(file_path, map_location='cpu')
        
        # 处理不同的存储格式（与CustomSparseStructure相同）
        if isinstance(data, dict):
            if 'coords' in data and 'feats' in data:
                coords = data['coords']
                feats = data['feats']
                if coords.dim() == 2 and feats.dim() == 2:
                    dense = torch.zeros(
                        feats.shape[1], 
                        self.resolution, 
                        self.resolution, 
                        self.resolution,
                        dtype=feats.dtype
                    )
                    coords_int = coords.long()
                    valid_mask = (
                        (coords_int >= 0).all(dim=1) & 
                        (coords_int < self.resolution).all(dim=1)
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
        elif isinstance(data, torch.sparse.FloatTensor) or isinstance(data, torch.sparse_coo_tensor):
            if data.layout == torch.sparse_coo:
                indices = data.indices()
                values = data.values()
                dense_shape = list(data.shape)
                dense = torch.zeros(dense_shape, dtype=values.dtype)
                if indices.shape[0] == 4:
                    dense[indices[0], indices[1], indices[2], indices[3]] = values
                elif indices.shape[0] == 3:
                    dense[0, indices[0], indices[1], indices[2]] = values
                    if dense.shape[0] == 1:
                        dense = dense.repeat(13, 1, 1, 1)
                return dense
            else:
                return data.to_dense()
        elif isinstance(data, torch.Tensor):
            if data.dim() == 4:
                return data
            elif data.dim() == 3:
                return data.unsqueeze(0).repeat(13, 1, 1, 1)
        
        raise ValueError(f"无法识别的数据格式: {type(data)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, f"{file_name}{self.file_ext}")
        
        dense_tensor = self._load_sparse_tensor(file_path)
        
        if dense_tensor.dim() != 4:
            raise ValueError(f"期望4维tensor [C, H, W, D]，但得到形状: {dense_tensor.shape}")
        
        if dense_tensor.shape[1:] != (self.resolution, self.resolution, self.resolution):
            dense_tensor = torch.nn.functional.interpolate(
                dense_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution, self.resolution),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        return {'ss': dense_tensor}




# ==================== Latent Dataset for Flow Matching Training ====================

class CustomSparseStructureLatent(StandardDatasetBase):
    """
    Custom sparse structure latent dataset for flow matching training.
    
    This dataset loads pre-encoded latent representations instead of raw sparse structures,
    which is much more efficient for training flow matching models.
    
    Dataset structure:
        root/
            metadata.csv (contains sha256 column)
            ss_latents/{latent_model}/
                {sha256}.npz (contains 'mean' array for latent code)
    
    Args:
        roots (str): Dataset root paths (comma-separated for multiple datasets)
        latent_model (str): Name of the latent encoder model used (e.g., 'ss_enc_conv3d_16l8_fp16')
        normalization (dict, optional): Normalization statistics {'mean': [...], 'std': [...]}
            If provided, latents will be normalized: z_normalized = (z - mean) / std
    """
    
    # Flag to indicate this dataset returns latents, not images
    is_latent_dataset = True
    
    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        normalization: Optional[dict] = None,
    ):
        self.latent_model = latent_model
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(roots)
        
        # Setup loads for balanced sampling
        # If metadata has 'num_voxels' column, use it; otherwise use uniform load
        if 'num_voxels' in self.metadata.columns:
            self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        else:
            # Use uniform load (assumes all samples have similar complexity)
            self.loads = [1.0] * len(self.instances)
        
        # Setup normalization if provided
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
    
    def filter_metadata(self, metadata: pd.DataFrame):
        """Filter metadata to only include instances with latent encodings."""
        stats = {}
        # Check if latent column exists
        latent_col = f'ss_latent_{self.latent_model}'
        if latent_col in metadata.columns:
            metadata = metadata[metadata[latent_col]]
            stats[f'With {self.latent_model} latents'] = len(metadata)
        else:
            stats['Warning: No latent column found, using all data'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        Load a single latent instance.
        
        Args:
            root: Dataset root directory
            instance: Instance ID (sha256)
            
        Returns:
            Dictionary containing 'x_0': latent tensor of shape [C, H, W, D]
        """
        latent_path = os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz')
        
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        # Load latent
        latent = np.load(latent_path)
        z = torch.tensor(latent['mean']).float()
        
        # Apply normalization if configured
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        
        return {'x_0': z}
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function for dense latent tensors.
        
        Args:
            batch: List of dictionaries containing 'x_0' tensors
            split_size: If specified, split batch into sub-batches (for gradient accumulation)
        
        Returns:
            Dictionary or list of dictionaries with batched data
        """
        if split_size is None:
            # No splitting, return single batch
            pack = {}
            pack['x_0'] = torch.stack([b['x_0'] for b in batch])
            
            # Collate other keys if present
            keys = [k for k in batch[0].keys() if k != 'x_0']
            for k in keys:
                if isinstance(batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in batch])
                elif isinstance(batch[0][k], list):
                    pack[k] = sum([b[k] for b in batch], [])
                else:
                    pack[k] = [b[k] for b in batch]
            
            return pack
        else:
            # Split into sub-batches for gradient accumulation
            from ...utils.data_utils import load_balanced_group_indices
            
            # Use uniform load for dense tensors (all same size)
            loads = [1] * len(batch)
            group_idx = load_balanced_group_indices(loads, split_size)
            
            packs = []
            for group in group_idx:
                sub_batch = [batch[i] for i in group]
                pack = {}
                pack['x_0'] = torch.stack([b['x_0'] for b in sub_batch])
                
                # Collate other keys
                keys = [k for k in sub_batch[0].keys() if k != 'x_0']
                for k in keys:
                    if isinstance(sub_batch[0][k], torch.Tensor):
                        pack[k] = torch.stack([b[k] for b in sub_batch])
                    elif isinstance(sub_batch[0][k], list):
                        pack[k] = sum([b[k] for b in sub_batch], [])
                    else:
                        pack[k] = [b[k] for b in sub_batch]
                
                packs.append(pack)
            
            return packs

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        """
        Visualize latent samples by showing slices of the latent tensor.
        
        Since latent representations are 4D (C, H, W, D), we visualize them by:
        1. Taking a middle slice along the depth dimension
        2. Displaying multiple channels as grayscale images
        
        Args:
            sample: Dictionary containing 'x_0' tensor of shape [B, C, H, W, D]
            
        Returns:
            Visualization tensor of shape [B, 3, H, W] suitable for save_image
        """
        x_0 = sample['x_0'] if isinstance(sample, dict) else sample
        
        # x_0 shape: [B, C, H, W, D] (e.g., [16, 8, 16, 16, 16])
        B, C, H, W, D = x_0.shape
        
        # Take middle slice along depth dimension
        mid_slice = x_0[:, :, :, :, D // 2]  # [B, C, H, W]
        
        # For visualization, take first 3 channels or repeat if less than 3
        if C >= 3:
            vis = mid_slice[:, :3, :, :]  # [B, 3, H, W]
        else:
            # Repeat the first channel to create RGB
            vis = mid_slice[:, 0:1, :, :].repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # Normalize to [0, 1] for visualization
        vis = vis - vis.min()
        if vis.max() > 0:
            vis = vis / vis.max()
        
        return vis


class TextConditionedCustomSparseStructureLatent(TextConditionedMixin, CustomSparseStructureLatent):
    """
    Text-conditioned custom sparse structure latent dataset.
    
    Requires 'captions' column in metadata.csv containing JSON-formatted caption lists.
    """
    pass


class ImageConditionedCustomSparseStructureLatent(ImageConditionedMixin, CustomSparseStructureLatent):
    """
    Image-conditioned custom sparse structure latent dataset.
    
    Requires rendered condition images and 'cond_rendered' column in metadata.csv.
    """
    pass


# ==================== Pseudo-Color Image Conditioned ====================

# 8类伪颜色映射 (对应通道1-8的one-hot编码)
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


class PseudoColorImageConditionedMixin:
    """
    用于 custom_sparse_structure 的伪颜色图像条件 Mixin。
    
    每个活跃体素的伪颜色由数据的第 1-8 通道的 one-hot 编码决定。
    训练时从 renders_cond/ 加载预渲染的伪颜色图像作为条件输入，
    由训练器的 ImageConditionedMixin 使用 DINOv2 编码为特征。
    
    数据集目录结构:
        root/
            metadata.csv  (需要 cond_rendered 列)
            ss_latents/{latent_model}/
            renders_cond/{sha256}/
                0000.png, 0001.png, ...
                transforms.json
    
    Args:
        image_size: 输入图像大小（默认518，DINOv2输入尺寸）
    """
    
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[metadata['cond_rendered'] == True]
            stats['With cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        
        # 加载条件渲染图像
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        frame_meta = metadata['frames'][view]
        
        image_path = os.path.join(image_root, frame_meta['file_path'])
        image = Image.open(image_path)
        
        # 基于 alpha 通道裁剪到目标区域
        if image.mode == 'RGBA':
            alpha = np.array(image.getchannel(3))
            bbox_yx = np.array(alpha > 0).nonzero()
            if len(bbox_yx[0]) > 0:
                y_min, y_max = bbox_yx[0].min(), bbox_yx[0].max()
                x_min, x_max = bbox_yx[1].min(), bbox_yx[1].max()
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                hsize = max(x_max - x_min, y_max - y_min) / 2
                aug_hsize = hsize * 1.2
                crop_box = (
                    int(center[0] - aug_hsize),
                    int(center[1] - aug_hsize),
                    int(center[0] + aug_hsize),
                    int(center[1] + aug_hsize),
                )
                image = image.crop(crop_box)
        
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        if image.mode == 'RGBA':
            alpha_ch = image.getchannel(3)
            image = image.convert('RGB')
            alpha_t = torch.tensor(np.array(alpha_ch)).float() / 255.0
        else:
            image = image.convert('RGB')
            alpha_t = torch.ones(self.image_size, self.image_size)
        
        image_t = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        # 白色背景合成
        image_t = image_t * alpha_t.unsqueeze(0) + (1.0 - alpha_t.unsqueeze(0))
        
        pack['cond'] = image_t
        return pack


class PseudoColorImageConditionedCustomSparseStructureLatent(
    PseudoColorImageConditionedMixin,
    CustomSparseStructureLatent,
):
    """
    伪颜色图像条件的 custom sparse structure latent 数据集。
    
    组合:
      - PseudoColorImageConditionedMixin: 加载伪颜色渲染图像作为条件
      - CustomSparseStructureLatent: 加载 VAE 编码后的 latent 作为训练目标
    
    完整使用流程:
      1. dataset_toolkits/render_cond_ss.py 渲染伪颜色条件图像
      2. 使用本数据集 + ImageConditionedFlowMatchingCFGTrainer 训练
      3. infer.py img_cond_gen 进行推理
    """
    pass

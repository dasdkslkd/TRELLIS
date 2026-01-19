import os
from typing import Union, Dict, Any
import torch
import numpy as np
import pandas as pd
from .components import StandardDatasetBase
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




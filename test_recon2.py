import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_voxel_tensor(tensor, name, mode='type', sample_rate=1):
    """
    可视化14通道体素张量（坐标系对齐版本）
    :param tensor: torch.Tensor, shape (14, 64, 64, 64)
    :param mode: 'type' 或 'property_0' 到 'property_5'
    :param sample_rate: 采样率，用于性能优化
    """
    # 转换并获取占用掩码
    tensor = tensor.detach().cpu()
    occ = tensor[0].numpy()
    mask = occ > 0
    
    # ====== 核心修改：索引转连续物理坐标 ======
    res = occ.shape[0]  # 分辨率，通常为64
    vs = 2.0 / res      # 体素大小，覆盖 [-1, 1] 范围
    
    # 获取整数索引并映射到体素中心坐标
    idx_coords = np.argwhere(mask).astype(np.float32)  # shape (N, 3), 值域 [0, res-1]
    
    # 映射公式: coord = (index + 0.5) * vs - 1.0
    # -1.0 是网格起点，+0.5 对齐体素中心
    coords = (idx_coords + 0.5) * vs - 1.0
    # ==========================================
    
    # 根据模式获取颜色数据（后续逻辑不变）
    if mode == 'type':
        types = tensor[1:8].argmax(dim=0).numpy()[mask]
        colors = plt.cm.tab10(types % 10)
        title = 'Voxel Types'
    else:
        prop_idx = int(mode.split('_')[1])
        attr = tensor[8 + prop_idx].numpy()[mask]
        attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        colors = plt.cm.viridis(attr_norm)
        title = f'Property {prop_idx}'
    
    # 创建交互式3D散点图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制体素点（现在使用连续坐标）
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=colors, s=30, alpha=0.8, edgecolors='none'
    )
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴范围确保比例一致
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # 设置初始视角（与vis_recon的iso视角接近）
    ax.view_init(elev=30, azim=315)
    
    plt.savefig(f'output/images/{name}.png')
    plt.show()


# 使用示例
# gt = torch.load('../TRELLIS/dataset/sparse_vae_dataset/val/data/135906_587b5c3d_11.pt', map_location='cpu')
# gt = gt.to_dense().permute(3,0,1,2)
# recon = torch.load('output/135906_587b5c3d_11_recon.pt', map_location='cpu')
# mode = 'type'
# visualize_voxel_tensor(gt, name='gt_aligned', mode=mode)
# visualize_voxel_tensor(recon[0], name='recon', mode=mode)

data = torch.load('output/generated/sample_0003.pt', map_location='cpu')
# data = torch.load('../TRELLIS/dataset/sparse_vae_dataset/val/data/86704_3f8f3bfe_7.pt').to_dense().permute(3,0,1,2).unsqueeze(0)
data = torch.load('output/recon/reconstructed.pt', map_location='cpu')
print(data.shape)  # 应为 (14, 64, 64, 64)
visualize_voxel_tensor(data[0], name='generated_sample_0000', mode='type')
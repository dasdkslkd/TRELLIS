import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

def visualize_3d_model(tensor, output_path=None):
    """
    可视化3D模型张量
    
    参数:
        tensor: numpy数组,形状[C,H,W,D],C=8
               第0维: occupancy field (0或1)
               第1-7维: one-hot标签(7个类别)
        output_path: 可选,保存图像的路径
    """
    # 提取占用场和标签
    occupancy = tensor[0]  # 形状 [H,W,D]
    labels_onehot = tensor[1:]  # 形状 [7,H,W,D]
    
    # 将one-hot标签转换为类别索引 (0-6)
    labels = np.argmax(labels_onehot, axis=0)  # 形状 [H,W,D]
    
    # 只保留被占用的体素的标签
    labels = labels * occupancy
    
    # 创建PyVista的体数据
    grid = pv.ImageData()
    grid.dimensions = np.array(occupancy.shape) + 1  # 体素维度+1
    grid.spacing = (1, 1, 1)  # 体素间距
    
    # 将占用场数据作为cell data添加
    grid.cell_data['occupancy'] = occupancy.astype(np.float32).ravel(order='F')
    
    # 将cell data转换为point data
    grid_point_data = grid.cell_data_to_point_data()
    
    # 使用marching cubes提取表面 (阈值0.5提取占用场中的1)
    contours = grid_point_data.contour(isosurfaces=[0.5], scalars='occupancy')
    
    if contours.n_points == 0:
        print("警告: 未提取到任何表面，占用场可能全为0")
        return
    
    # 为每个顶点插值标签值
    vertex_labels = np.zeros(contours.n_points, dtype=int)
    
    # 获取顶点坐标并转换为体素索引
    points = contours.points
    voxel_coords = np.floor(points).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, np.array(labels.shape) - 1)
    
    # 从标签体数据中获取每个顶点的标签
    for i, (x, y, z) in enumerate(voxel_coords):
        vertex_labels[i] = labels[x, y, z]
    
    # 为每个类别分配颜色 (7个类别)
    base_cmap = plt.cm.tab10
    colors = base_cmap(np.linspace(0, 1, 8))[:7]  # 取7个颜色
    custom_cmap = ListedColormap(colors)
    
    # 将标签映射到RGB颜色
    vertex_colors = custom_cmap(vertex_labels)
    
    # 为网格设置颜色
    contours['colors'] = vertex_colors
    
    # 创建可视化
    plotter = pv.Plotter(window_size=[1200, 1000])
    plotter.add_mesh(contours, scalars='colors', rgb=True, smooth_shading=True)
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 设置相机视角
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # 添加标签颜色图例
    for i in range(7):
        color = tuple(colors[i][:3])  # RGB值
        plotter.add_text(f"类别 {i}", position=(10, 10 + i*25), color=color, font_size=12)
    plotter.show()
    # 显示或保存
    if output_path:
        plotter.screenshot(output_path)
        print(f"图像已保存到: {output_path}")
    
    
    
    return plotter

# recon = torch.load('output/86704_3f8f3bfe_7_recon.pt', map_location='cpu')  # shape (N, 14, 64, 64, 64
# gt = torch.load('../TRELLIS/dataset/sparse_vae_dataset/val/data/86704_3f8f3bfe_7.pt', map_location='cpu')

gt = torch.load('output/86704_3f8f3bfe_7 copy.pt', map_location='cpu')

# 可视化第一个样本的重建结果
# recon_np = recon[0].numpy()  # shape (14, 64, 64, 64)
# visualize_3d_model(recon_np[:8], output_path='recon_visualization.png')
# 可视化第一个样本的真实数据
gt_np = gt.to_dense().permute(3,0,1,2).numpy()  # shape (14, 64, 64, 64)
visualize_3d_model(gt_np[:8], output_path='gt_visualization.png')
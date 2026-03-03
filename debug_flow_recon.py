"""
调试Flow模型重建质量的脚本

用于诊断重建性能差的原因
"""
import torch
import numpy as np
import json
from easydict import EasyDict as edict
import argparse
import os

# 使用 infer.py 中的函数
from infer import load_flow_model, load_model, diffuse_latent, sample_from_noisy, mIoU, mIoU_occ


def analyze_reconstruction_at_different_t():
    """
    测试不同加噪水平下的重建质量
    帮助诊断模型在哪个时间步开始出问题
    """
    parser = argparse.ArgumentParser(description='分析不同加噪水平的重建质量')
    parser.add_argument('--config', type=str, required=True, help='Flow模型配置文件')
    parser.add_argument('--ckpt_flow', type=str, required=True, help='Flow模型检查点')
    parser.add_argument('--vae_config', type=str, default=None, help='VAE配置文件')
    parser.add_argument('--ckpt_decoder', type=str, default=None, help='VAE Decoder检查点')
    parser.add_argument('--input', type=str, required=True, help='输入latent文件')
    parser.add_argument('--output', type=str, default='output/debug', help='输出目录')
    parser.add_argument('--steps', type=int, default=50, help='采样步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    opt = parser.parse_args()
    
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    os.makedirs(opt.output, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    cfg = edict(json.load(open(opt.config)))
    flow_model = load_flow_model(cfg.models.denoiser, opt.ckpt_flow)
    
    # 加载latent
    print(f"加载latent: {opt.input}")
    if opt.input.endswith('.npz'):
        data = np.load(opt.input)
        latent = torch.from_numpy(data['mean']).unsqueeze(0).cuda()
    else:
        latent = torch.load(opt.input, map_location='cuda', weights_only=True)
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Latent stats: mean={latent.mean():.4f}, std={latent.std():.4f}, min={latent.min():.4f}, max={latent.max():.4f}")
    
    # 加载decoder (可选)
    decoder = None
    if opt.vae_config and opt.ckpt_decoder:
        vae_cfg = edict(json.load(open(opt.vae_config)))
        decoder = load_model(vae_cfg.models.decoder, opt.ckpt_decoder, 'decoder')
    
    # 测试不同加噪水平
    t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("\n" + "="*80)
    print("不同加噪水平的重建质量分析")
    print("="*80)
    print(f"{'t_noise':>10} | {'Latent MSE':>12} | {'Latent Cosine':>14} | {'Voxel mIoU':>12} | {'Voxel mIoU_occ':>14}")
    print("-"*80)
    
    results = []
    
    for t in t_values:
        # 固定噪声以便比较
        noise = torch.randn_like(latent)
        
        # 加噪
        x_noisy, _ = diffuse_latent(latent, t, noise=noise)
        
        # 重建
        recon = sample_from_noisy(flow_model, x_noisy, t, num_steps=opt.steps)
        
        # 计算latent空间指标
        mse = torch.nn.functional.mse_loss(recon, latent).item()
        cosine = torch.nn.functional.cosine_similarity(
            recon.flatten(), latent.flatten(), dim=0
        ).item()
        
        # 计算voxel空间指标 (如果有decoder)
        voxel_miou = voxel_miou_occ = float('nan')
        if decoder is not None:
            orig_voxel = decoder(latent)
            recon_voxel = decoder(recon)
            voxel_miou = mIoU(recon_voxel, orig_voxel).item()
            voxel_miou_occ = mIoU_occ(recon_voxel[:, :1], orig_voxel[:, :1])
        
        print(f"{t:>10.1f} | {mse:>12.6f} | {cosine:>14.6f} | {voxel_miou:>12.4f} | {voxel_miou_occ:>14.4f}")
        
        results.append({
            't': t,
            'mse': mse,
            'cosine': cosine,
            'voxel_miou': voxel_miou,
            'voxel_miou_occ': voxel_miou_occ
        })
    
    print("="*80)
    
    # 保存结果
    results_data = results  # 避免变量名冲突
    with open(os.path.join(opt.output, 'reconstruction_analysis.json'), 'w') as f:
        import json as json_module
        json_module.dump(results_data, f, indent=2)
    
    print(f"\n结果已保存到: {opt.output}/reconstruction_analysis.json")
    
    # 诊断建议
    print("\n" + "="*80)
    print("诊断建议:")
    print("="*80)
    
    # 检查不同时间步的表现
    low_t_mse = np.mean([r['mse'] for r in results if r['t'] <= 0.3])
    mid_t_mse = np.mean([r['mse'] for r in results if 0.3 < r['t'] <= 0.7])
    high_t_mse = np.mean([r['mse'] for r in results if r['t'] > 0.7])
    
    print(f"低时间步 (t≤0.3) 平均MSE: {low_t_mse:.6f}")
    print(f"中时间步 (0.3<t≤0.7) 平均MSE: {mid_t_mse:.6f}")
    print(f"高时间步 (t>0.7) 平均MSE: {high_t_mse:.6f}")
    
    if high_t_mse > 5 * low_t_mse:
        print("\n⚠️ 高时间步重建误差显著高于低时间步")
        print("   建议: ")
        print("   1. 调整t_schedule，增加高时间步的采样概率")
        print("   2. 增加训练步数")
        print("   3. 使用更多采样步数进行推理")
    
    if low_t_mse > 0.1:
        print("\n⚠️ 即使低时间步的重建也有较大误差")
        print("   建议: ")
        print("   1. 检查模型是否正确加载")
        print("   2. 检查latent数据分布是否与训练时一致")
        print("   3. 可能需要更大的模型容量")


def test_sampling_steps():
    """
    测试不同采样步数对生成质量的影响
    """
    parser = argparse.ArgumentParser(description='测试采样步数影响')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_flow', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()
    
    torch.manual_seed(opt.seed)
    
    cfg = edict(json.load(open(opt.config)))
    flow_model = load_flow_model(cfg.models.denoiser, opt.ckpt_flow)
    
    if opt.input.endswith('.npz'):
        data = np.load(opt.input)
        latent = torch.from_numpy(data['mean']).unsqueeze(0).cuda()
    else:
        latent = torch.load(opt.input, map_location='cuda', weights_only=True)
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
    
    print("测试不同采样步数的重建质量 (t=1.0, 从纯噪声开始)")
    print("="*60)
    
    noise = torch.randn_like(latent)
    x_noisy, _ = diffuse_latent(latent, 1.0, noise=noise)
    
    for steps in [10, 25, 50, 100, 200]:
        recon = sample_from_noisy(flow_model, x_noisy, 1.0, num_steps=steps)
        mse = torch.nn.functional.mse_loss(recon, latent).item()
        cosine = torch.nn.functional.cosine_similarity(
            recon.flatten(), latent.flatten(), dim=0
        ).item()
        print(f"Steps={steps:>4}: MSE={mse:.6f}, Cosine={cosine:.6f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'steps':
        sys.argv.pop(1)
        test_sampling_steps()
    else:
        analyze_reconstruction_at_different_t()

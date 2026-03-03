import argparse
import json
import torch
import numpy as np
from easydict import EasyDict as edict
from trellis import models

def load_model(cfg, ckpt_path, key):
    """极简模型加载器"""
    model = getattr(models, cfg.name)(**cfg.args).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    model.load_state_dict(ckpt.get(f'{key}_state_dict', ckpt.get('model_state_dict', ckpt)))
    return model


def load_flow_model(cfg, ckpt_path):
    """加载flow matching denoiser模型"""
    model = getattr(models, cfg.name)(**cfg.args).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    # 支持EMA检查点和普通检查点
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model

def mIoU(pred, target, thr = 0.5):
    """
    计算3D稀疏体素的mIoU
    pred: 预测 (B,8,D,H,W)
    target: 真值 (B,8,D,H,W)
    thr: 占用阈值
    """
    occ = target[:, 0] > thr  # 占用掩码
    match = pred[:, 1:8].argmax(1) == target[:, 1:8].argmax(1)  # 类型匹配
    return (match & occ).sum().float() / occ.sum().float().clamp(min=1)

def mIoU_occ(p, t, thr=0.5):
    """
    计算占用场mIoU
    p: 预测占用场 (B, 1, D, H, W) 或 (B, D, H, W)
    t: 真值占用场 (B, 1, D, H, W) 或 (B, D, H, W)
    """
    p, t = (p > thr).flatten(1), (t > thr).flatten(1)  # (B, N)
    inter = (p & t).sum(1).float()
    union = (p | t).sum(1).float()
    return (inter / (union + 1e-8)).mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.pth')
    parser.add_argument('--mode', type=str, choices=['encode', 'decode', 'full'], required=True)
    parser.add_argument('--ckpt_encoder', type=str, help='encode/full模式必需')
    parser.add_argument('--ckpt_decoder', type=str, help='decode/full模式必需')
    opt = parser.parse_args()
    
    cfg = edict(json.load(open(opt.config)))
    data = torch.load(opt.input, map_location='cuda', weights_only=True)
    data = data.to_dense().permute(3,0,1,2).unsqueeze(0)  # [1 x C x H x W x D]
    
    with torch.no_grad():
        if opt.mode == 'encode':
            output = load_model(cfg.models.encoder, opt.ckpt_encoder, 'encoder')(data)
        elif opt.mode == 'decode':
            output = load_model(cfg.models.decoder, opt.ckpt_decoder, 'decoder')(data)
        else:  # full模式: 编码后解码
            latent = load_model(cfg.models.encoder, opt.ckpt_encoder, 'encoder')(data)
            output = load_model(cfg.models.decoder, opt.ckpt_decoder, 'decoder')(latent)
    
    torch.save(output.cpu(), opt.output)
    miou = mIoU(output, data)
    miou_occ = mIoU_occ(output[:, :1], data[:, :1])
    print(f'[{opt.mode}] 结果: {opt.output}\n  mIoU: {miou:.4f}\n  mIoU_occ: {miou_occ:.4f}')


def gen_latent_dataset():
    """
    批量生成latent数据集，用于flow matching训练
    
    使用方法:
        python infer.py gen_latent \
            --config configs/vae/ss_vae_enc_dec_gs_swin8_B_64l8_fp16.json \
            --ckpt_encoder ckpt/encoder_ema0.9999_step0050000.pt \
            --data_dir /path/to/dataset \
            --output_dir /path/to/dataset/ss_latents/ss_enc_conv3d_16l8_fp16 \
            --batch_size 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='VAE配置文件')
    parser.add_argument('--ckpt_encoder', type=str, required=True, help='Encoder检查点路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='Latent输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--data_dir_name', type=str, default='data', help='数据子目录名')
    parser.add_argument('--file_ext', type=str, default='.pt', help='文件扩展名')
    opt = parser.parse_args()
    
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    
    # 创建输出目录
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # 加载配置和模型
    print(f"加载配置: {opt.config}")
    cfg = edict(json.load(open(opt.config)))
    
    print(f"加载encoder: {opt.ckpt_encoder}")
    encoder = load_model(cfg.models.encoder, opt.ckpt_encoder, 'encoder')
    
    # 读取metadata
    metadata_path = os.path.join(opt.data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"错误: metadata.csv不存在: {metadata_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    if 'sha256' not in metadata.columns:
        print(f"错误: metadata.csv缺少sha256列")
        return
    
    print(f"找到 {len(metadata)} 个样本")
    
    # 批量编码
    data_dir = os.path.join(opt.data_dir, opt.data_dir_name)
    successful = 0
    failed = 0
    
    with torch.no_grad():
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="编码latent"):
            sha256 = row['sha256']
            input_path = os.path.join(data_dir, f"{sha256}{opt.file_ext}")
            output_path = os.path.join(opt.output_dir, f"{sha256}.npz")
            
            # 跳过已存在的文件
            if os.path.exists(output_path):
                successful += 1
                continue
            
            try:
                # 加载数据
                data = torch.load(input_path, map_location='cuda', weights_only=True)
                
                # 处理sparse tensor
                if hasattr(data, 'to_dense'):
                    data = data.to_dense()
                # print(data.shape)
                # 确保格式正确 [C, H, W, D]
                if data.dim() == 4:
                    # 可能是 [H, W, D, C] 格式，需要permute
                    data = data.permute(3, 0, 1, 2)
                
                # 添加batch维度 [1, C, H, W, D]
                data = data.unsqueeze(0).cuda()
                # print(data.shape)
                # 编码
                latent = encoder(data)
                
                # 保存为npz格式
                np.savez_compressed(
                    output_path,
                    mean=latent.cpu().numpy()[0]  # 移除batch维度
                )
                
                successful += 1
                
            except Exception as e:
                print(f"\n错误处理 {sha256}: {e}")
                failed += 1
                exit(1)
    
    print(f"\n完成!")
    print(f"  成功: {successful}/{len(metadata)}")
    print(f"  失败: {failed}/{len(metadata)}")
    print(f"  输出目录: {opt.output_dir}")
    
    # 更新metadata标记列
    latent_model = os.path.basename(opt.output_dir)
    col_name = f'ss_latent_{latent_model}'
    
    # 检查哪些文件存在
    metadata[col_name] = metadata['sha256'].apply(
        lambda x: os.path.exists(os.path.join(opt.output_dir, f"{x}.npz"))
    )
    
    # 保存更新的metadata
    metadata.to_csv(metadata_path, index=False)
    print(f"  已更新metadata.csv，添加列: {col_name}")
    print(f"  标记为有latent的样本数: {metadata[col_name].sum()}")


@torch.no_grad()
def sample_flow_ode(model, shape, num_steps=50, cfg_scale=1.0, cond=None, sigma_min=1e-5):
    """
    使用ODE采样器进行Flow Matching采样 (Euler方法)
    
    与官方FlowEulerSampler实现保持一致:
    - 时间步乘以1000后传给模型
    - 使用正确的速度场到x_prev的转换公式
    
    Args:
        model: Flow matching denoiser模型
        shape: 输出形状 (B, C, D, H, W)
        num_steps: 采样步数
        cfg_scale: CFG引导强度 (仅用于条件生成)
        cond: 条件嵌入 (可选)
        sigma_min: 最小噪声水平
    
    Returns:
        生成的latent tensor
    """
    device = next(model.parameters()).device
    
    # 从标准正态分布开始
    x = torch.randn(shape, device=device, dtype=torch.float32)
    
    # 时间步从1到0 (与官方实现一致)
    t_seq = np.linspace(1, 0, num_steps + 1)
    
    for i in range(num_steps):
        t = t_seq[i]
        t_prev = t_seq[i + 1]
        
        # 时间步乘以1000后传给模型 (与官方实现一致)
        t_batch = torch.tensor([1000 * t] * shape[0], device=device, dtype=torch.float32)
        
        # 计算速度场
        if cond is not None and cfg_scale > 1.0:
            # 有条件生成 + CFG
            v_cond = model(x, t_batch, cond)
            v_uncond = model(x, t_batch, None)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # 无条件生成
            v = model(x, t_batch, cond)
        
        # Euler step: x_{t-1} = x_t - (t - t_prev) * v
        x = x - (t - t_prev) * v
    
    return x


@torch.no_grad()
def sample_flow_heun(model, shape, num_steps=50, cfg_scale=1.0, cond=None, sigma_min=1e-5):
    """
    使用Heun方法进行Flow Matching采样 (二阶)
    
    与官方实现保持一致的时间步处理
    
    Args:
        model: Flow matching denoiser模型
        shape: 输出形状 (B, C, D, H, W)
        num_steps: 采样步数
        cfg_scale: CFG引导强度
        cond: 条件嵌入 (可选)
        sigma_min: 最小噪声水平
    
    Returns:
        生成的latent tensor
    """
    device = next(model.parameters()).device
    
    def get_v(x, t_float):
        # 时间步乘以1000后传给模型
        t_batch = torch.tensor([1000 * t_float] * shape[0], device=device, dtype=torch.float32)
        if cond is not None and cfg_scale > 1.0:
            v_cond = model(x, t_batch, cond)
            v_uncond = model(x, t_batch, None)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return model(x, t_batch, cond)
    
    x = torch.randn(shape, device=device, dtype=torch.float32)
    t_seq = np.linspace(1, 0, num_steps + 1)
    
    for i in range(num_steps):
        t = t_seq[i]
        t_prev = t_seq[i + 1]
        
        # Heun's method
        v1 = get_v(x, t)
        x_euler = x - (t - t_prev) * v1
        v2 = get_v(x_euler, t_prev)
        x = x - 0.5 * (t - t_prev) * (v1 + v2)
    
    return x


def uncond_generate():
    """
    无条件生成3D样本
    
    使用方法:
        # 仅生成latent
        python infer.py uncond_gen \
            --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
            --ckpt_flow ckpt/flow_ema0.9999_step0050000.pt \
            --output output/generated \
            --num_samples 4 \
            --steps 50
        
        # 生成latent并解码为3D体素
        python infer.py uncond_gen \
            --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
            --ckpt_flow ckpt/flow_ema0.9999_step0050000.pt \
            --vae_config configs/vae/ss_vae_conv3d_16l8_fp16.json \
            --ckpt_decoder ckpt/decoder_ema0.9999_step0040000.pt \
            --output output/generated \
            --num_samples 4 \
            --steps 50 \
            --decode
    """
    import os
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='无条件生成3D样本')
    parser.add_argument('--config', type=str, required=True, help='Flow模型配置文件')
    parser.add_argument('--ckpt_flow', type=str, required=True, help='Flow模型检查点')
    parser.add_argument('--vae_config', type=str, default=None, help='VAE配置文件 (解码时需要)')
    parser.add_argument('--ckpt_decoder', type=str, default=None, help='VAE Decoder检查点 (解码时需要)')
    parser.add_argument('--output', type=str, default='output/generated', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数量')
    parser.add_argument('--batch_size', type=int, default=4, help='每批生成数量')
    parser.add_argument('--steps', type=int, default=50, help='采样步数')
    parser.add_argument('--sampler', type=str, choices=['euler', 'heun'], default='euler', help='采样器类型')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--decode', action='store_true', help='是否解码为3D体素')
    parser.add_argument('--save_latent', action='store_true', help='是否同时保存latent')
    opt = parser.parse_args()

    normalize_mean = 0
    normalize_std = 2.26
    
    # 设置随机种子
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        print(f"随机种子: {opt.seed}")
    
    # 创建输出目录
    os.makedirs(opt.output, exist_ok=True)
    
    # 加载配置和Flow模型
    print(f"加载Flow配置: {opt.config}")
    cfg = edict(json.load(open(opt.config)))
    
    print(f"加载Flow模型: {opt.ckpt_flow}")
    flow_model = load_flow_model(cfg.models.denoiser, opt.ckpt_flow)
    
    # 获取模型参数
    resolution = cfg.models.denoiser.args.resolution  # 通常是16
    latent_channels = cfg.models.denoiser.args.in_channels  # 通常是8
    
    print(f"Latent shape: ({latent_channels}, {resolution}, {resolution}, {resolution})")
    
    # 加载VAE Decoder (如果需要解码)
    decoder = None
    if opt.decode:
        if opt.vae_config is None or opt.ckpt_decoder is None:
            print("错误: 解码模式需要 --vae_config 和 --ckpt_decoder")
            return
        print(f"加载VAE配置: {opt.vae_config}")
        vae_cfg = edict(json.load(open(opt.vae_config)))
        print(f"加载Decoder: {opt.ckpt_decoder}")
        decoder = load_model(vae_cfg.models.decoder, opt.ckpt_decoder, 'decoder')
    
    # 选择采样器
    sampler_fn = sample_flow_euler if opt.sampler == 'euler' else sample_flow_heun
    print(f"采样器: {opt.sampler}, 步数: {opt.steps}")
    
    # 批量生成
    generated_count = 0
    pbar = tqdm(total=opt.num_samples, desc="生成中")
    
    while generated_count < opt.num_samples:
        current_batch = min(opt.batch_size, opt.num_samples - generated_count)
        shape = (current_batch, latent_channels, resolution, resolution, resolution)
        
        # 生成latent
        latent = sampler_fn(flow_model, shape, num_steps=opt.steps)
        
        # 处理每个样本
        for i in range(current_batch):
            sample_idx = generated_count + i
            latent_i = latent[i:i+1]  # 保持batch维度
            latent_i = latent_i * normalize_std + normalize_mean  # 反归一化
            
            # 保存latent
            if opt.save_latent or not opt.decode:
                latent_path = os.path.join(opt.output, f'sample_{sample_idx:04d}_latent.pt')
                torch.save(latent_i.cpu(), latent_path)
            
            # 解码
            if opt.decode and decoder is not None:
                decoded = decoder(latent_i)
                decoded_path = os.path.join(opt.output, f'sample_{sample_idx:04d}.pt')
                torch.save(decoded.cpu(), decoded_path)
        
        generated_count += current_batch
        pbar.update(current_batch)
    
    pbar.close()
    
    print(f"\n生成完成!")
    print(f"  样本数量: {opt.num_samples}")
    print(f"  输出目录: {opt.output}")
    if opt.decode:
        print(f"  已解码为3D体素")


def sample_flow_euler(model, shape, num_steps=50, cfg_scale=1.0, cond=None, sigma_min=1e-5):
    """sample_flow_ode的别名"""
    return sample_flow_ode(model, shape, num_steps, cfg_scale, cond, sigma_min)


@torch.no_grad()
def diffuse_latent(x_0, t, noise=None, sigma_min=1e-5):
    """
    对latent进行前向扩散（加噪）
    
    Flow Matching扩散公式: x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
    
    Args:
        x_0: 原始latent (B, C, D, H, W)
        t: 扩散时间步 [0-1]，0表示无噪声，1表示纯噪声
        noise: 可选的噪声，如果不提供则随机生成
        sigma_min: 最小噪声水平
    
    Returns:
        x_t: 加噪后的latent
        noise: 使用的噪声（用于后续分析）
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    t_expanded = t if isinstance(t, float) else t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
    x_t = (1 - t_expanded) * x_0 + (sigma_min + (1 - sigma_min) * t_expanded) * noise
    
    return x_t, noise


@torch.no_grad()
def sample_from_noisy(model, x_t, t_start, num_steps=50, cfg_scale=1.0, cond=None, sigma_min=1e-5):
    """
    从加噪的latent开始采样重建
    
    Args:
        model: Flow matching denoiser模型
        x_t: 加噪后的latent (B, C, D, H, W)
        t_start: 起始时间步 [0-1]
        num_steps: 采样步数
        cfg_scale: CFG引导强度
        cond: 条件嵌入 (可选)
        sigma_min: 最小噪声水平
    
    Returns:
        重建的latent tensor
    """
    device = x_t.device
    shape = x_t.shape
    x = x_t.clone()
    
    # 时间步从t_start到0
    t_seq = np.linspace(t_start, 0, num_steps + 1)
    
    for i in range(num_steps):
        t = t_seq[i]
        t_prev = t_seq[i + 1]
        
        # 时间步乘以1000后传给模型
        t_batch = torch.tensor([1000 * t] * shape[0], device=device, dtype=torch.float32)
        
        # 计算速度场
        if cond is not None and cfg_scale > 1.0:
            v_cond = model(x, t_batch, cond)
            v_uncond = model(x, t_batch, None)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, cond)
        
        # Euler step
        x = x - (t - t_prev) * v
    
    return x


def recon_from_latent():
    """
    从指定latent扩散并重建
    
    使用方法:
        # 从latent文件重建（测试模型质量）
        python infer.py recon_latent \
            --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
            --ckpt_flow outputs/flow_test1/denoiser_ema0.9999_step0100000.pt \
            --input output/latent_sample.pt \
            --output output/recon \
            --t_noise 0.5 \
            --steps 50
        
        # 同时解码为3D体素
        python infer.py recon_latent \
            --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
            --ckpt_flow outputs/flow_test1/denoiser_ema0.9999_step0100000.pt \
            --vae_config configs/vae/ss_vae_conv3d_16l8_fp16.json \
            --ckpt_decoder ckpt/decoder_ema0.9999_step0040000.pt \
            --input output/latent_sample.pt \
            --output output/recon \
            --t_noise 0.5 \
            --steps 50 \
            --decode
            
        # 从npz格式的latent重建
        python infer.py recon_latent \
            --config configs/generation/ss_flow_custom_uncond_dit_B_16l8_fp16.json \
            --ckpt_flow outputs/flow_test1/denoiser_ema0.9999_step0100000.pt \
            --input ~/dataset/ss_latents/sample.npz \
            --output output/recon \
            --t_noise 1.0 \
            --steps 50
    """
    import os
    
    parser = argparse.ArgumentParser(description='从latent扩散并重建')
    parser.add_argument('--config', type=str, required=True, help='Flow模型配置文件')
    parser.add_argument('--ckpt_flow', type=str, required=True, help='Flow模型检查点')
    parser.add_argument('--vae_config', type=str, default=None, help='VAE配置文件 (解码时需要)')
    parser.add_argument('--ckpt_decoder', type=str, default=None, help='VAE Decoder检查点 (解码时需要)')
    parser.add_argument('--input', type=str, required=True, help='输入latent文件 (.pt或.npz)')
    parser.add_argument('--output', type=str, default='output/recon', help='输出目录')
    parser.add_argument('--t_noise', type=float, default=1.0, help='加噪时间步 [0-1]，1.0表示从纯噪声开始')
    parser.add_argument('--steps', type=int, default=50, help='采样步数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--decode', action='store_true', help='是否解码为3D体素')
    parser.add_argument('--save_noisy', action='store_true', help='是否保存加噪后的latent')
    opt = parser.parse_args()
    
    normalize_mean = 0
    normalize_std = 2.26
    # 设置随机种子
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        print(f"随机种子: {opt.seed}")
    
    # 创建输出目录
    os.makedirs(opt.output, exist_ok=True)
    
    # 加载输入latent
    print(f"加载latent: {opt.input}")
    if opt.input.endswith('.npz'):
        data = np.load(opt.input)
        latent = torch.from_numpy(data['mean']).unsqueeze(0).cuda()  # (1, C, D, H, W)
    else:
        latent = torch.load(opt.input, map_location='cuda', weights_only=True)
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)  # 添加batch维度
    
    print(f"Latent shape: {latent.shape}")
    
    # 加载Flow模型
    print(f"加载Flow配置: {opt.config}")
    cfg = edict(json.load(open(opt.config)))
    
    print(f"加载Flow模型: {opt.ckpt_flow}")
    flow_model = load_flow_model(cfg.models.denoiser, opt.ckpt_flow)
    
    # 加载VAE Decoder (如果需要解码)
    decoder = None
    if opt.decode:
        if opt.vae_config is None or opt.ckpt_decoder is None:
            print("错误: 解码模式需要 --vae_config 和 --ckpt_decoder")
            return
        print(f"加载VAE配置: {opt.vae_config}")
        vae_cfg = edict(json.load(open(opt.vae_config)))
        print(f"加载Decoder: {opt.ckpt_decoder}")
        decoder = load_model(vae_cfg.models.decoder, opt.ckpt_decoder, 'decoder')
    
    # 扩散（加噪）
    latent_norm = (latent - normalize_mean) / normalize_std
    print(f"扩散时间步: t={opt.t_noise}")
    x_noisy, noise = diffuse_latent(latent_norm, opt.t_noise)
    
    if opt.save_noisy:
        noisy_path = os.path.join(opt.output, 'noisy_latent.pt')
        torch.save(x_noisy.cpu(), noisy_path)
        print(f"保存加噪latent: {noisy_path}")
    
    # 重建
    print(f"采样重建: {opt.steps}步")
    recon_latent = sample_from_noisy(flow_model, x_noisy, opt.t_noise, num_steps=opt.steps)
    recon_latent = recon_latent * normalize_std + normalize_mean  # 反归一化
    
    # 保存重建的latent
    recon_latent_path = os.path.join(opt.output, 'recon_latent.pt')
    torch.save(recon_latent.cpu(), recon_latent_path)
    print(f"保存重建latent: {recon_latent_path}")
    
    # 计算latent空间的重建误差
    mse = torch.nn.functional.mse_loss(recon_latent, latent_norm).item()
    print(f"Latent MSE: {mse:.6f}")
    
    # 解码
    if opt.decode and decoder is not None:
        print("解码为3D体素...")
        
        # 解码原始latent
        orig_voxel = decoder(latent)
        orig_path = os.path.join(opt.output, 'original.pt')
        torch.save(orig_voxel.cpu(), orig_path)
        print(f"保存原始体素: {orig_path}")
        
        # 解码重建latent
        recon_voxel = decoder(recon_latent)
        recon_path = os.path.join(opt.output, 'reconstructed.pt')
        torch.save(recon_voxel.cpu(), recon_path)
        print(f"保存重建体素: {recon_path}")
        
        # 计算体素空间的mIoU
        miou = mIoU(recon_voxel, orig_voxel)
        miou_occ = mIoU_occ(recon_voxel[:, :1], orig_voxel[:, :1])
        print(f"Voxel mIoU: {miou:.4f}")
        print(f"Voxel mIoU_occ: {miou_occ:.4f}")
    
    print("\n重建完成!")


def img_cond_generate():
    """
    图像条件生成3D样本（伪颜色条件）
    
    使用 DINOv2 编码输入图像，然后用 flow matching 模型进行条件生成。
    支持从伪颜色渲染图或任意图像进行条件生成。
    
    使用方法:
        # 从单张图像生成latent
        python infer.py img_cond_gen \
            --config configs/generation/ss_flow_custom_pseudocolor_img_dit_B_16l8_fp16.json \
            --ckpt_flow ckpt/denoiser_ema0.9999_step0100000.pt \
            --image input.png \
            --output output/generated \
            --steps 50
        
        # 生成并解码为3D体素
        python infer.py img_cond_gen \
            --config configs/generation/ss_flow_custom_pseudocolor_img_dit_B_16l8_fp16.json \
            --ckpt_flow ckpt/denoiser_ema0.9999_step0100000.pt \
            --image input.png \
            --vae_config configs/vae/ss_vae_conv3d_16l8_fp16.json \
            --ckpt_decoder ckpt/decoder_ema0.9999_step0040000.pt \
            --output output/generated \
            --steps 50 \
            --decode \
            --cfg_scale 3.0
        
        # 批量处理目录中的图像
        python infer.py img_cond_gen \
            --config configs/generation/ss_flow_custom_pseudocolor_img_dit_B_16l8_fp16.json \
            --ckpt_flow ckpt/denoiser_ema0.9999_step0100000.pt \
            --image_dir renders_cond/ \
            --output output/generated \
            --steps 50
    """
    import os
    from PIL import Image
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='图像条件生成3D样本')
    parser.add_argument('--config', type=str, required=True, help='Flow模型配置文件')
    parser.add_argument('--ckpt_flow', type=str, required=True, help='Flow模型检查点')
    parser.add_argument('--image', type=str, default=None, help='输入图像路径')
    parser.add_argument('--image_dir', type=str, default=None, help='输入图像目录（批量处理）')
    parser.add_argument('--vae_config', type=str, default=None, help='VAE配置文件（解码时需要）')
    parser.add_argument('--ckpt_decoder', type=str, default=None, help='VAE Decoder检查点（解码时需要）')
    parser.add_argument('--output', type=str, default='output/img_cond_gen', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=1, help='每张图像生成的样本数')
    parser.add_argument('--steps', type=int, default=50, help='采样步数')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='Classifier-free guidance强度')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--decode', action='store_true', help='是否解码为3D体素')
    parser.add_argument('--save_latent', action='store_true', help='是否保存latent')
    parser.add_argument('--image_cond_model', type=str, default=None,
                        help='DINOv2模型名（默认从配置读取）')
    parser.add_argument('--image_size', type=int, default=518, help='图像输入尺寸')
    opt = parser.parse_args()
    
    if opt.image is None and opt.image_dir is None:
        print("错误: 需要 --image 或 --image_dir")
        return
    
    # 设置随机种子
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        print(f"随机种子: {opt.seed}")
    
    os.makedirs(opt.output, exist_ok=True)
    
    # ---------- 加载配置 ----------
    print(f"加载Flow配置: {opt.config}")
    cfg = edict(json.load(open(opt.config)))
    
    # 读取 normalization 参数
    normalize_mean = 0.0
    normalize_std = 2.26
    if 'dataset' in cfg and 'args' in cfg.dataset:
        ds_args = cfg.dataset.args
        if 'normalization' in ds_args and ds_args.normalization is not None:
            if 'mean' in ds_args.normalization:
                normalize_mean = ds_args.normalization['mean']
                if isinstance(normalize_mean, list):
                    normalize_mean = normalize_mean[0]
            if 'std' in ds_args.normalization:
                normalize_std = ds_args.normalization['std']
                if isinstance(normalize_std, list):
                    normalize_std = normalize_std[0]
    print(f"归一化参数: mean={normalize_mean}, std={normalize_std}")
    
    # ---------- 加载 DINOv2 ----------
    image_cond_model_name = opt.image_cond_model
    if image_cond_model_name is None:
        image_cond_model_name = cfg.get('trainer', {}).get('args', {}).get(
            'image_cond_model', 'dinov2_vitl14_reg')
    
    print(f"加载DINOv2: {image_cond_model_name}")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', image_cond_model_name, pretrained=True)
    dinov2_model.eval().cuda()
    dinov2_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # ---------- 加载 Flow 模型 ----------
    print(f"加载Flow模型: {opt.ckpt_flow}")
    flow_model = load_flow_model(cfg.models.denoiser, opt.ckpt_flow)
    
    resolution = cfg.models.denoiser.args.resolution
    latent_channels = cfg.models.denoiser.args.in_channels
    print(f"Latent shape: ({latent_channels}, {resolution}, {resolution}, {resolution})")
    
    # ---------- 加载 VAE Decoder ----------
    decoder = None
    if opt.decode:
        if opt.vae_config is None or opt.ckpt_decoder is None:
            print("错误: 解码模式需要 --vae_config 和 --ckpt_decoder")
            return
        vae_cfg = edict(json.load(open(opt.vae_config)))
        print(f"加载Decoder: {opt.ckpt_decoder}")
        decoder = load_model(vae_cfg.models.decoder, opt.ckpt_decoder, 'decoder')
    
    # ---------- 准备图像列表 ----------
    image_paths = []
    if opt.image is not None:
        image_paths = [opt.image]
    else:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        for f in sorted(os.listdir(opt.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(opt.image_dir, f))
    
    print(f"待处理图像: {len(image_paths)} 张")
    
    # ---------- 图像预处理函数 ----------
    def preprocess_image(img_path):
        """加载并预处理图像为 DINOv2 输入"""
        img = Image.open(img_path)
        
        # alpha 裁剪
        if img.mode == 'RGBA':
            alpha = np.array(img.getchannel(3))
            bbox_yx = np.array(alpha > 0).nonzero()
            if len(bbox_yx[0]) > 0:
                y_min, y_max = bbox_yx[0].min(), bbox_yx[0].max()
                x_min, x_max = bbox_yx[1].min(), bbox_yx[1].max()
                ctr = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                hs = max(x_max - x_min, y_max - y_min) / 2 * 1.2
                crop = (int(ctr[0] - hs), int(ctr[1] - hs),
                        int(ctr[0] + hs), int(ctr[1] + hs))
                img = img.crop(crop)
        
        img = img.resize((opt.image_size, opt.image_size), Image.Resampling.LANCZOS)
        
        if img.mode == 'RGBA':
            alpha_ch = np.array(img.getchannel(3)).astype(np.float32) / 255.0
            img = img.convert('RGB')
        else:
            alpha_ch = np.ones((opt.image_size, opt.image_size), dtype=np.float32)
            img = img.convert('RGB')
        
        t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        a = torch.tensor(alpha_ch)
        t = t * a.unsqueeze(0) + (1.0 - a.unsqueeze(0))  # 白色背景
        return t
    
    # ---------- DINOv2 编码函数 ----------
    @torch.no_grad()
    def encode_image(image_tensor):
        """编码图像为 DINOv2 patch tokens"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = dinov2_transform(image_tensor).cuda()
        features = dinov2_model(image_tensor, is_training=True)['x_prenorm']
        return F.layer_norm(features, features.shape[-1:])
    
    # ---------- CFG 采样函数 ----------
    @torch.no_grad()
    def sample_flow_cfg(model, shape, cond, neg_cond, num_steps, cfg_strength):
        """使用 CFG 的 flow matching 采样"""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device, dtype=torch.float32)
        t_seq = np.linspace(1, 0, num_steps + 1)
        
        for i in range(num_steps):
            t = t_seq[i]
            t_prev = t_seq[i + 1]
            t_batch = torch.tensor([1000 * t] * shape[0], device=device, dtype=torch.float32)
            
            # 正向 + 负向预测
            cond_expanded = cond.expand(shape[0], -1, -1) if cond.shape[0] == 1 else cond
            neg_expanded = neg_cond.expand(shape[0], -1, -1) if neg_cond.shape[0] == 1 else neg_cond
            
            v_cond = model(x, t_batch, cond_expanded)
            v_uncond = model(x, t_batch, neg_expanded)
            v = v_uncond + cfg_strength * (v_cond - v_uncond)
            
            x = x - (t - t_prev) * v
        
        return x
    
    # ---------- 主循环 ----------
    total_generated = 0
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="处理图像")):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n处理: {img_path}")
        
        # 预处理 + 编码
        img_t = preprocess_image(img_path)
        cond = encode_image(img_t)           # [1, N_patches, D_feat]
        neg_cond = torch.zeros_like(cond)    # 无条件 = 全零
        
        for sample_idx in range(opt.num_samples):
            shape = (1, latent_channels, resolution, resolution, resolution)
            
            # CFG 采样
            latent = sample_flow_cfg(
                flow_model, shape, cond, neg_cond,
                num_steps=opt.steps, cfg_strength=opt.cfg_scale,
            )
            
            # 反归一化
            latent = latent * normalize_std + normalize_mean
            
            prefix = f"{img_name}_s{sample_idx:02d}"
            
            # 保存 latent
            if opt.save_latent or not opt.decode:
                latent_path = os.path.join(opt.output, f'{prefix}_latent.pt')
                torch.save(latent.cpu(), latent_path)
            
            # 解码
            if opt.decode and decoder is not None:
                decoded = decoder(latent)
                decoded_path = os.path.join(opt.output, f'{prefix}.pt')
                torch.save(decoded.cpu(), decoded_path)
            
            total_generated += 1
    
    print(f"\n生成完成! 共 {total_generated} 个样本 → {opt.output}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'gen_latent':
        # 移除'gen_latent'参数，让argparse正常解析剩余参数
        sys.argv.pop(1)
        gen_latent_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == 'uncond_gen':
        sys.argv.pop(1)
        uncond_generate()
    elif len(sys.argv) > 1 and sys.argv[1] == 'recon_latent':
        sys.argv.pop(1)
        recon_from_latent()
    elif len(sys.argv) > 1 and sys.argv[1] == 'img_cond_gen':
        sys.argv.pop(1)
        img_cond_generate()
    else:
        main()
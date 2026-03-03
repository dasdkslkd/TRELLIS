import matplotlib.pyplot as plt, re
import numpy as np

def visualize_flow_matching_loss(file_path, window_size=50):
    """
    可视化 Flow Matching 训练的 loss 曲线。
    
    包括：
    1. 总体 MSE loss 曲线（含平滑）
    2. 各时间区间 (bin_0 ~ bin_9) 的 MSE 热力图
    3. Gradient norm 曲线
    
    Args:
        file_path: 日志文件路径
        window_size: 滑动平均窗口大小
    """
    steps, losses, grad_norms = [], [], []
    bin_losses = {i: [] for i in range(10)}  # bin_0 ~ bin_9
    bin_steps = {i: [] for i in range(10)}
    
    for line in open(file_path):
        if "'loss':" not in line:
            continue
        
        # 提取 step
        step_match = re.search(r"^(\d+):", line)
        if step_match:
            step = int(step_match.group(1))
        else:
            continue
        
        steps.append(step)
        
        # 提取总体 loss
        loss_match = re.search(r"'loss': np\.float\d+\(([\d.eE+-]+)\)", line)
        if loss_match:
            losses.append(float(loss_match.group(1)))
        
        # 提取 grad_norm
        grad_match = re.search(r"'grad_norm': np\.float\d+\(([\d.eE+-]+)\)", line)
        if grad_match:
            grad_norms.append(float(grad_match.group(1)))
        
        # 提取各 bin 的 mse
        for i in range(10):
            bin_match = re.search(rf"'bin_{i}': \{{'mse': np\.float\d+\(([\d.eE+-]+)\)\}}", line)
            if bin_match:
                bin_losses[i].append(float(bin_match.group(1)))
                bin_steps[i].append(step)
    np.savetxt('output/loss_log.txt', np.array(losses), fmt='%f')
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Flow Matching Training Visualization', fontsize=14)
    
    # 1. 总体 Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, alpha=0.3, label='Raw Loss', color='blue')
    if len(losses) >= window_size:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(steps[window_size-1:], smoothed, label=f'Smoothed (w={window_size})', color='red', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Overall Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log Loss 曲线
    ax2 = axes[0, 1]
    log_losses = np.log(np.array(losses) + 1e-8)
    ax2.plot(steps, log_losses, alpha=0.3, label='Raw Log Loss', color='green')
    if len(log_losses) >= window_size:
        smoothed_log = np.convolve(log_losses, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(steps[window_size-1:], smoothed_log, label=f'Smoothed (w={window_size})', color='darkgreen', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Log MSE Loss')
    ax2.set_title('Log Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 各时间区间 bin loss 箱线图
    ax3 = axes[1, 0]
    bin_data = [bin_losses[i] for i in range(10)]
    # 过滤空列表
    valid_bins = [(i, data) for i, data in enumerate(bin_data) if len(data) > 0]
    if valid_bins:
        positions = [v[0] for v in valid_bins]
        data = [v[1] for v in valid_bins]
        bp = ax3.boxplot(data, positions=positions, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        for patch, pos in zip(bp['boxes'], positions):
            patch.set_facecolor(colors[pos])
    ax3.set_xlabel('Time Bin (0=t∈[0,0.1), 9=t∈[0.9,1])')
    ax3.set_ylabel('MSE Loss')
    ax3.set_title('Loss Distribution by Time Bin')
    ax3.set_xticks(range(10))
    ax3.set_xticklabels([f'{i}' for i in range(10)])
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradient Norm 曲线
    ax4 = axes[1, 1]
    if grad_norms:
        ax4.plot(steps[:len(grad_norms)], grad_norms, alpha=0.3, label='Raw Grad Norm', color='orange')
        if len(grad_norms) >= window_size:
            smoothed_grad = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(steps[window_size-1:len(grad_norms)], smoothed_grad, 
                     label=f'Smoothed (w={window_size})', color='darkorange', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n=== Flow Matching Training Statistics ===")
    print(f"Total steps: {len(steps)}")
    print(f"Loss - Mean: {np.mean(losses):.4f}, Std: {np.std(losses):.4f}")
    print(f"Loss - Min: {np.min(losses):.4f}, Max: {np.max(losses):.4f}")
    if grad_norms:
        print(f"Grad Norm - Mean: {np.mean(grad_norms):.4f}, Max: {np.max(grad_norms):.4f}")
    print(f"\nPer-bin MSE statistics:")
    for i in range(10):
        if bin_losses[i]:
            print(f"  Bin {i} (t∈[{i/10:.1f},{(i+1)/10:.1f})): "
                  f"Mean={np.mean(bin_losses[i]):.4f}, Count={len(bin_losses[i])}")


def log_type1(file_path):
    L, S = {}, []
    for l in open(file_path):
        if "'loss':" in l:
            # 提取step和loss中的数值，忽略np类型
            S.append(float(re.search(r"'step': ([\d.eE+-]+)", l).group(1)))
            for k, v in re.findall(r"'(\w+)': np\.float\d+\(([\d.eE+-]+)\)", l):
                if k == 'kl': continue  # 忽略kl项
                L.setdefault(k, []).append(float(v))

    [plt.plot(np.log(v), label=k) for k, v in L.items() if k == 'dice' or k == 'ce' or k == 'mse' or k == 'loss']
    # plt.ylim(0, 0.5)
    plt.legend(), plt.xlabel('Step'), plt.ylabel('Log Loss'), plt.show()

def log_type2(file_path):
    t_m, t_mo, v_m, v_mo = [], [], [], []
    for l in open(file_path):
        if 'Validation mIoU_occ:' in l:  # 先检查更具体的
            v_mo.append(float(l.split()[-1]))
        elif 'Validation mIoU:' in l:
            v_m.append(float(l.split()[-1]))
        elif 'Train mIoU_occ:' in l:
            t_mo.append(float(l.split()[-1]))
        elif 'Train mIoU:' in l:
            t_m.append(float(l.split()[-1]))

    plt.plot(t_mo, label='Train mIoU occ')
    plt.plot(v_mo, label='Val mIoU occ')
    plt.plot(t_m, label='Train mIoU')
    plt.plot(v_m, label='Val mIoU')
    # plt.ylim(0, 1)
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('mIoU'), plt.show()

# log_type1('output/log (1).txt')
# log_type2('output/slurm-7583.out')
visualize_flow_matching_loss('output/log (4).txt', window_size=50)
#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render representative cases for a joint sparse-structure VAE run.')
    parser.add_argument('--config', required=True, help='Path to experiment config JSON.')
    parser.add_argument('--output-dir', required=True, help='Training output directory containing checkpoints and best_metric.json.')
    parser.add_argument('--val-data-dir', required=True, help='Validation dataset root.')
    parser.add_argument('--title', default='Joint VAE Case Visualization', help='Title shown in figures and summaries.')
    parser.add_argument('--case-name', default='joint_case_viz', help='Short folder name under output root.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=0, help='Cap number of validation samples. 0 means all.')
    parser.add_argument('--k-best', type=int, default=3)
    parser.add_argument('--k-middle', type=int, default=3)
    parser.add_argument('--k-worst', type=int, default=3)
    parser.add_argument(
        '--rank-by',
        default='miou_occ',
        choices=['miou_occ', 'miou', 'occ_precision', 'occ_recall', 'mse_stage2'],
        help='Metric used to rank best / middle / worst cases.',
    )
    parser.add_argument('--max-points', type=int, default=12000)
    parser.add_argument(
        '--checkpoint-source',
        default='plain',
        choices=['plain', 'ema', 'auto'],
        help='Checkpoint weights to use. plain matches trainer validate() behavior.',
    )
    parser.add_argument(
        '--output-root',
        default='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/joint_case_visualizations',
        help='Directory for rendered images and summaries.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _extract_available_steps(ckpt_dir: Path, prefix: str) -> List[int]:
    pattern = re.compile(rf'^{re.escape(prefix)}(?:_ema0\.9999)?_step(\d+)\.pt$')
    steps: List[int] = []
    for file_path in ckpt_dir.iterdir():
        match = pattern.match(file_path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def resolve_checkpoint_paths(output_dir: Path, checkpoint_source: str) -> Dict[str, Path]:
    ckpt_dir = output_dir / 'ckpts'
    best_metric_path = output_dir / 'best_metric.json'
    if not best_metric_path.exists():
        best_metric_path = ckpt_dir / 'best' / 'best_metric.json'

    if best_metric_path.exists():
        best_step = int(load_json(best_metric_path)['step'])
    else:
        steps = _extract_available_steps(ckpt_dir, 'encoder')
        if not steps:
            raise FileNotFoundError(f'No encoder checkpoints found under {ckpt_dir}')
        best_step = steps[-1]

    resolved: Dict[str, Path] = {}
    for name in ('encoder', 'decoder_stage1', 'decoder_stage2'):
        ema_path = ckpt_dir / f'{name}_ema0.9999_step{best_step:07d}.pt'
        plain_path = ckpt_dir / f'{name}_step{best_step:07d}.pt'
        if checkpoint_source == 'plain':
            candidates = [plain_path, ema_path]
        elif checkpoint_source == 'ema':
            candidates = [ema_path, plain_path]
        else:
            candidates = [ema_path, plain_path]
        chosen = next((candidate for candidate in candidates if candidate.exists()), None)
        if chosen is None:
            raise FileNotFoundError(f'Cannot resolve checkpoint for {name} at step {best_step} in {ckpt_dir}')
        resolved[name] = chosen
    resolved['step'] = Path(str(best_step))
    return resolved


def build_models(config: dict, checkpoint_paths: Dict[str, Path], device: torch.device) -> Dict[str, torch.nn.Module]:
    model_dict: Dict[str, torch.nn.Module] = {}
    for name in ('encoder', 'decoder_stage1', 'decoder_stage2'):
        model_cfg = config['models'][name]
        model = getattr(models, model_cfg['name'])(**model_cfg['args']).to(device)
        state_dict = torch.load(checkpoint_paths[name], map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model_dict[name] = model
    return model_dict


def load_sparse_tensor(file_path: Path, resolution: int) -> torch.Tensor:
    data = torch.load(file_path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        if 'coords' in data and 'feats' in data:
            coords = data['coords']
            feats = data['feats']
            dense = torch.zeros(feats.shape[1], resolution, resolution, resolution, dtype=feats.dtype)
            coords_int = coords.long()
            valid_mask = ((coords_int >= 0).all(dim=1) & (coords_int < resolution).all(dim=1))
            coords_int = coords_int[valid_mask]
            feats = feats[valid_mask]
            if coords_int.numel() > 0:
                dense[:, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = feats.t()
            return dense.float()
        if 'dense' in data:
            data = data['dense']
        elif 'tensor' in data:
            data = data['tensor']

    if isinstance(data, torch.Tensor) and data.layout == torch.sparse_coo:
        data = data.coalesce().to_dense().permute(3, 0, 1, 2)

    if not isinstance(data, torch.Tensor):
        raise TypeError(f'Unsupported tensor format from {file_path}: {type(data)}')

    if data.dim() != 4:
        raise ValueError(f'Expected 4D tensor from {file_path}, got {tuple(data.shape)}')

    if data.shape[0] in (8, 14, 80):
        dense = data
    elif data.shape[-1] in (8, 14, 80):
        dense = data.permute(3, 0, 1, 2)
    else:
        raise ValueError(f'Unrecognized channel layout from {file_path}: {tuple(data.shape)}')

    return dense.float()


def activate_stage1(logits: torch.Tensor, stage1_channels: int) -> torch.Tensor:
    occ = torch.sigmoid(logits[:, 0:1])
    cls_probs = torch.softmax(logits[:, 1:stage1_channels], dim=1)
    return torch.cat([occ, cls_probs], dim=1)


def discretize_stage1(activated: torch.Tensor, stage1_channels: int) -> torch.Tensor:
    recon = torch.zeros_like(activated)
    recon[:, 0:1] = (activated[:, 0:1] > 0.5).float()
    cls_idx = activated[:, 1:stage1_channels].argmax(dim=1, keepdim=True)
    recon[:, 1:stage1_channels].scatter_(1, cls_idx, 1.0)
    return recon


def compute_metrics(
    pred_stage1_discrete: torch.Tensor,
    pred_stage1_activated: torch.Tensor,
    pred_stage2: torch.Tensor,
    target: torch.Tensor,
    stage1_channels: int,
    stage2_channels: int,
) -> Dict[str, float]:
    occ = target[:, 0] > 0.5
    if occ.sum() == 0:
        miou = 0.0
    else:
        match = pred_stage1_discrete[:, 1:stage1_channels].argmax(1) == target[:, 1:stage1_channels].argmax(1)
        miou = ((match & occ).sum().float() / occ.sum().float().clamp(min=1)).item()

    pred_occ = pred_stage1_activated[:, :1] > 0.5
    gt_occ = target[:, :1] > 0.5
    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()
    pred_pos = pred_occ.sum().float()
    gt_pos = gt_occ.sum().float()

    occ_mask = gt_occ.float()
    target_stage2 = target[:, stage1_channels:stage1_channels + stage2_channels]
    if occ_mask.sum() > 0:
        mse_stage2 = (((pred_stage2 - target_stage2) ** 2) * occ_mask).sum() / (occ_mask.sum() * stage2_channels)
        mse_stage2_value = float(mse_stage2.item())
    else:
        mse_stage2_value = 0.0

    return {
        'miou': float(miou),
        'miou_occ': float((inter / (union + 1e-8)).item()),
        'occ_precision': float((inter / pred_pos.clamp(min=1)).item()),
        'occ_recall': float((inter / gt_pos.clamp(min=1)).item()),
        'pred_occ_ratio': float(pred_pos.div(pred_occ.numel()).item()),
        'gt_occ_ratio': float(gt_pos.div(gt_occ.numel()).item()),
        'mse_stage2': mse_stage2_value,
    }


def iter_validation_items(val_root: Path) -> Iterable[Tuple[str, Path]]:
    metadata = pd.read_csv(val_root / 'metadata.csv')
    for sha256 in metadata['sha256'].astype(str).tolist():
        yield sha256, val_root / 'data' / f'{sha256}.pt'


def sample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, num=max_points, dtype=int)


def voxel_center_coords(mask: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    chosen = sample_indices(len(idx), max_points)
    idx = idx[chosen].astype(np.float32)
    res = mask.shape[0]
    coords = (idx + 0.5) * (2.0 / res) - 1.0
    return coords, chosen


def stylize_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=315)
    ax.set_xlabel('X', labelpad=-6)
    ax.set_ylabel('Y', labelpad=-6)
    ax.set_zlabel('Z', labelpad=-6)
    ax.grid(False)
    ax.set_facecolor('#fafafa')


def render_type_panel(ax, tensor: torch.Tensor, title: str, max_points: int) -> None:
    tensor_np = tensor.detach().cpu().numpy()
    mask = tensor_np[0] > 0.5
    if not mask.any():
        stylize_axis(ax, f'{title}\n(empty)')
        return

    coords, chosen = voxel_center_coords(mask, max_points)
    classes = tensor_np[1:8].argmax(axis=0)[mask][chosen]
    colors = plt.cm.tab10(classes % 10)
    size = 22 if len(coords) < 4000 else 12
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=size, alpha=0.82, edgecolors='none')
    stylize_axis(ax, title)


def render_diff_panel(ax, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor, max_points: int) -> None:
    gt_occ = gt_tensor[0].detach().cpu().numpy() > 0.5
    pred_occ = pred_tensor[0].detach().cpu().numpy() > 0.5

    tp = pred_occ & gt_occ
    fp = pred_occ & ~gt_occ
    fn = gt_occ & ~pred_occ

    legend = []
    for mask, color, label, alpha in (
        (tp, '#6dbd45', 'TP', 0.35),
        (fp, '#ff8c42', 'FP', 0.85),
        (fn, '#3b82f6', 'FN', 0.85),
    ):
        coords, _ = voxel_center_coords(mask, max_points)
        if len(coords) > 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, s=14, alpha=alpha, edgecolors='none')
        legend.append(f'{label}:{int(mask.sum())}')

    stylize_axis(ax, 'Occupancy Diff')
    ax.text2D(0.02, 0.98, '  '.join(legend), transform=ax.transAxes, va='top', fontsize=9)


def stage2_magnitude(tensor: torch.Tensor, stage1_channels: int, stage2_channels: int) -> np.ndarray:
    feat = tensor[stage1_channels:stage1_channels + stage2_channels].detach().cpu().numpy()
    return np.linalg.norm(feat, axis=0)


def render_scalar_panel(ax, mask: np.ndarray, scalar: np.ndarray, title: str, max_points: int, cmap: str, vmin: float, vmax: float) -> None:
    if not mask.any():
        stylize_axis(ax, f'{title}\n(empty)')
        return

    coords, chosen = voxel_center_coords(mask, max_points)
    values = scalar[mask][chosen]
    denom = max(vmax - vmin, 1e-8)
    normalized = np.clip((values - vmin) / denom, 0.0, 1.0)
    colors = plt.get_cmap(cmap)(normalized)
    size = 20 if len(coords) < 4000 else 12
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=size, alpha=0.82, edgecolors='none')
    stylize_axis(ax, title)


def render_case_figure(
    title: str,
    case_kind: str,
    rank_index: int,
    sample_name: str,
    metrics: Dict[str, float],
    gt_tensor: torch.Tensor,
    pred_stage1: torch.Tensor,
    pred_full: torch.Tensor,
    stage1_channels: int,
    stage2_channels: int,
    output_path: Path,
    max_points: int,
) -> None:
    fig = plt.figure(figsize=(17.5, 9.8), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

    render_type_panel(axes[0], gt_tensor[:stage1_channels], 'GT Stage1', max_points)
    render_type_panel(axes[1], pred_stage1[:stage1_channels], 'Pred Stage1', max_points)
    render_diff_panel(axes[2], gt_tensor[:stage1_channels], pred_stage1[:stage1_channels], max_points)

    gt_occ = gt_tensor[0].detach().cpu().numpy() > 0.5
    pred_occ = pred_stage1[0].detach().cpu().numpy() > 0.5
    err_occ = gt_occ | pred_occ
    gt_mag = stage2_magnitude(gt_tensor, stage1_channels, stage2_channels)
    pred_mag = stage2_magnitude(pred_full, stage1_channels, stage2_channels)
    err_mag = np.linalg.norm(
        (pred_full[stage1_channels:stage1_channels + stage2_channels] - gt_tensor[stage1_channels:stage1_channels + stage2_channels]).detach().cpu().numpy(),
        axis=0,
    )

    common_vmin = float(min(gt_mag[gt_occ].min() if gt_occ.any() else 0.0, pred_mag[pred_occ].min() if pred_occ.any() else 0.0))
    common_vmax = float(max(gt_mag[gt_occ].max() if gt_occ.any() else 1.0, pred_mag[pred_occ].max() if pred_occ.any() else 1.0))
    err_vmax = float(err_mag[err_occ].max()) if err_occ.any() else 1.0

    render_scalar_panel(axes[3], gt_occ, gt_mag, 'GT Stage2 |mag|', max_points, 'viridis', common_vmin, common_vmax)
    render_scalar_panel(axes[4], pred_occ, pred_mag, 'Pred Stage2 |mag|', max_points, 'viridis', common_vmin, common_vmax)
    render_scalar_panel(axes[5], err_occ, err_mag, 'Stage2 Error |pred-gt|', max_points, 'magma', 0.0, err_vmax)

    score_text = ', '.join(
        [
            f"miou_occ={metrics['miou_occ']:.4f}",
            f"miou={metrics['miou']:.4f}",
            f"precision={metrics['occ_precision']:.4f}",
            f"recall={metrics['occ_recall']:.4f}",
            f"mse_s2={metrics['mse_stage2']:.6f}",
        ]
    )
    fig.suptitle(f'{title} | {case_kind} #{rank_index + 1} | {sample_name}\n{score_text}', fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_root = Path(args.output_root) / args.case_name
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_json(Path(args.config))
    trainer_args = config['trainer']['args']
    stage1_channels = int(trainer_args['stage1_channels'])
    stage2_channels = int(trainer_args['stage2_channels'])
    resolution = int(config['dataset']['args'].get('resolution', 64))

    checkpoint_paths = resolve_checkpoint_paths(Path(args.output_dir), args.checkpoint_source)
    models_dict = build_models(config, checkpoint_paths, device)

    records: List[Dict[str, object]] = []
    val_root = Path(args.val_data_dir)

    with torch.inference_mode():
        for sample_index, (sha256, tensor_path) in enumerate(iter_validation_items(val_root)):
            if args.max_samples > 0 and sample_index >= args.max_samples:
                break
            ss = load_sparse_tensor(tensor_path, resolution).unsqueeze(0).to(device)
            z = models_dict['encoder'](ss, sample_posterior=False)
            stage1_logits = models_dict['decoder_stage1'](z)
            stage1_activated = activate_stage1(stage1_logits, stage1_channels)
            pred_stage1 = discretize_stage1(stage1_activated, stage1_channels)
            pred_stage2 = models_dict['decoder_stage2'](z, stage1_activated)
            metrics = compute_metrics(pred_stage1, stage1_activated, pred_stage2, ss, stage1_channels, stage2_channels)
            records.append({
                'sample_index': sample_index,
                'sha256': sha256,
                'tensor_path': str(tensor_path),
                **metrics,
            })

    if not records:
        raise RuntimeError('No validation records were evaluated.')

    reverse = args.rank_by != 'mse_stage2'
    records_sorted = sorted(records, key=lambda item: float(item[args.rank_by]), reverse=reverse)

    selected_best = records_sorted[:args.k_best]
    middle_start = max((len(records_sorted) - args.k_middle) // 2, 0)
    selected_middle = records_sorted[middle_start:middle_start + args.k_middle]
    selected_worst = records_sorted[-args.k_worst:]
    if reverse:
        selected_worst = list(reversed(selected_worst))

    pd.DataFrame(records_sorted).to_csv(output_root / 'metrics_ranked.csv', index=False)

    summary = {
        'title': args.title,
        'config': args.config,
        'output_dir': args.output_dir,
        'val_data_dir': args.val_data_dir,
        'checkpoint_source': args.checkpoint_source,
        'checkpoint_step': int(str(checkpoint_paths['step'])),
        'checkpoint_paths': {k: str(v) for k, v in checkpoint_paths.items() if k != 'step'},
        'device': str(device),
        'rank_by': args.rank_by,
        'num_evaluated_samples': len(records_sorted),
        'selected_best': selected_best,
        'selected_middle': selected_middle,
        'selected_worst': selected_worst,
    }
    with (output_root / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    selected_cases = [('best', selected_best), ('middle', selected_middle), ('worst', selected_worst)]
    with torch.inference_mode():
        for case_kind, cases in selected_cases:
            for rank_index, record in enumerate(cases):
                gt_tensor = load_sparse_tensor(Path(str(record['tensor_path'])), resolution)
                gt = gt_tensor.unsqueeze(0).to(device)
                z = models_dict['encoder'](gt, sample_posterior=False)
                stage1_logits = models_dict['decoder_stage1'](z)
                stage1_activated = activate_stage1(stage1_logits, stage1_channels)
                pred_stage1 = discretize_stage1(stage1_activated, stage1_channels)[0].cpu()
                pred_stage2 = models_dict['decoder_stage2'](z, stage1_activated)[0].cpu()
                pred_full = torch.cat([pred_stage1, pred_stage2], dim=0)

                sample_name = str(record['sha256'])
                output_path = output_root / case_kind / f'{rank_index + 1:02d}_{sample_name}.png'
                render_case_figure(
                    title=args.title,
                    case_kind=case_kind,
                    rank_index=rank_index,
                    sample_name=sample_name,
                    metrics={
                        'miou_occ': float(record['miou_occ']),
                        'miou': float(record['miou']),
                        'occ_precision': float(record['occ_precision']),
                        'occ_recall': float(record['occ_recall']),
                        'mse_stage2': float(record['mse_stage2']),
                    },
                    gt_tensor=gt_tensor,
                    pred_stage1=pred_stage1,
                    pred_full=pred_full,
                    stage1_channels=stage1_channels,
                    stage2_channels=stage2_channels,
                    output_path=output_path,
                    max_points=args.max_points,
                )

    print(f'[JointCaseViz] Outputs saved to {output_root}')


if __name__ == '__main__':
    main()
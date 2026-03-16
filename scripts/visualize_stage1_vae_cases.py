#!/usr/bin/env python3

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    title: str
    config_path: str
    val_data: str
    output_dir: str


DEFAULT_EXPERIMENTS: Dict[str, ExperimentSpec] = {
    'fusion360': ExperimentSpec(
        name='fusion360',
        title='Fusion360 Stage1 BCE v3',
        config_path='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/configs/vae/ss_stage1_fusion360_80ch_bce_v3.json',
        val_data='/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val',
        output_dir='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_stage1_bce_v3',
    ),
    'abc1m': ExperimentSpec(
        name='abc1m',
        title='ABC-1M Stage1 v2',
        config_path='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/configs/vae/ss_stage1_abc1m_80ch_v2.json',
        val_data='/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split/val',
        output_dir='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/abc1m_stage1_v2',
    ),
    'mixed': ExperimentSpec(
        name='mixed',
        title='Fusion360 + ABC-1M Mixed Stage1 v1',
        config_path='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/configs/vae/ss_stage1_mix_fusion360_abc1m_80ch_v1.json',
        val_data='/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val,/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split/val',
        output_dir='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/mix_fusion360_abc1m_stage1_v1',
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render representative Stage1 VAE cases for multiple training runs.')
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['fusion360', 'abc1m', 'mixed'],
        choices=sorted(DEFAULT_EXPERIMENTS.keys()),
        help='Experiments to evaluate.',
    )
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=0, help='Cap the number of validation samples per experiment. 0 means all.')
    parser.add_argument('--k-best', type=int, default=2, help='Number of best cases to render per experiment.')
    parser.add_argument('--k-middle', type=int, default=0, help='Number of middle-ranked cases to render per experiment.')
    parser.add_argument('--k-worst', type=int, default=2, help='Number of worst cases to render per experiment.')
    parser.add_argument(
        '--rank-by',
        default='miou_occ',
        choices=['miou_occ', 'miou', 'occ_precision', 'occ_recall'],
        help='Metric used to rank good and bad samples.',
    )
    parser.add_argument('--max-points', type=int, default=12000, help='Maximum number of occupied voxels rendered per panel.')
    parser.add_argument(
        '--checkpoint-source',
        default='plain',
        choices=['plain', 'ema', 'auto'],
        help='Checkpoint weights to use. plain matches trainer validate() behavior; auto keeps backward-compatible preference order.',
    )
    parser.add_argument(
        '--output-root',
        default='/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/stage1_case_visualizations',
        help='Directory for rendered images and summaries.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def split_roots(roots: str) -> List[Path]:
    return [Path(part.strip()) for part in roots.split(',') if part.strip()]


def load_run_config(spec: ExperimentSpec) -> dict:
    output_config = Path(spec.output_dir) / 'config.json'
    if output_config.exists():
        return load_json(output_config)
    return load_json(Path(spec.config_path))


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
    best_step = None
    if best_metric_path.exists():
        best_step = int(load_json(best_metric_path)['step'])
    else:
        best_dir_metric = ckpt_dir / 'best' / 'best_metric.json'
        if best_dir_metric.exists():
            best_step = int(load_json(best_dir_metric)['step'])

    if best_step is None:
        steps = _extract_available_steps(ckpt_dir, 'encoder')
        if not steps:
            raise FileNotFoundError(f'No encoder checkpoints found under {ckpt_dir}')
        best_step = steps[-1]

    resolved: Dict[str, Path] = {}
    for name in ('encoder', 'decoder_stage1'):
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
    for name in ('encoder', 'decoder_stage1'):
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
            return dense
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

    if dense.shape[1:] != (resolution, resolution, resolution):
        dense = torch.nn.functional.interpolate(
            dense.unsqueeze(0).float(),
            size=(resolution, resolution, resolution),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)
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


def compute_metrics(pred_discrete: torch.Tensor, pred_activated: torch.Tensor, target: torch.Tensor, stage1_channels: int) -> Dict[str, float]:
    occ = target[:, 0] > 0.5
    if occ.sum() == 0:
        miou = 0.0
    else:
        match = pred_discrete[:, 1:stage1_channels].argmax(1) == target[:, 1:stage1_channels].argmax(1)
        miou = ((match & occ).sum().float() / occ.sum().float().clamp(min=1)).item()

    pred_occ = pred_activated[:, :1] > 0.5
    gt_occ = target[:, :1] > 0.5
    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()
    pred_pos = pred_occ.sum().float()
    gt_pos = gt_occ.sum().float()

    return {
        'miou': float(miou),
        'miou_occ': float((inter / (union + 1e-8)).item()),
        'occ_precision': float((inter / pred_pos.clamp(min=1)).item()),
        'occ_recall': float((inter / gt_pos.clamp(min=1)).item()),
        'pred_occ_ratio': float(pred_pos.div(pred_occ.numel()).item()),
        'gt_occ_ratio': float(gt_pos.div(gt_occ.numel()).item()),
    }


def iter_validation_items(val_roots: Sequence[Path]) -> Iterable[Tuple[Path, str, Path]]:
    for root in val_roots:
        metadata_path = root / 'metadata.csv'
        metadata = pd.read_csv(metadata_path)
        for sha256 in metadata['sha256'].astype(str).tolist():
            yield root, sha256, root / 'data' / f'{sha256}.pt'


def source_tag(root: Path) -> str:
    if root.name in ('train', 'val') and root.parent.name:
        return root.parent.name
    return root.name


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
    ax.set_title(title, fontsize=12, pad=8)
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
    classes = tensor_np[1:8].argmax(axis=0)[mask]
    classes = classes[chosen]
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
    ax.text2D(0.02, 0.98, '  '.join(legend), transform=ax.transAxes, va='top', fontsize=10)


def render_case_figure(
    experiment: ExperimentSpec,
    case_kind: str,
    rank_index: int,
    sample_name: str,
    metrics: Dict[str, float],
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    render_type_panel(axes[0], gt_tensor[:8], 'Ground Truth', max_points)
    render_type_panel(axes[1], pred_tensor[:8], 'Stage1 Recon', max_points)
    render_diff_panel(axes[2], gt_tensor[:8], pred_tensor[:8], max_points)

    score_text = ', '.join(
        f'{key}={metrics[key]:.4f}'
        for key in ('miou_occ', 'miou', 'occ_precision', 'occ_recall')
    )
    fig.suptitle(f'{experiment.title} | {case_kind} #{rank_index + 1} | {sample_name}\n{score_text}', fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def evaluate_experiment(
    spec: ExperimentSpec,
    output_root: Path,
    device: torch.device,
    max_samples: int,
    k_best: int,
    k_middle: int,
    k_worst: int,
    rank_by: str,
    max_points: int,
    checkpoint_source: str,
) -> None:
    config = load_run_config(spec)
    stage1_channels = int(config['trainer']['args']['stage1_channels'])
    resolution = int(config['dataset']['args'].get('resolution', 64))
    checkpoint_paths = resolve_checkpoint_paths(Path(spec.output_dir), checkpoint_source)
    models_dict = build_models(config, checkpoint_paths, device)

    records: List[Dict[str, object]] = []
    val_roots = split_roots(spec.val_data)

    with torch.inference_mode():
        for sample_index, (root, sha256, tensor_path) in enumerate(iter_validation_items(val_roots)):
            if max_samples > 0 and sample_index >= max_samples:
                break

            ss = load_sparse_tensor(tensor_path, resolution).unsqueeze(0).to(device)
            z = models_dict['encoder'](ss, sample_posterior=False)
            logits = models_dict['decoder_stage1'](z)
            activated = activate_stage1(logits, stage1_channels)
            pred = discretize_stage1(activated, stage1_channels)
            metrics = compute_metrics(pred, activated, ss[:, :stage1_channels], stage1_channels)
            records.append({
                'sample_index': sample_index,
                'source': source_tag(root),
                'sha256': sha256,
                'tensor_path': str(tensor_path),
                **metrics,
            })

    if not records:
        raise RuntimeError(f'No records evaluated for experiment {spec.name}')

    records_sorted = sorted(records, key=lambda item: (float(item[rank_by]), float(item['miou'])), reverse=True)
    selected_best = records_sorted[:k_best]
    middle_start = max((len(records_sorted) - k_middle) // 2, 0)
    selected_middle = records_sorted[middle_start:middle_start + k_middle]
    selected_worst = list(reversed(records_sorted[-k_worst:]))

    experiment_out = output_root / spec.name
    experiment_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records_sorted).to_csv(experiment_out / 'metrics_ranked.csv', index=False)

    summary = {
        'experiment': asdict(spec),
        'checkpoint_step': int(str(checkpoint_paths['step'])),
        'checkpoint_source': checkpoint_source,
        'checkpoint_paths': {
            'encoder': str(checkpoint_paths['encoder']),
            'decoder_stage1': str(checkpoint_paths['decoder_stage1']),
        },
        'device': str(device),
        'rank_by': rank_by,
        'num_evaluated_samples': len(records),
        'selected_best': selected_best,
        'selected_middle': selected_middle,
        'selected_worst': selected_worst,
    }
    with (experiment_out / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    selected_cases = [('best', selected_best), ('middle', selected_middle), ('worst', selected_worst)]
    with torch.inference_mode():
        for case_kind, cases in selected_cases:
            for rank_index, record in enumerate(cases):
                tensor_path = Path(str(record['tensor_path']))
                gt_tensor = load_sparse_tensor(tensor_path, resolution)
                gt = gt_tensor.unsqueeze(0).to(device)
                z = models_dict['encoder'](gt, sample_posterior=False)
                logits = models_dict['decoder_stage1'](z)
                activated = activate_stage1(logits, stage1_channels)
                pred = discretize_stage1(activated, stage1_channels)[0].cpu()

                sample_name = f"{record['source']}_{record['sha256']}"
                output_path = experiment_out / case_kind / f'{rank_index + 1:02d}_{sample_name}.png'
                render_case_figure(
                    experiment=spec,
                    case_kind=case_kind,
                    rank_index=rank_index,
                    sample_name=sample_name,
                    metrics={
                        'miou_occ': float(record['miou_occ']),
                        'miou': float(record['miou']),
                        'occ_precision': float(record['occ_precision']),
                        'occ_recall': float(record['occ_recall']),
                    },
                    gt_tensor=gt_tensor,
                    pred_tensor=pred,
                    output_path=output_path,
                    max_points=max_points,
                )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for experiment_name in args.experiments:
        spec = DEFAULT_EXPERIMENTS[experiment_name]
        print(f'\n[Stage1CaseViz] Evaluating {spec.title}...')
        evaluate_experiment(
            spec=spec,
            output_root=output_root,
            device=device,
            max_samples=args.max_samples,
            k_best=args.k_best,
            k_middle=args.k_middle,
            k_worst=args.k_worst,
            rank_by=args.rank_by,
            max_points=args.max_points,
            checkpoint_source=args.checkpoint_source,
        )
        print(f'[Stage1CaseViz] Finished {spec.title}. Outputs saved to {output_root / spec.name}')


if __name__ == '__main__':
    main()
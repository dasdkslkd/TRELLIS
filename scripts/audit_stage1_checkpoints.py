#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis import models


EXPERIMENTS = {
    'fusion360': {
        'title': 'Fusion360 Stage1 BCE v3',
        'config': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_stage1_bce_v3/config.json'),
        'val_roots': [Path('/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val')],
        'output': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/fusion360_stage1_bce_v3'),
        'step': 1500,
    },
    'abc1m': {
        'title': 'ABC-1M Stage1 v2',
        'config': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/abc1m_stage1_v2/config.json'),
        'val_roots': [Path('/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split/val')],
        'output': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/abc1m_stage1_v2'),
        'step': 1050,
    },
    'mixed': {
        'title': 'Fusion360 + ABC-1M Mixed Stage1 v1',
        'config': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/mix_fusion360_abc1m_stage1_v1/config.json'),
        'val_roots': [
            Path('/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val'),
            Path('/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split/val'),
        ],
        'output': Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/vae/mix_fusion360_abc1m_stage1_v1'),
        'step': 3850,
    },
}


def load_ss(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, torch.Tensor) and data.layout == torch.sparse_coo:
        data = data.coalesce().to_dense().permute(3, 0, 1, 2)
    elif isinstance(data, torch.Tensor) and data.shape[-1] in (8, 14, 80):
        data = data.permute(3, 0, 1, 2)
    return data.float()


def score_checkpoint(enc, dec, val_roots, limit: int, device: torch.device):
    count = 0
    sums = {
        'miou': 0.0,
        'miou_occ': 0.0,
        'occ_precision': 0.0,
        'occ_recall': 0.0,
        'pred_occ_ratio': 0.0,
        'gt_occ_ratio': 0.0,
    }
    with torch.no_grad():
        for root in val_roots:
            metadata = pd.read_csv(root / 'metadata.csv')
            for sha256 in metadata['sha256'].astype(str):
                ss = load_ss(root / 'data' / f'{sha256}.pt').unsqueeze(0).to(device)
                z = enc(ss, sample_posterior=False)
                logits = dec(z)
                activated = torch.cat([torch.sigmoid(logits[:, :1]), torch.softmax(logits[:, 1:8], dim=1)], dim=1)

                pred_occ = activated[:, :1] > 0.5
                gt_occ = ss[:, :1] > 0.5
                inter = (pred_occ & gt_occ).sum().float()
                union = (pred_occ | gt_occ).sum().float()
                pred_pos = pred_occ.sum().float()
                gt_pos = gt_occ.sum().float()

                occ = ss[:, 0] > 0.5
                cls_match = activated[:, 1:8].argmax(1) == ss[:, 1:8].argmax(1)

                sums['miou'] += ((cls_match & occ).sum().float() / occ.sum().float().clamp(min=1)).item()
                sums['miou_occ'] += (inter / (union + 1e-8)).item()
                sums['occ_precision'] += (inter / pred_pos.clamp(min=1)).item()
                sums['occ_recall'] += (inter / gt_pos.clamp(min=1)).item()
                sums['pred_occ_ratio'] += pred_pos.div(pred_occ.numel()).item()
                sums['gt_occ_ratio'] += gt_pos.div(gt_occ.numel()).item()

                count += 1
                if limit > 0 and count >= limit:
                    return {key: value / count for key, value in sums.items()}, count
    return {key: value / max(count, 1) for key, value in sums.items()}, count


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    limit = 32
    for name, spec in EXPERIMENTS.items():
        cfg = json.load(spec['config'].open('r', encoding='utf-8'))
        print(f'=== {name} | {spec["title"]} | step {spec["step"]} | device={device} ===')
        for kind, suffix in [('plain', ''), ('ema', '_ema0.9999')]:
            enc = models.SparseStructureEncoder(**cfg['models']['encoder']['args']).to(device)
            dec = models.SparseStructureDecoder(**cfg['models']['decoder_stage1']['args']).to(device)
            enc.load_state_dict(torch.load(spec['output'] / 'ckpts' / f'encoder{suffix}_step{spec["step"]:07d}.pt', map_location=device, weights_only=True))
            dec.load_state_dict(torch.load(spec['output'] / 'ckpts' / f'decoder_stage1{suffix}_step{spec["step"]:07d}.pt', map_location=device, weights_only=True))
            enc.eval()
            dec.eval()
            metrics, count = score_checkpoint(enc, dec, spec['val_roots'], limit=limit, device=device)
            rounded = {key: round(value, 6) for key, value in metrics.items()}
            print(kind, f'samples={count}', rounded)
        print()


if __name__ == '__main__':
    main()
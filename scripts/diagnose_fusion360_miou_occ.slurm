#!/bin/bash
#SBATCH --job-name=diag_miou_occ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_A800_2
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main
cd "$REPO_DIR"

python - <<'PY'
from pathlib import Path
import json
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from trellis import models, datasets

repo = Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main')
val_root = '/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val'

runs = {
    'stable_v2': {
        'config': repo / 'configs/vae/ss_two_stage_joint_fusion360_80ch_stable_v2.json',
        'ckpts': {
            'encoder': repo / 'outputs/vae/fusion360_ss_two_stage_joint_stable_v2/ckpts/encoder_step0003000.pt',
            'decoder_stage1': repo / 'outputs/vae/fusion360_ss_two_stage_joint_stable_v2/ckpts/decoder_stage1_step0003000.pt',
        },
    },
    'stage1_v1': {
        'config': repo / 'configs/vae/ss_stage1_fusion360_80ch_finetune_v1.json',
        'ckpts': {
            'encoder': repo / 'outputs/vae/fusion360_stage1_finetune_v1/ckpts/encoder_step0004000.pt',
            'decoder_stage1': repo / 'outputs/vae/fusion360_stage1_finetune_v1/ckpts/decoder_stage1_step0004000.pt',
        },
    },
}

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def activate_stage1(logits):
    occ = torch.sigmoid(logits[:, 0:1])
    cls = torch.softmax(logits[:, 1:8], dim=1)
    return torch.cat([occ, cls], dim=1)


with torch.no_grad():
    dataset = datasets.CustomSparseStructure(val_root, resolution=64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    gt_occ_sum = 0.0
    total_vox = 0
    for batch in loader:
        ss = batch['ss'].float()
        gt_occ_sum += ss[:, 0:1].sum().item()
        total_vox += ss[:, 0:1].numel()
    print('gt_occ_ratio', gt_occ_sum / total_vox)

    for run_name, info in runs.items():
        cfg = edict(json.load(open(info['config'])))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda().eval()
        decoder = getattr(models, cfg.models.decoder_stage1.name)(**cfg.models.decoder_stage1.args).cuda().eval()
        encoder.load_state_dict(torch.load(info['ckpts']['encoder'], map_location='cuda', weights_only=True))
        decoder.load_state_dict(torch.load(info['ckpts']['decoder_stage1'], map_location='cuda', weights_only=True))

        stats = {thr: {'inter': 0.0, 'union': 0.0, 'pred_pos': 0.0, 'gt_pos': 0.0} for thr in thresholds}
        prob_sum = 0.0
        prob_count = 0
        sample_means = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            ss = batch['ss'].float().cuda()
            gt = ss[:, 0:1] > 0.5
            z = encoder(ss, sample_posterior=False)
            logits = decoder(z)
            occ_prob = activate_stage1(logits)[:, 0:1]
            prob_sum += occ_prob.sum().item()
            prob_count += occ_prob.numel()
            sample_means.append(occ_prob.mean().item())
            for thr in thresholds:
                pred = occ_prob > thr
                inter = (pred & gt).sum().item()
                union = (pred | gt).sum().item()
                stats[thr]['inter'] += inter
                stats[thr]['union'] += union
                stats[thr]['pred_pos'] += pred.sum().item()
                stats[thr]['gt_pos'] += gt.sum().item()

        print('\nrun', run_name)
        print('avg_occ_prob', prob_sum / prob_count)
        print('sample_mean_prob_minmax', min(sample_means), max(sample_means))
        for thr in thresholds:
            inter = stats[thr]['inter']
            union = stats[thr]['union']
            pred_pos = stats[thr]['pred_pos']
            gt_pos = stats[thr]['gt_pos']
            precision = inter / pred_pos if pred_pos else 0.0
            recall = inter / gt_pos if gt_pos else 0.0
            iou = inter / union if union else 0.0
            pred_ratio = pred_pos / total_vox
            print(f'thr={thr:.1f} iou={iou:.6f} precision={precision:.6f} recall={recall:.6f} pred_ratio={pred_ratio:.6f}')
PY
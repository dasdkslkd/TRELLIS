#!/bin/bash
#SBATCH --job-name=render_f360_3d
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
from PIL import Image, ImageDraw
import torch
from easydict import EasyDict as edict
from torchvision.utils import save_image
from trellis import datasets, models

repo = Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main')
out_dir = repo / 'outputs/vae/fusion360_reports'
out_dir.mkdir(parents=True, exist_ok=True)

val_root = '/public/home/pb22000140/datasets/Fusion360_seg_voxelized_brep_64_v2_trellis_split/val'
dataset = datasets.CustomSparseStructure(val_root, resolution=64)
batch = torch.stack([dataset[i]['ss'] for i in range(8)], dim=0).cuda().float()


def activate_stage1(logits):
    occ = torch.sigmoid(logits[:, 0:1])
    cls = torch.softmax(logits[:, 1:8], dim=1)
    return torch.cat([occ, cls], dim=1)


def to_hard_stage1(activated):
    recon = torch.zeros_like(activated)
    recon[:, 0:1] = (activated[:, 0:1] > 0.5).float()
    cls_idx = activated[:, 1:8].argmax(dim=1, keepdim=True)
    recon[:, 1:8].scatter_(1, cls_idx, 1.0)
    return recon


runs = {
    'gt': None,
    'stable_v2': {
        'config': repo / 'configs/vae/ss_two_stage_joint_fusion360_80ch_stable_v2.json',
        'encoder': repo / 'outputs/vae/fusion360_ss_two_stage_joint_stable_v2/ckpts/encoder_step0003000.pt',
        'decoder_stage1': repo / 'outputs/vae/fusion360_ss_two_stage_joint_stable_v2/ckpts/decoder_stage1_step0003000.pt',
    },
    'stage1_v1': {
        'config': repo / 'configs/vae/ss_stage1_fusion360_80ch_finetune_v1.json',
        'encoder': repo / 'outputs/vae/fusion360_stage1_finetune_v1/ckpts/encoder_step0004000.pt',
        'decoder_stage1': repo / 'outputs/vae/fusion360_stage1_finetune_v1/ckpts/decoder_stage1_step0004000.pt',
    },
    'stage1_bce_v2': {
        'config': repo / 'configs/vae/ss_stage1_fusion360_80ch_bce_v2.json',
        'encoder': repo / 'outputs/vae/fusion360_stage1_bce_v2/ckpts/encoder_step0003000.pt',
        'decoder_stage1': repo / 'outputs/vae/fusion360_stage1_bce_v2/ckpts/decoder_stage1_step0003000.pt',
    },
}

rendered_paths = []

with torch.no_grad():
    gt_vis = dataset.visualize_sample(batch)
    gt_path = out_dir / 'gt_val8_3d.jpg'
    save_image(gt_vis, gt_path, nrow=4)
    rendered_paths.append(('GT', gt_path))

    for name, info in runs.items():
        if name == 'gt':
            continue
        cfg = edict(json.load(open(info['config'])))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda().eval()
        decoder = getattr(models, cfg.models.decoder_stage1.name)(**cfg.models.decoder_stage1.args).cuda().eval()
        encoder.load_state_dict(torch.load(info['encoder'], map_location='cuda', weights_only=True))
        decoder.load_state_dict(torch.load(info['decoder_stage1'], map_location='cuda', weights_only=True))
        z = encoder(batch, sample_posterior=False)
        logits = decoder(z)
        activated = activate_stage1(logits)
        hard = to_hard_stage1(activated)
        vis = dataset.visualize_sample(hard)
        out_path = out_dir / f'{name}_val8_3d.jpg'
        save_image(vis, out_path, nrow=4)
        rendered_paths.append((name, out_path))

images = [Image.open(path).convert('RGB') for _, path in rendered_paths]
label_h = 72
panel_w = images[0].width
panel_h = images[0].height + label_h
canvas = Image.new('RGB', (panel_w * len(images), panel_h), (8, 10, 14))
draw = ImageDraw.Draw(canvas)

for idx, ((label, _), image) in enumerate(zip(rendered_paths, images)):
    x = idx * panel_w
    canvas.paste(image, (x, label_h))
    draw.text((x + 24, 24), label, fill=(245, 245, 245))

montage_path = out_dir / 'stage1_3d_comparison_val8.jpg'
canvas.save(montage_path, quality=95)
print('saved', montage_path)
for label, path in rendered_paths:
    print(label, path)
PY
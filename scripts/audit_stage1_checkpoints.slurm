#!/bin/bash
#SBATCH --job-name=trellis_ckpt_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU_H800_8
#SBATCH --output=/public/home/pb22000140/TRELLIS-new/TRELLIS-main/logs/%x-%j.out

source ~/.bashrc
set -eo pipefail
conda init >/dev/null 2>&1
conda activate trellis

cd /public/home/pb22000140/TRELLIS-new/TRELLIS-main
python scripts/audit_stage1_checkpoints.py
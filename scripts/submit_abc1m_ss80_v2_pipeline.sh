#!/bin/bash
set -eo pipefail

REPO_DIR=/public/home/pb22000140/TRELLIS-new/TRELLIS-main

scan_job=$(sbatch --parsable "$REPO_DIR/scripts/scan_abc1m_ss80_v2_array.slurm")
select_job=$(sbatch --parsable --dependency=afterok:${scan_job} "$REPO_DIR/scripts/select_abc1m_ss80_v2.slurm")
build_job=$(sbatch --parsable --dependency=afterok:${select_job} "$REPO_DIR/scripts/build_abc1m_ss80_v2_array.slurm")
finalize_job=$(sbatch --parsable --dependency=afterok:${build_job} "$REPO_DIR/scripts/finalize_abc1m_ss80_v2.slurm")

echo "scan_job=${scan_job}"
echo "select_job=${select_job}"
echo "build_job=${build_job}"
echo "finalize_job=${finalize_job}"
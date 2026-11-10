#!/bin/bash
#PBS -N cmi_case_cam
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
WORKDIR=${WORKDIR:-/home/users/astar/ares/zhangrz3/scratch/flowcast-test}
ENV=/home/users/astar/ares/zhangrz3/scratch/conda_envs/flowcast
export PYTHONHOME="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export WANDB_DISABLED=true WANDB_MODE=disabled HDF5_USE_FILE_LOCKING=FALSE
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=1
export MPLBACKEND=Agg
cd "$WORKDIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${PBS_JOBID:-local}"
OUT="$WORKDIR/visualizations/outputs/cmi_case_$RUN_ID"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
export MPLCONFIGDIR="$OUT/matplotlib"
exec > >(tee -a "$OUT/submission.log") 2>&1

BASE=artifacts/sevir/lightning_diffusion_moe
CMI_CKPT=${CMI_CKPT:-$BASE/2026-10-18_21-29-47_lightning_diffusion_moe_cmi_gated_up_l0_t/models/best_lightning_diffusion_moe.pt}
CROSS_CKPT=${CROSS_CKPT:-$BASE/2026-09-20_21-38-45_lightning_diffusion_moe_cross_modal_gated_up_l0_t/models/best_lightning_diffusion_moe.pt}
python visualizations/cmi_case_cam.py --baseline "$CROSS_CKPT" --cmi "$CMI_CKPT" --sample "${SAMPLE_INDEX:-0}" --seed "${SEED:-2026}" --output "$OUT"
echo "Finished: $OUT"

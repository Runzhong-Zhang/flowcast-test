#!/bin/bash
#PBS -N moe_test_up_t
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/moe_test.log
#PBS -e logs/moe_test.err
#PBS -l walltime=40:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs
mkdir -p "$LOGDIR"
LOG=${LOGDIR}/running_moe_up_t_test.log
exec > >(tee -a "$LOG") 2>&1

echo "JobID: ${PBS_JOBID}"
hostname
date
nvidia-smi -L || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# ====== 2) Environment ======
ENV=/home/users/astar/ares/wangl4/scratch/zihao/.conda/envs/flowcast
export PYTHONHOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export NVIDIA_TF32_OVERRIDE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4

cd "$WORKDIR"

# ====== 3) Find checkpoint ======
# Usage: modify CKPT to point to the specific checkpoint you want to test
#   Option A: auto-find latest best checkpoint
CKPT=artifacts/sevir/lightning_diffusion_moe/2026-05-17_23-57-38_lightning_diffusion_moe_cuboid_e8k2_up_t/models/best_lightning_diffusion_moe.pt
#   Option B: specify manually (uncomment and edit)
# CKPT=artifacts/sevir/lightning_diffusion_moe/<run_dir>/models/best_lightning_diffusion_moe.pt

if [ -z "$CKPT" ]; then
    echo "[ERROR] No checkpoint found"
    exit 2
fi
echo "Using checkpoint: $CKPT"

# ====== 4) Config ======
# Change this to match the config used for training
CONFIG=lightning_diffusion_moe_config_server.yaml

# ---- sanity check ----
for f in \
  experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py \
  "$CONFIG" \
  "$CKPT" \
  data/nowcast_testing_vil.h5 \
  data/nowcast_testing_lght.h5 \
  data/nowcast_testing_META.csv
do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing file: $f"
    exit 2
  fi
done

# ====== 5) Inference ======
echo "================== START INFERENCE =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> logs/gpu_monitor_test.log 2>&1 &

python \
  experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --test_vil_file      data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta          data/nowcast_testing_META.csv \
  --batch_size 4 \
  --num_workers 4

echo "================== FINISHED =================="
date

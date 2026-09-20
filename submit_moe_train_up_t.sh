#!/bin/bash
#PBS -N moe_train_up_t
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/moe_train_up_t.log
#PBS -e logs/moe_train_up_t.err
#PBS -l walltime=72:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs
mkdir -p "$LOGDIR"
LOG=${LOGDIR}/running_moe_train_up_t.log
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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "$WORKDIR"

# ====== 3) Config & Data ======
CONFIG=lightning_diffusion_moe_config_server_up_t.yaml

# ---- sanity check ----
for f in \
  experiments/sevir/runner/flowcast/dist_train_lightning_diffusion_moe.py \
  "$CONFIG" \
  data/nowcast_training_vil.h5 \
  data/nowcast_training_lght.h5 \
  data/nowcast_training_META.csv \
  data/nowcast_validation_vil.h5 \
  data/nowcast_validation_lght.h5 \
  data/nowcast_validation_META.csv
do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing file: $f"
    exit 2
  fi
done

# ====== 4) DDP Training (single node, 4 GPUs) ======
echo "================== START TRAINING (up_t) =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> logs/gpu_monitor_train_up_t.log 2>&1 &

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  experiments/sevir/runner/flowcast/dist_train_lightning_diffusion_moe.py \
  --config "$CONFIG" \
  --train_vil_file       data/nowcast_training_vil.h5 \
  --train_lightning_file data/nowcast_training_lght.h5 \
  --train_meta           data/nowcast_training_META.csv \
  --val_vil_file         data/nowcast_validation_vil.h5 \
  --val_lightning_file   data/nowcast_validation_lght.h5 \
  --val_meta             data/nowcast_validation_META.csv

echo "================== FINISHED (up_t) =================="
date

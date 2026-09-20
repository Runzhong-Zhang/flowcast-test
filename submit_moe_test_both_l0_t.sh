#!/bin/bash
#PBS -N moe_test_both_l0_t
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/both_l0_t/test/pbs.log
#PBS -e logs/both_l0_t/test/pbs.err
#PBS -l walltime=04:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs/both_l0_t/test
mkdir -p "$LOGDIR"
LOG=${LOGDIR}/running.log
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
# Find the best checkpoint from both_l0_t training runs
CKPT=$(find artifacts/sevir/lightning_diffusion_moe -path "*both_l0_t*" -name "best_lightning_diffusion_moe.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$CKPT" ]; then
    echo "[ERROR] No both_l0_t checkpoint found"
    exit 2
fi
echo "Using checkpoint: $CKPT"

# ====== 4) Config ======
CONFIG=lightning_diffusion_moe_config_server_both_l0_t.yaml

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
echo "================== START INFERENCE (both_l0_t) =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> ${LOGDIR}/gpu_monitor.log 2>&1 &

python \
  experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --test_vil_file      data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta          data/nowcast_testing_META.csv \
  --batch_size 4 \
  --num_workers 4

echo "================== FINISHED (both_l0_t) =================="
date

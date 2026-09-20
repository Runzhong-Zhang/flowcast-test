#!/bin/bash
#PBS -N moe_shared_up_l0_t
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/shared/test/pbs.log
#PBS -e logs/shared/test/pbs.err
#PBS -l walltime=24:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs/shared/test
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
CKPT=artifacts/sevir/lightning_diffusion_moe/2026-06-10_16-25-10_lightning_diffusion_moe_shared_up_l0_t/models/best_lightning_diffusion_moe.pt
# CKPT=artifacts/sevir/lightning_diffusion_moe/<run_dir>/models/best_lightning_diffusion_moe.pt

if [ -z "$CKPT" ]; then
    echo "[ERROR] No checkpoint found"
    exit 2
fi
echo "Using checkpoint: $CKPT"

# ====== 4) Config ======
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

# ====== 5) Parse GPU IDs ======
# CUDA_VISIBLE_DEVICES contains comma-separated GPU UUIDs or indices
# Convert to array for per-shard assignment
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Available GPUs: $NUM_GPUS"

NUM_SHARDS=4
SCRIPT=experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py
COMMON_ARGS="--config $CONFIG --checkpoint $CKPT \
  --test_vil_file data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta data/nowcast_testing_META.csv \
  --batch_size 4 --num_workers 4 \
  --num_shards $NUM_SHARDS"

# ====== 6) Launch 4 shards in parallel ======
echo "================== START INFERENCE (4-GPU sharded) =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> ${LOGDIR}/gpu_monitor.log 2>&1 &
MONITOR_PID=$!

PIDS=()
for RANK in 0 1 2 3; do
    GPU_ID=${GPU_LIST[$RANK]}
    echo "Launching shard $RANK on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $COMMON_ARGS \
        --shard_rank $RANK \
        >> ${LOGDIR}/shard_${RANK}.log 2>&1 &
    PIDS+=($!)
done

# ====== 7) Wait for all shards ======
echo "Waiting for ${#PIDS[@]} shards to finish..."
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait ${PIDS[$i]}; then
        echo "[ERROR] Shard $i (PID ${PIDS[$i]}) failed"
        FAIL=1
    else
        echo "Shard $i finished successfully"
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "[ERROR] One or more shards failed, skipping merge"
    kill $MONITOR_PID 2>/dev/null || true
    exit 1
fi

# ====== 8) Merge shards ======
echo "================== MERGING SHARDS =================="
date

python $SCRIPT \
  --config $CONFIG \
  --checkpoint "$CKPT" \
  --test_vil_file      data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta          data/nowcast_testing_META.csv \
  --num_shards $NUM_SHARDS \
  --merge_shards

kill $MONITOR_PID 2>/dev/null || true

echo "================== FINISHED =================="
date

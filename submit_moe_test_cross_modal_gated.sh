#!/bin/bash
#PBS -N moe_test_cm_gated
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/cross_modal_gated/test/pbs.log
#PBS -e logs/cross_modal_gated/test/pbs.err
#PBS -l walltime=12:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs/cross_modal_gated/test
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
CKPT=$(find artifacts/sevir/lightning_diffusion_moe -path "*cross_modal_gated*" -name "best_lightning_diffusion_moe.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$CKPT" ]; then
    echo "[ERROR] No cross_modal_gated checkpoint found"
    exit 2
fi
echo "Using checkpoint: $CKPT"

# ====== 4) Config ======
CONFIG=lightning_diffusion_moe_config_server_cross_modal_gated.yaml

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

# ====== 5) 4-GPU sharded inference ======
echo "================== START INFERENCE (cross_modal_gated, 4-GPU) =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> ${LOGDIR}/gpu_monitor.log 2>&1 &
MONITOR_PID=$!

NUM_SHARDS=4
SCRIPT=experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py
COMMON_ARGS="--config $CONFIG --checkpoint $CKPT \
  --test_vil_file data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta data/nowcast_testing_META.csv \
  --batch_size 4 --num_workers 4 \
  --num_shards $NUM_SHARDS"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"

PIDS=()
for RANK in 0 1 2 3; do
    GPU_ID=${GPU_LIST[$RANK]}
    echo "Launching shard $RANK on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $COMMON_ARGS \
        --shard_rank $RANK \
        >> ${LOGDIR}/shard_${RANK}.log 2>&1 &
    PIDS+=($!)
done

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

# ====== 6) Merge shards ======
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

echo "================== FINISHED (cross_modal_gated) =================="
date

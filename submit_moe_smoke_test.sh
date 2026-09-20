#!/bin/bash
#PBS -N moe_smoke_test
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -M wang_lu@a-star.edu.sg
#PBS -o logs/moe_smoke_test.log
#PBS -e logs/moe_smoke_test.err
#PBS -l walltime=02:00:00
#PBS -j oe

set -e
set -o pipefail

# ====== 1) Directory ======
WORKDIR=/home/users/astar/ares/wangl4/scratch/zihao/flowcast-test
LOGDIR=${WORKDIR}/logs
mkdir -p "$LOGDIR"
LOG=${LOGDIR}/running_moe_smoke_test.log
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

# ====== 3) Smoke test config: debug_mode=true, 2 epochs ======
SMOKE_CONFIG=/tmp/moe_smoke_config_${PBS_JOBID}.yaml

python - <<'EOF' "$SMOKE_CONFIG"
import sys
from omegaconf import OmegaConf

out_path = sys.argv[1]
cfg = OmegaConf.load("lightning_diffusion_moe_config_server.yaml")

# smoke test overrides
cfg.run_params.debug_mode       = False
cfg.run_params.run_string       = "smoke_test"
cfg.training_params.num_epochs  = 1
cfg.training_params.micro_batch_size = 8
cfg.training_params.num_workers = 4
cfg.training_params.fp16        = True
cfg.training_params.init_checkpoint   = None
cfg.training_params.resume_optimizer  = False
cfg.training_params.resume_scheduler  = False
cfg.training_params.resume_scaler     = False

OmegaConf.save(cfg, out_path)
print(f"[smoke] config written to {out_path}")
EOF

# ---- sanity check ----
for f in \
  experiments/sevir/runner/flowcast/dist_train_lightning_diffusion_moe.py \
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

echo "================== START SMOKE TEST =================="
date

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> logs/gpu_monitor.log 2>&1 &
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  experiments/sevir/runner/flowcast/dist_train_lightning_diffusion_moe.py \
  --config "$SMOKE_CONFIG" \
  --train_vil_file       data/nowcast_training_vil.h5 \
  --train_lightning_file data/nowcast_training_lght.h5 \
  --train_meta           data/nowcast_training_META.csv \
  --val_vil_file         data/nowcast_validation_vil.h5 \
  --val_lightning_file   data/nowcast_validation_lght.h5 \
  --val_meta             data/nowcast_validation_META.csv

echo "================== SMOKE TEST PASSED =================="
date

# cleanup
rm -f "$SMOKE_CONFIG"

#!/bin/bash
#PBS -N moe_routing_maps
#PBS -q normal
#PBS -l select=1:ngpus=4
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
OUT="$WORKDIR/visualizations/outputs/$RUN_ID"
mkdir -p "$OUT"
export MPLCONFIGDIR="$OUT/matplotlib"
exec > >(tee -a "$OUT/submission.log") 2>&1

# Explicit overrides may be supplied with qsub -v CROSS_CKPT=...,CMI_CKPT=...
latest_checkpoint() {
    python - "$1" <<'PY'
import sys
from pathlib import Path
paths = [p for p in Path('artifacts/sevir/lightning_diffusion_moe').rglob(
    'best_lightning_diffusion_moe.pt') if sys.argv[1] in str(p)]
if not paths:
    raise SystemExit('No checkpoint found for ' + sys.argv[1])
print(max(paths, key=lambda p: p.stat().st_mtime))
PY
}
CROSS_CKPT=${CROSS_CKPT:-$(latest_checkpoint cross_modal_gated)}
CMI_CKPT=${CMI_CKPT:-$(latest_checkpoint cmi_gated)}
for file in "$CROSS_CKPT" "$CMI_CKPT" \
    lightning_diffusion_moe_config_server_cross_modal_gated.yaml \
    lightning_diffusion_moe_config_server_cmi_gated.yaml \
    data/nowcast_testing_vil.h5 data/nowcast_testing_lght.h5 data/nowcast_testing_META.csv \
    configs/autoencoder_kl_config.yaml configs/flowcast_config.yaml \
    saved_models/servir/autoencoder/early_stopping_model.pt \
    saved_models/servir/flowcast/early_stopping_model.pt; do
    test -f "$file" || { echo "Missing file: $file"; exit 2; }
done
python -c "import torch, numpy, omegaconf, torchdiffeq, matplotlib, h5py"
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:?PBS must provide CUDA_VISIBLE_DEVICES}"
if [ "${#GPU_LIST[@]}" -ne 4 ]; then
    echo "Expected four allocated GPUs, got ${CUDA_VISIBLE_DEVICES}"; exit 2
fi
echo "Cross-modal checkpoint: $CROSS_CKPT"
echo "CMI checkpoint: $CMI_CKPT"
echo "Outputs: $OUT"
SCRIPT=visualizations/expert_probability_maps.py
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT
for rank in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}"
        for label in cross_modal_gated cmi_gated; do
            if [ "$label" = cross_modal_gated ]; then ckpt="$CROSS_CKPT"; else ckpt="$CMI_CKPT"; fi
            python "$SCRIPT" infer --label "$label" --checkpoint "$ckpt" \
                --config "lightning_diffusion_moe_config_server_${label}.yaml" \
                --output_dir "$OUT" --num_shards 4 --shard_rank "$rank" \
                --euler_steps 4 --members 8 --seed 2026 --batch_size 4 --num_workers 4
        done
    ) > "$OUT/shard_${rank}.log" 2>&1 &
    PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=1; fi
done
PIDS=()
if [ "$FAIL" -ne 0 ]; then echo "Inference failed; inspect shard logs in $OUT"; exit 1; fi
python "$SCRIPT" plot --output_dir "$OUT" --num_shards 4
echo "Finished: $OUT"

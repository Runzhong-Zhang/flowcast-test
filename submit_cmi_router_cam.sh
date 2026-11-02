#!/bin/bash
#PBS -N cmi_router_cam
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
OUT="$WORKDIR/visualizations/outputs/cmi_cam_$RUN_ID"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
export MPLCONFIGDIR="$OUT/matplotlib"
exec > >(tee -a "$OUT/submission.log") 2>&1

# Explicit overrides may be supplied with qsub -v CMI_CKPT=...
CMI_CKPT=${CMI_CKPT:-artifacts/sevir/lightning_diffusion_moe/2026-10-18_21-29-47_lightning_diffusion_moe_cmi_gated_up_l0_t/models/best_lightning_diffusion_moe.pt}
for file in "$CMI_CKPT" \
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
echo "CMI checkpoint: $CMI_CKPT"
echo "Outputs: $OUT"
SCRIPT=visualizations/cmi_router_cam.py
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT
for rank in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}"
        exec python "$SCRIPT" infer --checkpoint "$CMI_CKPT" \
            --output_dir "$OUT" --num_shards 4 --shard_rank "$rank" \
            --members "${MEMBERS:-8}" --batch_size "${BATCH_SIZE:-4}" \
            --test_data_percentage "${TEST_FRACTION:-1}"
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

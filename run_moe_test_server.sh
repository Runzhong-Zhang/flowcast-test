#!/bin/bash
# Run MoE inference on server with relative paths
# Usage: cd flowcast-test && nohup bash run_moe_test_server.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"

# Find the latest best checkpoint
CKPT=$(find artifacts/sevir/lightning_diffusion_moe -name "best_lightning_diffusion_moe.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi
echo "Using checkpoint: $CKPT"

python \
  experiments/sevir/runner/flowcast/test_lightning_diffusion_moe.py \
  --config lightning_diffusion_moe_config_server.yaml \
  --checkpoint "$CKPT" \
  --test_vil_file      data/nowcast_testing_vil.h5 \
  --test_lightning_file data/nowcast_testing_lght.h5 \
  --test_meta          data/nowcast_testing_META.csv \
  --batch_size 3 \
  >> artifacts/sevir/lightning_diffusion_moe/test_run.log 2>&1

#!/bin/bash
# Run MoE training on server with relative paths
# Usage: cd flowcast-test && nohup bash run_moe_train_server.sh > /dev/null 2>&1 &
cd "$(dirname "$0")"

python \
  experiments/sevir/runner/flowcast/dist_train_lightning_diffusion_moe.py \
  --config lightning_diffusion_moe_config_server.yaml \
  --train_vil_file       data/nowcast_training_vil.h5 \
  --train_lightning_file data/nowcast_training_lght.h5 \
  --train_meta           data/nowcast_training_META.csv \
  --val_vil_file         data/nowcast_validation_vil.h5 \
  --val_lightning_file   data/nowcast_validation_lght.h5 \
  --val_meta             data/nowcast_validation_META.csv \
  >> artifacts/sevir/lightning_diffusion_moe/train_run.log 2>&1

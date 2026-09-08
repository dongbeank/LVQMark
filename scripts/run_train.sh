#!/usr/bin/env bash
# Train L-VQ-VAE (4 stages: pretrain → decoder → AR transformer → robust encoder)
# on every dataset.
#
# The robust encoder's size is read from each config's `robust_encoder:` section
# (stocks uses d_model=256, the rest 1024) — no per-dataset flags needed here.
#
# Override: GPU=0 WINDOW=64 scripts/run_train.sh
# Single dataset: scripts/run_train.sh etth
set -euo pipefail
cd "$(dirname "$0")/.."

# Make --gpu N mean the device nvidia-smi calls N. Without it torch may enumerate the GPUs
# in a different order, and on a multi-GPU host device 0 can be a different architecture
# from the rest -- numbers are only comparable within one architecture.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU index; on a multi-GPU host 0 may be a different architecture from the others.
GPU="${GPU:-0}"
WINDOW="${WINDOW:-64}"
DATASETS=("${@:-etth energy stocks fmri}")

for ds in ${DATASETS[@]}; do
  echo "============================================================"
  echo "  TRAIN  ds=$ds  window=$WINDOW  gpu=$GPU"
  echo "============================================================"
  python -u cli/train.py \
      --config_file "configs/lvqvae/${ds}.yaml" \
      --dataset "$ds" \
      --window_size "$WINDOW" \
      --gpu "$GPU"
done

#!/usr/bin/env bash
# Generate clean + LVQMark-watermarked samples (10000 each) and run detection
# (clean + 6 attacks × {WM, NoWM}) for every dataset.
#
# delta = 10 for etth/energy/fmri, 20 for stocks (paper setting).
#
# Override: GPU=0 WINDOW=64 scripts/run_generate.sh
# Single dataset: scripts/run_generate.sh stocks
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

delta_for() {
  case "$1" in
    stocks) echo 20 ;;
    *)      echo 10 ;;
  esac
}

for ds in ${DATASETS[@]}; do
  DELTA=$(delta_for "$ds")
  CFG="configs/lvqvae/${ds}.yaml"
  echo "============================================================"
  echo "  GENERATE  ds=$ds  window=$WINDOW  delta=$DELTA  gpu=$GPU"
  echo "============================================================"

  # 1) clean (no watermark) — needed once per (ds, window, delta-bucket)
  python -u cli/generate.py \
      --config_file "$CFG" --dataset "$ds" \
      --window_size "$WINDOW" --gpu "$GPU" \
      --watermark_delta "$DELTA"

  # 2) LVQMark watermarked
  python -u cli/generate.py \
      --config_file "$CFG" --dataset "$ds" \
      --window_size "$WINDOW" --gpu "$GPU" \
      --watermark --wm_method LVQMark --watermark_delta "$DELTA"

  # 3) detection over all attacks for LVQMark
  python -u cli/generate.py \
      --config_file "$CFG" --dataset "$ds" \
      --window_size "$WINDOW" --gpu "$GPU" \
      --watermark_delta "$DELTA" \
      --detect_all --methods LVQMark
done

#!/usr/bin/env bash
# Evaluate L-VQ-VAE: quality metrics (CFID/Corr/Disc/Pred) + watermark Z-score.
#
# delta = 10 for etth/energy/fmri, 20 for stocks (paper setting).
# MODE=full|quick|zscore (default: run all three)
#
# Override: MODE=zscore GPU=1 scripts/run_evaluate.sh
# Single dataset: scripts/run_evaluate.sh stocks
set -euo pipefail
cd "$(dirname "$0")/.."

# Make --gpu N mean the device nvidia-smi calls N. Without it torch may enumerate the GPUs
# in a different order, and on a multi-GPU host device 0 can be a different architecture
# from the rest -- numbers are only comparable within one architecture.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU index; on a multi-GPU host 0 may be a different architecture from the others.
# Evaluate on the same GPU that generated the samples.
GPU="${GPU:-0}"
WINDOW="${WINDOW:-64}"
MODES=("${MODE:-full quick zscore}")
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
  for mode in ${MODES[@]}; do
    # The quality metrics need --deterministic to repeat on GPU. The z-score is pure numpy
    # and already repeats without it; passing it there would only rename the CSV.
    DET=()
    [ "$mode" != "zscore" ] && DET=(--deterministic)
    echo "============================================================"
    echo "  EVAL  ds=$ds  window=$WINDOW  delta=$DELTA  mode=$mode  gpu=$GPU"
    echo "============================================================"
    python -u cli/evaluate.py \
        --config_file "$CFG" --dataset "$ds" \
        --window "$WINDOW" --delta "$DELTA" --mode "$mode" \
        --gpu "$GPU" ${DET[@]+"${DET[@]}"}
  done
done

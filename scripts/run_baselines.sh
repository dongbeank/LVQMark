#!/usr/bin/env bash
# Run one baseline generator end to end on every dataset: train -> generate (clean + each
# of that generator's watermarks) -> detect (clean + 6 attacks, on WM and NoWM samples) ->
# evaluate (quality full/quick + watermark z-score).
#
# GENERATOR=diffusionts (default): watermarks TimeWak, TR, GS. Diffusion-TS has no
#   watermark delta -- a latent watermark biases no logit -- so no delta flag is passed.
# GENERATOR=sdformer: watermark LVQMark, delta = 10 for etth/energy/fmri, 20 for stocks,
#   the same values the L-VQVAE launchers use.
#
# Override: GENERATOR=sdformer GPU=1 WINDOW=64 scripts/run_baselines.sh
# Single dataset: GENERATOR=sdformer scripts/run_baselines.sh stocks
#
# Unlike the three L-VQVAE launchers this one does not stop at the first failure: a
# generator that dies on one dataset should not hide the other three. Each failed step
# prints one FAILED line and the script exits non-zero at the end.
set -u
cd "$(dirname "$0")/.."

# Make --gpu N mean the device nvidia-smi calls N. Without it torch may enumerate the GPUs
# in a different order, and on a multi-GPU host device 0 can be a different architecture
# from the rest -- numbers are only comparable within one architecture.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GENERATOR="${GENERATOR:-diffusionts}"
GPU="${GPU:-0}"
WINDOW="${WINDOW:-64}"
MODES=("${MODE:-full quick zscore}")
DATASETS=("${@:-etth stocks energy fmri}")

case "$GENERATOR" in
  diffusionts) WM_METHODS=(TimeWak TR GS) ;;
  sdformer)    WM_METHODS=(LVQMark) ;;
  *) echo "unknown GENERATOR '$GENERATOR' (diffusionts | sdformer)"; exit 2 ;;
esac

FAILED=0

run () {  # run <label> <command...> -- report and continue instead of aborting the sweep
  local label="$1"; shift
  echo "------------------------------------------------------------"
  echo "  $label"
  echo "------------------------------------------------------------"
  if "$@"; then
    return 0
  fi
  echo "FAILED [$GENERATOR]: $label"
  FAILED=1
  return 1
}

delta_for() {
  case "$1" in
    stocks) echo 20 ;;
    *)      echo 10 ;;
  esac
}

for ds in ${DATASETS[@]}; do
  CFG="configs/${GENERATOR}/${ds}.yaml"
  # The delta flag is spelled --watermark_delta on the generate side and --delta on the
  # evaluate side; both are empty for the latent-watermark generator.
  GEN_DELTA=()
  EVAL_DELTA=()
  if [ "$GENERATOR" = "sdformer" ]; then
    GEN_DELTA=(--watermark_delta "$(delta_for "$ds")")
    EVAL_DELTA=(--delta "$(delta_for "$ds")")
  fi

  echo "============================================================"
  echo "  BASELINE  generator=$GENERATOR  ds=$ds  window=$WINDOW  gpu=$GPU ${GEN_DELTA[*]-}"
  echo "============================================================"

  run "train  $ds" \
      python -u cli/train_baselines.py --generator "$GENERATOR" \
          --config_file "$CFG" --dataset "$ds" \
          --window_size "$WINDOW" --gpu "$GPU" || continue

  # 1) clean (no watermark) -- the control every watermarked pool is scored against
  run "generate clean  $ds" \
      python -u cli/generate_baselines.py --generator "$GENERATOR" \
          --config_file "$CFG" --dataset "$ds" \
          --window_size "$WINDOW" --gpu "$GPU" ${GEN_DELTA[@]+"${GEN_DELTA[@]}"}

  # 2) one watermarked pool per method this generator carries
  for wm in "${WM_METHODS[@]}"; do
    run "generate $wm  $ds" \
        python -u cli/generate_baselines.py --generator "$GENERATOR" \
            --config_file "$CFG" --dataset "$ds" \
            --window_size "$WINDOW" --gpu "$GPU" ${GEN_DELTA[@]+"${GEN_DELTA[@]}"} \
            --watermark --wm_method "$wm"
  done

  # 3) detection over the clean condition + every attack, on WM and NoWM samples
  run "detect  $ds" \
      python -u cli/generate_baselines.py --generator "$GENERATOR" \
          --config_file "$CFG" --dataset "$ds" \
          --window_size "$WINDOW" --gpu "$GPU" ${GEN_DELTA[@]+"${GEN_DELTA[@]}"} \
          --detect_all

  # 4) metrics. The quality metrics need --deterministic to repeat on GPU; the z-score is
  #    pure numpy and already repeats without it, where it would only rename the CSV.
  for mode in ${MODES[@]}; do
    DET=()
    [ "$mode" != "zscore" ] && DET=(--deterministic)
    run "evaluate $mode  $ds" \
        python -u cli/evaluate_baselines.py --generator "$GENERATOR" \
            --config_file "$CFG" --dataset "$ds" \
            --window "$WINDOW" --gpu "$GPU" ${EVAL_DELTA[@]+"${EVAL_DELTA[@]}"} \
            --mode "$mode" ${DET[@]+"${DET[@]}"}
  done
done

if [ "$FAILED" -ne 0 ]; then
  echo "one or more steps FAILED (see the FAILED lines above)"
  exit 1
fi
echo "all steps completed [$GENERATOR]"

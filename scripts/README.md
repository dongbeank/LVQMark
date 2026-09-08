# scripts/

Shell launchers. No library code. Every script `cd`s to the repository root first, so it can
be started from anywhere.

| script | what it does |
|---|---|
| `run_train.sh` | train all datasets: pretrain → decoder → AR transformer → robust encoder |
| `run_generate.sh` | generate clean + watermarked samples, then detect over the attack grid |
| `run_evaluate.sh` | quality metrics (`full` / `quick`) + watermark z-score |
| `run_baselines.sh` | the same pipeline end to end for `GENERATOR=diffusionts \| sdformer` |

The `run_*` scripts take an optional dataset list and read `GPU` / `WINDOW` (and `MODE`, for
`run_evaluate.sh`) from the environment:

```bash
bash scripts/run_train.sh                             # all four datasets
GPU=1 WINDOW=24 bash scripts/run_train.sh
bash scripts/run_generate.sh stocks
MODE=zscore bash scripts/run_evaluate.sh etth energy
GENERATOR=sdformer bash scripts/run_baselines.sh
```

They pass `configs/<generator>/<ds>.yaml` and the default seeds, so they write the
all-default keys `…_trainseed12345[_delta<δ>_genseed12345_wmseed12345]`. Varying a seed needs
the `cli/` entry points directly (`--seed` / `--watermark_seed` / `--train_seed`).

Each `run_*` script exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `--gpu N` selects the device the driver
reports under index N. All arms of one experiment should run on the same GPU architecture;
numbers are only comparable within an architecture.

Nothing chains train → generate → evaluate in one command.

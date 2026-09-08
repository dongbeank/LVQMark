# LVQMark

Watermarking for time-series synthesis: a Local Vector-Quantized VAE tokenizes short windows,
an autoregressive transformer samples the token stream under a green-list logit bias, and a
robustness-trained encoder recovers the mark from the generated series.

This repository trains, generates and evaluates the model on four datasets: ETTh, Energy,
Stocks and fMRI.

---

## 1. Setup

```bash
pip install -r requirements.txt
```

Python 3.10, PyTorch 2.10 (CUDA 12.8), TensorFlow 2.12 (used only by the discriminative and
predictive metrics).

All four datasets are in `Data/datasets/`.


---

## 2. Running

```bash
bash scripts/run_train.sh       # train all four datasets
bash scripts/run_generate.sh    # generate clean + watermarked samples, then detect
bash scripts/run_evaluate.sh    # quality metrics + watermark z-score
```

Each takes an optional dataset list and reads `GPU`, `WINDOW` (and `MODE`, for
`run_evaluate.sh`) from the environment:

```bash
GPU=1 WINDOW=24 bash scripts/run_train.sh
bash scripts/run_generate.sh stocks
MODE=zscore bash scripts/run_evaluate.sh etth energy
```

`bash scripts/run_baselines.sh` does the same end to end for a baseline generator
(`GENERATOR=diffusionts` or `GENERATOR=sdformer`).

On a multi-GPU host, `export CUDA_DEVICE_ORDER=PCI_BUS_ID` so that `--gpu N` selects the
device `nvidia-smi` calls N; the scripts already do.

### The commands the scripts run

The entry points live in `cli/` and may be invoked from any directory.

```bash
# train (4 stages: pretrain -> decoder -> AR transformer -> robust encoder)
python -u cli/train.py --config_file configs/lvqvae/etth.yaml \
    --dataset etth --window_size 64 --gpu 0
# one stage only: --stage 4

# generate, without and with the watermark
python -u cli/generate.py --config_file configs/lvqvae/etth.yaml \
    --dataset etth --window_size 64 --gpu 0 --watermark_delta 10
python -u cli/generate.py --config_file configs/lvqvae/etth.yaml \
    --dataset etth --window_size 64 --gpu 0 --watermark --watermark_delta 10

# detect: the clean condition + 6 attacks, on watermarked and non-watermarked samples
python -u cli/generate.py --config_file configs/lvqvae/etth.yaml \
    --dataset etth --window_size 64 --gpu 0 --watermark_delta 10 --detect_all
# score the same samples under a different key (the wrong-key control): --detect_key 999

# evaluate: --mode full | quick | zscore
python -u cli/evaluate.py --config_file configs/lvqvae/etth.yaml \
    --dataset etth --window 64 --delta 10 --mode full --deterministic
```

Seeds: `--seed` everywhere (default 12345), plus `--watermark_seed` for the green-list key.
`cli/evaluate.py --deterministic` turns on `cudnn.deterministic`, which the quality metrics
need to repeat on GPU. `--gpu N` selects the CUDA device.

The watermark has two switches, both on by default: `--alternating_partition` flips the green
set with token-position parity, and `--unbiased_warmup` leaves the first `--warmup_positions`
token positions without a logit bias (default `min(4, token_length // 3)`).

The baseline generators use `cli/train_baselines.py`, `cli/generate_baselines.py` and
`cli/evaluate_baselines.py` with `--generator {sdformer,diffusionts}`.

---

## 3. Layout

```
cli/            train.py generate.py evaluate.py + the *_baselines.py twins
scripts/        run_train.sh run_generate.sh run_evaluate.sh run_baselines.sh
configs/        per-dataset YAML, one directory per generator
Models/         local_vqvae, transformer_ar, interpretable_diffusion/, sdformer/, ts2vec/
Utils/          io_utils, path_utils, greenlist, watermark_utils, metrics
Data/           build_dataloader.py + datasets/
```

Artifacts are created on first run, each tree starting with the generator level:

```
checkpoints/lvqvae/{dataset}/{exp}/{window}/{vqvae,transformer,robustencoder}/

outputs/lvqvae/{dataset}/{run}/{window}/
    sample/generated{,_norm}.npy            # no watermark
    LVQMark/sample/generated{,_norm}.npy    # watermarked
    LVQMark/detect/LVQMark.csv              # green fraction per window, clean condition
    LVQMark/{offset,crop,insert}-{0.05,0.3}{,-wo}/LVQMark.csv    # the 6 attacks
    samples/{dataset}_norm_truth_{window}_test.npy               # ground truth

results/lvqvae/{dataset}/{run}/
    w{window}_{zscore,quality_quick,quality_full}_seed{s}[_det][_key{k}].csv
```

`-wo` marks the same condition scored on the non-watermarked samples; `-key{k}` / `_key{k}`
mark a run scored under a different key; `_det` marks `--deterministic`.

`{exp}` and `{run}` are derived from the config and the seeds by `Utils/path_utils.py`:

```
exp = d{d_model}_cb{codebook_size}_st{stride}_ar{n_embd}_l{n_layer}_h{n_head}_trainseed{s}
run = {exp}_delta{δ}_genseed{s}_wmseed{s}
```

`exp` identifies the weights, `run` the samples. The full resolved configuration and the GPU
the run used are written next to the artifacts as `run_config.json`.

Generation and detection skip work that is already on disk: delete an artifact to recompute
it.

---

---

## Licence

MIT.

## Acknowledgement

This codebase builds on [TimeWak](https://github.com/soizhiwen/TimeWak),
[Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS),
[TS2Vec](https://github.com/zhihanyue/ts2vec) and [TimeGAN](https://github.com/jsyoon0823/TimeGAN).
Datasets: [ETDataset](https://github.com/zhouhaoyi/ETDataset),
[UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
[TimeGAN stock data](https://github.com/jsyoon0823/TimeGAN),
[FMRIB NetSim](https://www.fmrib.ox.ac.uk/datasets/netsim/).

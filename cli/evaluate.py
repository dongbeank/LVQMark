"""
Evaluate Local VQ-VAE (organized structure).

Paths (derived by Utils/path_utils.py — the same derivation train.py and generate.py use):
  output:  outputs/{generator}/{dataset}/{run}/{window}/
  results: results/{generator}/{dataset}/{run}/
  where generator = "lvqvae" (Utils.path_utils.DEFAULT_GENERATOR)
        run = d{d_model}_cb{codebook}_st{stride}_ar{n_embd}_l{n_layer}_h{n_head}
              _trainseed{s}_delta{d}_genseed{s}_wmseed{s}

stride is auto-determined from window: 24 -> 2, else -> 4.
--exp is auto-derived from --config_file if not specified.

Seeding: every number this file reports is seeded from a single --seed (default 12345,
the same value train.py/generate.py use). Iteration i of each N_QUALITY_ITER loop runs
under seed+i, and --mode zscore bootstraps under --seed. Both the seed and the flags that
move the numbers are recorded in the CSV file name.

--gpu N selects the CUDA device for the quality metrics and is recorded, with the device
name and compute capability, in run_config_{mode}.json.

--detect_key <k> (zscore mode) additionally reads the wrong-key detect CSVs that
`generate.py --detect_all --detect_key <k>` wrote and emits `<method>_wrongkey<k>` rows.
Watermarked samples scored under a key they were not generated with must give z ~ 0; that
is the key-specificity control. Without the flag nothing changes, down to the file name.

Usage:
  python -u cli/evaluate.py --config_file configs/lvqvae/etth.yaml --dataset etth \
      --window 64 --delta 10 --mode zscore
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss
from Utils.device_utils import cuda_index, restrict_tensorflow_to_gpu, select_torch_device
from Utils.io_utils import seed_all
from Utils.path_utils import (
    DEFAULT_EVAL_SEED,
    DEFAULT_GENERATOR,
    DEFAULT_GEN_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_WATERMARK_SEED,
    derive_exp_name,
    derive_run_name,
    get_output_dir,
    get_results_dir,
    stride_for_window,
    torch_provenance,
    write_run_config,
)
from Utils.greenlist import ATTACK_LABELS, detect_dir_name, resolve_detect_key
from Utils.watermark_metrics import get_zscore

_disc_fn = None
_pred_fn = None

def get_disc_fn(gpu=None):
    """The discriminative-score function, importing TensorFlow on first use.

    gpu: CUDA ordinal the run selected, or None to leave TensorFlow's device choice alone.
    TensorFlow is pinned to that GPU here because this is where it is first imported and
    its device list can only be narrowed before any op has run.
    """
    global _disc_fn
    if _disc_fn is None:
        import tensorflow as tf
        restrict_tensorflow_to_gpu(gpu)
        tf.compat.v1.disable_eager_execution()
        from Utils.discriminative_metric import discriminative_score_metrics
        _disc_fn = discriminative_score_metrics
    return _disc_fn

def get_pred_fn(gpu=None):
    """The predictive-score function. gpu: see :func:`get_disc_fn`."""
    global _pred_fn
    if _pred_fn is None:
        restrict_tensorflow_to_gpu(gpu)
        from Utils.predictive_metric import predictive_score_metrics
        _pred_fn = predictive_score_metrics
    return _pred_fn

GT_PREFIX = {
    "stocks": "stock", "etth": "etth", "energy": "energy", "fmri": "fMRI",
}
N_QUALITY_ITER = 5
METHODS = ["LVQMark"]

# get_zscore draws this many green-fractions per bootstrap replicate, without replacement.
# The split-half null pools must be at least this large.
ZSCORE_NUM_SAMPLE = 1000


def eval_full_quality(ori_data: np.ndarray, fake_data: np.ndarray, seed: int,
                      deterministic: bool = False, device=None) -> dict:
    """Four quality metrics, each averaged over N_QUALITY_ITER independently seeded repeats.

    ori_data / fake_data: (N, T, D) float arrays in the normalised [0, 1] scale.
    seed: int — iteration ``i`` runs under ``seed + i``. deterministic: bool, see --deterministic.
    device: the torch device --gpu selected; Context_FID runs there and TensorFlow is pinned
    to the same card. None means the process's current device.
    Returns {metric_{mean,std}: float}.

    The ± this returns is the spread of the metric *estimator* over repeats of the same two
    files (TS2Vec is retrained, the correlational subsample is redrawn, the TF discriminator
    and predictor are re-initialised and retrained). It is not a spread over generation or
    training seeds, of which there is one each.

    Seeding is applied immediately before each metric call rather than once per iteration,
    so that no metric's result depends on how much RNG the metric before it happened to
    consume — adding or removing a metric cannot move the others. Unseeded, the sign of the
    watermarked-vs-clean Context-FID gap is not stable between two runs on identical files.
    """
    n = min(len(ori_data), len(fake_data))
    o, f = ori_data[:n], fake_data[:n]

    cfids = []
    for i in range(N_QUALITY_ITER):
        seed_all(seed + i, cudnn_deterministic=deterministic)
        cfids.append(Context_FID(o, f, device=device))
        print(f"    Iter {i}: CFID={cfids[-1]:.4f}", flush=True)

    x_real, x_fake = torch.from_numpy(o), torch.from_numpy(f)
    size = n // N_QUALITY_ITER
    corrs = []
    for i in range(N_QUALITY_ITER):
        seed_all(seed + i, cudnn_deterministic=deterministic)
        ri = np.random.randint(0, n, size=size)
        fi = np.random.randint(0, n, size=size)
        corrs.append(CrossCorrelLoss(x_real[ri], name="c").compute(x_fake[fi]).item())
        print(f"    Iter {i}: Corr={corrs[-1]:.4f}", flush=True)
    torch.cuda.empty_cache()

    # Import TensorFlow *before* the first seed_all of this loop: seed_all only seeds a
    # TF that is already imported, and the discriminator's graph is built inside the call.
    gpu = cuda_index(device)
    disc_fn, pred_fn = get_disc_fn(gpu), get_pred_fn(gpu)
    discs, preds = [], []
    for i in range(N_QUALITY_ITER):
        seed_all(seed + i, cudnn_deterministic=deterministic)
        d, _, _ = disc_fn(o, f)
        discs.append(d)
        seed_all(seed + i, cudnn_deterministic=deterministic)
        preds.append(pred_fn(o, f))
        print(f"    Iter {i}: Disc={discs[-1]:.4f}, Pred={preds[-1]:.4f}", flush=True)

    return {
        "ContextFID_mean": np.mean(cfids), "ContextFID_std": np.std(cfids),
        "Correlational_mean": np.mean(corrs), "Correlational_std": np.std(corrs),
        "Discriminative_mean": np.mean(discs), "Discriminative_std": np.std(discs),
        "Predictive_mean": np.mean(preds), "Predictive_std": np.std(preds),
    }


def split_half_null(wo_data: np.ndarray, seed: int):
    """Split a no-watermark green-fraction pool into two disjoint halves.

    wo_data: (N, 1) float — one green fraction per generated window. seed: int.
    Returns (half_a, half_b), shapes (N//2, 1) and (N - N//2, 1).

    Half A and half B are two disjoint samples of unwatermarked windows, so scoring one
    against the other gives a real null: half A defines the reference mean/std and half B is
    bootstrapped against it. The z and FPR that come out are the detector's false-positive
    behaviour on clean data.

    The shuffle draws from a local ``RandomState`` rather than the global numpy stream that
    ``get_zscore``'s bootstrap consumes, so the clean row does not depend on how many other
    rows were computed first.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(wo_data))
    cut = len(wo_data) // 2
    return wo_data[order[:cut]], wo_data[order[cut:]]


def wrongkey_zscore_rows(base: str, method: str, detect_key: int, detect_suffix: str,
                         seed: int) -> list:
    """Key-specificity rows: the same windows scored with a green list from another key.

    base: str, ``outputs/{generator}/{dataset}/{run}/{window}``. method: e.g. "LVQMark".
    detect_key: int k, the wrong key. detect_suffix: "" (nothing to do) or "-key{k}", the
    directory suffix ``generate.py --detect_all --detect_key k`` wrote. seed: int, used for
    the split-half null only. Returns a list of row dicts, empty when detect_suffix is "".

    Every row is ``{method}_wrongkey{k}[_{attack}]``. The null pool is the *wrong-key* NoWM
    pool, not the correct-key one: under a different key the green fraction of unwatermarked
    windows has its own distribution, and comparing across keys would measure the difference
    between two green lists instead of the presence of a watermark. A row whose CSVs are
    absent is skipped rather than emitted empty, so a partial detect run is visible.

    Reads only; consumes the global numpy stream through ``get_zscore`` exactly as the rows
    above it do, which is why the caller runs it last.
    """
    if not detect_suffix:
        return []
    label = f"{method}_wrongkey{detect_key}"

    def _read(condition):
        path = os.path.join(base, method, detect_dir_name(condition, detect_suffix),
                            f"{method}.csv")
        return pd.read_csv(path, header=None).values if os.path.exists(path) else None

    wo_data = _read("detect-wo")
    w_data = _read("detect")
    if wo_data is None or w_data is None:
        print(f"  [SKIP] {label}: run `generate.py --detect_all --detect_key "
              f"{detect_key}` first (missing {base}/{method}/detect[-wo]{detect_suffix})",
              flush=True)
        return []

    rows = [{"method": label}]
    wm_z, wm_std, wm_tpr = get_zscore(method, wo_data, w_data)
    null_a, null_b = split_half_null(wo_data, seed)
    if len(null_a) < ZSCORE_NUM_SAMPLE:
        print(f"  [WARN] {label}: split-half null pools hold {len(null_a)} rows, fewer "
              f"than the {ZSCORE_NUM_SAMPLE} draws get_zscore takes without replacement.",
              flush=True)
    nw_z, nw_std, nw_tpr = get_zscore(method, null_a, null_b)
    rows[0].update({
        "WM_Zscore": wm_z, "WM_Zscore_std": wm_std, "WM_TPR": wm_tpr,
        "NoWM_Zscore": nw_z, "NoWM_Zscore_std": nw_std, "NoWM_TPR": nw_tpr,
    })
    print(f"  WRONG KEY {detect_key}: WM z={wm_z:.2f} TPR={wm_tpr:.2f} | "
          f"NoWM(split-half) z={nw_z:.2f} FPR={nw_tpr:.2f}   (expect z ~ 0 for both)",
          flush=True)

    for attack in ATTACK_LABELS:
        atk_data, atk_wo_data = _read(attack), _read(f"{attack}-wo")
        if atk_data is None and atk_wo_data is None:
            continue
        atk_row = {"method": f"{label}_{attack}"}
        if atk_data is not None:
            z, std, tpr = get_zscore(method, wo_data, atk_data)
            atk_row.update({"WM_Zscore": z, "WM_Zscore_std": std, "WM_TPR": tpr})
        if atk_wo_data is not None:
            z, std, tpr = get_zscore(method, wo_data, atk_wo_data)
            atk_row.update({"NoWM_Zscore": z, "NoWM_Zscore_std": std, "NoWM_TPR": tpr})
        rows.append(atk_row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp", default=None)
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--mode", default="full", choices=["zscore", "quick", "full"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device for the quality metrics (Context-FID/TS2Vec and "
                             "the TensorFlow discriminative/predictive scores). Set "
                             "CUDA_DEVICE_ORDER=PCI_BUS_ID so N matches nvidia-smi. The "
                             "selected device is recorded in run_config_{mode}.json; "
                             "--mode zscore is pure numpy and uses no GPU")
    parser.add_argument("--deterministic", action="store_true",
                        help="turn on cudnn.deterministic for the quality metrics. "
                             "Seeding alone does NOT make them reproducible on GPU, "
                             "because TS2Vec's backward pass is non-deterministic; with "
                             "this flag two runs of the same command on the same GPU give "
                             "identical numbers, at no measurable cost. OFF by default; "
                             "when on, '_det' is appended to the CSV name.")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED,
                        help="evaluation seed: quality iteration i runs under seed+i, and "
                             "--mode zscore bootstraps under it (default: 12345)")
    # Path-key components. They must match the run being evaluated; the defaults match the
    # defaults of train.py/generate.py, so all-default arguments address an all-default run.
    parser.add_argument("--train_seed", type=int, default=DEFAULT_TRAIN_SEED,
                        help="train.py --seed of the run being evaluated (path key only)")
    parser.add_argument("--gen_seed", type=int, default=DEFAULT_GEN_SEED,
                        help="generate.py --seed of the run being evaluated (path key only)")
    parser.add_argument("--watermark_seed", type=int, default=DEFAULT_WATERMARK_SEED,
                        help="generate.py --watermark_seed of the run (path key only)")
    parser.add_argument("--detect_key", type=int, default=None,
                        help="also report the wrong-key control from "
                             "<method>/detect-key<k>/ (written by generate.py --detect_all "
                             "--detect_key <k>): the same watermarked windows scored with a "
                             "green list built from key <k>, expected z ~ 0. Adds "
                             "<method>_wrongkey<k> rows and _key<k> to the CSV name; "
                             "changes no existing row. --mode zscore only")
    args = parser.parse_args()

    # One device for the whole run: torch's current device, and the same card for
    # TensorFlow once it is imported (Utils/device_utils).
    device = select_torch_device(args.gpu)

    # "" unless --detect_key was passed — the suffix generate.py used for the same control,
    # from the same function, so the two sides cannot drift apart.
    detect_key, detect_suffix = resolve_detect_key(args.detect_key, args.watermark_seed)
    if args.detect_key is not None and args.mode != "zscore":
        print(f"[WARN] --detect_key only affects --mode zscore; ignoring it in "
              f"--mode {args.mode}.", flush=True)
        detect_suffix = ""
    # Directory suffix is "-key<k>" (generate.py's grammar); file names use "_key<k>",
    # matching the "_seed<s>" / "_det" tags already in them.
    key_tag = f"_key{detect_key}" if detect_suffix else ""

    stride = stride_for_window(args.window)
    config = None
    if args.exp is None:
        if args.config_file is None:
            raise ValueError("--exp or --config_file required")
        from Utils.io_utils import load_yaml_config
        config = load_yaml_config(args.config_file)
        args.exp = derive_exp_name(config, stride, args.train_seed)

    run = derive_run_name(args.exp, args.delta, args.gen_seed, args.watermark_seed)
    # generator is the first level of every artifact tree (Utils/path_utils); passed
    # explicitly so it appears in the resolved run_config.json, not as a hidden default.
    base = get_output_dir(args.dataset, run, args.window, generator=DEFAULT_GENERATOR)
    results_dir = get_results_dir(args.dataset, run, generator=DEFAULT_GENERATOR)
    os.makedirs(results_dir, exist_ok=True)

    prefix = GT_PREFIX[args.dataset]
    gt_path = os.path.join(base, "samples", f"{prefix}_norm_truth_{args.window}_test.npy")
    ori_data = np.load(gt_path) if os.path.exists(gt_path) else None
    if ori_data is None:
        print(f"[SKIP] GT not found: {gt_path}")
        return

    print(f"\n{'='*60}")
    print(f"  {args.dataset}/{run} w={args.window} [{args.mode}] seed={args.seed}"
          f"{' deterministic' if args.deterministic else ''} device={device}")
    print(f"{'='*60}", flush=True)

    write_run_config(results_dir, {
        "entry_point": "evaluate.py",
        "argv": sys.argv,
        "args": vars(args),
        "derived": {"generator": DEFAULT_GENERATOR,
                    "run": run, "stride": stride, "output_dir": base,
                    "results_dir": results_dir, "n_quality_iter": N_QUALITY_ITER,
                    "device": str(device)},
        "config": config,
        "provenance": torch_provenance(args.gpu),
    }, filename=f"run_config_{args.mode}{key_tag}.json")

    rows = []

    # NoWM
    nowm_path = os.path.join(base, "sample", "generated_norm.npy")
    nowm_data = np.load(nowm_path) if os.path.exists(nowm_path) else None
    nowm_row = {"method": "NoWatermark"}
    if args.mode == "full" and nowm_data is not None:
        nowm_row.update(eval_full_quality(ori_data, nowm_data, seed=args.seed,
                                          deterministic=args.deterministic, device=device))
    elif args.mode == "quick" and nowm_data is not None:
        n = min(len(ori_data), len(nowm_data))
        seed_all(args.seed, cudnn_deterministic=args.deterministic)
        cfid = Context_FID(ori_data[:n], nowm_data[:n], device=device)
        nowm_row["CFID"] = cfid
        print(f"  NoWM: CFID={cfid:.4f}", flush=True)
    rows.append(nowm_row)

    # Methods
    for method in METHODS:
        print(f"\n  >> {method}")
        method_row = {"method": method}
        wrongkey_rows = []

        if args.mode in ("quick", "full"):
            fake_path = os.path.join(base, method, "sample", "generated_norm.npy")
            fake_data = np.load(fake_path) if os.path.exists(fake_path) else None
            if fake_data is not None:
                if args.mode == "full":
                    method_row.update(eval_full_quality(
                        ori_data, fake_data, seed=args.seed,
                        deterministic=args.deterministic, device=device))
                elif args.mode == "quick":
                    n = min(len(ori_data), len(fake_data))
                    seed_all(args.seed, cudnn_deterministic=args.deterministic)
                    cfid = Context_FID(ori_data[:n], fake_data[:n], device=device)
                    method_row["CFID"] = cfid
                    print(f"  CFID={cfid:.4f}", flush=True)

        if args.mode == "zscore":
            # One seed for the whole bootstrap: get_zscore draws 100 replicates per row from
            # the global numpy stream, in row order. Seeding here (and nowhere else in the
            # loop) is the RNG contract of this mode.
            seed_all(args.seed, seed_tf=False)
            wo_path = os.path.join(base, method, "detect-wo", f"{method}.csv")
            w_path = os.path.join(base, method, "detect", f"{method}.csv")
            wo_data = pd.read_csv(wo_path, header=None).values if os.path.exists(wo_path) else None
            w_data = pd.read_csv(w_path, header=None).values if os.path.exists(w_path) else None

            if wo_data is not None and w_data is not None:
                wm_z, wm_std, wm_tpr = get_zscore(method, wo_data, w_data)
                # Independent split-half null, not wo_data against itself — see split_half_null.
                null_a, null_b = split_half_null(wo_data, args.seed)
                if len(null_a) < ZSCORE_NUM_SAMPLE:
                    print(f"  [WARN] split-half null pools hold {len(null_a)} rows, fewer "
                          f"than the {ZSCORE_NUM_SAMPLE} draws get_zscore takes without "
                          f"replacement; it will raise. Generate more samples.", flush=True)
                nw_z, nw_std, nw_tpr = get_zscore(method, null_a, null_b)
                method_row.update({
                    "WM_Zscore": wm_z, "WM_Zscore_std": wm_std, "WM_TPR": wm_tpr,
                    "NoWM_Zscore": nw_z, "NoWM_Zscore_std": nw_std, "NoWM_TPR": nw_tpr,
                })
                print(f"  WM z={wm_z:.2f} TPR={wm_tpr:.2f} | "
                      f"NoWM(split-half) z={nw_z:.2f} FPR={nw_tpr:.2f}", flush=True)

            for attack in ATTACK_LABELS:
                atk_row = {"method": f"{method}_{attack}"}
                atk_csv = os.path.join(base, method, attack, f"{method}.csv")
                atk_wo_csv = os.path.join(base, method, f"{attack}-wo", f"{method}.csv")
                atk_data = pd.read_csv(atk_csv, header=None).values if os.path.exists(atk_csv) else None
                atk_wo_data = pd.read_csv(atk_wo_csv, header=None).values if os.path.exists(atk_wo_csv) else None
                if wo_data is not None and atk_data is not None:
                    wm_z, wm_std, wm_tpr = get_zscore(method, wo_data, atk_data)
                    atk_row.update({"WM_Zscore": wm_z, "WM_Zscore_std": wm_std, "WM_TPR": wm_tpr})
                if wo_data is not None and atk_wo_data is not None:
                    nw_z, nw_std, nw_tpr = get_zscore(method, wo_data, atk_wo_data)
                    atk_row.update({"NoWM_Zscore": nw_z, "NoWM_Zscore_std": nw_std, "NoWM_TPR": nw_tpr})
                rows.append(atk_row)

            # Wrong-key control, computed after every row above: get_zscore draws from the
            # global numpy stream in call order, so the position of a row in this loop is
            # part of what determines its numbers.
            wrongkey_rows = wrongkey_zscore_rows(base, method, detect_key, detect_suffix,
                                                 args.seed)
        rows.append(method_row)
        rows.extend(wrongkey_rows)

    df = pd.DataFrame(rows)
    # The file name records everything that moves the numbers in it.
    tag = f"seed{args.seed}" + ("_det" if args.deterministic else "") + key_tag
    if args.mode == "zscore":
        csv_name = f"w{args.window}_zscore_{tag}.csv"
    elif args.mode == "quick":
        csv_name = f"w{args.window}_quality_quick_{tag}.csv"
    elif args.mode == "full":
        csv_name = f"w{args.window}_quality_full_{tag}.csv"
    csv_path = os.path.join(results_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

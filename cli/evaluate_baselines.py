"""Evaluate a baseline generator: quality metrics and detection z-scores.

The baseline twin of ``evaluate.py``. It has its own loop -- the baseline entry points do
not import another entry point's code -- and calls exactly the same ``Utils/`` functions with
exactly the same seeding rule, so an L-VQVAE number and a baseline number are produced by
one implementation of every metric:

  Context_FID, CrossCorrelLoss, discriminative_score_metrics, predictive_score_metrics,
  Utils.watermark_metrics.get_zscore, Utils.io_utils.seed_all.

  outputs:  outputs/{generator}/{dataset}/{run}/{window}/
  results:  results/{generator}/{dataset}/{run}/w{window}_{mode}_seed{s}[_det][_key{k}].csv

Seeding. One ``--seed`` (default 12345): quality iteration ``i`` runs under ``seed + i`` and
``--mode zscore`` bootstraps under ``--seed``. Both the seed and the flags that move the
numbers are in the CSV name.

Reading a TR row. ``get_zscore`` computes TR's z by the paper's TR definition,
|mean_NoWM - mean| / sigma_NoWM with no /sqrt(n), while every other method uses the pooled
(mean - mean_NoWM) / (sigma_NoWM / sqrt(1000)). The column is headed "z" for all of them;
any table containing TR must carry that footnote.

Usage:
  python -u cli/evaluate_baselines.py --generator sdformer \
      --config_file configs/sdformer/etth.yaml --dataset etth --window 64 --mode zscore
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Utils.baseline_utils import (
    DEFAULT_LATENT_WATERMARK_SEED,
    GENERATORS,
    GT_PREFIX,
    derive_exp_name,
    derive_run_name,
    methods_for,
    split_half_null,
    zscore_method_key,
)
from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss
from Utils.device_utils import cuda_index, restrict_tensorflow_to_gpu, select_torch_device
from Utils.greenlist import ATTACK_LABELS, detect_dir_name, resolve_detect_key
from Utils.io_utils import load_yaml_config, seed_all
from Utils.path_utils import (
    DEFAULT_EVAL_SEED,
    DEFAULT_GEN_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_WATERMARK_SEED,
    get_output_dir,
    get_results_dir,
    torch_provenance,
    write_run_config,
)
from Utils.watermark_metrics import get_zscore

N_QUALITY_ITER = 5

# get_zscore draws this many values per bootstrap replicate, without replacement, so a pool
# smaller than this makes it raise. The split-half null halves the pool first.
ZSCORE_NUM_SAMPLE = 1000

SAMPLE_NORM = "generated_norm.npy"

_disc_fn = None
_pred_fn = None


def get_disc_fn(gpu=None):
    """The discriminative-score function, importing TensorFlow on first use.

    gpu: CUDA ordinal the run selected, or None to leave TensorFlow's device choice alone.
    TensorFlow is pinned to that GPU here because this is where it is first imported and its
    device list can only be narrowed before any op has run.
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


def eval_full_quality(ori_data, fake_data, seed, deterministic=False, device=None):
    """Four quality metrics, each averaged over N_QUALITY_ITER independently seeded repeats.

    ori_data / fake_data: (N, T, D) float arrays on the normalised [0, 1] scale.
    seed: int -- iteration ``i`` runs under ``seed + i``. deterministic: bool, see
    --deterministic. device: the torch device --gpu selected; Context_FID runs there and
    TensorFlow is pinned to the same card. None means the process's current device.
    Returns {metric_{mean,std}: float}.

    The +/- is the spread of the metric *estimator* over repeats of the same two files, not
    a spread over generation or training seeds, of which there is one each. Seeding is
    applied immediately before each metric call rather than once per iteration, so no
    metric's result depends on how much RNG the metric before it consumed.
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

    # Import TensorFlow *before* the first seed_all of this loop: seed_all only seeds a TF
    # that is already imported, and the discriminator's graph is built inside the call.
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


def read_detect_csv(base, method, condition, detect_suffix=""):
    """One condition's detect CSV as an (N, 1) array, or None if it is absent.

    base: ``outputs/{generator}/{dataset}/{run}/{window}``. method: e.g. "TimeWak".
    condition: "detect" | "detect-wo" | "<attack>-<factor>" | "<attack>-<factor>-wo".
    detect_suffix: "" or "-key{k}".
    """
    path = os.path.join(base, method, detect_dir_name(condition, detect_suffix),
                        f"{method}.csv")
    return pd.read_csv(path, header=None).values if os.path.exists(path) else None


def zscore_rows(base, method, seed, detect_suffix="", label=None):
    """Detection rows for one method. Returns (clean_row_updates, attack_rows).

    base / method / detect_suffix: see :func:`read_detect_csv`. seed: int, used for the
    split-half null only. label: the row name for the attack rows (defaults to ``method``).

    ``get_zscore`` draws 100 bootstrap replicates per call from the *global* numpy stream in
    call order, so the order of the calls below is part of the numbers. The clean row's null
    is a split half of the no-watermark pool, never the pool against itself: that would make
    z = 0 and FPR = 0 true by construction.
    """
    label = label or method
    zkey = zscore_method_key(method)
    wo_data = read_detect_csv(base, method, "detect-wo", detect_suffix)
    w_data = read_detect_csv(base, method, "detect", detect_suffix)
    updates, rows = {}, []
    if wo_data is None or w_data is None:
        want = detect_dir_name("detect", detect_suffix)
        want_wo = detect_dir_name("detect-wo", detect_suffix)
        print(f"  [SKIP] {label}: no {want}/ or {want_wo}/ CSVs under {base}/{method}; "
              f"run generate_baselines.py --detect_all"
              + (f" --detect_key {detect_suffix[len('-key'):]}" if detect_suffix else "")
              + " first", flush=True)
        return updates, rows

    wm_z, wm_std, wm_tpr = get_zscore(zkey, wo_data, w_data)
    null_a, null_b = split_half_null(wo_data, seed)
    if len(null_a) < ZSCORE_NUM_SAMPLE:
        print(f"  [WARN] {label}: split-half null pools hold {len(null_a)} rows, fewer "
              f"than the {ZSCORE_NUM_SAMPLE} draws get_zscore takes without replacement; "
              f"it will raise. Generate more samples.", flush=True)
    nw_z, nw_std, nw_tpr = get_zscore(zkey, null_a, null_b)
    updates = {"WM_Zscore": wm_z, "WM_Zscore_std": wm_std, "WM_TPR": wm_tpr,
               "NoWM_Zscore": nw_z, "NoWM_Zscore_std": nw_std, "NoWM_TPR": nw_tpr}
    print(f"  {label}: WM z={wm_z:.2f} TPR={wm_tpr:.2f} | "
          f"NoWM(split-half) z={nw_z:.2f} FPR={nw_tpr:.2f}", flush=True)

    for attack in ATTACK_LABELS:
        atk_data = read_detect_csv(base, method, attack, detect_suffix)
        atk_wo_data = read_detect_csv(base, method, f"{attack}-wo", detect_suffix)
        if atk_data is None and atk_wo_data is None:
            continue
        row = {"method": f"{label}_{attack}"}
        if atk_data is not None:
            z, std, tpr = get_zscore(zkey, wo_data, atk_data)
            row.update({"WM_Zscore": z, "WM_Zscore_std": std, "WM_TPR": tpr})
        if atk_wo_data is not None:
            z, std, tpr = get_zscore(zkey, wo_data, atk_wo_data)
            row.update({"NoWM_Zscore": z, "NoWM_Zscore_std": std, "NoWM_TPR": tpr})
        rows.append(row)
    return updates, rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generator", required=True, choices=list(GENERATORS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp", default=None)
    parser.add_argument("--config_file", default=None,
                        help="required unless --exp is given; the exp key is derived from it")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--mode", default="full", choices=["zscore", "quick", "full"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device for the quality metrics (Context-FID/TS2Vec and "
                             "the TensorFlow discriminative/predictive scores). Set "
                             "CUDA_DEVICE_ORDER=PCI_BUS_ID so N matches nvidia-smi. The "
                             "selected device is recorded in run_config_{mode}.json; "
                             "--mode zscore is pure numpy and uses no GPU")
    parser.add_argument("--methods", type=str, default=None,
                        help="comma-separated methods (default: all of the generator's)")
    parser.add_argument("--deterministic", action="store_true",
                        help="turn on cudnn.deterministic for the quality metrics. Seeding "
                             "alone does NOT make them reproducible on GPU, because "
                             "TS2Vec's backward pass is non-deterministic. OFF by default; "
                             "when on, '_det' is appended to the CSV name")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED,
                        help="evaluation seed: quality iteration i runs under seed+i, and "
                             "--mode zscore bootstraps under it (default: 12345)")

    # Path-key components. They must match the run being evaluated; the defaults match the
    # defaults of train_baselines.py / generate_baselines.py.
    parser.add_argument("--train_seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--gen_seed", type=int, default=DEFAULT_GEN_SEED)
    parser.add_argument("--watermark_seed", type=int, default=None,
                        help="default: 12345 for sdformer, 217 for diffusionts")
    parser.add_argument("--delta", type=float, default=10.0,
                        help="watermark delta of the run (sdformer run key)")
    parser.add_argument("--sampling_timesteps", type=int, default=None,
                        help="sampling_timesteps of the run (diffusionts run key); "
                             "default: the config's value")
    parser.add_argument("--detect_key", type=int, default=None,
                        help="also report the wrong-key control from "
                             "<method>/detect-key<k>/, expected z ~ 0. Adds "
                             "<method>_wrongkey<k> rows and _key<k> to the CSV name; "
                             "changes no existing row. --mode zscore only")
    args = parser.parse_args()
    if args.watermark_seed is None:
        args.watermark_seed = (DEFAULT_WATERMARK_SEED if args.generator == "sdformer"
                               else DEFAULT_LATENT_WATERMARK_SEED)
    return args


def main():
    args = parse_args()

    # One device for the whole run: torch's current device, and the same card for
    # TensorFlow once it is imported (Utils/device_utils).
    device = select_torch_device(args.gpu)

    detect_key, detect_suffix = resolve_detect_key(args.detect_key, args.watermark_seed)
    if args.detect_key is not None and args.mode != "zscore":
        print(f"[WARN] --detect_key only affects --mode zscore; ignoring it in "
              f"--mode {args.mode}.", flush=True)
        detect_suffix = ""
    key_tag = f"_key{detect_key}" if detect_suffix else ""

    config = None
    if args.config_file is not None:
        config = load_yaml_config(args.config_file)
    if args.exp is None:
        if config is None:
            raise SystemExit("--exp or --config_file required")
        args.exp = derive_exp_name(args.generator, config, args.train_seed)

    sampling_timesteps = args.sampling_timesteps
    if args.generator == "diffusionts" and sampling_timesteps is None:
        if config is None:
            raise SystemExit("--sampling_timesteps or --config_file required for "
                             "diffusionts: it is part of the run key")
        sampling_timesteps = config["model"]["params"]["sampling_timesteps"]

    run = derive_run_name(
        args.generator, args.exp, gen_seed=args.gen_seed,
        watermark_seed=args.watermark_seed,
        delta=args.delta if args.generator == "sdformer" else None,
        sampling_timesteps=sampling_timesteps if args.generator == "diffusionts" else None)

    base = get_output_dir(args.dataset, run, args.window, generator=args.generator)
    results_dir = get_results_dir(args.dataset, run, generator=args.generator)
    os.makedirs(results_dir, exist_ok=True)

    prefix = GT_PREFIX[args.dataset]
    gt_path = os.path.join(base, "samples",
                           f"{prefix}_norm_truth_{args.window}_test.npy")
    if not os.path.exists(gt_path):
        print(f"[SKIP] GT not found: {gt_path}")
        return
    ori_data = np.load(gt_path)

    methods = ([m.strip() for m in args.methods.split(",")] if args.methods
               else list(methods_for(args.generator)))
    unknown = [m for m in methods if m not in methods_for(args.generator)]
    if unknown:
        raise SystemExit(f"--methods {unknown} not available on {args.generator}")

    print(f"\n{'='*60}")
    print(f"  {args.generator}/{args.dataset}/{run} w={args.window} [{args.mode}] "
          f"seed={args.seed}{' deterministic' if args.deterministic else ''} "
          f"device={device}")
    print(f"{'='*60}", flush=True)

    write_run_config(results_dir, {
        "entry_point": "evaluate_baselines.py", "argv": sys.argv, "args": vars(args),
        "derived": {"generator": args.generator, "exp": args.exp, "run": run,
                    "output_dir": base, "results_dir": results_dir,
                    "methods": methods, "n_quality_iter": N_QUALITY_ITER,
                    "sampling_timesteps": sampling_timesteps,
                    "device": str(device)},
        "config": config, "provenance": torch_provenance(args.gpu),
    }, filename=f"run_config_{args.mode}{key_tag}.json")

    rows = []

    nowm_path = os.path.join(base, "sample", SAMPLE_NORM)
    nowm_data = np.load(nowm_path) if os.path.exists(nowm_path) else None
    nowm_row = {"method": "NoWatermark"}
    if nowm_data is not None and args.mode == "full":
        nowm_row.update(eval_full_quality(ori_data, nowm_data, seed=args.seed,
                                          deterministic=args.deterministic, device=device))
    elif nowm_data is not None and args.mode == "quick":
        n = min(len(ori_data), len(nowm_data))
        seed_all(args.seed, cudnn_deterministic=args.deterministic)
        cfid = Context_FID(ori_data[:n], nowm_data[:n], device=device)
        nowm_row["CFID"] = cfid
        print(f"  NoWM: CFID={cfid:.4f}", flush=True)
    rows.append(nowm_row)

    for method in methods:
        print(f"\n  >> {method}")
        method_row = {"method": method}
        wrongkey_rows = []

        if args.mode in ("quick", "full"):
            fake_path = os.path.join(base, method, "sample", SAMPLE_NORM)
            fake_data = np.load(fake_path) if os.path.exists(fake_path) else None
            if fake_data is None:
                print(f"  [SKIP] {fake_path} not found", flush=True)
            elif args.mode == "full":
                method_row.update(eval_full_quality(
                    ori_data, fake_data, seed=args.seed,
                    deterministic=args.deterministic, device=device))
            else:
                n = min(len(ori_data), len(fake_data))
                seed_all(args.seed, cudnn_deterministic=args.deterministic)
                cfid = Context_FID(ori_data[:n], fake_data[:n], device=device)
                method_row["CFID"] = cfid
                print(f"  CFID={cfid:.4f}", flush=True)

        if args.mode == "zscore":
            # One seed for the whole bootstrap: get_zscore draws 100 replicates per row
            # from the global numpy stream, in row order. Seeding here and nowhere else in
            # the loop is the RNG contract of this mode.
            seed_all(args.seed, seed_tf=False)
            updates, attack_rows = zscore_rows(base, method, args.seed)
            method_row.update(updates)
            rows.extend(attack_rows)

            # Wrong-key control last: get_zscore draws from the global numpy stream in
            # call order, so the position of a row in this loop is part of what
            # determines its numbers.
            if detect_suffix:
                label = f"{method}_wrongkey{detect_key}"
                wk_updates, wk_attack_rows = zscore_rows(
                    base, method, args.seed, detect_suffix=detect_suffix, label=label)
                if wk_updates:
                    wrongkey_rows = [dict({"method": label}, **wk_updates)]
                    wrongkey_rows.extend(wk_attack_rows)
                    print("  (expect z ~ 0 for both columns under a wrong key)",
                          flush=True)

        rows.append(method_row)
        rows.extend(wrongkey_rows)

    tag = f"seed{args.seed}" + ("_det" if args.deterministic else "") + key_tag
    csv_name = {"zscore": f"w{args.window}_zscore_{tag}.csv",
                "quick": f"w{args.window}_quality_quick_{tag}.csv",
                "full": f"w{args.window}_quality_full_{tag}.csv"}[args.mode]
    csv_path = os.path.join(results_dir, csv_name)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

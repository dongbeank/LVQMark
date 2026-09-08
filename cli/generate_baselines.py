"""Generate + detect for a baseline generator: SDformer or Diffusion-TS.

The baseline twin of ``generate.py``. It shares ``Models/``, ``Utils/``, ``Data/`` and the
artifact trees with the L-VQVAE entry points and imports no code from them; the watermark
protocol, the attack grid and the scoring all come from ``Utils/`` so that every method in
the paper is scored by one implementation.

  outputs/{generator}/{dataset}/{run}/{window}/
      samples/                              ground truth dumped by the dataloader
      sample/generated{,_norm}.npy          the no-watermark pool for this run
      {wm}/sample/generated{,_norm}.npy     the watermarked pool
      {wm}/sample/{gt_patch,watermarking_mask}.npy    (TR only: its key material)
      {wm}/{detect,detect-wo,<atk>-<f>,<atk>-<f>-wo}[-key<k>]/{wm}.csv
      run_config*.json                      the resolved config + torch/GPU provenance

Two watermark families, one CSV shape:
  sdformer     LVQMark, the green-list method from Utils.greenlist,
               embedded in the AR sampler and recovered by the robust conv encoder. One
               green fraction per window.
  diffusionts  the initial-latent family (TimeWak / TR / GS) inside
               Diffusion_TS.generate_mts, recovered by DDIM/BDIA inversion and scored by
               Utils.watermark_metrics.eval_{TimeWak,TR,GS}. One statistic per window.

Seeds. ``--seed`` (default 12345) drives sampling and is seeded ONCE at process start.
``--watermark_seed`` is the watermark key and is part of the run path: it defaults to 12345
for the green-list family and to 217 for the latent family, which is the key the latent
detectors in ``Utils.watermark_metrics`` expect.

Usage:
  python -u cli/generate_baselines.py --generator sdformer \
      --config_file configs/sdformer/etth.yaml --dataset etth --window_size 64 --gpu 2
  python -u cli/generate_baselines.py --generator sdformer ... --watermark --wm_method LVQMark
  python -u cli/generate_baselines.py --generator diffusionts ... --watermark --wm_method TimeWak
  python -u cli/generate_baselines.py --generator ... --detect_all [--detect_key 999]
"""

import argparse
import os
import sys

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Data.build_dataloader import build_dataloader
from Models.model_utils import unnormalize_to_zero_to_one
from Utils.baseline_utils import (
    DEFAULT_LATENT_WATERMARK_SEED,
    GENERATORS,
    KEYED_LATENT_WM_METHODS,
    LATENT_SCORER_CSV_STEM,
    build_tr_key_patches,
    derive_exp_name,
    derive_run_name,
    methods_for,
    resolve_checkpoint,
)
from Utils.greenlist import (
    ATTACK_GRID,
    WM_METHODS,
    compute_valid_token_range,
    create_fixed_watermark_table,
    detect_dir_name,
    green_fractions,
    resolve_detect_key,
    resolve_warmup_positions,
    resolve_wm_switches,
    warmup_delta_schedule,
)
from Utils.io_utils import instantiate_from_config, load_yaml_config, seed_everything
from Utils.metric_utils import write_csv
from Utils.path_utils import (
    DEFAULT_GEN_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_WATERMARK_SEED,
    OUTPUT_BASE,
    get_ckpt_dir,
    get_output_dir,
    torch_provenance,
    write_run_config,
)

NUM_SAMPLES = 10000     # production pool size, every pipeline in this repo
SIZE_EVERY = 2000       # sampling batch
DETECT_BATCH = 2000     # detect batch

SAMPLE_DENORM = "generated.npy"
SAMPLE_NORM = "generated_norm.npy"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generator", required=True, choices=list(GENERATORS))
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_GEN_SEED,
                        help="sampling seed; part of the output path")
    parser.add_argument("--train_seed", type=int, default=DEFAULT_TRAIN_SEED,
                        help="train_baselines.py --seed of the checkpoints to load; "
                             "selects the checkpoint directory only")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES,
                        help="windows per pool. get_zscore draws 1000 without replacement, "
                             "so anything below that makes --mode zscore raise")
    parser.add_argument("--size_every", type=int, default=SIZE_EVERY,
                        help="sampling batch size")
    parser.add_argument("--milestone", type=int, default=10,
                        help="checkpoint milestone to load")

    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--wm_method", type=str, default=None,
                        help="sdformer: LVQMark (default). "
                             "diffusionts: TimeWak (default) | TR | GS")
    parser.add_argument("--watermark_delta", type=float, default=10.0,
                        help="green-list logit bias (sdformer only; part of its run key)")
    parser.add_argument("--watermark_seed", type=int, default=None,
                        help="watermark key used to GENERATE; part of the output path. "
                             "Default: 12345 for sdformer, 217 for diffusionts")

    # The two green-list switches, under the preprint's names (p.6). None means "whatever
    # --wm_method's preset says". sdformer only; the latent family has no such switches.
    parser.add_argument("--alternating_partition", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="flip the green partition on odd token positions (Eq. 9)")
    parser.add_argument("--unbiased_warmup", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="leave the first --warmup_positions token positions unbiased")
    parser.add_argument("--warmup_positions", type=int, default=None,
                        help="m, the number of leading token positions with no logit bias "
                             "(default: min(4, token_length // 3))")

    # Diffusion-TS sampling / latent-watermark parameters.
    dt = parser.add_argument_group("diffusionts")
    dt.add_argument("--sampling_timesteps", type=int, default=None,
                    help="override model.params.sampling_timesteps. Part of the run key: "
                         "it selects DDIM/BDIA at sampling time without changing a weight")
    dt.add_argument("--bits", type=int, default=2, help="TimeWak bits per latent entry")
    dt.add_argument("--interval", type=int, default=2, help="TimeWak seed-row interval")
    dt.add_argument("--w_channel", type=int, default=-1)
    dt.add_argument("--w_mask_shape", type=str, default="circle")
    dt.add_argument("--w_measurement", type=str, default="l1_complex")
    dt.add_argument("--w_injection", type=str, default="complex")
    dt.add_argument("--w_pattern", type=str, default="ring")
    dt.add_argument("--w_pattern_const", type=float, default=0)

    parser.add_argument("--detect_all", action="store_true")
    parser.add_argument("--methods", type=str, default=None,
                        help="comma-separated methods to detect (default: all of the "
                             "generator's; a method with no samples on disk is skipped)")
    parser.add_argument("--detect_key", type=int, default=None,
                        help="key to build the detector from; default is --watermark_seed, "
                             "i.e. the key the samples were generated under. A different "
                             "value is the wrong-key control (expected z ~ 0) and writes "
                             "to <method>/detect-key<k>/ instead of overwriting detect/. "
                             "Supported for the green-list family and for TR; GS and "
                             "TimeWak have no detector key this repo can vary")

    args = parser.parse_args()
    if args.wm_method is None:
        args.wm_method = "LVQMark" if args.generator == "sdformer" else "TimeWak"
    if args.wm_method not in methods_for(args.generator):
        parser.error(f"--wm_method {args.wm_method} is not available on "
                     f"{args.generator}; choose from {methods_for(args.generator)}")
    if args.watermark_seed is None:
        args.watermark_seed = (DEFAULT_WATERMARK_SEED if args.generator == "sdformer"
                               else DEFAULT_LATENT_WATERMARK_SEED)
    # Tree-Ring's mask radius is derived from the window length, so it can never disagree
    # with the window the samples were drawn at.
    args.w_radius = args.window_size - 1
    return args


def prepare_config(args):
    """Load the yaml and apply the window / sampling overrides. Returns (config, derived).

    args: the parsed namespace. ``derived`` carries the numbers both branches need
    (token_length, codebook_size, ...) and is what lands in run_config.json.
    """
    config = load_yaml_config(args.config_file)
    ws = args.window_size

    if args.generator == "sdformer":
        config["vqvae"]["params"]["seq_length"] = ws
        downsample_rate = config["vqvae"]["params"].get("downsample_rate", 4)
        token_length = ws // downsample_rate
        config["transformer"]["params"]["token_length"] = token_length
        derived = {"downsample_rate": downsample_rate, "token_length": token_length,
                   "feature_size": config["vqvae"]["params"]["feature_size"],
                   "codebook_size": config["vqvae"]["params"].get("codebook_size", 512)}
    else:
        config["model"]["params"]["seq_length"] = ws
        if args.sampling_timesteps is not None:
            config["model"]["params"]["sampling_timesteps"] = args.sampling_timesteps
        derived = {"feature_size": config["model"]["params"]["feature_size"],
                   "timesteps": config["model"]["params"]["timesteps"],
                   "sampling_timesteps": config["model"]["params"]["sampling_timesteps"]}

    config["dataloader"]["train_dataset"]["params"]["window"] = ws
    if "test_dataset" in config["dataloader"]:
        config["dataloader"]["test_dataset"]["params"]["window"] = ws
    return config, derived


def make_detect_loader(args, base_dir, npy_path, attack=None, attack_factor=None):
    """A DataLoader over an existing ``.npy`` pool, optionally attacked. Returns dl_info.

    args: parsed namespace. base_dir: the run's ``{window}`` directory. npy_path: the
    denormalised sample array. attack / attack_factor: a row of ``Utils.greenlist``'s grid.

    ``save2npy=False`` is load-bearing: with the config's ``True`` the dataset class
    *overwrites* ``{base_dir}/samples/*_train.npy`` with the attacked synthetic array on
    every condition (measured on the L-VQVAE side). It is RNG-inert -- the
    dataset saves and restores the global numpy state around its own seeded draws -- so the
    detect stream is consumed identically either way.
    """
    cfg = load_yaml_config(args.config_file)
    ws = args.window_size
    if args.generator == "sdformer":
        cfg["vqvae"]["params"]["seq_length"] = ws
    else:
        cfg["model"]["params"]["seq_length"] = ws
    params = cfg["dataloader"]["train_dataset"]["params"]
    params["window"] = ws
    params["dataset"] = npy_path
    params["proportion"] = 1.0
    params["shuffle"] = False
    params["attack"] = attack
    params["attack_factor"] = attack_factor
    params["save2npy"] = False
    params["output_dir"] = base_dir
    cfg["dataloader"]["batch_size"] = DETECT_BATCH
    cfg["dataloader"]["shuffle"] = False

    class _FA:
        save_dir = base_dir
        output = OUTPUT_BASE
        name = f"{args.dataset}-detect"
        tensorboard = False

    return build_dataloader(cfg, _FA())


def loader_to_array(dl_info):
    """Concatenate a detect DataLoader into one array. Returns (N, T, D) float numpy."""
    chunks = []
    for batch in dl_info["dataloader"]:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        chunks.append(batch.numpy())
    return np.concatenate(chunks, axis=0)


def save_samples(save_dir, samples, dataset_obj):
    """Write the normalised and denormalised sample arrays. Returns None.

    save_dir: str. samples: (N, T, D) in the model's scale. dataset_obj: the dataset whose
    scaler was fitted on the real data, so the denormalised array is on the data's units.
    """
    samples_norm = (unnormalize_to_zero_to_one(samples) if dataset_obj.auto_norm
                    else samples.copy())
    np.save(os.path.join(save_dir, SAMPLE_NORM), samples_norm)
    feat = samples_norm.shape[-1]
    samples_denorm = dataset_obj.scaler.inverse_transform(
        samples_norm.reshape(-1, feat)).reshape(samples_norm.shape)
    np.save(os.path.join(save_dir, SAMPLE_DENORM), samples_denorm)
    print(f"Saved {samples.shape[0]} samples to {save_dir}", flush=True)


# ── SDformer: the green-list family ──────────────────────────────────────────────────────

def sdformer_load_models(args, config, derived, ckpt_dir):
    """Load the SDformer VQ-VAE and AR transformer in eval mode. Returns (vqvae, transformer)."""
    vqvae = instantiate_from_config(config["vqvae"]).cuda()
    vqvae.load_state_dict(torch.load(resolve_checkpoint(
        os.path.join(ckpt_dir, "vqvae", f"checkpoint-{args.milestone}.pt"),
        "the SDformer VQ-VAE checkpoint",
        f"run `python -u cli/train_baselines.py --generator sdformer --config_file "
        f"{args.config_file} --dataset {args.dataset} --window_size {args.window_size} "
        f"--gpu {args.gpu} --stage 1`"), map_location="cuda")["model"])
    vqvae.eval()

    tf_ckpt = torch.load(resolve_checkpoint(
        os.path.join(ckpt_dir, "transformer", f"checkpoint-{args.milestone}.pt"),
        "the SDformer AR transformer checkpoint",
        "run train_baselines.py --generator sdformer --stage 2"), map_location="cuda")
    train_tl = next(v.shape[-1] for k, v in tf_ckpt["model"].items() if "mask" in k)
    tf_config = dict(config["transformer"])
    tf_config["params"] = dict(tf_config["params"])
    tf_config["params"]["token_length"] = train_tl
    transformer = instantiate_from_config(tf_config).cuda()
    transformer.load_state_dict(tf_ckpt["model"])
    transformer.eval()
    transformer.token_length = derived["token_length"]
    return vqvae, transformer


def sdformer_generate(args, config, derived, ckpt_dir, base_dir, payload):
    """Sample one SDformer pool (clean or watermarked) into ``base_dir``. Returns None."""
    from Models.sdformer.solver import sample_mts

    vqvae, transformer = sdformer_load_models(args, config, derived, ckpt_dir)
    token_length = derived["token_length"]

    class FA:
        save_dir = base_dir
        output = OUTPUT_BASE
        name = f"{args.dataset}-gen"
        tensorboard = False

    dl_info = build_dataloader(config, FA())
    dataset_obj = dl_info["dataset"]

    watermark_mask, watermark_delta = None, 0.0
    sampling_alternating_partition = False
    if args.watermark:
        alternating_partition, unbiased_warmup = resolve_wm_switches(
            args.wm_method, args.alternating_partition, args.unbiased_warmup)
        warmup_positions = resolve_warmup_positions(token_length, args.warmup_positions)
        watermark_mask = create_fixed_watermark_table(
            derived["codebook_size"], args.watermark_seed).cuda()
        watermark_delta = args.watermark_delta
        sampling_alternating_partition = alternating_partition
        if unbiased_warmup:
            watermark_delta = warmup_delta_schedule(
                args.watermark_delta, token_length, warmup_positions)
        print(f"Watermarking [{args.wm_method}]: delta={args.watermark_delta}, "
              f"alternating_partition={alternating_partition}, "
              f"unbiased_warmup={unbiased_warmup}, "
              f"warmup_positions={warmup_positions}/{token_length}"
              f"{'' if unbiased_warmup else ' (unused)'}", flush=True)

    save_dir = (os.path.join(base_dir, args.wm_method, "sample") if args.watermark
                else os.path.join(base_dir, "sample"))
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, SAMPLE_DENORM)):
        print(f"[SKIP] {save_dir}/{SAMPLE_DENORM} exists", flush=True)
        return
    write_run_config(save_dir, payload)

    seed_everything(args.seed)
    samples = sample_mts(
        transformer, vqvae, num=args.num_samples, size_every=args.size_every,
        shape=[dataset_obj.window, dataset_obj.var_num],
        temperature=1.0, top_k=None,
        watermark_mask=watermark_mask, watermark_delta=watermark_delta,
        alternating_partition=sampling_alternating_partition,
    )
    save_samples(save_dir, samples, dataset_obj)


def sdformer_detect(args, config, derived, ckpt_dir, base_dir, methods,
                    detect_key, detect_suffix):
    """Score every method x condition through the robust conv encoder. Returns None.

    One CSV row per window, in window order, at
    ``{base_dir}/{method}/{condition}{detect_suffix}/{method}.csv`` -- the file format
    ``evaluate_baselines.py`` reads.
    """
    from Models.sdformer.vqvae import RobustConvEncoder

    downsample_rate = derived["downsample_rate"]
    K = derived["codebook_size"]

    robust_ckpt = torch.load(resolve_checkpoint(
        os.path.join(ckpt_dir, "robustencoder", "checkpoint-best.pt"),
        "the SDformer robust conv encoder",
        "run train_baselines.py --generator sdformer --stage 3"), map_location="cuda")
    # hidden_dim is recovered from the checkpoint so a drifted --robust_hidden cannot
    # silently build a different detector than the one that was trained.
    hidden_dim = robust_ckpt["model"]["net.0.weight"].shape[0]
    robust_enc = RobustConvEncoder(
        feature_dim=derived["feature_size"], hidden_dim=hidden_dim, n_classes=K,
        downsample_rate=downsample_rate, n_resblocks=2).cuda()
    robust_enc.load_state_dict(robust_ckpt["model"])
    robust_enc.eval()
    print(f"  Robust conv encoder: hidden_dim={hidden_dim}", flush=True)

    nowm_npy = os.path.join(base_dir, "sample", SAMPLE_DENORM)
    shared_mask = create_fixed_watermark_table(K, detect_key)

    def encode(data_scaled):
        out = []
        for i in range(0, len(data_scaled), DETECT_BATCH):
            batch = torch.tensor(data_scaled[i:i + DETECT_BATCH],
                                 dtype=torch.float32).cuda()
            with torch.no_grad():
                out.append(robust_enc(batch).argmax(dim=-1).cpu())
        return torch.cat(out, dim=0)

    def score(data_scaled, method, save_dir, alternating_partition):
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{method}.csv")
        if os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path}", flush=True)
            return
        indices = encode(data_scaled)
        start_tok, end_tok = compute_valid_token_range(
            data_scaled, rf=downsample_rate, stride=downsample_rate)
        for fraction in green_fractions(indices, shared_mask, start_tok, end_tok,
                                        alternating_partition):
            write_csv([fraction], method, save_dir)

    for wm in methods:
        wm_npy = os.path.join(base_dir, wm, "sample", SAMPLE_DENORM)
        if not os.path.exists(wm_npy):
            print(f"[SKIP] {wm}: {wm_npy} not found", flush=True)
            continue
        wm_alternating, _ = resolve_wm_switches(
            wm, args.alternating_partition, args.unbiased_warmup)
        print(f"[{wm}] alternating_partition={wm_alternating}, key={detect_key}", flush=True)
        for atk, factor in ATTACK_GRID:
            label = "detect" if atk is None else f"{atk}-{factor}"
            for pool_npy, suffix in ((wm_npy, ""), (nowm_npy, "-wo")):
                print(f"[{wm}] {label}{suffix}", flush=True)
                data = loader_to_array(make_detect_loader(
                    args, base_dir, pool_npy, attack=atk, attack_factor=factor))
                score(data, wm,
                      os.path.join(base_dir, wm,
                                   detect_dir_name(f"{label}{suffix}", detect_suffix)),
                      wm_alternating)


# ── Diffusion-TS: the initial-latent family ──────────────────────────────────────────────

def diffusionts_trainer(args, config, ckpt_dir, dl_info):
    """Build a Trainer with the Diffusion-TS checkpoint loaded. Returns the Trainer.

    The Trainer owns the EMA model (which samples) and the raw model (which inverts), so
    both branches go through it. ``args`` doubles as the namespace ``generate_mts`` reads
    its watermark parameters from.
    """
    from Models.interpretable_diffusion.solver import Trainer

    resolve_checkpoint(
        os.path.join(ckpt_dir, "diffusion", f"checkpoint-{args.milestone}.pt"),
        f"the Diffusion-TS checkpoint at milestone {args.milestone}",
        f"run `python -u cli/train_baselines.py --generator diffusionts --config_file "
        f"{args.config_file} --dataset {args.dataset} --window_size {args.window_size} "
        f"--gpu {args.gpu}` (or pass a --milestone that exists)")
    model = instantiate_from_config(config["model"]).cuda()
    trainer = Trainer(config=config, args=args, model=model, dataloader=dl_info,
                      logger=None, results_folder=os.path.join(ckpt_dir, "diffusion"))
    trainer.load(args.milestone)
    return trainer


def diffusionts_generate(args, config, ckpt_dir, base_dir, payload):
    """Sample one Diffusion-TS pool (clean or watermarked) into ``base_dir``. Returns None."""

    class FA:
        save_dir = base_dir
        output = OUTPUT_BASE
        name = f"{args.dataset}-gen"
        tensorboard = False

    dl_info = build_dataloader(config, FA())
    dataset_obj = dl_info["dataset"]

    save_dir = (os.path.join(base_dir, args.wm_method, "sample") if args.watermark
                else os.path.join(base_dir, "sample"))
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, SAMPLE_DENORM)):
        print(f"[SKIP] {save_dir}/{SAMPLE_DENORM} exists", flush=True)
        return

    watermark = args.wm_method if args.watermark else ""
    if watermark:
        print(f"Watermarking [{watermark}]: key={args.watermark_seed}, "
              f"sampling_timesteps={config['model']['params']['sampling_timesteps']}"
              + (f", bits={args.bits}, interval={args.interval}"
                 if watermark == "TimeWak" else "")
              + (f", w_radius={args.w_radius}, w_pattern={args.w_pattern}"
                 if watermark == "TR" else ""), flush=True)
        if watermark == "GS" and args.watermark_seed != DEFAULT_LATENT_WATERMARK_SEED:
            raise SystemExit(
                f"GS cannot be generated under key {args.watermark_seed}: its detector "
                f"(Utils.watermark_metrics.eval_GS) pins the key at "
                f"{DEFAULT_LATENT_WATERMARK_SEED}, and Utils/watermark_metrics.py is vendored unmodified. "
                f"Generating under another key would produce an undetectable watermark "
                f"with no error. Drop --watermark_seed to generate GS.")

    trainer = diffusionts_trainer(args, config, ckpt_dir, dl_info)
    samples = trainer.sample(num=args.num_samples, size_every=args.size_every,
                             shape=[dataset_obj.window, dataset_obj.var_num],
                             watermark=watermark, save_dir=save_dir)
    # The NaN-dropping loop makes the number of generate_mts calls -- and so the RNG
    # consumed -- data-dependent. Record both counts beside the samples.
    payload = dict(payload)
    payload["derived"] = dict(payload["derived"],
                              n_generate_calls=trainer.n_generate_calls,
                              n_nan_dropped=trainer.n_nan_dropped)
    write_run_config(save_dir, payload)
    save_samples(save_dir, samples, dataset_obj)


def diffusionts_detect(args, config, ckpt_dir, base_dir, methods, detect_key,
                       detect_suffix):
    """Invert each pool and score it with the method's detector. Returns None.

    The inversion has to match the sampler (BDIA for TimeWak, DDIM otherwise), which
    ``Trainer.detect`` selects from the method name. Scoring is
    ``Utils.watermark_metrics.eval_{TimeWak,TR,GS}``, unchanged: they write one row per
    window into a ``<lowercase stem>.csv``, which is renamed to ``{method}.csv`` so that
    both watermark families leave the same file for ``evaluate_baselines.py`` to read.
    """
    from Utils.watermark_metrics import eval_GS, eval_TimeWak, eval_TR

    nowm_npy = os.path.join(base_dir, "sample", SAMPLE_DENORM)
    # The trainer is built once; `watermark` is passed per call, so one load serves every
    # method and the inversion still follows the method being scored.
    dl_info = make_detect_loader(args, base_dir, nowm_npy)
    trainer = diffusionts_trainer(args, config, ckpt_dir, dl_info)

    for wm in methods:
        wm_npy = os.path.join(base_dir, wm, "sample", SAMPLE_DENORM)
        if not os.path.exists(wm_npy):
            print(f"[SKIP] {wm}: {wm_npy} not found", flush=True)
            continue
        if detect_suffix and wm not in KEYED_LATENT_WM_METHODS:
            print(f"[SKIP] {wm}: no wrong-key control. Its detector in "
                  f"Utils/watermark_metrics.py has no key this repo can vary "
                  f"(GS pins {DEFAULT_LATENT_WATERMARK_SEED}; TimeWak is keyless), and "
                  f"Utils/watermark_metrics.py is used unmodified.", flush=True)
            continue

        tr_mask = tr_patch = None
        if wm == "TR":
            art_dir = os.path.join(base_dir, "TR", "sample")
            mask_path = os.path.join(art_dir, "watermarking_mask.npy")
            patch_path = os.path.join(art_dir, "gt_patch.npy")
            if not (os.path.exists(mask_path) and os.path.exists(patch_path)):
                print(f"[SKIP] TR: {art_dir} has no gt_patch.npy / watermarking_mask.npy; "
                      f"re-run generation for TR", flush=True)
                continue
            tr_mask = torch.tensor(np.load(mask_path))
            tr_patch = torch.tensor(np.load(patch_path))
            if detect_suffix:
                # The saved array is (N, 1, T, D); get_watermarking_pattern draws one
                # (1, 1, T, D) pattern at a time, exactly as generate_mts's TR branch does.
                tr_patch = build_tr_key_patches(
                    args, torch.device("cpu"), len(tr_patch),
                    (1,) + tuple(tr_patch.shape[1:]), detect_key).to(tr_patch.dtype)
                print(f"  [WRONG KEY] TR patterns redrawn under key {detect_key}; the "
                      f"mask is geometry and does not depend on the key", flush=True)

        print(f"[{wm}] key={detect_key}", flush=True)
        for atk, factor in ATTACK_GRID:
            label = "detect" if atk is None else f"{atk}-{factor}"
            for pool_npy, suffix in ((wm_npy, ""), (nowm_npy, "-wo")):
                save_dir = os.path.join(
                    base_dir, wm, detect_dir_name(f"{label}{suffix}", detect_suffix))
                os.makedirs(save_dir, exist_ok=True)
                print(f"[{wm}] {label}{suffix}", flush=True)
                # Built before the skip check so a resumed run consumes the numpy stream
                # exactly as a fresh one does.
                loader = make_detect_loader(args, base_dir, pool_npy,
                                            attack=atk, attack_factor=factor)
                csv_path = os.path.join(save_dir, f"{wm}.csv")
                if os.path.exists(csv_path):
                    print(f"  [SKIP] {csv_path}", flush=True)
                    continue

                noises = trainer.detect(loader["dataloader"],
                                        shape=[loader["dataset"].window,
                                               loader["dataset"].var_num],
                                        watermark=wm)
                # eval_* append to <stem>.csv; a stale one from a crashed run would double
                # the rows, so it is removed rather than appended to.
                stem_path = os.path.join(save_dir, f"{LATENT_SCORER_CSV_STEM[wm]}.csv")
                if os.path.exists(stem_path):
                    os.remove(stem_path)

                # The eval_* functions read their parameters off an args namespace and
                # write into args.save_dir, so they get a copy pointed at this condition.
                scorer_args = argparse.Namespace(**vars(args))
                scorer_args.save_dir = save_dir
                scorer_args.watermark = wm

                if wm == "TR":
                    eval_TR(scorer_args, torch.tensor(noises), tr_mask, tr_patch)
                elif wm == "GS":
                    eval_GS(scorer_args, torch.tensor(noises))
                else:
                    eval_TimeWak(scorer_args, torch.tensor(noises))
                os.replace(stem_path, csv_path)


def main():
    args = parse_args()
    # One seed, once, at process start: --detect_all's conditions then consume a single
    # numpy stream in a fixed order, which is the RNG contract generate.py already has.
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    config, derived = prepare_config(args)
    ws = args.window_size
    if args.exp is None:
        args.exp = derive_exp_name(args.generator, config, args.train_seed)
    run = derive_run_name(
        args.generator, args.exp, gen_seed=args.seed, watermark_seed=args.watermark_seed,
        delta=args.watermark_delta if args.generator == "sdformer" else None,
        sampling_timesteps=(derived["sampling_timesteps"]
                            if args.generator == "diffusionts" else None))

    ckpt_dir = get_ckpt_dir(args.dataset, args.exp, ws, generator=args.generator)
    base_dir = get_output_dir(args.dataset, run, ws, generator=args.generator)
    print(f"  generator={args.generator}\n  exp={args.exp}\n  run={run}", flush=True)

    detect_key, detect_suffix = resolve_detect_key(args.detect_key, args.watermark_seed)
    # One grammar, two shapes: condition DIRECTORIES take "-key{k}", FILE names "_key{k}".
    key_tag = f"_key{detect_key}" if detect_suffix else ""

    if args.watermark and args.generator == "sdformer":
        preset = WM_METHODS[args.wm_method]
        resolved = resolve_wm_switches(args.wm_method, args.alternating_partition,
                                       args.unbiased_warmup)
        if resolved != (preset["alternating_partition"], preset["unbiased_warmup"]):
            print(f"  [WARN] the explicit switches override the {args.wm_method} preset; "
                  f"the artifacts still land under '{args.wm_method}/', which the "
                  f"directory name cannot distinguish. run_config.json records the "
                  f"resolved values.", flush=True)

    def payload(stage):
        extra = {}
        if args.generator == "sdformer" and args.watermark:
            ap, uw = resolve_wm_switches(args.wm_method, args.alternating_partition,
                                         args.unbiased_warmup)
            extra = {"alternating_partition": ap, "unbiased_warmup": uw,
                     "warmup_positions": resolve_warmup_positions(
                         derived["token_length"], args.warmup_positions)}
        return {
            "entry_point": "generate_baselines.py", "stage": stage, "argv": sys.argv,
            "args": vars(args),
            "derived": dict(derived, generator=args.generator, exp=args.exp, run=run,
                            ckpt_dir=ckpt_dir, base_dir=base_dir,
                            wm_method=args.wm_method, detect_key=detect_key, **extra),
            "config": config, "provenance": torch_provenance(args.gpu),
        }

    if args.detect_all:
        methods = ([m.strip() for m in args.methods.split(",")] if args.methods
                   else list(methods_for(args.generator)))
        unknown = [m for m in methods if m not in methods_for(args.generator)]
        if unknown:
            raise SystemExit(f"--methods {unknown} not available on {args.generator}")
        write_run_config(base_dir, payload(f"detect{detect_suffix}"),
                         filename=f"run_config_detect{key_tag}.json")
        if detect_suffix:
            print(f"  [WRONG-KEY CONTROL] detecting with key {detect_key} while the "
                  f"samples were generated under key {args.watermark_seed}. Writing to "
                  f"<method>/*{detect_suffix}/; the default detect/ dirs are untouched.",
                  flush=True)
        if args.generator == "sdformer":
            sdformer_detect(args, config, derived, ckpt_dir, base_dir, methods,
                            detect_key, detect_suffix)
        else:
            diffusionts_detect(args, config, ckpt_dir, base_dir, methods,
                               detect_key, detect_suffix)
        print("Detect done.", flush=True)
        return

    stage = f"generate-{args.wm_method}" if args.watermark else "generate-clean"
    if args.generator == "sdformer":
        sdformer_generate(args, config, derived, ckpt_dir, base_dir, payload(stage))
    else:
        diffusionts_generate(args, config, ckpt_dir, base_dir, payload(stage))


if __name__ == "__main__":
    main()

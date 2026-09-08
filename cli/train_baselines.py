"""Train a baseline generator: SDformer (VQ-VAE + AR Transformer) or Diffusion-TS.

This is the baseline twin of ``train.py``. It shares ``Models/``, ``Utils/``, ``Data/`` and
the artifact trees with the L-VQVAE entry points and imports **no** code from them: three
``cli/*_baselines.py`` entry points, no ``baselines/`` package, no import of another
entry point.

Paths (Utils/path_utils, with the generator as the first level):
  checkpoints: checkpoints/{generator}/{dataset}/{exp}/{window}/{stage}/
  outputs:     outputs/{generator}/{dataset}/{exp}/{window}/samples/   (ground-truth dump)
  where exp is derived by Utils.baseline_utils.derive_exp_name:
    sdformer     hid{hidden}_cb{codebook}_ds{downsample}_{quantizer}
                 _ar{n_embd}_l{n_layer}_h{n_head}_trainseed{s}
    diffusionts  dm{d_model}_enc{n_layer_enc}_dec{n_layer_dec}_t{timesteps}_trainseed{s}

Stages:
  sdformer     1 = VQ-VAE, 2 = AR transformer, 3 = robust conv encoder, 0 = all three
  diffusionts  1 = the diffusion model (the only stage), 0 = the same thing

Every quantity that governs run length is a CLI argument whose default is the parent's
production value, and the *resolved* value is written to run_config.json next to the
checkpoints. That is what lets a short run use the same code path as a full one.

Usage:
  python -u cli/train_baselines.py --generator sdformer \
      --config_file configs/sdformer/etth.yaml --dataset etth --window_size 64 --gpu 2
  python -u cli/train_baselines.py --generator diffusionts \
      --config_file configs/diffusionts/etth.yaml --dataset etth --window_size 64 --gpu 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm.auto import tqdm

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Data.build_dataloader import build_dataloader
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one
from Models.model_utils import unnormalize_to_zero_to_one
from Utils.baseline_utils import GENERATORS, derive_exp_name, resolve_checkpoint
from Utils.greenlist import ROBUST_TRAIN_ATTACK_GRID
from Utils.io_utils import instantiate_from_config, load_yaml_config, seed_everything
from Utils.path_utils import (
    DEFAULT_TRAIN_SEED,
    OUTPUT_BASE,
    get_ckpt_dir,
    get_train_output_dir,
    torch_provenance,
    write_run_config,
)
from Utils.watermark_utils import add_attack

# Production defaults, from the parent's train_vqvae_robust.py. Named constants rather than
# argparse literals so the values and their provenance sit together.
ROBUST_NUM_SAMPLES = 10000
ROBUST_GEN_SEEDS = "1,12,123,1234"
ROBUST_STEPS = 50000
ROBUST_HIDDEN = 512
ROBUST_LR = 1e-4
ROBUST_VAL_EVERY = 5000
ROBUST_BATCH_SIZE = 128
ROBUST_SPLIT_SEED = 42       # RandomState for the 90/10 train/val split
ROBUST_ATTACK_SEED = 123     # np.random.seed before each attacked chunk
SIZE_EVERY = 2000            # sampling batch, all pipelines in this repo


def cycle(dl):
    while True:
        for data in dl:
            yield data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generator", required=True, choices=list(GENERATORS))
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED,
                        help="training seed; part of the checkpoint path")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3],
                        help="sdformer: 1=vqvae, 2=transformer, 3=robust encoder, 0=all. "
                             "diffusionts: 1 (or 0) = the diffusion model")
    parser.add_argument("--exp", type=str, default=None,
                        help="experiment key (auto-derived from the config; hand-passing "
                             "one for a real run is how you load the wrong checkpoints)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="override dataloader.batch_size")

    # run-length knobs; when left at None the config supplies the value, so the config
    # stays the single source of truth and the resolved number lands in run_config.json
    sd = parser.add_argument_group("sdformer")
    sd.add_argument("--stage1_steps", type=int, default=None,
                    help="VQ-VAE steps (default: solver.stage1.max_epochs, 50000)")
    sd.add_argument("--stage2_steps", type=int, default=None,
                    help="AR transformer steps (default: solver.stage2.max_epochs, 50000)")
    sd.add_argument("--lr", type=float, default=None,
                    help="override solver.stage{1,2}.base_lr (3e-4)")
    sd.add_argument("--robust_steps", type=int, default=ROBUST_STEPS)
    sd.add_argument("--robust_num_samples", type=int, default=ROBUST_NUM_SAMPLES,
                    help="windows generated per --robust_gen_seeds entry")
    sd.add_argument("--robust_gen_seeds", type=str, default=ROBUST_GEN_SEEDS)
    sd.add_argument("--robust_hidden", type=int, default=ROBUST_HIDDEN)
    sd.add_argument("--robust_lr", type=float, default=ROBUST_LR)
    sd.add_argument("--robust_val_every", type=int, default=ROBUST_VAL_EVERY)
    sd.add_argument("--robust_batch_size", type=int, default=ROBUST_BATCH_SIZE)
    sd.add_argument("--size_every", type=int, default=SIZE_EVERY,
                    help="sampling batch size while generating the robust encoder's "
                         "training data")

    dt = parser.add_argument_group("diffusionts")
    dt.add_argument("--max_steps", type=int, default=None,
                    help="diffusion steps (default: solver.max_epochs, per-dataset)")

    parser.add_argument("--save_cycle", type=int, default=None,
                        help="steps between checkpoints. Default: the config's value for "
                             "diffusionts, steps//10 for sdformer. The generation entry "
                             "point loads milestone 10, so save_cycle has to divide the "
                             "step count at least ten times")
    return parser.parse_args()


def prepare_config(args):
    """Load the yaml and apply every CLI override. Returns (config, derived dict).

    args: the parsed namespace. The returned config is the one that is instantiated *and*
    the one dumped to run_config.json, so what the run resolved and what it recorded cannot
    disagree.
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
        for stage_key, steps in (("stage1", args.stage1_steps),
                                 ("stage2", args.stage2_steps)):
            if steps is not None:
                config["solver"][stage_key]["max_epochs"] = steps
                config["solver"][stage_key]["save_cycle"] = (
                    args.save_cycle if args.save_cycle is not None else max(1, steps // 10))
            elif args.save_cycle is not None:
                config["solver"][stage_key]["save_cycle"] = args.save_cycle
            if args.lr is not None:
                config["solver"][stage_key]["base_lr"] = args.lr
    else:
        config["model"]["params"]["seq_length"] = ws
        derived = {"feature_size": config["model"]["params"]["feature_size"],
                   "timesteps": config["model"]["params"]["timesteps"],
                   "sampling_timesteps": config["model"]["params"]["sampling_timesteps"]}
        if args.max_steps is not None:
            config["solver"]["max_epochs"] = args.max_steps
            config["solver"]["save_cycle"] = (
                args.save_cycle if args.save_cycle is not None
                else max(1, args.max_steps // 10))
        elif args.save_cycle is not None:
            config["solver"]["save_cycle"] = args.save_cycle

    config["dataloader"]["train_dataset"]["params"]["window"] = ws
    if "test_dataset" in config["dataloader"]:
        config["dataloader"]["test_dataset"]["params"]["window"] = ws
    if args.batch_size:
        config["dataloader"]["batch_size"] = args.batch_size
    return config, derived


def train_sdformer(args, config, derived, ckpt_dir, dl_info, dataset_obj):
    """Stages 1-3 of the SDformer line, into ``ckpt_dir/{vqvae,transformer,robustencoder}``.

    args: parsed namespace. config: the resolved config. derived: the dict from
    :func:`prepare_config`. ckpt_dir: ``pathlib.Path``. dl_info / dataset_obj: from
    ``Data.build_dataloader.build_dataloader``. Returns None; writes checkpoints.
    """
    from Models.sdformer.solver import TransformerTrainer, VQVAETrainer, sample_mts
    from Models.sdformer.vqvae import RobustConvEncoder

    ws = args.window_size
    feature_size = derived["feature_size"]
    codebook_size = derived["codebook_size"]
    downsample_rate = derived["downsample_rate"]

    class _FA:
        save_dir = str(ckpt_dir)
        name = f"{args.dataset}-{ws}"
        tensorboard = False

    if args.stage in (0, 1):
        vqvae_dir = ckpt_dir / "vqvae"
        steps = config["solver"]["stage1"]["max_epochs"]
        print(f"\n{'='*60}\n  SDformer stage 1: VQ-VAE ({steps} steps)\n{'='*60}",
              flush=True)
        model = instantiate_from_config(config["vqvae"]).cuda()
        VQVAETrainer(config=config, args=_FA(), model=model, dataloader=dl_info,
                     logger=None, results_folder=str(vqvae_dir)).train()
        del model

    if args.stage in (0, 2):
        vqvae_dir = ckpt_dir / "vqvae"
        tf_dir = ckpt_dir / "transformer"
        steps = config["solver"]["stage2"]["max_epochs"]
        print(f"\n{'='*60}\n  SDformer stage 2: AR transformer ({steps} steps)\n{'='*60}",
              flush=True)
        vqvae = instantiate_from_config(config["vqvae"]).cuda()
        vqvae_ckpt = resolve_checkpoint(
            str(vqvae_dir / "checkpoint-10.pt"), "the SDformer stage-1 VQ-VAE checkpoint",
            f"run stage 1 (`python -u cli/train_baselines.py --generator sdformer "
            f"--config_file {args.config_file} --dataset {args.dataset} "
            f"--window_size {ws} --gpu {args.gpu} --stage 1`)")
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location="cuda")["model"])
        vqvae.eval()
        transformer = instantiate_from_config(config["transformer"]).cuda()
        TransformerTrainer(config=config, args=_FA(), transformer=transformer, vqvae=vqvae,
                           dataloader=dl_info, logger=None,
                           results_folder=str(tf_dir)).train()
        del vqvae, transformer

    if args.stage not in (0, 3):
        return

    robust_dir = ckpt_dir / "robustencoder"
    os.makedirs(robust_dir, exist_ok=True)
    if (robust_dir / "checkpoint-best.pt").exists() and args.stage == 0:
        print(">>> Robust encoder exists, skipping stage 3", flush=True)
        return
    print(f"\n{'='*60}\n  SDformer stage 3: robust conv encoder "
          f"({args.robust_steps} steps)\n{'='*60}", flush=True)

    vqvae = instantiate_from_config(config["vqvae"]).cuda()
    vqvae.load_state_dict(torch.load(resolve_checkpoint(
        str(ckpt_dir / "vqvae" / "checkpoint-10.pt"),
        "the SDformer stage-1 VQ-VAE checkpoint", "run --stage 1"),
        map_location="cuda")["model"])
    vqvae.eval()

    tf_ckpt = torch.load(resolve_checkpoint(
        str(ckpt_dir / "transformer" / "checkpoint-10.pt"),
        "the SDformer stage-2 AR transformer checkpoint", "run --stage 2"),
        map_location="cuda")
    train_tl = next(v.shape[-1] for k, v in tf_ckpt["model"].items() if "mask" in k)
    tf_config = dict(config["transformer"])
    tf_config["params"] = dict(tf_config["params"])
    tf_config["params"]["token_length"] = train_tl
    transformer = instantiate_from_config(tf_config).cuda()
    transformer.load_state_dict(tf_ckpt["model"])
    transformer.eval()
    transformer.token_length = derived["token_length"]

    auto_norm = dataset_obj.auto_norm
    gen_seeds = [int(s) for s in args.robust_gen_seeds.split(",")]
    n_per_seed = args.robust_num_samples

    all_samples_raw, all_target_idx = [], []
    for gs in gen_seeds:
        print(f"  Generating {n_per_seed} samples (seed={gs})...", flush=True)
        seed_everything(gs)
        samples = sample_mts(transformer, vqvae, num=n_per_seed,
                             size_every=args.size_every,
                             shape=[ws, feature_size])
        samples_01 = unnormalize_to_zero_to_one(samples) if auto_norm else samples.copy()
        samples_raw = dataset_obj.scaler.inverse_transform(
            samples_01.reshape(-1, feature_size)).reshape(samples.shape)
        all_samples_raw.append(samples_raw)

        # Target tokens: raw -> a scaler fitted on this chunk -> [-1,1] -> encode. This is
        # the same route the detect pipeline takes (npy -> dataloader -> scaler), so the
        # encoder is trained against the tokens detection will ask it for.
        B, T, D = samples_raw.shape
        clean_scaled = MinMaxScaler().fit_transform(
            samples_raw.reshape(-1, D)).reshape(B, T, D)
        if auto_norm:
            clean_scaled = normalize_to_neg_one_to_one(clean_scaled)
        chunk = []
        with torch.no_grad():
            for i in range(0, len(clean_scaled), args.size_every):
                batch = torch.tensor(clean_scaled[i:i + args.size_every],
                                     dtype=torch.float32).cuda()
                chunk.append(vqvae.encode(batch).cpu())
        all_target_idx.append(torch.cat(chunk, dim=0).numpy())

    samples_raw = np.concatenate(all_samples_raw, axis=0)
    target_idx = np.concatenate(all_target_idx, axis=0)
    print(f"  Total: {samples_raw.shape[0]} samples, {target_idx.shape[1]} tokens",
          flush=True)

    # Attack augmentation. The grid is Utils.greenlist.ROBUST_TRAIN_ATTACK_GRID -- the one
    # the robust encoder is TRAINED on, which differs from the one detection is SCORED on
    # by a single attack. No entry point spells either name itself.
    all_scaled, all_targets = [], []
    for atk_name, atk_factor in ROBUST_TRAIN_ATTACK_GRID:
        print(f"  {'Clean' if atk_name is None else f'{atk_name}-{atk_factor}'}...",
              flush=True)
        for si in range(len(gen_seeds)):
            start = si * n_per_seed
            chunk_raw = samples_raw[start:start + n_per_seed].copy()
            B, T, D = chunk_raw.shape
            if atk_name is not None:
                np.random.seed(ROBUST_ATTACK_SEED)
                chunk_raw = add_attack(chunk_raw, atk_name, atk_factor)
            scaled = MinMaxScaler().fit_transform(
                chunk_raw.reshape(-1, D)).reshape(B, T, D)
            if auto_norm:
                scaled = normalize_to_neg_one_to_one(scaled)
            all_scaled.append(scaled.astype(np.float32))
            all_targets.append(target_idx[start:start + n_per_seed].copy())

    X = np.concatenate(all_scaled, axis=0)
    Y = np.concatenate(all_targets, axis=0)
    n_total = X.shape[0]
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    perm = np.random.RandomState(ROBUST_SPLIT_SEED).permutation(n_total)
    print(f"  Train: {n_train}, Val: {n_val}", flush=True)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X[perm[:n_train]]),
        torch.tensor(Y[perm[:n_train]], dtype=torch.long))
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X[perm[n_train:]]),
        torch.tensor(Y[perm[n_train:]], dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.robust_batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.robust_batch_size, shuffle=False)
    rdl = cycle(train_loader)

    robust_enc = RobustConvEncoder(
        feature_dim=feature_size, hidden_dim=args.robust_hidden,
        n_classes=codebook_size, downsample_rate=downsample_rate, n_resblocks=2,
    ).cuda()
    n_params = sum(p.numel() for p in robust_enc.parameters())
    opt = AdamW(robust_enc.parameters(), lr=args.robust_lr, betas=(0.9, 0.99))
    print(f"  Robust params: {n_params:,}", flush=True)

    @torch.no_grad()
    def evaluate_robust():
        robust_enc.eval()
        correct, total, total_loss, nb = 0, 0, 0.0, 0
        for xb, yb in val_loader:
            xb, yb = xb.cuda(), yb.cuda()
            logits = robust_enc(xb)
            total_loss += F.cross_entropy(
                logits.reshape(-1, codebook_size), yb.reshape(-1)).item()
            correct += (logits.argmax(dim=-1) == yb).float().sum().item()
            total += yb.numel()
            nb += 1
        robust_enc.train()
        return correct / total, total_loss / nb

    best_val_acc, best_state, best_step = 0.0, None, 0
    with tqdm(total=args.robust_steps, desc="Robust") as pbar:
        for step in range(1, args.robust_steps + 1):
            xb, yb = next(rdl)
            xb, yb = xb.cuda(), yb.cuda()
            logits = robust_enc(xb)
            loss = F.cross_entropy(logits.reshape(-1, codebook_size), yb.reshape(-1))
            loss.backward()
            clip_grad_norm_(robust_enc.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            acc = (logits.argmax(dim=-1) == yb).float().mean().item()
            pbar.set_description(f"CE:{loss.item():.4f} acc:{acc:.3f}")
            pbar.update(1)

            if step % args.robust_val_every == 0:
                val_acc, val_loss = evaluate_robust()
                pbar.write(f"  step {step}: val_acc={val_acc:.4f} val_CE={val_loss:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in robust_enc.state_dict().items()}

    if best_state is None:
        # A run shorter than --robust_val_every never validates. Save the final weights
        # rather than leaving checkpoint-best.pt absent, which generation would only
        # discover much later, and say so loudly.
        print(f"  [WARN] no validation happened ({args.robust_steps} steps < "
              f"--robust_val_every {args.robust_val_every}); checkpoint-best.pt holds the "
              f"FINAL weights, not the best ones.", flush=True)
        best_state = {k: v.clone() for k, v in robust_enc.state_dict().items()}
        best_step = args.robust_steps
    torch.save({"step": best_step, "model": best_state},
               str(robust_dir / "checkpoint-best.pt"))

    val_acc, _ = evaluate_robust()
    with open(robust_dir / "final_results.json", "w") as fh:
        json.dump({"best_step": best_step, "best_val_acc": best_val_acc,
                   "final_val_acc": val_acc, "params": n_params}, fh, indent=2)
    print(f"  Best val_acc: {best_val_acc:.4f} (step {best_step})", flush=True)


def train_diffusionts(args, config, ckpt_dir, dl_info):
    """The single Diffusion-TS stage. Writes ``ckpt_dir/diffusion/checkpoint-{n}.pt``.

    args: parsed namespace. config: the resolved config. ckpt_dir: ``pathlib.Path``.
    dl_info: from ``build_dataloader``. Returns None.
    """
    from Models.interpretable_diffusion.solver import Trainer

    steps = config["solver"]["max_epochs"]
    save_cycle = config["solver"]["save_cycle"]
    print(f"\n{'='*60}\n  Diffusion-TS ({steps} steps, save every {save_cycle})\n{'='*60}",
          flush=True)
    if steps // max(1, save_cycle) < 10:
        print(f"  [WARN] {steps} steps / save_cycle {save_cycle} produces "
              f"{steps // max(1, save_cycle)} milestones; generate_baselines.py loads "
              f"milestone 10 by default and will raise if it is absent.", flush=True)

    class _FA:
        save_dir = str(ckpt_dir)
        name = f"{args.dataset}-{args.window_size}"
        tensorboard = False
        watermark = ""

    model = instantiate_from_config(config["model"]).cuda()
    Trainer(config=config, args=_FA(), model=model, dataloader=dl_info, logger=None,
            results_folder=str(ckpt_dir / "diffusion")).train()


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    config, derived = prepare_config(args)
    ws = args.window_size
    if args.exp is None:
        args.exp = derive_exp_name(args.generator, config, args.seed)

    ckpt_dir = Path(get_ckpt_dir(args.dataset, args.exp, ws, generator=args.generator))
    train_out = get_train_output_dir(args.dataset, args.exp, ws, generator=args.generator)
    os.makedirs(train_out, exist_ok=True)
    print(f"  generator={args.generator}\n  exp={args.exp}\n  ckpt_dir={ckpt_dir}",
          flush=True)

    write_run_config(str(ckpt_dir), {
        "entry_point": "train_baselines.py", "argv": sys.argv, "args": vars(args),
        "derived": dict(derived, generator=args.generator, exp=args.exp,
                        ckpt_dir=str(ckpt_dir), output_dir=train_out),
        "config": config, "provenance": torch_provenance(args.gpu),
    })

    class FA:
        save_dir = train_out
        base_dir = train_out
        # Dead in practice: build_dataloader overwrites train_dataset.output_dir with
        # base_dir. Kept on the layout so it can never become a live path that skips the
        # generator level.
        output = os.path.join(OUTPUT_BASE, args.generator, args.dataset, args.exp)
        name = f"{args.dataset}-{ws}"
        tensorboard = False

    config["dataloader"]["train_dataset"]["params"]["output_dir"] = FA.output
    dl_info = build_dataloader(config, FA())

    if args.generator == "sdformer":
        train_sdformer(args, config, derived, ckpt_dir, dl_info, dl_info["dataset"])
    else:
        train_diffusionts(args, config, ckpt_dir, dl_info)

    print(f"\n{'='*60}\n  ALL DONE: {args.generator} {args.dataset} w={ws}\n{'='*60}")


if __name__ == "__main__":
    main()

"""
Generate + Detect for L-VQ-VAE (organized structure).

Paths (derived by Utils/path_utils.py — the same derivation train.py and evaluate.py use):
  checkpoints: checkpoints/{generator}/{dataset}/{exp}/{window}/
  output:      outputs/{generator}/{dataset}/{run}/{window}/
  where generator = "lvqvae" (Utils.path_utils.DEFAULT_GENERATOR — this is the L-VQVAE entry
                    point; the cli/*_baselines.py entry points pass generator=sdformer
                    / diffusionts against the same roots)
        exp = d{d_model}_cb{codebook}_st{stride}_ar{n_embd}_l{n_layer}_h{n_head}_trainseed{s}
        run = {exp}_delta{delta}_genseed{s}_wmseed{s}

stride is auto-determined from window_size: 24 -> 2, else -> 4.
--exp is auto-derived from config params if not specified.

Seeds: --seed (default 12345) drives sampling, --watermark_seed (default 12345) the green
list, --train_seed (default 12345) only selects the checkpoint directory. All three are part
of the artifact path, so a run under a different seed can never overwrite or silently reuse
another one's samples.

The watermark (LVQMark) uses one fixed green list and two switches, under the preprint's
names (arXiv:2608.19727 p.6): the *alternating partition* flips the green set on odd token
positions (Eq. 9), and the *unbiased warm-up* leaves the first m token positions without a
logit bias. LVQMark is both on, with m = min(4, token_length // 3).

--alternating_partition/--no-alternating_partition, --unbiased_warmup/--no-unbiased_warmup
and --warmup_positions m vary them individually. The resolved values (including m) are
printed and written to run_config.json.

Usage:
  # generate clean
  python -u cli/generate.py --dataset energy --window_size 24 --config_file configs/lvqvae/energy.yaml --gpu 0
  # generate watermarked (LVQMark)
  python -u cli/generate.py --dataset energy --window_size 24 --config_file configs/lvqvae/energy.yaml --gpu 0 --watermark --watermark_delta 10
  # detect (LVQMark only by default)
  python -u cli/generate.py --dataset energy --window_size 24 --config_file configs/lvqvae/energy.yaml --gpu 0 --detect_all
  # wrong-key control: detect the same samples with a green list built from another key
  python -u cli/generate.py --dataset energy --window_size 24 --config_file configs/lvqvae/energy.yaml --gpu 0 --detect_all --detect_key 999
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import numpy as np

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config
from Utils.path_utils import (
    DEFAULT_GENERATOR,
    DEFAULT_GEN_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_WATERMARK_SEED,
    OUTPUT_BASE,
    derive_exp_name,
    derive_run_name,
    get_ckpt_dir,
    get_output_dir,
    resolve_vqvae_checkpoint,
    stride_for_window,
    torch_provenance,
    write_run_config,
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
from Data.build_dataloader import build_dataloader
from Models.model_utils import unnormalize_to_zero_to_one
from Utils.metric_utils import write_csv
from Models.transformer_ar import sample_mts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='number of windows to sample (default 10000). NOT part of '
                             'the run key -- it is recorded in the run_config.json beside '
                             'the samples, so lowering it does not create a second '
                             'artifact tree. Detection statistics need at least a few '
                             'thousand windows')
    parser.add_argument('--seed', type=int, default=DEFAULT_GEN_SEED,
                        help='sampling seed; part of the output path')
    parser.add_argument('--train_seed', type=int, default=DEFAULT_TRAIN_SEED,
                        help="train.py --seed of the checkpoints to load; selects the "
                             "checkpoint directory only, no effect on sampling")

    parser.add_argument('--watermark', action='store_true')
    parser.add_argument('--wm_method', type=str, default='LVQMark',
                        choices=list(WM_METHODS.keys()),
                        help='watermark method; also the artifact subdirectory and the '
                             'label of every detect CSV it writes')
    parser.add_argument('--watermark_delta', type=float, default=10.0)
    parser.add_argument('--watermark_seed', type=int, default=DEFAULT_WATERMARK_SEED,
                        help='green-list seed used to GENERATE; part of the output path')

    # The two LVQMark switches, under the preprint's names (arXiv:2608.19727 p.6). Default
    # None means "whatever --wm_method's preset says"; passing either form overrides it.
    parser.add_argument('--alternating_partition', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='flip the green partition on odd token positions (Eq. 9). '
                             'Default: the --wm_method preset (LVQMark: on)')
    parser.add_argument('--unbiased_warmup', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='leave the first --warmup_positions token positions unbiased. '
                             'Default: the --wm_method preset (LVQMark: on)')
    parser.add_argument('--warmup_positions', type=int, default=None,
                        help='m, the number of leading token positions with no logit bias '
                             '(only used when the unbiased warm-up is on). '
                             'Default: min(4, token_length // 3) — 4 of 16 at window 64')

    parser.add_argument('--detect_all', action='store_true')
    parser.add_argument('--methods', type=str, default=None,
                        help='comma-separated WM methods to detect (default: LVQMark)')
    parser.add_argument('--detect_key', type=int, default=None,
                        help='key to build the green list from at DETECTION time; default '
                             'is --watermark_seed, i.e. the key the samples were generated '
                             'under. A different value is the wrong-key control (expected '
                             'z ~ 0) and writes to <method>/detect-key<k>/ etc. instead of '
                             'overwriting <method>/detect/')
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    ws = args.window_size
    stride = stride_for_window(ws)
    rf = config['vqvae']['params'].get('rf', 4)
    token_length = (ws - rf) // stride + 1
    K = config['vqvae']['params'].get('codebook_size', 512)

    config['vqvae']['params']['stride'] = stride
    config['vqvae']['params']['seq_length'] = ws
    config['transformer']['params']['token_length'] = token_length
    config['dataloader']['train_dataset']['params']['window'] = ws
    if 'test_dataset' in config['dataloader']:
        config['dataloader']['test_dataset']['params']['window'] = ws

    # Resolve the watermark switches here, before anything is written: they belong in
    # run_config.json, and both branches below need them. Pure argument arithmetic — it
    # touches no RNG stream (the green-list table uses its own torch.Generator, and the
    # sampling branch re-seeds with seed_everything immediately before sample_mts).
    alternating_partition, unbiased_warmup = resolve_wm_switches(
        args.wm_method, args.alternating_partition, args.unbiased_warmup)
    warmup_positions = resolve_warmup_positions(token_length, args.warmup_positions)
    detect_key, detect_suffix = resolve_detect_key(args.detect_key, args.watermark_seed)
    # One grammar, two shapes: condition DIRECTORIES take "-key{k}" (they sit beside
    # "detect" and the "<attack>-<factor>" dirs), FILE names take "_key{k}" (they sit
    # beside the "_seed{s}" / "_det" tags evaluate.py already uses).
    key_tag = f"_key{detect_key}" if detect_suffix else ""
    preset = WM_METHODS[args.wm_method]
    if (alternating_partition, unbiased_warmup) != (preset['alternating_partition'],
                                                    preset['unbiased_warmup']):
        print(f"  [WARN] the explicit switches override the {args.wm_method} preset "
              f"(alternating_partition={preset['alternating_partition']}->"
              f"{alternating_partition}, unbiased_warmup={preset['unbiased_warmup']}->"
              f"{unbiased_warmup}). The artifacts still land under '{args.wm_method}/', "
              f"which the directory name cannot distinguish; run_config.json records the "
              f"resolved values.", flush=True)

    if args.exp is None:
        args.exp = derive_exp_name(config, stride, args.train_seed)
    run = derive_run_name(args.exp, args.watermark_delta, args.seed, args.watermark_seed)

    # generator is the first level of every artifact tree (Utils/path_utils); passed
    # explicitly so it appears in the resolved run_config.json, not as a hidden default.
    ckpt_dir = get_ckpt_dir(args.dataset, args.exp, ws, generator=DEFAULT_GENERATOR)
    base_dir = get_output_dir(args.dataset, run, ws, generator=DEFAULT_GENERATOR)
    print(f"  generator={DEFAULT_GENERATOR}\n  exp={args.exp}\n  run={run}", flush=True)

    def run_config_payload(stage):
        return {
            "entry_point": "generate.py", "stage": stage, "argv": sys.argv,
            "args": vars(args),
            "derived": {"generator": DEFAULT_GENERATOR,
                        "exp": args.exp, "run": run, "stride": stride,
                        "token_length": token_length, "codebook_size": K,
                        "ckpt_dir": ckpt_dir, "base_dir": base_dir,
                        "wm_method": args.wm_method,
                        "alternating_partition": alternating_partition,
                        "unbiased_warmup": unbiased_warmup,
                        "warmup_positions": warmup_positions,
                        "num_samples": args.num_samples,
                        "detect_key": detect_key},
            "config": config, "provenance": torch_provenance(args.gpu),
        }

    # Load VQ-VAE. Raises if the stage-2 checkpoint is missing rather than falling back to
    # the stage-1 pretrain.pt, whose global decoder is randomly initialised (see
    # Utils/path_utils.resolve_vqvae_checkpoint).
    vqvae = instantiate_from_config(config['vqvae']).cuda()
    vqvae_ckpt = resolve_vqvae_checkpoint(ckpt_dir)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location='cuda')['model'])
    vqvae.eval()

    # Load Transformer
    tf_ckpt_path = os.path.join(ckpt_dir, 'transformer', 'checkpoint-10.pt')
    tf_ckpt = torch.load(tf_ckpt_path, map_location='cuda')
    for key in tf_ckpt['model']:
        if 'mask' in key:
            train_tl = tf_ckpt['model'][key].shape[-1]
            break
    tf_config = dict(config['transformer'])
    tf_config['params'] = dict(tf_config['params'])
    tf_config['params']['token_length'] = train_tl
    transformer = instantiate_from_config(tf_config).cuda()
    transformer.load_state_dict(tf_ckpt['model'])
    transformer.eval()
    transformer.token_length = token_length

    if args.detect_all:
        # ===== DETECT =====
        methods = [m.strip() for m in args.methods.split(",")] if args.methods else ["LVQMark"]
        write_run_config(base_dir, run_config_payload(f"detect{detect_suffix}"),
                         filename=f"run_config_detect{key_tag}.json")
        if detect_suffix:
            print(f"  [WRONG-KEY CONTROL] detecting with key {detect_key}, while the "
                  f"samples were generated under key {args.watermark_seed}. Writing to "
                  f"<method>/*{detect_suffix}/; the default detect/ dirs are untouched.",
                  flush=True)

        # Load robust encoder. d_model / n_layers are recovered from the
        # checkpoint itself so a drifted config can never silently mismatch;
        # only n_heads (which leaves no shape trace) comes from the config.
        from Models.local_vqvae import RobustEncoder, resolve_robust_cfg, infer_robust_cfg
        robust_path = os.path.join(ckpt_dir, 'robustencoder', 'checkpoint-best.pt')
        robust_ckpt = torch.load(robust_path, map_location='cuda')
        robust_d, robust_nl = infer_robust_cfg(robust_ckpt['model'])
        cfg_d, cfg_nl, robust_heads = resolve_robust_cfg(config)
        if (cfg_d, cfg_nl) != (robust_d, robust_nl):
            print(f"  [WARN] robust_encoder config (d_model={cfg_d}, n_layers={cfg_nl}) "
                  f"differs from checkpoint (d_model={robust_d}, n_layers={robust_nl}); "
                  f"using the checkpoint.", flush=True)
            robust_heads = max(4, robust_d // 64)
        print(f"  Robust arch: d_model={robust_d}, n_layers={robust_nl}, "
              f"n_heads={robust_heads}", flush=True)
        robust_enc = RobustEncoder(
            feature_size=config['vqvae']['params']['feature_size'],
            d_model=robust_d, n_heads=robust_heads, n_layers=robust_nl,
            rf=rf, stride=stride, n_classes=K
        ).cuda()
        robust_enc.load_state_dict(robust_ckpt['model'])
        robust_enc.eval()

        nowm_npy = os.path.join(base_dir, "sample", "generated.npy")

        def load_via_dataloader(npy_path, attack=None, attack_factor=None):
            cfg = load_yaml_config(args.config_file)
            cfg['vqvae']['params']['seq_length'] = ws
            cfg['dataloader']['train_dataset']['params']['window'] = ws
            cfg['dataloader']['train_dataset']['params']['dataset'] = npy_path
            cfg['dataloader']['train_dataset']['params']['proportion'] = 1.0
            cfg['dataloader']['train_dataset']['params']['shuffle'] = False
            cfg['dataloader']['train_dataset']['params']['attack'] = attack
            cfg['dataloader']['train_dataset']['params']['attack_factor'] = attack_factor
            # Detection must never write ground truth: with save2npy left at the config's
            # True, CustomDataset overwrites {base_dir}/samples/{name}_{ground,norm}_truth_
            # {window}_train.npy with the attacked *synthetic* array on every condition
            # (Utils/Data_utils/real_datasets.py:100-141). Read-only load; no RNG is
            # consumed either way, so the detect stream order is unchanged.
            cfg['dataloader']['train_dataset']['params']['save2npy'] = False
            cfg['dataloader']['batch_size'] = 2000
            cfg['dataloader']['shuffle'] = False
            cfg['dataloader']['train_dataset']['params']['output_dir'] = base_dir
            class _FA:
                save_dir = base_dir
                output = OUTPUT_BASE
                name = f'{args.dataset}-detect'
                tensorboard = False
            dl_info = build_dataloader(cfg, _FA())
            all_data = []
            for batch in dl_info['dataloader']:
                if isinstance(batch, (list, tuple)): batch = batch[0]
                all_data.append(batch.numpy())
            return np.concatenate(all_data, axis=0)

        def encode_data(data_scaled):
            all_indices = []
            for i in range(0, len(data_scaled), 2000):
                batch = torch.tensor(data_scaled[i:i+2000], dtype=torch.float32).cuda()
                with torch.no_grad():
                    idx = robust_enc(batch).argmax(dim=-1)
                all_indices.append(idx.cpu())
            return torch.cat(all_indices, dim=0)

        def run_detect_save(data_scaled, mask, method, save_dir, alternating_partition):
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"{method}.csv")
            if os.path.exists(csv_path):
                print(f"  [SKIP] {csv_path}", flush=True)
                return
            indices = encode_data(data_scaled)
            start_tok, end_tok = compute_valid_token_range(data_scaled, rf=rf, stride=stride)
            # One CSV row per window, in window order — the file format evaluate.py reads.
            for fraction in green_fractions(indices, mask, start_tok, end_tok,
                                            alternating_partition):
                write_csv([fraction], method, save_dir)

        # All methods share the same fixed table — build once, from the DETECTION key
        # (== the generation key unless --detect_key asked for the wrong-key control).
        shared_mask = create_fixed_watermark_table(K, detect_key)

        for wm in methods:
            wm_alternating, _ = resolve_wm_switches(
                wm, args.alternating_partition, args.unbiased_warmup)
            mask = shared_mask
            wm_npy = os.path.join(base_dir, wm, "sample", "generated.npy")
            if not os.path.exists(wm_npy):
                print(f"[SKIP] {wm}: {wm_npy} not found", flush=True)
                continue
            print(f"[{wm}] alternating_partition={wm_alternating}, key={detect_key}", flush=True)
            for atk, factor in ATTACK_GRID:
                if atk is None:
                    print(f"[{wm}] detect WM", flush=True)
                    data = load_via_dataloader(wm_npy)
                    run_detect_save(data, mask, wm,
                                    os.path.join(base_dir, wm, detect_dir_name("detect", detect_suffix)),
                                    wm_alternating)
                    print(f"[{wm}] detect NoWM", flush=True)
                    data = load_via_dataloader(nowm_npy)
                    run_detect_save(data, mask, wm,
                                    os.path.join(base_dir, wm, detect_dir_name("detect-wo", detect_suffix)),
                                    wm_alternating)
                else:
                    atk_label = f"{atk}-{factor}"
                    print(f"[{wm}] {atk_label} WM", flush=True)
                    data = load_via_dataloader(wm_npy, attack=atk, attack_factor=factor)
                    run_detect_save(data, mask, wm,
                                    os.path.join(base_dir, wm, detect_dir_name(atk_label, detect_suffix)),
                                    wm_alternating)
                    print(f"[{wm}] {atk_label} NoWM", flush=True)
                    data = load_via_dataloader(nowm_npy, attack=atk, attack_factor=factor)
                    run_detect_save(data, mask, wm,
                                    os.path.join(base_dir, wm, detect_dir_name(f"{atk_label}-wo", detect_suffix)),
                                    wm_alternating)
        print("Detect done.", flush=True)

    else:
        # ===== GENERATION =====
        class FA:
            save_dir = base_dir
            output = OUTPUT_BASE
            name = f'{args.dataset}-gen'
            tensorboard = False
        # (build_dataloader overwrites train_dataset.output_dir with FA.save_dir, so the
        # dataset's ground-truth dump lands in {base_dir}/samples.)
        dl_info = build_dataloader(config, FA())
        dataset_obj = dl_info['dataset']

        watermark_mask = None
        watermark_delta = 0.0
        # The clean arm samples with no bias at all, so the alternating partition is inert
        # there; it is passed as False, exactly as before.
        sampling_alternating_partition = False
        if args.watermark:
            watermark_mask = create_fixed_watermark_table(K, args.watermark_seed).cuda()
            watermark_delta = args.watermark_delta
            sampling_alternating_partition = alternating_partition
            if unbiased_warmup:
                watermark_delta = warmup_delta_schedule(
                    args.watermark_delta, token_length, warmup_positions)
            # m is printed as resolved (matching run_config.json) and marked unused when
            # the warm-up is off, rather than printed as 0 — the log and the JSON must not
            # disagree about the same field.
            print(f"Watermarking [{args.wm_method}]: delta={args.watermark_delta}, "
                  f"alternating_partition={alternating_partition}, "
                  f"unbiased_warmup={unbiased_warmup}, "
                  f"warmup_positions={warmup_positions}/{token_length}"
                  f"{'' if unbiased_warmup else ' (unused)'}", flush=True)

        if args.watermark:
            save_dir = os.path.join(base_dir, args.wm_method, 'sample')
        else:
            save_dir = os.path.join(base_dir, 'sample')
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(os.path.join(save_dir, 'generated.npy')):
            print(f"[SKIP] {save_dir}/generated.npy exists", flush=True)
            return

        write_run_config(save_dir, run_config_payload(
            f"generate-{args.wm_method}" if args.watermark else "generate-clean"))

        seed_everything(args.seed)
        samples = sample_mts(
            transformer, vqvae, num=args.num_samples, size_every=2000,
            shape=[dataset_obj.window, dataset_obj.var_num],
            temperature=1.0, top_k=None,
            watermark_mask=watermark_mask, watermark_delta=watermark_delta,
            alternating_partition=sampling_alternating_partition,
        )

        if dataset_obj.auto_norm:
            samples_norm = unnormalize_to_zero_to_one(samples)
        else:
            samples_norm = samples.copy()

        np.save(os.path.join(save_dir, "generated_norm.npy"), samples_norm)
        feat = samples_norm.shape[-1]
        samples_denorm = dataset_obj.scaler.inverse_transform(
            samples_norm.reshape(-1, feat)).reshape(samples_norm.shape)
        np.save(os.path.join(save_dir, "generated.npy"), samples_denorm)
        print(f'Saved {samples.shape[0]} samples to {save_dir}', flush=True)


if __name__ == '__main__':
    main()

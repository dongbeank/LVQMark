"""
Train Local VQ-VAE + AR Transformer + Robust Encoder.

Stage 1: LocalEncoder + LocalDecoder pretrain (codebook learning)
Stage 2: Freeze encoder+quantizer, train GlobalDecoder
Stage 3: AR Transformer
Stage 4: Robust Encoder

stride is auto-determined from window_size: 24 -> 2, else -> 4.
--exp is auto-derived from config params if not specified (Utils/path_utils.derive_exp_name):
  checkpoints/{generator}/{dataset}/
              d{d_model}_cb{codebook}_st{stride}_ar{n_embd}_l{n_layer}_h{n_head}
              _trainseed{seed}/{window}/{stage}/
  where generator = "lvqvae" (Utils.path_utils.DEFAULT_GENERATOR — the L-VQVAE entry point;
  the cli/*_baselines.py entry points pass generator=sdformer / diffusionts and write
  under the same roots)
The resolved configuration of the run — including the lr and step counts, which the directory
name does not carry — is dumped to run_config.json in that directory.

Usage:
  # Full pipeline
  python -u cli/train.py --config_file configs/lvqvae/etth.yaml --dataset etth --window_size 24 --gpu 0

  # Specific stage
  python -u cli/train.py --config_file configs/lvqvae/etth.yaml --dataset etth --window_size 24 --gpu 0 --stage 2
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import MinMaxScaler

import _bootstrap  # noqa: F401  -- repo root on sys.path + as cwd; keep first

from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config
from Utils.path_utils import (
    DEFAULT_GENERATOR,
    DEFAULT_TRAIN_SEED,
    OUTPUT_BASE,
    derive_exp_name,
    get_ckpt_dir,
    get_train_output_dir,
    stride_for_window,
    torch_provenance,
    write_run_config,
)
from Data.build_dataloader import build_dataloader
from Models.model_utils import unnormalize_to_zero_to_one
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one
from Utils.greenlist import ROBUST_TRAIN_ATTACK_GRID
from Utils.watermark_utils import add_attack


def cycle(dl):
    while True:
        for data in dl:
            yield data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train Local VQ-VAE')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=DEFAULT_TRAIN_SEED,
                        help='training seed; part of the checkpoint path')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0=all, 1=pretrain, 2=decoder, 3=transformer, 4=robust')

    # Steps
    parser.add_argument('--pretrain_steps', type=int, default=30000)
    parser.add_argument('--decoder_steps', type=int, default=10000)
    parser.add_argument('--transformer_steps', type=int, default=30000)
    parser.add_argument('--robust_steps', type=int, default=50000)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=None)

    # Robust encoder
    # Architecture (d_model / n_layers / n_heads) comes from the config's
    # `robust_encoder:` section; these flags override it when given.
    parser.add_argument('--gen_seeds', type=str, default='1,12,123,1234')
    parser.add_argument('--robust_val_every', type=int, default=5000,
                        help='validate the robust encoder every N steps and keep the best '
                             'state. Clamped to --robust_steps so a short run still '
                             'validates once and writes checkpoint-best.pt, which detection '
                             'needs; at the default step counts the clamp is inactive')
    parser.add_argument('--robust_d_model', type=int, default=None,
                        help='Override config robust_encoder.d_model')
    parser.add_argument('--robust_n_layers', type=int, default=None,
                        help='Override config robust_encoder.n_layers')
    parser.add_argument('--robust_n_heads', type=int, default=None,
                        help='Override config robust_encoder.n_heads '
                             '(default: auto = max(4, d_model // 64))')
    parser.add_argument('--robust_lr', type=float, default=1e-4)
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name (auto-derived from config if not specified)')
    return parser.parse_args()


def resolve_robust_val_every(robust_val_every: int, robust_steps: int) -> int:
    """Stage-4 validation interval, clamped so that a short run still validates once.

    robust_val_every / robust_steps: steps. Returns the interval to use, in steps.

    Stage 4 keeps the best-validating state and writes ``robustencoder/checkpoint-best.pt``
    only if it validated at least once, on a step that is a multiple of the interval. An
    interval larger than the run therefore ends with no best checkpoint, and
    ``generate.py --detect_all`` has nothing to load. Clamping is inert whenever
    ``robust_val_every <= robust_steps``, which includes the defaults (5000 <= 50000).
    """
    if robust_steps <= 0:
        return robust_val_every
    return max(1, min(robust_val_every, robust_steps))


def main():
    args = parse_args()

    val_every = resolve_robust_val_every(args.robust_val_every, args.robust_steps)
    if val_every != args.robust_val_every:
        print(f"  [INFO] --robust_val_every {args.robust_val_every} exceeds "
              f"--robust_steps {args.robust_steps}; using {val_every} so stage 4 still "
              f"writes checkpoint-best.pt.", flush=True)
        args.robust_val_every = val_every

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    ds = args.dataset
    ws = args.window_size
    stride = stride_for_window(ws)
    rf = config['vqvae']['params'].get('rf', 4)
    token_length = (ws - rf) // stride + 1
    feature_size = config['vqvae']['params']['feature_size']
    codebook_size = config['vqvae']['params'].get('codebook_size', 512)

    config['vqvae']['params']['stride'] = stride
    config['vqvae']['params']['seq_length'] = ws
    config['transformer']['params']['token_length'] = token_length
    config['dataloader']['train_dataset']['params']['window'] = ws
    if 'test_dataset' in config['dataloader']:
        config['dataloader']['test_dataset']['params']['window'] = ws
    if args.batch_size:
        config['dataloader']['batch_size'] = args.batch_size

    if args.exp is None:
        args.exp = derive_exp_name(config, stride, args.seed)

    # generator is the first level of every artifact tree (Utils/path_utils); this entry
    # point is the L-VQVAE one, and it is passed explicitly so the value shows up in the
    # resolved run_config.json rather than being an invisible default.
    ckpt_dir = Path(get_ckpt_dir(ds, args.exp, ws, generator=DEFAULT_GENERATOR))
    train_out = get_train_output_dir(ds, args.exp, ws, generator=DEFAULT_GENERATOR)
    print(f"  generator={DEFAULT_GENERATOR}\n  exp={args.exp}\n  ckpt_dir={ckpt_dir}", flush=True)

    # The resolved configuration of the run that produced these weights. The directory name
    # carries the keyed fields only (architecture + seed); lr and step counts live here, and
    # a rerun in the same directory that resolves differently prints a [WARN].
    write_run_config(str(ckpt_dir), {
        "entry_point": "train.py", "argv": sys.argv, "args": vars(args),
        "derived": {"generator": DEFAULT_GENERATOR,
                    "exp": args.exp, "stride": stride, "token_length": token_length,
                    "feature_size": feature_size, "codebook_size": codebook_size,
                    "ckpt_dir": str(ckpt_dir), "output_dir": train_out},
        "config": config, "provenance": torch_provenance(args.gpu),
    })

    # Dataloader
    class FA:
        save_dir = train_out
        base_dir = train_out
        # build_dataloader overwrites train_dataset.output_dir with base_dir, so this
        # value is not read; it spells out the same layout, generator level included.
        output = os.path.join(OUTPUT_BASE, DEFAULT_GENERATOR, ds, args.exp)
        name = f'{ds}-{ws}'
        tensorboard = False
    os.makedirs(FA.save_dir, exist_ok=True)
    config['dataloader']['train_dataset']['params']['output_dir'] = FA.output
    dl_info = build_dataloader(config, FA())
    dl = cycle(dl_info['dataloader'])
    dataset_obj = dl_info['dataset']
    batch_size = config['dataloader'].get('batch_size', 128)

    # ── Stage 1: Pretrain (LocalEncoder + LocalDecoder) ──
    if args.stage in [0, 1]:
        vqvae_dir = ckpt_dir / 'vqvae'
        os.makedirs(vqvae_dir, exist_ok=True)

        if (vqvae_dir / 'pretrain.pt').exists() and args.stage == 0:
            print(f">>> Pretrain checkpoint exists, skipping stage 1", flush=True)
        else:
            print(f"\n{'='*60}")
            print(f"  Stage 1: Pretrain ({args.pretrain_steps} steps)")
            print(f"  {ds} w={ws}")
            print(f"{'='*60}", flush=True)

            model = instantiate_from_config(config['vqvae']).cuda()
            opt = AdamW(model.parameters(), lr=args.lr, betas=[0.9, 0.99])

            with tqdm(total=args.pretrain_steps, desc='Pretrain') as pbar:
                for step in range(1, args.pretrain_steps + 1):
                    data = next(dl).cuda()
                    loss = model.forward(data, mode='pretrain')
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    pbar.set_description(f'pretrain: {loss.item():.6f}')
                    pbar.update(1)

            torch.save({'step': args.pretrain_steps, 'model': model.state_dict()},
                       str(vqvae_dir / 'pretrain.pt'))
            print(f"  Saved pretrain checkpoint", flush=True)

            # Idempotence check (local decoder)
            model.eval()
            total_idem, total_tokens = 0, 0
            unique_idx = set()
            with torch.no_grad():
                for batch in dl_info['dataloader']:
                    if isinstance(batch, (list, tuple)): batch = batch[0]
                    x = batch.cuda()
                    idx1 = model.encode(x)
                    windows = model.local_decoder(F.embedding(idx1, model.quantizer.embedding))
                    recon = model.local_decoder.reconstruct_full(windows)
                    idx2 = model.encode(recon)
                    total_idem += (idx1 == idx2).float().sum().item()
                    total_tokens += idx1.numel()
                    unique_idx.update(idx1.cpu().numpy().flatten().tolist())
            print(f"  Local idem: {total_idem/total_tokens*100:.1f}%, codebook: {len(unique_idx)}/{codebook_size}", flush=True)
            del model, opt

    # ── Stage 2: GlobalDecoder finetune ──
    if args.stage in [0, 2]:
        vqvae_dir = ckpt_dir / 'vqvae'

        if (vqvae_dir / 'decoder_done.pt').exists() and args.stage == 0:
            print(f">>> Decoder checkpoint exists, skipping stage 2", flush=True)
        else:
            print(f"\n{'='*60}")
            print(f"  Stage 2: GlobalDecoder ({args.decoder_steps} steps)")
            print(f"{'='*60}", flush=True)

            model = instantiate_from_config(config['vqvae']).cuda()
            pretrain_path = vqvae_dir / 'pretrain.pt'
            model.load_state_dict(torch.load(str(pretrain_path), map_location='cuda')['model'])

            # Freeze encoder + quantizer
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.quantizer.parameters():
                p.requires_grad = False
            model.encoder.eval()
            model.quantizer.eval()

            dec_params = list(model.global_decoder.parameters())
            opt = AdamW(dec_params, lr=args.lr, betas=[0.9, 0.99])
            n_params = sum(p.numel() for p in dec_params)
            print(f"  Decoder params: {n_params:,}", flush=True)

            with tqdm(total=args.decoder_steps, desc='Decoder') as pbar:
                for step in range(1, args.decoder_steps + 1):
                    data = next(dl).cuda()
                    loss = model.forward(data, mode='finetune')
                    loss.backward()
                    clip_grad_norm_(dec_params, 1.0)
                    opt.step()
                    opt.zero_grad()
                    pbar.set_description(f'decoder: {loss.item():.6f}')
                    pbar.update(1)

            # Save as checkpoint-10 (compatible with load_models) + decoder_done marker
            torch.save({'step': args.pretrain_steps + args.decoder_steps,
                        'model': model.state_dict()},
                       str(vqvae_dir / 'checkpoint-10.pt'))
            torch.save({'done': True}, str(vqvae_dir / 'decoder_done.pt'))
            print(f"  Saved checkpoint-10.pt", flush=True)

            # Idempotence check (global decoder)
            model.eval()
            total_idem, total_tokens = 0, 0
            with torch.no_grad():
                for batch in dl_info['dataloader']:
                    if isinstance(batch, (list, tuple)): batch = batch[0]
                    x = batch.cuda()
                    idx1 = model.encode(x)
                    recon = model.decode_from_indices(idx1)
                    idx2 = model.encode(recon)
                    total_idem += (idx1 == idx2).float().sum().item()
                    total_tokens += idx1.numel()
            print(f"  Global idem: {total_idem/total_tokens*100:.1f}%", flush=True)
            del model, opt

    # ── Stage 3: AR Transformer ──
    if args.stage in [0, 3]:
        tf_dir = ckpt_dir / 'transformer'
        os.makedirs(tf_dir, exist_ok=True)

        if (tf_dir / 'checkpoint-10.pt').exists() and args.stage == 0:
            print(f">>> Transformer checkpoint exists, skipping stage 3", flush=True)
        else:
            print(f"\n{'='*60}")
            print(f"  Stage 3: AR Transformer ({args.transformer_steps} steps)")
            print(f"{'='*60}", flush=True)

            # Load VQ-VAE
            vqvae = instantiate_from_config(config['vqvae']).cuda()
            vqvae.load_state_dict(torch.load(
                str(ckpt_dir / 'vqvae' / 'checkpoint-10.pt'), map_location='cuda')['model'])
            vqvae.eval()
            for p in vqvae.parameters():
                p.requires_grad = False

            # Build transformer
            config['transformer']['params']['token_length'] = token_length
            transformer = instantiate_from_config(config['transformer']).cuda()
            opt = AdamW(transformer.parameters(), lr=args.lr, betas=[0.9, 0.99])
            save_cycle = args.transformer_steps // 10

            milestone = 0
            with tqdm(total=args.transformer_steps, desc='Transformer') as pbar:
                for step in range(1, args.transformer_steps + 1):
                    data = next(dl).cuda()
                    with torch.no_grad():
                        indices = vqvae.encode(data)
                    loss = transformer(indices)
                    loss.backward()
                    clip_grad_norm_(transformer.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    pbar.set_description(f'CE: {loss.item():.6f}')
                    pbar.update(1)
                    if step % save_cycle == 0:
                        milestone += 1
                        torch.save({'step': step, 'model': transformer.state_dict()},
                                   str(tf_dir / f'checkpoint-{milestone}.pt'))

            print(f"  Saved transformer", flush=True)
            del vqvae, transformer, opt

    # ── Stage 4: Robust Encoder ──
    if args.stage in [0, 4]:
        robust_dir = ckpt_dir / 'robustencoder'
        os.makedirs(robust_dir, exist_ok=True)

        if (robust_dir / 'checkpoint-best.pt').exists() and args.stage == 0:
            print(f">>> Robust encoder exists, skipping stage 4", flush=True)
        else:
            print(f"\n{'='*60}")
            print(f"  Stage 4: Robust Encoder ({args.robust_steps} steps)")
            print(f"{'='*60}", flush=True)

            from Models.local_vqvae import RobustEncoder, resolve_robust_cfg

            # Load VQ-VAE + Transformer
            vqvae = instantiate_from_config(config['vqvae']).cuda()
            vqvae.load_state_dict(torch.load(
                str(ckpt_dir / 'vqvae' / 'checkpoint-10.pt'), map_location='cuda')['model'])
            vqvae.eval()

            tf_ckpt = torch.load(str(ckpt_dir / 'transformer' / 'checkpoint-10.pt'), map_location='cuda')
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

            auto_norm = dataset_obj.auto_norm

            # Generate training data with multiple seeds
            from Models.transformer_ar import sample_mts
            gen_seeds = [int(s) for s in args.gen_seeds.split(',')]

            all_samples_raw = []
            all_target_idx = []

            for gs in gen_seeds:
                print(f"  Generating 10000 samples (seed={gs})...", flush=True)
                seed_everything(gs)
                samples = sample_mts(transformer, vqvae, num=10000, size_every=2000,
                                     shape=[ws, feature_size])

                if auto_norm:
                    samples_01 = unnormalize_to_zero_to_one(samples)
                else:
                    samples_01 = samples.copy()
                samples_raw = dataset_obj.scaler.inverse_transform(
                    samples_01.reshape(-1, feature_size)).reshape(samples.shape)
                all_samples_raw.append(samples_raw)

                # Target: raw → MinMaxScaler fit on this 10k → [-1,1] → encode
                # This matches the detect pipeline (npy → dataloader → scaler)
                B, T, D = samples_raw.shape
                scaler_clean = MinMaxScaler()
                clean_scaled = scaler_clean.fit_transform(
                    samples_raw.reshape(-1, D)).reshape(B, T, D)
                if auto_norm:
                    clean_scaled = normalize_to_neg_one_to_one(clean_scaled)

                seed_target = []
                with torch.no_grad():
                    for i in range(0, len(clean_scaled), 2000):
                        batch = torch.tensor(clean_scaled[i:i+2000], dtype=torch.float32).cuda()
                        seed_target.append(vqvae.encode(batch).cpu())
                all_target_idx.append(torch.cat(seed_target, dim=0).numpy())

            samples_raw = np.concatenate(all_samples_raw, axis=0)
            target_idx = np.concatenate(all_target_idx, axis=0)
            print(f"  Total: {samples_raw.shape[0]} samples, {target_idx.shape[1]} tokens", flush=True)

            # Apply attacks. The grid is Utils/greenlist.ROBUST_TRAIN_ATTACK_GRID — the
            # same seven conditions as before, defined in one place so no entry point
            # spells an attack name itself (crop vs crop_var vs crop_zero is a live trap).

            all_scaled = []
            all_targets = []

            n_per_seed = 10000
            n_seeds = len(gen_seeds)

            for atk_name, atk_factor in ROBUST_TRAIN_ATTACK_GRID:
                if atk_name is None:
                    print(f"  Clean...", flush=True)
                else:
                    print(f"  Attack: {atk_name}-{atk_factor}...", flush=True)

                for si in range(n_seeds):
                    start = si * n_per_seed
                    end = start + n_per_seed
                    chunk_raw = samples_raw[start:end].copy()
                    B, T, D = chunk_raw.shape

                    if atk_name is not None:
                        np.random.seed(123)
                        chunk_raw = add_attack(chunk_raw, atk_name, atk_factor)

                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(chunk_raw.reshape(-1, D)).reshape(B, T, D)
                    if auto_norm:
                        scaled = normalize_to_neg_one_to_one(scaled)
                    all_scaled.append(scaled.astype(np.float32))
                    all_targets.append(target_idx[start:end].copy())

            X = np.concatenate(all_scaled, axis=0)
            Y = np.concatenate(all_targets, axis=0)
            n_total = X.shape[0]
            n_val = int(n_total * 0.1)
            n_train = n_total - n_val
            perm = np.random.RandomState(42).permutation(n_total)
            X_train, Y_train = X[perm[:n_train]], Y[perm[:n_train]]
            X_val, Y_val = X[perm[n_train:]], Y[perm[n_train:]]
            print(f"  Train: {n_train}, Val: {n_val}", flush=True)

            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_train), torch.tensor(Y_train, dtype=torch.long))
            val_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_val), torch.tensor(Y_val, dtype=torch.long))
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            rdl = cycle(train_loader)

            # Robust encoder — architecture from config['robust_encoder'],
            # overridable via --robust_{d_model,n_layers,n_heads}.
            robust_d, robust_nl, robust_heads = resolve_robust_cfg(
                config, args.robust_d_model, args.robust_n_layers, args.robust_n_heads)
            robust_enc = RobustEncoder(
                feature_size=feature_size, d_model=robust_d, n_heads=robust_heads,
                n_layers=robust_nl, rf=rf, stride=stride, n_classes=codebook_size
            ).cuda()
            n_params = sum(p.numel() for p in robust_enc.parameters())
            opt = AdamW(robust_enc.parameters(), lr=args.robust_lr, betas=(0.9, 0.99))
            print(f"  Robust arch: d_model={robust_d}, n_layers={robust_nl}, "
                  f"n_heads={robust_heads}", flush=True)
            print(f"  Robust params: {n_params:,}", flush=True)

            best_val_acc = 0.
            best_state = None
            best_step = 0

            @torch.no_grad()
            def evaluate_robust():
                robust_enc.eval()
                correct, total, total_loss, nb = 0, 0, 0, 0
                for xb, yb in val_loader:
                    xb, yb = xb.cuda(), yb.cuda()
                    logits = robust_enc(xb)
                    loss = F.cross_entropy(logits.reshape(-1, codebook_size), yb.reshape(-1))
                    correct += (logits.argmax(dim=-1) == yb).float().sum().item()
                    total += yb.numel()
                    total_loss += loss.item()
                    nb += 1
                robust_enc.train()
                return correct / total, total_loss / nb

            with tqdm(total=args.robust_steps, desc='Robust') as pbar:
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
                    pbar.set_description(f'CE:{loss.item():.4f} acc:{acc:.3f}')
                    pbar.update(1)

                    if step % args.robust_val_every == 0:
                        val_acc, val_loss = evaluate_robust()
                        pbar.write(f'  step {step}: val_acc={val_acc:.4f} val_CE={val_loss:.4f}')
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_step = step
                            best_state = {k: v.clone() for k, v in robust_enc.state_dict().items()}

                    if step % 10000 == 0:
                        torch.save({'step': step, 'model': robust_enc.state_dict()},
                                   str(robust_dir / f'checkpoint-{step // 10000}.pt'))

            if best_state is not None:
                torch.save({'step': best_step, 'model': best_state},
                           str(robust_dir / 'checkpoint-best.pt'))

            val_acc, val_loss = evaluate_robust()
            results = {'best_step': best_step, 'best_val_acc': best_val_acc,
                       'final_val_acc': val_acc, 'params': n_params}
            with open(robust_dir / 'final_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Best val_acc: {best_val_acc:.4f} (step {best_step})", flush=True)

    print(f"\n{'='*60}")
    print(f"  ALL DONE: {ds} w={ws}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

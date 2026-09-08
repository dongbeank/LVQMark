# Adapted from https://github.com/soizhiwen/TimeWak

"""
Two-stage training solver for VQ-TS.

Stage 1: Train VQ-VAE to learn discrete token representations of time series.
Stage 2: Train Autoregressive Transformer to model token distributions.
"""

import os
import json
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import get_model_parameters_info


def cycle(dl):
    while True:
        for data in dl:
            yield data


def _require_results_folder(results_folder, who):
    """Checkpoint directory for a trainer. results_folder: str. Returns str.

    Raises ``ValueError`` rather than falling back to a config key: a silent fallback
    here writes checkpoints somewhere the entry point will not look for them later.
    """
    if not results_folder:
        raise ValueError(
            f"{who} needs an explicit results_folder "
            f"(Utils.path_utils.get_ckpt_dir); the shipped configs carry no "
            f"solver.results_folder.")
    return results_folder


class VQVAETrainer(object):
    """Stage 1: Train VQ-VAE for time series tokenization."""

    def __init__(self, config, args, model, dataloader, logger=None,
                 results_folder=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.train_num_steps = config['solver']['stage1']['max_epochs']
        self.gradient_accumulate_every = config['solver']['stage1'].get('gradient_accumulate_every', 2)
        self.save_cycle = config['solver']['stage1'].get('save_cycle', 500)
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.config = config
        self.logger = logger

        # results_folder comes from Utils/path_utils via the entry point; the shipped
        # configs carry no solver.results_folder key.
        self.results_folder = Path(
            _require_results_folder(results_folder, 'VQVAETrainer'))
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver']['stage1'].get('base_lr', 3.0e-4)
        self.opt = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=start_lr, betas=[0.9, 0.99])

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Stage 1: Start VQ-VAE training...')

        loss_history = []

        with tqdm(initial=step, total=self.train_num_steps, desc='Stage1-VQVAE') as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'VQ-VAE loss: {total_loss:.6f}')
                loss_history.append({'step': self.step + 1, 'loss': total_loss})

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='stage1/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        # Save loss history
        with open(self.results_folder / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f)

        print('Stage 1 (VQ-VAE) training complete')
        if self.logger is not None:
            self.logger.log_info('Stage 1 done, time: {:.2f}'.format(time.time() - tic))


class TransformerTrainer(object):
    """Stage 2: Train Autoregressive Transformer on VQ tokens."""

    def __init__(self, config, args, transformer, vqvae, dataloader, logger=None,
                 results_folder=None):
        super().__init__()
        self.transformer = transformer
        self.vqvae = vqvae
        self.vqvae.eval()  # Freeze VQ-VAE
        self.device = next(transformer.parameters()).device
        self.train_num_steps = config['solver']['stage2']['max_epochs']
        self.gradient_accumulate_every = config['solver']['stage2'].get('gradient_accumulate_every', 2)
        self.save_cycle = config['solver']['stage2'].get('save_cycle', 500)
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.config = config
        self.logger = logger

        # See the note in VQVAETrainer.__init__.
        self.results_folder = Path(
            _require_results_folder(results_folder, 'TransformerTrainer'))
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver']['stage2'].get('base_lr', 3.0e-4)
        self.opt = AdamW(filter(lambda p: p.requires_grad, self.transformer.parameters()),
                         lr=start_lr, betas=[0.9, 0.99])

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.transformer)))
        self.log_frequency = 100

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.transformer.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.transformer.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Stage 2: Start Transformer training...')

        loss_history = []

        with tqdm(initial=step, total=self.train_num_steps, desc='Stage2-Transformer') as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    # Encode time series to tokens with the VQ-VAE, no gradients
                    with torch.no_grad():
                        indices = self.vqvae.encode(data)  # (B, L)

                    # Train transformer on token sequences
                    loss = self.transformer(indices)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'Transformer CE loss: {total_loss:.6f}')
                loss_history.append({'step': self.step + 1, 'loss': total_loss})

                clip_grad_norm_(self.transformer.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='stage2/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        # Save loss history
        with open(self.results_folder / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f)

        print('Stage 2 (Transformer) training complete')
        if self.logger is not None:
            self.logger.log_info('Stage 2 done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None, temperature=1.0, top_k=None,
               watermark_mask=None, watermark_delta=0.0, alt_context=False, alt_position=False):
        """Generate time series samples."""
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')

        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.transformer.generate_mts(
                batch_size=size_every,
                vqvae=self.vqvae,
                temperature=temperature,
                top_k=top_k,
                watermark_mask=watermark_mask,
                watermark_delta=watermark_delta,
                alt_context=alt_context,
                alt_position=alt_position,
            )
            samples = np.vstack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        samples = samples[:num]
        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples


def sample_mts(transformer, vqvae, num, size_every, shape, temperature=1.0, top_k=None,
               watermark_mask=None, watermark_delta=0.0, alternating_partition=False):
    """Sample ``num`` SDformer windows. Returns (num, T, D) float64 numpy, model scale.

    transformer: a loaded ``transformer_ar_vqvae.AutoregressiveTransformer``, eval mode.
    vqvae: a loaded ``vqvae.VQVAE``, eval mode. num: total windows. size_every: batch
    size per ``generate_mts`` call. shape: [T, D]. watermark_mask: (K+1, K) bool green-list
    table on the model's device, or None. watermark_delta: float, or a per-position list
    of length L (the unbiased warm-up schedule). alternating_partition: bool, flip the
    green set on odd token positions (preprint Eq. 9).

    Body copied from ``TransformerTrainer.sample`` minus the logger, so that generation
    needs neither a trainer object (constructing one ``makedirs``es a checkpoint
    directory) nor a ``solver:`` config block. Mirrors ``Models.transformer_ar.sample_mts``
    on the L-VQVAE side. ``alternating_partition`` is this repo's name for the vendored
    model's ``alt_position`` kwarg; the translation happens here and nowhere else.
    """
    samples = np.empty([0, shape[0], shape[1]])
    num_cycle = int(num // size_every) + 1

    for _ in range(num_cycle):
        sample = transformer.generate_mts(
            batch_size=size_every,
            vqvae=vqvae,
            temperature=temperature,
            top_k=top_k,
            watermark_mask=watermark_mask,
            watermark_delta=watermark_delta,
            alt_context=False,
            alt_position=alternating_partition,
        )
        samples = np.vstack([samples, sample.detach().cpu().numpy()])
        torch.cuda.empty_cache()

    return samples[:num]

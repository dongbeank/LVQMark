# Adapted from https://github.com/soizhiwen/TimeWak

import os
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None,
                 results_folder=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config["solver"]["max_epochs"]
        self.gradient_accumulate_every = config["solver"]["gradient_accumulate_every"]
        self.save_cycle = config["solver"]["save_cycle"]
        self.dl = cycle(dataloader["dataloader"])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        # results_folder comes from Utils/path_utils via the entry point. The shipped
        # configs carry no solver.results_folder key, so a missing argument raises.
        if not results_folder:
            raise ValueError(
                "Trainer needs an explicit results_folder "
                "(Utils.path_utils.get_ckpt_dir); the shipped configs carry no "
                "solver.results_folder.")
        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config["solver"].get("base_lr", 1.0e-4)
        ema_decay = config["solver"]["ema"]["decay"]
        ema_update_every = config["solver"]["ema"]["update_interval"]

        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=[0.9, 0.96],
        )
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        sc_cfg = config["solver"]["scheduler"]
        sc_cfg["params"]["optimizer"] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                "Save current model to {}".format(
                    str(self.results_folder / f"checkpoint-{milestone}.pt")
                )
            )
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f"checkpoint-{milestone}.pt"))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                "Resume from {}".format(
                    str(self.results_folder / f"checkpoint-{milestone}.pt")
                )
            )
        device = self.device
        data = torch.load(
            str(self.results_folder / f"checkpoint-{milestone}.pt"), map_location=device
        )
        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(
                "{}: start training...".format(self.args.name), check_primary=False
            )

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f"loss: {total_loss:.6f}")

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(
                            tag="train/loss",
                            scalar_value=total_loss,
                            global_step=self.step,
                        )

                pbar.update(1)

        print("training complete")
        if self.logger is not None:
            self.logger.log_info(
                "Training done, time: {:.2f}".format(time.time() - tic)
            )

    def sample(self, num, size_every, shape=None, watermark=None, save_dir=None):
        """Sample ``num`` windows. Returns (num, T, D) float64 numpy, model scale.

        num: total windows. size_every: batch size per ``generate_mts`` call.
        shape: [T, D]. watermark: "" (none) or a method name; defaults to
        ``self.args.watermark``. save_dir: where TR's ``gt_patch.npy`` and
        ``watermarking_mask.npy`` go; defaults to ``self.args.save_dir``.

        Sets ``self.n_generate_calls`` and ``self.n_nan_dropped``. The loop below keeps
        drawing until ``num`` non-NaN windows exist, so the number of ``generate_mts``
        calls -- and therefore the RNG consumed -- is data-dependent. Recording the two
        counts in run_config.json is what makes a rerun's stream auditable.
        """
        watermark = self.args.watermark if watermark is None else watermark
        save_dir = self.args.save_dir if save_dir is None else save_dir
        self.n_generate_calls = 0
        self.n_nan_dropped = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to sample...")
        samples = np.empty([0, shape[0], shape[1]])

        if watermark == "TR":
            gt_patches = np.empty([0, 1, shape[0], shape[1]])
            watermarking_masks = np.empty([0, 1, shape[0], shape[1]], dtype=bool)

        while len(samples) < num:
            self.n_generate_calls += 1
            if watermark == "TR":
                sample, gt_patch, watermarking_mask = self.ema.ema_model.generate_mts(
                    args=self.args,
                    batch_size=size_every,
                    watermark=watermark,
                )
                # Filter out NaN samples
                nan_indices = torch.isnan(sample).any(dim=(1, 2))
                self.n_nan_dropped += int(nan_indices.sum().item())
                sample = sample[~nan_indices]
                nan_indices = nan_indices.detach().cpu().numpy()
                gt_patch = gt_patch[~nan_indices]
                watermarking_mask = watermarking_mask[~nan_indices]

                samples = np.vstack([samples, sample.detach().cpu().numpy()])
                gt_patches = np.vstack([gt_patches, gt_patch])
                watermarking_masks = np.vstack(
                    [watermarking_masks, watermarking_mask]
                )
                samples = samples[:num]
                gt_patches = gt_patches[:num]
                watermarking_masks = watermarking_masks[:num]

            else:
                sample = self.ema.ema_model.generate_mts(
                    args=self.args,
                    batch_size=size_every,
                    watermark=watermark,
                )
                # Filter out NaN samples
                nan_indices = torch.isnan(sample).any(dim=(1, 2))
                self.n_nan_dropped += int(nan_indices.sum().item())
                sample = sample[~nan_indices]
                samples = np.vstack([samples, sample.detach().cpu().numpy()])
                samples = samples[:num]
            torch.cuda.empty_cache()

        if watermark == "TR":
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/gt_patch.npy", gt_patches)
            np.save(f"{save_dir}/watermarking_mask.npy", watermarking_masks)

        if self.logger is not None:
            self.logger.log_info(
                "Sampling done, time: {:.2f}".format(time.time() - tic)
            )
        return samples

    def detect(self, raw_dataloader, shape=None, watermark=None):
        """Invert a dataloader of windows back to latents. Returns (N, T, D) numpy.

        raw_dataloader: yields (B, T, D) tensors. shape: [T, D]. watermark: method name;
        defaults to ``self.args.watermark``. It selects the inversion, which must match
        the sampler the windows were drawn with -- BDIA for TimeWak, DDIM otherwise.
        """
        watermark = self.args.watermark if watermark is None else watermark
        noises = np.empty([0, shape[0], shape[1]])

        for x in raw_dataloader:
            if watermark in ["TimeWak", "SpatBDIA"]:
                noise = self.model.inverse_bdia_sample(x)
            else:
                noise = self.model.inverse_fast_sample(x)
            noises = np.vstack([noises, noise.detach().cpu().numpy()])

        return noises

    def restore(
        self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50
    ):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to restore...")
        model_kwargs = {}
        model_kwargs["coef"] = coef
        model_kwargs["learning_rate"] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = self.ema.ema_model.fast_sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                )

            samples = np.vstack([samples, sample.detach().cpu().numpy()])
            reals = np.vstack([reals, x.detach().cpu().numpy()])
            masks = np.vstack([masks, t_m.detach().cpu().numpy()])

        if self.logger is not None:
            self.logger.log_info(
                "Imputation done, time: {:.2f}".format(time.time() - tic)
            )
        return samples, reals, masks
        # return samples

"""Experiment-key derivation and artifact paths — one definition for all three entry points.

What this module is for
-----------------------
It holds the single definition of every experiment key and artifact path, so that the entry
points cannot spell a naming rule differently. A disagreement would be silent:
``generate.py`` skips generation when the sample file already exists, so a key that fails to
distinguish two runs makes the second run *report the first run's numbers*.

The artifact layout
-------------------
Every artifact tree carries a ``generator`` level first, so that L-VQVAE and the baseline
generators (SDformer, Diffusion-TS) can share one set of roots without their dataset
directories colliding::

    checkpoints/{generator}/{dataset}/{exp}/{window}/{stage}/
    outputs/{generator}/{dataset}/{run}/{window}/
    results/{generator}/{dataset}/{run}/

``generator`` defaults to :data:`DEFAULT_GENERATOR` (``"lvqvae"``), which is what the three
L-VQVAE entry points use; the ``cli/*_baselines.py`` entry points pass their own. It is a
*path level* and not part of the key strings below, which describe the run's
hyperparameters only.

The key scheme
--------------
A directory name, not a hash — it has to be readable in ``ls``.

``exp``  (identifies the **weights**; used for ``checkpoints/``)::

    d{d_model}_cb{codebook_size}_st{stride}_ar{n_embd}_l{n_layer}_h{n_head}_trainseed{seed}
    e.g. d512_cb512_st4_ar1024_l2_h8_trainseed12345

``run``  (identifies the **samples**; used for ``outputs/`` and ``results/``)::

    {exp}_delta{delta}_genseed{gen_seed}_wmseed{watermark_seed}
    e.g. d512_cb512_st4_ar1024_l2_h8_trainseed12345_delta10_genseed12345_wmseed12345

What the keys do and do not carry:

* **Training seed** is in ``exp``: same architecture + different seed = different weights, so
  without it two such runs would share a checkpoint directory.
* **``n_layer`` / ``n_head``** are in ``exp``: they change the transformer's weights and are the
  first thing an ablation varies.
* **Generation seed and watermark seed** are in ``run`` but not in ``exp``: they change the
  samples, not the weights. The clean (NoWM) pool lives in the same ``run`` directory as the
  watermarked pool it is the control for, so ``delta`` and ``wmseed`` appear in its path too
  even though they do not affect it — that is what keeps a WM/NoWM pair together.
* **Robust-encoder architecture is NOT in ``exp``.** It does not affect stages 1-3, so
  keying the shared directory on it would force a full retrain whenever the detector's
  width changed. A mismatch there is *loud*: ``generate.py`` recovers
  ``d_model`` / ``n_layers`` from the checkpoint itself and prints a ``[WARN]`` when the config
  disagrees.
* **lr, step counts, and the rest of the config are NOT in the name.** They would make the name
  unreadable, and a hash would make it uninvertible. They go into ``run_config.json`` instead
  (:func:`write_run_config`), which dumps the *resolved* configuration next to the artifacts
  and prints a loud ``[WARN]`` when a rerun in the same directory resolves differently. So the
  name is the human index and the JSON is the complete record.

Layer: ``Utils/`` — imports nothing from ``Data/`` or ``Models/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# Repo-relative artifact roots. Every path in the pipeline hangs off one of these, and
# ``cli/_bootstrap.py`` makes the repository root the working directory, so they always
# resolve to the same trees.
CKPT_BASE = "checkpoints"
OUTPUT_BASE = "outputs"
RESULTS_BASE = "results"

# First level under each root: which generator produced the artifacts. This is the value the
# L-VQVAE entry points use; the ``cli/*_baselines.py`` entry points pass
# "sdformer" / "diffusionts".
DEFAULT_GENERATOR = "lvqvae"

RUN_CONFIG_NAME = "run_config.json"

# The one place these three defaults are written down. Every entry point's argparse default
# points here, so "all-default arguments" produce the same fully descriptive path everywhere.
DEFAULT_TRAIN_SEED = 12345
DEFAULT_GEN_SEED = 12345
DEFAULT_WATERMARK_SEED = 12345
DEFAULT_EVAL_SEED = 12345


def stride_for_window(window: int) -> int:
    """Encoder stride implied by the window length. window: int (time steps). Returns int."""
    return 2 if window == 24 else 4


def format_delta(delta: float) -> str:
    """Render a watermark delta for a directory name: 10.0 -> '10', 7.5 -> '7.5'.

    ``%g`` rather than ``int()`` so that two deltas that differ only fractionally cannot
    collide in one directory (``int()`` mapped 10.0 and 10.5 to the same 'delta10').
    Integral deltas render exactly as before, so no existing directory name changes.
    """
    return f"{float(delta):g}"


def derive_exp_name(config: Dict[str, Any], stride: int, train_seed: int) -> str:
    """Experiment key identifying the trained weights. Returns a directory-safe str.

    config: parsed yaml (needs ``vqvae.params`` and ``transformer.params``).
    stride: int, from :func:`stride_for_window`. train_seed: int, ``train.py --seed``.

    A missing key raises ``KeyError`` rather than falling back to a default, which would
    let the name diverge from the yaml without saying so.
    """
    vq = config["vqvae"]["params"]
    tr = config["transformer"]["params"]
    return (
        f"d{vq['d_model']}"
        f"_cb{vq['codebook_size']}"
        f"_st{stride}"
        f"_ar{tr['n_embd']}"
        f"_l{tr['n_layer']}"
        f"_h{tr['n_head']}"
        f"_trainseed{train_seed}"
    )


def derive_run_name(exp: str, delta: float, gen_seed: int, watermark_seed: int) -> str:
    """Run key identifying a generated sample set. Returns a directory-safe str.

    exp: from :func:`derive_exp_name` (or a hand-passed ``--exp``). delta: watermark logit
    bias. gen_seed: sampling seed (``generate.py --seed``). watermark_seed: green-list table
    seed (``generate.py --watermark_seed``).
    """
    return (
        f"{exp}"
        f"_delta{format_delta(delta)}"
        f"_genseed{gen_seed}"
        f"_wmseed{watermark_seed}"
    )


def get_ckpt_dir(dataset: str, exp: str, window: int, *,
                 generator: str = DEFAULT_GENERATOR) -> str:
    """``checkpoints/{generator}/{dataset}/{exp}/{window}`` — the four stages write subdirs.

    generator: str, see :data:`DEFAULT_GENERATOR`. Keyword-only so that a caller adding it
    can never land it in the ``window`` slot.
    """
    return os.path.join(CKPT_BASE, generator, dataset, exp, str(window))


def get_train_output_dir(dataset: str, exp: str, window: int, *,
                         generator: str = DEFAULT_GENERATOR) -> str:
    """``outputs/{generator}/{dataset}/{exp}/{window}`` — training's ground-truth dump.

    No ``run`` suffix: this content depends on the dataset and the window only, and training
    never generates watermarked samples.
    """
    return os.path.join(OUTPUT_BASE, generator, dataset, exp, str(window))


def get_output_dir(dataset: str, run: str, window: int, *,
                   generator: str = DEFAULT_GENERATOR) -> str:
    """``outputs/{generator}/{dataset}/{run}/{window}`` — samples, detect CSVs, ground truth."""
    return os.path.join(OUTPUT_BASE, generator, dataset, run, str(window))


def get_results_dir(dataset: str, run: str, *,
                    generator: str = DEFAULT_GENERATOR) -> str:
    """``results/{generator}/{dataset}/{run}`` — the metric CSVs ``evaluate.py`` writes.

    The curated ``results/*.md`` records stay at the root of ``results/``: they are written by
    hand, span generators, and are not addressed by this function.
    """
    return os.path.join(RESULTS_BASE, generator, dataset, run)


def resolve_vqvae_checkpoint(ckpt_dir: str) -> str:
    """Path of the stage-2 VQ-VAE checkpoint. Raises ``FileNotFoundError`` if it is missing.

    ckpt_dir: str from :func:`get_ckpt_dir`. Returns ``{ckpt_dir}/vqvae/checkpoint-10.pt``.

    There is no fallback to ``vqvae/pretrain.pt``. That file is the stage-1 checkpoint: its
    global decoder is randomly initialised, but its keys match, so ``load_state_dict`` would
    succeed and generation would produce plausible-looking garbage that no downstream check
    catches. Stage-1 weights are never the intended model here, so a missing stage-2
    checkpoint raises rather than warning.
    """
    stage2 = os.path.join(ckpt_dir, "vqvae", "checkpoint-10.pt")
    if os.path.exists(stage2):
        return stage2
    stage1 = os.path.join(ckpt_dir, "vqvae", "pretrain.pt")
    stage1_state = "exists" if os.path.exists(stage1) else "is also missing"
    raise FileNotFoundError(
        f"Stage-2 VQ-VAE checkpoint not found: {stage2}\n"
        f"  The stage-1 checkpoint {stage1} {stage1_state}, and this used to be loaded "
        f"instead without a message. Its global decoder is randomly initialised, so "
        f"generation would have succeeded and produced garbage.\n"
        f"  Fix: run stage 2 (`python -u cli/train.py --config_file <cfg> --dataset <ds> "
        f"--window_size <w> --gpu <n> --stage 2`), or point --exp at a trained experiment."
    )


def _flatten(payload: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten a nested dict to ``{'a.b': repr}`` so two dumps can be diffed key by key."""
    flat: Dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = repr(value)
    return flat


def write_run_config(directory: str, payload: Dict[str, Any],
                     filename: str = RUN_CONFIG_NAME) -> str:
    """Dump the *resolved* run configuration next to the artifacts. Returns the file path.

    directory: str, created if absent. payload: JSON-serialisable dict (args, derived values,
    the full config, torch/GPU provenance).

    The directory name carries only the keyed fields (see the module docstring); everything
    else that determines a number lives here. If a dump already exists and disagrees, the
    differing keys are printed as a ``[WARN]`` before it is overwritten — a rerun with a
    different lr or step count in an existing directory is then loud instead of silent.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        try:
            with open(path) as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict):
            old, new = _flatten(previous), _flatten(payload)
            differing = sorted(
                key for key in set(old) | set(new)
                if old.get(key, "<absent>") != new.get(key, "<absent>")
            )
            # argv/timestamps differ on every rerun and say nothing about the numbers.
            differing = [k for k in differing if not k.startswith(("argv", "args.gpu"))]
            if differing:
                print(f"  [WARN] {path} was written by a run with a different resolved "
                      f"configuration; the artifacts in this directory may be a mixture.",
                      flush=True)
                for key in differing:
                    print(f"    [WARN] {key}: {old.get(key, '<absent>')} -> "
                          f"{new.get(key, '<absent>')}", flush=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return path


def torch_provenance(gpu: Optional[int] = None) -> Dict[str, Any]:
    """Torch / GPU provenance for a ``run_config.json`` payload.

    The GPU *architecture* is part of an experiment's identity in this repo — two different
    architectures can give different numbers from the same seed — so it is recorded with every
    run rather than reconstructed afterwards.
    """
    import torch  # local import: this module must stay importable without a GPU present

    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "<unset>"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if gpu is not None and torch.cuda.is_available():
        info["gpu_index"] = gpu
        info["gpu_name"] = torch.cuda.get_device_name(gpu)
        major, minor = torch.cuda.get_device_capability(gpu)
        info["gpu_capability"] = f"sm_{major}{minor}"
    return info

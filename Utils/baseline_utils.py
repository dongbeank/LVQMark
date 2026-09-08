"""Shared leaf helpers for the baseline generators (SDformer, Diffusion-TS).

What this module is for
-----------------------
``train_baselines.py`` / ``generate_baselines.py`` / ``evaluate_baselines.py`` are three
independent entry points: there is no ``baselines/`` package, and they may not import the
L-VQVAE ``train.py`` / ``generate.py`` / ``evaluate.py``. Everything the three
of them agree on — which generators exist, which watermark methods each carries, how an
``exp`` / ``run`` key is spelled, which key ``get_zscore`` must be called with — therefore
has to live in exactly one place below them. This is that place.

Layer: ``Utils/`` — a leaf. It imports ``numpy``, ``torch`` (lazily, for the TR key) and
the sibling leaves ``Utils.path_utils`` / ``Utils.greenlist`` / ``Utils.watermark_utils``.
It imports nothing from ``Models/``, ``Data/`` or any entry point.

The artifact layout is the one ``Utils/path_utils`` already defines; this module only adds
the two key-derivation rules the baselines need, because ``derive_exp_name`` there encodes
the L-VQVAE architecture fields and the baselines have different ones. Nothing here is
reached by ``generator="lvqvae"``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from Utils.greenlist import WM_METHODS
from Utils.path_utils import derive_run_name as _derive_lvqvae_run_name
from Utils.path_utils import format_delta

# ── The generators and what each carries ─────────────────────────────────────────────────

GENERATORS: Tuple[str, ...] = ("sdformer", "diffusionts")

#: The green-list (token) family lives on SDformer; the initial-latent family on Diffusion-TS.
#: One list per generator so that no entry point invents a method name of its own.
TOKEN_WM_METHODS: Tuple[str, ...] = tuple(WM_METHODS)            # LVQMark
LATENT_WM_METHODS: Tuple[str, ...] = ("TimeWak", "TR", "GS")

WM_METHODS_BY_GENERATOR: Dict[str, Tuple[str, ...]] = {
    "sdformer": TOKEN_WM_METHODS,
    "diffusionts": LATENT_WM_METHODS,
}

#: Method string to hand ``Utils.watermark_metrics.get_zscore``. It is not cosmetic: the
#: function branches on ``method == "tr"`` and computes TR's z by the paper's definition
#: (|mean_NoWM - mean| / sigma_NoWM, no /sqrt(n)) instead of the pooled statistic every
#: other method uses. Passing "TR" here would silently score TR on the wrong scale.
#: Anything absent from this map is passed through unchanged (the token family).
ZSCORE_METHOD_KEY: Dict[str, str] = {"TimeWak": "timewak", "TR": "tr", "GS": "gs"}

#: ``Utils.watermark_metrics.eval_{TR,GS,TimeWak}`` write ``<this>.csv`` into the directory
#: they are given. That module is used unmodified, so the entry point renames the file to
#: ``<wm_method>.csv`` afterwards — one CSV shape for both families, which is what lets one
#: ``evaluate_baselines.py`` loop read them all.
LATENT_SCORER_CSV_STEM: Dict[str, str] = {"TimeWak": "timewak", "TR": "tr", "GS": "gs"}

#: Methods for which a wrong-key control (``--detect_key``) is expressible without changing
#: ``Utils/watermark_metrics.py``:
#:   * TR   — its key is the seed the ring pattern ``gt_patch`` was drawn under, and this
#:            entry point both draws it and hands it to ``eval_TR``.
#:   * GS   — NOT here: ``eval_GS`` hardcodes ``torch.manual_seed(217)`` as the detector key.
#:   * TimeWak — NOT here: its detector has no key at all (the chained-hash permutations are
#:            seeded by the position index, and the bit levels are recovered by quantiles).
#: The token family takes its key from ``Utils.greenlist``, so every method there is keyed.
KEYED_LATENT_WM_METHODS: Tuple[str, ...] = ("TR",)

#: Ground-truth file prefix written by the dataset class, per dataset key. Same map
#: ``cli/evaluate.py`` uses; the baselines read the same ``samples/`` dumps.
GT_PREFIX: Dict[str, str] = {
    "stocks": "stock", "etth": "etth", "energy": "energy", "fmri": "fMRI",
}

#: The latent family's key: 217, the value its detectors in ``Utils.watermark_metrics``
#: expect. Exposed as ``--watermark_seed`` and recorded in the run key as ``wmseed217``.
#: The token family uses 12345 (``Utils.path_utils.DEFAULT_WATERMARK_SEED``).
DEFAULT_LATENT_WATERMARK_SEED = 217


def methods_for(generator: str) -> Tuple[str, ...]:
    """Watermark methods available on ``generator``. Returns a tuple of names."""
    try:
        return WM_METHODS_BY_GENERATOR[generator]
    except KeyError:
        raise ValueError(
            f"unknown generator {generator!r}; expected one of {GENERATORS}") from None


def zscore_method_key(wm_method: str) -> str:
    """Method string for ``get_zscore``. wm_method: e.g. "TR". Returns e.g. "tr"."""
    return ZSCORE_METHOD_KEY.get(wm_method, wm_method)


# ── Experiment / run keys ────────────────────────────────────────────────────────────────
#
# Same grammar as Utils/path_utils: `exp` names everything that determines the WEIGHTS,
# `run` everything that determines the SAMPLES. The generator itself is a path level, never
# a field of the key (Utils/path_utils module docstring), so neither string carries it.


def derive_sdformer_exp_name(config: Dict[str, Any], train_seed: int) -> str:
    """SDformer weight key. config: parsed yaml. train_seed: int. Returns a directory name.

    ``hid{hidden_dim}_cb{codebook_size}_ds{downsample_rate}_{quantizer_type}
    _ar{n_embd}_l{n_layer}_h{n_head}_trainseed{seed}`` — the fields ``Config_vqvae`` exposes
    that change the trained weights. ``quantizer_type`` is in the name because cosine and L2
    are different models with the same shapes, so nothing downstream would catch a mix-up.
    ``embed_loss_weight``, ``ema_decay``, ``n_resblocks``, lr and step counts are not in the
    name (they go to ``run_config.json``); ``KeyError`` on a missing field is deliberate.
    """
    vq = config["vqvae"]["params"]
    tr = config["transformer"]["params"]
    return (
        f"hid{vq['hidden_dim']}"
        f"_cb{vq['codebook_size']}"
        f"_ds{vq['downsample_rate']}"
        f"_{vq['quantizer_type']}"
        f"_ar{tr['n_embd']}"
        f"_l{tr['n_layer']}"
        f"_h{tr['n_head']}"
        f"_trainseed{train_seed}"
    )


def derive_diffusionts_exp_name(config: Dict[str, Any], train_seed: int) -> str:
    """Diffusion-TS weight key. config: parsed yaml. train_seed: int. Returns a dir name.

    ``dm{d_model}_enc{n_layer_enc}_dec{n_layer_dec}_t{timesteps}_trainseed{seed}``.
    ``sampling_timesteps`` is *not* here: it selects DDIM/BDIA at sampling time and changes
    no weight, so it belongs to the run key instead.
    """
    m = config["model"]["params"]
    return (
        f"dm{m['d_model']}"
        f"_enc{m['n_layer_enc']}"
        f"_dec{m['n_layer_dec']}"
        f"_t{m['timesteps']}"
        f"_trainseed{train_seed}"
    )


def derive_exp_name(generator: str, config: Dict[str, Any], train_seed: int) -> str:
    """Weight key for ``generator``. Returns a directory name. Dispatches on the generator."""
    if generator == "sdformer":
        return derive_sdformer_exp_name(config, train_seed)
    if generator == "diffusionts":
        return derive_diffusionts_exp_name(config, train_seed)
    raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATORS}")


def derive_diffusionts_run_name(exp: str, sampling_timesteps: int, gen_seed: int,
                                watermark_seed: int) -> str:
    """Diffusion-TS sample key. Returns ``{exp}_s{steps}_genseed{s}_wmseed{s}``.

    There is no ``delta``: the latent family biases nothing, it replaces the initial latent.
    ``sampling_timesteps`` takes delta's slot because it is the one knob that changes the
    samples without changing the weights (it also picks DDIM vs the full chain).
    """
    return (
        f"{exp}"
        f"_s{int(sampling_timesteps)}"
        f"_genseed{gen_seed}"
        f"_wmseed{watermark_seed}"
    )


def derive_run_name(generator: str, exp: str, *, gen_seed: int, watermark_seed: int,
                    delta: Optional[float] = None,
                    sampling_timesteps: Optional[int] = None) -> str:
    """Sample key for ``generator``. Returns a directory name.

    generator: "sdformer" (needs ``delta``) or "diffusionts" (needs ``sampling_timesteps``).
    The two families key on different things, so the wrong argument raises rather than
    silently producing a name that collides with another arm's.
    """
    if generator == "sdformer":
        if delta is None:
            raise ValueError("sdformer runs are keyed on the watermark delta; pass delta=")
        return _derive_lvqvae_run_name(exp, delta, gen_seed, watermark_seed)
    if generator == "diffusionts":
        if sampling_timesteps is None:
            raise ValueError(
                "diffusionts runs are keyed on sampling_timesteps; pass sampling_timesteps=")
        return derive_diffusionts_run_name(exp, sampling_timesteps, gen_seed, watermark_seed)
    raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATORS}")


# ── Loud checkpoint resolution ───────────────────────────────────────────────────────────

def resolve_checkpoint(path: str, what: str, how_to_make_it: str) -> str:
    """Return ``path``, or raise ``FileNotFoundError`` naming the command that creates it.

    path: str. what: human name of the checkpoint ("the SDformer AR transformer").
    how_to_make_it: the exact command to run.

    There is no fallback: state-dict keys match across training stages, so loading a
    neighbouring checkpoint would succeed and produce plausible-looking output that nothing
    downstream catches. Every baseline checkpoint lookup goes through here.
    """
    import os

    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"{what} not found: {path}\n"
        f"  There is no fallback: loading a different checkpoint here would succeed and "
        f"produce plausible-looking garbage.\n"
        f"  Fix: {how_to_make_it}"
    )


# ── Statistics shared with the root evaluate.py ──────────────────────────────────────────

def split_half_null(wo_data: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a no-watermark detection-statistic pool into two disjoint halves.

    wo_data: (N, 1) float — one value per generated window. seed: int.
    Returns (half_a, half_b), shapes (N//2, 1) and (N - N//2, 1).

    Byte-for-byte the rule ``evaluate.py`` uses, so the L-VQVAE and the baseline tables
    report the same kind of clean-row null: scoring the pool against *itself* makes z = 0 and
    FPR = 0 true by construction, which measures nothing. ``tests/test_baselines.py``
    asserts this function and ``evaluate.py``'s agree on the same input — the duplication
    exists only because the baselines may not import a root entry point, and the test is
    what keeps the two from drifting.

    The local ``RandomState`` is load-bearing: the shuffle must not perturb the global numpy
    stream ``get_zscore``'s bootstrap draws from, or the clean row would depend on how many
    rows were computed before it.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(wo_data))
    cut = len(wo_data) // 2
    return wo_data[order[:cut]], wo_data[order[cut:]]


# ── Tree-Ring key material ───────────────────────────────────────────────────────────────

def build_tr_key_patches(args, device, n: int, patch_shape: Sequence[int], key: int):
    """Draw ``n`` Tree-Ring ``gt_patch`` tensors under green-list key ``key``.

    args: the namespace ``Utils.watermark_utils.get_watermarking_pattern`` reads
    (``w_pattern``, ``w_radius``, ``w_pattern_const``). device: torch device. n: number of
    windows. patch_shape: the per-window pattern shape, ``(1, 1, T, D)``. key: int.
    Returns a (n, 1, T, D) complex tensor on ``device``.

    This is the wrong-key control for TR and nothing else. It reproduces the draw order of
    ``Diffusion_TS.generate_mts``'s TR branch — one ``get_watermarking_pattern`` call per
    window under a single ``manual_seed`` — but with ``key`` instead of the generation key,
    and it restores the global torch RNG state afterwards exactly as that branch does, so
    calling it consumes nothing. The circular mask is key-independent (it is geometry), so
    only the pattern changes; that is precisely the "same detector, different key" control.

    Under a wrong key the i-th patch has no relationship to the i-th window anyway, so it
    does not matter that NaN-dropping during generation may have shifted the correspondence.
    """
    import torch

    from Utils.watermark_utils import get_watermarking_pattern

    st0 = torch.get_rng_state()
    torch.manual_seed(int(key))
    patches = [get_watermarking_pattern(args, device, shape=tuple(patch_shape))
               for _ in range(n)]
    torch.set_rng_state(st0)
    return torch.cat(patches, dim=0)


__all__ = [
    "GENERATORS", "TOKEN_WM_METHODS", "LATENT_WM_METHODS", "WM_METHODS_BY_GENERATOR",
    "ZSCORE_METHOD_KEY", "LATENT_SCORER_CSV_STEM", "KEYED_LATENT_WM_METHODS", "GT_PREFIX",
    "DEFAULT_LATENT_WATERMARK_SEED", "methods_for", "zscore_method_key",
    "derive_sdformer_exp_name", "derive_diffusionts_exp_name", "derive_exp_name",
    "derive_diffusionts_run_name", "derive_run_name", "resolve_checkpoint",
    "split_half_null", "build_tr_key_patches", "format_delta",
]

"""The LVQMark watermark protocol: green list, method presets, attack grid, scoring.

What this module is for
-----------------------
Everything here is generator-independent. ``cli/generate.py`` (L-VQVAE) and the baseline
entry points score *the same* watermark against *the same* attacks on different backbones,
so the definition of the green list, of the method, and of a window's green fraction lives
here in exactly one place and every entry point imports it.

Layer: ``Utils/`` — a leaf. It imports numpy and torch and nothing from ``Models/``,
``Data/`` or any entry point. The attack *implementations* stay in
``Utils.watermark_utils`` (``add_attack``); this module only names the grid they are called
with, and ``tests/test_greenlist.py`` checks every name in the grid is one ``add_attack``
implements.

Naming
------
The two switches carry the preprint's names (arXiv:2608.19727 p.6): the *alternating
partition* flips the green set on odd token positions (Eq. 9), and the *unbiased warm-up*
leaves the first m token positions without a logit bias.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


# ── Watermark table + method registry ──
#
# There is ONE green list: a single 50/50 partition of the codebook drawn from
# --watermark_seed (default 12345). It is the same list at every token position
# and for every context; the (K+1, K) "table" below is that one list broadcast
# over K+1 identical rows, which lets the sampler index it by previous token
# without a special case. This is the intended design, not a
# degenerate case of a context-dependent scheme.
#
# LVQMark = (alternating_partition=True, unbiased_warmup=True) is the paper's method and
# the only preset. The two switches are exposed on the command line, so any other
# combination is reachable without a second named method.


def create_fixed_watermark_table(codebook_size, seed=12345):
    """One fixed green list, broadcast to a (K+1, K) bool table.

    codebook_size: K. seed: int, the green-list seed. Returns (K+1, K) bool tensor whose
    rows are all THE SAME green list — exactly half of the K codes are True. The K+1 rows
    exist so the sampler and the detector can index by previous token (0..K-1 plus the BOS
    id K) uniformly; the row index carries no information.
    """
    K = codebook_size
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(K, generator=rng)
    mask = torch.zeros(K, dtype=torch.bool)
    mask[perm[:K // 2]] = True
    return mask.unsqueeze(0).expand(K + 1, -1).clone()


WM_METHODS = {
    'LVQMark': {'alternating_partition': True,  'unbiased_warmup': True},
}


def resolve_wm_switches(method, alternating_partition=None, unbiased_warmup=None):
    """Resolve the two LVQMark switches. Returns (alternating_partition, unbiased_warmup).

    method: a key of WM_METHODS — its preset supplies any switch left at None.
    alternating_partition / unbiased_warmup: None (take the preset) or a bool from the CLI,
    which wins. An override produces an off-menu run that still lands in the preset's output
    directory, so ``cli/generate.py`` prints a [WARN] for it.
    """
    preset = WM_METHODS[method]
    return (
        preset['alternating_partition'] if alternating_partition is None
        else bool(alternating_partition),
        preset['unbiased_warmup'] if unbiased_warmup is None else bool(unbiased_warmup),
    )


def resolve_warmup_positions(token_length, warmup_positions=None):
    """m, the number of leading token positions that receive no logit bias. Returns int.

    token_length: int L, the number of tokens per window. warmup_positions: None for the
    paper's default m = min(4, L // 3), or an explicit int in [0, L].

    Making m a number rather than an implicit formula is the point of --warmup_positions:
    the LVQMark arm skips 4 of 16 positions on every dataset in this repo, and that ratio is
    what any dilution argument about the method has to reason about.
    """
    m = min(4, token_length // 3) if warmup_positions is None else int(warmup_positions)
    if not 0 <= m <= token_length:
        raise ValueError(
            f"--warmup_positions must be in [0, token_length={token_length}], got {m}")
    return m


def warmup_delta_schedule(delta, token_length, warmup_positions):
    """Per-position logit bias under the unbiased warm-up. Returns a list of L floats.

    delta: float, the watermark bias. token_length: int L. warmup_positions: int m.
    Positions 0..m-1 get 0.0, positions m..L-1 get delta — the schedule the sampler indexes
    by token position (Models/transformer_ar.AutoregressiveTransformer.generate).
    """
    return [0.0] * warmup_positions + [float(delta)] * (token_length - warmup_positions)


def build_watermark_table(method, codebook_size, seed=12345,
                          alternating_partition=None, unbiased_warmup=None):
    """Green-list table plus the resolved switches for ``method``.

    method: a key of WM_METHODS. codebook_size: K. seed: the green-list seed.
    alternating_partition / unbiased_warmup: None for the method's preset, or a bool that
    overrides it. Returns ``(table, alternating_partition, unbiased_warmup)`` — the (K+1, K)
    bool table from :func:`create_fixed_watermark_table` and the two resolved switches.

    One call for entry points that want both; ``generate.py`` resolves the switches earlier
    than it builds the table (it has to record them in ``run_config.json`` before either
    branch runs) and so calls the two pieces separately.
    """
    table = create_fixed_watermark_table(codebook_size, seed)
    return (table,) + resolve_wm_switches(method, alternating_partition, unbiased_warmup)


def resolve_detect_key(detect_key, watermark_seed):
    """Which key detection scores with, and the directory suffix that records it.

    detect_key: None (score with the key the samples were generated under) or an int.
    watermark_seed: the generation key. Returns (key: int, dir_suffix: str).

    The suffix is empty exactly when --detect_key was not passed, so the default run writes
    the same detect/ , detect-wo/ and <attack>/ directories it always has. A wrong-key run
    writes alongside them under a -key<k> suffix instead of overwriting them: the run
    directory stays keyed on the *generation* key (the samples did not change), and the
    condition directory records the *detection* key.
    """
    if detect_key is None:
        return watermark_seed, ""
    return int(detect_key), f"-key{int(detect_key)}"


def detect_dir_name(condition: str, detect_suffix: str = "") -> str:
    """Directory holding one detect condition's CSV, under ``<run>/<window>/<wm_method>/``.

    condition: "detect" | "detect-wo" | "<attack>-<factor>" | "<attack>-<factor>-wo".
    detect_suffix: "" or "-key{k}" from :func:`resolve_detect_key`. Returns a directory name.

    ``generate.py`` writes these directories and ``evaluate.py`` reads them through this one
    function, so both spell the name the same way. The CSV inside is always
    ``<wm_method>.csv``, one green fraction per row, in window order.
    """
    return f"{condition}{detect_suffix}"


# ── The conditions the watermark is scored under ──
#
# One definition, imported by every entry point, so a condition can never be spelled
# differently in the code that produces a CSV and the code that reads it. ``(None, None)``
# is the clean condition and must stay first: ``generate.py`` relies on it to write the
# unattacked ``detect/`` pair before any attacked one.
#
# ⚠ The names must be the ones ``Utils.watermark_utils.add_attack`` implements. Here
# ``crop`` is the midpoint-fill crop — pinned by tests/test_greenlist.py.
ATTACK_GRID: List[Tuple[Optional[str], Optional[float]]] = [
    (None, None),
    ("offset", 0.05), ("offset", 0.3),
    ("crop", 0.05), ("crop", 0.3),
    ("insert", 0.05), ("insert", 0.3),
]

# The attacked conditions as directory / row labels, in grid order: "offset-0.05", ...
ATTACK_LABELS: List[str] = [
    f"{name}-{factor}" for name, factor in ATTACK_GRID if name is not None
]

# Stage 4 (robust encoder) trains on the same grid with ``crop_var`` in place of ``crop``:
# a variable-length crop, so the encoder does not overfit one crop length. It is a separate
# constant because the detector is *scored* on ATTACK_GRID and *trained* on this one.
ROBUST_TRAIN_ATTACK_GRID: List[Tuple[Optional[str], Optional[float]]] = [
    (None, None),
    ("offset", 0.05), ("offset", 0.3),
    ("crop_var", 0.05), ("crop_var", 0.3),
    ("insert", 0.05), ("insert", 0.3),
]


# ── Valid token range (skip constant-padded regions) ──

def compute_valid_token_range(data, rf=4, stride=2):
    N, T, D = data.shape
    n_tokens = (T - rf) // stride + 1
    diff = np.diff(data, axis=1)
    is_const = (np.abs(diff) < 1e-8).all(axis=2)

    start_tok = np.zeros(N, dtype=np.int64)
    end_tok = np.full(N, n_tokens, dtype=np.int64)

    for i in range(N):
        front_crop = 0
        for t in range(T - 1):
            if is_const[i, t]:
                front_crop = t + 2
            else:
                break
        back_crop = 0
        for t in range(T - 2, -1, -1):
            if is_const[i, t]:
                back_crop = T - t
            else:
                break

        valid_start = front_crop
        if valid_start % stride != 0:
            valid_start += 1
        valid_end = T - back_crop

        st = 0
        while st < n_tokens and st * stride < valid_start:
            st += 1
        et = n_tokens
        while et > 0 and (et - 1) * stride + rf > valid_end:
            et -= 1

        start_tok[i] = st
        end_tok[i] = et

    return start_tok, end_tok


def green_fractions(indices, mask, start_tok, end_tok, alternating_partition):
    """Per-window fraction of tokens that fall in the green list. Returns a list of B floats.

    indices: (B, L) long tensor of token ids recovered from the (possibly attacked) series.
    mask: (K+1, K) bool table from :func:`create_fixed_watermark_table`, indexed by the
    previous token (row K is the BOS id). start_tok / end_tok: (B,) int arrays from
    :func:`compute_valid_token_range` — the half-open range of token positions that is not
    constant padding. alternating_partition: bool, flip the partition on odd positions.

    A window with an empty valid range scores 0.5, the value an unwatermarked window has in
    expectation, so that it neither helps nor hurts the detector. The values it returns are
    written one per CSV row, in order.
    """
    B, L = indices.shape
    K_cb = mask.shape[1]
    bos_id = K_cb
    prev_tokens = torch.cat([
        torch.full((B, 1), bos_id, dtype=torch.long), indices[:, :-1]], dim=1)
    fractions = []
    for b in range(B):
        st, et = start_tok[b], end_tok[b]
        if et <= st:
            fractions.append(0.5)
            continue
        green = 0
        for t in range(st, et):
            m = mask[prev_tokens[b, t].item()]
            if alternating_partition and t % 2 == 1: m = ~m
            if m[indices[b, t]]: green += 1
        fractions.append(green / (et - st))
    return fractions

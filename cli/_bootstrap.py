"""Repo-root bootstrap, imported first by every ``cli/`` entry point.

The pipeline is written in the repository root's frame of reference: ``Models`` /
``Utils`` / ``Data`` are imported as top-level packages, every artifact root in
``Utils/path_utils.py`` (``checkpoints`` / ``outputs`` / ``results``) is a relative path,
and each config's ``data_root`` is ``./Data/datasets/...``. This module establishes both
halves of that frame before any project import runs:

1. the repository root goes to ``sys.path[0]``, so ``Models`` / ``Utils`` / ``Data`` resolve
   however the script was invoked (``python cli/train.py``, an absolute path, a symlink);
2. the process changes directory to the repository root, so every relative path in the
   pipeline keeps meaning "relative to the repo", not "relative to wherever you stood".

The working directory a command is issued from therefore has no effect on where a run reads
or writes.

Import it for its side effects, before the first project import::

    import _bootstrap  # noqa: F401  (repo root on sys.path + as cwd)

It is silent, idempotent, and imports nothing but the standard library.

Layer: ``cli/`` — the entry-point layer. Imports nothing from the project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Absolute path of the repository root (the parent of ``cli/``).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

_ROOT = str(REPO_ROOT)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

os.chdir(_ROOT)

__all__ = ["REPO_ROOT"]

"""Device selection shared by the entry points.

An entry point picks one CUDA device with ``--gpu N``; everything downstream follows that
choice, so every number a run reports comes from one card. Two device lists are involved and
they are indexed the same way:

* **torch** — ``select_torch_device`` makes ``cuda:N`` the process-wide current device, and
  anything that does not name a device explicitly (``.cuda()``, ``torch.device("cuda")``)
  lands there.
* **TensorFlow** — used only by the discriminative and predictive metrics. It keeps its own
  device list, ignores ``torch.cuda.set_device``, and by default maps *every* visible GPU and
  places its ops on ``/GPU:0``. ``restrict_tensorflow_to_gpu`` narrows that list to the
  selected card.

Both lists come from the CUDA runtime, so ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER``
apply to both and ``N`` means the same device in each. Set ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` on
a multi-GPU host to make ``N`` the index ``nvidia-smi`` reports.
"""

import os
from typing import Optional, Union

import torch

DeviceLike = Union[int, str, torch.device, None]


def _visible_hint() -> str:
    """``CUDA_VISIBLE_DEVICES`` as a string, for out-of-range error messages."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Normalise a device argument. ``None`` means "the process's current device".

    device: a CUDA ordinal, a string such as ``"cuda:1"`` / ``"cpu"``, a ``torch.device``,
    or None. Returns a ``torch.device``; with ``device=None`` that is
    ``cuda:<current>`` when CUDA is available and ``cpu`` otherwise, so a caller that never
    saw a ``--gpu`` flag still follows the device the entry point selected.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def select_torch_device(gpu: Optional[int]) -> torch.device:
    """Make ``cuda:gpu`` the current torch device and return the device that is now current.

    gpu: CUDA ordinal, or None to leave the current device alone. Returns ``cpu`` when no
    CUDA device is visible, so the same code path runs on a CPU-only machine.

    Raises ValueError if ``gpu`` is not a visible device — loudly, because the alternative is
    a run that silently computes on a different card than it was asked to.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu is not None:
        count = torch.cuda.device_count()
        if not 0 <= gpu < count:
            raise ValueError(
                f"--gpu {gpu} is not a visible CUDA device: {count} visible "
                f"(CUDA_VISIBLE_DEVICES={_visible_hint()})")
        torch.cuda.set_device(gpu)
    return torch.device("cuda", torch.cuda.current_device())


def cuda_index(device: DeviceLike) -> Optional[int]:
    """The CUDA ordinal of ``device``, or None when it is not a CUDA device.

    device: anything :func:`resolve_device` accepts. Used to hand a torch device selection
    over to a library that only understands an index.
    """
    if device is None:
        return None
    resolved = resolve_device(device)
    return resolved.index if resolved.type == "cuda" else None


def restrict_tensorflow_to_gpu(gpu: Optional[int]) -> None:
    """Limit an already-imported TensorFlow to a single GPU.

    gpu: CUDA ordinal, or None to leave TensorFlow's own selection untouched. Returns None.

    Call this directly after ``import tensorflow`` and before any TensorFlow op runs:
    ``set_visible_devices`` is only accepted while TensorFlow's devices are uninitialised. A
    no-op when TensorFlow sees no GPU, or when the visible list already is exactly the
    requested card. Raises ValueError if ``gpu`` is out of range for TensorFlow's list.
    """
    if gpu is None:
        return
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    if not 0 <= gpu < len(gpus):
        raise ValueError(
            f"--gpu {gpu} is not a GPU TensorFlow can see: {len(gpus)} visible "
            f"(CUDA_VISIBLE_DEVICES={_visible_hint()})")
    target = [gpus[gpu]]
    if tf.config.get_visible_devices("GPU") == target:
        return
    try:
        tf.config.set_visible_devices(target, "GPU")
    except RuntimeError:
        print("  [WARN] TensorFlow had already initialised its devices; the discriminative "
              "and predictive metrics keep TensorFlow's own device choice.", flush=True)

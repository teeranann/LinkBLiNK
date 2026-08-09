"""Torch device selection.

Shared by the detector and the embedder so both always agree on where tensors
live — a mismatch surfaces as an opaque "expected all tensors on the same
device" error deep inside a forward pass.
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return the best available device.

    ``torch.cuda.is_available()`` alone is not sufficient: with
    ``CUDA_VISIBLE_DEVICES=""`` it can report True while ``device_count()`` is
    0, after which loading a GPU-saved checkpoint fails. Requiring at least one
    visible device makes CPU-only and GPU-masked machines work.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")

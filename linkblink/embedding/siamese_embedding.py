"""Appearance embeddings for particle re-identification.

Each detection gets a 128-d vector describing how it looks. The judge compares
vectors across a blinking gap: same particle, similar appearance.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import SiameseConfig
from ..models import SiameseNet
from ..utils.device import resolve_device
from ..utils.logging import get_logger

log = get_logger(__name__)

UINT16_MAX = 65535.0


def extract_patch(frame: np.ndarray, centre_x: float, centre_y: float, patch_size: int) -> np.ndarray:
    """Crop a square patch centred on a particle, zero-padded at frame edges.

    Padding rather than clipping keeps every patch exactly ``patch_size`` square,
    which the network's fixed fully-connected layer requires.
    """
    half = patch_size // 2
    height, width = frame.shape

    x1 = int(centre_x - half)
    y1 = int(centre_y - half)
    x2 = int(centre_x + half)
    y2 = int(centre_y + half)

    patch = np.zeros((patch_size, patch_size), dtype=frame.dtype)

    # Source window, clipped to the frame.
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(width, x2), min(height, y2)

    # Matching destination window inside the padded patch.
    dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        patch[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return patch


class SiameseEmbedder:
    """Holds the Siamese network and turns detections into embedding vectors."""

    def __init__(self, config: SiameseConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or resolve_device()
        self.model: SiameseNet | None = None

    def load(self) -> None:
        """Load weights onto the target device. Safe to call more than once."""
        if self.model is not None:
            return

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Siamese model not found at {self.config.model_path}")

        model = SiameseNet(patch_size=self.config.patch_size).to(self.device)
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.eval()

        self.model = model
        log.info("Siamese network loaded successfully.")

    @torch.no_grad()
    def embed(self, frame: np.ndarray, centre_x: float, centre_y: float) -> np.ndarray:
        """Return the embedding for the particle centred at ``(centre_x, centre_y)``."""
        assert self.model is not None, "call load() first"

        patch = extract_patch(frame, centre_x, centre_y, self.config.patch_size)
        tensor = torch.from_numpy(patch.astype(np.float32) / UINT16_MAX)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return self.model.forward_once(tensor).squeeze().cpu().numpy()

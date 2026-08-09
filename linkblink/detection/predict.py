"""Stage 2 — particle detection with the U-Net.

Wrapped in a class so the weights are loaded once and reused across every video
in a batch run; the old function reloaded the checkpoint per video.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from ..config import UNetConfig
from ..io.images import list_frames, load_normalised_frame, mask_name_for
from ..models import UNet
from ..utils.device import resolve_device
from ..utils.logging import get_logger

log = get_logger(__name__)


class UNetDetector:
    """Runs U-Net inference over a folder of frames, writing one mask per frame."""

    def __init__(self, config: UNetConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or resolve_device()
        self.net: UNet | None = None

        # Normalisation must match training; the checkpoint is meaningless otherwise.
        self.transform = A.Compose(
            [A.Normalize(mean=(config.norm_mean,), std=(config.norm_std,)), ToTensorV2()]
        )

    def load(self) -> None:
        """Load weights onto the target device. Safe to call more than once."""
        if self.net is not None:
            return

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"U-Net model not found at {self.config.model_path}")

        log.info("U-Net will run on: %s", self.device)

        net = UNet(n_channels=1, n_classes=1, bilinear=False).to(self.device)
        state_dict = torch.load(self.config.model_path, map_location=self.device)
        # Older checkpoints carry this bookkeeping entry, which is not a parameter.
        state_dict.pop("mask_values", None)
        net.load_state_dict(state_dict)
        net.eval()

        self.net = net
        log.info("U-Net model loaded from %s.", self.config.model_path)

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a 0/255 uint8 mask for one normalised frame."""
        assert self.net is not None, "call load() first"
        original_h, original_w = frame.shape

        tensor = self.transform(image=frame)["image"].unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)

        logits = self.net(tensor)
        # The network's output can be off by a pixel or two after pooling;
        # resize back so mask coordinates match the source frame exactly.
        logits = F.interpolate(
            logits, size=(original_h, original_w), mode="bilinear", align_corners=False
        )

        heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()
        if heatmap.ndim == 1:
            heatmap = heatmap.reshape(original_h, original_w)
        elif heatmap.ndim == 0:
            heatmap = np.array([[heatmap]])

        return (heatmap > self.config.threshold).astype(np.uint8) * 255

    def predict_directory(self, frames_dir: Path, mask_dir: Path) -> bool:
        """Predict masks for every frame in ``frames_dir``.

        Returns True on success, including the no-frames case (nothing to do is
        not an error for a batch run).
        """
        log.info("\n--- Stage 2: Running U-Net prediction on '%s' ---", frames_dir)
        self.load()

        frame_paths = list_frames(frames_dir)
        if not frame_paths:
            log.warning("WARNING: No images found in %s for U-Net prediction. Skipping.", frames_dir)
            return True

        mask_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frame_paths, desc="U-Net Predicting Frames"):
            frame = load_normalised_frame(frame_path)

            if self.config.save_debug_images:
                self._save_debug_original(frame, frame_path, mask_dir)

            mask = self.predict_frame(frame)
            Image.fromarray(mask).save(mask_dir / mask_name_for(frame_path))

        log.info("U-Net prediction completed successfully.")
        return True

    def _save_debug_original(self, frame: np.ndarray, frame_path: Path, mask_dir: Path) -> None:
        out = mask_dir / f"{frame_path.stem}_debug_01_original.png"
        Image.fromarray((frame * 255).astype(np.uint8)).save(out)

        tensor = self.transform(image=frame)["image"]
        normalised = tensor.squeeze().cpu().numpy()
        spread = normalised.max() - normalised.min() + 1e-8
        scaled = (normalised - normalised.min()) / spread
        debug_path = mask_dir / f"{frame_path.stem}_debug_03_normalized_input.png"
        Image.fromarray((scaled * 255).astype(np.uint8)).save(debug_path)

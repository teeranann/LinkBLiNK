"""Frame and mask loading.

Two different loaders on purpose:

* :func:`load_normalised_frame` scales to 0-1 for the U-Net, which was trained
  on normalised input.
* :func:`load_frame_and_mask` keeps the raw 16-bit counts, because photometry
  turns those counts into photon numbers and any rescaling would corrupt them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

UINT16_MAX = 65535.0
UINT8_MAX = 255.0


def list_frames(directory: Path) -> list[Path]:
    """Return the image files in ``directory``, sorted by name (i.e. frame order)."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_normalised_frame(path: Path) -> np.ndarray:
    """Load a frame as float32 in 0-1, preserving the source bit depth.

    16-bit TIFFs divide by 65535, 8-bit images by 255, so the same physical
    brightness maps to the same value regardless of how the frame was stored.
    """
    img = Image.open(path)

    if img.mode in ("I;16", "I"):
        return np.array(img, dtype=np.float32) / UINT16_MAX
    return np.array(img.convert("L"), dtype=np.float32) / UINT8_MAX


def load_frame_and_mask(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw (un-normalised) frame and its binarised U-Net mask.

    Raises:
        FileNotFoundError: if either file is missing or unreadable.
    """
    original = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(f"Original image not found at {image_path}")
    if mask is None:
        raise FileNotFoundError(f"U-Net mask not found at {mask_path}")

    mask = (mask > 0).astype(np.uint8) * 255
    return original, mask


def parse_frame_number(path: Path, default: int = 0) -> int:
    """Extract the frame index from a filename like ``Real_1_frame_0042.tif``.

    Takes the digits from the last underscore-separated token. Returns
    ``default`` when the name carries no trailing number.
    """
    last_token = path.stem.split("_")[-1]
    digits = "".join(filter(str.isdigit, last_token))
    if not digits:
        return default
    return int(digits)


def mask_name_for(frame_path: Path) -> str:
    """Filename convention linking a frame to its U-Net mask."""
    return f"{frame_path.stem}_predict_mask.png"


def read_frame_shape(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of a frame, or ``(100, 100)`` if unreadable."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return 100, 100
    return image.shape[0], image.shape[1]

"""Background estimation and conversion of camera counts into photon numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CameraConfig

BACKGROUND_BORDER_PX = 1
"""How far outside the bounding box to sample local background."""


@dataclass
class PhotonCounts:
    """One particle's brightness at four points along the detection chain."""

    background_subtracted_counts: float
    """``Ibcnt`` — raw ADU above local background."""

    photons_emitted: float
    """``Ib`` — photons leaving the emitter, after undoing collection efficiency."""

    photons_at_camera: float
    """``Idetect`` — photons reaching the sensor."""

    photoelectrons: float
    """``Iphelect`` — electrons actually generated."""


def local_background(
    frame: np.ndarray,
    particle_mask_roi: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    fallback: float,
) -> float:
    """Median of the non-particle pixels in a one-pixel ring around the bbox.

    Local rather than global because SMLM background varies across the field.
    Falls back to the frame-wide median when the ring is empty (particle
    touching the frame edge).
    """
    frame_h, frame_w = frame.shape

    x_end = min(x + width, frame_w)
    y_end = min(y + height, frame_h)

    x_start_padded = max(0, x - BACKGROUND_BORDER_PX)
    y_start_padded = max(0, y - BACKGROUND_BORDER_PX)
    x_end_padded = min(x + width + BACKGROUND_BORDER_PX, frame_w)
    y_end_padded = min(y + height + BACKGROUND_BORDER_PX, frame_h)

    padded_roi = frame[y_start_padded:y_end_padded, x_start_padded:x_end_padded]

    padded_mask = np.zeros_like(padded_roi, dtype=np.uint8)
    padded_mask[
        y - y_start_padded : y_end - y_start_padded,
        x - x_start_padded : x_end - x_start_padded,
    ] = particle_mask_roi

    background_pixels = padded_roi[(padded_mask == 0)]
    if background_pixels.size == 0:
        return float(fallback)
    return float(np.median(background_pixels))


def photon_counts(
    roi: np.ndarray,
    particle_mask_roi: np.ndarray,
    area: int,
    background: float,
    camera: CameraConfig,
) -> PhotonCounts:
    """Integrate particle intensity and walk it back up the detection chain.

    Counts are clamped at zero: a negative background-subtracted sum means the
    background estimate overshot, not that the particle emitted negative light.
    """
    total_intensity = float(np.sum(roi[particle_mask_roi > 0]))
    counts = max(0.0, total_intensity - (background * area))

    photoelectrons = counts * camera.gain_electrons_per_count
    photons_at_camera = photoelectrons / camera.quantum_efficiency
    photons_emitted = photons_at_camera / camera.collection_efficiency

    return PhotonCounts(
        background_subtracted_counts=counts,
        photons_emitted=photons_emitted,
        photons_at_camera=photons_at_camera,
        photoelectrons=photoelectrons,
    )


def frame_background(frame: np.ndarray) -> float:
    """Frame-wide median, used as the fallback for edge particles."""
    return float(np.median(frame.flatten()))

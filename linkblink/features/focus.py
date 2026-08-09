"""Focus measure used to separate in-focus particles from defocused blur."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(image_roi: np.ndarray, mask_roi: np.ndarray) -> float:
    """Variance of the Laplacian over the masked pixels of a bounding-box crop.

    A sharp, in-focus particle has strong second derivatives and therefore high
    variance; a defocused blob is smooth and scores near zero.

    16-bit crops are rescaled to 0-255 first so the threshold means the same
    thing regardless of camera bit depth.
    """
    if image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
        return 0.0

    pixels = image_roi.astype(np.float32)

    # Heuristic for 16-bit data; 8-bit crops are already in range.
    if 255 < pixels.max() <= 65535:
        cv2.normalize(pixels, pixels, 0, 255, cv2.NORM_MINMAX)

    # Mask after normalising, so background zeros do not skew the rescale.
    as_uint8 = pixels.astype(np.uint8)
    masked = cv2.bitwise_and(as_uint8, as_uint8, mask=(mask_roi > 0).astype(np.uint8))

    laplacian = cv2.Laplacian(masked, cv2.CV_32F)
    particle_response = laplacian[mask_roi > 0]

    if particle_response.size == 0:
        return 0.0
    return float(np.var(particle_response))

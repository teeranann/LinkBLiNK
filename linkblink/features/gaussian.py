"""Rotated 2-D Gaussian PSF fitting.

The fitted widths give FWHM (a size feature the judge uses) and the residual
sum gives a goodness-of-fit the defocusing filter thresholds on: a real
diffraction-limited particle fits a Gaussian well, overlapping or defocused
blobs do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import curve_fit

SIGMA_TO_FWHM = 2.355
"""FWHM = 2*sqrt(2*ln2)*sigma."""


@dataclass
class GaussianFit:
    """Outcome of one fit. Failure leaves the permissive defaults in place."""

    success: bool = False
    amplitude: float = np.nan
    sigma_x: float = np.nan
    sigma_y: float = np.nan
    fwhm_x: float = np.nan
    fwhm_y: float = np.nan
    fwhm_avg: float = np.nan
    offset: float = np.nan
    r_squared: float = np.nan
    rmse: float = np.nan
    residual_sum: float = float("inf")
    sigma_aspect_ratio: float = float("inf")


def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Rotated 2-D Gaussian, flattened for :func:`scipy.optimize.curve_fit`."""
    x, y = coords
    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)

    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def prepare_fit_pixels(
    roi: np.ndarray, mask_roi: np.ndarray, frame_is_16bit: bool
) -> np.ndarray:
    """Rescale a crop to 0-255 and zero everything outside the particle mask.

    The fit bounds below are written in 8-bit units, so the input has to be in
    that range for the initial guesses and bounds to be meaningful.
    """
    pixels = roi.astype(np.float32)
    if pixels.max() > 0 and frame_is_16bit:
        cv2.normalize(pixels, pixels, 0, 255, cv2.NORM_MINMAX)

    as_uint8 = pixels.astype(np.uint8)
    return cv2.bitwise_and(as_uint8, as_uint8, mask=mask_roi)


def fit_gaussian_2d(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    intensities: np.ndarray,
    bbox_w: int,
    bbox_h: int,
    max_iterations: int = 5000,
) -> GaussianFit:
    """Fit a rotated Gaussian to the masked pixels of one particle.

    Args:
        x_coords, y_coords: pixel positions within the bounding-box crop.
        intensities: pixel values at those positions, in 0-255.
        bbox_w, bbox_h: crop dimensions, used for the initial width guess and
            to bound the centre inside the crop.

    Returns:
        A :class:`GaussianFit`; check ``success`` before trusting the values.
    """
    if intensities.size == 0:
        return GaussianFit()

    initial_amplitude = np.max(intensities) - np.min(intensities)
    initial_offset = np.min(intensities)
    initial_xo = x_coords.mean()
    initial_yo = y_coords.mean()
    # Half the bbox is a rough FWHM; convert to sigma and keep it non-degenerate.
    initial_sigma_x = max((bbox_w / 2) / SIGMA_TO_FWHM if bbox_w > 0 else 1.0, 0.5)
    initial_sigma_y = max((bbox_h / 2) / SIGMA_TO_FWHM if bbox_h > 0 else 1.0, 0.5)

    p0 = [
        initial_amplitude,
        initial_xo,
        initial_yo,
        initial_sigma_x,
        initial_sigma_y,
        0.0,
        initial_offset,
    ]
    bounds = (
        [0, 0, 0, 0.1, 0.1, -np.pi, 0],
        [256, bbox_w, bbox_h, max(bbox_w, bbox_h), max(bbox_w, bbox_h), np.pi, 255],
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x_coords, y_coords),
            intensities,
            p0=p0,
            bounds=bounds,
            maxfev=max_iterations,
        )
    except Exception:
        # Non-convergence, bad bounds, and singular Jacobians are all expected
        # for junk components; a failed fit is a filter signal, not an error.
        return GaussianFit()

    amplitude, _xo, _yo, sigma_x, sigma_y, _theta, offset = popt

    fitted = gaussian_2d((x_coords, y_coords), *popt)
    residual_sum = float(np.sum((intensities - fitted) ** 2))

    ss_total = np.sum((intensities - np.mean(intensities)) ** 2)
    r_squared = 1 - (residual_sum / (ss_total + 1e-9)) if ss_total > 0 else 0.0
    rmse = float(np.sqrt(residual_sum / intensities.size))

    fwhm_x = sigma_x * SIGMA_TO_FWHM
    fwhm_y = sigma_y * SIGMA_TO_FWHM

    if sigma_x != 0 and sigma_y != 0:
        sigma_aspect_ratio = max(sigma_x, sigma_y) / min(sigma_x, sigma_y)
    else:
        sigma_aspect_ratio = float("inf")

    return GaussianFit(
        success=True,
        amplitude=amplitude,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        fwhm_avg=(fwhm_x + fwhm_y) / 2,
        offset=offset,
        r_squared=r_squared,
        rmse=rmse,
        residual_sum=residual_sum,
        sigma_aspect_ratio=sigma_aspect_ratio,
    )

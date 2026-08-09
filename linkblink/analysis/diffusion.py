"""Diffusion models fitted to MSD curves.

MSD(τ) = 2·n·D·τ + (V·τ)² + Z, where *n* is the number of dimensions:

* ``D`` — diffusion coefficient (µm²/s), the linear term.
* ``V`` — drift velocity (µm/s); directed motion makes the curve superlinear.
* ``Z`` — intercept, i.e. twice the squared localisation error. A real
  measurement has a non-zero intercept even at zero lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class DiffusionFit:
    """Fitted parameters, in µm²/s, µm/s and µm²."""

    success: bool
    D: float = np.nan
    V: float = np.nan
    Z: float = np.nan
    error: str | None = None


def diffusion_model_1d(lag_time, D, V, Z):
    """MSD along a single axis: 2·D·τ + (V·τ)² + Z."""
    return 2 * D * lag_time + (V * lag_time) ** 2 + Z


def diffusion_model_2d(lag_time, D, V, Z):
    """MSD in the plane: 4·D·τ + (V·τ)² + Z."""
    return 4 * D * lag_time + (V * lag_time) ** 2 + Z


def fit_msd_curve(
    lag_times_s: np.ndarray, msd_um2: np.ndarray, two_dimensional: bool
) -> DiffusionFit:
    """Fit a diffusion model to one MSD curve.

    Args:
        lag_times_s: lag times in seconds.
        msd_um2: MSD values in µm² (not nm² — the bounds below assume µm²).
        two_dimensional: True for the combined xy curve, False for x or y alone.

    Returns:
        A :class:`DiffusionFit`; ``success`` is False if the fit did not converge.
    """
    if len(lag_times_s) <= 1:
        return DiffusionFit(success=False, error="not enough points")

    model = diffusion_model_2d if two_dimensional else diffusion_model_1d
    dimension_factor = 4 if two_dimensional else 2

    # Seed D from the average slope; the epsilon guards a zero-length lag span.
    span = lag_times_s[-1] - lag_times_s[0]
    initial_D = (msd_um2[-1] - msd_um2[0]) / (dimension_factor * span + 1e-9)
    initial_D = max(1e-12, initial_D)
    initial_Z = msd_um2[0] if msd_um2[0] > -1e-12 else 0
    initial_V = 1e-8

    try:
        popt, _ = curve_fit(
            model,
            lag_times_s,
            msd_um2,
            p0=[initial_D, initial_V, initial_Z],
            # D and V cannot be negative; Z is bounded near zero because a
            # localisation-error intercept is small by construction.
            bounds=([0, 0, -0.01], [10.0, 10.0, 0.01]),
            maxfev=5000,
        )
    except Exception as exc:
        return DiffusionFit(success=False, error=str(exc))

    return DiffusionFit(success=True, D=popt[0], V=popt[1], Z=popt[2])

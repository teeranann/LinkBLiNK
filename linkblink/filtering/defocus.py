"""The defocusing filter: decides which detections survive to tracking.

Every criterion is still computed and recorded as a feature even when the
filter is disabled, so a downstream project can re-threshold offline from the
saved CSV without re-running detection.
"""

from __future__ import annotations

from ..config import FilterConfig
from ..features.gaussian import GaussianFit
from ..features.morphology import Component


def should_keep(
    component: Component,
    ecc: float,
    laplacian_var: float,
    fit: GaussianFit,
    config: FilterConfig,
) -> bool:
    """True if a detection looks like a single, in-focus, diffraction-limited particle.

    With ``config.disabled`` every detection is kept unconditionally.
    """
    if config.disabled:
        return True

    if not (config.min_particle_area <= component.area <= config.max_particle_area):
        return False
    if not (config.aspect_ratio_min <= component.aspect_ratio <= config.aspect_ratio_max):
        return False
    if component.extent < config.extent_min:
        return False
    if ecc > config.max_eccentricity:
        return False

    # An unfittable PSF is treated as a rejection, not as a missing measurement.
    if not fit.success:
        return False
    if fit.residual_sum > config.max_gaussian_residual_sum:
        return False
    if fit.sigma_aspect_ratio > config.max_gaussian_sigma_aspect_ratio:
        return False

    # Low Laplacian variance means smooth, i.e. out of focus.
    if laplacian_var <= config.laplacian_var_threshold:
        return False

    return True

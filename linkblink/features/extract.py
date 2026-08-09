"""Stage 3 — turn one frame's mask into measured, filtered particle records.

This is the orchestration layer over :mod:`morphology`, :mod:`gaussian`,
:mod:`focus` and :mod:`photometry`: for each connected component it measures
everything, asks the filter whether to keep it, and (for survivors) attaches an
appearance embedding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import CameraConfig, FilterConfig
from ..embedding import SiameseEmbedder
from ..filtering import should_keep
from .focus import laplacian_variance
from .gaussian import GaussianFit, fit_gaussian_2d, prepare_fit_pixels
from .morphology import Component, eccentricity, find_components
from .photometry import PhotonCounts, frame_background, local_background, photon_counts


@dataclass
class ParticleRecord:
    """One surviving detection.

    Field order defines the column order of
    ``filtered_particle_data_for_tracking.csv``; downstream tools and the
    published result files depend on it, so append rather than reorder.
    """

    frame: int
    x: float
    y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area: int
    aspect_ratio: float
    extent: float
    eccentricity: float
    laplacian_variance: float
    gaussian_fit_residual_sum: float
    fitted_sigma_aspect_ratio: float
    Ag_gaussian_amplitude: float
    sigma_x_pixels: float
    sigma_y_pixels: float
    fwhm_x_pixels: float
    fwhm_y_pixels: float
    fwhm_avg_pixels: float
    gaussian_offset: float
    r_squared_gaussian: float
    rmse_gaussian: float
    Ibcnt: float
    Ib_photons_emitted: float
    Idetect_photons_camera: float
    Iphelect_photoelectrons: float
    siamese_embedding: Any = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_record(
    component: Component,
    frame_number: int,
    ecc: float,
    laplacian_var: float,
    fit: GaussianFit,
    photons: PhotonCounts,
    embedding: np.ndarray | None,
) -> ParticleRecord:
    return ParticleRecord(
        frame=frame_number,
        x=component.centroid_x,
        y=component.centroid_y,
        bbox_x=component.x,
        bbox_y=component.y,
        bbox_w=component.width,
        bbox_h=component.height,
        area=component.area,
        aspect_ratio=component.aspect_ratio,
        extent=component.extent,
        eccentricity=ecc,
        laplacian_variance=laplacian_var,
        gaussian_fit_residual_sum=fit.residual_sum,
        fitted_sigma_aspect_ratio=fit.sigma_aspect_ratio,
        Ag_gaussian_amplitude=fit.amplitude,
        sigma_x_pixels=fit.sigma_x,
        sigma_y_pixels=fit.sigma_y,
        fwhm_x_pixels=fit.fwhm_x,
        fwhm_y_pixels=fit.fwhm_y,
        fwhm_avg_pixels=fit.fwhm_avg,
        gaussian_offset=fit.offset,
        r_squared_gaussian=fit.r_squared,
        rmse_gaussian=fit.rmse,
        Ibcnt=photons.background_subtracted_counts,
        Ib_photons_emitted=photons.photons_emitted,
        Idetect_photons_camera=photons.photons_at_camera,
        Iphelect_photoelectrons=photons.photoelectrons,
        siamese_embedding=embedding,
    )


def extract_frame_particles(
    frame: np.ndarray,
    mask: np.ndarray,
    frame_number: int,
    filter_config: FilterConfig,
    camera: CameraConfig,
    embedder: SiameseEmbedder | None = None,
) -> tuple[np.ndarray, list[ParticleRecord]]:
    """Measure and filter every particle in one frame.

    Args:
        frame: raw (un-normalised) frame — photometry needs the original counts.
        mask: binary U-Net mask for this frame.
        frame_number: index written into each record.
        embedder: if given, a loaded embedder; each kept particle gets a vector.

    Returns:
        ``(filtered_mask, records)`` where ``filtered_mask`` contains only the
        components that passed the filter.
    """
    height, width = frame.shape
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    records: list[ParticleRecord] = []

    labels, components = find_components(mask)
    if not components:
        return filtered_mask, records

    global_background = frame_background(frame)
    frame_is_16bit = frame.dtype == np.uint16 or frame.max() > 255

    for component in components:
        if component.width == 0 or component.height == 0:
            continue

        particle_mask = (labels == component.label).astype(np.uint8) * 255

        x_end = min(component.x + component.width, width)
        y_end = min(component.y + component.height, height)
        roi = frame[component.y : y_end, component.x : x_end]
        mask_roi = particle_mask[component.y : y_end, component.x : x_end]

        background = local_background(
            frame, mask_roi, component.x, component.y,
            component.width, component.height, global_background,
        )

        if roi.size == 0 or mask_roi.size == 0:
            continue

        ecc = eccentricity(mask_roi)
        laplacian_var = laplacian_variance(roi, mask_roi)

        fit_pixels = prepare_fit_pixels(roi, mask_roi, frame_is_16bit)
        y_coords, x_coords = np.where(mask_roi > 0)
        intensities = fit_pixels[y_coords, x_coords]
        fit = fit_gaussian_2d(
            x_coords, y_coords, intensities, component.width, component.height
        )

        photons = photon_counts(roi, mask_roi, component.area, background, camera)

        if not should_keep(component, ecc, laplacian_var, fit, filter_config):
            continue

        cv2.add(filtered_mask, particle_mask, dst=filtered_mask)

        embedding = None
        if embedder is not None:
            embedding = embedder.embed(frame, component.centroid_x, component.centroid_y)

        records.append(
            _build_record(component, frame_number, ecc, laplacian_var, fit, photons, embedding)
        )

    return filtered_mask, records

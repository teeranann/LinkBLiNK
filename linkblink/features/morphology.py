"""Shape descriptors derived from a connected component and its mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_POINTS_FOR_ELLIPSE = 5
"""cv2.fitEllipse needs at least five contour points."""


@dataclass
class Component:
    """One connected component of a U-Net mask."""

    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid_x: float
    centroid_y: float

    @property
    def aspect_ratio(self) -> float:
        return float(self.width) / self.height

    @property
    def extent(self) -> float:
        """Filled fraction of the bounding box. A blob fills more of it than a streak."""
        return float(self.area) / (self.width * self.height)


def find_components(mask: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    """Label a binary mask.

    Returns the label image and one :class:`Component` per particle. The
    background label (0) is excluded.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    components = [
        Component(
            label=i,
            x=int(stats[i, cv2.CC_STAT_LEFT]),
            y=int(stats[i, cv2.CC_STAT_TOP]),
            width=int(stats[i, cv2.CC_STAT_WIDTH]),
            height=int(stats[i, cv2.CC_STAT_HEIGHT]),
            area=int(stats[i, cv2.CC_STAT_AREA]),
            centroid_x=float(centroids[i][0]),
            centroid_y=float(centroids[i][1]),
        )
        for i in range(1, count)
    ]
    return labels, components


def eccentricity(mask_roi: np.ndarray) -> float:
    """Eccentricity of the best-fit ellipse: 0 is a circle, 1 is a line.

    Returns 1.0 (maximally elongated, so rejected by the filter) whenever the
    contour is too small or degenerate to fit — an unfittable shape is not a
    well-formed particle.
    """
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0

    contour = contours[0]
    if len(contour) < MIN_POINTS_FOR_ELLIPSE:
        return 1.0

    try:
        _centre, (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
    except cv2.error:
        return 1.0

    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if major == 0:
        return 0.0
    return float(np.sqrt(1 - (minor / major) ** 2))

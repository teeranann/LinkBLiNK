"""Shape, focus, PSF-fit and photometry measurements."""

import numpy as np
import pytest

from linkblink.config import CameraConfig
from linkblink.features import (
    eccentricity,
    find_components,
    fit_gaussian_2d,
    laplacian_variance,
    local_background,
    photon_counts,
)
from linkblink.features.gaussian import gaussian_2d


def disc_mask(size=40, centre=(20, 20), radius=5):
    yy, xx = np.ogrid[:size, :size]
    inside = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius**2
    return (inside.astype(np.uint8)) * 255


class TestComponents:
    def test_finds_two_blobs_and_excludes_background(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        mask[30:35, 30:35] = 255

        _labels, components = find_components(mask)
        assert len(components) == 2
        assert all(c.area == 25 for c in components)

    def test_bbox_geometry(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:14, 10:20] = 255  # 10 wide, 4 tall

        _labels, (component,) = find_components(mask)
        assert (component.width, component.height) == (10, 4)
        assert component.aspect_ratio == pytest.approx(2.5)
        assert component.extent == pytest.approx(1.0)

    def test_empty_mask_yields_no_components(self):
        _labels, components = find_components(np.zeros((20, 20), dtype=np.uint8))
        assert components == []


class TestEccentricity:
    def test_circle_is_near_zero(self):
        assert eccentricity(disc_mask()) < 0.5

    def test_elongated_shape_is_high(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[20:23, 5:35] = 255
        assert eccentricity(mask) > 0.9

    def test_degenerate_shape_returns_maximum(self):
        # Too few contour points to fit an ellipse.
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255
        assert eccentricity(mask) == 1.0


class TestLaplacianVariance:
    def test_sharp_edge_scores_higher_than_flat_region(self):
        mask = np.ones((20, 20), dtype=np.uint8) * 255

        flat = np.full((20, 20), 100, dtype=np.uint8)
        sharp = np.zeros((20, 20), dtype=np.uint8)
        sharp[8:12, 8:12] = 255

        assert laplacian_variance(sharp, mask) > laplacian_variance(flat, mask)

    def test_empty_roi_returns_zero(self):
        assert laplacian_variance(np.zeros((0, 0)), np.zeros((0, 0))) == 0.0


class TestGaussianFit:
    def test_recovers_known_parameters(self):
        yy, xx = np.mgrid[0:21, 0:21]
        x_coords = xx.ravel()
        y_coords = yy.ravel()

        truth = gaussian_2d((x_coords, y_coords), 200.0, 10.0, 10.0, 2.0, 3.0, 0.0, 20.0)
        fit = fit_gaussian_2d(x_coords, y_coords, truth, bbox_w=21, bbox_h=21)

        assert fit.success
        # theta is free, so the fit may report the widths in either order.
        assert sorted([fit.sigma_x, fit.sigma_y]) == pytest.approx([2.0, 3.0], abs=0.05)
        assert fit.sigma_aspect_ratio == pytest.approx(1.5, abs=0.05)
        assert fit.r_squared > 0.999

    def test_fwhm_follows_sigma(self):
        yy, xx = np.mgrid[0:21, 0:21]
        truth = gaussian_2d((xx.ravel(), yy.ravel()), 200.0, 10.0, 10.0, 2.0, 2.0, 0.0, 10.0)
        fit = fit_gaussian_2d(xx.ravel(), yy.ravel(), truth, 21, 21)
        assert fit.fwhm_avg == pytest.approx(2.0 * 2.355, rel=0.05)

    def test_empty_input_fails_permissively(self):
        fit = fit_gaussian_2d(np.array([]), np.array([]), np.array([]), 5, 5)
        assert not fit.success
        # Defaults must not accidentally pass the filter thresholds.
        assert fit.residual_sum == float("inf")
        assert fit.sigma_aspect_ratio == float("inf")


class TestPhotometry:
    def test_background_is_the_local_median(self):
        frame = np.full((40, 40), 50, dtype=np.uint16)
        frame[18:22, 18:22] = 1000
        mask_roi = np.ones((4, 4), dtype=np.uint8) * 255

        assert local_background(frame, mask_roi, 18, 18, 4, 4, fallback=0.0) == 50.0

    def test_falls_back_when_no_background_pixels_exist(self):
        # Particle fills the whole frame, so the surrounding ring is empty.
        frame = np.full((4, 4), 500, dtype=np.uint16)
        mask_roi = np.ones((4, 4), dtype=np.uint8) * 255
        assert local_background(frame, mask_roi, 0, 0, 4, 4, fallback=42.0) == 42.0

    def test_photon_chain_is_consistent(self):
        camera = CameraConfig()
        roi = np.full((3, 3), 100, dtype=np.uint16)
        mask_roi = np.ones((3, 3), dtype=np.uint8) * 255

        counts = photon_counts(roi, mask_roi, area=9, background=10.0, camera=camera)

        assert counts.background_subtracted_counts == pytest.approx(900 - 90)
        assert counts.photoelectrons == pytest.approx(810 * camera.gain_electrons_per_count)
        assert counts.photons_at_camera == pytest.approx(
            counts.photoelectrons / camera.quantum_efficiency
        )
        assert counts.photons_emitted == pytest.approx(
            counts.photons_at_camera / camera.collection_efficiency
        )
        # Each step upstream recovers more photons than the last.
        assert counts.photons_emitted > counts.photons_at_camera > counts.photoelectrons

    def test_counts_clamp_at_zero(self):
        roi = np.full((2, 2), 10, dtype=np.uint16)
        mask_roi = np.ones((2, 2), dtype=np.uint8) * 255
        counts = photon_counts(roi, mask_roi, area=4, background=1000.0, camera=CameraConfig())
        assert counts.background_subtracted_counts == 0.0

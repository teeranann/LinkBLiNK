"""MSD computation against analytically known trajectories."""

import numpy as np
import pandas as pd
import pytest

from linkblink.analysis import calculate_msd


def trajectory(xs, ys, particle=0, start_frame=1):
    return pd.DataFrame({
        "frame": np.arange(start_frame, start_frame + len(xs)),
        "x": xs,
        "y": ys,
        "particle": particle,
    })


def test_constant_velocity_gives_quadratic_msd():
    # One pixel per frame along x. With pixelsize 1 nm, MSD_x(lag) = lag^2.
    df = trajectory(np.arange(6, dtype=float), np.zeros(6))
    msd = calculate_msd(df, pixelsize_nm=1.0, frame_rate_hz=1.0)

    assert msd["msd_x_nm2"].tolist() == pytest.approx([1.0, 4.0, 9.0, 16.0, 25.0])
    assert msd["msd_y_nm2"].tolist() == pytest.approx([0.0] * 5)
    assert msd["msd_xy_nm2"].tolist() == pytest.approx([1.0, 4.0, 9.0, 16.0, 25.0])


def test_stationary_particle_has_zero_msd():
    df = trajectory(np.full(5, 7.0), np.full(5, 3.0))
    msd = calculate_msd(df, pixelsize_nm=35.9, frame_rate_hz=100.0)
    assert msd["msd_xy_nm2"].tolist() == pytest.approx([0.0] * 4)


def test_pixelsize_scales_msd_quadratically():
    df = trajectory(np.arange(4, dtype=float), np.zeros(4))
    at_1nm = calculate_msd(df, pixelsize_nm=1.0)
    at_10nm = calculate_msd(df, pixelsize_nm=10.0)
    assert at_10nm["msd_x_nm2"].values == pytest.approx(at_1nm["msd_x_nm2"].values * 100)


def test_lag_time_uses_frame_rate():
    df = trajectory(np.arange(4, dtype=float), np.zeros(4))
    msd = calculate_msd(df, pixelsize_nm=1.0, frame_rate_hz=100.0)
    assert msd["lag_time_s"].tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_max_lag_frames_caps_the_curve():
    df = trajectory(np.arange(20, dtype=float), np.zeros(20))
    capped = calculate_msd(df, pixelsize_nm=1.0, max_lag_frames=5)
    assert capped["lag_time_frames"].max() == 5


def test_zero_max_lag_uses_full_trajectory():
    df = trajectory(np.arange(8, dtype=float), np.zeros(8))
    full = calculate_msd(df, pixelsize_nm=1.0, max_lag_frames=0)
    assert full["lag_time_frames"].max() == 7


def test_single_point_trajectory_is_skipped():
    df = trajectory([1.0], [1.0])
    assert calculate_msd(df, pixelsize_nm=1.0).empty


def test_multiple_particles_are_reported_separately():
    df = pd.concat([
        trajectory(np.arange(5, dtype=float), np.zeros(5), particle=0),
        trajectory(np.zeros(5), np.arange(5, dtype=float), particle=1),
    ])
    msd = calculate_msd(df, pixelsize_nm=1.0)
    assert set(msd["particle"]) == {0, 1}
    particle_1 = msd[msd["particle"] == 1]
    assert particle_1["msd_x_nm2"].tolist() == pytest.approx([0.0] * 4)

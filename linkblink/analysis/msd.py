"""Mean squared displacement.

MSD(τ) averaged over every starting point in a trajectory. Its slope gives the
diffusion coefficient; curvature separates directed motion from free diffusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


def calculate_msd(
    trajectories: pd.DataFrame,
    pixelsize_nm: float,
    frame_rate_hz: float = 1.0,
    max_lag_frames: int = 0,
) -> pd.DataFrame:
    """Compute MSD in x, y and xy for every trajectory.

    Args:
        trajectories: linked tracks with ``x``, ``y``, ``frame``, ``particle``.
        pixelsize_nm: converts pixel coordinates to nanometres.
        frame_rate_hz: converts lag in frames to lag in seconds.
        max_lag_frames: cap on lag time. 0 uses the full trajectory length.
            Long lags average over few pairs and are dominated by noise, so a
            cap of roughly a tenth of the track length is typical.

    Returns:
        Long-format frame with one row per (particle, lag), or an empty frame
        when no trajectory is longer than a single point.
    """
    log.info("\nCalculating Mean Squared Displacement (MSD) for X, Y, and XY...")

    per_particle: list[pd.DataFrame] = []

    for particle_id, trajectory in trajectories.groupby("particle"):
        trajectory = trajectory.reset_index(drop=True).sort_values(by="frame")

        x_nm = trajectory["x"].values * pixelsize_nm
        y_nm = trajectory["y"].values * pixelsize_nm

        longest_possible_lag = len(trajectory) - 1
        if longest_possible_lag == 0:
            continue  # a single detection has no displacement

        if max_lag_frames == 0 or max_lag_frames >= longest_possible_lag:
            effective_max_lag = longest_possible_lag
        else:
            effective_max_lag = max_lag_frames

        lags = np.arange(1, effective_max_lag + 1)

        msd_x, msd_y, msd_xy = [], [], []
        for lag in lags:
            # Every pair separated by `lag` samples, not just consecutive ones.
            dx_squared = (x_nm[lag:] - x_nm[:-lag]) ** 2
            dy_squared = (y_nm[lag:] - y_nm[:-lag]) ** 2

            msd_x.append(np.mean(dx_squared))
            msd_y.append(np.mean(dy_squared))
            msd_xy.append(np.mean(dx_squared + dy_squared))

        per_particle.append(
            pd.DataFrame(
                {
                    "particle": particle_id,
                    "lag_time_frames": lags,
                    "lag_time_s": lags / frame_rate_hz,
                    "msd_x_nm2": msd_x,
                    "msd_y_nm2": msd_y,
                    "msd_xy_nm2": msd_xy,
                }
            )
        )

    if not per_particle:
        log.info("No particles to calculate MSD for.")
        return pd.DataFrame()

    log.info("MSD calculation complete.")
    return pd.concat(per_particle, ignore_index=True)

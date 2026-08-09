"""Detected photons per particle over time — the photobleaching view."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.logging import get_logger
from .style import apply_style

log = get_logger(__name__)

INTENSITY_COLUMN = "Idetect_photons_camera"


def plot_detected_photons_over_time(
    trajectories: pd.DataFrame,
    output_dir: Path,
    video_name: str,
    frame_rate_hz: float,
    smoothing_window: int = 5,
) -> None:
    """Plot every particle's smoothed brightness against elapsed time.

    A centred moving average takes out shot noise so the underlying bleaching
    decay is visible. Time is measured from each particle's own first frame.
    """
    if trajectories.empty:
        log.info("No linked particle data to plot detected photons.")
        return

    if INTENSITY_COLUMN not in trajectories.columns:
        log.error("Error: '%s' column not found in trajectories.", INTENSITY_COLUMN)
        return

    apply_style()

    particles = trajectories["particle"].unique()
    log.info("Generating Detected Photons over Time plots for %d particles...", len(particles))

    fig, ax = plt.subplots(figsize=(10, 7))

    for particle_id in particles:
        particle_data = (
            trajectories[trajectories["particle"] == particle_id]
            .reset_index(drop=True)
            .sort_values(by="frame")
            .copy()
        )
        if particle_data.empty:
            continue

        elapsed_s = (particle_data["frame"] - particle_data["frame"].min()) / frame_rate_hz
        smoothed = (
            particle_data[INTENSITY_COLUMN]
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
        )

        ax.plot(elapsed_s, smoothed, label=f"Particle {particle_id}", linewidth=1.5)

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Detected photons/Particle/Frame", fontsize=14)
    ax.set_title(f"Detected Photons Over Time for {video_name}", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10, loc="upper right", frameon=True, edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = output_dir / f"{video_name}_detected_photons_over_time_all_particles.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Detected Photons Over Time plot saved to %s", plot_path)

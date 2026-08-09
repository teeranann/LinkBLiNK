"""One MSD figure per particle: measured curves plus fitted diffusion models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.diffusion import diffusion_model_1d, diffusion_model_2d, fit_msd_curve
from ..utils.logging import get_logger
from .style import apply_style, frame_axes

log = get_logger(__name__)

NM2_TO_UM2 = 1e-6
"""(1e-3 µm/nm)² — MSD is an area, so the length conversion is squared."""

CURVES = {
    "msd_x_um2": ("tab:blue", r"MSD_x"),
    "msd_y_um2": ("tab:red", r"MSD_y"),
    "msd_xy_um2": ("tab:orange", r"MSD_{xy}"),
}


def plot_individual_msd_curves(
    msd_df: pd.DataFrame, output_dir: Path, video_name: str
) -> None:
    """Save ``<video>_particle_<id>_msd_plot.png`` for every particle.

    Solid lines are measured MSD, dashed lines the fitted diffusion model.
    """
    if msd_df.empty:
        log.info("No MSD data to plot.")
        return

    apply_style()

    msd_df = msd_df.copy()
    for axis in ("x", "y", "xy"):
        msd_df[f"msd_{axis}_um2"] = msd_df[f"msd_{axis}_nm2"] * NM2_TO_UM2

    particles = msd_df["particle"].unique()
    log.info("Generating individual MSD plots for %d particles...", len(particles))

    for particle_id in particles:
        particle_msd = msd_df[msd_df["particle"] == particle_id].copy()
        fig, ax = plt.subplots(figsize=(8, 6))

        for column, (colour, label) in CURVES.items():
            values = particle_msd[column].dropna()
            lag_times = particle_msd["lag_time_s"][values.index]

            values_np = values.values
            lag_times_np = lag_times.values

            fit = fit_msd_curve(lag_times_np, values_np, two_dimensional=column == "msd_xy_um2")
            if fit.success:
                model = diffusion_model_2d if column == "msd_xy_um2" else diffusion_model_1d
                ax.plot(
                    lag_times_np, model(lag_times_np, fit.D, fit.V, fit.Z),
                    color=colour, linestyle="--", alpha=0.8, linewidth=1.5,
                )
                log.info(
                    "Particle %s - Fit for %s: D = %.4e um^2/s, V = %.4e um/s, Z = %.4e um^2",
                    particle_id, column, fit.D, fit.V, fit.Z,
                )
            elif fit.error:
                log.info("Particle %s - Could not fit %s: %s", particle_id, column, fit.error)

            ax.plot(lag_times_np, values_np, color=colour, label=label, linewidth=2.0)

        ax.set_xlabel("Lag time (s)", fontsize=14)
        ax.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=14)
        ax.set_title(f"MSD for {video_name} - Particle {particle_id}", fontsize=16)
        frame_axes(ax)
        ax.legend(fontsize=12, loc="upper left", frameon=True, edgecolor="black")
        ax.grid(False)

        max_lag = particle_msd["lag_time_s"].max()
        ax.set_xlim(left=0, right=max_lag + max_lag * 0.05)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plot_path = output_dir / f"{video_name}_particle_{particle_id}_msd_plot.png"
        plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("MSD plot for Particle %s saved to %s", particle_id, plot_path)

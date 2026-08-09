"""Overview figure: every trajectory drawn in image coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.logging import get_logger
from .style import apply_style

log = get_logger(__name__)


def plot_trajectories(
    trajectories: pd.DataFrame,
    output_dir: Path,
    image_width: int,
    image_height: int,
    show: bool = False,
) -> Path | None:
    """Save ``particle_trajectories_with_legend.png``.

    Axes match the image frame: y runs downward and the aspect is locked, so
    the plot overlays the raw video without distortion.
    """
    if trajectories.empty:
        log.info("No trajectories to plot after filtering.")
        return None

    apply_style()

    fig, ax = plt.subplots(figsize=(12, 10))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, particle_id in enumerate(trajectories["particle"].unique()):
        track = trajectories[trajectories["particle"] == particle_id]
        ax.plot(track["x"], track["y"], label=f"ID {particle_id}", color=colours[i % len(colours)])

    ax.set_title("Particle Trajectories")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(True)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # image convention: origin top-left
    ax.set_aspect("equal", adjustable="box")
    ax.legend(title="Particle ID", bbox_to_anchor=(1.05, 1), loc="upper left")

    plot_path = output_dir / "particle_trajectories_with_legend.png"
    fig.savefig(str(plot_path), bbox_inches="tight")
    log.info("Trajectory plot with legend saved to %s", plot_path)

    if show:
        plt.show()
    plt.close(fig)

    return plot_path

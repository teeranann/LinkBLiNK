"""Figure generation. Every function saves to disk and closes its figure."""

from .msd_plots import plot_individual_msd_curves
from .photon_plots import plot_detected_photons_over_time
from .style import apply_style
from .trajectory_plots import plot_trajectories

__all__ = [
    "apply_style",
    "plot_detected_photons_over_time",
    "plot_individual_msd_curves",
    "plot_trajectories",
]

"""Shared matplotlib styling.

Applied explicitly rather than at import time, so importing the package never
mutates a host application's global matplotlib state.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

STYLE = "seaborn-v0_8-whitegrid"

_APPLIED = False


def apply_style() -> None:
    """Install the project plot style once per process."""
    global _APPLIED
    if _APPLIED:
        return
    plt.style.use(STYLE)
    _APPLIED = True


def frame_axes(ax) -> None:
    """Apply the publication axis treatment: inward ticks, heavy black spines."""
    ax.tick_params(
        axis="both", which="both", direction="in",
        length=6, width=1.5, color="black", labelsize=12,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

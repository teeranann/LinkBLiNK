"""Stage 4 — building trajectories, then repairing them across blinking gaps."""

from .judge import apply_judge, parse_embedding, rf_features
from .nearest_neighbor import drop_short_trajectories, link_nearest_neighbour
from .segments import Segment, build_segments

__all__ = [
    "Segment",
    "apply_judge",
    "build_segments",
    "drop_short_trajectories",
    "link_nearest_neighbour",
    "parse_embedding",
    "rf_features",
]

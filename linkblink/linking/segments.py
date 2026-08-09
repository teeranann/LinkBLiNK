"""Trajectory segments — the unit the judge reasons about.

A segment collapses a whole trajectory down to what matters when deciding
whether two tracks are the same particle: when it started, when it ended, and
the detections at each end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Segment:
    """One continuous trajectory, reduced to its endpoints."""

    particle: int
    start_frame: int
    end_frame: int
    first_row: pd.Series
    last_row: pd.Series


def build_segments(df: pd.DataFrame) -> list[Segment]:
    """Reduce a linked-trajectory table to one :class:`Segment` per particle."""
    reindexed = df.reset_index(drop=True)

    segments: list[Segment] = []
    for particle_id, group in reindexed.groupby("particle"):
        ordered = group.sort_values("frame")
        segments.append(
            Segment(
                particle=int(particle_id),
                start_frame=int(ordered["frame"].iloc[0]),
                end_frame=int(ordered["frame"].iloc[-1]),
                first_row=ordered.iloc[0],
                last_row=ordered.iloc[-1],
            )
        )
    return segments

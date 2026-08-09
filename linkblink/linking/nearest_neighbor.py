"""Frame-to-frame nearest-neighbour linking.

Deliberately conservative: it only connects detections in *consecutive* frames
and never bridges a gap. Blinking gaps are the judge's job, and keeping the two
concerns separate means a mis-link here cannot be laundered into a long
trajectory by the gap-bridging logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def link_nearest_neighbour(df: pd.DataFrame, max_displacement_px: float) -> pd.DataFrame:
    """Assign integer ``particle`` IDs by greedy nearest-neighbour matching.

    Candidate pairs within ``max_displacement_px`` are sorted by distance and
    claimed shortest-first, so each track and each detection is used at most
    once. Unmatched detections start new tracks.

    Args:
        df: detections with ``frame``, ``x``, ``y`` columns.
        max_displacement_px: maximum centroid movement between frames.

    Returns:
        A copy of ``df`` sorted by frame with a ``particle`` column added.
    """
    if df.empty:
        return df

    df = df.sort_values("frame").reset_index(drop=True)
    frames = df["frame"].unique()

    particle_ids = np.full(len(df), -1, dtype=int)
    next_id = 0

    # track_id -> row index of that track's most recent detection
    active_tracks: dict[int, int] = {}

    first_frame_rows = df.index[df["frame"] == frames[0]].tolist()
    for idx in first_frame_rows:
        particle_ids[idx] = next_id
        active_tracks[next_id] = idx
        next_id += 1

    frame_set = set(frames)
    for frame in frames[1:]:
        previous_frame = frame - 1
        # A missing intermediate frame breaks every track: no gap bridging here.
        if previous_frame not in frame_set:
            active_tracks = {}

        current_rows = df.index[df["frame"] == frame].tolist()
        if not current_rows:
            continue

        candidates: list[tuple[float, int, int]] = []
        for track_id, last_idx in active_tracks.items():
            if df.at[last_idx, "frame"] != previous_frame:
                continue
            x1 = df.at[last_idx, "x"]
            y1 = df.at[last_idx, "y"]
            for idx in current_rows:
                distance = float(np.hypot(df.at[idx, "x"] - x1, df.at[idx, "y"] - y1))
                if distance <= max_displacement_px:
                    candidates.append((distance, track_id, idx))

        candidates.sort(key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        new_active_tracks: dict[int, int] = {}

        for _distance, track_id, idx in candidates:
            if track_id in used_tracks or idx in used_detections:
                continue
            particle_ids[idx] = track_id
            used_tracks.add(track_id)
            used_detections.add(idx)
            new_active_tracks[track_id] = idx

        for idx in current_rows:
            if idx in used_detections:
                continue
            particle_ids[idx] = next_id
            new_active_tracks[next_id] = idx
            next_id += 1

        active_tracks = new_active_tracks

    df = df.copy()
    df["particle"] = particle_ids.astype(int)
    return df


def drop_short_trajectories(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Remove tracks shorter than ``min_length`` frames.

    Short stubs are usually noise detections that happened to fall within the
    search radius for a frame or two.
    """
    lengths = df.groupby("particle").size()
    keep_ids = lengths[lengths >= min_length].index
    return df[df["particle"].isin(keep_ids)].copy()

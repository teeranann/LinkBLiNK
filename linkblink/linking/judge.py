"""Random-forest re-identification judge.

Nearest-neighbour linking breaks a trajectory every time a fluorophore blinks
off. The judge repairs those breaks: for each track that *starts*, it looks
back at tracks that recently *ended* nearby and asks a trained random forest
whether they are the same particle, combining appearance (Siamese distance),
kinematics (gap, distance) and photometry (size, brightness).

Merging is iterative and restarts after every accepted merge, because a merge
changes the segment structure the next decision is made against.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import JudgeConfig
from ..utils.dataframe import normalise_track_frame
from ..utils.logging import get_logger
from .segments import Segment, build_segments

log = get_logger(__name__)

INF_REPLACEMENT = 1e9
"""Sentinel the RF sees in place of inf/NaN — trees split fine on a large finite value."""


def parse_embedding(embedding: Any) -> np.ndarray:
    """Read a Siamese embedding that may be an array or its CSV string form.

    Records round-trip through CSV between stages, so ``"[0.1 0.2 ...]"`` and a
    real ndarray both have to work. Anything unparseable yields an empty array,
    which the caller treats as "no appearance evidence".
    """
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32)
    if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
        return np.array([], dtype=np.float32)

    text = str(embedding).strip()
    if len(text) >= 2 and text[0] in "[(" and text[-1] in "])":
        text = text[1:-1]
    text = text.replace(",", " ")

    try:
        return np.asarray([float(p) for p in text.split() if p], dtype=np.float32)
    except ValueError:
        return np.array([], dtype=np.float32)


def _safe_difference(a: Any, b: Any) -> float:
    try:
        return float(abs(float(a) - float(b)))
    except (TypeError, ValueError):
        return float("inf")


def _safe_ratio(a: Any, b: Any) -> float:
    try:
        denominator = float(b) if float(b) != 0 else 1e-6
        return float(a) / denominator
    except (TypeError, ValueError):
        return float("inf")


def rf_features(row_end: pd.Series, row_start: pd.Series) -> dict[str, float]:
    """Build the feature vector comparing a track's last row to another's first.

    Args:
        row_end: final detection of the earlier trajectory.
        row_start: first detection of the later trajectory.
    """
    emb_end = parse_embedding(row_end.get("siamese_embedding", None))
    emb_start = parse_embedding(row_start.get("siamese_embedding", None))

    if emb_end.size and emb_start.size and emb_end.shape == emb_start.shape:
        siamese_distance = float(np.linalg.norm(emb_end - emb_start))
    else:
        # No comparable appearance evidence: inf strongly discourages a merge.
        siamese_distance = float("inf")

    return {
        "siamese_euclidean_distance": siamese_distance,
        "position_euclidean_distance": float(
            np.hypot(row_end["x"] - row_start["x"], row_end["y"] - row_start["y"])
        ),
        "time_difference": int(row_start["frame"] - row_end["frame"]),
        "area_difference": _safe_difference(row_end.get("area", 0), row_start.get("area", 0)),
        "Ibcnt_difference": _safe_difference(row_end.get("Ibcnt", 0), row_start.get("Ibcnt", 0)),
        "fwhm_avg_difference": _safe_difference(
            row_end.get("fwhm_avg_pixels", 0), row_start.get("fwhm_avg_pixels", 0)
        ),
        "area_ratio": _safe_ratio(row_end.get("area", 0), row_start.get("area", 0)),
        "Ibcnt_ratio": _safe_ratio(row_end.get("Ibcnt", 0), row_start.get("Ibcnt", 0)),
        "fwhm_avg_ratio": _safe_ratio(
            row_end.get("fwhm_avg_pixels", 0), row_start.get("fwhm_avg_pixels", 0)
        ),
    }


def _candidate_predecessors(
    segment: Segment, segments: list[Segment], config: JudgeConfig
) -> list[tuple[Segment, int, float]]:
    """Segments that ended shortly before ``segment`` started, and close by.

    Cheap gating before the model runs: only gap and distance, no inference.
    """
    candidates: list[tuple[Segment, int, float]] = []
    for other in segments:
        if other.particle == segment.particle:
            continue

        gap = segment.start_frame - other.end_frame
        if not (1 <= gap <= config.max_time_gap_frames):
            continue

        distance = float(
            np.hypot(
                other.last_row["x"] - segment.first_row["x"],
                other.last_row["y"] - segment.first_row["y"],
            )
        )
        if distance <= config.max_start_distance_px:
            candidates.append((other, gap, distance))

    return candidates


def apply_judge(
    linked_df: pd.DataFrame, config: JudgeConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Merge trajectory IDs that the random forest believes are one particle.

    Returns:
        ``(judged_df, merge_log_df, merge_count)``. When the judge is disabled
        or the model is missing, the input is returned normalised but unchanged.
    """
    judged = normalise_track_frame(linked_df)

    if not config.enabled:
        log.info("RF judge disabled in config. Skipping.")
        return judged, pd.DataFrame(), 0

    if not config.model_path or not config.model_path.exists():
        log.warning("RF model not found at: %s. Skipping judge step.", config.model_path)
        return judged, pd.DataFrame(), 0

    log.info("\n--- RF Judge: Loading model ---")
    model = joblib.load(config.model_path)
    # Read the feature set from the model itself, so a retrained forest with a
    # different feature list keeps working without a code change.
    expected_features = list(model.feature_names_in_)

    merges: list[dict[str, Any]] = []
    changed = True

    while changed:
        changed = False

        segments = build_segments(judged)
        by_start = sorted(segments, key=lambda s: s.start_frame)
        by_end = sorted(segments, key=lambda s: s.end_frame)

        for segment in by_start:
            # This ID may have been absorbed by an earlier merge in this pass.
            live_ids = set(
                pd.to_numeric(judged["particle"], errors="coerce").dropna().astype(int).unique()
            )
            if segment.particle not in live_ids:
                continue

            candidates = _candidate_predecessors(segment, by_end, config)
            if not candidates:
                continue

            best: tuple[Segment, int, float, dict[str, float]] | None = None
            best_proba = -1.0

            for other, gap, distance in candidates:
                features = rf_features(other.last_row, segment.first_row)
                row = {name: features[name] for name in expected_features}

                X = pd.DataFrame([row], columns=expected_features)
                X = X.replace([np.inf, -np.inf], INF_REPLACEMENT).fillna(INF_REPLACEMENT)

                try:
                    proba = float(model.predict_proba(X)[:, 1][0])
                except Exception as exc:
                    log.warning("Warning: RF prediction failed: %s", exc)
                    continue

                if proba > best_proba:
                    best_proba = proba
                    best = (other, gap, distance, features)

            if best and best_proba >= config.min_confidence:
                other, gap, distance, features = best
                judged.loc[judged["particle"] == segment.particle, "particle"] = other.particle
                merges.append(
                    {
                        "merged_into": int(other.particle),
                        "merged_from": int(segment.particle),
                        "confidence": best_proba,
                        "time_gap_frames": int(gap),
                        "start_distance_px": float(distance),
                        **features,
                    }
                )
                changed = True
                break  # segment table is now stale; rebuild it

    merge_log = pd.DataFrame(merges)
    judged = judged.sort_values(["particle", "frame"]).reset_index(drop=True)
    return judged, merge_log, len(merges)

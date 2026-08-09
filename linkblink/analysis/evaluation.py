"""Benchmarking against ground truth.

Two independent families of metric:

* **Detection** (precision/recall/F1) is scored on the *pre-linking* table, so
  it measures the U-Net and the filter alone.
* **Tracking** (fragmentation, false linkage, completeness, association F1) is
  scored on the *post-judge* trajectories, so it measures linking and
  re-identification.

Ground-truth columns are ``Frame``, ``ID``, ``X_pix``, ``Y_pix``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import EVALUATION_METRICS_CSV, FILTERED_PARTICLES_CSV
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DetectionMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class TrackingMetrics:
    fragmentation_rate: float = 0.0
    false_linkage_rate: float = 0.0
    avg_trajectory_completeness: float = 0.0
    assoc_precision: float = 0.0
    assoc_recall: float = 0.0
    assoc_f1_score: float = 0.0


def score_detections(
    pre_link_csv: Path, ground_truth: pd.DataFrame, distance_threshold: float
) -> DetectionMetrics:
    """Precision/recall/F1 of raw detections against ground-truth positions."""
    if not pre_link_csv.exists():
        log.error("Error: Could not find %s for detection evaluation.", pre_link_csv)
        return DetectionMetrics()

    detected = pd.read_csv(pre_link_csv)[["frame", "x", "y"]].copy()
    detected = detected.rename(columns={"x": "x_det", "y": "y_det"})
    detected["det_id"] = detected.index

    merged = pd.merge(detected, ground_truth, left_on="frame", right_on="Frame", how="left")
    merged = merged.dropna(subset=["X_pix", "Y_pix"])

    merged["distance"] = np.sqrt(
        (merged["x_det"] - merged["X_pix"]) ** 2 + (merged["y_det"] - merged["Y_pix"]) ** 2
    )

    # Closest-first, then one match per GT point and one per detection.
    matches = merged[merged["distance"] <= distance_threshold].sort_values("distance")
    matches = matches.drop_duplicates(subset=["frame", "ID"])
    matches = matches.drop_duplicates(subset=["frame", "det_id"])

    true_positives = len(matches)
    precision = true_positives / len(detected) if len(detected) > 0 else 0.0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return DetectionMetrics(precision=precision, recall=recall, f1_score=f1)


def score_tracking(
    judged_df: pd.DataFrame, ground_truth: pd.DataFrame, distance_threshold: float
) -> TrackingMetrics:
    """Trajectory-level metrics for linked and judged tracks."""
    if judged_df.empty:
        log.warning("Judged dataframe is empty. Tracking metrics will be 0.")
        return TrackingMetrics()

    tracked = judged_df.rename(
        columns={"x": "x_tracked", "y": "y_tracked", "particle": "particle_tracked"}
    )

    merged = pd.merge(tracked, ground_truth, left_on="frame", right_on="Frame", how="left")
    merged["distance"] = np.sqrt(
        (merged["x_tracked"] - merged["X_pix"]) ** 2
        + (merged["y_tracked"] - merged["Y_pix"]) ** 2
    )

    # One GT assignment per (frame, tracked particle): the nearest.
    nearest = merged.groupby(["frame", "particle_tracked"])["distance"].idxmin()
    matched = merged.loc[nearest].reset_index(drop=True)
    matched.rename(columns={"ID": "gt_particle"}, inplace=True)

    gt_groups = ground_truth.groupby("ID")
    gt_count = ground_truth["ID"].nunique()

    # Fragmentation: one true particle split across several tracked IDs.
    fragmented = sum(
        1
        for gt_id, _ in gt_groups
        if len(matched[matched["gt_particle"] == gt_id]["particle_tracked"].unique()) > 1
    )
    fragmentation_rate = fragmented / gt_count if gt_count > 0 else 0.0

    # False linkage: one tracked ID spanning several true particles.
    tracked_count = tracked["particle_tracked"].nunique()
    false_linked = sum(
        1
        for tracked_id, _ in tracked.groupby("particle_tracked")
        if len(matched[matched["particle_tracked"] == tracked_id]["gt_particle"].unique()) > 1
    )
    false_linkage_rate = false_linked / tracked_count if tracked_count > 0 else 0.0

    # Completeness: fraction of each true trajectory actually recovered.
    completeness_scores = []
    for gt_id, gt_traj in gt_groups:
        if len(gt_traj) == 0:
            continue
        recovered = matched[matched["gt_particle"] == gt_id]["frame"].nunique()
        completeness_scores.append(recovered / len(gt_traj))
    avg_completeness = float(np.mean(completeness_scores)) if completeness_scores else 0.0

    assoc = _association_f1(matched, ground_truth, distance_threshold)

    return TrackingMetrics(
        fragmentation_rate=fragmentation_rate,
        false_linkage_rate=false_linkage_rate,
        avg_trajectory_completeness=avg_completeness,
        assoc_precision=assoc[0],
        assoc_recall=assoc[1],
        assoc_f1_score=assoc[2],
    )


def _association_f1(
    matched: pd.DataFrame, ground_truth: pd.DataFrame, distance_threshold: float
) -> tuple[float, float, float]:
    """Score individual frame-to-frame *links* rather than whole trajectories.

    A link is correct when both of its endpoints map to the same true particle.
    This catches ID swaps that trajectory-level metrics can average away.
    """
    matched = matched.copy()
    # Beyond the threshold there is no valid GT assignment at all.
    matched.loc[matched["distance"] > distance_threshold, "gt_particle"] = np.nan

    # Precision: of the links the tracker made, how many were right?
    predicted = matched.sort_values(["particle_tracked", "frame"]).copy()
    predicted["next_particle_tracked"] = predicted["particle_tracked"].shift(-1)
    predicted["next_gt_particle"] = predicted["gt_particle"].shift(-1)

    made_links = predicted[predicted["particle_tracked"] == predicted["next_particle_tracked"]]
    correct = (made_links["gt_particle"] == made_links["next_gt_particle"]) & (
        made_links["gt_particle"].notna()
    )
    true_positive_links = correct.sum()
    false_positive_links = (~correct).sum()

    # Recall: of the links that truly exist, how many did the tracker make?
    gt_sorted = ground_truth.sort_values(["ID", "Frame"]).copy()
    gt_mapped = pd.merge(
        gt_sorted,
        matched[["frame", "gt_particle", "particle_tracked"]],
        left_on=["Frame", "ID"],
        right_on=["frame", "gt_particle"],
        how="left",
    )
    gt_mapped["next_ID"] = gt_mapped["ID"].shift(-1)
    gt_mapped["next_particle_tracked"] = gt_mapped["particle_tracked"].shift(-1)

    gt_links = gt_mapped[gt_mapped["ID"] == gt_mapped["next_ID"]]
    recovered = (gt_links["particle_tracked"] == gt_links["next_particle_tracked"]) & (
        gt_links["particle_tracked"].notna()
    )
    false_negative_links = (~recovered).sum()

    denominator_p = true_positive_links + false_positive_links
    denominator_r = true_positive_links + false_negative_links

    precision = true_positive_links / denominator_p if denominator_p > 0 else 0.0
    recall = true_positive_links / denominator_r if denominator_r > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


def evaluate_tracking_performance(
    judged_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    video_name: str,
    output_dir: Path,
    distance_threshold: float = 3.0,
) -> dict[str, float] | None:
    """Score one video and append the row to ``evaluation_metrics.csv``."""
    log.info("\n--- Evaluating performance for '%s' ---", video_name)

    if ground_truth.empty:
        log.warning("Skipping evaluation: ground truth dataframe is empty.")
        return None

    detection = score_detections(
        output_dir / FILTERED_PARTICLES_CSV, ground_truth, distance_threshold
    )
    tracking = score_tracking(judged_df, ground_truth, distance_threshold)

    metrics = {
        "video": video_name,
        "precision": detection.precision,
        "recall": detection.recall,
        "f1_score": detection.f1_score,
        "fragmentation_rate": tracking.fragmentation_rate,
        "false_linkage_rate": tracking.false_linkage_rate,
        "avg_trajectory_completeness": tracking.avg_trajectory_completeness,
        "assoc_precision": tracking.assoc_precision,
        "assoc_recall": tracking.assoc_recall,
        "assoc_f1_score": tracking.assoc_f1_score,
    }

    log_path = output_dir / EVALUATION_METRICS_CSV
    header_needed = not log_path.exists()
    pd.DataFrame([metrics]).to_csv(
        log_path, mode="w" if header_needed else "a", header=header_needed, index=False
    )

    log.info("\n--- Evaluation Results ---")
    log.info("Detection Precision (Pre-Link): %.4f", detection.precision)
    log.info("Detection Recall (Pre-Link):    %.4f", detection.recall)
    log.info("Detection F1-Score (Pre-Link):  %.4f", detection.f1_score)
    log.info("-" * 26)
    log.info("Fragmentation Rate (Post-Link): %.4f", tracking.fragmentation_rate)
    log.info("False Linkage Rate (Post-Link): %.4f", tracking.false_linkage_rate)
    log.info("Avg Completeness (Post-Link):   %.4f", tracking.avg_trajectory_completeness)
    log.info("-" * 26)
    log.info("Assoc Precision (Post-Link):    %.4f", tracking.assoc_precision)
    log.info("Assoc Recall (Post-Link):       %.4f", tracking.assoc_recall)
    log.info("Assoc F1-Score (Post-Link):     %.4f", tracking.assoc_f1_score)
    log.info("Metrics saved to: %s", log_path)

    return metrics

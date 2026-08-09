"""Pipeline orchestration.

One code path (:meth:`Pipeline.run_video`) serves both batch and single-video
mode; the two entry points differ only in how they decide which folders to feed
it. Networks are loaded once per :class:`Pipeline` and reused across videos.

Stages::

    frames -> [U-Net] masks -> [measure + filter] detections
           -> [nearest-neighbour] tracks -> [RF judge] trajectories
           -> [MSD / plots / evaluation]
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from . import artifacts
from .analysis.evaluation import evaluate_tracking_performance
from .analysis.msd import calculate_msd
from .config import Config
from .detection import UNetDetector
from .embedding import SiameseEmbedder
from .features import extract_frame_particles
from .io.gui import select_video_file
from .io.images import (
    list_frames,
    load_frame_and_mask,
    mask_name_for,
    parse_frame_number,
    read_frame_shape,
)
from .io.paths import RunPaths, find_video_dirs, mask_cache_is_usable, setup_directories
from .io.seq import seq_to_tif
from .linking import apply_judge, drop_short_trajectories, link_nearest_neighbour
from .utils.dataframe import ensure_frame_is_column
from .utils.logging import get_logger
from .viz import (
    plot_detected_photons_over_time,
    plot_individual_msd_curves,
    plot_trajectories,
)

log = get_logger(__name__)


class Pipeline:
    """Runs LinkBLiNK over one or many videos."""

    def __init__(self, config: Config):
        self.config = config
        self.detector = UNetDetector(config.unet)
        self.embedder = SiameseEmbedder(config.siamese)
        self._embedder_ready = False

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_batch(self) -> None:
        """Process every video subfolder under ``paths.input_dir``."""
        log.info("\n=== Starting SMLM Particle Tracking Pipeline (Batch Mode) ===")
        setup_directories(self.config.paths)

        video_dirs = find_video_dirs(self.config.paths.input_dir)
        if not video_dirs:
            log.error(
                "ERROR: No video subdirectories found in %s. Aborting.",
                self.config.paths.input_dir,
            )
            return

        log.info(
            "\nFound %d video folders to process in %s.",
            len(video_dirs), self.config.paths.input_dir,
        )

        for i, video_dir in enumerate(video_dirs, start=1):
            log.info("\n--- Processing Video %d/%d: '%s' ---", i, len(video_dirs), video_dir.name)
            paths = RunPaths.for_video(self.config.paths, video_dir.name, video_dir)
            self.run_video(paths)
            log.info("\n=== Finished processing '%s' ===", video_dir.name)

        log.info("\n=== SMLM Particle Tracking Pipeline (Batch Mode) Complete ===")

    def run_single(self, video_path: Path | None = None) -> None:
        """Process one video, asking for it via GUI when not supplied."""
        log.info("\n=== Starting SMLM Particle Tracking Pipeline (Single File Mode) ===")
        setup_directories(self.config.paths)

        if video_path is None:
            video_path = select_video_file(self.config.paths.input_dir)
        if video_path is None:
            log.warning("No file selected. Pipeline aborted.")
            return

        paths = self._prepare_single_video(video_path)
        if paths is None:
            return

        if self.run_video(paths):
            log.info("\n=== SMLM Particle Tracking Pipeline Finished Successfully ===")
            log.info("Results for '%s' are located in: %s", paths.video_name, paths.results_dir)

    def _prepare_single_video(self, video_path: Path) -> RunPaths | None:
        """Resolve one selected file into a frames directory, converting if needed."""
        # A file inside a folder of frames: treat the whole folder as the video.
        if video_path.is_dir():
            return RunPaths.for_video(self.config.paths, video_path.name, video_path)

        suffix = video_path.suffix.lower()
        if suffix in (".tif", ".tiff") and self._looks_like_frame_folder(video_path.parent):
            folder = video_path.parent
            return RunPaths.for_video(self.config.paths, folder.name, folder)

        video_name = video_path.stem
        frames_dir = self.config.paths.temp_frames_dir / video_name
        frames_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".seq":
            converted = seq_to_tif(
                video_path,
                frames_dir,
                self.config.matlab,
                self.config.paths.matlab_scripts_dir,
            )
            if not converted:
                log.error("Pipeline aborted due to MATLAB preprocessing failure.")
                return None
        elif suffix in (".tif", ".tiff"):
            import shutil

            shutil.copy(video_path, frames_dir / video_path.name)
        else:
            log.error("Unsupported file type '%s'. Pipeline aborted.", suffix)
            return None

        return RunPaths.for_video(
            self.config.paths, video_name, frames_dir, frames_are_temporary=True
        )

    @staticmethod
    def _looks_like_frame_folder(folder: Path) -> bool:
        """Heuristic: more than one TIFF alongside it means it is a frame sequence."""
        tiffs = [p for p in folder.glob("*.tif")] + [p for p in folder.glob("*.tiff")]
        return len(tiffs) > 1

    # ------------------------------------------------------------------
    # The single shared code path
    # ------------------------------------------------------------------

    def run_video(self, paths: RunPaths) -> bool:
        """Run every stage for one video. Returns False if a stage bailed out."""
        use_cached_masks = self.config.mask_cache.enabled and mask_cache_is_usable(
            paths.frames_dir, paths.unet_masks_dir, self.config.mask_cache
        )

        if self.config.run.clean_previous_run_data:
            paths.clean(preserve_unet_masks=use_cached_masks)
        paths.create_output_dirs()

        log.info("Processing video frames from: %s", paths.frames_dir)
        log.info("U-Net masks in:   %s", paths.unet_masks_dir)
        log.info("Filtered masks in: %s", paths.filtered_masks_dir)
        log.info("Final results in:  %s", paths.results_dir)
        log.info(
            "Fixed Normalization Stats: Mean=%.6f, Std=%.6f",
            self.config.unet.norm_mean, self.config.unet.norm_std,
        )

        if use_cached_masks:
            log.info(
                "--- Stage 2: Skipping U-Net prediction for '%s' (using cached masks) ---",
                paths.video_name,
            )
        elif not self.detector.predict_directory(paths.frames_dir, paths.unet_masks_dir):
            log.error("Skipping rest of pipeline for '%s': U-Net prediction failed.", paths.video_name)
            return False

        detections = self._measure_frames(paths)
        if detections.empty:
            log.warning(
                "\n--- No particles were detected and kept across all frames "
                "based on current thresholds. ---"
            )
            log.warning("Please review image data, U-Net performance, and filtering thresholds.")
            return False

        trajectories = self._link_and_judge(detections, paths)
        self._analyse(trajectories, paths)
        self._evaluate(trajectories, paths)
        return True

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _measure_frames(self, paths: RunPaths) -> pd.DataFrame:
        """Stage 3 — measure and filter every particle in every frame."""
        log.info(
            "\n--- Stage 3: Filtering Defocused Particles, Extracting Properties & Tracking ---"
        )

        embedder = self._get_embedder()

        frame_paths = list_frames(paths.frames_dir)
        if not frame_paths:
            log.warning("WARNING: No original image files found in %s.", paths.frames_dir)
            return pd.DataFrame()

        records: list[dict] = []
        for frame_path in tqdm(frame_paths, desc="Filtering & Extracting Properties"):
            mask_path = paths.unet_masks_dir / mask_name_for(frame_path)

            try:
                frame, mask = load_frame_and_mask(frame_path, mask_path)
            except FileNotFoundError as exc:
                log.warning("Error: %s. Skipping frame %s.", exc, frame_path.stem)
                continue

            if mask.sum() == 0:
                continue

            frame_number = parse_frame_number(frame_path)
            if frame_number == 0 and not any(c.isdigit() for c in frame_path.stem.split("_")[-1]):
                log.warning(
                    "Warning: Could not extract numeric frame ID from '%s'. Using 0.",
                    frame_path.stem,
                )

            try:
                filtered_mask, frame_records = extract_frame_particles(
                    frame,
                    mask,
                    frame_number,
                    self.config.filtering,
                    self.config.camera,
                    embedder,
                )
            except Exception as exc:
                log.warning("An unexpected error occurred while processing %s: %s", frame_path.name, exc)
                continue

            out_path = paths.filtered_masks_dir / f"{frame_path.stem}{artifacts.FILTERED_MASK_SUFFIX}"
            cv2.imwrite(str(out_path), filtered_mask)

            records.extend(record.to_dict() for record in frame_records)

        if not records:
            return pd.DataFrame()

        detections = pd.DataFrame(records)
        csv_path = paths.results_dir / artifacts.FILTERED_PARTICLES_CSV
        detections.to_csv(str(csv_path), index=False)

        log.info("\n--- Initial particle data (pre-tracking) saved to %s ---", csv_path)
        log.info("Total detections before tracking: %d", len(detections))
        log.info(
            "X coordinates range: min=%.2f, max=%.2f",
            detections["x"].min(), detections["x"].max(),
        )
        log.info(
            "Y coordinates range: min=%.2f, max=%.2f",
            detections["y"].min(), detections["y"].max(),
        )
        return detections

    def _link_and_judge(self, detections: pd.DataFrame, paths: RunPaths) -> pd.DataFrame:
        """Stage 4 — link detections into trajectories, then repair blinking gaps."""
        log.info("\n--- Starting nearest-neighbor particle linking ---")

        detections = detections.copy()
        detections["frame"] = detections["frame"].astype(int)

        linked = link_nearest_neighbour(detections, self.config.linking.search_range_px)
        linked = linked.reset_index(drop=True)

        log.info("Custom nearest-neighbor linking complete.")
        log.info("Total raw trajectories found: %d", linked["particle"].nunique())

        linked = drop_short_trajectories(linked, self.config.linking.min_trajectory_length)
        linked = ensure_frame_is_column(linked)
        log.info(
            "Total trajectories after filtering (min length %d frames): %d",
            self.config.linking.min_trajectory_length, linked["particle"].nunique(),
        )

        raw_path = paths.results_dir / artifacts.LINKED_RAW_CSV
        linked.to_csv(str(raw_path), index=False)
        log.info("Raw linked particle data (pre-judge) saved to %s", raw_path)

        if self.config.judge.enabled:
            log.info("\n--- Applying RF Judge to correct re-identifications ---")
            judged, merge_log, merge_count = apply_judge(linked, self.config.judge)
            log.info("RF Judge finished: merges performed = %d", merge_count)
        else:
            log.info("RF Judge disabled; skipping ID corrections.")
            judged, merge_log, _ = apply_judge(linked, self.config.judge)

        judged_path = paths.results_dir / artifacts.LINKED_JUDGED_CSV
        judged.to_csv(str(judged_path), index=False)
        log.info("Judged particle data saved to %s", judged_path)

        if not merge_log.empty:
            merge_log_path = paths.results_dir / artifacts.JUDGE_MERGE_LOG_CSV
            merge_log.to_csv(str(merge_log_path), index=False)
            log.info("Judge merge log saved to %s", merge_log_path)

        return judged

    def _analyse(self, trajectories: pd.DataFrame, paths: RunPaths) -> None:
        """Stage 5 — MSD, photon decay, and the trajectory overview figure."""
        if trajectories.empty:
            log.info("No linked particles found after judge; skipping MSD and plotting.")
        else:
            msd_df = calculate_msd(
                trajectories,
                self.config.camera.pixelsize_nm,
                self.config.camera.frame_rate_hz,
                self.config.msd.max_lag_frames,
            )
            if not msd_df.empty:
                msd_path = paths.results_dir / artifacts.MSD_RESULTS_CSV
                msd_df.to_csv(str(msd_path), index=False)
                log.info("MSD results saved to %s", msd_path)

                log.info("\n--- Generating MSD plot ---")
                plot_individual_msd_curves(msd_df, paths.results_dir, paths.video_name)

                log.info("\n--- Generating Detected Photons Over Time plot ---")
                plot_detected_photons_over_time(
                    trajectories,
                    paths.results_dir,
                    paths.video_name,
                    self.config.camera.frame_rate_hz,
                    self.config.viz.photon_plot_smoothing_window,
                )

        log.info("\n--- Generating a simple trajectory plot ---")
        frame_paths = list_frames(paths.frames_dir)
        height, width = read_frame_shape(frame_paths[0]) if frame_paths else (100, 100)
        try:
            plot_trajectories(
                trajectories, paths.results_dir, width, height, self.config.viz.show_trajectory_plot
            )
        except Exception as exc:
            log.warning("Error generating trajectory plot with legend: %s", exc)

    def _evaluate(self, trajectories: pd.DataFrame, paths: RunPaths) -> None:
        """Optional benchmarking against a ground-truth CSV."""
        if not self.config.run.evaluate_tracking_performance:
            return

        gt_path = (
            self.config.paths.ground_truth_dir
            / f"{paths.video_name}{self.config.run.ground_truth_filename_pattern}"
        )
        if not gt_path.exists():
            log.warning("Ground truth file not found at '%s'. Skipping evaluation.", gt_path)
            return

        try:
            ground_truth = pd.read_csv(gt_path)
        except Exception as exc:
            log.error("ERROR: Failed to load ground truth file '%s': %s", gt_path, exc)
            return

        evaluate_tracking_performance(
            trajectories,
            ground_truth,
            paths.video_name,
            paths.results_dir,
            self.config.run.evaluation_distance_threshold_px,
        )

    def _get_embedder(self) -> SiameseEmbedder | None:
        """Load the Siamese network on first use; None if it is unavailable."""
        if self._embedder_ready:
            return self.embedder
        try:
            log.info("\n--- Stage 3.1: Loading Siamese Network for Linking ---")
            self.embedder.load()
            self._embedder_ready = True
            return self.embedder
        except FileNotFoundError as exc:
            log.warning("Skipping Siamese-enhanced linking: %s", exc)
            return None

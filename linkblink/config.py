"""Typed configuration for the LinkBLiNK pipeline.

Every stage takes the dataclass slice it needs, never the whole config, so a
stage can be reused or unit-tested without constructing the rest of the tree.

Defaults here reproduce the values used for the published results. A YAML file
overrides any subset of them::

    cfg = Config.load()                        # defaults only
    cfg = Config.load("configs/default.yaml")  # defaults + overrides

Relative paths in YAML resolve against the project root (the directory holding
``run.py``), so a config file stays portable between machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    """Where the pipeline reads from and writes to."""

    input_dir: Path = PROJECT_ROOT / "data" / "input"
    ground_truth_dir: Path = PROJECT_ROOT / "data" / "ground_truth"
    temp_frames_dir: Path = PROJECT_ROOT / "outputs" / "temp_frames"
    unet_masks_dir: Path = PROJECT_ROOT / "outputs" / "unet_masks"
    filtered_masks_dir: Path = PROJECT_ROOT / "outputs" / "filtered_masks"
    results_dir: Path = PROJECT_ROOT / "outputs" / "results"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    matlab_scripts_dir: Path = PROJECT_ROOT / "matlab_scripts"

    def all_dirs(self) -> list[Path]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class RunConfig:
    """Top-level behaviour of a pipeline invocation."""

    batch_mode: bool = True
    """True: process every video folder under ``input_dir``. False: pick one via GUI."""

    clean_previous_run_data: bool = False
    """Delete this video's temp/mask/result folders before running."""

    evaluate_tracking_performance: bool = False
    ground_truth_filename_pattern: str = "_ground_truth.csv"
    evaluation_distance_threshold_px: float = 3.0


@dataclass
class MaskCacheConfig:
    """Reuse of previously written U-Net masks (skips the detection stage)."""

    enabled: bool = False
    glob: str = "*_predict_mask.png"
    require_full_match: bool = True
    """Only accept the cache if every input frame has a matching mask."""


@dataclass
class MatlabConfig:
    """Only needed to convert ``.seq`` input into TIFF frames."""

    exe_path: Path = Path(r"C:\Program Files\MATLAB\R2021b\bin\matlab.exe")
    seq_to_tif_script: str = "seq_to_tif.m"
    timeout_s: int = 300


@dataclass
class UNetConfig:
    """Stage 2 — particle detection."""

    model_path: Path = PROJECT_ROOT / "checkpoints" / "R2G3B3.pth"
    threshold: float = 0.5
    """Sigmoid probability above which a pixel is called particle."""

    norm_mean: float = 0.021386
    norm_std: float = 0.0230073
    """Fixed normalisation stats — must match the values used at training time."""

    save_debug_images: bool = False


@dataclass
class FilterConfig:
    """Stage 3 — defocusing rejection.

    With ``disabled=True`` every connected component is kept and the thresholds
    below are ignored (they are still evaluated and recorded as features).
    """

    disabled: bool = True
    min_particle_area: int = 10
    max_particle_area: int = 1000
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 2.5
    extent_min: float = 0.6
    max_eccentricity: float = 0.9
    laplacian_var_threshold: float = 1.0
    max_gaussian_residual_sum: float = 100000.0
    max_gaussian_sigma_aspect_ratio: float = 2.5


@dataclass
class SiameseConfig:
    """Appearance embeddings used by the re-identification judge."""

    model_path: Path = PROJECT_ROOT / "checkpoints" / "S1.pth"
    patch_size: int = 32


@dataclass
class LinkingConfig:
    """Frame-to-frame nearest-neighbour linking."""

    search_range_px: float = 5.0
    """Maximum centroid displacement between consecutive frames."""

    min_trajectory_length: int = 3
    """Trajectories shorter than this are discarded as stubs."""


@dataclass
class JudgeConfig:
    """Random-forest re-identification across blinking gaps."""

    enabled: bool = True
    model_path: Path = PROJECT_ROOT / "checkpoints" / "F5.pkl"
    max_time_gap_frames: int = 120
    max_start_distance_px: float = 30.0
    min_confidence: float = 0.50


@dataclass
class CameraConfig:
    """Detector and optics constants used to convert counts into photons."""

    pixelsize_nm: float = 35.9
    quantum_efficiency: float = 0.71
    collection_efficiency: float = 0.015
    gain_electrons_per_count: float = 0.071
    frame_rate_hz: float = 100.0


@dataclass
class MSDConfig:
    max_lag_frames: int = 30
    """0 uses the full trajectory length."""


@dataclass
class VizConfig:
    show_trajectory_plot: bool = False
    """Block on an interactive window after saving the trajectory figure."""

    photon_plot_smoothing_window: int = 5


@dataclass
class Config:
    """Root config object handed to :class:`linkblink.pipeline.Pipeline`."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    mask_cache: MaskCacheConfig = field(default_factory=MaskCacheConfig)
    matlab: MatlabConfig = field(default_factory=MatlabConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    siamese: SiameseConfig = field(default_factory=SiameseConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    msd: MSDConfig = field(default_factory=MSDConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    @classmethod
    def load(cls, yaml_path: str | Path | None = None, project_root: Path = PROJECT_ROOT) -> "Config":
        """Build a config from defaults, optionally overlaid with a YAML file."""
        cfg = cls()
        if yaml_path is not None:
            import yaml  # imported lazily so the package works without PyYAML

            with open(yaml_path, "r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
            _apply_overrides(cfg, overrides)
        _resolve_paths(cfg, project_root)
        return cfg


def _apply_overrides(node: Any, overrides: Mapping[str, Any], trail: str = "") -> None:
    """Recursively write a nested mapping onto a dataclass tree, rejecting typos."""
    valid = {f.name: f for f in fields(node)}
    for key, value in overrides.items():
        if key not in valid:
            where = f"{trail}{key}"
            raise ValueError(
                f"Unknown config key '{where}'. Valid keys here: {sorted(valid)}"
            )
        current = getattr(node, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_overrides(current, value, trail=f"{trail}{key}.")
        elif isinstance(current, Path) or valid[key].type in ("Path", Path):
            setattr(node, key, Path(value))
        else:
            setattr(node, key, value)


def _resolve_paths(node: Any, project_root: Path) -> None:
    """Make every relative Path in the tree absolute against the project root."""
    for f in fields(node):
        value = getattr(node, f.name)
        if is_dataclass(value):
            _resolve_paths(value, project_root)
        elif isinstance(value, Path) and not value.is_absolute():
            setattr(node, f.name, (project_root / value).resolve())

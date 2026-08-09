"""Command-line interface.

    python run.py                              # batch mode, default config
    python run.py --config configs/default.yaml
    python run.py --single                     # pick one video via GUI
    python run.py --single --input path/to.tif
    python run.py --input-dir data/other --no-judge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .pipeline import Pipeline
from .utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linkblink",
        # Plain ASCII: this reaches the console, and a cp1252 Windows terminal
        # raises UnicodeEncodeError on anything outside it.
        description="LinkBLiNK - SMLM particle detection, tracking and re-identification.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config overlaying the built-in defaults.",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Process one video instead of every folder in the input directory.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Video file or frame folder for --single. Opens a file picker if omitted.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Override the batch input directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the results directory.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete this video's previous outputs before running.",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip random-forest re-identification (raw linking only).",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Score results against ground_truth/ after each video.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Let explicit flags win over whatever the YAML said."""
    if args.input_dir is not None:
        config.paths.input_dir = args.input_dir.resolve()
    if args.output_dir is not None:
        config.paths.results_dir = args.output_dir.resolve()
    if args.clean:
        config.run.clean_previous_run_data = True
    if args.no_judge:
        config.judge.enabled = False
    if args.evaluate:
        config.run.evaluate_tracking_performance = True
    if args.single:
        config.run.batch_mode = False
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    config = Config.load(args.config, project_root=PROJECT_ROOT)
    config = apply_cli_overrides(config, args)

    pipeline = Pipeline(config)

    if config.run.batch_mode and args.input is None:
        pipeline.run_batch()
    else:
        pipeline.run_single(args.input.resolve() if args.input else None)

    return 0

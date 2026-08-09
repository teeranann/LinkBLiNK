"""Optional ``.seq`` support: converts a Photron/StreamPix sequence to TIFF frames.

This shells out to MATLAB because the SEQ reader (``matlab_scripts/readSEQ.m``)
is MATLAB code. Everything downstream works on TIFF, so this is the only place
in the pipeline that needs MATLAB installed — TIFF input skips it entirely.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import MatlabConfig
from ..utils.logging import get_logger

log = get_logger(__name__)


def seq_to_tif(seq_path: Path, output_dir: Path, matlab: MatlabConfig, scripts_dir: Path) -> bool:
    """Convert ``seq_path`` into numbered TIFFs under ``output_dir``.

    Returns True on success. Failures are logged and reported as False rather
    than raised, so batch runs can skip one bad file and continue.
    """
    log.info("\n--- Stage 1: Preprocessing '%s' with MATLAB ---", seq_path.name)

    script_path = scripts_dir / matlab.seq_to_tif_script
    if not script_path.exists():
        log.error("ERROR: MATLAB script '%s' not found.", script_path)
        log.error("Please ensure '%s' is in the 'matlab_scripts' folder.", matlab.seq_to_tif_script)
        return False

    command = [
        str(matlab.exe_path),
        "-batch",
        (
            f"addpath('{scripts_dir.as_posix()}'); "
            f"{Path(matlab.seq_to_tif_script).stem}"
            f"('{seq_path.as_posix()}', '{output_dir.as_posix()}')"
        ),
    ]

    try:
        log.info("Executing MATLAB command: %s", " ".join(command))
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=matlab.timeout_s
        )
        log.info("MATLAB Output:\n%s", result.stdout)
        if result.stderr:
            log.warning("MATLAB Errors (if any):\n%s", result.stderr)
        log.info("MATLAB preprocessing completed successfully.")
        return True

    except subprocess.CalledProcessError as exc:
        log.error("ERROR: MATLAB script failed with error code %s", exc.returncode)
        log.error("STDOUT: %s", exc.stdout)
        log.error("STDERR: %s", exc.stderr)
    except FileNotFoundError:
        log.error("ERROR: MATLAB executable not found at '%s'.", matlab.exe_path)
        log.error("Please verify 'matlab.exe_path' in the configuration.")
    except subprocess.TimeoutExpired:
        log.error("ERROR: MATLAB script timed out after %d seconds.", matlab.timeout_s)

    return False

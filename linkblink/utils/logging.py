"""Central logging setup.

The pipeline used to print directly to stdout. Routing through ``logging``
means the follow-on project can capture, filter, or redirect pipeline output
without touching pipeline code.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, quiet_libraries: bool = True) -> None:
    """Install a stdout handler on the ``linkblink`` logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger("linkblink")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    if quiet_libraries:
        for noisy in ("matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``linkblink`` namespace."""
    if not name.startswith("linkblink"):
        name = f"linkblink.{name}"
    return logging.getLogger(name)

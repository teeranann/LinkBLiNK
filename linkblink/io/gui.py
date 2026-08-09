"""Tkinter file picker for single-video mode.

Kept optional: tkinter is imported inside the function so headless machines can
import the package (and run batch mode) without a display or a Tk build.
"""

from __future__ import annotations

from pathlib import Path


def select_video_file(initial_dir: Path | None = None) -> Path | None:
    """Ask the user for a ``.seq`` or ``.tif`` file. Returns None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        start_dir = initial_dir if initial_dir and initial_dir.exists() else Path.cwd()
        selected = filedialog.askopenfilename(
            title="Select an SMLM Video File (.seq or .tif)",
            initialdir=str(start_dir),
            filetypes=[("Video files", "*.seq *.tif"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None

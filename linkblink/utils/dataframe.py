"""DataFrame hygiene helpers.

Linking and judging both assume ``frame`` and ``particle`` are plain integer
columns. Various pandas operations upstream can leave ``frame`` sitting on the
index instead, or duplicate it, which silently breaks the merge logic. These
helpers normalise that in one place.
"""

from __future__ import annotations

import pandas as pd


def ensure_frame_is_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where ``frame`` exists as a column and never as an index level."""
    out = df.copy()

    index_names: list[str] = []
    if isinstance(out.index, pd.MultiIndex):
        index_names = [n for n in out.index.names if n is not None]
    elif out.index.name is not None:
        index_names = [out.index.name]

    if "frame" in index_names:
        # Already a column too: drop the index. Otherwise promote it to a column.
        out = out.reset_index(drop=True) if "frame" in out.columns else out.reset_index()
    else:
        out = out.reset_index(drop=True)

    return out


def normalise_track_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fully normalise a linked-trajectory frame before the judge runs.

    Promotes ``frame`` off the index, flattens MultiIndex columns, drops
    duplicate ``frame`` columns, and coerces ``frame``/``particle`` to int.
    """
    out = df.copy()

    # Push a 'frame' index level into the columns without duplicating it.
    if isinstance(out.index, pd.MultiIndex) and ("frame" in out.index.names):
        out = out.reset_index(drop=True) if "frame" in out.columns else out.reset_index()
    elif getattr(out.index, "name", None) == "frame":
        out = out.reset_index(drop=True) if "frame" in out.columns else out.reset_index()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(map(str, c)).strip("_") for c in out.columns]

    if (out.columns == "frame").sum() > 1:
        # Select by position, not label: `.loc[:, [...]]` with duplicate labels
        # would re-select every 'frame' column and defeat the deduplication.
        seen_frame = False
        keep_positions: list[int] = []
        for position, col in enumerate(out.columns):
            if col == "frame":
                if seen_frame:
                    continue
                seen_frame = True
            keep_positions.append(position)
        out = out.iloc[:, keep_positions]

    for col in ("frame", "particle"):
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .astype("Int64")
                .astype(int, errors="ignore")
            )

    return out

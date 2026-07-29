"""Reusable time-series statistics for simulation observables.

The functions in this module deliberately separate a time window from the
number of independent samples.  Block means may estimate time-series
uncertainty, but they are not independent network realizations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xvg(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read numeric XVG data and return ``(legends, array)``.

    Comment/directive lines are ignored except for ``legend`` labels.  The
    first numeric column is normally time, but this function does not assign
    semantic meaning to columns.
    """
    legends: list[str] = []
    rows: list[list[float]] = []
    width: int | None = None
    for line_no, raw in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            if " legend " in line and '"' in line:
                legends.append(line.split('"', 1)[1].rsplit('"', 1)[0])
            continue
        if line.startswith("#"):
            continue
        try:
            row = [float(token) for token in line.split()]
        except ValueError as exc:
            raise ValueError(f"non-numeric XVG row at {path}:{line_no}") from exc
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"inconsistent XVG column count at {path}:{line_no}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no numeric XVG rows: {path}")
    return legends, np.asarray(rows, dtype=float)


def select_time_window(
    times: np.ndarray,
    values: np.ndarray,
    start: float | None = None,
    end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select an inclusive time window with shape and monotonicity checks."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.shape[0] != t.size:
        raise ValueError("times must be 1-D and match values along axis 0")
    if t.size == 0 or np.any(np.diff(t) < 0):
        raise ValueError("times must be non-empty and monotonically increasing")
    mask = np.ones(t.size, dtype=bool)
    if start is not None:
        mask &= t >= float(start)
    if end is not None:
        mask &= t <= float(end)
    if not np.any(mask):
        raise ValueError("selected time window is empty")
    return t[mask], y[mask]


def block_statistics(values: np.ndarray, n_blocks: int = 5) -> dict[str, object]:
    """Return sample statistics and SEM across contiguous block means."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("values must contain at least two scalar samples")
    if n_blocks < 2 or n_blocks > y.size:
        raise ValueError("n_blocks must be between 2 and the number of samples")
    blocks = [block for block in np.array_split(y, n_blocks) if block.size]
    block_means = np.asarray([np.mean(block) for block in blocks], dtype=float)
    return {
        "n_samples": int(y.size),
        "n_blocks": int(len(blocks)),
        "mean": float(np.mean(y)),
        "sample_std": float(np.std(y, ddof=1)),
        "block_means": block_means.tolist(),
        "block_sem": float(np.std(block_means, ddof=1) / np.sqrt(len(blocks))),
    }


def linear_drift(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit a linear drift and report total/relative change over the window."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        raise ValueError("times and values must be matching 1-D arrays")
    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError("time window duration must be positive")
    slope, intercept = np.polyfit(t, y, 1)
    change = float(slope * duration)
    mean = float(np.mean(y))
    return {
        "slope_per_time": float(slope),
        "intercept": float(intercept),
        "window_duration": duration,
        "fitted_change": change,
        "relative_change": float(change / mean) if mean != 0 else float("nan"),
    }

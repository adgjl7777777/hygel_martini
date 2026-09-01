"""Reusable time-series statistics for simulation observables.

Owns pure-NumPy primitives — XVG reading (:func:`read_xvg`), inclusive
time-window selection, contiguous block statistics, and linear drift —
used by the mechanics window analyses and the equilibration checks.

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

    Args:
        path: Path to the .xvg file.

    Returns:
        Tuple of legend labels (in declaration order) and a 2-D float
        array with one row per data line.

    Raises:
        ValueError: On a non-numeric token, an inconsistent column
            count (both reported with file:line), or a file with no
            numeric rows.
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
    """Select an inclusive time window with shape and monotonicity checks.

    Args:
        times: 1-D, non-empty, monotonically non-decreasing time axis.
        values: Samples aligned with ``times`` along axis 0 (extra
            trailing axes are allowed).
        start: Inclusive lower time bound; ``None`` means unbounded.
        end: Inclusive upper time bound; ``None`` means unbounded.

    Returns:
        ``(times_in_window, values_in_window)``.

    Raises:
        ValueError: On shape mismatch, a non-monotonic time axis, or an
            empty selection (an empty window must fail loudly rather
            than silently produce statistics over nothing).
    """
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
    """Return sample statistics and SEM across contiguous block means.

    The series is split into ``n_blocks`` contiguous blocks; the SEM of
    the block means estimates time-series uncertainty.  Blocks are NOT
    independent network replicates — do not report them as such.

    Args:
        values: 1-D series with at least two samples.
        n_blocks: Number of contiguous blocks (2 <= n_blocks <= len).

    Returns:
        Dict with ``n_samples``, ``n_blocks``, ``mean``, ``sample_std``
        (ddof=1), ``block_means`` and ``block_sem``.

    Raises:
        ValueError: If the series is too short or ``n_blocks`` is out
            of range.
    """
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
    """Fit a linear drift and report total/relative change over the window.

    Args:
        times: 1-D time axis with positive total duration (same time
            unit as the source data, typically ps).
        values: 1-D series aligned with ``times``.

    Returns:
        Dict with ``slope_per_time``, ``intercept``,
        ``window_duration``, ``fitted_change`` (slope * duration) and
        ``relative_change`` (fitted change / mean; NaN when the mean is
        exactly zero).

    Raises:
        ValueError: On shape mismatch, fewer than two samples, or a
            non-positive window duration.
    """
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

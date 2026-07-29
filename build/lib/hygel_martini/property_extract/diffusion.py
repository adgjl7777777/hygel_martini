"""PBC-safe translational MSD primitives for within-model mobility analysis."""

from __future__ import annotations

import numpy as np

from .geometry import minimum_image_displacement, orthorhombic_box_lengths


def unwrap_trajectory(positions: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Cumulatively unwrap particle positions for orthorhombic boxes.

    Positions have shape ``(n_frames, n_particles, 3)``. Boxes may be one
    length vector or one vector per frame. Coordinates and box lengths must use
    the same units.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3 or pos.shape[0] < 2:
        raise ValueError("positions must have shape (n_frames,n_particles,3), n_frames >= 2")
    raw_boxes = np.asarray(boxes, dtype=float)
    if raw_boxes.shape == (3,):
        lengths = np.repeat(orthorhombic_box_lengths(raw_boxes)[None, :], pos.shape[0], axis=0)
    elif raw_boxes.shape == (pos.shape[0], 3):
        lengths = np.asarray([orthorhombic_box_lengths(box) for box in raw_boxes])
    else:
        raise ValueError("boxes must have shape (3,) or (n_frames,3)")
    out = np.empty_like(pos)
    out[0] = pos[0]
    for frame in range(1, pos.shape[0]):
        # Use the current box for the stored wrapped displacement. This is an
        # analysis convention, not a correction for barostat affine scaling.
        delta = minimum_image_displacement(pos[frame] - pos[frame - 1], lengths[frame])
        out[frame] = out[frame - 1] + delta
    return out


def multi_origin_msd(
    unwrapped_positions: np.ndarray,
    frame_dt: float,
    max_lag_frames: int | None = None,
    origin_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lag times, 3-D MSD, and number of time origins per lag."""
    pos = np.asarray(unwrapped_positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3 or pos.shape[0] < 2:
        raise ValueError("unwrapped_positions must have shape (frames,particles,3)")
    if frame_dt <= 0 or origin_stride < 1:
        raise ValueError("frame_dt and origin_stride must be positive")
    n_frames = pos.shape[0]
    max_lag = n_frames - 1 if max_lag_frames is None else int(max_lag_frames)
    if max_lag < 1 or max_lag >= n_frames:
        raise ValueError("max_lag_frames must be in [1, n_frames-1]")
    msd = np.empty(max_lag + 1, dtype=float)
    counts = np.empty(max_lag + 1, dtype=int)
    msd[0] = 0.0
    counts[0] = n_frames
    for lag in range(1, max_lag + 1):
        origins = np.arange(0, n_frames - lag, origin_stride)
        displacement = pos[origins + lag] - pos[origins]
        msd[lag] = float(np.mean(np.sum(displacement**2, axis=2)))
        counts[lag] = int(len(origins))
    return np.arange(max_lag + 1, dtype=float) * frame_dt, msd, counts


def fit_diffusion_coefficient(
    lag_times: np.ndarray,
    msd: np.ndarray,
    fit_start: float,
    fit_end: float,
    dimensions: int = 3,
) -> dict[str, float]:
    """Fit ``MSD = 2*d*D*t + intercept`` over an explicit lag window."""
    t = np.asarray(lag_times, dtype=float)
    y = np.asarray(msd, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("lag_times and msd must be matching 1-D arrays")
    if dimensions < 1 or fit_end <= fit_start:
        raise ValueError("invalid dimensions or fit window")
    mask = (t >= fit_start) & (t <= fit_end) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        raise ValueError("fit window contains fewer than two points")
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    predicted = slope * t[mask] + intercept
    residual = float(np.sum((y[mask] - predicted) ** 2))
    total = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    coefficient = float(slope / (2 * dimensions))
    return {
        "diffusion_coefficient_coordinate2_per_time": coefficient,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "fit_start": float(fit_start),
        "fit_end": float(fit_end),
        "dimensions": int(dimensions),
        "n_fit_points": int(np.count_nonzero(mask)),
    }

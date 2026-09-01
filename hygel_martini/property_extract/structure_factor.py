"""Static structure-factor primitives for periodic coarse-grained systems.

The functions in this module calculate observables only.  They do not convert
``2*pi/q`` into a pore or mesh size and do not decide whether a simulation is
commensurate with a scattering experiment.

Owns three pure-numpy layers: cloud-in-cell density deposition
(:func:`cic_density_grid`), the FFT-based ``S(q)`` over all
reciprocal-lattice modes with optional CIC-window deconvolution
(:func:`fft_structure_factor`) plus its radial binning
(:func:`radial_bin_structure_factor`), and an exact direct-summation
``S(q)`` along the box axes (:func:`reciprocal_axis_structure_factor`)
for cross-checking the gridded estimate.  Lengths follow the caller's
coordinate unit, so ``q`` is in the reciprocal of that unit (1/nm for
nm coordinates).  Malformed inputs raise ``ValueError`` — refusal
semantics live in the calling layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import orthorhombic_box_lengths


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    """Coerce to a float ``(n, 3)`` array; reject empty/non-finite input."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
        raise ValueError("positions must be a non-empty array with shape (n,3)")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    return pos


def _validate_grid_shape(grid_shape: int | Sequence[int]) -> tuple[int, int, int]:
    """Expand an int to a cubic shape; require three integers >= 4."""
    if isinstance(grid_shape, int):
        shape = (grid_shape, grid_shape, grid_shape)
    else:
        shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value < 4 for value in shape):
        raise ValueError("grid_shape must contain three integers >= 4")
    return shape


def cic_density_grid(
    positions: np.ndarray,
    box: np.ndarray,
    grid_shape: int | Sequence[int],
) -> np.ndarray:
    """Deposit particles on a periodic grid using cloud-in-cell assignment.

    The returned grid contains particle counts, so its sum equals the number of
    input particles.  Coordinates are wrapped into the orthorhombic box.

    Each particle's weight is split over its 8 surrounding cells with
    trilinear (first-order B-spline) weights, wrapped periodically.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        box: Orthorhombic box accepted by
            :func:`.geometry.orthorhombic_box_lengths`.
        grid_shape: Cells per axis — an int (cubic) or three ints,
            each >= 4.

    Returns:
        Float64 count grid of shape ``grid_shape``.

    Raises:
        ValueError: Invalid positions or grid shape.
        RuntimeError: Deposited weights fail to sum to the particle
            count (internal conservation check).
    """
    pos = _validate_positions(positions)
    lengths = orthorhombic_box_lengths(box)
    shape = _validate_grid_shape(grid_shape)
    shape_array = np.asarray(shape, dtype=np.int64)

    scaled = np.mod(pos / lengths, 1.0) * shape_array
    lower = np.floor(scaled).astype(np.int64)
    fraction = scaled - lower
    density = np.zeros(shape, dtype=np.float64)

    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        ix = (lower[:, 0] + dx) % shape[0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            iy = (lower[:, 1] + dy) % shape[1]
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                iz = (lower[:, 2] + dz) % shape[2]
                np.add.at(density, (ix, iy, iz), wx * wy * wz)

    if not np.isclose(float(density.sum()), float(len(pos)), rtol=0.0, atol=1e-8):
        raise RuntimeError("cloud-in-cell particle-count conservation failed")
    return density


def fft_structure_factor(
    positions: np.ndarray,
    box: np.ndarray,
    grid_shape: int | Sequence[int] = 64,
    *,
    q_max: float | None = None,
    deconvolve_cic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reciprocal-vector magnitudes and static ``S(q)`` values.

    ``S(k) = |sum_j exp(-i k.r_j)|^2 / N`` is evaluated from a periodic
    cloud-in-cell density grid.  The zero mode is removed.  When requested, the
    separable CIC assignment window is deconvolved from the Fourier amplitude.
    Comparisons should still be restricted below the grid Nyquist region and
    repeated at more than one grid resolution.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        box: Orthorhombic box specification.
        grid_shape: Cells per axis (int or three ints, each >= 4).
        q_max: Optional upper cut on |q| (reciprocal length unit);
            must be finite and positive when given.
        deconvolve_cic: Divide out the per-dimension ``sinc**2`` CIC
            window before squaring (default True).

    Returns:
        Tuple ``(q_magnitude, structure)`` — flat arrays over the kept
        modes (zero mode always excluded), unsorted in q.

    Raises:
        ValueError: Invalid positions, grid shape, or ``q_max``.
    """
    pos = _validate_positions(positions)
    lengths = orthorhombic_box_lengths(box)
    shape = _validate_grid_shape(grid_shape)
    density = cic_density_grid(pos, lengths, shape)
    amplitude = np.fft.rfftn(density)

    nx = np.fft.fftfreq(shape[0]) * shape[0]
    ny = np.fft.fftfreq(shape[1]) * shape[1]
    nz = np.fft.rfftfreq(shape[2]) * shape[2]
    qx = 2.0 * np.pi * nx / lengths[0]
    qy = 2.0 * np.pi * ny / lengths[1]
    qz = 2.0 * np.pi * nz / lengths[2]
    q_magnitude = np.sqrt(
        qx[:, None, None] ** 2
        + qy[None, :, None] ** 2
        + qz[None, None, :] ** 2
    )

    if deconvolve_cic:
        # np.sinc(x) = sin(pi*x)/(pi*x).  CIC is a first-order B-spline,
        # whose Fourier-amplitude window is sinc(n/N)^2 per dimension.
        wx = np.sinc(nx / shape[0]) ** 2
        wy = np.sinc(ny / shape[1]) ** 2
        wz = np.sinc(nz / shape[2]) ** 2
        assignment_window = (
            wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        )
        amplitude = amplitude / assignment_window

    structure = np.abs(amplitude) ** 2 / len(pos)
    keep = q_magnitude > 0.0
    if q_max is not None:
        if not np.isfinite(q_max) or q_max <= 0:
            raise ValueError("q_max must be finite and positive")
        keep &= q_magnitude <= float(q_max)
    return q_magnitude[keep], structure[keep]


def radial_bin_structure_factor(
    q_magnitude: np.ndarray,
    structure_factor: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Radially average reciprocal modes into explicit ``q`` bins.

    Args:
        q_magnitude: Mode magnitudes (any shape; flattened).
        structure_factor: Matching per-mode ``S(q)`` values.
        bin_edges: Strictly increasing 1-D bin edges; modes outside
            the outer edges are dropped.

    Returns:
        Dict of per-bin arrays: ``q_lower``/``q_upper``/``q_center``,
        ``mean_structure_factor`` (NaN for empty bins), and
        ``n_modes`` counts.

    Raises:
        ValueError: Size mismatch, empty input, non-increasing edges,
            or non-finite values.
    """
    q = np.asarray(q_magnitude, dtype=float).ravel()
    s = np.asarray(structure_factor, dtype=float).ravel()
    edges = np.asarray(bin_edges, dtype=float)
    if q.shape != s.shape or q.size == 0:
        raise ValueError("q_magnitude and structure_factor must have equal nonzero size")
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing 1-D array")
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(s)):
        raise ValueError("q and structure-factor values must be finite")

    index = np.digitize(q, edges, right=False) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    count = np.bincount(index[valid], minlength=len(edges) - 1).astype(np.int64)
    total = np.bincount(index[valid], weights=s[valid], minlength=len(edges) - 1)
    mean = np.full(len(edges) - 1, np.nan, dtype=float)
    np.divide(total, count, out=mean, where=count > 0)
    return {
        "q_lower": edges[:-1],
        "q_upper": edges[1:],
        "q_center": 0.5 * (edges[:-1] + edges[1:]),
        "mean_structure_factor": mean,
        "n_modes": count,
    }


def reciprocal_axis_structure_factor(
    positions: np.ndarray,
    box: np.ndarray,
    max_mode: int,
) -> dict[str, np.ndarray]:
    """Calculate exact static structure factors along box ``x/y/z`` axes.

    Only reciprocal-lattice vectors ``q = 2*pi*n/L_axis`` are used, which keeps
    the finite periodic cell and directional comparison explicit.

    Direct summation over particles (no grid, no assignment window),
    so this also serves as an exact cross-check of
    :func:`fft_structure_factor` at the shared modes.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        box: Orthorhombic box specification.
        max_mode: Highest integer mode ``n`` per axis (>= 1).

    Returns:
        Dict with ``mode`` (1..max_mode) and, per axis label,
        ``q_<axis>`` (reciprocal length unit) and ``S_<axis>``.

    Raises:
        ValueError: Invalid positions or non-positive/non-integer
            ``max_mode``.
    """
    pos = _validate_positions(positions)
    lengths = orthorhombic_box_lengths(box)
    if int(max_mode) != max_mode or max_mode < 1:
        raise ValueError("max_mode must be a positive integer")
    modes = np.arange(1, int(max_mode) + 1, dtype=np.int64)
    output: dict[str, np.ndarray] = {"mode": modes}
    for axis, label in enumerate(("x", "y", "z")):
        q = 2.0 * np.pi * modes / lengths[axis]
        phase = np.outer(q, pos[:, axis])
        amplitude = np.exp(-1j * phase).sum(axis=1)
        output[f"q_{label}"] = q
        output[f"S_{label}"] = np.abs(amplitude) ** 2 / len(pos)
    return output


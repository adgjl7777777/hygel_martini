"""Resolution-explicit voxel heterogeneity analysis.

Pure-numpy helpers that rasterize particle positions onto a periodic
orthorhombic grid and characterize the resulting count field:
occupancy statistics (:func:`voxel_counts`,
:func:`summarize_voxel_counts`), translation-aware persistence between
two fields (:func:`periodic_field_correlation`), and
amplitude-preserving phase-randomized surrogates
(:func:`phase_randomized_field`) for null-model comparisons.

Everything here is a resolution-dependent composition diagnostic — the
summary says so explicitly and nothing is converted into a physical
pore size.  Coordinates/box lengths share the caller's length unit;
malformed inputs raise ``ValueError`` rather than returning partial
results.
"""

from __future__ import annotations

import numpy as np

from .geometry import orthorhombic_box_lengths, wrap_positions


def voxel_counts(
    positions: np.ndarray,
    box: np.ndarray,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Count positions in a periodic grid and return counts and actual spacing.

    The grid divides each box edge into an integer number of cells no
    smaller than ``target_spacing`` (at least one cell per axis), so
    the realized spacing can exceed the target; positions are wrapped
    into the box before binning.

    Args:
        positions: Particle coordinates, shape ``(n, 3)``.
        box: Orthorhombic box specification accepted by
            :func:`.geometry.orthorhombic_box_lengths`.
        target_spacing: Requested voxel edge length (same length unit
            as the coordinates); must be positive.

    Returns:
        Tuple ``(counts, spacing)`` — integer occupancy grid of shape
        ``(nx, ny, nz)`` and the realized per-axis spacing.

    Raises:
        ValueError: Non-positive ``target_spacing``.
    """
    if target_spacing <= 0:
        raise ValueError("target_spacing must be positive")
    lengths = orthorhombic_box_lengths(box)
    n_cells = np.maximum(np.floor(lengths / float(target_spacing)).astype(int), 1)
    spacing = lengths / n_cells
    wrapped = wrap_positions(positions, lengths)
    indices = np.floor(wrapped / spacing).astype(int)
    indices = np.minimum(indices, n_cells - 1)
    counts = np.zeros(tuple(int(value) for value in n_cells), dtype=int)
    np.add.at(counts, tuple(indices.T), 1)
    return counts, spacing


def summarize_voxel_counts(counts: np.ndarray) -> dict[str, object]:
    """Summarize count heterogeneity without converting it to a pore size.

    Args:
        counts: Voxel occupancy grid (any shape; flattened here).

    Returns:
        Dict of occupancy statistics — voxel count, mean/std,
        coefficient of variation (NaN for zero mean), empty-voxel
        fraction, the 5/25/50/75/95 percentiles, min/max — plus an
        ``interpretation`` string stating this is a
        resolution-dependent diagnostic, not a physical pore size.

    Raises:
        ValueError: Empty ``counts``.
    """
    values = np.asarray(counts, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("counts must be non-empty")
    mean = float(np.mean(values))
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n_voxels": int(values.size),
        "mean_count": mean,
        "std_count": float(np.std(values)),
        "coefficient_of_variation": float(np.std(values) / mean) if mean else float("nan"),
        "empty_fraction": float(np.mean(values == 0)),
        "percentiles_5_25_50_75_95": percentiles.tolist(),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "interpretation": "resolution-dependent composition diagnostic; not a physical pore size",
    }


def periodic_field_correlation(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, object]:
    """Correlate equal-shaped periodic fields before and after translation.

    The zero-shift coefficient measures persistence in the stored coordinate
    frame. The maximum circular coefficient removes a whole-field periodic
    translation, but does not remove rotation, deformation, or topology memory.

    Both fields are mean-centered, then the full circular
    cross-correlation is evaluated via FFT and normalized like a
    Pearson coefficient.

    Args:
        reference: 3-D field (e.g. voxel counts at a reference time).
        current: 3-D field of the same shape at another time.

    Returns:
        Dict with ``zero_shift_correlation``,
        ``translation_aligned_correlation`` (maximum over circular
        shifts), and ``best_periodic_shift_cells`` — the maximizing
        shift as signed cell offsets (wrapped into +-size/2).

    Raises:
        ValueError: Shape mismatch, non-3-D input, empty fields, or a
            field with zero/non-finite variance.
    """
    first = np.asarray(reference, dtype=float)
    second = np.asarray(current, dtype=float)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("reference and current must be equal-shaped 3-D fields")
    if first.size == 0:
        raise ValueError("fields must be non-empty")
    first = first - np.mean(first)
    second = second - np.mean(second)
    norm = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("both fields must have nonzero finite variance")

    zero_shift = float(np.sum(first * second) / norm)
    cross = np.fft.ifftn(
        np.conj(np.fft.fftn(first)) * np.fft.fftn(second)
    ).real
    maximum_index = np.unravel_index(int(np.argmax(cross)), cross.shape)
    maximum = float(cross[maximum_index] / norm)
    signed_shift = tuple(
        int(index if index <= size // 2 else index - size)
        for index, size in zip(maximum_index, cross.shape)
    )
    return {
        "zero_shift_correlation": zero_shift,
        "translation_aligned_correlation": maximum,
        "best_periodic_shift_cells": signed_shift,
    }


def phase_randomized_field(
    field: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize spatial phase while preserving mean and Fourier amplitude.

    A real Gaussian field supplies a Hermitian-symmetric set of random phases,
    so the inverse transform remains real. The result is a spatial surrogate
    with the input field's mean and power spectrum but without its particular
    coordinate-phase arrangement.

    Args:
        field: Non-empty, all-finite 3-D array to build a surrogate of.
        rng: Numpy Generator supplying the random phases (pass a seeded
            one for reproducible surrogates).

    Returns:
        Real 3-D surrogate field with the same shape, mean, and Fourier
        amplitude spectrum as ``field``.

    Raises:
        ValueError: Non-3-D/empty input or non-finite values.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("field must be a non-empty 3-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")

    mean = float(np.mean(values))
    amplitude = np.abs(np.fft.fftn(values - mean))
    noise_spectrum = np.fft.fftn(rng.normal(size=values.shape))
    magnitude = np.abs(noise_spectrum)
    unit_phase = np.divide(
        noise_spectrum,
        magnitude,
        out=np.ones_like(noise_spectrum),
        where=magnitude > 0,
    )
    return np.fft.ifftn(amplitude * unit_phase).real + mean

"""Resolution-explicit voxel heterogeneity analysis."""

from __future__ import annotations

import numpy as np

from .geometry import orthorhombic_box_lengths, wrap_positions


def voxel_counts(
    positions: np.ndarray,
    box: np.ndarray,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Count positions in a periodic grid and return counts and actual spacing."""
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
    """Summarize count heterogeneity without converting it to a pore size."""
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

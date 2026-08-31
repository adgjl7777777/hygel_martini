"""Periodic geometry primitives shared by PEG and Pluronic analyses."""

from __future__ import annotations

import numpy as np

from hygel_martini.core.pbc import minimum_image as core_minimum_image


def orthorhombic_box_lengths(box: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector or diagonal 3x3 box to positive lengths."""
    arr = np.asarray(box, dtype=float)
    if arr.shape == (3,):
        lengths = arr
    elif arr.shape == (3, 3):
        off_diagonal = arr - np.diag(np.diag(arr))
        if not np.allclose(off_diagonal, 0.0, atol=1e-12):
            raise ValueError("triclinic boxes are not supported by this primitive")
        lengths = np.diag(arr)
    else:
        raise ValueError("box must have shape (3,) or diagonal (3,3)")
    if np.any(lengths <= 0):
        raise ValueError("box lengths must be positive")
    return lengths.astype(float, copy=True)


def minimum_image_displacement(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply the orthorhombic minimum-image convention to displacement(s).

    The orthorhombic contract is kept deliberately: trajectory analysis here
    assumes rectangular boxes and validates that. The convention itself lives
    in :mod:`hygel_martini.core.pbc` so there is one implementation to be
    right, and that one also handles triclinic cells for the builder.
    """
    lengths = orthorhombic_box_lengths(box)
    return core_minimum_image(delta, lengths)


def wrap_positions(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap Cartesian positions into ``[0, L)`` for an orthorhombic box."""
    lengths = orthorhombic_box_lengths(box)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (n,3)")
    return np.mod(pos, lengths)


def unwrap_ordered_chain(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Unwrap a bonded/ordered chain using consecutive minimum images."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
        raise ValueError("positions must be a non-empty array with shape (n,3)")
    out = np.empty_like(pos)
    out[0] = pos[0]
    for idx in range(1, len(pos)):
        out[idx] = out[idx - 1] + minimum_image_displacement(pos[idx] - pos[idx - 1], box)
    return out


def gyration_metrics(positions: np.ndarray) -> dict[str, object]:
    """Return Rg, end-to-end distance, eigenvalues, and relative anisotropy."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 2:
        raise ValueError("positions must have shape (n,3) with n >= 2")
    centered = pos - np.mean(pos, axis=0)
    tensor = centered.T @ centered / len(pos)
    eigenvalues = np.linalg.eigvalsh(tensor)
    total = float(np.sum(eigenvalues))
    if total <= 0:
        kappa2 = 0.0
    else:
        mean_eigenvalue = total / 3.0
        kappa2 = float(1.5 * np.sum((eigenvalues - mean_eigenvalue) ** 2) / total**2)
    return {
        "radius_of_gyration": float(np.sqrt(max(total, 0.0))),
        "end_to_end": float(np.linalg.norm(pos[-1] - pos[0])),
        "gyration_eigenvalues": eigenvalues.tolist(),
        "relative_shape_anisotropy": kappa2,
    }


def bond_orientation_metrics(vectors: np.ndarray) -> dict[str, object]:
    """Return second-rank orientational order from PBC-corrected bond vectors.

    The caller is responsible for defining the bond population and applying
    minimum-image corrections.  The returned tensor is
    ``Q = 3/2 <u u> - I/2`` for normalized vectors ``u``.  Its largest
    eigenvalue is the conventional uniaxial nematic order parameter only when
    that interpretation is appropriate for the selected bond population.
    """
    vec = np.asarray(vectors, dtype=float)
    if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) == 0:
        raise ValueError("vectors must be a non-empty array with shape (n,3)")
    lengths = np.linalg.norm(vec, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("vectors must be finite and nonzero")
    unit = vec / lengths[:, None]
    second_moment = unit.T @ unit / len(unit)
    tensor = 1.5 * second_moment - 0.5 * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    principal = eigenvectors[:, -1]
    return {
        "n_vectors": int(len(unit)),
        "second_moment": second_moment.tolist(),
        "orientation_tensor": tensor.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "principal_axis": principal.tolist(),
    }

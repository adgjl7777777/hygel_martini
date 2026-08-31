"""One minimum-image convention for the whole package.

Before this module the same primitive existed seven times: twice in
``hydrogel_builder/core_utils/common/utility.py``, three times inline in
``layout/isotropic_builder.py``, once in ``runtime/dynamic_crosslink.py``, and
once in ``property_extract/geometry.py``.  Six of them applied
``delta -= box * round(delta / box)``, which is the *orthorhombic* convention.
Applied to a triclinic cell it does not find the nearest image at all: on the
GROMACS-legal cell ``[[4,0,0],[0,4,0],[2,2,3]]`` it overstates a separation of
1.04 nm as 2.96 nm.  That distance decides which chain ends a crosslinker
reaches, so the error is not cosmetic.

The worst case was ``dynamic_crosslink.normalize_box_vector``, which accepted a
full 3x3 cell and reduced it with ``np.diag`` --- silently discarding the
off-diagonal terms that make the cell triclinic in the first place.

This module keeps one implementation, correct for both cases, and states its
own domain of validity rather than assuming one.

Conventions
-----------
A cell is a 3x3 array whose **rows** are the cell vectors, matching GROMACS
box order.  A 3-vector is accepted as shorthand for the orthorhombic cell with
those diagonal lengths.  ``None`` means no periodicity.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "normalize_cell",
    "is_orthorhombic",
    "nearest_image_reach",
    "minimum_image",
    "minimum_image_distance",
    "wrap_into_cell",
]

def _shift_grid(reach: int) -> np.ndarray:
    """All integer shifts within ``reach`` cells along each axis."""
    span = range(-reach, reach + 1)
    return np.array(
        [(i, j, k) for i in span for j in span for k in span], dtype=float
    )


_SHIFT_CACHE = {reach: _shift_grid(reach) for reach in (1, 2)}
_MAX_REACH = 4


def normalize_cell(box) -> np.ndarray | None:
    """Coerce a box specification to a 3x3 cell with rows as cell vectors."""
    if box is None:
        return None
    cell = np.asarray(box, dtype=float)
    if cell.shape == (3,):
        if np.any(cell <= 0.0):
            raise ValueError(f"box lengths must be positive, got {cell.tolist()}")
        return np.diag(cell)
    if cell.shape == (3, 3):
        volume = float(abs(np.linalg.det(cell)))
        if volume <= 0.0:
            raise ValueError("cell vectors are degenerate (zero volume)")
        return cell
    raise ValueError(
        f"box must be a 3-vector or a 3x3 matrix of cell vectors, got shape {cell.shape}"
    )


def is_orthorhombic(cell: np.ndarray, tolerance: float = 1e-9) -> bool:
    """True when the cell is diagonal to within ``tolerance``."""
    off_diagonal = cell - np.diag(np.diag(cell))
    return bool(np.all(np.abs(off_diagonal) <= tolerance))


def nearest_image_reach(cell: np.ndarray, max_reach: int = _MAX_REACH) -> int:
    """Smallest shift range whose optimum is strictly interior.

    A nearest-image search over neighbouring cells is only exhaustive if the
    winning shift is not on the boundary of the range searched --- otherwise a
    shift one step further out might be shorter still. Rather than assume a
    particular cell convention, the range is widened until the optimum is
    interior for a set of probe displacements spanning the cell. An earlier
    version instead tested the GROMACS lower-triangular reduction condition,
    which rejected perfectly usable cells such as the FCC primitive basis of
    the diamond seed, whose diagonal contains zeros.
    """
    probes = _shift_grid(1) @ cell * 0.5
    for reach in range(1, int(max_reach) + 1):
        shifts = _SHIFT_CACHE.setdefault(reach, _shift_grid(reach))
        fractional = np.linalg.solve(cell.T, probes.T).T
        fractional -= np.round(fractional)
        candidates = (fractional[:, None, :] + shifts[None, :, :]) @ cell
        best = np.argmin(np.einsum("nsi,nsi->ns", candidates, candidates), axis=1)
        if np.all(np.abs(shifts[best]) < reach):
            return reach
    raise ValueError(
        f"cell {cell.tolist()} is too skewed for a nearest-image search within "
        f"{max_reach} cells; reduce the lattice basis before applying the "
        "minimum-image convention"
    )


def minimum_image(delta, cell) -> np.ndarray:
    """Shortest periodic image of a displacement, or of a stack of them.

    ``delta`` may be a single 3-vector or an ``(n, 3)`` array.  For an
    orthorhombic cell this reduces to the familiar rounding formula; otherwise
    the fractional-wrapped displacement is compared against its 27 neighbouring
    images.
    """
    displacement = np.asarray(delta, dtype=float)
    if displacement.shape[-1] != 3:
        raise ValueError("displacement last dimension must be 3")
    if cell is None:
        return displacement

    matrix = cell if isinstance(cell, np.ndarray) and cell.shape == (3, 3) else normalize_cell(cell)
    if matrix is None:
        return displacement

    if is_orthorhombic(matrix):
        lengths = np.diag(matrix)
        return displacement - lengths * np.round(displacement / lengths)

    reach = nearest_image_reach(matrix)
    shifts = _SHIFT_CACHE.setdefault(reach, _shift_grid(reach))

    single = displacement.ndim == 1
    stack = displacement.reshape(1, 3) if single else displacement.reshape(-1, 3)

    # rows are cell vectors, so fractional coordinates solve f @ cell = delta
    fractional = np.linalg.solve(matrix.T, stack.T).T
    fractional -= np.round(fractional)
    candidates = (fractional[:, None, :] + shifts[None, :, :]) @ matrix
    best = np.argmin(np.einsum("nsi,nsi->ns", candidates, candidates), axis=1)
    result = candidates[np.arange(len(stack)), best]
    return result[0] if single else result.reshape(displacement.shape)


def minimum_image_distance(first, second, cell) -> float:
    """Shortest periodic distance between two positions."""
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return float(np.linalg.norm(minimum_image(delta, cell)))


def wrap_into_cell(positions, cell) -> np.ndarray:
    """Wrap Cartesian positions into the primary cell."""
    coordinates = np.asarray(positions, dtype=float)
    if cell is None:
        return coordinates
    matrix = normalize_cell(cell)
    if is_orthorhombic(matrix):
        return np.mod(coordinates, np.diag(matrix))
    fractional = np.linalg.solve(matrix.T, coordinates.reshape(-1, 3).T).T
    fractional -= np.floor(fractional)
    return (fractional @ matrix).reshape(coordinates.shape)

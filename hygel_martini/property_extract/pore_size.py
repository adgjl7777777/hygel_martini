"""Periodic, definition-explicit grid clearance analysis.

This module intentionally reports a local-clearance/probe-admissible-volume
observable.  It is not a method-identical replacement for PoreBlazer, a
percolating pore-limiting diameter, or an experimental mesh size.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .result import PropertyResult

_GRID_POINT_WARN_THRESHOLD = 5_000_000  # 이 이상이면 메모리 경고
_DEFAULT_CHUNK_SIZE = 250_000


def parse_gro_coords(gro_file, selection_residue=None, selection_residues=None):
    """
    .gro 파일을 파싱해 지정 residue의 좌표와 box 크기를 반환.

    selection_residues : residue 이름 집합 (우선). 예: ["PEO", "HYDROGEL"]
    selection_residue  : 단일 residue 이름 (selection_residues가 None일 때).
    둘 다 None이면 기본값 {"PEO", "HYDROGEL"} 사용.
    exact match만 허용 (substring match 금지).
    """
    if selection_residues is not None:
        allowed = set(selection_residues)
    elif selection_residue is not None:
        allowed = {selection_residue}
    else:
        allowed = {"PEO", "HYDROGEL"}

    coords = []
    box = None
    with open(gro_file, 'r') as f:
        lines = f.readlines()
        try:
            n_atoms = int(lines[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid .gro file format: {gro_file}")

        for i in range(2, 2 + n_atoms):
            line = lines[i]
            resname = line[5:10].strip()
            if resname in allowed:
                try:
                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])
                    coords.append([x, y, z])
                except ValueError:
                    continue

        try:
            box_parts = lines[2 + n_atoms].split()
            if len(box_parts) < 3:
                raise ValueError("box line에 좌표가 3개 미만")
            box = np.array([float(v) for v in box_parts[:3]])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f".gro box line 파싱 실패: {gro_file}\n원인: {e}"
            ) from e

    return np.array(coords), box


def _validate_box(box_size: np.ndarray) -> np.ndarray:
    box = np.asarray(box_size, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("box_size must contain three positive finite lengths")
    return box


def _grid_shape_and_spacing(
    box_size: np.ndarray,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(target_spacing) or target_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite")
    shape = np.maximum(np.floor(box_size / float(target_spacing)).astype(int), 1)
    return shape, box_size / shape


def periodic_clearance_grid(
    obstacle_groups: Sequence[tuple[np.ndarray, float]],
    box_size: np.ndarray,
    grid_spacing: float = 0.2,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest obstacle-surface clearance on a periodic cell-centred grid.

    Each obstacle group is ``(positions_nm, radius_nm)``. Groups make mixed
    Martini radii exact without assuming that the nearest centre also has the
    largest radius. Coordinates are wrapped into an orthorhombic periodic box.

    The returned field is positive in geometric void and negative inside an
    obstacle sphere. ``spacing`` is the actual grid spacing after fitting an
    integer number of cells to each box length.
    """
    box = _validate_box(box_size)
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")

    trees: list[tuple[cKDTree, float]] = []
    for positions, radius in obstacle_groups:
        coords = np.asarray(positions, dtype=float)
        radius_value = float(radius)
        if coords.ndim != 2 or coords.shape[1:] != (3,):
            raise ValueError("obstacle positions must have shape (n, 3)")
        if not np.all(np.isfinite(coords)):
            raise ValueError("obstacle positions must be finite")
        if not np.isfinite(radius_value) or radius_value < 0:
            raise ValueError("obstacle radii must be non-negative and finite")
        if len(coords):
            trees.append((cKDTree(np.mod(coords, box), boxsize=box), radius_value))
    if not trees:
        raise ValueError("at least one non-empty obstacle group is required")

    shape, spacing = _grid_shape_and_spacing(box, grid_spacing)
    n_points = int(np.prod(shape, dtype=np.int64))
    if n_points > _GRID_POINT_WARN_THRESHOLD:
        warnings.warn(
            f"grid point 수 {n_points:,}개 — 메모리 사용량이 클 수 있습니다. "
            f"grid_spacing을 늘리거나 box를 축소하세요. "
            f"현재: box={box}, grid_spacing={grid_spacing} nm",
            UserWarning,
        )

    clearance = np.empty(n_points, dtype=np.float64)
    shape_tuple = tuple(int(value) for value in shape)
    for start in range(0, n_points, int(chunk_size)):
        stop = min(start + int(chunk_size), n_points)
        flat = np.arange(start, stop, dtype=np.int64)
        indices = np.column_stack(np.unravel_index(flat, shape_tuple))
        points = (indices.astype(float) + 0.5) * spacing
        nearest_surface = np.full(len(points), np.inf, dtype=np.float64)
        for tree, radius in trees:
            distances, _ = tree.query(points, k=1, workers=1)
            np.minimum(nearest_surface, distances - radius, out=nearest_surface)
        clearance[start:stop] = nearest_surface
    return clearance.reshape(shape_tuple), spacing


def periodic_component_summary(mask: np.ndarray) -> dict[str, object]:
    """Summarize 6-connected components after merging periodic face contacts.

    This reports component size only. A face-merging component is not
    automatically labelled as a winding/percolating network.
    """
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("mask must be a non-empty 3-D array")
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_labels = ndimage.label(values, structure=structure)
    if n_labels == 0:
        return {
            "n_periodic_components": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction_of_grid": 0.0,
            "largest_component_fraction_of_admissible": 0.0,
        }

    parent = np.arange(n_labels + 1, dtype=int)

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for axis in range(3):
        first_face = np.take(labels, 0, axis=axis)
        last_face = np.take(labels, -1, axis=axis)
        contacts = (first_face > 0) & (last_face > 0)
        for first, second in zip(first_face[contacts], last_face[contacts]):
            union(int(first), int(second))

    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    merged: dict[int, int] = {}
    for label in range(1, n_labels + 1):
        root = find(label)
        merged[root] = merged.get(root, 0) + int(counts[label])
    largest = max(merged.values())
    admissible = int(np.count_nonzero(values))
    return {
        "n_periodic_components": len(merged),
        "largest_component_voxels": largest,
        "largest_component_fraction_of_grid": float(largest / values.size),
        "largest_component_fraction_of_admissible": float(largest / admissible),
    }


def summarize_periodic_clearance(
    clearance_nm: np.ndarray,
    probe_radius_nm: float = 0.1657,
    bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Summarize local-clearance diameters and probe-admissible volume.

    ``probe_radius_nm`` is applied to centre clearance. The histogram therefore
    contains local obstacle-surface diameters at grid points that can admit the
    probe centre. It is not a pore-limiting diameter or a mesh-size estimate.
    """
    field = np.asarray(clearance_nm, dtype=float)
    probe_radius = float(probe_radius_nm)
    if field.ndim != 3 or field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("clearance_nm must be a non-empty finite 3-D field")
    if not np.isfinite(probe_radius) or probe_radius < 0:
        raise ValueError("probe_radius_nm must be non-negative and finite")
    if not isinstance(bins, (int, np.integer)) or int(bins) < 1:
        raise ValueError("bins must be a positive integer")

    void = field > 0.0
    admissible = field >= probe_radius
    diameters = 2.0 * field[admissible]
    component = periodic_component_summary(admissible)
    summary: dict[str, object] = {
        "method": "periodic_nearest_surface_clearance_grid",
        "probe_radius_nm": probe_radius,
        "n_grid_points": int(field.size),
        "n_void_grid_points": int(np.count_nonzero(void)),
        "n_probe_admissible_grid_points": int(np.count_nonzero(admissible)),
        "geometric_void_fraction": float(np.mean(void)),
        "probe_admissible_fraction": float(np.mean(admissible)),
        **component,
        "interpretation": (
            "local clearance and probe-admissible volume; not PoreBlazer PLD, "
            "not a winding proof, and not experimental mesh size"
        ),
    }
    if not len(diameters):
        summary.update({
            "local_clearance_diameter_peak_nm": 0.0,
            "local_clearance_diameter_max_nm": 0.0,
            "local_clearance_diameter_percentiles_nm": [],
        })
        return np.array([]), np.array([]), summary

    hist, edges = np.histogram(diameters, bins=int(bins), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    summary.update({
        "local_clearance_diameter_peak_nm": float(centres[int(np.argmax(hist))]),
        "local_clearance_diameter_max_nm": float(np.max(diameters)),
        "local_clearance_diameter_percentiles_nm": np.percentile(
            diameters, [5, 25, 50, 75, 95]
        ).tolist(),
    })
    return centres, hist, summary


def calculate_periodic_clearance_distribution(
    obstacle_groups: Sequence[tuple[np.ndarray, float]],
    box_size: np.ndarray,
    grid_spacing: float = 0.2,
    probe_radius: float = 0.1657,
    bins: int = 50,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Calculate a mixed-radius periodic clearance distribution."""
    clearance, actual_spacing = periodic_clearance_grid(
        obstacle_groups,
        box_size,
        grid_spacing=grid_spacing,
        chunk_size=chunk_size,
    )
    centres, hist, summary = summarize_periodic_clearance(
        clearance,
        probe_radius_nm=probe_radius,
        bins=bins,
    )
    summary.update({
        "requested_grid_spacing_nm": float(grid_spacing),
        "actual_grid_spacing_nm": actual_spacing.tolist(),
        "grid_shape": list(clearance.shape),
        "obstacle_groups": [
            {"n_obstacles": int(len(np.asarray(coords))), "radius_nm": float(radius)}
            for coords, radius in obstacle_groups
        ],
    })
    return centres, hist, summary


def calculate_pore_size_distribution(
    coords, box_size,
    grid_spacing=0.2,
    bead_radius=0.24,
    bins=50,
):
    """
    Grid 기반 void radius distribution 계산.
    주의: Poreblazer와 방법론이 다름 (nearest polymer surface distance 기반).

    coords      : (N, 3) polymer bead 좌표 [nm]
    box_size    : (3,) box 크기 [nm]  ※ orthorhombic만 지원
    grid_spacing: grid 간격 [nm]
    bead_radius : polymer bead 유효 반경 [nm]
    bins        : histogram bin 수

    반환값: (bin_centers [nm], hist [probability density], metadata dict)
    """
    coords_array = np.asarray(coords, dtype=float)
    meta = {
        "grid_spacing_nm": float(grid_spacing),
        "bead_radius_nm": float(bead_radius),
        "bins": int(bins),
        "method": "periodic_nearest_surface_grid",
        "n_polymer_atoms": int(len(coords_array)),
    }
    if len(coords_array) == 0:
        return np.array([]), np.array([]), meta

    centres, hist, periodic_meta = calculate_periodic_clearance_distribution(
        [(coords_array, float(bead_radius))],
        np.asarray(box_size, dtype=float),
        grid_spacing=float(grid_spacing),
        probe_radius=0.0,
        bins=int(bins),
    )
    meta.update(periodic_meta)
    # Backward-compatible function historically returned radius bins.
    return centres / 2.0, hist * 2.0, meta


def get_peak_pore_size(
    coords, box_size, grid_spacing=0.2, bead_radius=0.24, bins=50
) -> PropertyResult:
    """
    peak pore diameter [nm] 를 PropertyResult 로 반환.

    validation_role = "proxy":
        nearest_surface_grid (single frame) 는 Poreblazer trajectory 결과와
        방법론이 다르므로 실험 target 과 직접 비교 불가.
    """
    bin_centers, hist, meta = calculate_pore_size_distribution(
        coords, box_size,
        grid_spacing=grid_spacing,
        bead_radius=bead_radius,
        bins=bins,
    )
    if len(hist) == 0:
        meta['peak_pore_diameter_nm'] = 0.0
        return PropertyResult.insufficient_data(
            "pore_size_single_frame_grid",
            reason="void grid points가 0개입니다 — box가 polymer로 완전히 채워져 있거나 grid_spacing이 너무 큽니다.",
            validation_role="proxy",
            metadata={
                **meta,
                "target_aliases": ["pore_diameter_nm"],
            },
        )

    peak_diameter = float(bin_centers[np.argmax(hist)]) * 2.0
    meta['peak_pore_diameter_nm'] = peak_diameter

    return PropertyResult(
        property="pore_size_single_frame_grid",
        value=peak_diameter,
        status="computed",
        direct_experiment_comparison_allowed=False,
        validation_role="proxy",
        metadata={
            **meta,
            "target_aliases": ["pore_diameter_nm"],
            "note": (
                "single-frame nearest-surface-grid; "
                "not comparable to Poreblazer trajectory pore diameter"
            ),
        },
    )

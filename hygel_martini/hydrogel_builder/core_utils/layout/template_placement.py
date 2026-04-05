"""Shared geometry helpers for template placement and side-chain tuning."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_alignment_basis(axis: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Build an orthonormal basis whose x-axis follows ``axis``."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        norm = 1.0
    x_axis = axis / norm
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(x_axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = ref - np.dot(ref, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis -= np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
            y_norm = np.linalg.norm(y_axis)
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        z_axis /= z_norm
    return np.column_stack((x_axis, y_axis, z_axis))


def place_template_coords(
    coords: np.ndarray,
    origin: np.ndarray | list[float] | tuple[float, ...],
    axis_vector: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Rotate and translate template coordinates onto an axis-aligned frame."""
    basis = build_alignment_basis(axis_vector)
    return np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ basis.T


def compute_template_positions(
    coords: np.ndarray,
    origin: np.ndarray | list[float] | tuple[float, ...],
    normal_vector: np.ndarray | list[float] | tuple[float, ...],
    tangent_vector: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray | None:
    """Build a local side-chain frame from normal and tangent vectors."""
    normal_vector = np.asarray(normal_vector, dtype=float)
    tangent_vector = np.asarray(tangent_vector, dtype=float)
    n_norm = np.linalg.norm(normal_vector)
    t_norm = np.linalg.norm(tangent_vector)
    if n_norm < 1e-8 or t_norm < 1e-8:
        return None
    n_vec = normal_vector / n_norm
    t_vec = tangent_vector / t_norm
    b_vec = np.cross(n_vec, t_vec)
    b_norm = np.linalg.norm(b_vec)
    if b_norm < 1e-8:
        return None
    b_vec /= b_norm
    n_vec = np.cross(t_vec, b_vec)
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-8:
        return None
    n_vec /= n_norm
    rotation = np.column_stack((n_vec, b_vec, t_vec))
    return np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ rotation.T


def resolve_sidechain_placement_tuning(sim_params: Dict[str, Any], atom_count: int) -> Dict[str, float | int]:
    """Resolve side-chain search settings from config with large-system fallback."""
    tuning: Dict[str, float | int] = {
        "num_candidate_vectors": int(sim_params.get("sidechain_num_candidates", 72)),
        "overlap_threshold_factor": float(sim_params.get("sidechain_overlap_threshold_factor", 0.8)),
        "search_radius_factor": float(sim_params.get("sidechain_search_radius_factor", 10.0)),
        "placement_scale": float(sim_params.get("sidechain_placement_scale", 0.5)),
        "nearby_atom_limit": int(sim_params.get("sidechain_nearby_atom_limit", 800)),
        "auto_threshold": int(sim_params.get("sidechain_auto_threshold", 20000)),
    }
    if atom_count > int(tuning["auto_threshold"]):
        tuning["num_candidate_vectors"] = max(12, int(tuning["num_candidate_vectors"]) // 3)
        tuning["search_radius_factor"] = min(float(tuning["search_radius_factor"]), 6.0)
        tuning["nearby_atom_limit"] = min(int(tuning["nearby_atom_limit"]), 400)
    return tuning

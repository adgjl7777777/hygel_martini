"""Helpers for assigning linker stubs to backbone ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class StubAssignment:
    """Chosen backbone end for a single linker stub."""

    linker_index: int
    stub_atom: object
    backbone_atom: object
    chain_index: int
    distance: float


def normalize_box_vector(box_vec) -> np.ndarray | None:
    """Return a 3-vector box size or ``None`` when PBC is disabled."""
    if box_vec is None:
        return None
    try:
        arr = np.asarray(box_vec, dtype=float)
    except Exception:
        return None
    if arr.shape == (3,):
        return arr
    if arr.shape == (3, 3):
        return np.diag(arr)
    return None


def pbc_distance(first, second, box_size: np.ndarray | None) -> float:
    """Compute the minimum-image distance between two coordinates."""
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if box_size is not None:
        delta -= box_size * np.round(delta / box_size)
    return float(np.linalg.norm(delta))


def group_linker_stubs(atoms: Iterable[object]) -> Dict[int, List[object]]:
    """Group linker terminal stubs by linker chain index."""
    grouped: Dict[int, List[object]] = {}
    for atom in atoms:
        if getattr(atom, "stub_type", None) not in ("backbone_1", "backbone_2"):
            continue
        linker_index = getattr(atom, "linker_chain_index", None)
        if linker_index is None:
            continue
        grouped.setdefault(int(linker_index), []).append(atom)
    for linker_index in grouped:
        grouped[linker_index].sort(
            key=lambda atom: (
                getattr(atom, "stub_type", ""),
                getattr(atom, "atom_id", -1),
            )
        )
    return grouped


def collect_backbone_ends(atoms: Iterable[object]) -> Dict[int, List[object]]:
    """Collect true backbone end atoms keyed by backbone chain index."""
    ends_by_chain: Dict[int, List[object]] = {}
    for atom in atoms:
        if getattr(atom, "end_tag", 0) != 1:
            continue
        if getattr(atom, "linker_chain_index", None) is not None:
            continue
        if getattr(atom, "chain_type", None) not in (None, "backbone"):
            continue
        chain_index = getattr(atom, "chain_index", None)
        if chain_index is None:
            continue
        ends_by_chain.setdefault(int(chain_index), []).append(atom)
    return ends_by_chain


def _stub_target_backbone(stub: object):
    target = getattr(stub, "target_backbone", None)
    if target not in (None, "", "dummy_id"):
        return target
    if getattr(stub, "linker_chain_index", None) is not None:
        fallback = getattr(stub, "backbone_type", None)
        if fallback not in (None, "", "dummy_id"):
            return fallback
    return None


def _is_compatible_target(stub: object, backbone_atom: object) -> bool:
    target = _stub_target_backbone(stub)
    if not target:
        return True
    return getattr(backbone_atom, "backbone_type", None) == target


def _candidate_end_options(
    stub: object,
    backbone_ends: Dict[int, List[object]],
    box_size: np.ndarray | None,
    candidate_limit: int,
) -> List[Tuple[float, int, object]]:
    options: List[Tuple[float, int, object]] = []
    stub_pos = getattr(stub, "position", None)
    if stub_pos is None:
        return options
    for chain_index, ends in backbone_ends.items():
        best_end = None
        best_distance = None
        for end_atom in ends:
            if not _is_compatible_target(stub, end_atom):
                continue
            distance = pbc_distance(stub_pos, getattr(end_atom, "position", None), box_size)
            if best_end is None or distance < best_distance:
                best_end = end_atom
                best_distance = distance
        if best_end is None:
            continue
        options.append((float(best_distance), int(chain_index), best_end))
    options.sort(key=lambda item: (item[0], item[1], getattr(item[2], "atom_id", -1)))
    if candidate_limit > 0:
        return options[:candidate_limit]
    return options


def plan_dynamic_crosslinks(
    linker_stubs: Dict[int, List[object]],
    backbone_ends: Dict[int, List[object]],
    box_vec,
    candidate_limit: int = 8,
):
    """Assign two compatible backbone ends to each linker.

    The matcher operates on true backbone end atoms rather than arbitrary
    backbone beads. It also avoids reusing the same end atom across multiple
    linkers whenever a unique assignment is available.
    """
    box_size = normalize_box_vector(box_vec)
    candidate_limit = max(int(candidate_limit), 1)
    pairing_options = []
    notes: List[str] = []

    for linker_index, stubs in sorted(linker_stubs.items()):
        if len(stubs) != 2:
            notes.append(
                f"Linker {linker_index}: Has {len(stubs)} stubs. Skipping (need 2)."
            )
            continue

        first_options = _candidate_end_options(stubs[0], backbone_ends, box_size, candidate_limit)
        second_options = _candidate_end_options(stubs[1], backbone_ends, box_size, candidate_limit)
        if not first_options or not second_options:
            notes.append(
                f"Linker {linker_index}: No compatible backbone-end candidates."
            )
            continue

        pairings = []
        for dist_a, chain_a, end_a in first_options:
            for dist_b, chain_b, end_b in second_options:
                if chain_a == chain_b:
                    continue
                if getattr(end_a, "atom_id", None) == getattr(end_b, "atom_id", None):
                    continue
                assignment_a = StubAssignment(
                    linker_index=linker_index,
                    stub_atom=stubs[0],
                    backbone_atom=end_a,
                    chain_index=chain_a,
                    distance=dist_a,
                )
                assignment_b = StubAssignment(
                    linker_index=linker_index,
                    stub_atom=stubs[1],
                    backbone_atom=end_b,
                    chain_index=chain_b,
                    distance=dist_b,
                )
                pairings.append((dist_a + dist_b, assignment_a, assignment_b))

        if not pairings:
            notes.append(
                f"Linker {linker_index}: Could not find 2 distinct compatible chains."
            )
            continue

        pairings.sort(
            key=lambda item: (
                item[0],
                item[1].distance,
                item[2].distance,
                item[1].chain_index,
                item[2].chain_index,
            )
        )
        pairing_options.append((pairings[0][0], linker_index, pairings))

    assignments: Dict[int, Tuple[StubAssignment, StubAssignment]] = {}
    used_end_atoms = set()

    for _, linker_index, pairings in sorted(pairing_options, key=lambda item: (item[0], item[1])):
        chosen = None
        for _, assignment_a, assignment_b in pairings:
            end_a = getattr(assignment_a.backbone_atom, "atom_id", None)
            end_b = getattr(assignment_b.backbone_atom, "atom_id", None)
            if end_a in used_end_atoms or end_b in used_end_atoms:
                continue
            chosen = (assignment_a, assignment_b)
            break

        if chosen is None:
            _, assignment_a, assignment_b = pairings[0]
            chosen = (assignment_a, assignment_b)
            notes.append(
                "Linker {}: Reusing a backbone end because no unique-end pairing remained.".format(
                    linker_index
                )
            )

        assignments[linker_index] = chosen
        used_end_atoms.add(getattr(chosen[0].backbone_atom, "atom_id", None))
        used_end_atoms.add(getattr(chosen[1].backbone_atom, "atom_id", None))

    return assignments, notes

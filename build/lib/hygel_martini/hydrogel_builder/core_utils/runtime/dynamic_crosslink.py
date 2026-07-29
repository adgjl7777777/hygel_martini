"""Helpers for assigning placed linker stubs to nearby backbone ends.

This module is intentionally geometry-only.  It does not choose linker
directions, rotate already-built linkers, or repair a global graph by assigning
arbitrary chain pairs.  Layout code must choose the local x/y/z matching state
first; once the BCK positions exist, each BCK bonds to the nearest compatible
backbone end(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
    respect_target_backbone: bool = True,
) -> List[Tuple[float, int, object]]:
    """Return the nearest compatible end for each compatible backbone chain."""
    options: List[Tuple[float, int, object]] = []
    stub_pos = getattr(stub, "position", None)
    if stub_pos is None:
        return options
    for chain_index, ends in backbone_ends.items():
        best_end = None
        best_distance = None
        for end_atom in ends:
            if respect_target_backbone and not _is_compatible_target(stub, end_atom):
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


def _pick_stub_targets(
    linker_index: int,
    stub: object,
    options: List[Tuple[float, int, object]],
    targets_per_stub: int,
    used_end_atoms: set,
    local_used_atoms: set,
    linker_used_chains: set,
) -> List[StubAssignment]:
    picked: List[StubAssignment] = []
    for distance, chain_index, end_atom in options:
        end_id = getattr(end_atom, "atom_id", None)
        if end_id in used_end_atoms or end_id in local_used_atoms:
            continue
        if chain_index in linker_used_chains:
            continue
        picked.append(
            StubAssignment(
                linker_index=linker_index,
                stub_atom=stub,
                backbone_atom=end_atom,
                chain_index=chain_index,
                distance=distance,
            )
        )
        used_end_atoms.add(end_id)
        local_used_atoms.add(end_id)
        linker_used_chains.add(chain_index)
        if len(picked) >= targets_per_stub:
            break
    return picked


def plan_dynamic_crosslinks(
    linker_stubs: Dict[int, List[object]],
    backbone_ends: Dict[int, List[object]],
    box_vec,
    candidate_limit: int = 8,
    targets_per_stub: int = 1,
    respect_target_backbone_policy: bool = False,
):
    """Assign compatible backbone ends to each placed linker stub.

    ``targets_per_stub=1`` keeps the historical pairwise linker behavior.
    ``targets_per_stub=2`` means each BCK stub creates one local two-chain
    junction, so one two-BCK linker creates two polymer junctions and four
    BCK-backbone bonds in total.
    """
    box_size = normalize_box_vector(box_vec)
    candidate_limit = max(int(candidate_limit), 1)
    targets_per_stub = max(int(targets_per_stub), 1)
    pairing_options = []
    notes: List[str] = []

    if targets_per_stub > 1:
        linker_state_candidates = {}
        notes.append(
            "targets_per_stub={} over {} linkers and {} true backbone ends; "
            "target filtering {}.".format(
                targets_per_stub,
                len(linker_stubs),
                sum(len(v) for v in backbone_ends.values()),
                "enabled" if respect_target_backbone_policy else "disabled",
            )
        )

        for linker_index, stubs in sorted(linker_stubs.items()):
            if len(stubs) != 2:
                notes.append(
                    f"Linker {linker_index}: Has {len(stubs)} stubs. Skipping (need 2)."
                )
                continue

            per_stub_candidates = []
            for stub in stubs:
                respect_target = respect_target_backbone_policy
                options = _candidate_end_options(
                    stub,
                    backbone_ends,
                    box_size,
                    candidate_limit,
                    respect_target_backbone=respect_target,
                )
                if len(options) < targets_per_stub and candidate_limit > 0:
                    options = _candidate_end_options(
                        stub,
                        backbone_ends,
                        box_size,
                        0,
                        respect_target_backbone=respect_target,
                    )
                if len(options) < targets_per_stub:
                    notes.append(
                        f'Linker {linker_index}: Stub {getattr(stub, "atom_id", None)} has only {len(options)} compatible backbone-end candidates.'
                    )
                    per_stub_candidates = []
                    break

                stub_candidates = []
                for combo in combinations(options, targets_per_stub):
                    chain_ids = [chain_index for _, chain_index, _ in combo]
                    end_ids = [getattr(end_atom, "atom_id", None) for _, _, end_atom in combo]
                    if len(set(chain_ids)) != len(chain_ids):
                        continue
                    if len(set(end_ids)) != len(end_ids):
                        continue
                    assignments_for_stub = tuple(
                        StubAssignment(
                            linker_index=linker_index,
                            stub_atom=stub,
                            backbone_atom=end_atom,
                            chain_index=chain_index,
                            distance=distance,
                        )
                        for distance, chain_index, end_atom in combo
                    )
                    stub_candidates.append(
                        (
                            sum(item[0] for item in combo),
                            assignments_for_stub,
                            frozenset(end_ids),
                            frozenset(chain_ids),
                        )
                    )
                stub_candidates.sort(key=lambda item: (item[0], tuple(sorted(item[3]))))
                if candidate_limit > 0:
                    stub_candidates = stub_candidates[:candidate_limit]
                if not stub_candidates:
                    notes.append(
                        f'Linker {linker_index}: Stub {getattr(stub, "atom_id", None)} has no valid {targets_per_stub}-end candidate set.'
                    )
                    per_stub_candidates = []
                    break
                per_stub_candidates.append(stub_candidates)

            if len(per_stub_candidates) != 2:
                linker_state_candidates[linker_index] = []
                continue

            states = []
            for left in per_stub_candidates[0]:
                for right in per_stub_candidates[1]:
                    end_ids = set(left[2]) | set(right[2])
                    chain_ids = set(left[3]) | set(right[3])
                    if len(end_ids) != 2 * targets_per_stub:
                        continue
                    if len(chain_ids) != 2 * targets_per_stub:
                        continue
                    states.append(
                        (
                            left[0] + right[0],
                            left[1] + right[1],
                            frozenset(end_ids),
                        )
                    )
            states.sort(key=lambda item: (item[0], tuple(a.chain_index for a in item[1])))
            if candidate_limit > 0:
                states = states[: candidate_limit * candidate_limit]
            linker_state_candidates[linker_index] = states
            if not states:
                notes.append(f"Linker {linker_index}: No valid two-junction candidate state.")

        selected_states = {}
        state_count = sum(len(states) for states in linker_state_candidates.values())
        search_budget = max(100000, state_count * max(len(linker_state_candidates), 1) * 16)
        search_visits = 0
        notes.append(
            f"Backtracking search prepared {state_count} candidate linker states; budget={search_budget}."
        )

        linker_order = sorted(
            [k for k, v in linker_state_candidates.items() if v],
            key=lambda k: len(linker_state_candidates[k])
        )

        def _search(depth: int, used_end_ids: set) -> bool:
            nonlocal search_visits
            search_visits += 1
            if search_visits > search_budget:
                return False
            if depth == len(linker_order):
                return True

            linker_index = linker_order[depth]
            states = linker_state_candidates[linker_index]
            for _, chosen, end_ids in states:
                if used_end_ids.intersection(end_ids):
                    continue
                selected_states[linker_index] = chosen
                used_end_ids.update(end_ids)
                if _search(depth + 1, used_end_ids):
                    return True
                used_end_ids.difference_update(end_ids)
                selected_states.pop(linker_index, None)
            return False

        if not _search(0, set()):
            notes.append(
                "No globally unique endpoint assignment found for targets_per_stub>1 "
                f"within search_budget={search_budget}."
            )

        assignments: Dict[int, Tuple[StubAssignment, ...]] = {
            linker_index: tuple(chosen)
            for linker_index, chosen in selected_states.items()
        }

        return assignments, notes

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

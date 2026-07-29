#!/usr/bin/env python3
"""Generic bonded-topology audit for hydrogel_builder outputs.

The audit intentionally keeps chemistry-specific expectations out of the code.
Pass expected chain patterns, linker residues, target residues, and count
thresholds through CLI options so the same tool can check PEG, Pluronic, or
other hydrogel validation cases.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Iterable


def _csv_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {token.strip() for token in value.split(",") if token.strip()}


def _parse_pattern(value: str | None) -> list[tuple[str, int]]:
    if not value:
        return []
    pattern = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid pattern token {token!r}; expected RES:COUNT")
        residue, count = token.split(":", 1)
        pattern.append((residue.strip(), int(count.strip())))
    return pattern


def parse_itp(path: Path):
    atoms = {}
    bonds = []
    angles = []
    dihedrals = []
    section = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        fields = line.split()
        if section == "atoms" and len(fields) >= 8 and fields[0].isdigit():
            atoms[int(fields[0])] = {
                "type": fields[1],
                "resnr": fields[2],
                "residue": fields[3],
                "atom": fields[4],
                "charge": fields[6],
                "mass": fields[7],
            }
        elif section == "bonds" and len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            bonds.append((int(fields[0]), int(fields[1]), fields[2:]))
        elif section == "angles" and len(fields) >= 3 and all(token.isdigit() for token in fields[:3]):
            angles.append((int(fields[0]), int(fields[1]), int(fields[2]), fields[3:]))
        elif section == "dihedrals" and len(fields) >= 4 and all(token.isdigit() for token in fields[:4]):
            dihedrals.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), fields[4:]))
    return atoms, bonds, angles, dihedrals


def connected_components(atom_ids: Iterable[int], bonds: list[tuple[int, int, list[str]]]):
    atom_set = set(atom_ids)
    adj = {atom_id: [] for atom_id in atom_set}
    for a, b, _params in bonds:
        if a in atom_set and b in atom_set:
            adj[a].append(b)
            adj[b].append(a)

    seen = set()
    components = []
    for atom_id in sorted(atom_set):
        if atom_id in seen:
            continue
        queue = deque([atom_id])
        seen.add(atom_id)
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(comp))
    return sorted(components, key=len, reverse=True), adj


def path_order(component: list[int], adj: dict[int, list[int]]) -> list[int]:
    degrees = {atom_id: len([n for n in adj.get(atom_id, []) if n in component]) for atom_id in component}
    endpoints = [atom_id for atom_id, degree in degrees.items() if degree <= 1]
    start = min(endpoints or component)
    ordered = []
    prev = None
    cur = start
    comp_set = set(component)
    while cur is not None:
        ordered.append(cur)
        candidates = [n for n in adj.get(cur, []) if n in comp_set and n != prev]
        nxt = min(candidates) if candidates else None
        prev, cur = cur, nxt
        if cur in ordered:
            break
    return ordered


def expected_sequence(pattern: list[tuple[str, int]]) -> list[str]:
    seq = []
    for residue, count in pattern:
        seq.extend([residue] * count)
    return seq


def audit(args):
    atoms, bonds, angles, dihedrals = parse_itp(args.itp)
    residues = {atom_id: atom["residue"] for atom_id, atom in atoms.items()}

    linker_residues = _csv_tokens(args.linker_residues)
    backbone_residues = _csv_tokens(args.backbone_residues)
    allowed_targets = _csv_tokens(args.allowed_dynamic_target_residues)
    chain_pattern = _parse_pattern(args.expect_chain_pattern)
    expected_chain = expected_sequence(chain_pattern)

    all_components, all_adj = connected_components(atoms.keys(), bonds)
    largest_fraction = len(all_components[0]) / len(atoms) if atoms and all_components else 0.0

    linker_atoms = {atom_id for atom_id, residue in residues.items() if residue in linker_residues}
    backbone_atoms = {atom_id for atom_id, residue in residues.items() if residue in backbone_residues}

    linker_internal_bonds = []
    dynamic_bonds = []
    dynamic_target_counts = Counter()
    for a, b, params in bonds:
        a_linker = a in linker_atoms
        b_linker = b in linker_atoms
        if a_linker and b_linker:
            linker_internal_bonds.append((a, b, params))
        elif a_linker != b_linker:
            target = b if a_linker else a
            dynamic_bonds.append((a, b, params))
            dynamic_target_counts[residues.get(target, "?")] += 1

    backbone_components, backbone_adj = connected_components(backbone_atoms, bonds)
    chain_reports = []
    bad_chain_reports = []
    for idx, component in enumerate(backbone_components):
        ordered = path_order(component, backbone_adj)
        seq = [residues[atom_id] for atom_id in ordered]
        report = {
            "index": idx,
            "length": len(component),
            "ends": [seq[0], seq[-1]] if seq else [],
            "sequence": seq,
        }
        chain_reports.append(report)
        if expected_chain and seq != expected_chain and list(reversed(seq)) != expected_chain:
            bad_chain_reports.append(report)

    junction_angles = []
    junction_angle_params = Counter()
    for a, b, c, params in angles:
        ids = {a, b, c}
        if ids & linker_atoms and ids & backbone_atoms:
            junction_angles.append((a, b, c, params))
            junction_angle_params[tuple(params)] += 1

    junction_dihedrals = []
    junction_dihedral_params = Counter()
    for a, b, c, d, params in dihedrals:
        ids = {a, b, c, d}
        if ids & linker_atoms and ids & backbone_atoms:
            junction_dihedrals.append((a, b, c, d, params))
            junction_dihedral_params[tuple(params)] += 1

    # Calculate degree-based theoretical angles
    theory_angles = 0
    for atom_id, neighbors in all_adj.items():
        deg = len(neighbors)
        if deg >= 2:
            theory_angles += deg * (deg - 1) // 2

    # Graph-based search for theoretical unique dihedrals to classify by residues
    theory_dihedrals_set = set()
    for a, b, _params in bonds:
        if a in all_adj and b in all_adj:
            for w in all_adj[a]:
                if w == b:
                    continue
                for x in all_adj[b]:
                    if x == a or x == w:
                        continue
                    if w < x:
                        theory_dihedrals_set.add((w, a, b, x))
                    else:
                        theory_dihedrals_set.add((x, b, a, w))

    # Classify dihedrals by residue types (PEO-like, PPO-only, Hybrid)
    peo_like_count = 0
    ppo_only_count = 0
    hybrid_count = 0
    unknown_count = 0
    for a, b, c, d in theory_dihedrals_set:
        res_a = residues.get(a, "?")
        res_b = residues.get(b, "?")
        res_c = residues.get(c, "?")
        res_d = residues.get(d, "?")
        res_set = {res_a, res_b, res_c, res_d}
        
        has_peo = any(r in {"PEO", "BCK", "LNK", "XLEE"} for r in res_set)
        has_ppo = "PPO" in res_set
        
        if has_peo and has_ppo:
            hybrid_count += 1
        elif has_peo:
            peo_like_count += 1
        elif has_ppo:
            ppo_only_count += 1
        else:
            unknown_count += 1

    if args.exclude_hybrid_dihedrals:
        theory_dihedrals = len(theory_dihedrals_set) - hybrid_count
    else:
        theory_dihedrals = len(theory_dihedrals_set)

    # Count actual unique physical angles and dihedrals
    unique_angles_set = set()
    for a, b, c, _params in angles:
        if a in atoms and b in atoms and c in atoms:
            unique_angles_set.add((min(a, c), b, max(a, c)))
    actual_unique_angles = len(unique_angles_set)

    unique_dihedrals_set = set()
    for a, b, c, d, _params in dihedrals:
        if a in atoms and b in atoms and c in atoms and d in atoms:
            if a < d:
                unique_dihedrals_set.add((a, b, c, d))
            else:
                unique_dihedrals_set.add((d, c, b, a))
    actual_unique_dihedrals = len(unique_dihedrals_set)

    issues = []
    if actual_unique_angles != theory_angles:
        issues.append(f"actual unique angles {actual_unique_angles} != theoretical angles {theory_angles}")
    if actual_unique_dihedrals != theory_dihedrals:
        issues.append(f"actual unique dihedrals {actual_unique_dihedrals} != theoretical dihedrals {theory_dihedrals}")
    if args.expect_components is not None and len(all_components) != args.expect_components:
        issues.append(f"component count {len(all_components)} != expected {args.expect_components}")
    if args.min_largest_component_fraction is not None and largest_fraction < args.min_largest_component_fraction:
        issues.append(
            f"largest component fraction {largest_fraction:.6f} < expected {args.min_largest_component_fraction}"
        )
    if args.expect_dynamic_bonds is not None and len(dynamic_bonds) != args.expect_dynamic_bonds:
        issues.append(f"dynamic bond count {len(dynamic_bonds)} != expected {args.expect_dynamic_bonds}")
    if args.expect_linker_internal_bonds is not None and len(linker_internal_bonds) != args.expect_linker_internal_bonds:
        issues.append(
            f"linker internal bond count {len(linker_internal_bonds)} != expected {args.expect_linker_internal_bonds}"
        )
    if allowed_targets:
        bad_targets = {res: count for res, count in dynamic_target_counts.items() if res not in allowed_targets}
        if bad_targets:
            issues.append(f"dynamic bonds target disallowed residues: {bad_targets}")
    if expected_chain and bad_chain_reports:
        issues.append(f"{len(bad_chain_reports)} backbone chains do not match expected pattern")
    if args.require_junction_angles and not junction_angles:
        issues.append("no linker-backbone junction angles found")
    if args.require_junction_dihedrals and not junction_dihedrals:
        issues.append("no linker-backbone junction dihedrals found")

    summary = {
        "itp": str(args.itp),
        "atoms": len(atoms),
        "bonds": len(bonds),
        "angles": len(angles),
        "theoretical_angles": theory_angles,
        "actual_unique_angles": actual_unique_angles,
        "dihedrals": len(dihedrals),
        "theoretical_dihedrals_total_graph": len(theory_dihedrals_set),
        "theoretical_dihedrals_peo_like": peo_like_count,
        "theoretical_dihedrals_ppo_only": ppo_only_count,
        "theoretical_dihedrals_hybrid": hybrid_count,
        "theoretical_dihedrals_expected": theory_dihedrals,
        "actual_unique_dihedrals": actual_unique_dihedrals,
        "components": len(all_components),
        "largest_component_fraction": round(largest_fraction, 6),
        "backbone_chains": len(backbone_components),
        "bad_chain_count": len(bad_chain_reports),
        "linker_internal_bonds": len(linker_internal_bonds),
        "dynamic_bonds": len(dynamic_bonds),
        "dynamic_target_counts": dict(sorted(dynamic_target_counts.items())),
        "junction_angles": len(junction_angles),
        "junction_angle_param_counts": {" ".join(k): v for k, v in sorted(junction_angle_params.items())},
        "junction_dihedrals": len(junction_dihedrals),
        "junction_dihedral_param_counts": {" ".join(k): v for k, v in sorted(junction_dihedral_params.items())},
        "issues": issues,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            if key == "issues":
                continue
            print(f"{key}: {value}")
        if issues:
            print("issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("issues: none")

    return 1 if issues and args.fail_on_issue else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itp", required=True, type=Path)
    parser.add_argument("--backbone-residues", default="")
    parser.add_argument("--linker-residues", default="")
    parser.add_argument("--allowed-dynamic-target-residues", default="")
    parser.add_argument("--expect-chain-pattern", default="")
    parser.add_argument("--expect-components", type=int)
    parser.add_argument("--min-largest-component-fraction", type=float)
    parser.add_argument("--expect-dynamic-bonds", type=int)
    parser.add_argument("--expect-linker-internal-bonds", type=int)
    parser.add_argument("--require-junction-angles", action="store_true")
    parser.add_argument("--require-junction-dihedrals", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-issue", action="store_true")
    parser.add_argument("--exclude-hybrid-dihedrals", action="store_true")
    args = parser.parse_args()
    raise SystemExit(audit(args))



if __name__ == "__main__":
    main()

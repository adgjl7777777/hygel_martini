"""Serialize logical endpoint pairs before routing and audit written partitions.

The input metadata originates in the layout planner. This checks its transfer,
not planner optimality, force-field validity, or periodic embedding equivalence.
"""
import json
from collections import Counter
from pathlib import Path


def _logical_key(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def capture_planner_input(linker_stubs, backbone_ends):
    endpoint_map = {}
    for atoms in backbone_ends.values():
        for atom in atoms:
            value = getattr(atom, "planned_endpoint_id", None)
            if value is None:
                raise ValueError("Missing logical endpoint identity before routing")
            key = _logical_key(value)
            if key in endpoint_map:
                raise ValueError("Duplicate logical endpoint before routing")
            endpoint_map[key] = int(atom.atom_id) + 1
    if len(set(endpoint_map.values())) != len(endpoint_map):
        raise ValueError("Non-injective endpoint-to-atom map")
    linkers = []
    for linker, stubs in sorted(linker_stubs.items()):
        if len(stubs) != 2:
            raise ValueError("Logical witness requires two stubs per linker")
        plan = getattr(stubs[0], "planned_endpoint_edges", None)
        if not plan or plan != getattr(stubs[1], "planned_endpoint_edges", None):
            raise ValueError("Explicit crosslink plan required: missing or inconsistent planner input")
        if len(plan) != 2 or any(len(edge) != 2 for edge in plan):
            raise ValueError("Logical witness requires two endpoint pairs")
        pairs = [[_logical_key(value) for value in edge] for edge in plan]
        if any(key not in endpoint_map for edge in pairs for key in edge):
            raise ValueError("Planner endpoint absent from independent atom map")
        linkers.append({"linker": int(linker),
                        "stubs": [int(a.atom_id) + 1 for a in stubs],
                        "logical_pairs": pairs})
    used = [key for row in linkers for pair in row["logical_pairs"] for key in pair]
    if Counter(used) != Counter(endpoint_map.keys()):
        raise ValueError("Planner input does not use every endpoint exactly once")
    return {"schema": 1, "origin": "layout metadata captured before router call",
            "indexing": "one-based ITP", "endpoint_atom_map": endpoint_map,
            "linkers": linkers,
            "equivalence": "unordered endpoint pairs; swap of two stubs within each linker"}


def guard_written_planner_input(itp_path, witness, report_path):
    report = {"status": "FAIL", "scope": "pre-router logical plan to written linker partitions"}
    try:
        if not witness or witness.get("schema") != 1:
            raise ValueError("Missing pre-router planner witness")
        endpoint_map = witness["endpoint_atom_map"]
        endpoint_ids = set(endpoint_map.values())
        if len(endpoint_ids) != len(endpoint_map):
            raise ValueError("Non-injective endpoint map")
        atoms, bonds, section = [], [], None
        for line in Path(itp_path).read_text().splitlines():
            line = line.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[] ").lower()
            elif section == "atoms":
                atoms.append(int(line.split()[0]))
            elif section == "bonds":
                bonds.append(tuple(map(int, line.split()[:2])))
        stubs = [a for row in witness["linkers"] for a in row["stubs"]]
        if len(stubs) != len(set(stubs)) or set(stubs) & endpoint_ids:
            raise ValueError("Duplicate or overlapping stub identities")
        if len(atoms) != len(set(atoms)) or not set(stubs).union(endpoint_ids) <= set(atoms):
            raise ValueError("Missing or duplicate written atom IDs")
        all_used = [key for row in witness["linkers"] for pair in row["logical_pairs"] for key in pair]
        if Counter(all_used) != Counter(endpoint_map.keys()):
            raise ValueError("Logical plan endpoint multiplicities invalid")
        adjacency = {s: [] for s in stubs}
        for a, b in bonds:
            if a in adjacency and b in endpoint_ids:
                adjacency[a].append(b)
            if b in adjacency and a in endpoint_ids:
                adjacency[b].append(a)
        mismatches = []
        for row in witness["linkers"]:
            expected = sorted(tuple(sorted(endpoint_map[k] for k in pair)) for pair in row["logical_pairs"])
            actual = sorted(tuple(sorted(adjacency[s])) for s in row["stubs"])
            if expected != actual:
                mismatches.append({"linker": row["linker"], "expected": expected, "observed": actual})
        report.update(linkers=len(witness["linkers"]), mismatches=mismatches)
        if mismatches:
            raise ValueError("Written topology differs from pre-router planner input")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(str(error)) from error
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report

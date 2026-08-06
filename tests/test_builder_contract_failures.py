from __future__ import annotations

from pathlib import Path

import pytest

from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import (
    plan_dynamic_crosslinks,
)
from hygel_martini.property_extract.network_topology import audit_reduced_network


class _Atom:
    def __init__(self, atom_id, position, chain_index=None):
        self.atom_id = atom_id
        self.position = position
        self.chain_index = chain_index
        self.end_tag = 1
        self.chain_type = "backbone"
        self.linker_chain_index = None
        self.planned_endpoint_id = None
        self.planned_endpoint_edges = None
        self.stub_type = None


def _valid_planned_fixture():
    ends = {}
    positions = {
        "a": (0.0, 0.0, 0.0),
        "b": (9.0, 0.0, 0.0),
        "c": (1.0, 0.0, 0.0),
        "d": (10.0, 0.0, 0.0),
    }
    for chain_index, name in enumerate(("a", "b", "c", "d")):
        atom = _Atom(chain_index, positions[name], chain_index)
        atom.planned_endpoint_id = name
        ends[chain_index] = [atom]

    left_stub = _Atom(10, (0.2, 0.0, 0.0))
    right_stub = _Atom(11, (9.8, 0.0, 0.0))
    for stub, stub_type in ((left_stub, "backbone_1"), (right_stub, "backbone_2")):
        stub.end_tag = 2
        stub.chain_type = "linker"
        stub.linker_chain_index = 7
        stub.stub_type = stub_type
        stub.planned_endpoint_edges = (("a", "b"), ("c", "d"))
    return ends, [left_stub, right_stub]


def test_missing_planned_endpoint_is_rejected() -> None:
    ends, stubs = _valid_planned_fixture()
    del ends[3]

    with pytest.raises(ValueError, match="planned endpoint 'd' was not materialized"):
        plan_dynamic_crosslinks(
            {7: stubs},
            ends,
            box_vec=None,
            candidate_limit=64,
            targets_per_stub=2,
        )


def test_duplicate_planned_endpoint_id_is_rejected() -> None:
    ends, stubs = _valid_planned_fixture()
    duplicate = _Atom(99, (4.0, 0.0, 0.0), 99)
    duplicate.planned_endpoint_id = "a"
    ends[99] = [duplicate]

    with pytest.raises(ValueError, match="Duplicate planned endpoint id: 'a'"):
        plan_dynamic_crosslinks(
            {7: stubs},
            ends,
            box_vec=None,
            candidate_limit=64,
            targets_per_stub=2,
        )


def test_partial_planner_metadata_is_rejected() -> None:
    ends, stubs = _valid_planned_fixture()
    stubs[1].planned_endpoint_edges = None

    with pytest.raises(ValueError, match="Partial layout-planner endpoint metadata"):
        plan_dynamic_crosslinks(
            {7: stubs},
            ends,
            box_vec=None,
            candidate_limit=64,
            targets_per_stub=2,
        )


def test_reused_planned_endpoint_is_rejected_across_linkers() -> None:
    ends, stubs = _valid_planned_fixture()
    second = [_Atom(20, (0.3, 0.0, 0.0)), _Atom(21, (9.7, 0.0, 0.0))]
    for stub, stub_type in zip(second, ("backbone_1", "backbone_2")):
        stub.end_tag = 2
        stub.chain_type = "linker"
        stub.linker_chain_index = 8
        stub.stub_type = stub_type
        stub.planned_endpoint_edges = (("a", "b"), ("c", "d"))

    with pytest.raises(ValueError, match="planned endpoint 'a' is reused"):
        plan_dynamic_crosslinks(
            {7: stubs, 8: second},
            ends,
            box_vec=None,
            candidate_limit=64,
            targets_per_stub=2,
        )


def _write_topology_fixture(directory: Path, remove_attachment: bool):
    bonds = [
        "1 2 1",
        "3 4 1",
        "1 5 1",
        "5 6 1",
        "6 3 1",
        "2 7 1",
        "7 8 1",
        "8 4 1",
    ]
    if remove_attachment:
        bonds.remove("8 4 1")
    itp_text = """\
[ moleculetype ]
TOY 1

[ atoms ]
1 T 1 BCK J1 1 0 1
2 T 1 BCK J2 1 0 1
3 T 2 BCK J1 2 0 1
4 T 2 BCK J2 2 0 1
5 T 3 PEO E1 3 0 1
6 T 3 PEO E2 3 0 1
7 T 4 PEO E1 4 0 1
8 T 4 PEO E2 4 0 1

[ bonds ]
""" + "\n".join(bonds) + "\n"
    coordinates = [
        (1.0, 1.0, 1.0),
        (1.1, 1.0, 1.0),
        (4.0, 1.0, 1.0),
        (4.1, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (3.0, 1.0, 1.0),
        (0.2, 1.0, 1.0),
        (9.0, 1.0, 1.0),
    ]
    gro_lines = ["toy", str(len(coordinates))]
    for index, (x, y, z) in enumerate(coordinates, start=1):
        gro_lines.append(
            f"{index:5d}{'TOY':<5}{'B':>5}{index:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    gro_lines.append("  10.00000  10.00000  10.00000")

    suffix = "mutated" if remove_attachment else "valid"
    itp = directory / f"{suffix}.itp"
    gro = directory / f"{suffix}.gro"
    itp.write_text(itp_text)
    gro.write_text("\n".join(gro_lines) + "\n")
    return itp, gro


def test_persisted_bond_mutation_changes_audit_and_creates_defect(tmp_path: Path) -> None:
    valid_itp, valid_gro = _write_topology_fixture(tmp_path, remove_attachment=False)
    mutated_itp, mutated_gro = _write_topology_fixture(tmp_path, remove_attachment=True)

    valid = audit_reduced_network(valid_itp, valid_gro)
    mutated = audit_reduced_network(mutated_itp, mutated_gro)

    assert valid["junction_attachment_bond_count"] == 4
    assert mutated["junction_attachment_bond_count"] == 3
    assert valid["atom_bond_connectivity_sha256"] != mutated["atom_bond_connectivity_sha256"]
    assert mutated["junction_degree_distribution"] != valid["junction_degree_distribution"]

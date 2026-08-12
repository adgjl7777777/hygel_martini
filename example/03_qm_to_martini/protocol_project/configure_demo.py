#!/usr/bin/env python3
"""Create a fully populated, unsealed synthetic protocol project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from hygel_martini.param_opt.qm_to_martini.protocol.engine import (
    initialize_project,
    refresh_checksums,
)
from hygel_martini.param_opt.qm_to_martini.protocol.io import (
    atomic_write_text,
    atomic_write_yaml,
    load_yaml,
    sha256_file,
)


PASS_VALUES: Dict[str, Dict[str, Any]] = {
    "E0": {
        "e0_provenance_complete": True,
        "e0_run_integrity": 1.0,
    },
    "E1": {
        "e1_graph_and_canonicalization": True,
        "e1_rank_condition_support": 4200.0,
    },
    "E2": {
        "e2_effect_necessity": 8,
        "e2_exact_selection_stability": 0.7777778,
        "e2_physical_admissibility": True,
    },
    "E3": {
        "e3_exact_topology": True,
        "e3_complete_manifest": 1.0,
    },
    "E4": {
        "e4_target_distribution": 0.018,
        "e4_upstream_nonregression": 8,
        "e4_complete_topology_replay": 1.0,
    },
    "E5": {
        "e5_one_shot_confirmation": 1.0,
    },
    "E6": {
        "e6_transfer_qualification": 0.75,
        "e6_no_parameter_feedback": True,
    },
}


def _set_numeric_rule(
    contract: Dict[str, Any], gate: str, criterion: str, operator: str, expected: Any, description: str
) -> None:
    gate_row = next(row for row in contract["gates"] if row["id"] == gate)
    rule = next(row for row in gate_row["criteria"] if row["id"] == criterion)
    rule["operator"] = operator
    rule["expected"] = expected
    rule["description"] = description


def create_demo(root: Path) -> None:
    initialize_project(
        root,
        project_id="synthetic_backbone_bond",
        title="Synthetic backbone-bond candidate-to-release demonstration",
        claim_domain="synthetic mapping m1 and nonbonded parent nb1 at 310 K",
    )
    inputs = {
        "mapping.yaml": "mapping_id: synthetic_m1\nbeads: [B0, B1]\n",
        "topology_graph.yaml": "graph_id: synthetic_graph_v1\nedges: [[B0, B1]]\n",
        "bead_model.yaml": "bead_model_id: synthetic_beads_v1\ntypes: {B0: P1, B1: P1}\ncharges: {B0: 0, B1: 0}\n",
        "nonbonded_parent.yaml": "parent_id: synthetic_nb1\nmartini_generation: 3\n",
        "exclusions.yaml": "exclusion_id: synthetic_exclusions_v1\nrule: graph_distance_le_1\n",
    }
    for name, content in inputs.items():
        atomic_write_text(root / "inputs" / name, content)
    groups = {
        "development_groups.tsv": ["dev01", "dev02", "dev03", "dev04", "dev05", "dev06", "dev07", "dev08", "dev09"],
        "validation_groups.tsv": ["val01", "val02", "val03"],
        "stress_groups.tsv": ["length20", "length50", "dilute01", "material01"],
        "confirmation_groups.tsv": ["sealed01", "sealed02", "sealed03"],
    }
    for name, group_ids in groups.items():
        lines = ["group_id\tchemistry\tindependent_family\tsource\topened"]
        lines.extend(
            f"{group_id}\tX\t{group_id}\tsynthetic/{group_id}\tfalse"
            for group_id in group_ids
        )
        atomic_write_text(root / "data" / name, "\n".join(lines) + "\n")

    contract_path = root / "iterations" / "v001" / "contract.yaml"
    contract = load_yaml(contract_path)
    contract["design"] = {
        "coordinate": "synthetic B0-B1 backbone bond",
        "predecessor": "explicit omission",
        "candidate_ladder": ["omission", "harmonic"],
        "maximum_complexity": 1,
        "primary_objective": "family-balanced held-out bond-distance W1 improvement",
        "sensitivity_objectives": ["family-win count", "support-exit fraction"],
        "grouping_unit": "independent_start_family",
        "stop_rule": "weakest-link; two corrections maximum for one mechanism",
        "claim_ceiling": "synthetic tested-domain release; transfer only after E6",
    }
    for artifact in contract["scientific_identity"].values():
        artifact["placeholder"] = False
    for artifact in contract["data_groups"]:
        artifact["placeholder"] = False

    _set_numeric_rule(
        contract, "E0", "e0_run_integrity", "eq", 1.0,
        "Fraction of required tasks with exact completion and integrity markers.",
    )
    _set_numeric_rule(
        contract, "E1", "e1_rank_condition_support", "le", 100000.0,
        "Maximum registered design condition number after full-rank and support checks.",
    )
    _set_numeric_rule(
        contract, "E2", "e2_effect_necessity", "ge", 6,
        "Independent held-out families favoring the candidate over omission.",
    )
    _set_numeric_rule(
        contract, "E2", "e2_exact_selection_stability", "ge", 0.6666667,
        "Exact model identity selection frequency under grouped outer refitting.",
    )
    e2_physical = next(
        row for row in contract["gates"][2]["criteria"] if row["id"] == "e2_physical_admissibility"
    )
    e2_physical["operator"] = "truthy"
    e2_physical.pop("expected", None)
    _set_numeric_rule(
        contract, "E3", "e3_complete_manifest", "eq", 1.0,
        "Fraction of exact topology, force-consistency, minimization, and finite-MD tasks passing.",
    )
    _set_numeric_rule(
        contract, "E4", "e4_target_distribution", "gt", 0.0,
        "Family-balanced improvement over the frozen predecessor after support checks.",
    )
    _set_numeric_rule(
        contract, "E4", "e4_upstream_nonregression", "ge", 6,
        "Independent families noninferior for every frozen upstream observable.",
    )
    _set_numeric_rule(
        contract, "E4", "e4_complete_topology_replay", "eq", 1.0,
        "Fraction of required complete-topology replay tasks passing all original gates.",
    )
    _set_numeric_rule(
        contract, "E5", "e5_one_shot_confirmation", "eq", 1.0,
        "Fraction of genuinely unopened confirmation groups passing once without retuning.",
    )
    _set_numeric_rule(
        contract, "E6", "e6_transfer_qualification", "ge", 0.6666667,
        "Fraction of declared length, chain, solution, and material transfer groups passing.",
    )
    atomic_write_yaml(contract_path, contract)
    refresh_checksums(root, write=True)

    for gate, observations in PASS_VALUES.items():
        analysis_path = root / "artifacts" / f"{gate.lower()}_metrics.json"
        atomic_write_text(
            analysis_path,
            json.dumps({"gate": gate, "synthetic": True, "observations": observations}, indent=2) + "\n",
        )
        if gate == "E5":
            group_ids = ["confirmation_groups"]
        elif gate == "E6":
            group_ids = ["stress_groups"]
        else:
            group_ids = ["development_groups"]
        evidence = {
            "schema_version": "1.0",
            "project_id": "synthetic_backbone_bond",
            "iteration_id": "v001",
            "gate": gate,
            "evidence_id": f"synthetic_{gate.lower()}_v1",
            "data_group_ids": group_ids,
            "artifacts": [
                {
                    "id": f"{gate.lower()}_metrics",
                    "path": str(analysis_path.relative_to(root)),
                    "sha256": sha256_file(analysis_path),
                }
            ],
            "observations": observations,
            "notes": "Synthetic demonstration only; values are not scientific defaults.",
        }
        atomic_write_yaml(root / "evidence" / f"{gate}.yaml", evidence)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    create_demo(args.project_root.expanduser().resolve())
    print(args.project_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Built-in project skeletons for the evidence-gated protocol."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .schema import GATE_ORDER, SCHEMA_VERSION


PLACEHOLDER_SHA256 = "0" * 64


def protocol_template(project_id: str, title: str, claim_domain: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "id": project_id,
            "title": title,
            "claim_domain": claim_domain,
        },
        "active_iteration": "v001",
        "policy": {
            "gate_order": list(GATE_ORDER),
            "strict_sequence": True,
            "weakest_link": True,
            "e6_parameter_feedback": "prohibited",
            "max_correction_iterations_per_mechanism": 2,
        },
    }


def _artifact(artifact_id: str, path: str) -> Dict[str, Any]:
    return {
        "id": artifact_id,
        "path": path,
        "sha256": PLACEHOLDER_SHA256,
        "placeholder": True,
    }


def _criterion(
    criterion_id: str,
    description: str,
    *,
    operator: str = "truthy",
    expected: Any = None,
    on_fail: str,
    on_inconclusive: str = "DATA_LIMITED",
) -> Dict[str, Any]:
    criterion = {
        "id": criterion_id,
        "description": description,
        "operator": operator,
        "on_fail": on_fail,
        "on_inconclusive": on_inconclusive,
    }
    if operator not in ("truthy", "status"):
        criterion["expected"] = expected
    return criterion


def contract_template(project_id: str) -> Dict[str, Any]:
    """Return a conservative contract that must be customized before sealing."""

    gates = [
        {
            "id": "E0",
            "name": "provenance_and_run_integrity",
            "on_pass": "ADVANCE_TO_E1",
            "on_fail": "INELIGIBLE_EVIDENCE",
            "on_inconclusive": "INELIGIBLE_EVIDENCE",
            "criteria": [
                _criterion(
                    "e0_provenance_complete",
                    "Mapping, source, software, seed, manifest, and checksum provenance is complete.",
                    on_fail="INELIGIBLE_EVIDENCE",
                    on_inconclusive="INELIGIBLE_EVIDENCE",
                ),
                _criterion(
                    "e0_run_integrity",
                    "Every required task and output marker passes the frozen integrity check.",
                    on_fail="INELIGIBLE_EVIDENCE",
                    on_inconclusive="INELIGIBLE_EVIDENCE",
                ),
            ],
        },
        {
            "id": "E1",
            "name": "analytic_fit_identifiability_and_support",
            "on_pass": "ADVANCE_TO_E2",
            "on_fail": "NONRELEASE",
            "on_inconclusive": "DATA_LIMITED",
            "criteria": [
                _criterion(
                    "e1_graph_and_canonicalization",
                    "The term exists on the graph and symmetry/reversal canonicalization is correct.",
                    on_fail="REJECT_FUNCTION",
                ),
                _criterion(
                    "e1_rank_condition_support",
                    "The frozen rank, condition, leverage, and support criteria pass.",
                    operator="status",
                    on_fail="DATA_LIMITED",
                ),
            ],
        },
        {
            "id": "E2",
            "name": "grouped_prediction_and_selection",
            "on_pass": "ADVANCE_TO_E3",
            "on_fail": "NONRELEASE",
            "on_inconclusive": "SELECTION_LIMITED",
            "criteria": [
                _criterion(
                    "e2_effect_necessity",
                    "The registered candidate beats its explicit predecessor/omission under grouped prediction.",
                    operator="status",
                    on_fail="DEMOTE_TO_OMISSION",
                    on_inconclusive="DATA_LIMITED",
                ),
                _criterion(
                    "e2_exact_selection_stability",
                    "The complete registered selection is stable under the frozen outer grouped resampling.",
                    operator="status",
                    on_fail="SELECTION_LIMITED",
                    on_inconclusive="SELECTION_LIMITED",
                ),
                _criterion(
                    "e2_physical_admissibility",
                    "The coefficient region, support, curvature, boundary, and convention screens pass.",
                    operator="status",
                    on_fail="REJECT_FUNCTION",
                    on_inconclusive="DATA_LIMITED",
                ),
            ],
        },
        {
            "id": "E3",
            "name": "materialization_and_numerical_realization",
            "on_pass": "ADVANCE_TO_E4",
            "on_fail": "NUMERICAL_FAIL",
            "on_inconclusive": "NUMERICAL_INCONCLUSIVE",
            "criteria": [
                _criterion(
                    "e3_exact_topology",
                    "An independent verification confirms the exact graph, function, coefficients, units, and signs.",
                    on_fail="NUMERICAL_FAIL",
                    on_inconclusive="NUMERICAL_INCONCLUSIVE",
                ),
                _criterion(
                    "e3_complete_manifest",
                    "All frozen grompp, minimization, force-consistency, and finite-MD tasks pass.",
                    operator="status",
                    on_fail="NUMERICAL_FAIL",
                    on_inconclusive="NUMERICAL_INCONCLUSIVE",
                ),
            ],
        },
        {
            "id": "E4",
            "name": "realized_target_and_upstream_nonregression",
            "on_pass": "ADVANCE_TO_E5",
            "on_fail": "WEAKEST_LINK_FAIL_RETURN_TO_PREDECESSOR",
            "on_inconclusive": "REALIZATION_INCONCLUSIVE",
            "criteria": [
                _criterion(
                    "e4_target_distribution",
                    "Family-balanced target distribution, support-exit, and whole-family uncertainty pass.",
                    operator="status",
                    on_fail="REALIZATION_FAIL",
                    on_inconclusive="REALIZATION_INCONCLUSIVE",
                ),
                _criterion(
                    "e4_upstream_nonregression",
                    "All upstream bonded observables pass the frozen mean, family-count, and support non-regression gates.",
                    operator="status",
                    on_fail="WEAKEST_LINK_FAIL_RETURN_TO_PREDECESSOR",
                    on_inconclusive="REALIZATION_INCONCLUSIVE",
                ),
                _criterion(
                    "e4_complete_topology_replay",
                    "The single complete topology passes all original absolute gates under the frozen replay design.",
                    operator="status",
                    on_fail="COMPLETE_TOPOLOGY_NOT_RELEASED",
                    on_inconclusive="REALIZATION_INCONCLUSIVE",
                ),
            ],
        },
        {
            "id": "E5",
            "name": "unopened_one_shot_confirmation",
            "on_pass": "TESTED_DOMAIN_RELEASE",
            "on_fail": "EXTERNAL_VERIFICATION_FAIL_NONRELEASE",
            "on_inconclusive": "EXTERNAL_VERIFICATION_INCONCLUSIVE",
            "criteria": [
                _criterion(
                    "e5_one_shot_confirmation",
                    "The frozen topology passes once on genuinely unopened groups without retuning.",
                    operator="status",
                    on_fail="EXTERNAL_VERIFICATION_FAIL_NONRELEASE",
                    on_inconclusive="EXTERNAL_VERIFICATION_INCONCLUSIVE",
                )
            ],
        },
        {
            "id": "E6",
            "name": "length_chain_solution_and_material_transfer",
            "on_pass": "DOMAIN_QUALIFIED",
            "on_fail": "TRANSFER_BOUNDARY_OBSERVED",
            "on_inconclusive": "TRANSFER_INCONCLUSIVE",
            "criteria": [
                _criterion(
                    "e6_transfer_qualification",
                    "The frozen E5 parameters pass the declared length/single-chain/solution/material domain gates.",
                    operator="status",
                    on_fail="TRANSFER_BOUNDARY_OBSERVED",
                    on_inconclusive="TRANSFER_INCONCLUSIVE",
                ),
                _criterion(
                    "e6_no_parameter_feedback",
                    "No E6 observation was used to change an E5 coefficient, function, mapping, or nonbonded parent.",
                    on_fail="PROTOCOL_VIOLATION",
                    on_inconclusive="PROTOCOL_VIOLATION",
                ),
            ],
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "iteration_id": "v001",
        "parent": None,
        "iteration_class": "TYPE_I",
        "failure_mechanism_addressed": "INITIAL_REGISTERED_CANDIDATE_LADDER",
        "scientific_identity": {
            "mapping": _artifact("mapping", "inputs/mapping.yaml"),
            "topology_graph": _artifact("topology_graph", "inputs/topology_graph.yaml"),
            "bead_model": _artifact("bead_model", "inputs/bead_model.yaml"),
            "nonbonded_parent": _artifact("nonbonded_parent", "inputs/nonbonded_parent.yaml"),
            "exclusions": _artifact("exclusions", "inputs/exclusions.yaml"),
        },
        "data_groups": [
            {
                **_artifact("development_groups", "data/development_groups.tsv"),
                "role": "development",
                "sealed": False,
            },
            {
                **_artifact("validation_groups", "data/validation_groups.tsv"),
                "role": "validation",
                "sealed": False,
            },
            {
                **_artifact("stress_groups", "data/stress_groups.tsv"),
                "role": "stress",
                "sealed": False,
            },
            {
                **_artifact("confirmation_groups", "data/confirmation_groups.tsv"),
                "role": "confirmation",
                "sealed": True,
            },
        ],
        "design": {
            "coordinate": "REPLACE_WITH_BOND_ANGLE_DIHEDRAL_OR_COMPLETE_TOPOLOGY",
            "predecessor": "EXPLICIT_OMISSION_OR_FROZEN_UPSTREAM_TOPOLOGY",
            "candidate_ladder": ["omission", "REPLACE_WITH_REGISTERED_FUNCTION"],
            "maximum_complexity": 1,
            "primary_objective": "REPLACE_WITH_FAMILY_GROUPED_PRIMARY_OBJECTIVE",
            "sensitivity_objectives": ["REPLACE_WITH_REGISTERED_SENSITIVITY"],
            "grouping_unit": "independent_start_family",
            "stop_rule": "weakest_link; at most two corrections for one unchanged mechanism",
            "claim_ceiling": "tested-domain bonded-topology release; no universal transfer claim",
        },
        "gates": deepcopy(gates),
        "permitted_repairs": [
            "parser or sign correction with unchanged scientific identity",
            "runtime repair with identical frozen inputs, seeds, and task manifest",
        ],
        "prohibited_after_seal": [
            "mapping, graph, bead, nonbonded, or exclusion change",
            "candidate, function, grouping, objective, threshold, or stop-rule change",
            "data-role reassignment or premature confirmation access",
            "discarding an unfavorable required task or resample",
        ],
    }


PROJECT_README = """# Sealed bonded-parameter decision project

This directory is one scientific decision track.  It may represent one bond,
angle, dihedral block, or one complete-topology release.  It does not contain
universal Martini thresholds: the scientifically justified candidate ladder,
grouping unit, operators, and thresholds must be frozen in
`iterations/v001/contract.yaml` before evidence is opened.

## Required preparation

1. Replace every placeholder input and data-group manifest with the exact
   mapping, graph, bead/nonbonded/exclusion definitions, and role registry.
2. Edit the design and E0--E6 criteria.  Use numeric operators (`le`, `ge`,
   `between`) wherever a result can be evaluated directly; use `status` only
   when a separately verified analysis produces PASS/FAIL/INCONCLUSIVE.
3. Set every replaced artifact's `placeholder` field to `false`.
4. Run `hygel-parameter-protocol hash-inputs . --write`, then `validate .`.
5. Run `seal .`.  Any scientific-identity, data-role, candidate, objective,
   threshold, or stop-rule change after this point requires `new-iteration`.

## Evidence sequence

Copy `evidence_template.yaml` to a new file, enter the exact current gate and
all observations, attach checksummed analysis artifacts, and run:

```bash
hygel-parameter-protocol evaluate . evidence/my_E0.yaml --commit
hygel-parameter-protocol status .
```

Gates are cumulative and strict: E0 provenance, E1 analytic eligibility, E2
grouped selection, E3 numerical realization, E4 local realization plus
upstream non-regression, E5 unopened one-shot confirmation, and E6 transfer.
E6 never feeds coefficients back into the E5 release.

The ledger is hash chained.  Preserve failed and superseded artifacts.  Never
edit an existing ledger row; create a new protocol iteration instead.
"""


EVIDENCE_TEMPLATE = {
    "schema_version": SCHEMA_VERSION,
    "project_id": "REPLACE_PROJECT_ID",
    "iteration_id": "v001",
    "gate": "E0",
    "evidence_id": "REPLACE_UNIQUE_EVIDENCE_ID",
    "data_group_ids": ["development_groups"],
    "artifacts": [
        {
            "id": "analysis_output",
            "path": "evidence/REPLACE_ANALYSIS_OUTPUT.json",
            "sha256": PLACEHOLDER_SHA256,
        }
    ],
    "observations": {
        "e0_provenance_complete": True,
        "e0_run_integrity": True,
    },
    "notes": "Replace this template; results are computed against the sealed contract.",
}

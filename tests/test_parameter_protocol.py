from __future__ import annotations

from pathlib import Path

import pytest

from hygel_martini.param_opt.qm_to_martini.protocol.engine import (
    ProtocolError,
    evaluate_evidence,
    initialize_project,
    new_iteration,
    project_status,
    refresh_checksums,
    seal_iteration,
    validate_project,
)
from hygel_martini.param_opt.qm_to_martini.protocol.io import (
    atomic_write_text,
    atomic_write_yaml,
    load_yaml,
    sha256_file,
)
from hygel_martini.param_opt.qm_to_martini.protocol.schema import GATE_ORDER


def _accepted_project(tmp_path: Path) -> Path:
    root = tmp_path / "protocol_project"
    initialize_project(
        root,
        project_id="demo_term",
        title="Synthetic bonded-term protocol",
        claim_domain="unit-test chemistry and mapping only",
    )
    contract_path = root / "iterations" / "v001" / "contract.yaml"
    contract = load_yaml(contract_path)
    for path in sorted((root / "inputs").glob("*.yaml")):
        atomic_write_text(path, f"id: {path.stem}\nvalue: frozen-test-input\n")
    for path in sorted((root / "data").glob("*.tsv")):
        atomic_write_text(
            path,
            "group_id\tchemistry\tindependent_family\tsource\topened\n"
            f"{path.stem}\tX\tfamily_01\tsynthetic\tfalse\n",
        )
    for artifact in contract["scientific_identity"].values():
        artifact["placeholder"] = False
    for artifact in contract["data_groups"]:
        artifact["placeholder"] = False
    atomic_write_yaml(contract_path, contract)
    refresh_checksums(root, write=True)
    seal_iteration(root)
    return root


def _evidence(root: Path, gate_id: str, *, overrides=None) -> Path:
    contract = load_yaml(root / "iterations" / "v001" / "contract.yaml")
    gate = next(item for item in contract["gates"] if item["id"] == gate_id)
    observations = {}
    for rule in gate["criteria"]:
        if rule["operator"] == "status":
            observations[rule["id"]] = "PASS"
        elif rule["operator"] == "truthy":
            observations[rule["id"]] = True
        else:
            observations[rule["id"]] = rule["expected"]
    observations.update(overrides or {})
    artifact_path = root / "artifacts" / f"{gate_id.lower()}_analysis.json"
    atomic_write_text(artifact_path, '{"integrity": true}\n')
    if gate_id == "E5":
        data_groups = ["confirmation_groups"]
    elif gate_id == "E6":
        data_groups = ["stress_groups"]
    else:
        data_groups = ["development_groups"]
    payload = {
        "schema_version": "1.0",
        "project_id": "demo_term",
        "iteration_id": "v001",
        "gate": gate_id,
        "evidence_id": f"evidence_{gate_id.lower()}",
        "data_group_ids": data_groups,
        "artifacts": [
            {
                "id": f"artifact_{gate_id.lower()}",
                "path": str(artifact_path.relative_to(root)),
                "sha256": sha256_file(artifact_path),
            }
        ],
        "observations": observations,
        "notes": "synthetic test evidence",
    }
    evidence_path = root / "evidence" / f"{gate_id}.yaml"
    atomic_write_yaml(evidence_path, payload)
    return evidence_path


def test_full_e0_to_e6_release_is_replayable(tmp_path: Path) -> None:
    root = _accepted_project(tmp_path)

    for gate_id in GATE_ORDER:
        preview = evaluate_evidence(root, _evidence(root, gate_id), commit=False)
        assert preview["result"] == "PASS"
        assert preview["committed"] is False
        committed = evaluate_evidence(root, root / "evidence" / f"{gate_id}.yaml", commit=True)
        assert committed["result"] == "PASS"

    status = project_status(root)
    validation = validate_project(root)
    assert status["state"] == "DOMAIN_QUALIFIED"
    assert status["completed_gates"] == list(GATE_ORDER)
    assert status["opened_confirmation_group_ids"] == ["confirmation_groups"]
    assert validation["decision"] == "PASS"
    assert validation["ledger_rows"] == 9  # initialization, seal, and seven decisions


def test_strict_sequence_and_sealed_confirmation_boundary(tmp_path: Path) -> None:
    root = _accepted_project(tmp_path)
    evidence = load_yaml(_evidence(root, "E1"))
    evidence["gate"] = "E0"
    evidence["data_group_ids"] = ["confirmation_groups"]
    evidence["observations"] = {
        "e0_provenance_complete": True,
        "e0_run_integrity": True,
    }
    evidence_path = root / "evidence" / "invalid_e0.yaml"
    atomic_write_yaml(evidence_path, evidence)

    with pytest.raises(ProtocolError, match="may not access sealed confirmation"):
        evaluate_evidence(root, evidence_path)

    wrong_gate = _evidence(root, "E1")
    with pytest.raises(ProtocolError, match="strict sequence requires gate E0"):
        evaluate_evidence(root, wrong_gate)


def test_weakest_link_nonpass_is_terminal(tmp_path: Path) -> None:
    root = _accepted_project(tmp_path)
    evaluate_evidence(root, _evidence(root, "E0"), commit=True)
    failing = _evidence(
        root,
        "E1",
        overrides={"e1_rank_condition_support": "FAIL"},
    )
    decision = evaluate_evidence(root, failing, commit=True)

    assert decision["result"] == "FAIL"
    assert decision["action"] == "NONRELEASE"
    failed_rule = next(row for row in decision["criteria"] if row["result"] == "FAIL")
    assert failed_rule["nonpass_action"] == "DATA_LIMITED"
    assert project_status(root)["state"] == "TERMINAL_NONPASS"
    with pytest.raises(ProtocolError, match="non-pass terminal"):
        evaluate_evidence(root, _evidence(root, "E2"), commit=True)


def test_numeric_threshold_is_taken_only_from_sealed_contract(tmp_path: Path) -> None:
    root = tmp_path / "numeric_project"
    initialize_project(
        root,
        project_id="numeric_demo",
        title="Numeric gate test",
        claim_domain="test only",
    )
    contract_path = root / "iterations" / "v001" / "contract.yaml"
    contract = load_yaml(contract_path)
    e0_rule = contract["gates"][0]["criteria"][0]
    e0_rule["operator"] = "ge"
    e0_rule["expected"] = 6
    for path in sorted((root / "inputs").glob("*.yaml")):
        atomic_write_text(path, f"id: {path.stem}\n")
    for path in sorted((root / "data").glob("*.tsv")):
        atomic_write_text(path, f"group_id\n{path.stem}\n")
    for artifact in contract["scientific_identity"].values():
        artifact["placeholder"] = False
    for artifact in contract["data_groups"]:
        artifact["placeholder"] = False
    atomic_write_yaml(contract_path, contract)
    refresh_checksums(root, write=True)
    seal_iteration(root)
    evidence = _evidence(root, "E0", overrides={"e0_provenance_complete": 5})
    payload = load_yaml(evidence)
    payload["project_id"] = "numeric_demo"
    atomic_write_yaml(evidence, payload)

    result = evaluate_evidence(root, evidence)
    assert result["result"] == "FAIL"
    assert result["criteria"][0]["observed"] == 5


def test_artifact_tampering_invalidates_project_and_blocks_evaluation(tmp_path: Path) -> None:
    root = _accepted_project(tmp_path)
    atomic_write_text(root / "inputs" / "mapping.yaml", "id: tampered\n")

    validation = validate_project(root)
    assert validation["decision"] == "FAIL"
    assert any("checksum mismatch" in error for error in validation["errors"])
    with pytest.raises(ProtocolError, match="contract/artifact validation failed"):
        evaluate_evidence(root, _evidence(root, "E0"))


def test_opened_confirmation_is_reclassified_in_new_iteration(tmp_path: Path) -> None:
    root = _accepted_project(tmp_path)
    for gate_id in GATE_ORDER[:5]:
        evaluate_evidence(root, _evidence(root, gate_id), commit=True)
    failed_e5 = _evidence(
        root,
        "E5",
        overrides={"e5_one_shot_confirmation": "FAIL"},
    )
    evaluate_evidence(root, failed_e5, commit=True)

    result = new_iteration(
        root,
        new_iteration_id="v002",
        iteration_class="TYPE_IV",
        failure_mechanism="EXTERNAL_DOMAIN_FAILURE",
    )
    contract = load_yaml(root / "iterations" / "v002" / "contract.yaml")
    former_confirmation = next(
        row for row in contract["data_groups"] if row["id"] == "confirmation_groups"
    )
    assert result["reclassified_opened_confirmation_ids"] == ["confirmation_groups"]
    assert former_confirmation["role"] == "development"
    assert former_confirmation["sealed"] is False
    assert validate_project(root)["decision"] == "FAIL"  # a fresh sealed E5 group is mandatory


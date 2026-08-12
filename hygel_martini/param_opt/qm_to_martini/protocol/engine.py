"""State engine for a sealed, weakest-link parameterization protocol."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .io import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    canonical_json_bytes,
    load_yaml,
    safe_project_path,
    sha256_bytes,
    sha256_file,
)
from .ledger import append_event, events_for_iteration, validate_ledger
from .schema import (
    GATE_ORDER,
    ID_RE,
    ITERATION_CLASSES,
    SCHEMA_VERSION,
    SHA256_RE,
    contract_artifacts,
    evaluate_rule,
    load_project_documents,
    scientific_identity_hash,
    validate_contract_document,
    validate_protocol_document,
)
from .templates import EVIDENCE_TEMPLATE, PROJECT_README, contract_template, protocol_template


class ProtocolError(RuntimeError):
    """Raised when an operation would violate the frozen protocol."""


def _root(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _iteration_dir(root: Path, iteration_id: str) -> Path:
    return root / "iterations" / iteration_id


def _seal_path(root: Path, iteration_id: str) -> Path:
    return _iteration_dir(root, iteration_id) / "seal.json"


def _contract_hash(contract: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(contract))


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"JSON document must be an object: {path}")
    return payload


def _placeholder_text(name: str) -> str:
    return (
        f"# PLACEHOLDER: replace {name} with the exact frozen input.\n"
        "# Then set placeholder: false in the contract and run hash-inputs.\n"
    )


def initialize_project(
    project_root: Path,
    *,
    project_id: str,
    title: str,
    claim_domain: str,
) -> Dict[str, Any]:
    """Create a non-overwriting project skeleton."""

    root = _root(project_root)
    if not ID_RE.fullmatch(project_id):
        raise ProtocolError("project_id must contain only letters, digits, dot, underscore, or hyphen")
    if not title.strip() or not claim_domain.strip():
        raise ProtocolError("title and claim_domain must be non-empty")
    if root.exists() and any(root.iterdir()):
        raise ProtocolError(f"refusing to initialize a non-empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("inputs", "data", "evidence", "artifacts", "iterations/v001/decisions"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    protocol = protocol_template(project_id, title.strip(), claim_domain.strip())
    contract = contract_template(project_id)
    evidence = deepcopy(EVIDENCE_TEMPLATE)
    evidence["project_id"] = project_id
    atomic_write_yaml(root / "protocol.yaml", protocol)
    atomic_write_yaml(_iteration_dir(root, "v001") / "contract.yaml", contract)
    atomic_write_yaml(root / "evidence_template.yaml", evidence)
    atomic_write_text(root / "README.md", PROJECT_README)
    for name in ("mapping", "topology_graph", "bead_model", "nonbonded_parent", "exclusions"):
        atomic_write_text(root / "inputs" / f"{name}.yaml", _placeholder_text(name))
    header = "group_id\tchemistry\tindependent_family\tsource\topened\n"
    for role in ("development", "validation", "stress", "confirmation"):
        atomic_write_text(
            root / "data" / f"{role}_groups.tsv",
            header + f"REPLACE_{role.upper()}\tREPLACE\tREPLACE\tREPLACE\tfalse\n",
        )
    atomic_write_text(root / "ledger.jsonl", "")
    event = append_event(
        root / "ledger.jsonl",
        event_type="PROJECT_INITIALIZED",
        iteration_id="v001",
        payload={"project_id": project_id, "schema_version": SCHEMA_VERSION},
    )
    return {
        "decision": "INITIALIZED",
        "project_root": str(root),
        "project_id": project_id,
        "active_iteration": "v001",
        "ledger_event_hash": event["event_hash"],
        "next_action": "replace placeholders and freeze the prospective contract",
    }


def _artifact_rows(contract: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return list(contract_artifacts(contract))


def refresh_checksums(
    project_root: Path,
    *,
    iteration_id: Optional[str] = None,
    write: bool = False,
) -> Dict[str, Any]:
    """Compute checksums for explicitly accepted, project-local frozen inputs."""

    root = _root(project_root)
    protocol = load_yaml(root / "protocol.yaml")
    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol.yaml must contain a mapping")
    iteration = iteration_id or str(protocol.get("active_iteration", ""))
    protocol, contract = load_project_documents(root, iteration)
    if _seal_path(root, iteration).exists() and write:
        raise ProtocolError("refusing to modify checksums after the iteration was sealed")
    updated = deepcopy(contract)
    rows = _artifact_rows(updated)
    results: List[Dict[str, Any]] = []
    for artifact in rows:
        artifact_id = str(artifact.get("id", ""))
        if artifact.get("placeholder", False):
            raise ProtocolError(
                f"artifact {artifact_id!r} is still marked placeholder; replace it and set placeholder: false"
            )
        relative = str(artifact.get("path", ""))
        try:
            path = safe_project_path(root, relative)
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        if not path.is_file():
            raise ProtocolError(f"artifact does not exist: {relative}")
        digest = sha256_file(path)
        if write:
            artifact["sha256"] = digest
        results.append({"id": artifact_id, "path": relative, "sha256": digest})
    if write:
        atomic_write_yaml(_iteration_dir(root, iteration) / "contract.yaml", updated)
    return {
        "decision": "CHECKSUMS_WRITTEN" if write else "CHECKSUMS_COMPUTED",
        "iteration_id": iteration,
        "artifact_count": len(results),
        "artifacts": results,
    }


def _normalized_data_groups(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    normalized = []
    for group in contract.get("data_groups", []):
        if isinstance(group, Mapping):
            normalized.append(
                {
                    "id": group.get("id"),
                    "path": group.get("path"),
                    "sha256": group.get("sha256"),
                    "role": group.get("role"),
                    "sealed": group.get("sealed", False),
                }
            )
    return normalized


def _validate_transition(root: Path, contract: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    parent = contract.get("parent")
    if not isinstance(parent, Mapping):
        return errors
    parent_id = str(parent.get("iteration_id", ""))
    parent_contract_path = _iteration_dir(root, parent_id) / "contract.yaml"
    if not parent_contract_path.is_file():
        return [f"parent contract does not exist: {parent_id!r}"]
    parent_contract = load_yaml(parent_contract_path)
    if not isinstance(parent_contract, Mapping):
        return [f"parent contract is not a mapping: {parent_id!r}"]
    iteration_class = contract.get("iteration_class")
    current_identity = contract.get("scientific_identity")
    parent_identity = parent_contract.get("scientific_identity")
    if iteration_class in ("TYPE_I", "TYPE_II", "TYPE_III") and current_identity != parent_identity:
        errors.append(
            f"{iteration_class} must preserve mapping, graph, bead, nonbonded, and exclusion identity; use TYPE_IV"
        )
    if iteration_class == "TYPE_I":
        for key in ("design", "gates"):
            if contract.get(key) != parent_contract.get(key):
                errors.append(f"TYPE_I predeclared continuation must preserve parent {key}")
    if iteration_class == "TYPE_II":
        for key in ("design", "gates"):
            if contract.get(key) != parent_contract.get(key):
                errors.append(f"TYPE_II implementation repair must preserve parent {key}")
        if _normalized_data_groups(contract) != _normalized_data_groups(parent_contract):
            errors.append("TYPE_II implementation repair must preserve data manifests and roles")
    return errors


def _verify_seal(
    root: Path,
    iteration_id: str,
    contract: Mapping[str, Any],
    records: Optional[List[Mapping[str, Any]]] = None,
) -> Tuple[Optional[Mapping[str, Any]], List[str]]:
    errors: List[str] = []
    path = _seal_path(root, iteration_id)
    if not path.is_file():
        return None, ["seal.json is absent"]
    try:
        seal = _load_json(path)
    except ProtocolError as error:
        return None, [str(error)]
    observed_contract_hash = _contract_hash(contract)
    if seal.get("contract_sha256") != observed_contract_hash:
        errors.append("sealed contract hash does not match current contract.yaml")
    observed_identity_hash = scientific_identity_hash(contract)
    if seal.get("scientific_identity_sha256") != observed_identity_hash:
        errors.append("sealed scientific identity hash does not match current contract")
    sealed_artifacts = seal.get("artifacts")
    if not isinstance(sealed_artifacts, list):
        errors.append("seal artifacts must be a list")
    else:
        current_artifacts = [
            {"id": row.get("id"), "path": row.get("path"), "sha256": row.get("sha256")}
            for row in _artifact_rows(contract)
        ]
        if sealed_artifacts != current_artifacts:
            errors.append("sealed artifact registry does not match the contract")
    if records is not None:
        matching = [
            row
            for row in records
            if row.get("iteration_id") == iteration_id
            and row.get("event_type") == "ITERATION_SEALED"
            and isinstance(row.get("payload"), Mapping)
            and row["payload"].get("contract_sha256") == observed_contract_hash
        ]
        if len(matching) != 1:
            errors.append("ledger must contain exactly one matching ITERATION_SEALED event")
    return seal, errors


def seal_iteration(
    project_root: Path, *, iteration_id: Optional[str] = None
) -> Dict[str, Any]:
    """Freeze a complete prospective contract and its referenced artifacts."""

    root = _root(project_root)
    protocol = load_yaml(root / "protocol.yaml")
    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol.yaml must contain a mapping")
    iteration = iteration_id or str(protocol.get("active_iteration", ""))
    protocol, contract = load_project_documents(root, iteration)
    protocol_errors = validate_protocol_document(protocol)
    contract_errors, warnings = validate_contract_document(
        contract, protocol, root, iteration, check_files=True
    )
    transition_errors = _validate_transition(root, contract)
    errors = protocol_errors + contract_errors + transition_errors
    placeholders = [warning for warning in warnings if "placeholder is true" in warning]
    if placeholders:
        errors.extend(item.replace("placeholder is true; sealing is prohibited", "placeholder must be replaced before sealing") for item in placeholders)
    if errors:
        raise ProtocolError("cannot seal invalid iteration:\n- " + "\n- ".join(errors))
    records, ledger_errors = validate_ledger(root / "ledger.jsonl")
    if ledger_errors:
        raise ProtocolError("cannot seal with invalid ledger:\n- " + "\n- ".join(ledger_errors))
    existing, existing_errors = _verify_seal(root, iteration, contract, records)
    if existing is not None:
        if existing_errors:
            raise ProtocolError("existing seal is invalid:\n- " + "\n- ".join(existing_errors))
        return {**dict(existing), "decision": "ALREADY_SEALED"}

    artifacts = [
        {"id": row.get("id"), "path": row.get("path"), "sha256": row.get("sha256")}
        for row in _artifact_rows(contract)
    ]
    seal: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_id": protocol["project"]["id"],
        "iteration_id": iteration,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_sha256": _contract_hash(contract),
        "scientific_identity_sha256": scientific_identity_hash(contract),
        "artifacts": artifacts,
        "sealed_confirmation_group_ids": [
            row["id"]
            for row in contract.get("data_groups", [])
            if row.get("role") == "confirmation" and row.get("sealed") is True
        ],
    }
    atomic_write_json(_seal_path(root, iteration), seal)
    event = append_event(
        root / "ledger.jsonl",
        event_type="ITERATION_SEALED",
        iteration_id=iteration,
        payload={
            "contract_sha256": seal["contract_sha256"],
            "scientific_identity_sha256": seal["scientific_identity_sha256"],
            "sealed_confirmation_group_ids": seal["sealed_confirmation_group_ids"],
        },
    )
    return {**seal, "decision": "SEALED", "ledger_event_hash": event["event_hash"]}


def _decision_events(records: Sequence[Mapping[str, Any]], iteration_id: str) -> List[Mapping[str, Any]]:
    return [
        row
        for row in records
        if row.get("iteration_id") == iteration_id and row.get("event_type") == "DECISION"
    ]


def _validate_evidence_artifacts(root: Path, artifacts: Any) -> List[Dict[str, Any]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ProtocolError("evidence.artifacts must be a non-empty list")
    normalized: List[Dict[str, Any]] = []
    ids = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ProtocolError(f"evidence.artifacts[{index}] must be a mapping")
        artifact_id = str(artifact.get("id", ""))
        relative = str(artifact.get("path", ""))
        checksum = str(artifact.get("sha256", ""))
        if not ID_RE.fullmatch(artifact_id) or artifact_id in ids:
            raise ProtocolError(f"invalid or duplicate evidence artifact id: {artifact_id!r}")
        ids.add(artifact_id)
        if not SHA256_RE.fullmatch(checksum):
            raise ProtocolError(f"invalid evidence artifact SHA-256 for {artifact_id!r}")
        try:
            path = safe_project_path(root, relative)
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        if not path.is_file():
            raise ProtocolError(f"evidence artifact does not exist: {relative}")
        observed = sha256_file(path)
        if observed != checksum:
            raise ProtocolError(
                f"evidence artifact checksum mismatch for {artifact_id!r}: expected {checksum}, observed {observed}"
            )
        normalized.append({"id": artifact_id, "path": relative, "sha256": checksum})
    return normalized


def evaluate_evidence(
    project_root: Path,
    evidence_path: Path,
    *,
    commit: bool = False,
) -> Dict[str, Any]:
    """Evaluate one gate against a sealed contract; mutate only with ``commit``."""

    root = _root(project_root)
    evidence_file = Path(evidence_path).expanduser().resolve()
    try:
        evidence_relative = str(evidence_file.relative_to(root))
    except ValueError as error:
        raise ProtocolError("evidence file must be stored inside the project root") from error
    evidence = load_yaml(evidence_file)
    if not isinstance(evidence, Mapping):
        raise ProtocolError("evidence document must be a mapping")
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(f"evidence.schema_version must equal {SCHEMA_VERSION!r}")
    protocol = load_yaml(root / "protocol.yaml")
    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol.yaml must contain a mapping")
    project_id = protocol.get("project", {}).get("id")
    if evidence.get("project_id") != project_id:
        raise ProtocolError(f"evidence.project_id must equal {project_id!r}")
    iteration = str(evidence.get("iteration_id", ""))
    if not ID_RE.fullmatch(iteration):
        raise ProtocolError("evidence.iteration_id is invalid")
    protocol, contract = load_project_documents(root, iteration)
    contract_errors, contract_warnings = validate_contract_document(
        contract, protocol, root, iteration, check_files=True
    )
    if contract_errors or contract_warnings:
        problems = contract_errors + contract_warnings
        raise ProtocolError(
            "sealed contract/artifact validation failed:\n- " + "\n- ".join(problems)
        )
    records, ledger_errors = validate_ledger(root / "ledger.jsonl")
    if ledger_errors:
        raise ProtocolError("ledger validation failed:\n- " + "\n- ".join(ledger_errors))
    seal, seal_errors = _verify_seal(root, iteration, contract, records)
    if seal is None or seal_errors:
        raise ProtocolError("iteration seal validation failed:\n- " + "\n- ".join(seal_errors))

    evidence_id = str(evidence.get("evidence_id", ""))
    if not ID_RE.fullmatch(evidence_id):
        raise ProtocolError("evidence.evidence_id is invalid")
    used_evidence_ids = {
        row.get("payload", {}).get("evidence_id")
        for row in records
        if isinstance(row.get("payload"), Mapping)
    }
    if evidence_id in used_evidence_ids:
        raise ProtocolError(f"evidence_id has already been committed: {evidence_id!r}")

    prior_decisions = _decision_events(records, iteration)
    for row in prior_decisions:
        if row.get("payload", {}).get("result") != "PASS":
            raise ProtocolError("iteration already reached a non-pass terminal; create a new iteration")
    if len(prior_decisions) >= len(GATE_ORDER):
        raise ProtocolError("all E0-E6 gates have already been evaluated")
    expected_gate = GATE_ORDER[len(prior_decisions)]
    gate_id = evidence.get("gate")
    if gate_id != expected_gate:
        raise ProtocolError(f"strict sequence requires gate {expected_gate}, not {gate_id!r}")
    gate = contract["gates"][len(prior_decisions)]

    groups_by_id = {
        row["id"]: row for row in contract.get("data_groups", []) if isinstance(row, Mapping)
    }
    data_group_ids = evidence.get("data_group_ids")
    if not isinstance(data_group_ids, list) or not data_group_ids:
        raise ProtocolError("evidence.data_group_ids must be a non-empty list")
    if len(set(data_group_ids)) != len(data_group_ids):
        raise ProtocolError("evidence.data_group_ids contains duplicates")
    unknown_groups = sorted(set(data_group_ids) - set(groups_by_id))
    if unknown_groups:
        raise ProtocolError(f"evidence references unknown data groups: {unknown_groups}")
    selected_groups = [groups_by_id[group_id] for group_id in data_group_ids]
    if gate_id in GATE_ORDER[:5]:
        forbidden = [row["id"] for row in selected_groups if row.get("sealed") is True]
        if forbidden:
            raise ProtocolError(f"E0-E4 may not access sealed confirmation groups: {forbidden}")
    if gate_id == "E5":
        invalid = [
            row["id"]
            for row in selected_groups
            if row.get("role") != "confirmation" or row.get("sealed") is not True
        ]
        if invalid:
            raise ProtocolError(f"E5 may use only sealed confirmation groups: {invalid}")
    if gate_id == "E6" and not any(row.get("role") == "stress" for row in selected_groups):
        raise ProtocolError("E6 must include at least one stress/transfer data group")

    observations = evidence.get("observations")
    if not isinstance(observations, Mapping):
        raise ProtocolError("evidence.observations must be a mapping")
    rules = {rule["id"]: rule for rule in gate["criteria"]}
    missing = sorted(set(rules) - set(observations))
    extra = sorted(set(observations) - set(rules))
    if missing or extra:
        raise ProtocolError(f"observation ids must exactly match the frozen criteria; missing={missing}, extra={extra}")
    criterion_results: List[Dict[str, Any]] = []
    for criterion_id, rule in rules.items():
        try:
            result = evaluate_rule(rule, observations[criterion_id])
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"invalid observation for {criterion_id!r}: {error}") from error
        action_key = "on_fail" if result == "FAIL" else "on_inconclusive"
        criterion_results.append(
            {
                "id": criterion_id,
                "observed": observations[criterion_id],
                "result": result,
                "nonpass_action": None if result == "PASS" else rule[action_key],
            }
        )
    results = [row["result"] for row in criterion_results]
    if "FAIL" in results:
        aggregate = "FAIL"
        action = gate["on_fail"]
    elif "INCONCLUSIVE" in results:
        aggregate = "INCONCLUSIVE"
        action = gate["on_inconclusive"]
    else:
        aggregate = "PASS"
        action = gate["on_pass"]

    artifacts = _validate_evidence_artifacts(root, evidence.get("artifacts"))
    evidence_sha256 = sha256_file(evidence_file)
    claim_ceiling = contract["design"]["claim_ceiling"]
    if aggregate == "PASS" and gate_id not in ("E5", "E6"):
        next_permitted = f"evaluate {GATE_ORDER[len(prior_decisions) + 1]} under the same seal"
    elif aggregate == "PASS" and gate_id == "E5":
        next_permitted = "use within the tested domain or evaluate E6 without coefficient feedback"
    elif aggregate == "PASS" and gate_id == "E6":
        next_permitted = "use only within the E6-qualified domain"
    else:
        next_permitted = "close this iteration or create a prospectively frozen new iteration"
    decision: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "iteration_id": iteration,
        "gate": gate_id,
        "gate_name": gate["name"],
        "evidence_id": evidence_id,
        "evidence_path": evidence_relative,
        "evidence_sha256": evidence_sha256,
        "contract_sha256": seal["contract_sha256"],
        "scientific_identity_sha256": seal["scientific_identity_sha256"],
        "data_group_ids": list(data_group_ids),
        "artifacts": artifacts,
        "criteria": criterion_results,
        "result": aggregate,
        "action": action,
        "claim_ceiling": claim_ceiling,
        "next_permitted_action": next_permitted,
        "confirmation_opened": gate_id == "E5",
        "notes": str(evidence.get("notes", "")),
        "committed": commit,
    }
    try:
        canonical_json_bytes(decision)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"evidence contains a non-JSON-compatible value: {error}") from error
    if commit:
        ledger_payload = dict(decision)
        ledger_payload.pop("committed", None)
        event = append_event(
            root / "ledger.jsonl",
            event_type="DECISION",
            iteration_id=iteration,
            payload=ledger_payload,
        )
        decision["ledger_event_hash"] = event["event_hash"]
        decision_path = _iteration_dir(root, iteration) / "decisions" / f"{gate_id}_{evidence_id}.json"
        atomic_write_json(decision_path, decision)
        decision["decision_path"] = str(decision_path.relative_to(root))
    return decision


def _validate_ledger_semantics(
    root: Path,
    records: List[Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    errors: List[str] = []
    for iteration_id, contract in contracts.items():
        iteration_events = events_for_iteration(records, iteration_id)
        seal_positions = [
            index for index, row in enumerate(iteration_events) if row.get("event_type") == "ITERATION_SEALED"
        ]
        decisions = [row for row in iteration_events if row.get("event_type") == "DECISION"]
        if decisions and len(seal_positions) != 1:
            errors.append(f"{iteration_id}: decisions require exactly one prior seal event")
        observed_gates: List[Any] = []
        terminal_seen = False
        for row in decisions:
            payload = row.get("payload", {})
            gate = payload.get("gate") if isinstance(payload, Mapping) else None
            observed_gates.append(gate)
            if terminal_seen:
                errors.append(f"{iteration_id}: decision exists after a non-pass terminal")
            if isinstance(payload, Mapping) and payload.get("result") != "PASS":
                terminal_seen = True
            if isinstance(payload, Mapping):
                if payload.get("contract_sha256") != _contract_hash(contract):
                    errors.append(f"{iteration_id}/{gate}: decision contract hash mismatch")
                evidence_relative = payload.get("evidence_path")
                evidence_hash = payload.get("evidence_sha256")
                if isinstance(evidence_relative, str):
                    try:
                        evidence_path = safe_project_path(root, evidence_relative)
                    except ValueError as error:
                        errors.append(f"{iteration_id}/{gate}: {error}")
                    else:
                        if not evidence_path.is_file():
                            errors.append(f"{iteration_id}/{gate}: evidence file is absent")
                        elif sha256_file(evidence_path) != evidence_hash:
                            errors.append(f"{iteration_id}/{gate}: evidence file changed after commit")
                for artifact in payload.get("artifacts", []):
                    if not isinstance(artifact, Mapping):
                        continue
                    try:
                        artifact_path = safe_project_path(root, str(artifact.get("path", "")))
                    except ValueError as error:
                        errors.append(f"{iteration_id}/{gate}: {error}")
                        continue
                    if not artifact_path.is_file():
                        errors.append(f"{iteration_id}/{gate}: evidence artifact is absent: {artifact.get('path')}")
                    elif sha256_file(artifact_path) != artifact.get("sha256"):
                        errors.append(f"{iteration_id}/{gate}: evidence artifact changed: {artifact.get('path')}")
        if observed_gates != list(GATE_ORDER[: len(observed_gates)]):
            errors.append(f"{iteration_id}: decision gates violate strict sequence: {observed_gates}")
    return errors


def validate_project(project_root: Path) -> Dict[str, Any]:
    """Validate schemas, artifacts, seals, ledger hashes, and state semantics."""

    root = _root(project_root)
    errors: List[str] = []
    warnings: List[str] = []
    protocol_path = root / "protocol.yaml"
    if not protocol_path.is_file():
        return {"decision": "FAIL", "project_root": str(root), "errors": ["protocol.yaml is absent"], "warnings": []}
    try:
        protocol = load_yaml(protocol_path)
    except (OSError, ValueError) as error:
        return {"decision": "FAIL", "project_root": str(root), "errors": [str(error)], "warnings": []}
    errors.extend(validate_protocol_document(protocol))
    iteration_root = root / "iterations"
    iteration_ids = sorted(
        path.name for path in iteration_root.iterdir() if path.is_dir() and (path / "contract.yaml").is_file()
    ) if iteration_root.is_dir() else []
    if not iteration_ids:
        errors.append("no iteration contract exists")
    if protocol.get("active_iteration") not in iteration_ids:
        errors.append("protocol.active_iteration does not name an existing iteration")
    records, ledger_errors = validate_ledger(root / "ledger.jsonl")
    errors.extend(ledger_errors)
    contracts: Dict[str, Mapping[str, Any]] = {}
    iteration_reports: List[Dict[str, Any]] = []
    for iteration_id in iteration_ids:
        try:
            contract = load_yaml(_iteration_dir(root, iteration_id) / "contract.yaml")
        except (OSError, ValueError) as error:
            errors.append(f"{iteration_id}: {error}")
            continue
        contracts[iteration_id] = contract
        contract_errors, contract_warnings = validate_contract_document(
            contract, protocol, root, iteration_id, check_files=True
        )
        transition_errors = _validate_transition(root, contract)
        errors.extend(f"{iteration_id}: {item}" for item in contract_errors + transition_errors)
        warnings.extend(f"{iteration_id}: {item}" for item in contract_warnings)
        seal, seal_errors = _verify_seal(root, iteration_id, contract, records)
        if seal is None:
            if _decision_events(records, iteration_id):
                errors.append(f"{iteration_id}: decisions exist without seal.json")
            seal_state = "DRAFT"
        else:
            errors.extend(f"{iteration_id}: {item}" for item in seal_errors)
            seal_state = "SEALED" if not seal_errors else "INVALID_SEAL"
            if contract_warnings:
                errors.append(f"{iteration_id}: a sealed contract contains placeholder artifacts")
        iteration_reports.append(
            {
                "iteration_id": iteration_id,
                "seal_state": seal_state,
                "decision_count": len(_decision_events(records, iteration_id)),
                "contract_sha256": _contract_hash(contract),
            }
        )
    errors.extend(_validate_ledger_semantics(root, records, contracts))
    return {
        "decision": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "project_id": protocol.get("project", {}).get("id"),
        "active_iteration": protocol.get("active_iteration"),
        "iterations": iteration_reports,
        "ledger_rows": len(records),
        "errors": errors,
        "warnings": warnings,
    }


def project_status(project_root: Path) -> Dict[str, Any]:
    """Report the exact current gate and claim ceiling without changing state."""

    root = _root(project_root)
    validation = validate_project(root)
    protocol = load_yaml(root / "protocol.yaml")
    iteration = str(protocol.get("active_iteration", ""))
    contract = load_yaml(_iteration_dir(root, iteration) / "contract.yaml")
    records, _ = validate_ledger(root / "ledger.jsonl")
    decisions = _decision_events(records, iteration)
    if not _seal_path(root, iteration).is_file():
        state = "DRAFT"
        next_gate = "E0"
        terminal = None
    elif not decisions:
        state = "SEALED_AWAITING_E0"
        next_gate = "E0"
        terminal = None
    else:
        payload = decisions[-1]["payload"]
        if payload["result"] != "PASS":
            state = "TERMINAL_NONPASS"
            next_gate = None
            terminal = payload["action"]
        elif payload["gate"] == "E6":
            state = "DOMAIN_QUALIFIED"
            next_gate = None
            terminal = payload["action"]
        elif payload["gate"] == "E5":
            state = "TESTED_DOMAIN_RELEASE"
            next_gate = "E6"
            terminal = payload["action"]
        else:
            index = GATE_ORDER.index(payload["gate"])
            next_gate = GATE_ORDER[index + 1]
            state = f"AWAITING_{next_gate}"
            terminal = None
    opened_confirmation = []
    for row in decisions:
        payload = row.get("payload", {})
        if payload.get("confirmation_opened"):
            opened_confirmation.extend(payload.get("data_group_ids", []))
    return {
        "decision": validation["decision"],
        "project_id": protocol.get("project", {}).get("id"),
        "active_iteration": iteration,
        "state": state,
        "next_gate": next_gate,
        "terminal": terminal,
        "completed_gates": [row["payload"]["gate"] for row in decisions],
        "opened_confirmation_group_ids": sorted(set(opened_confirmation)),
        "claim_ceiling": contract.get("design", {}).get("claim_ceiling"),
        "validation_error_count": len(validation["errors"]),
        "validation_warning_count": len(validation["warnings"]),
    }


def new_iteration(
    project_root: Path,
    *,
    new_iteration_id: str,
    iteration_class: str,
    failure_mechanism: str,
    from_iteration_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fork a closed non-pass iteration without rewriting its terminal."""

    root = _root(project_root)
    if not ID_RE.fullmatch(new_iteration_id):
        raise ProtocolError("new iteration id is invalid")
    if iteration_class not in ITERATION_CLASSES:
        raise ProtocolError(f"iteration_class must be one of {list(ITERATION_CLASSES)!r}")
    if not failure_mechanism.strip():
        raise ProtocolError("failure_mechanism must be non-empty")
    target_dir = _iteration_dir(root, new_iteration_id)
    if target_dir.exists():
        raise ProtocolError(f"target iteration already exists: {new_iteration_id}")
    protocol = load_yaml(root / "protocol.yaml")
    source_iteration = from_iteration_id or str(protocol.get("active_iteration", ""))
    source_contract = load_yaml(_iteration_dir(root, source_iteration) / "contract.yaml")
    records, ledger_errors = validate_ledger(root / "ledger.jsonl")
    if ledger_errors:
        raise ProtocolError("ledger validation failed:\n- " + "\n- ".join(ledger_errors))
    decisions = _decision_events(records, source_iteration)
    if not decisions:
        raise ProtocolError("source iteration has no decision; finish or retain the current iteration")
    last = decisions[-1]["payload"]
    if last.get("result") == "PASS" and last.get("gate") != "E6":
        raise ProtocolError("source iteration has not reached a non-pass terminal; continue its next gate")
    if last.get("gate") == "E6":
        raise ProtocolError("E6 cannot feed parameter changes back into the released iteration; start a separate project/domain")

    if iteration_class in ("TYPE_I", "TYPE_II"):
        same_mechanism = 0
        for path in (root / "iterations").iterdir():
            if not path.is_dir() or not (path / "contract.yaml").is_file():
                continue
            prior = load_yaml(path / "contract.yaml")
            if (
                prior.get("failure_mechanism_addressed") == failure_mechanism
                and prior.get("iteration_class") in ("TYPE_I", "TYPE_II")
            ):
                same_mechanism += 1
        maximum = int(protocol["policy"]["max_correction_iterations_per_mechanism"])
        if same_mechanism >= maximum:
            raise ProtocolError(
                f"correction cap reached for {failure_mechanism!r}; classify DATA_LIMITED/MODEL_CLASS_LIMITED or use a justified TYPE_III/TYPE_IV"
            )

    opened_confirmation = set()
    for row in decisions:
        payload = row.get("payload", {})
        if payload.get("confirmation_opened"):
            opened_confirmation.update(payload.get("data_group_ids", []))
    contract = deepcopy(source_contract)
    contract["iteration_id"] = new_iteration_id
    contract["parent"] = {
        "iteration_id": source_iteration,
        "terminal": last.get("action"),
        "gate": last.get("gate"),
        "result": last.get("result"),
    }
    contract["iteration_class"] = iteration_class
    contract["failure_mechanism_addressed"] = failure_mechanism
    for group in contract.get("data_groups", []):
        if group.get("id") in opened_confirmation:
            group["role"] = "development"
            group["sealed"] = False
    target_dir.mkdir(parents=True, exist_ok=False)
    (target_dir / "decisions").mkdir()
    atomic_write_yaml(target_dir / "contract.yaml", contract)
    updated_protocol = deepcopy(protocol)
    updated_protocol["active_iteration"] = new_iteration_id
    atomic_write_yaml(root / "protocol.yaml", updated_protocol)
    event = append_event(
        root / "ledger.jsonl",
        event_type="ITERATION_CREATED",
        iteration_id=new_iteration_id,
        payload={
            "parent_iteration_id": source_iteration,
            "parent_terminal": last.get("action"),
            "iteration_class": iteration_class,
            "failure_mechanism_addressed": failure_mechanism,
            "reclassified_opened_confirmation_ids": sorted(opened_confirmation),
        },
    )
    return {
        "decision": "DRAFT_ITERATION_CREATED",
        "iteration_id": new_iteration_id,
        "parent_iteration_id": source_iteration,
        "parent_terminal": last.get("action"),
        "iteration_class": iteration_class,
        "reclassified_opened_confirmation_ids": sorted(opened_confirmation),
        "ledger_event_hash": event["event_hash"],
        "next_action": "edit the draft contract, refresh checksums, validate, and seal before opening evidence",
    }

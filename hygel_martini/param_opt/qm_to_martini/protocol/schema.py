"""Schema and frozen-criterion validation for parameterization contracts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .io import canonical_json_bytes, load_yaml, safe_project_path, sha256_bytes, sha256_file


SCHEMA_VERSION = "1.0"
GATE_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
ITERATION_CLASSES = ("TYPE_I", "TYPE_II", "TYPE_III", "TYPE_IV")
DATA_ROLES = ("development", "validation", "stress", "confirmation")
OPERATORS = ("status", "truthy", "eq", "ne", "lt", "le", "gt", "ge", "between", "in")
STATUSES = ("PASS", "FAIL", "INCONCLUSIVE")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

REQUIRED_IDENTITY_KEYS = (
    "mapping",
    "topology_graph",
    "bead_model",
    "nonbonded_parent",
    "exclusions",
)


def _error(errors: List[str], location: str, message: str) -> None:
    errors.append(f"{location}: {message}")


def _require_mapping(value: Any, errors: List[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(errors, location, "must be a mapping")
        return {}
    return value


def _require_nonempty_string(
    value: Any, errors: List[str], location: str
) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(errors, location, "must be a non-empty string")
        return ""
    return value.strip()


def _validate_identifier(value: Any, errors: List[str], location: str) -> str:
    normalized = _require_nonempty_string(value, errors, location)
    if normalized and not ID_RE.fullmatch(normalized):
        _error(errors, location, "must contain only letters, digits, dot, underscore, or hyphen")
    return normalized


def _validate_artifact_spec(
    spec: Any,
    root: Path,
    errors: List[str],
    warnings: List[str],
    location: str,
    *,
    check_files: bool,
) -> None:
    artifact = _require_mapping(spec, errors, location)
    _require_nonempty_string(artifact.get("id"), errors, f"{location}.id")
    relative = _require_nonempty_string(artifact.get("path"), errors, f"{location}.path")
    checksum = _require_nonempty_string(
        artifact.get("sha256"), errors, f"{location}.sha256"
    )
    if checksum and not SHA256_RE.fullmatch(checksum):
        _error(errors, f"{location}.sha256", "must be a lowercase SHA-256 digest")
    if artifact.get("placeholder", False):
        warnings.append(f"{location}: placeholder is true; sealing is prohibited")
    if check_files and relative:
        try:
            path = safe_project_path(root, relative)
        except ValueError as exc:
            _error(errors, f"{location}.path", str(exc))
            return
        if not path.is_file():
            _error(errors, f"{location}.path", f"file does not exist: {relative}")
        elif checksum and SHA256_RE.fullmatch(checksum) and not artifact.get("placeholder", False):
            observed = sha256_file(path)
            if observed != checksum:
                _error(
                    errors,
                    f"{location}.sha256",
                    f"checksum mismatch: expected {checksum}, observed {observed}",
                )


def validate_protocol_document(payload: Any) -> List[str]:
    errors: List[str] = []
    document = _require_mapping(payload, errors, "protocol")
    if document.get("schema_version") != SCHEMA_VERSION:
        _error(errors, "protocol.schema_version", f"must equal {SCHEMA_VERSION!r}")
    project = _require_mapping(document.get("project"), errors, "protocol.project")
    _validate_identifier(project.get("id"), errors, "protocol.project.id")
    _require_nonempty_string(project.get("title"), errors, "protocol.project.title")
    _require_nonempty_string(
        project.get("claim_domain"), errors, "protocol.project.claim_domain"
    )
    _validate_identifier(
        document.get("active_iteration"), errors, "protocol.active_iteration"
    )
    policy = _require_mapping(document.get("policy"), errors, "protocol.policy")
    if tuple(policy.get("gate_order", ())) != GATE_ORDER:
        _error(errors, "protocol.policy.gate_order", f"must equal {list(GATE_ORDER)!r}")
    for key in ("strict_sequence", "weakest_link"):
        if policy.get(key) is not True:
            _error(errors, f"protocol.policy.{key}", "must be true")
    if policy.get("e6_parameter_feedback") != "prohibited":
        _error(errors, "protocol.policy.e6_parameter_feedback", "must equal 'prohibited'")
    maximum = policy.get("max_correction_iterations_per_mechanism")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        _error(
            errors,
            "protocol.policy.max_correction_iterations_per_mechanism",
            "must be an integer >= 1",
        )
    return errors


def validate_contract_document(
    payload: Any,
    protocol: Mapping[str, Any],
    project_root: Path,
    expected_iteration: str,
    *,
    check_files: bool = True,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    contract = _require_mapping(payload, errors, "contract")
    if contract.get("schema_version") != SCHEMA_VERSION:
        _error(errors, "contract.schema_version", f"must equal {SCHEMA_VERSION!r}")
    project_id = protocol.get("project", {}).get("id")
    if contract.get("project_id") != project_id:
        _error(errors, "contract.project_id", f"must equal {project_id!r}")
    iteration_id = _validate_identifier(
        contract.get("iteration_id"), errors, "contract.iteration_id"
    )
    if iteration_id and iteration_id != expected_iteration:
        _error(
            errors,
            "contract.iteration_id",
            f"must match directory name {expected_iteration!r}",
        )
    iteration_class = contract.get("iteration_class")
    if iteration_class not in ITERATION_CLASSES:
        _error(
            errors,
            "contract.iteration_class",
            f"must be one of {list(ITERATION_CLASSES)!r}",
        )
    _require_nonempty_string(
        contract.get("failure_mechanism_addressed"),
        errors,
        "contract.failure_mechanism_addressed",
    )
    parent = contract.get("parent")
    if parent is not None:
        parent_map = _require_mapping(parent, errors, "contract.parent")
        _validate_identifier(parent_map.get("iteration_id"), errors, "contract.parent.iteration_id")
        _require_nonempty_string(parent_map.get("terminal"), errors, "contract.parent.terminal")

    identity = _require_mapping(
        contract.get("scientific_identity"), errors, "contract.scientific_identity"
    )
    missing_identity = sorted(set(REQUIRED_IDENTITY_KEYS) - set(identity))
    extra_identity = sorted(set(identity) - set(REQUIRED_IDENTITY_KEYS))
    if missing_identity:
        _error(errors, "contract.scientific_identity", f"missing keys: {missing_identity}")
    if extra_identity:
        _error(errors, "contract.scientific_identity", f"unknown keys: {extra_identity}")
    for key in REQUIRED_IDENTITY_KEYS:
        if key in identity:
            _validate_artifact_spec(
                identity[key],
                project_root,
                errors,
                warnings,
                f"contract.scientific_identity.{key}",
                check_files=check_files,
            )

    groups = contract.get("data_groups")
    if not isinstance(groups, list) or not groups:
        _error(errors, "contract.data_groups", "must be a non-empty list")
        groups = []
    group_ids = set()
    group_sources: Dict[Tuple[str, str], Tuple[str, str]] = {}
    sealed_ids = set()
    for index, group in enumerate(groups):
        location = f"contract.data_groups[{index}]"
        item = _require_mapping(group, errors, location)
        group_id = _validate_identifier(item.get("id"), errors, f"{location}.id")
        if group_id in group_ids:
            _error(errors, f"{location}.id", f"duplicate group id {group_id!r}")
        group_ids.add(group_id)
        role = item.get("role")
        if role not in DATA_ROLES:
            _error(errors, f"{location}.role", f"must be one of {list(DATA_ROLES)!r}")
        sealed = item.get("sealed", False)
        if not isinstance(sealed, bool):
            _error(errors, f"{location}.sealed", "must be a boolean")
        if sealed and role != "confirmation":
            _error(errors, f"{location}.sealed", "only confirmation groups may be sealed")
        if sealed:
            sealed_ids.add(group_id)
        source_key = (str(item.get("path", "")), str(item.get("sha256", "")))
        if source_key in group_sources:
            prior_id, prior_role = group_sources[source_key]
            _error(
                errors,
                location,
                f"duplicates source assigned to {prior_id!r} with role {prior_role!r}",
            )
        else:
            group_sources[source_key] = (group_id, str(role))
        _validate_artifact_spec(
            item,
            project_root,
            errors,
            warnings,
            location,
            check_files=check_files,
        )
    if not sealed_ids:
        _error(
            errors,
            "contract.data_groups",
            "must contain at least one sealed confirmation group for the E5 one-shot gate",
        )

    design = _require_mapping(contract.get("design"), errors, "contract.design")
    for key in (
        "coordinate",
        "predecessor",
        "primary_objective",
        "grouping_unit",
        "stop_rule",
        "claim_ceiling",
    ):
        _require_nonempty_string(design.get(key), errors, f"contract.design.{key}")
    ladder = design.get("candidate_ladder")
    if not isinstance(ladder, list) or not ladder:
        _error(errors, "contract.design.candidate_ladder", "must be a non-empty list")
    elif any(not isinstance(item, str) or not item.strip() for item in ladder):
        _error(errors, "contract.design.candidate_ladder", "all entries must be non-empty strings")
    maximum_complexity = design.get("maximum_complexity")
    if not isinstance(maximum_complexity, int) or isinstance(maximum_complexity, bool) or maximum_complexity < 0:
        _error(errors, "contract.design.maximum_complexity", "must be an integer >= 0")
    sensitivities = design.get("sensitivity_objectives")
    if not isinstance(sensitivities, list):
        _error(errors, "contract.design.sensitivity_objectives", "must be a list")

    gates = contract.get("gates")
    if not isinstance(gates, list):
        _error(errors, "contract.gates", "must be a list")
        gates = []
    observed_gate_order = [gate.get("id") if isinstance(gate, Mapping) else None for gate in gates]
    if tuple(observed_gate_order) != GATE_ORDER:
        _error(errors, "contract.gates", f"ids must occur exactly once in order {list(GATE_ORDER)!r}")
    criterion_ids = set()
    for gate_index, gate in enumerate(gates):
        location = f"contract.gates[{gate_index}]"
        item = _require_mapping(gate, errors, location)
        gate_id = item.get("id")
        _require_nonempty_string(item.get("name"), errors, f"{location}.name")
        for outcome in ("on_pass", "on_fail", "on_inconclusive"):
            _require_nonempty_string(item.get(outcome), errors, f"{location}.{outcome}")
        criteria = item.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            _error(errors, f"{location}.criteria", "must be a non-empty list")
            criteria = []
        for criterion_index, criterion in enumerate(criteria):
            criterion_location = f"{location}.criteria[{criterion_index}]"
            rule = _require_mapping(criterion, errors, criterion_location)
            criterion_id = _validate_identifier(
                rule.get("id"), errors, f"{criterion_location}.id"
            )
            if criterion_id in criterion_ids:
                _error(errors, f"{criterion_location}.id", f"duplicate criterion id {criterion_id!r}")
            criterion_ids.add(criterion_id)
            _require_nonempty_string(
                rule.get("description"), errors, f"{criterion_location}.description"
            )
            operator = rule.get("operator")
            if operator not in OPERATORS:
                _error(errors, f"{criterion_location}.operator", f"must be one of {list(OPERATORS)!r}")
            if operator not in ("status", "truthy") and "expected" not in rule:
                _error(errors, f"{criterion_location}.expected", f"is required for operator {operator!r}")
            if operator == "between":
                expected = rule.get("expected")
                if not isinstance(expected, list) or len(expected) != 2:
                    _error(errors, f"{criterion_location}.expected", "must be [lower, upper]")
            if operator == "in" and not isinstance(rule.get("expected"), list):
                _error(errors, f"{criterion_location}.expected", "must be a list")
            for outcome in ("on_fail", "on_inconclusive"):
                _require_nonempty_string(rule.get(outcome), errors, f"{criterion_location}.{outcome}")
        if gate_id == "E6" and item.get("on_fail") == "NONRELEASE":
            _error(errors, f"{location}.on_fail", "E6 failure cannot revoke an E5 tested-domain release")

    repairs = contract.get("permitted_repairs")
    if not isinstance(repairs, list) or not repairs:
        _error(errors, "contract.permitted_repairs", "must be a non-empty list")
    prohibited = contract.get("prohibited_after_seal")
    if not isinstance(prohibited, list) or not prohibited:
        _error(errors, "contract.prohibited_after_seal", "must be a non-empty list")

    return errors, warnings


def contract_artifacts(contract: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    identity = contract.get("scientific_identity", {})
    for key in REQUIRED_IDENTITY_KEYS:
        artifact = identity.get(key)
        if isinstance(artifact, Mapping):
            yield artifact
    for group in contract.get("data_groups", []):
        if isinstance(group, Mapping):
            yield group


def scientific_identity_hash(contract: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(contract.get("scientific_identity", {})))


def evaluate_rule(rule: Mapping[str, Any], observed: Any) -> str:
    """Evaluate one frozen criterion without importing a post-result threshold."""

    if isinstance(observed, str) and observed.strip().upper() == "INCONCLUSIVE":
        return "INCONCLUSIVE"
    operator = rule["operator"]
    expected = rule.get("expected")
    if operator == "status":
        normalized = str(observed).strip().upper()
        if normalized not in STATUSES:
            raise ValueError(f"status observation must be one of {list(STATUSES)!r}")
        return normalized
    if operator == "truthy":
        if not isinstance(observed, bool):
            raise ValueError("truthy observation must be a YAML boolean")
        passed = observed
    elif operator in ("lt", "le", "gt", "ge", "between"):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(f"{operator} observation must be numeric")
        value = float(observed)
        if not math.isfinite(value):
            raise ValueError("numeric observation must be finite")
        if operator == "lt":
            passed = value < float(expected)
        elif operator == "le":
            passed = value <= float(expected)
        elif operator == "gt":
            passed = value > float(expected)
        elif operator == "ge":
            passed = value >= float(expected)
        else:
            lower, upper = (float(item) for item in expected)
            passed = lower <= value <= upper
    elif operator == "eq":
        passed = observed == expected
    elif operator == "ne":
        passed = observed != expected
    elif operator == "in":
        passed = observed in expected
    else:  # pragma: no cover - contract validation excludes this path
        raise ValueError(f"unsupported operator: {operator}")
    return "PASS" if passed else "FAIL"


def load_project_documents(
    project_root: Path, iteration_id: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = load_yaml(project_root / "protocol.yaml")
    contract = load_yaml(project_root / "iterations" / iteration_id / "contract.yaml")
    if not isinstance(protocol, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("protocol and contract documents must be mappings")
    return protocol, contract

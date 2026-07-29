from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any


REQUIRED_MANIFEST_KEYS = {"schema_version", "references", "targets"}
REQUIRED_TARGET_KEYS = {"reference", "formulation", "properties"}
REQUIRED_PROPERTY_KEYS = {"unit"}
REQUIRED_REFERENCE_KEYS = {"doi", "system", "notes"}


@dataclass
class ManifestProperty:
    name: str
    unit: str
    value: float | None = None
    min: float | None = None
    max: float | None = None
    uncertainty: float | None = None
    tolerance: float | None = None
    definition: str = ""
    method: str = ""
    comparable_to: list[str] = field(default_factory=list)
    not_comparable_to: list[str] = field(default_factory=list)
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def is_directly_comparable(self, simulation_property: str) -> bool:
        return simulation_property in self.comparable_to

    def is_not_comparable(self, simulation_property: str) -> bool:
        return simulation_property in self.not_comparable_to


@dataclass
class ManifestTarget:
    target_id: str
    reference: str
    formulation: dict[str, Any]
    properties: dict[str, ManifestProperty]

    def get_property(self, name: str) -> ManifestProperty | None:
        return self.properties.get(name)


def _parse_property(name: str, spec: dict) -> ManifestProperty:
    if not isinstance(spec, dict):
        raise ValueError(f"property {name!r}: dict 형식이어야 합니다.")
    if "unit" not in spec:
        raise ValueError(f"property {name!r}: 'unit' 필드가 없습니다.")

    comparable_to = _as_string_list(spec.get("comparable_to"), f"{name}.comparable_to")
    not_comparable_to = _as_string_list(
        spec.get("not_comparable_to"), f"{name}.not_comparable_to"
    )

    return ManifestProperty(
        name=name,
        unit=str(spec["unit"]),
        value=float(spec["value"]) if spec.get("value") is not None else None,
        min=float(spec["min"]) if spec.get("min") is not None else None,
        max=float(spec["max"]) if spec.get("max") is not None else None,
        uncertainty=float(spec["uncertainty"]) if spec.get("uncertainty") is not None else None,
        tolerance=float(spec["tolerance"]) if spec.get("tolerance") is not None else None,
        definition=str(spec.get("definition", "")),
        method=str(spec.get("method", "")),
        comparable_to=comparable_to,
        not_comparable_to=not_comparable_to,
        notes=str(spec.get("notes", "")),
        raw=spec,
    )


def _parse_target(target_id: str, spec: dict) -> ManifestTarget:
    missing = REQUIRED_TARGET_KEYS - set(spec.keys())
    if missing:
        raise ValueError(f"target {target_id!r}: 필수 키 누락 — {sorted(missing)}")

    properties_spec = spec.get("properties") or {}
    if not isinstance(properties_spec, dict) or not properties_spec:
        raise ValueError(f"target {target_id!r}: properties는 비어 있지 않은 dict여야 합니다.")

    properties = {}
    for prop_name, prop_spec in properties_spec.items():
        properties[prop_name] = _parse_property(prop_name, prop_spec)

    return ManifestTarget(
        target_id=target_id,
        reference=str(spec["reference"]),
        formulation=dict(spec.get("formulation") or {}),
        properties=properties,
    )


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name}: list[str] 형식이어야 합니다.")
    bad = [item for item in value if not isinstance(item, str)]
    if bad:
        raise ValueError(f"{field_name}: 모든 항목은 문자열이어야 합니다. bad={bad!r}")
    return list(value)


def _validate_references(references: dict[str, Any]) -> None:
    if not isinstance(references, dict) or not references:
        raise ValueError("validation_manifest.references는 비어 있지 않은 dict여야 합니다.")

    for ref_id, ref in references.items():
        if not isinstance(ref, dict):
            raise ValueError(f"reference {ref_id!r}: dict 형식이어야 합니다.")
        missing = REQUIRED_REFERENCE_KEYS - set(ref.keys())
        if missing:
            raise ValueError(f"reference {ref_id!r}: 필수 키 누락 — {sorted(missing)}")
        path = ref.get("path")
        if path is not None and not isinstance(path, str):
            raise ValueError(f"reference {ref_id!r}: path는 string 또는 null이어야 합니다.")


def _validate_property_value(target_id: str, prop: ManifestProperty) -> None:
    # 일부 manifest 항목은 아직 trend benchmark나 placeholder일 수 있으므로
    # 수치가 없는 property도 허용한다. 실제 numeric compare 단계에서만
    # value/min/max 존재 여부를 다시 확인한다.
    return None


def _validate_targets_against_references(
    targets: list[ManifestTarget],
    references: dict[str, Any],
) -> None:
    for target in targets:
        if target.reference not in references:
            raise ValueError(
                f"target {target.target_id!r}: reference {target.reference!r}가 "
                "references에 없습니다."
            )
        for prop in target.properties.values():
            _validate_property_value(target.target_id, prop)


def load_manifest(path: str) -> list[ManifestTarget]:
    """
    validation_manifest.yaml을 읽어 ManifestTarget 리스트로 반환.
    schema_version, references, targets 키가 없으면 ValueError.
    """
    import yaml

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"validation_manifest 파일 없음: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"validation_manifest.yaml이 dict가 아닙니다: {path}")

    missing_top = REQUIRED_MANIFEST_KEYS - set(data.keys())
    if missing_top:
        raise ValueError(
            f"validation_manifest.yaml 최상위 키 누락 — {sorted(missing_top)}: {path}"
        )

    references = data.get("references") or {}
    _validate_references(references)

    targets: list[ManifestTarget] = []
    for target_id, target_spec in (data.get("targets") or {}).items():
        targets.append(_parse_target(target_id, target_spec))

    _validate_targets_against_references(targets, references)
    return targets


def get_target_property(
    targets: list[ManifestTarget],
    property_name: str,
) -> tuple[ManifestTarget, ManifestProperty] | tuple[None, None]:
    """
    전체 target 목록에서 property_name을 가진 첫 번째 (target, property) 반환.
    없으면 (None, None).
    """
    for target in targets:
        prop = target.get_property(property_name)
        if prop is not None:
            return target, prop
    return None, None


def find_target_properties_for_simulation_property(
    targets: list[ManifestTarget],
    simulation_property: str,
) -> list[tuple[ManifestTarget, ManifestProperty, str]]:
    """
    simulation property 이름으로 manifest target property를 찾는다.

    반환 relation:
    - "same_property": target property key와 simulation property가 같음
    - "comparable_to": manifest가 직접 비교 가능하다고 선언함
    - "not_comparable_to": manifest가 직접 비교 금지라고 선언함
    """
    matches: list[tuple[ManifestTarget, ManifestProperty, str]] = []
    for target in targets:
        for prop in target.properties.values():
            if prop.name == simulation_property:
                matches.append((target, prop, "same_property"))
            elif simulation_property in prop.comparable_to:
                matches.append((target, prop, "comparable_to"))
            elif simulation_property in prop.not_comparable_to:
                matches.append((target, prop, "not_comparable_to"))
    return matches

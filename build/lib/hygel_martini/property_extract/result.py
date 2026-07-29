from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

ALLOWED_STATUSES = {
    "computed",
    "missing_required_md",
    "invalid_input",
    "analysis_failed",
    "insufficient_data",
    "not_implemented",
}

ALLOWED_VALIDATION_ROLES = {
    "",
    "composition_check_only",
    "proxy",
    "direct",
    "trend_only",
    "structural_audit",
    "finite_rate",
}


@dataclass
class PropertyResult:
    """
    모든 extractor의 표준 반환 구조.

    status:
        "computed"             — 정상 계산 완료
        "missing_required_md"  — 필요한 MD output 파일이 없음
        "invalid_input"        — 파일은 있지만 설정/컬럼/선택이 잘못됨
        "analysis_failed"      — 예기치 못한 분석 실패
        "insufficient_data"    — 데이터는 있으나 판단/통계에 부족함
        "not_implemented"      — extractor 미구현

    direct_experiment_comparison_allowed:
        False 이면 _report_targets() 가 비교를 블록하고 이유를 출력한다.

    validation_role:
        "composition_check_only"  — 조성값, 실험 swelling ratio와 비교 불가
        "proxy"                   — 방법론 mismatch 있음, screening 용도
        "direct"                  — 실험 target과 직접 비교 가능
        "trend_only"              — 절댓값 비교 불가, trend/rank-order 만 유효
        "structural_audit"        — construction/topology 자체 검증
        "finite_rate"             — 등록된 유한 속도 mechanics observable
    """

    property: str
    value: Any = None
    status: str = "computed"
    direct_experiment_comparison_allowed: bool = True
    validation_role: str = "direct"
    missing_required_inputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ALLOWED_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_STATUSES))
            raise ValueError(f"Invalid PropertyResult.status={self.status!r}; allowed: {allowed}")

        if self.validation_role not in ALLOWED_VALIDATION_ROLES:
            allowed = ", ".join(sorted(ALLOWED_VALIDATION_ROLES))
            raise ValueError(
                f"Invalid PropertyResult.validation_role={self.validation_role!r}; "
                f"allowed: {allowed}"
            )

        if not isinstance(self.missing_required_inputs, list):
            self.missing_required_inputs = list(self.missing_required_inputs)
        if not isinstance(self.metadata, dict):
            raise TypeError("PropertyResult.metadata must be a dict")

        if self.status != "computed":
            self.direct_experiment_comparison_allowed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "property": self.property,
            "value": self.value,
            "status": self.status,
            "calculability": self.status,
            "direct_experiment_comparison_allowed": self.direct_experiment_comparison_allowed,
            "validation_role": self.validation_role,
            "missing_required_inputs": self.missing_required_inputs,
            "metadata": self.metadata,
        }

    @staticmethod
    def missing(
        property_name: str,
        missing_inputs: list[str],
        validation_role: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PropertyResult":
        return PropertyResult(
            property=property_name,
            value=None,
            status="missing_required_md",
            direct_experiment_comparison_allowed=False,
            validation_role=validation_role,
            missing_required_inputs=missing_inputs,
            metadata=metadata or {},
        )

    @staticmethod
    def invalid_input(
        property_name: str,
        reason: str,
        inputs: list[str] | None = None,
        validation_role: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PropertyResult":
        meta = dict(metadata or {})
        meta["reason"] = reason
        return PropertyResult(
            property=property_name,
            value=None,
            status="invalid_input",
            direct_experiment_comparison_allowed=False,
            validation_role=validation_role,
            missing_required_inputs=inputs or [],
            metadata=meta,
        )

    @staticmethod
    def analysis_failed(
        property_name: str,
        error: str,
        inputs: list[str] | None = None,
        validation_role: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PropertyResult":
        meta = dict(metadata or {})
        meta["error"] = error
        return PropertyResult(
            property=property_name,
            value=None,
            status="analysis_failed",
            direct_experiment_comparison_allowed=False,
            validation_role=validation_role,
            missing_required_inputs=inputs or [],
            metadata=meta,
        )

    @staticmethod
    def insufficient_data(
        property_name: str,
        reason: str,
        validation_role: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PropertyResult":
        meta = dict(metadata or {})
        meta["reason"] = reason
        return PropertyResult(
            property=property_name,
            value=None,
            status="insufficient_data",
            direct_experiment_comparison_allowed=False,
            validation_role=validation_role,
            metadata=meta,
        )

    @staticmethod
    def not_implemented(property_name: str, reason: str = "") -> "PropertyResult":
        return PropertyResult(
            property=property_name,
            value=None,
            status="not_implemented",
            direct_experiment_comparison_allowed=False,
            validation_role="",
            metadata={"reason": reason},
        )

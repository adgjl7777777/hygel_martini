"""Standard result container shared by every property extractor.

Owns :class:`PropertyResult`, the single return type through which all
extractors, the job runner (``analysis_jobs``) and the CLI communicate.
It encodes the package's claim-boundary design in data: a result always
carries an explicit computability ``status`` and an interpretation
``validation_role``, and any non-``computed`` status forcibly revokes
``direct_experiment_comparison_allowed`` so downstream reporting cannot
compare a failed/missing analysis against experimental targets.

Invariants:
    * ``status`` must be one of :data:`ALLOWED_STATUSES`;
      ``validation_role`` one of :data:`ALLOWED_VALIDATION_ROLES`
      (enforced in ``__post_init__``, which raises otherwise).
    * ``status != "computed"`` implies
      ``direct_experiment_comparison_allowed is False``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Closed vocabulary for PropertyResult.status; __post_init__ rejects
# anything else so new failure modes must be registered here first.
ALLOWED_STATUSES = {
    "computed",
    "missing_required_md",
    "invalid_input",
    "analysis_failed",
    "insufficient_data",
    "not_implemented",
}

# Closed vocabulary for PropertyResult.validation_role ("" = unassigned).
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
    """Standard return structure for every extractor.

    Fields:
        property: Canonical observable name (e.g.
            ``paired_step_finite_rate_apparent_shear_response``).
        value: Computed value (scalar, dict, list, ...); ``None`` when
            the status is anything other than ``computed``.
        status: Computability outcome.  One of:
            ``"computed"`` — analysis finished normally;
            ``"missing_required_md"`` — a required MD output file is
            absent;
            ``"invalid_input"`` — files exist but a setting, column, or
            selection is wrong;
            ``"analysis_failed"`` — unexpected failure during analysis;
            ``"insufficient_data"`` — data exist but are too few for a
            decision or statistic;
            ``"not_implemented"`` — no extractor implements this
            property.
        direct_experiment_comparison_allowed: When ``False``,
            report-time target comparison is blocked and the reason is
            printed instead.  Forced ``False`` for any non-``computed``
            status.
        validation_role: How the value may be interpreted.  One of:
            ``"composition_check_only"`` — composition value, not
            comparable to experimental swelling ratios;
            ``"proxy"`` — methodological mismatch, screening use only;
            ``"direct"`` — directly comparable to experimental targets;
            ``"trend_only"`` — absolute values not comparable, only
            trends/rank order are valid;
            ``"structural_audit"`` — audits construction/topology
            itself;
            ``"finite_rate"`` — registered finite-rate mechanics
            observable, not an equilibrium modulus;
            ``""`` — role unassigned (typical for failure results).
        missing_required_inputs: File paths / input names that were
            required but missing (populated for failure statuses).
        metadata: Free-form provenance (parameters, reasons, errors,
            window definitions, ...).  Failure constructors put the
            explanation under ``metadata["reason"]`` or
            ``metadata["error"]``.
    """

    property: str
    value: Any = None
    status: str = "computed"
    direct_experiment_comparison_allowed: bool = True
    validation_role: str = "direct"
    missing_required_inputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate vocabularies and enforce the claim-boundary rule.

        Raises:
            ValueError: If ``status`` or ``validation_role`` is not in
                the registered vocabulary.
            TypeError: If ``metadata`` is not a dict.
        """
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
        """Serialize to a plain dict for JSON output.

        ``calculability`` duplicates ``status`` for backward-compatible
        consumers of the serialized form.
        """
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
        """Build a ``missing_required_md`` result listing absent inputs."""
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
        """Build an ``invalid_input`` result; ``reason`` goes to metadata."""
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
        """Build an ``analysis_failed`` result; ``error`` goes to metadata."""
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
        """Build an ``insufficient_data`` result; ``reason`` to metadata."""
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
        """Build a ``not_implemented`` result for an unsupported property."""
        return PropertyResult(
            property=property_name,
            value=None,
            status="not_implemented",
            direct_experiment_comparison_allowed=False,
            validation_role="",
            metadata={"reason": reason},
        )

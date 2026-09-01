"""Manifest-driven analysis job loading and execution.

Owns the ``analysis_jobs.yaml`` schema (:class:`AnalysisJob`) and the
job runner :func:`run_analysis`, which dispatches each job to a plugin
from ``extractors.EXTRACTOR_REGISTRY``.  Called by the CLI
(``__main__``) and by :mod:`config`.

Claim-boundary behavior of the runner (nothing is silently skipped and
no exception escapes a single job):

* extractor name not in the registry     -> ``not_implemented``
* md_requirements gate not satisfied     -> ``missing_required_md``
* extractor's own ``can_compute`` fails  -> ``missing_required_md``
* extractor raises during ``compute``    -> ``analysis_failed``

Every per-job :class:`PropertyResult` is also written to the job's
``output.report`` JSON (if declared) via an atomic tmp-file rename.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .result import PropertyResult


@dataclass
class AnalysisJob:
    """One entry of the ``analysis_jobs`` mapping in analysis_jobs.yaml.

    Fields:
        job_id: YAML key of the job (unique within one file).
        property: Canonical observable name the job claims to produce.
        extractor: Registry key of the extractor plugin
            (e.g. ``"topology.reduced_network"``).
        inputs: Mapping of input name -> file path (possibly relative to
            the YAML's directory; ``None`` allowed for optional slots).
        parameters: Extractor-specific keyword parameters, passed
            through verbatim.
        output: Output options; ``output["report"]`` names a JSON file
            to write the result to.
        validation: Free-form validation options carried from the YAML
            (not interpreted here).
        requires_md_job: Optional MD-job id this analysis depends on
            (declarative; checked via md_requirements.yaml).
        raw: The original YAML dict for the job, kept for provenance.
    """

    job_id: str
    property: str
    extractor: str
    inputs: dict[str, str | None] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    requires_md_job: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def resolved_inputs(self, base_dir: str) -> dict[str, str | None]:
        """Resolve relative input paths against ``base_dir``.

        Absolute paths and ``None`` values pass through unchanged;
        relative paths become absolute paths rooted at ``base_dir``
        (the directory of the analysis YAML).
        """
        resolved: dict[str, str | None] = {}
        for key, value in self.inputs.items():
            if value is None:
                resolved[key] = None
                continue
            value_str = str(value)
            resolved[key] = (
                value_str
                if os.path.isabs(value_str)
                else os.path.abspath(os.path.join(base_dir, value_str))
            )
        return resolved


def load_analysis_jobs(path: str, allow_template: bool = False) -> tuple[list[AnalysisJob], str]:
    """Read analysis_jobs.yaml and return the job list plus its base dir.

    Args:
        path: Path to the analysis-jobs YAML file.
        allow_template: When ``False`` (default), a file marked
            ``template: true`` is rejected so a placeholder file cannot
            be executed by accident.

    Returns:
        Tuple ``(jobs, base_dir)`` where ``base_dir`` is the YAML's
        directory, used later to resolve relative input paths.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is not a mapping, is a template, lacks
            the ``analysis_jobs`` key, or a job entry is malformed
            (missing ``property``/``extractor``/``inputs``, or inputs
            not a dict).
    """
    import yaml

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"analysis_jobs.yaml 없음: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"analysis_jobs.yaml이 dict가 아닙니다: {path}")
    if data.get("template") and not allow_template:
        raise ValueError(
            f"analysis_jobs.yaml은 template: true 입니다. 실행용 파일이 아닙니다: {path}"
        )
    if "analysis_jobs" not in data:
        raise ValueError(f"analysis_jobs.yaml에 'analysis_jobs' 키가 없습니다: {path}")

    jobs_spec = data.get("analysis_jobs") or {}
    if not isinstance(jobs_spec, dict) or not jobs_spec:
        raise ValueError("analysis_jobs는 비어 있지 않은 dict여야 합니다.")

    jobs: list[AnalysisJob] = []
    for job_id, spec in jobs_spec.items():
        if not isinstance(spec, dict):
            raise ValueError(f"analysis job {job_id!r}: dict 형식이어야 합니다.")
        for required in ("property", "extractor", "inputs"):
            if required not in spec:
                raise ValueError(f"analysis job {job_id!r}: '{required}' 필드가 없습니다.")
        inputs = spec.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ValueError(f"analysis job {job_id!r}: inputs는 dict여야 합니다.")

        jobs.append(
            AnalysisJob(
                job_id=str(job_id),
                property=str(spec["property"]),
                extractor=str(spec["extractor"]),
                inputs=dict(inputs),
                parameters=dict(spec.get("parameters") or {}),
                output=dict(spec.get("output") or {}),
                validation=dict(spec.get("validation") or {}),
                requires_md_job=(
                    str(spec["requires_md_job"])
                    if spec.get("requires_md_job") is not None
                    else None
                ),
                raw=spec,
            )
        )

    return jobs, os.path.dirname(path)


def run_analysis(
    analysis_path: str,
    md_requirements_path: str | None = None,
) -> dict[str, PropertyResult]:
    """Run every job in analysis_jobs.yaml through its extractor.

    If ``md_requirements_path`` is omitted, ``md_requirements.yaml``
    next to the analysis YAML is used when present; when a requirements
    file is in play, each job is gated by
    :func:`requirements.check_requirements` before its extractor runs.

    Args:
        analysis_path: Path to analysis_jobs.yaml.
        md_requirements_path: Optional path to md_requirements.yaml.

    Returns:
        ``{job_id: PropertyResult}``.  Per-job failures are encoded as
        result statuses, never raised:
        extractor absent from the registry -> ``not_implemented``;
        required files missing (per requirements gate or the
        extractor's ``can_compute``) -> ``missing_required_md``;
        an exception inside ``compute`` -> ``analysis_failed``.
    """
    from .extractors import EXTRACTOR_REGISTRY
    jobs, base_dir = load_analysis_jobs(analysis_path)

    if md_requirements_path is None:
        candidate = os.path.join(base_dir, "md_requirements.yaml")
        if os.path.exists(candidate):
            md_requirements_path = candidate

    results: dict[str, PropertyResult] = {}

    for job in jobs:
        inputs = job.resolved_inputs(base_dir)

        if md_requirements_path is not None:
            from .requirements import check_requirements

            requirement = check_requirements(
                job.property,
                {key: value or "" for key, value in inputs.items()},
                md_requirements_path,
                base_dir=base_dir,
            )
            if not requirement.satisfied:
                results[job.job_id] = PropertyResult.missing(
                    job.property,
                    missing_inputs=(
                        requirement.missing_required_inputs
                        + requirement.invalid_inputs
                    ),
                    validation_role=requirement.validation_role,
                    metadata={"requirement_status": requirement.to_dict()},
                )
                _write_job_output(job, results[job.job_id], base_dir)
                continue

        extractor_cls = EXTRACTOR_REGISTRY.get(job.extractor)

        if extractor_cls is None:
            results[job.job_id] = PropertyResult.not_implemented(
                job.property,
                reason=f"extractor '{job.extractor}'이 registry에 없습니다.",
            )
            _write_job_output(job, results[job.job_id], base_dir)
            continue

        extractor = extractor_cls()

        if not extractor.can_compute(inputs):
            missing = extractor.missing_inputs_list(inputs)
            results[job.job_id] = PropertyResult.missing(
                job.property,
                missing_inputs=missing,
            )
            _write_job_output(job, results[job.job_id], base_dir)
            continue

        try:
            results[job.job_id] = extractor.compute(inputs, job.parameters)
        except Exception as e:
            results[job.job_id] = PropertyResult.analysis_failed(
                job.property,
                error=str(e),
            )
        _write_job_output(job, results[job.job_id], base_dir)

    return results


def _write_job_output(
    job: AnalysisJob,
    result: PropertyResult,
    base_dir: str,
) -> None:
    """Write ``output.report`` atomically when the analysis job requests it.

    Relative report paths are resolved against ``base_dir``.  The JSON
    is first written to ``<report>.tmp`` and then renamed, so a crash
    never leaves a truncated report behind.  No-op if the job declares
    no ``output.report``.
    """
    report = job.output.get("report")
    if not report:
        return
    report_path = Path(str(report))
    if not report_path.is_absolute():
        report_path = Path(base_dir) / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)

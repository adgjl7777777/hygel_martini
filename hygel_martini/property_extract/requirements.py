"""MD requirement gate driven by ``md_requirements.yaml``.

Decides, per property, whether the declared MD prerequisites (static
inputs, MD output files, required XVG columns) are actually present on
disk before an analysis is allowed to run.  Called by the job runner
(:func:`analysis_jobs.run_analysis`) and the CLI ``requirements``
subcommand.

Gate semantics (an unsatisfied gate never raises for the caller; it is
reported through :class:`RequirementStatus`):

* a property absent from ``property_requirements`` is UNSATISFIED with
  an explanatory note — an unregistered property cannot pass the gate;
* ``required_inputs`` / ``required_outputs`` may be matched by the
  caller-supplied ``available_files`` mapping (by key or basename) or
  by globbing relative to ``base_dir``;
* ``required_columns`` are verified inside every found ``.xvg``
  artifact; parse failures and missing columns are ``invalid_inputs``;
* ``required_md_jobs`` is declarative only — jobs are reported missing
  when required outputs are absent, not probed directly.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequirementStatus:
    """Outcome of the requirement gate for one property.

    Fields:
        property_name: Property that was checked.
        md_required: Whether the YAML declares MD output as required.
        satisfied: True only if no inputs are missing and none are
            invalid; the runner refuses to analyze otherwise.
        validation_role: Interpretation role declared in the YAML
            (propagated into the eventual PropertyResult).
        required_md_jobs: MD job ids declared as prerequisites.
        missing_required_inputs: Declared inputs/outputs not found on
            disk (paths or glob patterns as written in the YAML).
        missing_md_jobs: ``required_md_jobs`` echoed when MD is
            required and outputs are missing; empty when satisfied.
        invalid_inputs: Artifacts that exist but fail column checks or
            cannot be parsed ("path: reason" strings).
        notes: Free-text note from the YAML entry, or the reason the
            property is unregistered.
    """

    property_name: str
    md_required: bool
    satisfied: bool
    validation_role: str = ""
    required_md_jobs: list[str] = field(default_factory=list)
    missing_required_inputs: list[str] = field(default_factory=list)
    missing_md_jobs: list[str] = field(default_factory=list)
    invalid_inputs: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON reports."""
        return {
            "property": self.property_name,
            "md_required": self.md_required,
            "satisfied": self.satisfied,
            "validation_role": self.validation_role,
            "required_md_jobs": self.required_md_jobs,
            "missing_required_inputs": self.missing_required_inputs,
            "missing_md_jobs": self.missing_md_jobs,
            "invalid_inputs": self.invalid_inputs,
            "notes": self.notes,
        }


def _resolve_glob(pattern: str, base_dir: str) -> list[str]:
    """Expand a glob pattern relative to ``base_dir``.

    Absolute patterns are used as-is.  A pattern without wildcards
    degenerates to a plain existence check (glob returns the path only
    if it exists).

    Returns:
        List of existing matching paths (possibly empty).
    """
    import glob as _glob
    full_pattern = pattern if os.path.isabs(pattern) else os.path.join(base_dir, pattern)
    matches = _glob.glob(full_pattern)
    return matches


def _existing_input_by_basename(
    filename: str,
    available_files: dict[str, str],
) -> str | None:
    """Return the first available-file value whose basename matches.

    Only values that actually exist on disk qualify; returns ``None``
    when nothing matches.
    """
    for value in available_files.values():
        if not value:
            continue
        if os.path.basename(str(value)) == filename and os.path.exists(str(value)):
            return str(value)
    return None


def _existing_input_by_key(
    key: str,
    available_files: dict[str, str],
) -> str | None:
    """Return ``available_files[key]`` if it points to an existing file."""
    value = available_files.get(key)
    if value and os.path.exists(str(value)):
        return str(value)
    return None


def _candidate_output_matches(
    pattern: str,
    base_dir: str,
    available_files: dict[str, str],
) -> list[str]:
    """Find on-disk artifacts satisfying one ``required_outputs`` entry.

    Tries a glob relative to ``base_dir`` first; if that finds nothing,
    falls back to matching the pattern's basename against the values of
    ``available_files`` (see inline comment for why).

    Returns:
        Existing paths accepted for this pattern (empty if none).
    """
    matches = _resolve_glob(pattern, base_dir)
    if matches:
        return matches

    # analysis_jobs.yaml은 보통 "energy_xvg: energy.xvg"처럼 job input으로
    # output을 넘긴다. md_requirements.yaml의 "production/energy.xvg"와
    # basename이 같으면 같은 MD artifact로 인정한다.
    basename = os.path.basename(pattern)
    by_basename = _existing_input_by_basename(basename, available_files)
    return [by_basename] if by_basename else []


def _validate_required_columns(paths: list[str], columns: list[str]) -> list[str]:
    """Check that every found .xvg artifact carries the required columns.

    Non-XVG paths are ignored; when no .xvg is present the check is
    vacuous (missing files are reported elsewhere as missing inputs).

    Args:
        paths: Artifacts already found on disk.
        columns: Legend names required in each XVG (e.g. "Pres-XY").

    Returns:
        "path: reason" strings for XVGs that fail to parse or lack a
        required column; empty when everything passes.
    """
    if not columns:
        return []

    invalid: list[str] = []
    xvg_paths = [p for p in paths if str(p).endswith(".xvg")]
    if not xvg_paths:
        return invalid

    from .gmx_utils import parse_xvg

    for path in xvg_paths:
        try:
            data = parse_xvg(path)
        except Exception as exc:
            invalid.append(f"{path}: XVG parse failed ({exc})")
            continue
        missing = [col for col in columns if col not in data]
        if missing:
            invalid.append(f"{path}: missing columns {missing}")
    return invalid


def check_requirements(
    property_name: str,
    available_files: dict[str, str],
    md_requirements_path: str,
    base_dir: str | None = None,
) -> RequirementStatus:
    """Evaluate one property's MD requirement gate against the filesystem.

    Reads the ``property_requirements[property_name]`` entry of
    md_requirements.yaml and checks its ``required_inputs``,
    ``required_outputs`` (glob patterns allowed) and
    ``required_columns`` against ``available_files`` and disk state.
    Each declared input is matched in order by (1) ``available_files``
    key, (2) basename of an ``available_files`` value, (3) glob under
    ``base_dir``.

    Args:
        property_name: Canonical property name to look up.
        available_files: Input mapping such as
            ``{"top": "/path/system.top", "itp": "/path/hydrogel.itp"}``.
        md_requirements_path: Path to md_requirements.yaml.
        base_dir: Root for relative/glob patterns in the YAML; defaults
            to the current working directory.

    Returns:
        A :class:`RequirementStatus`.  An unregistered property yields
        ``satisfied=False`` with an explanatory note rather than an
        exception.

    Raises:
        FileNotFoundError: If md_requirements.yaml itself is absent.
    """
    import yaml

    if not os.path.exists(md_requirements_path):
        raise FileNotFoundError(f"md_requirements.yaml 없음: {md_requirements_path}")

    with open(md_requirements_path, "r") as f:
        data = yaml.safe_load(f)

    prop_reqs = (data or {}).get("property_requirements", {})
    req = prop_reqs.get(property_name)

    if req is None:
        return RequirementStatus(
            property_name=property_name,
            md_required=False,
            satisfied=False,
            validation_role="",
            required_md_jobs=[],
            missing_md_jobs=[],
            missing_required_inputs=[],
            invalid_inputs=[],
            notes=f"md_requirements.yaml에 '{property_name}' 항목이 없습니다.",
        )

    md_required = bool(req.get("md_required", False))
    validation_role = str(req.get("validation_role", ""))
    notes = str(req.get("notes", ""))

    if base_dir is None:
        base_dir = os.getcwd()

    missing_inputs: list[str] = []
    invalid_inputs: list[str] = []
    missing_jobs: list[str] = []
    required_jobs = [str(job) for job in (req.get("required_md_jobs") or [])]
    found_artifacts: list[str] = []

    # available_files 값에서 basename → 절대경로 역매핑 생성
    basename_map: dict[str, str] = {}
    for _v in available_files.values():
        if _v:
            basename_map[os.path.basename(str(_v))] = str(_v)

    # 정적 입력 파일 체크
    for inp_key in (req.get("required_inputs") or []):
        # 1) available_files 키 직접 매핑 (예: "top" → "/path/system.top")
        resolved = available_files.get(inp_key)
        if resolved and os.path.exists(resolved):
            found_artifacts.append(str(resolved))
            continue
        # 2) available_files 값의 basename 매핑 (예: "system.top" → "/path/system.top")
        resolved = basename_map.get(inp_key)
        if resolved and os.path.exists(resolved):
            found_artifacts.append(str(resolved))
            continue
        # 3) base_dir 기준 glob 시도
        matches = _resolve_glob(inp_key, base_dir)
        if not matches:
            missing_inputs.append(inp_key)
        else:
            found_artifacts.extend(matches)

    # MD output 파일 체크 (glob 패턴 지원)
    for out_pattern in (req.get("required_outputs") or []):
        # dict 형태인 경우 key만 사용 (예: "production/energy.xvg: Volume")
        if isinstance(out_pattern, dict):
            for k in out_pattern:
                out_pattern = k
                break
        matches = _candidate_output_matches(str(out_pattern), base_dir, available_files)
        if not matches:
            missing_inputs.append(str(out_pattern))
        else:
            found_artifacts.extend(matches)

    required_columns = [str(col) for col in (req.get("required_columns") or [])]
    invalid_inputs.extend(_validate_required_columns(found_artifacts, required_columns))

    # required_md_jobs는 "선언"이다. missing 여부는 관련 output 부재로 판단한다.
    if md_required and missing_inputs:
        missing_jobs = required_jobs

    satisfied = (
        not missing_inputs
        and not invalid_inputs
    )

    return RequirementStatus(
        property_name=property_name,
        md_required=md_required,
        satisfied=satisfied,
        validation_role=validation_role,
        required_md_jobs=required_jobs,
        missing_required_inputs=missing_inputs,
        missing_md_jobs=missing_jobs if not satisfied else [],
        invalid_inputs=invalid_inputs,
        notes=notes,
    )


def check_all_requirements(
    properties: list[str],
    available_files: dict[str, str],
    md_requirements_path: str,
    base_dir: str | None = None,
) -> dict[str, RequirementStatus]:
    """Run :func:`check_requirements` for every property in ``properties``.

    Returns:
        ``{property_name: RequirementStatus}`` in input order.
    """
    return {
        prop: check_requirements(prop, available_files, md_requirements_path, base_dir)
        for prop in properties
    }

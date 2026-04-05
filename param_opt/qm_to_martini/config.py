from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from fractions import Fraction

from ..core.utils import parse_csv_list

CONNECTION_CUTOFF = 2.2

@dataclass(frozen=True)
class ConnectionDetectionConfig:
    indicator: str
    cutoff: float

@dataclass(frozen=True)
class TermGenerationConfig:
    mode: str
    n: int

@dataclass(frozen=True)
class WeightedAtomRef:
    atom_index: int
    denominator: int = 1

    @property
    def weight(self) -> Fraction:
        return Fraction(1, self.denominator)

    def format(self) -> str:
        if self.denominator == 1:
            return str(self.atom_index)
        return f"{self.atom_index}/{self.denominator}"

@dataclass
class ValidationReport:
    target: str
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [f"Target: {self.target}", f"Status: {'OK' if self.ok else 'FAILED'}"]
        if self.problems:
            lines.append("Problems:")
            lines.extend(f"- {item}" for item in self.problems)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines) + "\n"

@dataclass
class MonomerTemplate:
    path: Path
    preamble: List[str]
    beads: Dict[int, List[WeightedAtomRef]]
    bonds: List[Tuple[int, int]]
    constraints: List[Tuple[int, int]]
    angles: List[Tuple[int, int, int]]
    dihedrals: List[Tuple[int, int, int, int]]
    impropers: List[Tuple[int, int, int, int]]

    @property
    def bead_count(self) -> int:
        return len(self.beads)

    @property
    def atom_count(self) -> int:
        if not self.beads:
            return 0
        return max(ref.atom_index for refs in self.beads.values() for ref in refs)

@dataclass
class PolymerInputBundle:
    base: MonomerTemplate
    augmented: MonomerTemplate
    base_text: str
    augmented_text: str
    base_report: ValidationReport
    augmented_report: ValidationReport
    connection_bonds: List[Tuple[int, int]]
    connection_beads: List[int]
    backbone_beads: List[int]

@dataclass(frozen=True)
class ParamLine:
    section: str
    indices: Tuple[int, ...]
    tokens: Tuple[str, ...]
    commented: bool
    inline_comment: str
    rmsd: Optional[float]
    raw: str

@dataclass(frozen=True)
class TypedRecord:
    section: str
    category: str
    angle_dist: str
    type_names: Tuple[str, ...]
    display_labels: Tuple[str, ...]
    indices: Tuple[int, ...]
    tokens: Tuple[str, ...]
    commented: bool
    inline_comment: str
    rmsd: Optional[float]
    source_tag: str
    source_path: str

@dataclass
class MergedVariant:
    section: str
    category: str
    angle_dist: str
    type_names: Tuple[str, ...]
    display_labels: List[Tuple[str, ...]]
    tokens: Tuple[str, ...]
    commented: bool
    sources: List[str]
    indices_examples: List[Tuple[int, ...]]
    inline_comments: List[str]
    rmsd_values: List[float]
    primary: bool

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def shell_assign(name: str, value: str) -> str:
    return f'{name}={shlex.quote(value)}'

def resolve_under_base(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def resolve_connection_detection_config(pipeline_cfg: Dict[str, Any]) -> ConnectionDetectionConfig:
    indicator = str(pipeline_cfg.get("connection_indicator", "Br")).strip() or "Br"
    cutoff = float(pipeline_cfg.get("connection_cutoff", CONNECTION_CUTOFF))
    if cutoff <= 0:
        raise ValueError("bartender_pipeline.connection_cutoff must be > 0")
    return ConnectionDetectionConfig(
        indicator=indicator,
        cutoff=cutoff,
    )

def resolve_term_generation_config(pipeline_cfg: Dict[str, Any]) -> TermGenerationConfig:
    raw_cfg = pipeline_cfg.get("term_generation", {})
    if raw_cfg is None:
        raw_cfg = {}
    if isinstance(raw_cfg, str):
        mode = raw_cfg
        n = 0
    elif isinstance(raw_cfg, dict):
        mode = raw_cfg.get("mode", "all_unique")
        n = raw_cfg.get("n", 0)
    else:
        raise TypeError("bartender_pipeline.term_generation must be a mapping or string when provided")

    aliases = {
        "exhaustive": "all_unique",
        "original": "init_only",
        "all": "all_unique",
    }
    normalized_mode = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    supported = {"init_only", "all_unique", "polymer_backbone", "topology_n", "topology_swap_n"}
    if normalized_mode not in supported:
        raise ValueError(
            "bartender_pipeline.term_generation.mode must be one of "
            f"{sorted(supported)} (or alias exhaustive/original), got {mode!r}"
        )

    budget = int(n)
    if budget < 0:
        raise ValueError("bartender_pipeline.term_generation.n must be >= 0")
    return TermGenerationConfig(mode=normalized_mode, n=budget)

_WORKDIR_NAMES: Dict[Tuple[str, str], str] = {
    # (md, relaxation) -> workdir name
    ("bartender",       "xtb"):  "relax_xtb_geoopt",
    ("bartender",       "orca"): "relax_orca_geoopt",
    ("bartender",       "off"):  "relax_input_geometry",
    ("existing",        "xtb"):  "relax_xtb_geoopt_existing_traj",
    ("existing",        "orca"): "relax_orca_geoopt_existing_traj",
    ("existing",        "off"):  "existing_traj_refit",
    ("off",             "xtb"):  "relax_xtb_geoopt_only",
    ("off",             "orca"): "relax_orca_geoopt_only",
    ("off",             "off"):  "polymer_geometry_only",
    ("xtb_nobartender", "xtb"):  "relax_xtb_geoopt_xtb_nvt_only",
    ("xtb_nobartender", "orca"): "relax_orca_geoopt_xtb_nvt_only",
    ("xtb_nobartender", "off"):  "relax_xtb_nvt_only",
    ("xtb",             "xtb"):  "relax_xtb_geoopt_xtb_nvt",
    ("xtb",             "orca"): "relax_orca_geoopt_xtb_nvt",
    ("xtb",             "off"):  "relax_xtb_nvt",
}

def default_workdir_name(relaxation: str, md: str) -> str:
    key = (md, relaxation)
    name = _WORKDIR_NAMES.get(key)
    if name is not None:
        return name
    # md known but relaxation unexpected: fall back to off-relaxation default
    fallback = _WORKDIR_NAMES.get((md, "off"))
    if fallback is not None:
        return fallback
    raise ValueError(f"Unsupported md mode: {md}")

def _normalize_pipeline_mode(value: Any, default: str, field_name: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        if value is False:
            return "off"
        raise ValueError(f"{field_name} must be one of the documented string modes, not boolean true")
    return str(value).strip().lower()

def resolve_pipeline_modes(pipeline_cfg: Dict[str, Any]) -> Dict[str, str]:
    top_relaxation = pipeline_cfg.get("relaxation")
    top_md = pipeline_cfg.get("md")
    if (
        "workdir_name" in pipeline_cfg
        or top_md is not None
        or (top_relaxation is not None and not isinstance(top_relaxation, dict))
    ):
        relaxation = _normalize_pipeline_mode(
            top_relaxation,
            "xtb",
            "bartender_pipeline.relaxation",
        )
        md = _normalize_pipeline_mode(top_md, "bartender", "bartender_pipeline.md")
        workdir_name = str(
            pipeline_cfg.get("workdir_name") or default_workdir_name(relaxation, md)
        ).strip()
    else:
        mode_cfg = pipeline_cfg.get("mode")
        if isinstance(mode_cfg, dict):
            relaxation = _normalize_pipeline_mode(
                mode_cfg.get("relaxation"),
                "xtb",
                "bartender_pipeline.mode.relaxation",
            )
            md = _normalize_pipeline_mode(
                mode_cfg.get("md"),
                "bartender",
                "bartender_pipeline.mode.md",
            )
            workdir_name = str(
                mode_cfg.get("workdir_name") or default_workdir_name(relaxation, md)
            ).strip()
        else:
            legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
            if not isinstance(legacy_relax_cfg, dict):
                legacy_relax_cfg = {}
            legacy_bartender_cfg = pipeline_cfg.get("bartender", {})
            if not isinstance(legacy_bartender_cfg, dict):
                legacy_bartender_cfg = {}

            backend = _normalize_pipeline_mode(
                legacy_relax_cfg.get("backend"),
                "xtb",
                "bartender_pipeline.relaxation.backend",
            )
            if backend == "xtb":
                relaxation = "xtb"
            elif backend in {"orca", "orca_then_xtb"}:
                relaxation = "orca"
            elif backend == "off":
                relaxation = "off"
            else:
                raise ValueError(f"Unsupported legacy relaxation backend: {backend}")

            geometry_source = str(
                legacy_bartender_cfg.get("geometry_source", "polymer_xyz")
            ).strip()
            md = "xtb" if geometry_source == "relaxation_output" else "bartender"
            workdir_name = str(
                legacy_relax_cfg.get("workdir_name")
                or default_workdir_name(relaxation, md)
            ).strip()

    if relaxation not in {"xtb", "orca", "off"}:
        raise ValueError("bartender_pipeline.relaxation must be one of: xtb, orca, off")
    if md not in {"bartender", "xtb", "existing", "xtb_nobartender", "off"}:
        raise ValueError(
            "bartender_pipeline.md must be one of: bartender, xtb, existing, xtb_nobartender, off"
        )

    return {
        "relaxation": relaxation,
        "md": md,
        "workdir_name": workdir_name or default_workdir_name(relaxation, md),
    }

def resolve_spin_state(
    uhf_value: Any,
    multiplicity_value: Any,
    *,
    label: str,
) -> Tuple[int, int]:
    uhf = None if uhf_value is None else int(uhf_value)
    multiplicity = None if multiplicity_value is None else int(multiplicity_value)

    if uhf is None and multiplicity is None:
        return 0, 1
    if uhf is None:
        uhf = multiplicity - 1
    if multiplicity is None:
        multiplicity = uhf + 1

    if uhf < 0:
        raise ValueError(f"{label}: uhf must be >= 0")
    if multiplicity < 1:
        raise ValueError(f"{label}: multiplicity must be >= 1")
    if multiplicity != uhf + 1:
        raise ValueError(
            f"{label}: multiplicity ({multiplicity}) must equal uhf + 1 ({uhf + 1})"
        )
    return uhf, multiplicity

def _normalize_index_list(raw: Any, *, label: str) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, str)):
        values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values = list(raw)
    else:
        raise TypeError(f"{label} must be an integer or a list of integers")

    normalized: List[int] = []
    seen: set[int] = set()
    for value in values:
        index = int(value)
        if index < 0:
            raise ValueError(f"{label} must contain 0-based atom indices")
        converted = index + 1
        if converted not in seen:
            normalized.append(converted)
            seen.add(converted)
    return normalized

def resolve_backbone_atom_config(raw: Any, *, label: str) -> Dict[str, List[int]]:
    if raw is None:
        return {"head": [1], "tail": [2], "body": []}
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a mapping with optional head/body/tail atom lists")

    head = _normalize_index_list(raw.get("head"), label=f"{label}.head")
    tail = _normalize_index_list(raw.get("tail"), label=f"{label}.tail")
    body = _normalize_index_list(raw.get("body"), label=f"{label}.body")
    if not head and not tail:
        raise ValueError(f"{label} must define at least one of head or tail")
    return {"head": head, "tail": tail, "body": body}

def export_backbone_atom_config(cfg: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {
        key: [int(value) - 1 for value in cfg.get(key, [])]
        for key in ("head", "tail", "body")
    }

def normalize_monomer_configs(
    raw_monomers: Dict[str, Any],
    legacy_init_templates: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for token, raw_entry in raw_monomers.items():
        if isinstance(raw_entry, str):
            entry: Dict[str, Any] = {"xyz": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            raise TypeError(f"monomers.{token} must be a string or mapping")

        xyz = entry.get("xyz")
        if not xyz:
            raise ValueError(f"monomers.{token}.xyz is required")
        uhf, multiplicity = resolve_spin_state(
            entry.get("uhf"),
            entry.get("multiplicity"),
            label=f"monomers.{token}",
        )
        normalized[token] = {
            "xyz": str(xyz),
            "init_template": entry.get("init_template", legacy_init_templates.get(token)),
            "charge": int(entry.get("charge", 0)),
            "uhf": uhf,
            "multiplicity": multiplicity,
            "backbone_atoms": resolve_backbone_atom_config(
                entry.get("backbone_atoms"),
                label=f"monomers.{token}.backbone_atoms",
            ),
        }
    return normalized

def resolve_case_electronic_state(
    tokens: Sequence[str],
    monomer_cfg: Dict[str, Dict[str, Any]],
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, int]:
    inferred_charge = sum(int(monomer_cfg[token]["charge"]) for token in tokens)
    inferred_uhf = sum(int(monomer_cfg[token]["uhf"]) for token in tokens)
    state_cfg = pipeline_cfg.get("electronic_state")
    if state_cfg is None and isinstance(pipeline_cfg.get("relaxation"), dict):
        state_cfg = pipeline_cfg["relaxation"]
    if state_cfg is None:
        state_cfg = {}
    if not isinstance(state_cfg, dict):
        raise TypeError("bartender_pipeline.electronic_state must be a mapping")

    charge_raw = state_cfg.get("charge")
    charge = inferred_charge if charge_raw is None else int(charge_raw)

    if state_cfg.get("uhf") is None and state_cfg.get("multiplicity") is None:
        uhf = inferred_uhf
        multiplicity = inferred_uhf + 1
    else:
        uhf, multiplicity = resolve_spin_state(
            state_cfg.get("uhf"),
            state_cfg.get("multiplicity"),
            label="bartender_pipeline.electronic_state",
        )

    return {
        "charge": charge,
        "uhf": uhf,
        "multiplicity": multiplicity,
        "inferred_charge": inferred_charge,
        "inferred_uhf": inferred_uhf,
    }

def resolve_optional_path(base_dir: Path, raw_value: Any) -> Optional[Path]:
    value = str(raw_value or "").strip()
    if not value:
        return None
    return resolve_under_base(base_dir, value)

def resolve_xtb_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
    if not isinstance(legacy_relax_cfg, dict):
        legacy_relax_cfg = {}

    xtb_cfg = pipeline_cfg.get("xtb")
    if xtb_cfg is None:
        xtb_cfg = legacy_relax_cfg.get("xtb", {})
    if not isinstance(xtb_cfg, dict):
        raise TypeError("bartender_pipeline.xtb must be a mapping")
    md_cfg = xtb_cfg.get("md", {})
    if not isinstance(md_cfg, dict):
        raise TypeError("bartender_pipeline.xtb.md must be a mapping")

    solvent_model = str(xtb_cfg.get("solvent_model", "alpb")).strip().lower() or "off"
    if solvent_model not in {"off", "alpb", "gbsa"}:
        raise ValueError("bartender_pipeline.xtb.solvent_model must be one of: off, alpb, gbsa")

    return {
        "env_script": str(
            xtb_cfg.get("env_script", legacy_relax_cfg.get("xtb_env_script", ""))
        ).strip(),
        "binary": str(xtb_cfg.get("binary", legacy_relax_cfg.get("xtb_binary", "xtb"))).strip(),
        "gfn": int(xtb_cfg.get("gfn", 2)),
        "parallel": int(xtb_cfg.get("parallel", legacy_relax_cfg.get("nprocs", 32))),
        "opt_level": str(xtb_cfg.get("opt_level", "normal")).strip(),
        "opt_cycles": int(xtb_cfg.get("opt_cycles", 10000)),
        "acc": float(xtb_cfg.get("acc", 1.0)),
        "etemp": float(xtb_cfg.get("etemp", 300.0)),
        "solvent_model": solvent_model,
        "solvent": str(xtb_cfg.get("solvent", legacy_relax_cfg.get("solvent", "water"))).strip(),
        "solvent_reference": str(xtb_cfg.get("solvent_reference", "")).strip(),
        "md_input_template_path": str(xtb_cfg.get("md_input_template_path", "")).strip(),
        "md_temp_k": float(md_cfg.get("temp_k", legacy_relax_cfg.get("temp_k", 310.0))),
        "md_time_ps": float(md_cfg.get("time_ps", legacy_relax_cfg.get("time_ps", 5000))),
        "md_dump_fs": float(md_cfg.get("dump_fs", 50.0)),
        "md_step_fs": float(md_cfg.get("step_fs", 4.0)),
        "md_velo": parse_bool(md_cfg.get("velo", False)),
        "md_hmass": int(md_cfg.get("hmass", 4)),
        "md_shake": int(md_cfg.get("shake", 2)),
        "md_sccacc": float(md_cfg.get("sccacc", 2.0)),
        "md_restart": parse_bool(md_cfg.get("restart", False)),
    }

def resolve_orca_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
    if not isinstance(legacy_relax_cfg, dict):
        legacy_relax_cfg = {}

    orca_cfg = pipeline_cfg.get("orca")
    if orca_cfg is None:
        orca_cfg = legacy_relax_cfg.get("orca", {})
    if not isinstance(orca_cfg, dict):
        raise TypeError("bartender_pipeline.orca must be a mapping")
    return {
        "binary": str(orca_cfg.get("binary", legacy_relax_cfg.get("orca_binary", "orca"))).strip(),
        "nprocs": int(orca_cfg.get("nprocs", legacy_relax_cfg.get("nprocs", 32))),
        "method_line": str(
            orca_cfg.get(
                "method_line",
                f"{legacy_relax_cfg.get('orca_method', 'r2scan-3c')} CPCM({legacy_relax_cfg.get('solvent', 'water')}) Opt TightSCF",
            )
        ).strip(),
        "max_iter": int(orca_cfg.get("max_iter", 300)),
        "input_template_path": str(orca_cfg.get("input_template_path", "")).strip(),
    }

def _inspect_configured_executable(base_dir: Path, raw_value: Any) -> Dict[str, Any]:
    configured = str(raw_value or "").strip()
    if not configured:
        return {
            "configured": configured,
            "resolved": None,
            "exists": False,
            "lookup": "missing",
        }
    if "/" in configured or configured.startswith("."):
        path = resolve_under_base(base_dir, configured)
        fallback = None
        if not path.exists():
            fallback = shutil.which(path.name)
        return {
            "configured": configured,
            "resolved": fallback or str(path),
            "exists": path.exists() or fallback is not None,
            "lookup": "path->PATH" if fallback else "path",
        }
    found = shutil.which(configured)
    return {
        "configured": configured,
        "resolved": found,
        "exists": found is not None,
        "lookup": "PATH",
    }

def resolve_executable_command(base_dir: Path, raw_value: Any) -> str:
    payload = _inspect_configured_executable(base_dir, raw_value)
    resolved = str(payload.get("resolved") or "").strip()
    if resolved:
        return resolved
    return str(payload.get("configured") or "").strip()

def _inspect_optional_file(base_dir: Path, raw_value: Any) -> Dict[str, Any]:
    configured = str(raw_value or "").strip()
    if not configured:
        return {
            "configured": configured,
            "resolved": None,
            "exists": True,
            "lookup": "optional-empty",
        }
    path = resolve_under_base(base_dir, configured)
    return {
        "configured": configured,
        "resolved": str(path),
        "exists": path.exists(),
        "lookup": "path",
    }

def check_configured_tools(cfg: Dict[str, Any], requested: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    requested_tools = [str(value).strip().lower() for value in (requested or ("xtb", "orca", "bartender"))]
    base_dir = Path(str(cfg["paths"]["base_dir"])).resolve()
    pipeline_cfg = cfg.get("bartender_pipeline", {})
    if not isinstance(pipeline_cfg, dict):
        raise TypeError("bartender_pipeline must be a mapping")

    xtb_cfg = resolve_xtb_settings(pipeline_cfg)
    orca_cfg = resolve_orca_settings(pipeline_cfg)
    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        raise TypeError("bartender_pipeline.bartender must be a mapping")

    tools: List[Dict[str, Any]] = []
    if "xtb" in requested_tools:
        tools.append(
            {
                "name": "xtb",
                "binary": _inspect_configured_executable(base_dir, xtb_cfg.get("binary")),
                "env_script": _inspect_optional_file(base_dir, xtb_cfg.get("env_script")),
            }
        )
    if "orca" in requested_tools:
        tools.append(
            {
                "name": "orca",
                "binary": _inspect_configured_executable(base_dir, orca_cfg.get("binary")),
            }
        )
    if "bartender" in requested_tools:
        tools.append(
            {
                "name": "bartender",
                "binary": _inspect_configured_executable(base_dir, bartender_cfg.get("binary")),
                "env_script": _inspect_optional_file(base_dir, bartender_cfg.get("env_script")),
                "root": _inspect_optional_file(base_dir, bartender_cfg.get("root")),
            }
        )

    ok = True
    for tool in tools:
        ok = ok and bool(tool["binary"]["exists"])
    return {
        "ok": ok,
        "base_dir": str(base_dir),
        "tools": tools,
    }

def resolve_execution_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    exec_cfg = pipeline_cfg.get("execution", {})
    if exec_cfg is None:
        exec_cfg = {}
    if not isinstance(exec_cfg, dict):
        raise TypeError("bartender_pipeline.execution must be a mapping")

    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        bartender_cfg = {}

    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    use_srun = parse_bool(exec_cfg.get("use_srun", False), False)
    if use_srun and not slurm_enabled:
        slurm_enabled = True

    return {
        "run_relaxation": parse_bool(exec_cfg.get("run_relaxation", False)),
        "run_bartender": parse_bool(exec_cfg.get("run_bartender", bartender_cfg.get("execute", False))),
        "shell": str(exec_cfg.get("shell", "bash")).strip() or "bash",
        "slurm": slurm_enabled,
        "use_srun": use_srun,
    }

def _get_slurm_cpu_count() -> int:
    val = str(os.environ.get("SLURM_CPUS_PER_TASK", "")).strip()
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            return 0
    return 0

def resolve_log_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_cfg = pipeline_cfg.get("logs", {})
    if log_cfg is None:
        log_cfg = {}
    if not isinstance(log_cfg, dict):
        raise TypeError("bartender_pipeline.logs must be a mapping")

    return {
        "enabled": parse_bool(log_cfg.get("enabled", True), True),
        "dirname": str(log_cfg.get("dirname", "logs")).strip() or "logs",
        "write_validation": parse_bool(log_cfg.get("write_validation", True), True),
        "capture_runtime": parse_bool(log_cfg.get("capture_runtime", True), True),
    }

def ensure_case_logs_dir(case_dir: Path, log_cfg: Dict[str, Any]) -> Optional[Path]:
    if not log_cfg.get("enabled", True):
        return None
    logs_dir = case_dir / str(log_cfg["dirname"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def execute_case_script(
    label: str,
    script_path: Path,
    cwd: Path,
    exec_cfg: Dict[str, Any],
    logs_dir: Optional[Path],
) -> Dict[str, Any]:
    capture_runtime = bool(logs_dir) and bool(exec_cfg.get("capture_runtime", True))
    command = [str(exec_cfg.get("shell", "bash")), script_path.name]
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    use_srun = parse_bool(exec_cfg.get("use_srun", False), False)
    if slurm_enabled and use_srun:
        if not shutil.which("srun"):
             raise RuntimeError("execution.use_srun=true but 'srun' was not found in PATH")
        srun_command = ["srun", "--export=ALL", "--ntasks=1"]
        slurm_cpus = str(os.environ.get("SLURM_CPUS_PER_TASK", "")).strip()
        if slurm_cpus:
            srun_command.extend(["--cpus-per-task", slurm_cpus])
        command = srun_command + command

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture_runtime,
    )

    stdout_name = None
    stderr_name = None
    if capture_runtime and logs_dir is not None:
        stdout_name = f"{label}.stdout"
        stderr_name = f"{label}.stderr"
        write_text(logs_dir / stdout_name, result.stdout)
        write_text(logs_dir / stderr_name, result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() if capture_runtime else f"{label} failed with exit code {result.returncode}")

    return {
        "script": script_path.name,
        "cwd": str(cwd),
        "shell": str(exec_cfg.get("shell", "bash")),
        "slurm": slurm_enabled,
        "use_srun": use_srun,
        "command": command,
        "returncode": result.returncode,
        "stdout": stdout_name,
        "stderr": stderr_name,
    }

def render_xtb_md_input(md_mode: str, xtb_cfg: Dict[str, Any], template_text: Optional[str]) -> str:
    if template_text is not None:
        return template_text.rstrip() + "\n"
    if md_mode != "nvt":
        raise ValueError("xTB MD input generation only supports md_mode 'nvt'")
    return (
        "$md\n"
        f" temp={xtb_cfg['md_temp_k']:.3f}\n"
        f" time={xtb_cfg['md_time_ps']:.3f}\n"
        f" dump={xtb_cfg['md_dump_fs']:.3f}\n"
        f" step={xtb_cfg['md_step_fs']:.3f}\n"
        f" velo={'true' if xtb_cfg['md_velo'] else 'false'}\n"
        " nvt=true\n"
        f" hmass={xtb_cfg['md_hmass']}\n"
        f" shake={xtb_cfg['md_shake']}\n"
        f" sccacc={xtb_cfg['md_sccacc']:.3f}\n"
        f" restart={'true' if xtb_cfg['md_restart'] else 'false'}\n"
        "$end\n"
    )

def render_orca_input(
    local_xyz_name: str,
    state: Dict[str, Any],
    orca_cfg: Dict[str, Any],
    template_text: Optional[str],
) -> str:
    if template_text is not None:
        nprocs = int(orca_cfg.get("nprocs", 1))
        if nprocs > 1:
            if re.search(r"%pal\s+nprocs\s+\d+\s+end", template_text, re.IGNORECASE):
                template_text = re.sub(
                    r"%pal\s+nprocs\s+\d+\s+end",
                    f"%pal nprocs {nprocs} end",
                    template_text,
                    flags=re.IGNORECASE,
                )
            elif not re.search(r"%pal", template_text, re.IGNORECASE):
                template_text = f"%pal nprocs {nprocs} end\n{template_text}"

        return template_text.rstrip() + "\n\n" + (
            f"* xyzfile {int(state['charge'])} {int(state.get('multiplicity', 1))} {local_xyz_name}\n"
        )
    method_line = orca_cfg["method_line"].strip()
    if not method_line.startswith("!"):
        method_line = f"! {method_line}"
    return (
        f"{method_line}\n"
        f"%pal nprocs {int(orca_cfg['nprocs'])} end\n\n"
        "%geom\n"
        f"   MaxIter {int(orca_cfg['max_iter'])}\n"
        "end\n\n"
        f"* xyzfile {int(state['charge'])} {int(state['multiplicity'])} {local_xyz_name}\n"
    )

def normalize_sequence(sequence: Sequence[str] | str) -> List[str]:
    if isinstance(sequence, str):
        text = sequence.strip()
        if not text:
            raise ValueError("Sequence is empty.")
        if "," in text:
            tokens = [token.strip() for token in text.split(",") if token.strip()]
        elif " " in text:
            tokens = [token for token in text.split() if token]
        else:
            tokens = list(text)
    else:
        tokens = [str(token).strip() for token in sequence if str(token).strip()]
    if not tokens:
        raise ValueError("Sequence produced no tokens.")
    return tokens

def sequence_stem(tokens: Sequence[str]) -> str:
    if all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return "_".join(tokens)

def parse_sequence_entry(entry: Any, monomer_keys: set[str]) -> List[str]:
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            raise ValueError("Empty sequence entry is not allowed")
        if "," in text:
            tokens = parse_csv_list(text)
        elif " " in text:
            tokens = [token for token in text.split() if token]
        elif text in monomer_keys:
            tokens = [text]
        else:
            tokens = list(text)
    elif isinstance(entry, (list, tuple)):
        tokens = [str(token).strip() for token in entry if str(token).strip()]
    else:
        raise TypeError(f"Unsupported sequence entry type: {type(entry)!r}")

    if not tokens:
        raise ValueError("Sequence entry produced no tokens")
    return tokens

def build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str]) -> List[List[str]]:
    explicit_sequences = system_cfg.get("sequences")
    if explicit_sequences is not None:
        if not isinstance(explicit_sequences, list):
            raise ValueError("system.sequences must be a list when provided")
        jobs = [parse_sequence_entry(entry, monomer_keys) for entry in explicit_sequences]
        if not jobs:
            raise ValueError("system.sequences is empty")
        return jobs

    symbols = list(system_cfg["symbols"])
    lengths = list(system_cfg["lengths"])
    jobs: List[List[str]] = []
    for symbol in symbols:
        for repeat in lengths:
            jobs.append([symbol] * int(repeat))
    return jobs

def parse_xyz(path: Path) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty xyz: {path}")
    natoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ {path} declares {natoms} atoms but contains {len(atom_lines)} coordinates.")
    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords

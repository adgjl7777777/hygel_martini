"""Existing-data fitting for stage 02: trim + Bartender refit job setup.

Owns the ``workflow.mode=existing_data_fit`` branch: it consumes
user-provided OPLS/GROMACS trajectory cases (``opls_data.cases``) and
writes, per case, a ``trim/run_prepare_md.sh`` script (trjconv/energy
extraction + equilibration trim via ``gromacs_traj_to_pdb``) and a
``bartender_job/run_bartender.sh`` refit script, plus case.json /
run_all.sh / summary.json bookkeeping.  Depending on the
``opls_data.execution`` preset the scripts are either only written or
also executed immediately.  Also provides the ``--check-gmx`` /
``--check-bartender`` tool probes and the ``--postprocess-only``
screening entry (delegating to qm_to_martini.postprocess).

Called by ``generator.run_opls_to_martini``; no MD production runs are
launched from here.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hygel_martini.param_opt.qm_to_martini.postprocess import run_screening_postprocess

from .writers import write_text


def _bool(value: Any, default: bool = False) -> bool:
    """Coerce config values to bool ("1"/"true"/"yes"/"on" are truthy strings)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_path(base_dir: Path, value: Any, *, required: bool = True) -> Path | None:
    """Resolve a config path against base_dir (absolute paths pass through).

    Returns None for empty/missing values when not required.

    Raises:
        ValueError: If the value is empty but required.
    """
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError("A required path value is empty.")
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _q(value: str | Path) -> str:
    """Shell-quote a value for the generated bash scripts."""
    return shlex.quote(str(value))


def _rel(path: Path, start: Path) -> str:
    """Relative path from start to path (scripts cd into their own dir)."""
    return os.path.relpath(str(path), start=str(start))


def _repo_root() -> Path:
    """Installable package root (3 levels up), exported as PYTHONPATH."""
    return Path(__file__).resolve().parents[3]


def _merge_case_variant(case: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay one variant dict on its parent case (shallow; drops "variants")."""
    merged = dict(case)
    merged.update(variant)
    merged.pop("variants", None)
    return merged


def _iter_case_variants(cases: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Expand opls_data.cases: yield each case, or one merged dict per variant.

    Raises:
        TypeError: If a variants[] entry is not a mapping.
    """
    for case in cases:
        variants = case.get("variants")
        if variants:
            for variant in variants:
                if not isinstance(variant, dict):
                    raise TypeError("Each opls_data.cases[].variants[] entry must be a mapping.")
                yield _merge_case_variant(case, variant)
        else:
            yield dict(case)


def _mode_tag(case: Dict[str, Any]) -> str:
    """Middle directory-name component: mode_tag > mode > name > "default"."""
    return str(case.get("mode_tag") or case.get("mode") or case.get("name") or "default").strip()


def _label(case: Dict[str, Any]) -> str:
    """Case label used for directory names: label > sequence > "CASE"."""
    return str(case.get("label") or case.get("sequence") or "CASE").strip()


def _case_dir(out_root: Path, case: Dict[str, Any]) -> Path:
    """Case directory layout: <out_root>/<label>/<mode_tag>/<label>."""
    return out_root / _label(case) / _mode_tag(case) / _label(case)


def _resolve_md_mode(cfg: Dict[str, Any]) -> str:
    """Normalize bartender_pipeline.md to one of md / md_notrim / trim / off.

    Stage 02 names the trajectory source "md" (stage 03 calls it "xtb");
    legacy aliases (existing, gromacs, bartender-noxtb, ...) are mapped.

    Raises:
        ValueError: For an unrecognized mode string.
    """
    pipeline = cfg.get("bartender_pipeline", {})
    mode = str(pipeline.get("md", "md")).strip().lower()
    aliases = {
        "existing": "md",
        "existing_notrim": "md_notrim",
        "gromacs": "md",
        "gromacs_notrim": "md_notrim",
        "bartender_noxtb": "md",
        "bartender-noxtb": "md",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"md", "md_notrim", "trim", "off"}:
        raise ValueError("For 02 existing_data_fit, bartender_pipeline.md must be one of md, md_notrim, trim, off.")
    return mode


def _apply_execution_preset(cfg: Dict[str, Any]) -> str:
    """Apply the opls_data.execution.mode preset onto the config in place.

    A single mode alias (e.g. "setup", "md", "md_notrim", "trim",
    "off"; many spelling variants accepted) expands into four knobs at
    once: ``bartender_pipeline.md``, ``bartender_pipeline.bartender.
    enabled``, ``execution.run_trim`` and ``execution.run_bartender``.
    The resolved values are also recorded under ``execution.effective``
    for the summary output.  An empty mode leaves the config untouched.

    Returns:
        The normalized mode string, or "" when no preset was requested.

    Raises:
        ValueError: For an unknown mode alias.
        TypeError: If the touched config sections are not mappings.
    """
    data_cfg = cfg.setdefault("opls_data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("opls_data must be a mapping")
    execution = data_cfg.setdefault("execution", {})
    if not isinstance(execution, dict):
        raise TypeError("opls_data.execution must be a mapping")

    raw_mode = str(execution.get("mode", "") or "").strip().lower().replace("-", "_")
    if not raw_mode:
        return ""

    presets: Dict[str, Dict[str, Any]] = {
        "setup": {"md": "md", "bartender": True, "run_trim": False, "run_bartender": False},
        "setup_md": {"md": "md", "bartender": True, "run_trim": False, "run_bartender": False},
        "setup_notrim": {"md": "md_notrim", "bartender": True, "run_trim": False, "run_bartender": False},
        "setup_md_notrim": {"md": "md_notrim", "bartender": True, "run_trim": False, "run_bartender": False},
        "md": {"md": "md", "bartender": True, "run_trim": True, "run_bartender": True},
        "trim_md": {"md": "md", "bartender": True, "run_trim": True, "run_bartender": True},
        "bartender_noxtb": {"md": "md", "bartender": True, "run_trim": True, "run_bartender": True},
        "run": {"md": "md", "bartender": True, "run_trim": True, "run_bartender": True},
        "both": {"md": "md", "bartender": True, "run_trim": True, "run_bartender": True},
        "bartender": {"md": "md", "bartender": True, "run_trim": False, "run_bartender": True},
        "bartender_trim": {"md": "md", "bartender": True, "run_trim": False, "run_bartender": True},
        "md_notrim": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "md_no_trim": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "notrim": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "notrim_md": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "no_trim_md": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "bartender_noxtb_notrim": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "bartender_notrim_noxtb": {"md": "md_notrim", "bartender": True, "run_trim": True, "run_bartender": True},
        "bartender_notrim": {"md": "md_notrim", "bartender": True, "run_trim": False, "run_bartender": True},
        "notrim_bartender": {"md": "md_notrim", "bartender": True, "run_trim": False, "run_bartender": True},
        "no_trim_bartender": {"md": "md_notrim", "bartender": True, "run_trim": False, "run_bartender": True},
        "trim": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "trim_only": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "prepare": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "prepare_only": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "nobartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "no_bartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "md_nobartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "md_no_bartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "trim_nobartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "trim_no_bartender": {"md": "trim", "bartender": False, "run_trim": True, "run_bartender": False},
        "notrim_nobartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "notrim_no_bartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "md_notrim_nobartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "md_notrim_no_bartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "no_trim_nobartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "no_trim_no_bartender": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "prepare_notrim": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "prepare_no_trim": {"md": "md_notrim", "bartender": False, "run_trim": True, "run_bartender": False},
        "off": {"md": "off", "bartender": False, "run_trim": False, "run_bartender": False},
        "metadata": {"md": "off", "bartender": False, "run_trim": False, "run_bartender": False},
        "scaffold": {"md": "off", "bartender": False, "run_trim": False, "run_bartender": False},
    }
    if raw_mode not in presets:
        allowed = ", ".join(sorted(presets))
        raise ValueError(f"Unknown opls_data.execution.mode={raw_mode!r}. Allowed aliases: {allowed}")

    preset = presets[raw_mode]
    pipeline = cfg.setdefault("bartender_pipeline", {})
    if not isinstance(pipeline, dict):
        raise TypeError("bartender_pipeline must be a mapping")
    bartender_cfg = pipeline.setdefault("bartender", {})
    if not isinstance(bartender_cfg, dict):
        raise TypeError("bartender_pipeline.bartender must be a mapping")

    pipeline["md"] = preset["md"]
    bartender_cfg["enabled"] = preset["bartender"]
    execution["run_trim"] = preset["run_trim"]
    execution["run_bartender"] = preset["run_bartender"]
    execution["mode"] = raw_mode
    execution["effective"] = {
        "bartender_pipeline.md": preset["md"],
        "bartender_pipeline.bartender.enabled": preset["bartender"],
        "run_trim": preset["run_trim"],
        "run_bartender": preset["run_bartender"],
    }
    return raw_mode


def _resolve_tool(cfg: Dict[str, Any], name: str, default: str) -> str:
    """Look up a tool command in cfg["tools"], falling back to default."""
    tools = cfg.get("tools", {})
    if not isinstance(tools, dict):
        tools = {}
    return str(tools.get(name) or default)


def check_existing_data_tools(cfg: Dict[str, Any], tools: Iterable[str]) -> Dict[str, Any]:
    """Probe the configured gmx / bartender commands without running them.

    Resolution order per tool: existing base_dir-relative path, then
    PATH lookup (shutil.which), then existing absolute path.

    Args:
        cfg: Merged stage-02 config.
        tools: Subset of {"gmx", "bartender"} to check.

    Returns:
        {"ok": all_found, "tools": [{name, configured, resolved, exists}]}.
    """
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    pipeline = cfg.get("bartender_pipeline", {})
    bartender_cfg = pipeline.get("bartender", {}) if isinstance(pipeline, dict) else {}
    gmx_cmd = _resolve_tool(cfg, "gmx", "gmx_mpi")
    bartender_bin = str(bartender_cfg.get("binary", "bartender"))
    configured = {
        "gmx": gmx_cmd,
        "bartender": bartender_bin,
    }
    checks = []
    for name in tools:
        value = configured[name]
        path = Path(value)
        resolved = str((base_dir / path).resolve()) if not path.is_absolute() and (base_dir / path).exists() else shutil.which(value)
        if path.is_absolute() and path.exists():
            resolved = str(path)
        checks.append({"name": name, "configured": value, "resolved": resolved, "exists": bool(resolved)})
    return {"ok": all(item["exists"] for item in checks), "tools": checks}


def _prepare_script_lines(
    cfg: Dict[str, Any],
    case: Dict[str, Any],
    trim_dir: Path,
    base_dir: Path,
    md_mode: str,
) -> tuple[List[str], Path]:
    """Assemble the bash lines of trim/run_prepare_md.sh for one case.

    The script (executed from trim_dir) converts the case trajectory to
    a multi-model PDB (``gmx trjconv`` for non-PDB inputs, plain copy
    otherwise), extracts the configured energy term to energy.xvg when
    an .edr is given, and calls ``gromacs_traj_to_pdb`` to write the
    trimmed ``md_traj.pdb``.  ``--auto-trim`` is suppressed when
    md_mode is "md_notrim".  Case keys override the ``opls_data.trim``
    defaults for the per-case knobs (selections, energy_term, ...).

    Returns:
        (script lines, path of the trimmed PDB the script will write).

    Raises:
        ValueError: When the trajectory is missing, or a non-PDB
            trajectory has no tpr for trjconv.
    """
    data_cfg = cfg.get("opls_data", {})
    trim_cfg = data_cfg.get("trim", {}) if isinstance(data_cfg, dict) else {}
    tools_cfg = cfg.get("tools", {})
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}

    trajectory = _as_path(base_dir, case.get("trajectory") or case.get("md_traj"), required=md_mode in {"md", "md_notrim", "trim"})
    tpr = _as_path(base_dir, case.get("tpr"), required=False)
    edr = _as_path(base_dir, case.get("edr"), required=False)
    gmx_cmd = str(tools_cfg.get("gmx", "gmx_mpi"))
    gmxrc_path = str(tools_cfg.get("gmxrc_path", "")).strip()
    selections = case.get("trjconv_selections") or trim_cfg.get("trjconv_selections") or ["System"]
    if isinstance(selections, str):
        selections = [selections]
    trjconv_extra = case.get("trjconv_extra") or trim_cfg.get("trjconv_extra") or []
    if isinstance(trjconv_extra, str):
        trjconv_extra = [trjconv_extra]
    energy_term = str(case.get("energy_term") or trim_cfg.get("energy_term") or "Potential")

    auto_trim = _bool(case.get("auto_trim", trim_cfg.get("auto_trim", True)), True)
    write_plots = _bool(trim_cfg.get("write_plots", True), True)
    skip_frames = int(case.get("skip_frames", trim_cfg.get("skip_frames", 0)))
    nskip = int(trim_cfg.get("nskip", 1))
    max_trim_fraction = float(trim_cfg.get("max_trim_fraction", 1.0))
    trim_method = str(trim_cfg.get("method", "pymbar"))
    ref_fraction = float(trim_cfg.get("ref_fraction", 0.2))
    threshold_sigma = float(trim_cfg.get("threshold_sigma", 1.0))
    fast = _bool(trim_cfg.get("fast", True), True)

    source_pdb = trim_dir / "source_traj.pdb"
    output_pdb = trim_dir / "md_traj.pdb"
    energy_xvg = trim_dir / "energy.xvg"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        "PYTHON_BIN=${PYTHON_BIN:-python3}",
        f"export PYTHONPATH={_q(_repo_root())}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"GMX_CMD=${{GMX_CMD:-{_q(gmx_cmd)}}}",
    ]
    if gmxrc_path:
        lines.extend(
            [
                "set +u",
                f"if [ -f {_q(gmxrc_path)} ]; then",
                f"  source {_q(gmxrc_path)}",
                "fi",
                "set -u",
            ]
        )

    if trajectory is None:
        raise ValueError("trajectory is required for md/md_notrim/trim modes")
    traj_suffix = trajectory.suffix.lower()
    if traj_suffix == ".pdb":
        lines.append(f"cp {_q(_rel(trajectory, trim_dir))} source_traj.pdb")
    else:
        if tpr is None:
            raise ValueError(f"Non-PDB trajectory requires a tpr path: {trajectory}")
        selection_text = "\\n".join(str(item) for item in selections) + "\\n"
        lines.append(
            f"printf '%b' {_q(selection_text)} | \"$GMX_CMD\" trjconv"
            f" -s {_q(_rel(tpr, trim_dir))}"
            f" -f {_q(_rel(trajectory, trim_dir))}"
            " -o source_traj.pdb"
            f" {' '.join(_q(str(item)) for item in trjconv_extra)}"
        )

    if edr is not None:
        lines.append(
            f"printf '%s\\n' {_q(energy_term)} | \"$GMX_CMD\" energy"
            f" -f {_q(_rel(edr, trim_dir))} -o energy.xvg -xvg none || true"
        )
    else:
        lines.append(": > energy.xvg")

    auto_flag = " --auto-trim" if (auto_trim and md_mode != "md_notrim") else ""
    fast_flag = " --fast" if fast else ""
    plots_flag = "" if write_plots else " --no-plots"
    lines.append(
        "\"$PYTHON_BIN\" -m hygel_martini.param_opt.opls_to_martini.gromacs_traj_to_pdb"
        f" {_q(source_pdb.name)} {_q(output_pdb.name)}"
        " --energy-xvg energy.xvg"
        f"{auto_flag}"
        f" --skip-frames {skip_frames}"
        f" --nskip {nskip}"
        f" --max-trim-fraction {max_trim_fraction}"
        f" --trim-method {_q(trim_method)}"
        f" --ref-fraction {ref_fraction}"
        f" --threshold-sigma {threshold_sigma}"
        f"{fast_flag}{plots_flag}"
    )
    return lines, output_pdb


def _write_prepare_md_job(
    cfg: Dict[str, Any],
    case: Dict[str, Any],
    case_dir: Path,
    base_dir: Path,
    md_mode: str,
) -> Path | None:
    """Write trim/run_prepare_md.sh for one case (skipped when md_mode=off).

    Returns:
        Path of the trimmed PDB the script will produce, or None.
    """
    if md_mode == "off":
        return None
    trim_dir = case_dir / "trim"
    trim_dir.mkdir(parents=True, exist_ok=True)
    lines, output_pdb = _prepare_script_lines(cfg, case, trim_dir, base_dir, md_mode)
    script = trim_dir / "run_prepare_md.sh"
    write_text(script, "\n".join(lines) + "\n")
    script.chmod(0o755)
    return output_pdb


def _write_bartender_job(
    cfg: Dict[str, Any],
    case: Dict[str, Any],
    case_dir: Path,
    base_dir: Path,
    md_mode: str,
    trajectory_pdb: Path | None,
) -> Path | None:
    """Write the per-case Bartender refit job directory and script.

    Skipped (returns None) for md_mode trim/off or when the pipeline
    disables Bartender.  Copies the .inp next to the script, then emits
    ``run_bartender.sh`` invoking ``bartender ... -owntraj <trimmed pdb>
    -refit`` (with optional -skip), honoring BTROOT / env_script /
    HYGEL_BARTENDER_CPUS overrides, and a bartender_job.json manifest.

    Returns:
        The job directory, or None when the job is not applicable.

    Raises:
        ValueError: If geometry, bartender_inp, or the prepared
            trajectory is missing while Bartender is enabled.
    """
    if md_mode in {"trim", "off"}:
        return None
    pipeline = cfg.get("bartender_pipeline", {})
    bartender_cfg = pipeline.get("bartender", {}) if isinstance(pipeline, dict) else {}
    if not _bool(bartender_cfg.get("enabled", True), True):
        return None

    geometry = _as_path(base_dir, case.get("geometry") or case.get("reference_geometry"), required=True)
    inp = _as_path(base_dir, case.get("bartender_inp") or case.get("inp"), required=True)
    if geometry is None or inp is None or trajectory_pdb is None:
        raise ValueError("geometry, bartender_inp, and prepared trajectory are required for Bartender.")

    outdir = case_dir / str(bartender_cfg.get("output_dirname", "bartender_job"))
    outdir.mkdir(parents=True, exist_ok=True)
    local_inp = outdir / inp.name
    local_inp.write_text(inp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    bartender_bin = str(bartender_cfg.get("binary", "bartender"))
    bartender_root = str(bartender_cfg.get("root", "")).strip()
    env_script = str(bartender_cfg.get("env_script", "")).strip()
    cpus = int(bartender_cfg.get("cpus", 1))
    charge = int(case.get("charge", bartender_cfg.get("charge", 0) or 0))
    skip = int(bartender_cfg.get("skip", 1))

    traj_rel = _rel(trajectory_pdb, outdir)
    geometry_rel = _rel(geometry, outdir)
    script = outdir / "run_bartender.sh"
    args = ["-cpus", "\"$HYGEL_BARTENDER_CPUS\"", "-charge", str(charge), "-owntraj", traj_rel, "-refit"]
    if skip > 1:
        args += ["-skip", str(skip)]
    args += [geometry_rel, local_inp.name]
    command = ["$BARTENDER_BIN"] + [_q(item) if item != "\"$HYGEL_BARTENDER_CPUS\"" else item for item in args]

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"export HYGEL_BARTENDER_CPUS=${{HYGEL_BARTENDER_CPUS:-{cpus}}}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$HYGEL_BARTENDER_CPUS}",
        f"BARTENDER_BIN=${{BARTENDER_BIN:-{_q(bartender_bin)}}}",
    ]
    if bartender_root:
        lines.append(f"export BTROOT={_q(bartender_root)}")
    if env_script:
        lines.extend(
            [
                "set +u",
                f"if [ -f {_q(env_script)} ]; then",
                f"  source {_q(env_script)}",
                "fi",
                "set -u",
            ]
        )
    lines.append(" ".join(command))
    write_text(script, "\n".join(lines) + "\n")
    script.chmod(0o755)

    manifest = {
        "mode": md_mode,
        "geometry": geometry_rel,
        "inp": local_inp.name,
        "trajectory": traj_rel,
        "command": [bartender_bin] + args,
        "outdir": str(outdir),
    }
    write_text(outdir / "bartender_job.json", json.dumps(manifest, indent=2))
    return outdir


def _run_script(path: Path) -> None:
    """Execute a generated bash script in its own directory (check=True)."""
    subprocess.run(["bash", str(path)], cwd=str(path.parent), check=True)


def run_existing_data_fit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare (and optionally run) trim + Bartender jobs for every case.

    Applies the execution preset, expands case variants, writes the
    per-case scripts plus case.json, aggregates everything into
    ``run_all.sh`` and ``summary.json`` under ``paths.out_root``, and
    executes the trim/Bartender scripts inline when the preset sets
    run_trim/run_bartender (a missing trimmed PDB is prepared on demand
    before a Bartender run).

    Args:
        cfg: Merged stage-02 config with opls_data.cases populated.

    Returns:
        Summary dict (settings, execution/md modes, case records,
        run_all path); also written as summary.json.

    Raises:
        ValueError: When opls_data.cases is empty.
        TypeError: When opls_data is not a mapping.
        subprocess.CalledProcessError: If an inline script run fails.
    """
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("opls_data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("opls_data must be a mapping")
    cases = data_cfg.get("cases") or []
    if not cases:
        raise ValueError("workflow.mode=existing_data_fit requires opls_data.cases")
    execution_mode = _apply_execution_preset(cfg)
    md_mode = _resolve_md_mode(cfg)
    execution = data_cfg.get("execution", {}) if isinstance(data_cfg.get("execution", {}), dict) else {}
    run_trim = _bool(execution.get("run_trim", False), False)
    run_bartender = _bool(execution.get("run_bartender", False), False)

    records: List[Dict[str, Any]] = []
    run_all_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd \"$(dirname \"$0\")\""]
    for case in _iter_case_variants(cases):
        label = _label(case)
        mode_tag = _mode_tag(case)
        case_dir = _case_dir(out_root, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        prepared_pdb = _write_prepare_md_job(cfg, case, case_dir, base_dir, md_mode)
        bartender_dir = _write_bartender_job(cfg, case, case_dir, base_dir, md_mode, prepared_pdb)

        case_json = {
            "label": label,
            "mode_tag": mode_tag,
            "case_dir": str(case_dir),
            "input": case,
            "trim": {
                "script": "trim/run_prepare_md.sh" if md_mode != "off" else None,
                "prepared_pdb": str(prepared_pdb) if prepared_pdb is not None else None,
            },
            "bartender": {
                "job_dir": str(bartender_dir) if bartender_dir is not None else None,
                "script": "bartender_job/run_bartender.sh" if bartender_dir is not None else None,
            },
        }
        write_text(case_dir / "case.json", json.dumps(case_json, indent=2, ensure_ascii=False))

        if prepared_pdb is not None:
            rel_script = _rel(case_dir / "trim" / "run_prepare_md.sh", out_root)
            run_all_lines.append(f"bash {_q(rel_script)}")
            if run_trim:
                _run_script(case_dir / "trim" / "run_prepare_md.sh")
        if bartender_dir is not None:
            rel_script = _rel(bartender_dir / "run_bartender.sh", out_root)
            run_all_lines.append(f"bash {_q(rel_script)}")
            if run_bartender:
                if not run_trim and prepared_pdb is not None and not prepared_pdb.exists():
                    _run_script(case_dir / "trim" / "run_prepare_md.sh")
                _run_script(bartender_dir / "run_bartender.sh")

        records.append(case_json)

    run_all = out_root / "run_all.sh"
    write_text(run_all, "\n".join(run_all_lines) + "\n")
    run_all.chmod(0o755)

    result = {
        "settings": cfg,
        "workflow": "existing_data_fit",
        "execution_mode": execution_mode,
        "md_mode": md_mode,
        "cases": records,
        "run_all": str(run_all),
    }
    write_text(out_root / "summary.json", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_postprocess_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run only the Bartender screening postprocess on existing results.

    Delegates to ``qm_to_martini.postprocess.run_screening_postprocess``
    and writes postprocess_summary.json under
    ``paths.postprocess_output_root`` (falling back to out_root).

    Raises:
        ValueError: If screening is not enabled in the config.
    """
    post_cfg = cfg.get("bartender_pipeline", {}).get("postprocess", {})
    if not post_cfg.get("screening", {}).get("enabled", False):
        raise ValueError(
            "Postprocess-only mode runs screening only. Set bartender_pipeline.postprocess.screening.enabled=true."
        )
    summary = {
        "settings": cfg,
        "postprocess_only": True,
        "screening": run_screening_postprocess(cfg),
    }
    summary_root_value = cfg["paths"].get("postprocess_output_root") or cfg["paths"].get("out_root") or "."
    summary_root = Path(str(summary_root_value)).resolve()
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "postprocess_summary.json"
    summary["summary_json"] = str(summary_path)
    write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

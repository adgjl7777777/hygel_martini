from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..polymer_maker.maker import build_polymer as build_polymer_structure
from ..polymer_maker.maker import load_monomer_library

from .postprocess import run_screening_postprocess
from .config import (
    _get_slurm_cpu_count,
    build_sequence_jobs,
    ensure_case_logs_dir,
    execute_case_script,
    normalize_monomer_configs,
    parse_bool,
    render_orca_input,
    render_xtb_md_input,
    resolve_case_electronic_state,
    resolve_connection_detection_config,
    resolve_executable_command,
    resolve_execution_settings,
    resolve_log_settings,
    resolve_optional_path,
    resolve_orca_settings,
    resolve_pipeline_modes,
    resolve_term_generation_config,
    resolve_under_base,
    resolve_xtb_settings,
    parse_xyz,
    sequence_stem,
    shell_assign,
    write_text,
)

from .workflow_logic.loader import (
    parse_bartender_inp,
    validate_template,
    infer_connection_metadata,
)
from .workflow_logic.builder import build_polymer_input
from .workflow_logic.merger import (
    summarize_itp,
    merge_records,
    write_merged_forcefield,
    merged_summary_payload,
    typed_records_for_result,
)

_XTB_TRAJ_TO_PDB_SRC = Path(__file__).resolve().parents[2] / "tools" / "xtb_traj_to_pdb.py"

def _srun_reentry_lines(exec_cfg: Dict[str, Any], cpu_fallback_var: str) -> List[str]:
    if not (parse_bool(exec_cfg.get("slurm", False), False) and parse_bool(exec_cfg.get("use_srun", False), False)):
        return []
    return [
        "if [ -n \"${SLURM_JOB_ID:-}\" ] && [ -z \"${SLURM_STEP_ID:-}\" ]; then",
        "  if ! command -v srun >/dev/null 2>&1; then",
        "    echo \"[ERROR] execution.use_srun=true but srun was not found\" >&2",
        "    exit 1",
        "  fi",
        f"  exec srun --export=ALL --ntasks=1 --cpus-per-task \"${{SLURM_CPUS_PER_TASK:-${cpu_fallback_var}}}\" bash \"$0\" \"$@\"",
        "fi",
    ]

def _bartender_mode_args(
    flow: Dict[str, str],
    bartender_cfg: Dict[str, Any],
    bartender_charge: int,
    skip: int,
    trajectory: Optional[Path],
    outdir: Path,
) -> List[str]:
    """md 모드에 따른 Bartender CLI 인자 목록 반환 (quoting 없이 raw 값)."""
    args: List[str] = ["-charge", str(int(bartender_charge))]
    if flow["md"] == "bartender":
        args += ["-method", "gfn2",
                 "-time", str(int(bartender_cfg.get("time_ps", 5000))),
                 "-temperature", f"{float(bartender_cfg.get('temperature_k', 310.0)):.3f}"]
        solvent = str(bartender_cfg.get("solvent", "h2o")).strip()
        if solvent:
            args += ["-solvent", solvent]
        dcd_save = str(bartender_cfg.get("dcd_save", "")).strip()
        if dcd_save:
            args += ["-dcdSave", dcd_save]
        if skip > 1:
            args += ["-skip", str(skip)]
    elif flow["md"] in {"xtb", "existing", "existing_notrim"}:
        if trajectory is None:
            raise ValueError("Trajectory reuse mode requires a trajectory path.")
        trajectory_arg = os.path.relpath(str(trajectory), start=str(outdir))
        args += ["-owntraj", trajectory_arg, "-refit"]
        if skip > 1:
            args += ["-skip", str(skip)]
    else:
        raise ValueError(f"Unsupported md mode: {flow['md']}")
    return args

def prepare_relaxation_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    pipeline_cfg: Dict[str, Any],
    base_dir: Path,
    exec_cfg: Dict[str, Any],
) -> Optional[Path]:
    if flow["relaxation"] == "off" and flow["md"] not in {"xtb", "xtb_nobartender", "xtb_nobartender_notrim"}:
        case["relaxation"] = {
            "mode": flow["relaxation"],
            "md": flow["md"],
            "workdir": None,
            "run_script": None,
            "geometry_hint": str(case["artifacts"]["polymer_xyz"]),
            "trajectory_hint": None,
            "geometry_opt": False,
        }
        return None

    workdir_name = flow["workdir_name"]
    workdir = case_dir / workdir_name
    workdir.mkdir(parents=True, exist_ok=True)

    xyz_name = str(case["artifacts"].get("relax_input_xyz") or case["artifacts"]["polymer_xyz"])
    polymer_xyz = case_dir / xyz_name
    local_xyz = workdir / polymer_xyz.name
    local_xyz.write_text(polymer_xyz.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        raise TypeError("case.electronic_state must be a mapping")
    charge = int(state.get("charge", 0))
    uhf = int(state.get("uhf", 0))
    xtb_cfg = resolve_xtb_settings(pipeline_cfg)
    orca_cfg = resolve_orca_settings(pipeline_cfg)
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    slurm_cpu_count = _get_slurm_cpu_count() if slurm_enabled else 0
    resolved_xtb_bin = resolve_executable_command(base_dir, xtb_cfg["binary"])
    resolved_orca_bin = resolve_executable_command(base_dir, orca_cfg["binary"])
    xtb_env_script = xtb_cfg["env_script"]
    xtb_parallel = int(xtb_cfg["parallel"])
    orca_nprocs = max(1, int(orca_cfg["nprocs"]))
    if slurm_cpu_count > 0:
        xtb_parallel = slurm_cpu_count
        orca_nprocs = slurm_cpu_count
    xtb_parallel_expr = f"${{XTB_PARALLEL:-{xtb_parallel}}}"
    xtb_solvent_flags = ""
    if xtb_cfg["solvent_model"] != "off":
        xtb_solvent_flags = f" --{xtb_cfg['solvent_model']} {shlex.quote(xtb_cfg['solvent'])}"
        if xtb_cfg["solvent_reference"]:
            xtb_solvent_flags += f" {shlex.quote(xtb_cfg['solvent_reference'])}"
    xtb_common_flags = (
        f"--gfn {xtb_cfg['gfn']} "
        f"--chrg {charge} "
        f"--uhf {uhf} "
        f"--acc {xtb_cfg['acc']:.3f} "
        f"--etemp {xtb_cfg['etemp']:.3f}"
        f"{xtb_solvent_flags} "
        f"--parallel {xtb_parallel_expr}"
    )

    needs_xtb = flow["relaxation"] == "xtb" or flow["md"] in {"xtb", "xtb_nobartender", "xtb_nobartender_notrim"}
    needs_orca = flow["relaxation"] == "orca"
    geometry_hint = local_xyz.name
    trajectory_hint: Optional[str] = None
    geometry_opt = flow["relaxation"] == "xtb"
    thread_hint_default = max(
        xtb_parallel if needs_xtb else 1,
        orca_nprocs if needs_orca else 1,
    )

    thread_hint_expr = (
        f"${{HYGEL_THREAD_HINT:-${{SLURM_CPUS_PER_TASK:-{thread_hint_default}}}}}"
        if slurm_enabled
        else f"${{HYGEL_THREAD_HINT:-{thread_hint_default}}}"
    )

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"export HYGEL_THREAD_HINT={thread_hint_expr}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$HYGEL_THREAD_HINT}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OMP_STACKSIZE=1G",
    ]
    lines.extend(_srun_reentry_lines(exec_cfg, "HYGEL_THREAD_HINT"))
    if needs_xtb:
        lines.append("export XTB_PARALLEL=${XTB_PARALLEL:-$HYGEL_THREAD_HINT}")
    if xtb_env_script and needs_xtb:
        lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(xtb_env_script)} ]; then",
                f"  source {shlex.quote(xtb_env_script)}",
                "fi",
                "set -u",
            ]
        )
    if needs_orca:
        lines.append(shell_assign("ORCA_BIN", resolved_orca_bin))
    if needs_xtb:
        lines.append(shell_assign("XTB_BIN", resolved_xtb_bin))

    # Restart & Trajectory handling logic
    xtb_md_flags = xtb_common_flags
    if flow["md"] in {"xtb", "xtb_nobartender", "xtb_nobartender_notrim"}:
        lines.append("# Check for restart possibility")
        lines.append("XTB_RESTART_FLAG=\"\"")
        lines.append("if [ -f xtbrestart ] && [ -f xtb.trj ]; then")
        lines.append("  echo \"[INFO] Found xtbrestart and xtb.trj. Enabling restart mode.\"")
        lines.append("  XTB_RESTART_FLAG=\"--restart\"")
        lines.append("  mv xtb.trj xtb.trj.old")
        lines.append("fi")
        xtb_md_flags += " $XTB_RESTART_FLAG"

    if flow["relaxation"] == "xtb":
        lines.append(
            f"$XTB_BIN {shlex.quote(local_xyz.name)} {xtb_common_flags} --opt {shlex.quote(xtb_cfg['opt_level'])} --cycles {xtb_cfg['opt_cycles']} > xtb_opt.out"
        )
        geometry_hint = "xtbopt.xyz"
    elif flow["relaxation"] == "orca":
        orca_template_path = resolve_optional_path(base_dir, orca_cfg["input_template_path"])
        orca_template_text = (
            orca_template_path.read_text(encoding="utf-8", errors="replace")
            if orca_template_path is not None
            else None
        )
        effective_orca_cfg = dict(orca_cfg)
        effective_orca_cfg["nprocs"] = orca_nprocs
        write_text(workdir / "relax.inp", render_orca_input(local_xyz.name, state, effective_orca_cfg, orca_template_text))
        lines.append("$ORCA_BIN relax.inp > relax.out")
        geometry_hint = "relax.xyz"
    elif flow["relaxation"] != "off":
        raise ValueError(f"Unsupported relaxation mode: {flow['relaxation']}")

    if flow["md"] in {"xtb", "xtb_nobartender", "xtb_nobartender_notrim"}:
        md_template_path = resolve_optional_path(base_dir, xtb_cfg["md_input_template_path"])
        md_template_text = (
            md_template_path.read_text(encoding="utf-8", errors="replace")
            if md_template_path is not None
            else None
        )
        write_text(workdir / "gochem.inp", render_xtb_md_input("nvt", xtb_cfg, md_template_text))
        lines.append(
            f"$XTB_BIN {shlex.quote(geometry_hint)} {xtb_md_flags} --md --input gochem.inp > xtb_md.out"
        )

        # Merge trajectories if restarted
        lines.append("if [ -f xtb.trj.old ]; then")
        lines.append("  cat xtb.trj.old xtb.trj > xtb_combined.trj")
        lines.append("  mv xtb_combined.trj xtb.trj")
        lines.append("  rm xtb.trj.old")
        lines.append("fi")

        # Convert XYZ trajectory to PDB.
        # Trim only when Bartender will immediately consume the trajectory.
        # xtb_nobartender is for inspection only — keep all frames.
        xtb_md_cfg = xtb_cfg
        skip_frames = int(xtb_md_cfg.get("md_skip_frames", 0))
        nskip = int(xtb_md_cfg.get("trim_nskip", 1))
        max_trim = float(xtb_md_cfg.get("trim_max_fraction", 1.0))
        trim_method = str(xtb_md_cfg.get("trim_method", "pymbar"))
        ref_fraction = float(xtb_md_cfg.get("trim_ref_fraction", 0.2))
        threshold_sigma = float(xtb_md_cfg.get("trim_threshold_sigma", 1.0))
        detrend_flag = " --detrend" if xtb_md_cfg.get("trim_detrend", False) else ""
        fast_flag = " --fast" if xtb_md_cfg.get("trim_fast", True) else ""
        method_flags = (
            f" --trim-method energy_threshold --ref-fraction {ref_fraction}"
            f" --threshold-sigma {threshold_sigma}"
            if trim_method == "energy_threshold"
            else ""
        )
        traj_to_pdb = shlex.quote(str(_XTB_TRAJ_TO_PDB_SRC))
        if flow["md"] == "xtb":
            lines.append("mkdir -p ../trim")
            lines.append(
                f"python3 {traj_to_pdb} xtb.trj ../trim/xtb_traj.pdb --auto-trim"
                f" --skip-frames {skip_frames} --nskip {nskip} --max-trim-fraction {max_trim}"
                f"{detrend_flag}{fast_flag}{method_flags}"
            )
            trajectory_hint = "../trim/xtb_traj.pdb"
        elif flow["md"] == "xtb_nobartender":
            lines.append("mkdir -p ../trim")
            lines.append(
                f"python3 {traj_to_pdb} xtb.trj ../trim/xtb_traj.pdb --auto-trim"
                f" --skip-frames {skip_frames} --nskip {nskip} --max-trim-fraction {max_trim}"
                f"{detrend_flag}{fast_flag}{method_flags}"
            )
            trajectory_hint = "../trim/xtb_traj.pdb"
        else:
            # xtb_nobartender_notrim: skip PDB conversion, raw trajectory only
            trajectory_hint = None

    script_path = workdir / "run_relax.sh"
    write_text(script_path, "\n".join(lines) + "\n")
    script_path.chmod(0o755)
    case["relaxation"] = {
        "mode": flow["relaxation"],
        "md": flow["md"],
        "workdir": workdir_name,
        "run_script": script_path.name,
        "geometry_hint": geometry_hint,
        "trajectory_hint": trajectory_hint,
        "raw_trajectory_hint": "xtb.trj" if flow["md"] in {"xtb", "xtb_nobartender", "xtb_nobartender_notrim"} else None,
        "geometry_opt": geometry_opt,
    }
    return workdir

def prepare_bartender_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    pipeline_cfg: Dict[str, Any],
    base_dir: Path,
    exec_cfg: Dict[str, Any],
) -> Optional[Path]:
    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        raise TypeError("bartender_pipeline.bartender must be a mapping")

    if flow["md"] in {"off", "xtb_nobartender", "xtb_nobartender_notrim"}:
        relax = case.get("relaxation", {})
        if not isinstance(relax, dict):
            relax = {}
        trajectory_path = None
        if flow["md"] in {"xtb_nobartender", "xtb_nobartender_notrim"}:
            relax_workdir = relax.get("workdir")
            relax_trajectory = relax.get("trajectory_hint") or relax.get("raw_trajectory_hint")
            if relax_workdir and relax_trajectory:
                trajectory_path = str(case_dir / str(relax_workdir) / str(relax_trajectory))
        case["bartender"] = {
            "mode": flow["md"],
            "workdir": None,
            "run_script": None,
            "geometry_source": None,
            "trajectory_source": "relaxation_output" if flow["md"] in {"xtb_nobartender", "xtb_nobartender_notrim"} else None,
            "trajectory_path": trajectory_path,
        }
        return None

    if flow["md"] == "trim":
        md_traj = resolve_optional_path(base_dir, pipeline_cfg.get("md_traj"))
        if md_traj is None:
            raise ValueError("bartender_pipeline.md=trim requires bartender_pipeline.md_traj")
        trim_dir = case_dir / "trim"
        trim_dir.mkdir(parents=True, exist_ok=True)
        xtb_settings = resolve_xtb_settings(pipeline_cfg)
        skip_f = int(xtb_settings.get("md_skip_frames", 0))
        nskip_t = int(xtb_settings.get("trim_nskip", 1))
        max_trim_t = float(xtb_settings.get("trim_max_fraction", 1.0))
        traj_to_pdb_t = shlex.quote(str(_XTB_TRAJ_TO_PDB_SRC))
        traj_rel_t = shlex.quote(os.path.relpath(str(md_traj), start=str(trim_dir)))
        traj_suffix_t = Path(str(md_traj)).suffix.lower()
        trim_lines = ["#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"']
        trim_method_t = str(xtb_settings.get("trim_method", "pymbar"))
        ref_frac_t = float(xtb_settings.get("trim_ref_fraction", 0.2))
        thresh_sigma_t = float(xtb_settings.get("trim_threshold_sigma", 1.0))
        detrend_t = " --detrend" if xtb_settings.get("trim_detrend", False) else ""
        fast_t = " --fast" if xtb_settings.get("trim_fast", True) else ""
        method_flags_t = (
            f" --trim-method energy_threshold --ref-fraction {ref_frac_t}"
            f" --threshold-sigma {thresh_sigma_t}"
            if trim_method_t == "energy_threshold"
            else ""
        )
        if traj_suffix_t not in {".pdb"}:
            trim_lines.append(
                f"python3 {traj_to_pdb_t} {traj_rel_t} xtb_traj.pdb --auto-trim"
                f" --skip-frames {skip_f} --nskip {nskip_t} --max-trim-fraction {max_trim_t}"
                f"{detrend_t}{fast_t}{method_flags_t}"
            )
        else:
            trim_lines.append(
                f"python3 {traj_to_pdb_t} {traj_rel_t} xtb_traj.pdb --pdb-info"
            )
        # Also copy results back to the source trajectory directory.
        traj_src_dir = shlex.quote(os.path.dirname(os.path.relpath(str(md_traj), start=str(trim_dir))) or ".")
        trim_lines.append(
            'for f in xtb_traj.pdb xtb_traj_trim_info.json'
            ' xtb_traj_energy_convergence.png xtb_traj_block_avg.png; do'
        )
        trim_lines.append(f'  [ -f "$f" ] && cp "$f" {traj_src_dir}/ || true')
        trim_lines.append("done")
        trim_script = trim_dir / "run_trim.sh"
        write_text(trim_script, "\n".join(trim_lines) + "\n")
        trim_script.chmod(0o755)
        case["bartender"] = {
            "mode": "trim",
            "workdir": trim_dir.name,
            "run_script": trim_script.name,
            "geometry_source": None,
            "trajectory_source": "existing_md_traj",
            "trajectory_path": str(md_traj),
        }
        return trim_dir

    if not bartender_cfg.get("enabled", True):
        return None

    polymer_geometry = case_dir / str(case["artifacts"]["polymer_xyz"])
    relax = case.get("relaxation", {})
    if not isinstance(relax, dict):
        relax = {}
    if flow["relaxation"] == "off":
        geometry = polymer_geometry
        geometry_source = "polymer_xyz"
    else:
        relax_workdir = relax.get("workdir")
        relax_geometry = relax.get("geometry_hint")
        if not relax_workdir or not relax_geometry:
            raise ValueError("Relaxation metadata is incomplete; cannot determine Bartender geometry input.")
        geometry = case_dir / str(relax_workdir) / str(relax_geometry)
        geometry_source = "relaxation_output"

    trajectory: Optional[Path] = None
    if flow["md"] == "xtb":
        relax_workdir = relax.get("workdir")
        relax_trajectory = relax.get("trajectory_hint")
        if not relax_workdir or not relax_trajectory:
            raise ValueError("xTB reuse mode requires relaxation.trajectory_hint metadata.")
        trajectory = case_dir / str(relax_workdir) / str(relax_trajectory)
        trajectory_source = "relaxation_output"
    elif flow["md"] in {"existing", "existing_notrim"}:
        md_traj = resolve_optional_path(base_dir, pipeline_cfg.get("md_traj"))
        if md_traj is None:
            raise ValueError(f"bartender_pipeline.md={flow['md']} requires bartender_pipeline.md_traj")
        trajectory = md_traj
        trajectory_source = "existing_md_traj"
    else:
        trajectory_source = None

    inp = case_dir / str(case["artifacts"]["bartender_inp"])
    if not inp.exists():
        raise FileNotFoundError(f"Bartender inp does not exist: {inp}")

    outdir = case_dir / str(bartender_cfg.get("output_dirname", "bartender_job"))
    outdir.mkdir(parents=True, exist_ok=True)
    local_inp = outdir / inp.name
    local_inp.write_text(inp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    geometry_arg = os.path.relpath(str(geometry), start=str(outdir))
    resolved_bartender_bin = resolve_executable_command(base_dir, bartender_cfg.get("binary"))
    bartender_cpus = int(bartender_cfg.get("cpus", 1))
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    slurm_cpu_count = _get_slurm_cpu_count() if slurm_enabled else 0
    if slurm_cpu_count > 0:
        bartender_cpus = slurm_cpu_count
    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        state = {}
    bartender_charge = bartender_cfg.get("charge")
    if bartender_charge is None:
        bartender_charge = int(state.get("charge", 0))

    skip = int(bartender_cfg.get("skip", 1))
    mode_args = _bartender_mode_args(flow, bartender_cfg, bartender_charge, skip, trajectory, outdir)

    # JSON manifest
    command = [resolved_bartender_bin, "-cpus", "$HYGEL_BARTENDER_CPUS"] + mode_args + [geometry_arg, local_inp.name]

    bt_root = str(bartender_cfg.get("root", "")).strip()
    bt_env_script = str(bartender_cfg.get("env_script", "")).strip()
    script_path = outdir / "run_bartender.sh"
    bartender_cpu_expr = (
        f"${{HYGEL_BARTENDER_CPUS:-${{SLURM_CPUS_PER_TASK:-{bartender_cpus}}}}}"
        if slurm_enabled
        else f"${{HYGEL_BARTENDER_CPUS:-{bartender_cpus}}}"
    )
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"export HYGEL_BARTENDER_CPUS={bartender_cpu_expr}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$HYGEL_BARTENDER_CPUS}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}",
        shell_assign("BARTENDER_BIN", resolved_bartender_bin),
    ]
    script_lines.extend(_srun_reentry_lines(exec_cfg, "HYGEL_BARTENDER_CPUS"))
    if bt_root:
        script_lines.append(f"export {shell_assign('BTROOT', bt_root)}")
    if bt_env_script:
        script_lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(bt_env_script)} ]; then",
                f"  source {shlex.quote(bt_env_script)}",
                "fi",
                "set -u",
            ]
        )
    
    # Trajectory pre-processing: trim (existing) or convert-only (existing_notrim).
    traj_to_pdb_q = shlex.quote(str(_XTB_TRAJ_TO_PDB_SRC))
    if flow["md"] == "existing" and trajectory is not None:
        traj_suffix = Path(str(trajectory)).suffix.lower()
        traj_rel = os.path.relpath(str(trajectory), start=str(outdir))
        traj_dir_rel = os.path.dirname(traj_rel) or "."
        traj_stem = Path(str(trajectory)).stem
        xtb_settings = resolve_xtb_settings(pipeline_cfg)
        skip_f = int(xtb_settings.get("md_skip_frames", 0))
        if traj_suffix not in {".pdb"}:
            nskip = int(xtb_settings.get("trim_nskip", 1))
            max_trim = float(xtb_settings.get("trim_max_fraction", 1.0))
            trim_method_e = str(xtb_settings.get("trim_method", "pymbar"))
            ref_frac_e = float(xtb_settings.get("trim_ref_fraction", 0.2))
            thresh_sigma_e = float(xtb_settings.get("trim_threshold_sigma", 1.0))
            detrend_flag = " --detrend" if xtb_settings.get("trim_detrend", False) else ""
            fast_flag = " --fast" if xtb_settings.get("trim_fast", True) else ""
            method_flags_e = (
                f" --trim-method energy_threshold --ref-fraction {ref_frac_e}"
                f" --threshold-sigma {thresh_sigma_e}"
                if trim_method_e == "energy_threshold"
                else ""
            )
            script_lines.append(
                f"python3 {traj_to_pdb_q} {shlex.quote(traj_rel)} trimmed_traj.pdb --auto-trim"
                f" --skip-frames {skip_f} --nskip {nskip} --max-trim-fraction {max_trim}"
                f"{detrend_flag}{fast_flag}{method_flags_e}"
            )
            mode_args = [a if a != traj_rel else "trimmed_traj.pdb" for a in mode_args]
        else:
            # PDB input: count frames + write trim_info.json; copy energy plots if available
            script_lines.append(
                f"python3 {traj_to_pdb_q} {shlex.quote(traj_rel)} trim_info_only.pdb --pdb-info"
            )
            for suffix in ("_energy_convergence.png", "_block_avg.png"):
                src = f"{traj_dir_rel}/{traj_stem}{suffix}"
                script_lines.append(f'[ -f {shlex.quote(src)} ] && cp {shlex.quote(src)} . || true')
    elif flow["md"] == "existing_notrim" and trajectory is not None:
        traj_suffix = Path(str(trajectory)).suffix.lower()
        traj_rel = os.path.relpath(str(trajectory), start=str(outdir))
        xtb_settings = resolve_xtb_settings(pipeline_cfg)
        skip_f = int(xtb_settings.get("md_skip_frames", 0))
        if traj_suffix not in {".pdb"}:
            # Convert .trj → .pdb without trimming, then update Bartender args
            script_lines.append(
                f"python3 {traj_to_pdb_q} {shlex.quote(traj_rel)} notrim_traj.pdb"
                f" --skip-frames {skip_f}"
            )
            mode_args = [a if a != traj_rel else "notrim_traj.pdb" for a in mode_args]
        # .pdb: use as-is, no conversion needed

    command_parts = (
        ["$BARTENDER_BIN", "-cpus", "\"$HYGEL_BARTENDER_CPUS\""]
        + [shlex.quote(a) for a in mode_args]
        + [shlex.quote(geometry_arg), shlex.quote(local_inp.name)]
    )
    script_lines.append(" ".join(command_parts))
    write_text(script_path, "\n".join(script_lines) + "\n")
    script_path.chmod(0o755)

    manifest = {
        "mode": flow["md"],
        "geometry": geometry_arg,
        "inp": local_inp.name,
        "trajectory": os.path.relpath(str(trajectory), start=str(outdir)) if trajectory is not None else None,
        "command": command,
        "outdir": str(outdir),
    }
    write_text(outdir / "bartender_job.json", json.dumps(manifest, indent=2))

    case.setdefault("bartender", {})
    case["bartender"]["job_dir"] = outdir.name
    case["bartender"]["run_script"] = script_path.name
    case["bartender"]["mode"] = flow["md"]
    case["bartender"]["geometry_source"] = geometry_source
    case["bartender"]["geometry_path"] = str(geometry)
    case["bartender"]["trajectory_source"] = trajectory_source
    case["bartender"]["trajectory_path"] = str(trajectory) if trajectory is not None else None

    return outdir

def find_case_json(start: Path) -> Optional[Path]:
    current = start.resolve()
    for _ in range(6):
        candidate = current / "case.json"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None

def build_bead_maps(
    case: Dict[str, object],
    overrides: Dict[str, Dict[str, List[str]]],
) -> tuple[Dict[int, str], Dict[int, str], set[int]]:
    from .config import normalize_label_spec
    monomers = case.get("monomers")
    tokens = case.get("sequence_tokens")
    if not isinstance(monomers, dict) or not isinstance(tokens, list):
        raise ValueError("case.json must contain 'monomers' and 'sequence_tokens'.")

    case_specs = case.get("bead_specs", {})
    if not isinstance(case_specs, dict):
        case_specs = {}

    label_map: Dict[int, str] = {}
    type_map: Dict[int, str] = {}
    backbone_beads = set(int(value) for value in case.get("backbone_beads", []))
    use_case_backbone = bool(backbone_beads)
    if not backbone_beads:
        backbone_beads = set(int(value) for value in case.get("connection_beads", []))
        use_case_backbone = bool(backbone_beads)
    offset = 0
    for token in tokens:
        if token not in monomers:
            raise KeyError(f"Token {token} is not present in case['monomers'].")
        bead_count = int(monomers[token]["bead_count"])
        spec = normalize_label_spec(token, bead_count, overrides.get(token) or case_specs.get(token))
        for local_index in range(1, bead_count + 1):
            global_index = offset + local_index
            label_map[global_index] = spec["labels"][local_index - 1]
            type_map[global_index] = spec["types"][local_index - 1]
        if not use_case_backbone:
            local_backbone = monomers[token].get("backbone_beads", [])
            if isinstance(local_backbone, list) and local_backbone:
                for bead_id in local_backbone:
                    backbone_beads.add(offset + int(bead_id))
            else:
                backbone_beads.add(offset + int(monomers[token].get("left_connection_bead", 1)))
        offset += bead_count
    return label_map, type_map, backbone_beads

def collect_results(root: Path, output: Path) -> Dict[str, Any]:
    records = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        summary = summarize_itp(itp_path)
        case_json = find_case_json(itp_path.parent)
        if case_json:
            case = json.loads(case_json.read_text(encoding="utf-8"))
            summary["case"] = {
                "sequence_stem": case.get("sequence_stem"),
                "sequence_tokens": case.get("sequence_tokens"),
                "case_json": str(case_json),
            }
        records.append(summary)
    payload = {"root": str(root), "count": len(records), "records": records}
    write_text(output, json.dumps(payload, indent=2))
    return payload

def merge_results(root: Path, output_itp: Path, output_json: Path, label_map_path: Optional[Path]) -> Dict[str, Any]:
    from .config import load_label_map
    label_overrides = load_label_map(label_map_path) if label_map_path else {}
    records: List[Any] = []
    skipped: List[Dict[str, str]] = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        case_path = find_case_json(itp_path.parent)
        if case_path is None:
            skipped.append({"path": str(itp_path), "reason": "case.json not found in parent chain"})
            continue
        try:
            records.extend(typed_records_for_result(itp_path, case_path, label_overrides))
        except Exception as exc:
            skipped.append({"path": str(itp_path), "reason": str(exc)})

    merged = merge_records(records)
    write_merged_forcefield(output_itp, merged, root=root, label_map_path=label_map_path)
    payload = merged_summary_payload(root, merged, skipped)
    write_text(output_json, json.dumps(payload, indent=2))
    return payload

def run_postprocess_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    pipeline_cfg = cfg["bartender_pipeline"]
    post_cfg = pipeline_cfg["postprocess"]
    if not post_cfg.get("screening", {}).get("enabled", False):
        raise ValueError("Postprocess-only mode now runs screening only. Set bartender_pipeline.postprocess.screening.enabled=true.")

    summary: Dict[str, Any] = {
        "settings": cfg,
        "postprocess_only": True,
        "screening": run_screening_postprocess(cfg),
    }

    summary_root_value = cfg["paths"].get("postprocess_output_root") or cfg["paths"].get("out_root") or base_dir
    summary_root = Path(str(summary_root_value)).resolve()
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "postprocess_summary.json"
    summary["summary_json"] = str(summary_path)
    write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = cfg["bartender_pipeline"]
    flow = resolve_pipeline_modes(pipeline_cfg)
    exec_cfg = resolve_execution_settings(pipeline_cfg)
    log_cfg = resolve_log_settings(pipeline_cfg)
    connection_cfg = resolve_connection_detection_config(pipeline_cfg)
    term_cfg = resolve_term_generation_config(pipeline_cfg)
    
    legacy_init_templates = pipeline_cfg.get("init_templates", {})
    if legacy_init_templates is None:
        legacy_init_templates = {}

    monomer_cfg = normalize_monomer_configs(cfg["monomers"], legacy_init_templates)
    monomer_paths = {
        token: str(resolve_under_base(base_dir, entry["xyz"]))
        for token, entry in monomer_cfg.items()
    }
    library = load_monomer_library(monomer_paths, base_dir=base_dir)
    monomer_keys = set(library.keys())

    init_templates = {
        token: resolve_under_base(base_dir, entry["init_template"])
        for token, entry in monomer_cfg.items()
        if entry.get("init_template")
    }
    sequence_jobs = build_sequence_jobs(cfg["system"], monomer_keys)

    template_cache: Dict[str, Any] = {}
    metadata_cache: Dict[str, Any] = {}
    validation_cache: Dict[str, Any] = {}

    trim_only = flow["md"] == "trim"

    cases: List[Dict[str, Any]] = []
    for tokens in sequence_jobs:
        if not trim_only:
            for token in tokens:
                if token not in monomer_paths:
                    raise KeyError(f"Unknown monomer token: {token}")
                if token not in init_templates:
                    raise KeyError(f"Missing init template for monomer token: {token}")
                if token not in template_cache:
                    template = parse_bartender_inp(init_templates[token])
                    template_cache[token] = template
                    validation_cache[token] = validate_template(template, Path(monomer_paths[token]), connection_cfg)
                    metadata_cache[token] = infer_connection_metadata(
                        template,
                        Path(monomer_paths[token]),
                        connection_cfg,
                        monomer_cfg[token]["backbone_atoms"],
                    )

        stem = sequence_stem(tokens)
        case_dir = out_root / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        torsion_mode = str(cfg["system"].get("n_torsion_mode", "repeat"))
        torsion = len(tokens) if torsion_mode == "repeat" else max(1, len(tokens) - 1)

        if not trim_only:
            builder_tmp = case_dir / "_builder_tmp"
            shutil.rmtree(builder_tmp, ignore_errors=True)
            builder_tmp.mkdir(parents=True, exist_ok=True)

            build_polymer_structure(
                tokens,
                monomer_dict=library,
                n_torsion=torsion,
                output_filename=f"{stem}.xyz",
                output_dir=builder_tmp,
            )

            built_xyz = builder_tmp / f"{stem}.xyz"
            final_xyz = case_dir / f"{stem}.xyz"
            shutil.copyfile(built_xyz, final_xyz)
            shutil.rmtree(builder_tmp, ignore_errors=True)

            bundle = build_polymer_input(tokens, final_xyz, template_cache, metadata_cache, term_cfg)
            base_inp = case_dir / f"{stem}_base.inp"
            final_inp = case_dir / f"{stem}_bartender.inp"
            write_text(base_inp, bundle.base_text)
            write_text(final_inp, bundle.augmented_text)

            restart_xyz = resolve_optional_path(base_dir, pipeline_cfg.get("initial_geometry_xyz"))
            relax_input_xyz = final_xyz
            if restart_xyz is not None:
                built_symbols, _ = parse_xyz(final_xyz)
                restart_symbols, _ = parse_xyz(restart_xyz)
                if built_symbols != restart_symbols:
                    raise ValueError(
                        f"initial_geometry_xyz atom ordering does not match the built polymer for sequence {stem}."
                    )
                restart_copy = case_dir / f"{stem}_restart_input.xyz"
                shutil.copyfile(restart_xyz, restart_copy)
                relax_input_xyz = restart_copy

            artifacts: Dict[str, Any] = {
                "polymer_xyz": final_xyz.name,
                "base_inp": base_inp.name,
                "bartender_inp": final_inp.name,
            }
            if relax_input_xyz != final_xyz:
                artifacts["relax_input_xyz"] = relax_input_xyz.name
            bead_specs = {
                token: {"labels": [f"{token}{i+1}" for i in range(template_cache[token].bead_count)]}
                for token in sorted(set(tokens))
            }
            monomers_meta = {
                token: {
                    "xyz": monomer_paths[token],
                    "init_inp": str(init_templates[token]),
                    "bead_count": template_cache[token].bead_count,
                }
                for token in sorted(set(tokens))
            }
            connection_bonds = bundle.connection_bonds
            connection_beads = bundle.connection_beads
            backbone_beads = bundle.backbone_beads
        else:
            artifacts = {}
            bead_specs = {}
            monomers_meta = {}
            connection_bonds = []
            connection_beads = []
            backbone_beads = []

        logs_dir = ensure_case_logs_dir(case_dir, log_cfg)
        electronic_state = resolve_case_electronic_state(tokens, monomer_cfg, pipeline_cfg)

        case: Dict[str, Any] = {
            "sequence_tokens": tokens,
            "sequence_stem": stem,
            "torsion": torsion,
            "workflow_mode": flow,
            "artifacts": artifacts,
            "electronic_state": electronic_state,
            "term_generation": {
                "mode": term_cfg.mode,
                "n": term_cfg.n,
            },
            "connection_detection": {
                "indicator": connection_cfg.indicator,
                "cutoff": connection_cfg.cutoff,
            },
            "bead_specs": bead_specs,
            "monomers": monomers_meta,
            "connection_bonds": connection_bonds,
            "connection_beads": connection_beads,
            "backbone_beads": backbone_beads,
            "execution": {
                "relaxation": None,
                "bartender": None,
            },
        }

        if exec_cfg.get("run_relaxation", False) and not trim_only:
            relax_dir = prepare_relaxation_job(case_dir, case, flow, pipeline_cfg, base_dir, exec_cfg)
            if relax_dir:
                case["execution"]["relaxation"] = execute_case_script("relaxation", relax_dir / case["relaxation"]["run_script"], relax_dir, exec_cfg, logs_dir)

        if exec_cfg.get("run_bartender", False):
            bt_dir = prepare_bartender_job(case_dir, case, flow, pipeline_cfg, base_dir, exec_cfg)
            if bt_dir:
                case["execution"]["bartender"] = execute_case_script("bartender", bt_dir / case["bartender"]["run_script"], bt_dir, exec_cfg, logs_dir)

        write_text(case_dir / "case.json", json.dumps(case, indent=2))
        cases.append(case)

    summary = {
        "cases": cases,
        "config": cfg,
    }
    
    if pipeline_cfg.get("postprocess", {}).get("screening", {}).get("enabled", False):
         summary["screening"] = run_screening_postprocess(cfg)

    write_text(out_root / "summary.json", json.dumps(summary, indent=2))
    return summary

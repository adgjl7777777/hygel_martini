"""Guarded, fixed-increment box compression for hydrated network structures.

Owns the ``workflow.mode=hard_em_shrink`` stage of the post-build
relaxation: it compresses the box toward ``target_box_nm`` in small
multiplicative steps (default 1% per accepted step), and accepts each
step only after the structure passes guards — finite coordinates,
finite potential energy, and an EM Fmax below ``fmax_max`` (kJ/mol/nm).
A failing step optionally runs a short NVT recovery MD and retries; if
still failing, the shrink increment is halved down to
``min_shrink_fraction`` before giving up.  Only compression is allowed;
a target larger than the current box is rejected.

Called by ``generator.run_relax_workflow``; reuses the GROMACS helpers
from ``soft_em``.  State lives in ``paths.workdir``: ``last_valid.gro``
(latest accepted structure), ``state.json`` (restart point),
``history.jsonl`` (one record per attempt), and ``final.gro`` on
success.  Box lengths are in nm throughout.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .soft_em import (
    _build_mdrun_cmd,
    _ensure_dir,
    _extract_xvg,
    _grompp_and_run_em,
    _parse_em_fmax,
    _parse_gro_box,
    _read_xvg_rows,
    _run,
    _wrap_pbc_gro,
)


def _finite_gro(path: Path) -> bool:
    """Check a .gro file is structurally sane with finite numbers.

    Verifies the atom count matches, all fixed-column coordinates are
    finite, and the box line holds three finite positive lengths.  Any
    parse problem counts as not finite (returns False, never raises).
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        atom_count = int(lines[1].strip())
        if len(lines) < atom_count + 3:
            return False
        for line in lines[2 : atom_count + 2]:
            values = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
            if not all(math.isfinite(value) for value in values):
                return False
        box = [float(value) for value in lines[atom_count + 2].split()[:3]]
        return len(box) == 3 and all(math.isfinite(value) and value > 0.0 for value in box)
    except (IndexError, ValueError, OSError):
        return False


def _energy_is_finite(gmx: str, edr: Path, out_xvg: Path, env: Dict[str, str]) -> Tuple[bool, float | None]:
    """Extract the last Potential (kJ/mol) from an .edr and check finiteness.

    Returns:
        (is_finite, potential); (False, None) on any extraction failure.
    """
    try:
        _extract_xvg(gmx, edr, out_xvg, ["Potential"], env)
        rows = _read_xvg_rows(out_xvg)
        potential = rows[-1][1]
        return math.isfinite(potential), potential
    except Exception:
        return False, None


def _mdp_values(path: "Path", keys=("dt", "nsteps", "ref_t", "ref-t", "constraints")) -> Dict[str, Any]:
    """The handful of mdp settings that describe a recovery run.

    Recorded into every history event that used the recovery (integrated
    report §18C.5: timestep, temperature and recovery count must be on the
    record), so a stalled or healed shrink can be read back without opening
    the mdp that may since have changed. Missing keys are simply absent.
    """
    wanted = {k.replace("-", "_") for k in keys}
    out: Dict[str, Any] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            text = line.split(";", 1)[0].strip()
            if "=" not in text:
                continue
            key, _, value = text.partition("=")
            key = key.strip().lower().replace("-", "_")
            if key in wanted:
                value = value.strip()
                try:
                    out[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                except ValueError:
                    out[key] = value
    except OSError:
        pass
    return out


def _run_nvt_recovery(
    *,
    gmx: str,
    mdp: Path,
    gro: Path,
    top: Path,
    outdir: Path,
    ntomp: int,
    maxwarn: int,
    env: Dict[str, str],
    gpu_id: Optional[str],
    mpi_np: Optional[int],
    mpi_args: List[str],
    mdrun_args: List[str],
) -> Path:
    """Run a short NVT recovery MD to relax a structure that failed EM guards.

    Returns:
        The resulting nvt.gro path.

    Raises:
        RuntimeError: If the run produces no output or a non-finite
            structure.
    """
    _ensure_dir(outdir)
    tpr = outdir / "nvt.tpr"
    deffnm = outdir / "nvt"
    _run(
        [gmx, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", str(maxwarn)],
        cwd=outdir,
        env=env,
    )
    command, extra_env = _build_mdrun_cmd(
        gmx,
        ["-deffnm", "nvt", "-ntomp", str(ntomp)],
        gpu_id,
        mpi_np,
        mpi_args,
        mdrun_args,
    )
    run_env = dict(env)
    run_env.update(extra_env)
    _run(command, cwd=outdir, env=run_env)
    output = deffnm.with_suffix(".gro")
    if not output.exists() or not _finite_gro(output):
        raise RuntimeError(f"NVT recovery did not produce a finite structure: {output}")
    return output


def _target_box(config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Normalize target_box_nm (scalar or 3-list, in nm) to an (x, y, z) tuple.

    Raises:
        ValueError: For any other shape.
    """
    target = config.get("target_box_nm")
    if isinstance(target, (int, float)):
        return float(target), float(target), float(target)
    if isinstance(target, list) and len(target) == 3:
        return tuple(float(value) for value in target)
    raise ValueError("hard_em_shrink.target_box_nm must be a scalar or a three-value list")


def run_hard_em_shrink(cfg: Dict[str, Any]) -> Path:
    """Compress the box stepwise to the target, guarding every step with EM.

    Loop per accepted step: scale coordinates+box down by the attempt
    fraction (never below the per-axis target), wrap atoms back into the
    box, run EM, and accept only if structure/energy/Fmax guards pass —
    optionally after one NVT recovery.  A rejected attempt halves the
    fraction (down to ``min_shrink_fraction``); every attempt is
    appended to history.jsonl and the accepted state saved to
    state.json, enabling ``restart: true`` resumption.  Without restart
    an existing workdir is moved aside to ``<name>_bak``.

    Args:
        cfg: Normalized relax config; uses paths/tools/runtime plus the
            ``hard_em_shrink`` section (target_box_nm [nm],
            shrink_fraction, min_shrink_fraction, fmax_max [kJ/mol/nm],
            max_steps, nvt_recovery_enabled, minim_mdp,
            nvt_recovery_mdp, maxwarn, restart).

    Returns:
        ``<workdir>/final.gro`` once every axis is at (or within 1e-6
        relative of) the target.

    Raises:
        ValueError: For invalid fractions or an expanding target box.
        FileNotFoundError: For missing input files.
        RuntimeError: For a non-finite start structure, guards that
            cannot be passed even at the minimum fraction, or reaching
            max_steps before the target.
    """
    tools = cfg.get("tools", {})
    runtime = cfg.get("runtime", {})
    paths = cfg.get("paths", {})
    shrink = cfg.get("hard_em_shrink", {})

    gmx = str(tools.get("gmx", "gmx_mpi"))
    top = Path(str(paths["system_top"])).resolve()
    start_gro = Path(str(paths["start_gro"])).resolve()
    minim_mdp = Path(str(shrink["minim_mdp"])).resolve()
    nvt_mdp = Path(str(shrink["nvt_recovery_mdp"])).resolve()
    recovery_settings = _mdp_values(nvt_mdp)
    workdir = Path(str(paths["workdir"])).resolve()
    target = _target_box(shrink)
    ntomp = int(runtime.get("omp_threads", 1))
    maxwarn = int(shrink.get("maxwarn", 2))
    shrink_fraction = float(shrink.get("shrink_fraction", 0.01))
    min_shrink_fraction = float(shrink.get("min_shrink_fraction", 0.00125))
    fmax_max = float(shrink.get("fmax_max", 1000.0))
    max_steps = int(shrink.get("max_steps", 1000))
    recovery_enabled = bool(shrink.get("nvt_recovery_enabled", True))
    do_restart = bool(shrink.get("restart", False))
    gpu_id_raw = runtime.get("gpu_id")
    gpu_id: Optional[str] = str(gpu_id_raw) if gpu_id_raw is not None else None
    mpi_np_raw = runtime.get("mpi_np")
    mpi_np: Optional[int] = int(mpi_np_raw) if mpi_np_raw is not None else None
    mpi_args: List[str] = [str(value) for value in runtime.get("mpi_args", [])]
    mdrun_args: List[str] = [str(value) for value in runtime.get("mdrun_args", [])]

    if not 0.0 < shrink_fraction < 1.0:
        raise ValueError("hard_em_shrink.shrink_fraction must lie in (0, 1)")
    if not 0.0 < min_shrink_fraction <= shrink_fraction:
        raise ValueError("hard_em_shrink.min_shrink_fraction must lie in (0, shrink_fraction]")
    for path in (top, start_gro, minim_mdp, nvt_mdp):
        if not path.exists():
            raise FileNotFoundError(path)
    if not _finite_gro(start_gro):
        raise RuntimeError(f"Initial structure is not finite: {start_gro}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ntomp)
    env["GMX_OPENMP_MAX_THREADS"] = str(ntomp)
    current = workdir / "last_valid.gro"
    state_path = workdir / "state.json"
    history_path = workdir / "history.jsonl"

    if do_restart and current.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step_index = int(state["step_index"])
        print(f"=== hard_em_shrink restart at accepted step {step_index} ===")
    else:
        if workdir.exists():
            backup = workdir.with_name(workdir.name + "_bak")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.move(str(workdir), str(backup))
        _ensure_dir(workdir)
        shutil.copy2(start_gro, current)
        state = {"step_index": 0, "target_box_nm": target}
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        step_index = 0
        print("=== hydrogel_builder.relax hard_em_shrink ===")

    start_box = _parse_gro_box(current)
    if any(target_axis > current_axis for target_axis, current_axis in zip(target, start_box)):
        raise ValueError(f"Target box {target} expands current box {start_box}; hard_em_shrink only compresses")

    while step_index < max_steps:
        box = _parse_gro_box(current)
        if all(current_axis <= target_axis * (1.0 + 1e-6) for current_axis, target_axis in zip(box, target)):
            final = workdir / "final.gro"
            shutil.copy2(current, final)
            print(f"Reached target box {target}; wrote {final}")
            return final

        attempted_fraction = shrink_fraction
        accepted = False
        while attempted_fraction >= min_shrink_fraction and not accepted:
            next_box = tuple(max(target_axis, current_axis * (1.0 - attempted_fraction)) for current_axis, target_axis in zip(box, target))
            scales = tuple(next_axis / current_axis for next_axis, current_axis in zip(next_box, box))
            attempt_dir = workdir / f"step_{step_index + 1:04d}_scale_{attempted_fraction:.6f}"
            _ensure_dir(attempt_dir)
            scaled = attempt_dir / "scaled.gro"
            _run(
                [gmx, "editconf", "-f", str(current), "-o", str(scaled), "-scale", *(f"{value:.10f}" for value in scales)],
                env=env,
            )
            wrapped = attempt_dir / "wrapped.gro"
            _wrap_pbc_gro(scaled, wrapped)
            candidate = wrapped
            recovery_used = False

            def em_check(input_gro: Path, label: str) -> Tuple[bool, Path | None, float | None, float | None, str | None]:
                """EM the candidate and evaluate the acceptance guards.

                Returns:
                    (valid, em_gro, fmax, potential, error); exceptions
                    are captured into ``error`` instead of propagating.
                """
                try:
                    em_dir = attempt_dir / label
                    _, edr, log, em_gro = _grompp_and_run_em(
                        gmx_cmd=gmx,
                        mdp=minim_mdp,
                        gro=input_gro,
                        top=top,
                        outdir=em_dir,
                        ntomp=ntomp,
                        maxwarn=maxwarn,
                        env=env,
                        gpu_id=gpu_id,
                        mpi_np=mpi_np,
                        mpi_args=mpi_args,
                        mdrun_args=mdrun_args,
                    )
                    finite_energy, potential = _energy_is_finite(gmx, edr, em_dir / "potential.xvg", env)
                    fmax = _parse_em_fmax(log)
                    valid = _finite_gro(em_gro) and finite_energy and math.isfinite(fmax) and fmax <= fmax_max
                    return valid, em_gro, fmax, potential, None
                except Exception as exc:
                    return False, None, None, None, str(exc)

            valid, em_gro, fmax, potential, error = em_check(candidate, "em_initial")
            if not valid and recovery_enabled:
                recovery_used = True
                # Recovery must retain this attempt's compressed box. Falling
                # back to ``current`` would let an uncompressed structure pass
                # the guard while incorrectly advancing the shrink counter.
                recovery_input = em_gro if em_gro is not None and _finite_gro(em_gro) else candidate
                try:
                    recovered = _run_nvt_recovery(
                        gmx=gmx,
                        mdp=nvt_mdp,
                        gro=recovery_input,
                        top=top,
                        outdir=attempt_dir / "nvt_recovery",
                        ntomp=ntomp,
                        maxwarn=maxwarn,
                        env=env,
                        gpu_id=gpu_id,
                        mpi_np=mpi_np,
                        mpi_args=mpi_args,
                        mdrun_args=mdrun_args,
                    )
                    valid, em_gro, fmax, potential, error = em_check(recovered, "em_after_nvt")
                except Exception as exc:
                    error = f"NVT recovery failed: {exc}"

            event = {
                "step": step_index + 1,
                "attempted_shrink_fraction": attempted_fraction,
                "box_before_nm": box,
                "box_after_nm": next_box,
                "fmax": fmax,
                "potential": potential,
                "nvt_recovery_used": recovery_used,
                # dt (ps), nsteps, ref_t (K), constraints of the recovery that
                # ran, or None when this attempt did not need one.
                "nvt_recovery_settings": recovery_settings if recovery_used else None,
                "accepted": valid,
                "error": error,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(
                f"[step {step_index + 1:04d}] shrink={attempted_fraction:.5f} "
                f"box={tuple(round(value, 3) for value in next_box)} "
                f"Fmax={fmax} recovery={recovery_used} accepted={valid}"
            )

            if valid and em_gro is not None:
                shutil.copy2(em_gro, current)
                step_index += 1
                state = {"step_index": step_index, "target_box_nm": target, "last_event": event}
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                accepted = True
            else:
                attempted_fraction *= 0.5

        if not accepted:
            raise RuntimeError(
                f"Compression could not pass EM/NVT guards at accepted step {step_index + 1}. "
                f"Last valid structure: {current}"
            )

    raise RuntimeError(f"hard_em_shrink reached max_steps={max_steps}; last valid structure: {current}")

"""Soft-EM stage of the post-build relaxation (workflow.mode=soft_em).

Owns the iterative energy-minimization loop that gently relaxes a
freshly built hydrogel: per iteration it (1) writes a scaled copy of
the bonded .itp whose force constants ramp from ``bonded_start`` to
``bonded_end`` over ``bonded_ramp_iters`` iterations, (2) runs GROMACS
EM, (3) reads potential energy (kJ/mol), Fmax (kJ/mol/nm) and the
pressure tensor (bar) from the EM outputs, and (4) rescales the box —
coordinates included — toward ``p_target`` (bar) with per-iteration
length changes capped at ``max_dlen`` (fractional).  Convergence needs
energy-change, Fmax and pressure tolerances simultaneously satisfied
at full bonded strength; the result is ``<workdir>/final.gro``.

Called by ``generator.run_relax_workflow``; box lengths are in nm.
Also hosts the GROMACS helpers (_run, _grompp_and_run_em,
_build_mdrun_cmd, _wrap_pbc_gro, ...) reused by ``hard_em_shrink``.
Supports ``restart: true`` (resume after the last iteration that
completed its box scaling); otherwise an existing workdir is moved
aside to ``<name>_bak``.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


# Matches .itp/.top section headers like "[ bonds ]".
SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$")


def _run(
    cmd: List[str],
    *,
    cwd: Path | None = None,
    env: Dict[str, str] | None = None,
    input_str: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command with merged stdout/stderr captured as text.

    Args:
        cmd: Argument vector.
        cwd: Working directory (None keeps the current one).
        env: Environment for the child (None inherits).
        input_str: Text fed to stdin (e.g. gmx energy term selection).
        check: Raise on nonzero exit instead of returning the process.

    Raises:
        RuntimeError: When ``check`` and the command fails; the captured
            output is embedded so GROMACS diagnostics survive.
    """
    process = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        input=input_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if check and process.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={process.returncode}): {' '.join(map(shlex.quote, cmd))}\n"
            f"--- output ---\n{process.stdout}"
        )
    return process


def _ensure_dir(path: Path) -> None:
    """Create the directory (and parents) if needed."""
    path.mkdir(parents=True, exist_ok=True)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def _parse_gro_box(gro_path: Path) -> Tuple[float, float, float]:
    """Read the box diagonal (nm) from the last line of a .gro file.

    Raises:
        ValueError: When fewer than three values are present.
    """
    last = gro_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1].split()
    if len(last) < 3:
        raise ValueError(f"Cannot parse box from {gro_path}")
    return tuple(map(float, last[:3]))


def _parse_em_fmax(log_path: Path) -> float:
    """Parse the final maximum force (kJ/mol/nm) from an EM log.

    Tries the "Maximum force = ..." summary line first, then "Fmax =".

    Raises:
        RuntimeError: If neither pattern is found.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum\s+force\s*=\s*([0-9.Ee+\-]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"Fmax\s*=\s*([0-9.Ee+\-]+)", text)
    if match:
        return float(match.group(1))
    raise RuntimeError(f"Could not parse Fmax from {log_path}")


def _find_energy_indices(gmx_cmd: str, edr_file: Path, wanted_names: List[str], env: Dict[str, str]) -> List[int]:
    """Map energy-term names to their ``gmx energy`` menu indices.

    Probes the interactive menu by sending "0" and scanning the printed
    listing, since index numbering varies between .edr files.

    Returns:
        Indices in the same order as ``wanted_names``.

    Raises:
        RuntimeError: If any wanted term is absent (menu excerpt included).
    """
    probe = _run(
        [gmx_cmd, "energy", "-f", str(edr_file), "-o", os.devnull, "-xvg", "none"],
        input_str="0\n",
        env=env,
        check=False,
    )
    idx_map: Dict[str, int] = {}
    for line in probe.stdout.splitlines():
        for match in re.finditer(r"(\d+)\s+([A-Za-z0-9#\-\*\.\(\)]+)", line):
            idx = int(match.group(1))
            name = match.group(2).strip()
            if name != "0":
                idx_map[name] = idx

    missing = [name for name in wanted_names if name not in idx_map]
    if missing:
        excerpt = "\n".join(
            line
            for line in probe.stdout.splitlines()
            if re.search(r"\b\d+\s+\S+", line)
        )
        raise RuntimeError(
            f"Could not find energy terms in {edr_file}: {missing}\n"
            f"--- energy menu excerpt ---\n{excerpt}"
        )
    return [idx_map[name] for name in wanted_names]


def _extract_xvg(
    gmx_cmd: str,
    edr_file: Path,
    out_xvg: Path,
    terms: List[str],
    env: Dict[str, str],
) -> None:
    """Extract the given energy terms from an .edr into a plain .xvg.

    Term columns follow the order of ``terms``; ``-xvg none`` keeps the
    output free of Grace headers for simple numeric parsing.
    """
    indices = _find_energy_indices(gmx_cmd, edr_file, terms, env)
    selection = "\n".join(str(idx) for idx in indices) + "\n0\n"
    _run(
        [gmx_cmd, "energy", "-f", str(edr_file), "-o", str(out_xvg), "-xvg", "none"],
        input_str=selection,
        env=env,
        check=True,
    )


def _read_xvg_rows(path: Path) -> List[List[float]]:
    """Parse numeric rows from an .xvg (skipping #/@ lines and bad rows).

    Raises:
        RuntimeError: When no numeric data is found at all.
    """
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("@"):
            continue
        try:
            rows.append([float(value) for value in stripped.split()])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric data found in {path}")
    return rows


def _summarize_series(path: Path, column: int, mode: str) -> float:
    """Reduce one .xvg column to a scalar: mode "last" or mean (default)."""
    rows = _read_xvg_rows(path)
    if mode == "last":
        return rows[-1][column]
    return mean(row[column] for row in rows)


def _split_comment(line: str) -> Tuple[str, str]:
    """Split an .itp line into (code, ";comment") preserving the comment."""
    if ";" in line:
        code, comment = line.split(";", 1)
        return code.rstrip(), ";" + comment
    return line.rstrip(), ""


def _is_int_token(token: str) -> bool:
    """True when the token parses as int."""
    try:
        int(token)
        return True
    except Exception:
        return False


def _is_float_token(token: str) -> bool:
    """True when the token parses as float."""
    try:
        float(token)
        return True
    except Exception:
        return False


def scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float) -> None:
    """Write a copy of an .itp with bonded force constants multiplied.

    Only the [bonds], [angles] and [dihedrals] sections are touched;
    line comments are kept.  For bonds/angles the last float token on
    the line (the force constant) is scaled.  For dihedrals: funct 3
    (Ryckaert-Bellemans) scales every C0..C5 coefficient, otherwise the
    7th token (the k column of periodic/improper forms) is scaled with
    a last-float fallback.  Lines that cannot be interpreted are copied
    verbatim, so unknown content degrades to "unscaled", not corrupted.

    Args:
        in_itp: Source topology include.
        out_itp: Destination (overwritten).
        factor: Multiplier for the force constants (kJ/mol-based units).
    """
    current_section: str | None = None
    output: List[str] = []

    for raw in in_itp.read_text(encoding="utf-8", errors="replace").splitlines(True):
        line = raw.rstrip("\n")
        match = SECTION_RE.match(line.strip())
        if match:
            current_section = match.group(1).lower()
            output.append(line + "\n")
            continue

        if current_section not in {"bonds", "angles", "dihedrals"}:
            output.append(line + "\n")
            continue

        code, comment = _split_comment(line)
        stripped = code.strip()
        if not stripped or stripped.startswith(";"):
            output.append(line + "\n")
            continue

        tokens = stripped.split()
        prefix_ws = re.match(r"^\s*", code).group(0)

        try:
            if current_section in {"bonds", "angles"}:
                idx = None
                for i in range(len(tokens) - 1, -1, -1):
                    if _is_float_token(tokens[i]):
                        idx = i
                        break
                if idx is None:
                    output.append(line + "\n")
                    continue
                tokens[idx] = f"{float(tokens[idx]) * factor:.8g}"
                output.append(prefix_ws + " ".join(tokens) + ((" " + comment) if comment else "") + "\n")
                continue

            if len(tokens) < 5 or not _is_int_token(tokens[4]):
                output.append(line + "\n")
                continue
            funct = int(tokens[4])
            if funct == 3:
                for idx in range(5, len(tokens)):
                    if _is_float_token(tokens[idx]):
                        tokens[idx] = f"{float(tokens[idx]) * factor:.8g}"
            else:
                if len(tokens) >= 8 and _is_float_token(tokens[6]):
                    tokens[6] = f"{float(tokens[6]) * factor:.8g}"
                else:
                    idx = None
                    for i in range(len(tokens) - 1, -1, -1):
                        if _is_float_token(tokens[i]):
                            idx = i
                            break
                    if idx is not None:
                        tokens[idx] = f"{float(tokens[idx]) * factor:.8g}"
            output.append(prefix_ws + " ".join(tokens) + ((" " + comment) if comment else "") + "\n")
        except Exception:
            output.append(line + "\n")

    out_itp.write_text("".join(output), encoding="utf-8")


def patch_system_top(in_top: Path, out_top: Path, bonded_itp_basename: str, new_local_itp_name: str) -> None:
    """Copy a .top, redirecting the bonded include to a local scaled .itp.

    Any ``#include`` whose basename matches ``bonded_itp_basename`` is
    rewritten to ``new_local_itp_name`` (relative, next to the new .top);
    all other lines pass through unchanged.
    """
    pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')
    output: List[str] = []

    for raw in in_top.read_text(encoding="utf-8", errors="replace").splitlines(True):
        line = raw.rstrip("\n")
        match = pattern.match(line.strip())
        if match:
            include_path = match.group(1)
            if Path(include_path).name == bonded_itp_basename:
                output.append(f'#include "{new_local_itp_name}"\n')
                continue
        output.append(raw if raw.endswith("\n") else raw + "\n")

    out_top.write_text("".join(output), encoding="utf-8")


def _build_mdrun_cmd(
    gmx: str,
    base_args: List[str],
    gpu_id: Optional[str],
    mpi_np: Optional[int],
    mpi_args: List[str],
    extra: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Build the full mdrun command and required env overrides.

    No MPI (mpi_np=None):
      gpu_id=None    → single process, CPU-only flags (-nb cpu -update cpu)
      gpu_id="N"     → single process, GPU via CUDA_VISIBLE_DEVICES=N
    With MPI (mpi_np=N):
      gpu_id=None    → mpirun -np N [mpi_args] gmx mdrun ...
      gpu_id="0123"  → mpirun -np N [mpi_args] gmx mdrun ... -gpu_id 0123
      CUDA_VISIBLE_DEVICES is NOT set with MPI; ranks select GPU via -gpu_id.
    """
    env_extra: Dict[str, str] = {}
    mdrun = [gmx, "mdrun"] + base_args
    if mpi_np is None:
        if gpu_id is None:
            mdrun += ["-nb", "cpu", "-update", "cpu"]
        else:
            env_extra["CUDA_VISIBLE_DEVICES"] = gpu_id
        return mdrun + extra, env_extra
    else:
        if gpu_id is not None:
            mdrun += ["-gpu_id", gpu_id]
        return ["mpirun", "-np", str(mpi_np)] + mpi_args + mdrun + extra, env_extra


def _compute_box_deltas(
    pxx: float,
    pyy: float,
    pzz: float,
    box_x: float,
    box_y: float,
    box_z: float,
    p_target: float,
    scale_factor: float,
    max_dlen: float,
    box_mode: str,
    cubic_rate: float,
    cubic_max_dlen: float,
) -> Tuple[float, float, float]:
    """Compute (d_lx, d_ly, d_lz) — fractional length change for each axis.

    anisotropic (default):
      Each axis scaled independently by its own pressure component.
      Can produce elongated boxes when pressure is uneven across axes.

    isotropic:
      Mean pressure drives equal scaling of all three axes.
      The current a:b:c aspect ratio is preserved exactly.

    cubic:
      Mean pressure drives overall volume change (same as isotropic).
      Additionally applies a per-iter shape correction that nudges each axis
      toward the cube side L_cube = (Lx*Ly*Lz)^(1/3), which is the cube whose
      volume equals the current box.  The correction is volume-neutral (the
      three d_shape terms sum to ~0) so it does not fight the pressure signal.
    """
    if box_mode == "isotropic":
        p_mean = (pxx + pyy + pzz) / 3.0
        d = _clamp(scale_factor * (p_mean - p_target), -max_dlen, max_dlen)
        return d, d, d

    if box_mode == "cubic":
        p_mean = (pxx + pyy + pzz) / 3.0
        d_pressure = _clamp(scale_factor * (p_mean - p_target), -max_dlen, max_dlen)
        L_cube = (box_x * box_y * box_z) ** (1.0 / 3.0)
        d_sx = _clamp(cubic_rate * (L_cube - box_x) / box_x, -cubic_max_dlen, cubic_max_dlen)
        d_sy = _clamp(cubic_rate * (L_cube - box_y) / box_y, -cubic_max_dlen, cubic_max_dlen)
        d_sz = _clamp(cubic_rate * (L_cube - box_z) / box_z, -cubic_max_dlen, cubic_max_dlen)
        return (
            _clamp(d_pressure + d_sx, -max_dlen, max_dlen),
            _clamp(d_pressure + d_sy, -max_dlen, max_dlen),
            _clamp(d_pressure + d_sz, -max_dlen, max_dlen),
        )

    # Default: anisotropic
    return (
        _clamp(scale_factor * (pxx - p_target), -max_dlen, max_dlen),
        _clamp(scale_factor * (pyy - p_target), -max_dlen, max_dlen),
        _clamp(scale_factor * (pzz - p_target), -max_dlen, max_dlen),
    )


def _wrap_pbc_gro(gro_path: Path, out_path: Path) -> None:
    """Wrap atom coordinates into the simulation box using modulo arithmetic.

    editconf -box only resizes the box header; atoms near the old boundary end
    up outside the new (smaller) box.  The next grompp run applies PBC and
    effectively teleports those atoms to the opposite face, creating severe
    clashes.  Wrapping here prevents that.

    Uses atom-level modulo, not molecule-level.  Safe for CG Martini chains
    whose contour length (~38 nm for L110) is well below box/2 (~65 nm), so
    no chain spans more than one box image.
    """
    lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(lines[1].strip())
    box_vals = [float(v) for v in lines[n + 2].split()[:3]]
    out_lines = list(lines[:2])
    for line in lines[2 : n + 2]:
        try:
            x = float(line[20:28]) % box_vals[0]
            y = float(line[28:36]) % box_vals[1]
            z = float(line[36:44]) % box_vals[2]
            out_lines.append(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}{line[44:]}")
        except (ValueError, IndexError):
            out_lines.append(line)
    out_lines.append(lines[n + 2])
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _grompp_and_run_em(
    gmx_cmd: str,
    mdp: Path,
    gro: Path,
    top: Path,
    outdir: Path,
    ntomp: int,
    maxwarn: int,
    env: Dict[str, str],
    gpu_id: Optional[str] = None,
    mpi_np: Optional[int] = None,
    mpi_args: Optional[List[str]] = None,
    mdrun_args: Optional[List[str]] = None,
) -> Tuple[Path, Path, Path, Path]:
    """Run one grompp + EM mdrun pass in ``outdir`` (deffnm "em").

    Returns:
        (em.tpr, em.edr, em.log, em.gro) paths.

    Raises:
        RuntimeError: If a command fails or an expected output is missing.
    """
    _ensure_dir(outdir)
    tpr = outdir / "em.tpr"
    edr = outdir / "em.edr"
    log = outdir / "em.log"
    gro_out = outdir / "em.gro"

    mdrun_cmd, env_extra = _build_mdrun_cmd(
        gmx_cmd,
        ["-deffnm", "em", "-ntomp", str(ntomp)],
        gpu_id,
        mpi_np,
        mpi_args or [],
        mdrun_args or [],
    )
    run_env = dict(env)
    run_env.update(env_extra)
    _run([gmx_cmd, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", str(maxwarn)], cwd=outdir, env=run_env)
    _run(mdrun_cmd, cwd=outdir, env=run_env)
    for path in (tpr, edr, log, gro_out):
        if not path.exists():
            raise RuntimeError(f"Missing EM output file: {path}")
    return tpr, edr, log, gro_out


def run_soft_em(cfg: Dict[str, Any]) -> Path:
    """Run the iterative soft-EM loop until converged (see module docstring).

    Per iteration under ``<workdir>/iter_NNN/``: scaled bonded .itp and
    patched .top, an ``em/`` run, extracted potential/pressure .xvg
    files, and the rescaled+wrapped structure fed to the next iteration.
    Convergence (checked from ``min_iter`` on) requires |dEpot| <
    ``e_threshold`` (kJ/mol), Fmax < ``f_threshold`` (kJ/mol/nm), every
    pressure component within ``p_tol`` of ``p_target`` (bar), and the
    bonded ramp completed.  Box updates follow ``box_mode``
    (anisotropic / isotropic / cubic, see :func:`_compute_box_deltas`);
    when the pressure tensor terms are unavailable in the .edr, the
    scalar "Pressure" is used for all three axes.

    Args:
        cfg: Normalized relax config; uses tools/paths/runtime plus the
            ``soft_em`` section (thresholds and ramp settings above,
            stress_terms/stress_use, scale_factor [fraction per bar],
            max_dlen, min_iter/max_iter, box_mode, cubic_rate,
            cubic_max_dlen, maxwarn, restart).

    Returns:
        ``<workdir>/final.gro`` on convergence.

    Raises:
        FileNotFoundError: For missing input files.
        ValueError: For an unknown ``box_mode``.
        RuntimeError: If GROMACS fails or ``max_iter`` is exhausted.
    """
    tools = cfg.get("tools", {})
    runtime = cfg.get("runtime", {})
    paths = cfg.get("paths", {})
    soft_em = cfg.get("soft_em", {})

    gmx = str(tools.get("gmx", "gmx_mpi"))
    system_top = Path(str(paths["system_top"])).resolve()
    bonded_itp = Path(str(paths["bonded_itp"])).resolve()
    start_gro = Path(str(paths["start_gro"])).resolve()
    minim_mdp = Path(str(soft_em["minim_mdp"])).resolve()
    workdir = Path(str(paths["workdir"])).resolve()
    ntomp = int(runtime.get("omp_threads", 1))
    maxwarn = int(soft_em.get("maxwarn", 2))
    gpu_id_raw = runtime.get("gpu_id")
    gpu_id: Optional[str] = str(gpu_id_raw) if gpu_id_raw is not None else None
    mpi_np_raw = runtime.get("mpi_np")
    mpi_np: Optional[int] = int(mpi_np_raw) if mpi_np_raw is not None else None
    mpi_args: List[str] = [str(a) for a in runtime.get("mpi_args", [])]
    mdrun_args: List[str] = [str(value) for value in runtime.get("mdrun_args", [])]

    for path in (system_top, bonded_itp, start_gro, minim_mdp):
        if not path.exists():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    # OMP_NUM_THREADS must exactly match -ntomp passed to mdrun.
    # ntomp is per-rank threads; with MPI: total = mpi_np * ntomp.
    env["OMP_NUM_THREADS"] = str(ntomp)
    env["GMX_OPENMP_MAX_THREADS"] = str(ntomp)

    stress_terms = [str(term) for term in soft_em.get("stress_terms", ["Pres-XX", "Pres-YY", "Pres-ZZ"])]
    stress_use = str(soft_em.get("stress_use", "mean"))
    p_target = float(soft_em.get("p_target", 1.0))
    p_tol = float(soft_em.get("p_tol", 50.0))
    scale_factor = float(soft_em.get("scale_factor", 1e-4))
    max_dlen = float(soft_em.get("max_dlen", 0.001))
    e_threshold = float(soft_em.get("e_threshold", 200.0))
    f_threshold = float(soft_em.get("f_threshold", 300.0))
    min_iter = int(soft_em.get("min_iter", 3))
    max_iter = int(soft_em.get("max_iter", 300))
    bonded_start = float(soft_em.get("bonded_start", 0.1))
    bonded_end = float(soft_em.get("bonded_end", 1.0))
    bonded_ramp_iters = int(soft_em.get("bonded_ramp_iters", 150))
    do_restart = bool(soft_em.get("restart", False))
    box_mode = str(soft_em.get("box_mode", "anisotropic"))
    cubic_rate = float(soft_em.get("cubic_rate", 0.02))
    cubic_max_dlen = float(soft_em.get("cubic_max_dlen", 0.02))
    if box_mode not in {"anisotropic", "isotropic", "cubic"}:
        raise ValueError(f"box_mode must be anisotropic/isotropic/cubic, got '{box_mode}'")

    current = workdir / "current.gro"
    prev_energy: float | None = None
    start_iteration = 1

    if do_restart and workdir.exists():
        # Find the highest iter_N that has both em.gro and scaled.gro
        last_done = 0
        for i in range(1, max_iter + 1):
            if (workdir / f"iter_{i:03d}" / "scaled.gro").exists():
                last_done = i
            else:
                break
        if last_done >= 1:
            start_iteration = last_done + 1
            shutil.copy2(workdir / f"iter_{last_done:03d}" / "scaled.gro", current)
            pot_xvg = workdir / f"iter_{last_done:03d}" / "em" / "em_potential.xvg"
            if pot_xvg.exists():
                prev_energy = _summarize_series(pot_xvg, 1, "mean")
            print(f"=== hydrogel_builder.relax soft_em (RESTART from iter {start_iteration}) ===")
        else:
            _ensure_dir(workdir)
            shutil.copy2(start_gro, current)
            print("=== hydrogel_builder.relax soft_em ===")
    else:
        if workdir.exists():
            backup = workdir.with_name(workdir.name + "_bak")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.move(str(workdir), str(backup))
        _ensure_dir(workdir)
        shutil.copy2(start_gro, current)
        print("=== hydrogel_builder.relax soft_em ===")

    print(f"GMX={gmx}, ntomp={ntomp}")
    print(f"Top: {system_top}")
    print(f"Bonded itp: {bonded_itp}")
    print(f"Start: {current}")

    for iteration in range(start_iteration, max_iter + 1):
        iter_dir = workdir / f"iter_{iteration:03d}"
        _ensure_dir(iter_dir)

        if bonded_ramp_iters <= 1:
            bonded_factor = bonded_end
        else:
            ramp_position = _clamp((iteration - 1) / (bonded_ramp_iters - 1), 0.0, 1.0)
            bonded_factor = bonded_start + ramp_position * (bonded_end - bonded_start)

        iter_top = iter_dir / "system.top"
        iter_itp = iter_dir / "bonded_scaled.itp"
        patch_system_top(system_top, iter_top, bonded_itp.name, iter_itp.name)
        scale_itp_bonded(bonded_itp, iter_itp, bonded_factor)

        em_dir = iter_dir / "em"
        em_tpr, em_edr, em_log, em_gro = _grompp_and_run_em(
            gmx_cmd=gmx,
            mdp=minim_mdp,
            gro=current,
            top=iter_top,
            outdir=em_dir,
            ntomp=ntomp,
            maxwarn=maxwarn,
            env=env,
            gpu_id=gpu_id,
            mpi_np=mpi_np,
            mpi_args=mpi_args,
            mdrun_args=mdrun_args,
        )

        potential_xvg = em_dir / "em_potential.xvg"
        _extract_xvg(gmx, em_edr, potential_xvg, ["Potential"], env)
        energy = _summarize_series(potential_xvg, 1, "mean")
        fmax = _parse_em_fmax(em_log)
        delta_e = None if prev_energy is None else abs(energy - prev_energy)
        prev_energy = energy

        use_tensor = True
        stress_xvg = em_dir / "em_stress.xvg"
        try:
            _extract_xvg(gmx, em_edr, stress_xvg, stress_terms, env)
        except Exception:
            use_tensor = False
            _extract_xvg(gmx, em_edr, stress_xvg, ["Pressure"], env)

        if use_tensor:
            pxx = _summarize_series(stress_xvg, 1, stress_use)
            pyy = _summarize_series(stress_xvg, 2, stress_use)
            pzz = _summarize_series(stress_xvg, 3, stress_use)
        else:
            pressure = _summarize_series(stress_xvg, 1, stress_use)
            pxx = pyy = pzz = pressure

        box_x, box_y, box_z = _parse_gro_box(em_gro)
        d_lx, d_ly, d_lz = _compute_box_deltas(
            pxx, pyy, pzz, box_x, box_y, box_z,
            p_target, scale_factor, max_dlen,
            box_mode, cubic_rate, cubic_max_dlen,
        )

        L_cube = (box_x * box_y * box_z) ** (1.0 / 3.0)
        cubic_dev_pct = (
            max(abs(box_x - L_cube), abs(box_y - L_cube), abs(box_z - L_cube))
            / L_cube * 100.0
        )
        print(
            f"[iter {iteration:03d}] bonded={bonded_factor:.4f} "
            f"Epot={energy:.3e} Fmax={fmax:.3e} "
            f"P=({pxx:.3f},{pyy:.3f},{pzz:.3f}) "
            f"box=({box_x:.1f},{box_y:.1f},{box_z:.1f}) dev={cubic_dev_pct:.1f}%"
        )

        energy_ok = (delta_e is not None) and (delta_e < e_threshold)
        force_ok = fmax < f_threshold
        pressure_ok = (
            abs(pxx - p_target) < p_tol
            and abs(pyy - p_target) < p_tol
            and abs(pzz - p_target) < p_tol
        )
        bonded_ok = abs(bonded_factor - bonded_end) < 1e-8
        if iteration >= min_iter and energy_ok and force_ok and pressure_ok and bonded_ok:
            final_path = workdir / "final.gro"
            shutil.copy2(em_gro, final_path)
            print(f"Converged at iter {iteration}. Wrote {final_path}")
            return final_path

        # Use -scale (not -box) so that atomic coordinates are rescaled
        # proportionally with the box.  -box only updates the box header while
        # leaving atom positions unchanged; atoms near the old boundary then
        # fall outside the new (smaller) box and get PBC-teleported to the
        # opposite face, creating severe clashes (Epot explosions).
        # -scale sx sy sz is the GROMACS equivalent of ASE scale_atoms=True.
        scaled = iter_dir / "scaled.gro"
        _run(
            [
                gmx,
                "editconf",
                "-f",
                str(em_gro),
                "-o",
                str(scaled),
                "-scale",
                f"{1.0 + d_lx:.6f}",
                f"{1.0 + d_ly:.6f}",
                f"{1.0 + d_lz:.6f}",
            ],
            env=env,
            check=True,
        )
        # Safety-net: EM can leave a handful of atoms fractionally outside the
        # box.  After scaling those stay outside (proportionally), so wrap them
        # in to prevent grompp warnings and rare clash events.
        wrapped = iter_dir / "wrapped.gro"
        _wrap_pbc_gro(scaled, wrapped)
        shutil.copy2(wrapped, current)

    raise RuntimeError(f"soft_em did not converge within max_iter={max_iter}. Last structure: {current}")

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$")


def _run(
    cmd: List[str],
    *,
    cwd: Path | None = None,
    env: Dict[str, str] | None = None,
    input_str: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
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
    path.mkdir(parents=True, exist_ok=True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_gro_box(gro_path: Path) -> Tuple[float, float, float]:
    last = gro_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1].split()
    if len(last) < 3:
        raise ValueError(f"Cannot parse box from {gro_path}")
    return tuple(map(float, last[:3]))


def _parse_em_fmax(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum\s+force\s*=\s*([0-9.Ee+\-]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"Fmax\s*=\s*([0-9.Ee+\-]+)", text)
    if match:
        return float(match.group(1))
    raise RuntimeError(f"Could not parse Fmax from {log_path}")


def _find_energy_indices(gmx_cmd: str, edr_file: Path, wanted_names: List[str], env: Dict[str, str]) -> List[int]:
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
    indices = _find_energy_indices(gmx_cmd, edr_file, terms, env)
    selection = "\n".join(str(idx) for idx in indices) + "\n0\n"
    _run(
        [gmx_cmd, "energy", "-f", str(edr_file), "-o", str(out_xvg), "-xvg", "none"],
        input_str=selection,
        env=env,
        check=True,
    )


def _read_xvg_rows(path: Path) -> List[List[float]]:
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
    rows = _read_xvg_rows(path)
    if mode == "last":
        return rows[-1][column]
    return mean(row[column] for row in rows)


def _split_comment(line: str) -> Tuple[str, str]:
    if ";" in line:
        code, comment = line.split(";", 1)
        return code.rstrip(), ";" + comment
    return line.rstrip(), ""


def _is_int_token(token: str) -> bool:
    try:
        int(token)
        return True
    except Exception:
        return False


def _is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float) -> None:
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


def _grompp_and_run_em(
    gmx_cmd: str,
    mdp: Path,
    gro: Path,
    top: Path,
    outdir: Path,
    ntomp: int,
    maxwarn: int,
    env: Dict[str, str],
) -> Tuple[Path, Path, Path, Path]:
    _ensure_dir(outdir)
    tpr = outdir / "em.tpr"
    edr = outdir / "em.edr"
    log = outdir / "em.log"
    gro_out = outdir / "em.gro"

    _run([gmx_cmd, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", str(maxwarn)], cwd=outdir, env=env)
    _run([gmx_cmd, "mdrun", "-deffnm", "em", "-ntomp", str(ntomp)], cwd=outdir, env=env)
    for path in (tpr, edr, log, gro_out):
        if not path.exists():
            raise RuntimeError(f"Missing EM output file: {path}")
    return tpr, edr, log, gro_out


def run_soft_em(cfg: Dict[str, Any]) -> Path:
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
    ntomp = int(soft_em.get("ntomp", runtime.get("omp_threads", 1)))
    maxwarn = int(soft_em.get("maxwarn", 2))

    for path in (system_top, bonded_itp, start_gro, minim_mdp):
        if not path.exists():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    omp_threads = int(runtime.get("omp_threads", ntomp))
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["GMX_OPENMP_MAX_THREADS"] = str(omp_threads)

    if workdir.exists():
        backup = workdir.with_name(workdir.name + "_bak")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(workdir), str(backup))
    _ensure_dir(workdir)

    current = workdir / "current.gro"
    shutil.copy2(start_gro, current)

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

    print("=== hydrogel_builder.relax soft_em ===")
    print(f"GMX={gmx}, ntomp={ntomp}")
    print(f"Top: {system_top}")
    print(f"Bonded itp: {bonded_itp}")
    print(f"Start: {current}")
    prev_energy: float | None = None

    for iteration in range(1, max_iter + 1):
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
        _, em_edr, em_log, em_gro = _grompp_and_run_em(
            gmx_cmd=gmx,
            mdp=minim_mdp,
            gro=current,
            top=iter_top,
            outdir=em_dir,
            ntomp=ntomp,
            maxwarn=maxwarn,
            env=env,
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
        d_lx = _clamp(scale_factor * (pxx - p_target), -max_dlen, max_dlen)
        d_ly = _clamp(scale_factor * (pyy - p_target), -max_dlen, max_dlen)
        d_lz = _clamp(scale_factor * (pzz - p_target), -max_dlen, max_dlen)
        new_box_x = box_x * (1.0 + d_lx)
        new_box_y = box_y * (1.0 + d_ly)
        new_box_z = box_z * (1.0 + d_lz)

        print(
            f"[iter {iteration:03d}] bonded_factor={bonded_factor:.4f} "
            f"Epot={energy:.3e} Fmax={fmax:.3e} "
            f"P=({pxx:.3f}, {pyy:.3f}, {pzz:.3f})"
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

        scaled = iter_dir / "scaled.gro"
        _run(
            [
                gmx,
                "editconf",
                "-f",
                str(em_gro),
                "-o",
                str(scaled),
                "-box",
                f"{new_box_x:.6f}",
                f"{new_box_y:.6f}",
                f"{new_box_z:.6f}",
            ],
            env=env,
            check=True,
        )
        shutil.copy2(scaled, current)

    raise RuntimeError(f"soft_em did not converge within max_iter={max_iter}. Last structure: {current}")

#!/usr/bin/env python3
"""cell_opt_em.py

EM-only anisotropic box relaxation loop for GROMACS.

Per iteration:
  - Build a per-iteration topology that includes a *local* scaled bonded .itp
  - Run EM (grompp + mdrun)
  - Extract Pres-XX/Pres-YY/Pres-ZZ (or fallback to Pressure)
  - Update box lengths with small fractional steps (anisotropic if tensor available)
  - Repeat

Notes
-----
- This is a *preconditioner* for difficult systems (e.g., stretched CG gels) and
  does not replace a proper NPT equilibration.
- EM pressures can be noisy/biased. Use small scale steps (max_dlen ~ 0.0005-0.001).

Example
-------
python -u cell_opt_em.py \
  --top system.top \
  --bonded-itp /path/to/initial_hydrogel.itp \
  --start-gro current.gro \
  --gmx gmx_mpi --ntomp 32 \
  --p-target 1.0 --p-tol 50 \
  --scale-factor 1e-4 --max-dlen 0.001 \
  --bonded-start 0.1 --bonded-end 1.0 --bonded-ramp-iters 75
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean

# -------------------------
# subprocess helper
# -------------------------

def run(cmd, *, input_str: str | None = None, cwd: Path | None = None, env=None, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        input=(input_str.encode() if input_str is not None else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={p.returncode}): {' '.join(map(str, cmd))}\n"
            f"--- output ---\n{p.stdout.decode(errors='replace')}"
        )
    return p


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_gro_box(gro_path: Path) -> tuple[float, float, float]:
    last = gro_path.read_text().strip().splitlines()[-1].split()
    if len(last) < 3:
        raise ValueError(f"Cannot parse box from {gro_path}")
    return tuple(map(float, last[:3]))


def parse_em_fmax_from_log(log_path: Path) -> float:
    text = log_path.read_text(errors="replace")
    # Typical lines:
    # "Maximum force =  7.829e+02 on atom ..."
    m = re.search(r"Maximum\s+force\s*=\s*([0-9.Ee+\-]+)", text)
    if m:
        return float(m.group(1))
    # fallback
    m = re.search(r"Fmax\s*=\s*([0-9.Ee+\-]+)", text)
    if m:
        return float(m.group(1))
    raise RuntimeError(f"Could not parse Fmax from {log_path}.")


# -------------------------
# gmx energy extraction
# -------------------------

def find_energy_indices(gmx_cmd: str, edr_file: Path, wanted_names: list[str]) -> list[int]:
    # Probe menu with selection 0
    probe = run(
        [gmx_cmd, "energy", "-f", str(edr_file), "-o", os.devnull, "-xvg", "none"],
        input_str="0\n",
        check=False,
    )
    text = probe.stdout.decode(errors="replace")

    idx_map: dict[str, int] = {}
    for line in text.splitlines():
        # multiple pairs per line: " 7 Potential  8 Pressure ..."
        for m in re.finditer(r"(\d+)\s+([A-Za-z0-9#\-\*\.\(\)]+)", line):
            idx = int(m.group(1))
            name = m.group(2).strip()
            if name == "0":
                continue
            idx_map[name] = idx

    missing = [n for n in wanted_names if n not in idx_map]
    if missing:
        sample = "\n".join([ln for ln in text.splitlines() if re.search(r"\b\d+\s+\S+", ln)][:120])
        raise RuntimeError(
            f"Could not find energy terms in {edr_file}: {missing}\n"
            f"--- energy menu excerpt ---\n{sample}\n"
            f"Tip: run `{gmx_cmd} energy -f {edr_file}` interactively to see exact names."
        )

    return [idx_map[n] for n in wanted_names]


def extract_xvg(gmx_cmd: str, edr_file: Path, out_xvg: Path, terms: list[str], b_ps: float | None = None, e_ps: float | None = None) -> None:
    indices = find_energy_indices(gmx_cmd, edr_file, terms)
    flags = [gmx_cmd, "energy", "-f", str(edr_file), "-o", str(out_xvg), "-xvg", "none"]
    if b_ps is not None:
        flags += ["-b", str(b_ps)]
    if e_ps is not None:
        flags += ["-e", str(e_ps)]
    sel = "\n".join(str(i) for i in indices) + "\n0\n"
    run(flags, input_str=sel, check=True)


def read_xvg_rows(xvg_path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in xvg_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No data parsed from {xvg_path}")
    return rows


def summarize_series(xvg_path: Path, col_index: int, use: str) -> float:
    rows = read_xvg_rows(xvg_path)
    if use == "last":
        return rows[-1][col_index]
    return mean([r[col_index] for r in rows])


# -------------------------
# topology scaling
# -------------------------

SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$")


def split_comment(line: str) -> tuple[str, str]:
    if ";" in line:
        code, comment = line.split(";", 1)
        return code.rstrip(), ";" + comment
    return line.rstrip(), ""


def is_int_token(tok: str) -> bool:
    try:
        int(tok)
        return True
    except Exception:
        return False


def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float) -> None:
    """Scale bonded force constants in Martini-ish itp.

    - [ bonds ]: scale last numeric token (kb)
    - [ angles ]: scale last numeric token (ktheta)
    - [ dihedrals ]:
        funct==3 (RB): scale coefficients c0.. (tokens after funct)
        else: scale k at token index 6 if present, else scale last float token

    If a line can't be parsed safely, it is copied unchanged.
    """
    cur_section: str | None = None
    out_lines: list[str] = []

    for raw in in_itp.read_text(errors="replace").splitlines(True):
        line = raw.rstrip("\n")
        m = SECTION_RE.match(line.strip())
        if m:
            cur_section = m.group(1).lower()
            out_lines.append(line + "\n")
            continue

        if cur_section not in {"bonds", "angles", "dihedrals"}:
            out_lines.append(line + "\n")
            continue

        code, comment = split_comment(line)
        stripped = code.strip()
        if not stripped or stripped.startswith(";"):
            out_lines.append(line + "\n")
            continue

        toks = stripped.split()
        prefix_ws = re.match(r"^\s*", code).group(0)

        try:
            if cur_section in {"bonds", "angles"}:
                idx = None
                for i in range(len(toks) - 1, -1, -1):
                    if is_float_token(toks[i]):
                        idx = i
                        break
                if idx is None:
                    out_lines.append(line + "\n")
                    continue
                toks[idx] = f"{float(toks[idx]) * factor:.8g}"
                out_lines.append(prefix_ws + " ".join(toks) + ((" " + comment) if comment else "") + "\n")
                continue

            # dihedrals
            if len(toks) < 5 or not is_int_token(toks[4]):
                out_lines.append(line + "\n")
                continue
            funct = int(toks[4])

            if funct == 3:
                # RB dihedrals: scale all floats after funct
                for j in range(5, len(toks)):
                    if is_float_token(toks[j]):
                        toks[j] = f"{float(toks[j]) * factor:.8g}"
            else:
                # periodic-like: i j k l funct phi0 k mult
                if len(toks) >= 8 and is_float_token(toks[6]):
                    toks[6] = f"{float(toks[6]) * factor:.8g}"
                else:
                    idx = None
                    for i in range(len(toks) - 1, -1, -1):
                        if is_float_token(toks[i]):
                            idx = i
                            break
                    if idx is not None:
                        toks[idx] = f"{float(toks[idx]) * factor:.8g}"

            out_lines.append(prefix_ws + " ".join(toks) + ((" " + comment) if comment else "") + "\n")

        except Exception:
            out_lines.append(line + "\n")

    out_itp.write_text("".join(out_lines))


def patch_system_top(in_top: Path, out_top: Path, bonded_itp_basename: str, new_local_itp_name: str) -> None:
    """Replace the #include line whose included file basename == bonded_itp_basename."""
    pat = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')
    out: list[str] = []

    for raw in in_top.read_text(errors="replace").splitlines(True):
        line = raw.rstrip("\n")
        m = pat.match(line.strip())
        if m:
            inc = m.group(1)
            if Path(inc).name == bonded_itp_basename:
                out.append(f'#include "{new_local_itp_name}"\n')
                continue
        out.append(raw if raw.endswith("\n") else raw + "\n")

    out_top.write_text("".join(out))


# -------------------------
# gromacs EM runner
# -------------------------

def grompp_and_run_em(gmx_cmd: str, mdp: Path, gro: Path, top: Path, outdir: Path, ntomp: int, maxwarn: int) -> tuple[Path, Path, Path, Path]:
    ensure_dir(outdir)
    tpr = outdir / "em.tpr"
    edr = outdir / "em.edr"
    log = outdir / "em.log"
    gro_out = outdir / "em.gro"

    run([gmx_cmd, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", str(maxwarn)], cwd=outdir)
    run([gmx_cmd, "mdrun", "-deffnm", "em", "-ntomp", str(ntomp)], cwd=outdir)

    if not (tpr.exists() and edr.exists() and log.exists() and gro_out.exists()):
        raise RuntimeError(f"Missing EM outputs in {outdir}")

    return tpr, edr, log, gro_out


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="EM-only anisotropic box relaxation loop (Pres-XX/YY/ZZ).")
    ap.add_argument("--gmx", default=os.environ.get("GMX_CMD", "gmx_mpi"))
    ap.add_argument("--ntomp", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "32")))

    ap.add_argument("--top", required=True, help="system.top used for grompp")
    ap.add_argument("--bonded-itp", required=True, help=".itp that contains the bonded terms to scale")
    ap.add_argument("--start-gro", required=True)
    ap.add_argument("--minim-mdp", default="minim.mdp")

    ap.add_argument("--workdir", default="box_relax_loop_em")
    ap.add_argument("--maxwarn", type=int, default=2)

    # stress extraction
    ap.add_argument("--stress-terms", nargs="+", default=["Pres-XX", "Pres-YY", "Pres-ZZ"],
                    help="Energy terms to use. Default tries tensor components; fallback to Pressure if missing.")
    ap.add_argument("--stress-use", choices=["mean", "last"], default="mean")

    # control
    ap.add_argument("--p-target", type=float, default=1.0)
    ap.add_argument("--p-tol", type=float, default=50.0, help="bar tolerance (applied per-axis if tensor used)")
    ap.add_argument("--scale-factor", type=float, default=1e-4, help="fractional dL/L per bar")
    ap.add_argument("--max-dlen", type=float, default=0.001, help="max |dL/L| per iteration")

    # stopping
    ap.add_argument("--e-threshold", type=float, default=200.0, help="|ΔE| threshold (kJ/mol)")
    ap.add_argument("--f-threshold", type=float, default=300.0, help="Fmax threshold (kJ/mol/nm)")
    ap.add_argument("--min-iter", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=300)

    # bonded ramp
    ap.add_argument("--bonded-start", type=float, default=0.1)
    ap.add_argument("--bonded-end", type=float, default=1.0)
    ap.add_argument("--bonded-ramp-iters", type=int, default=150)

    args = ap.parse_args()

    gmx = args.gmx
    top = Path(args.top).resolve()
    bonded_itp = Path(args.bonded_itp).resolve()
    start_gro = Path(args.start_gro).resolve()
    minim_mdp = Path(args.minim_mdp).resolve()
    workdir = Path(args.workdir).resolve()

    for p in [top, bonded_itp, start_gro, minim_mdp]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Workdir reset
    if workdir.exists():
        bak = workdir.with_name(workdir.name + "_bak")
        if bak.exists():
            shutil.rmtree(bak)
        shutil.move(str(workdir), str(bak))
    ensure_dir(workdir)

    current = workdir / "current.gro"
    shutil.copy2(start_gro, current)

    print("=== EM-only box relaxation loop ===", flush=True)
    print(f"GMX={gmx}, ntomp={args.ntomp}", flush=True)
    print(f"Top: {top}", flush=True)
    print(f"Bonded itp: {bonded_itp}", flush=True)
    print(f"Start: {current}", flush=True)
    print(f"Stress terms: {args.stress-terms if hasattr(args,'stress-terms') else args.stress_terms}", flush=True)
    print(f"Stop: it>={args.min_iter} AND |ΔE|<{args.e_threshold} AND Fmax<{args.f_threshold} AND |Pii-Pt|<{args.p_tol}", flush=True)
    print(f"Bonded ramp: {args.bonded_start} -> {args.bonded_end} over {args.bonded_ramp_iters} iters", flush=True)

    prev_E: float | None = None

    for it in range(1, args.max_iter + 1):
        itdir = workdir / f"iter_{it:03d}"
        ensure_dir(itdir)

        # bonded factor ramp (linear)
        if args.bonded_ramp_iters <= 1:
            bf = args.bonded_end
        else:
            t = clamp((it - 1) / (args.bonded_ramp_iters - 1), 0.0, 1.0)
            bf = args.bonded_start + t * (args.bonded_end - args.bonded_start)

        it_top = itdir / "system.top"
        it_itp = itdir / "bonded_scaled.itp"
        patch_system_top(top, it_top, bonded_itp.name, it_itp.name)
        scale_itp_bonded(bonded_itp, it_itp, bf)

        emdir = itdir / "em"
        _, em_edr, em_log, em_gro = grompp_and_run_em(
            gmx_cmd=gmx, mdp=minim_mdp, gro=current, top=it_top, outdir=emdir, ntomp=args.ntomp, maxwarn=args.maxwarn
        )

        # Extract potential
        em_pot_xvg = emdir / "em_potential.xvg"
        extract_xvg(gmx, em_edr, em_pot_xvg, terms=["Potential"])
        E = summarize_series(em_pot_xvg, 1, "mean")
        Fmax = parse_em_fmax_from_log(em_log)
        dE = None if prev_E is None else abs(E - prev_E)
        prev_E = E

        # Extract stress tensor, fallback to Pressure
        use_tensor = True
        stress_terms = args.stress_terms
        em_stress_xvg = emdir / "em_stress.xvg"
        try:
            extract_xvg(gmx, em_edr, em_stress_xvg, terms=stress_terms)
        except Exception:
            use_tensor = False
            stress_terms = ["Pressure"]
            extract_xvg(gmx, em_edr, em_stress_xvg, terms=stress_terms)

        # Summarize
        if use_tensor:
            Pxx = summarize_series(em_stress_xvg, 1, args.stress_use)
            Pyy = summarize_series(em_stress_xvg, 2, args.stress_use)
            Pzz = summarize_series(em_stress_xvg, 3, args.stress_use)
        else:
            P = summarize_series(em_stress_xvg, 1, args.stress_use)
            Pxx = Pyy = Pzz = P

        bx0, by0, bz0 = parse_gro_box(em_gro)

        # Anisotropic scaling (per-axis)
        dLx = clamp(args.scale_factor * (Pxx - args.p_target), -args.max_dlen, args.max_dlen)
        dLy = clamp(args.scale_factor * (Pyy - args.p_target), -args.max_dlen, args.max_dlen)
        dLz = clamp(args.scale_factor * (Pzz - args.p_target), -args.max_dlen, args.max_dlen)

        bx1 = bx0 * (1.0 + dLx)
        by1 = by0 * (1.0 + dLy)
        bz1 = bz0 * (1.0 + dLz)

        print(f"[iter {it:03d}] bonded_factor={bf:.4f}", flush=True)
        print(f"[iter {it:03d}] EM: Epot={E: .3e} kJ/mol, Fmax={Fmax: .3e} kJ/mol/nm" + (f", |ΔE|={dE: .3e}" if dE is not None else ", |ΔE|=N/A"), flush=True)
        if use_tensor:
            print(f"[iter {it:03d}] EM stress (bar): Pxx={Pxx: .3f}  Pyy={Pyy: .3f}  Pzz={Pzz: .3f}", flush=True)
        else:
            print(f"[iter {it:03d}] EM pressure (bar): P={Pxx: .3f} (tensor unavailable)", flush=True)
        print(f"[iter {it:03d}] Box(cur)={bx0:.4f} {by0:.4f} {bz0:.4f} nm", flush=True)
        print(f"[iter {it:03d}] dL/L ={dLx:+.5f} {dLy:+.5f} {dLz:+.5f}  -> Box(new)={bx1:.4f} {by1:.4f} {bz1:.4f} nm", flush=True)

        # Convergence checks
        e_ok = (dE is not None) and (dE < args.e_threshold)
        f_ok = Fmax < args.f_threshold
        p_ok = (abs(Pxx - args.p_target) < args.p_tol) and (abs(Pyy - args.p_target) < args.p_tol) and (abs(Pzz - args.p_target) < args.p_tol)
        bf_ok = abs(bf - args.bonded_end) < 1e-8

        if it >= args.min_iter and e_ok and f_ok and p_ok and bf_ok:
            final = workdir / "final.gro"
            shutil.copy2(em_gro, final)
            print(f"=== Converged at iter {it}. Wrote {final} ===", flush=True)
            return

        # Apply new box based on EM structure
        scaled = itdir / "scaled.gro"
        run([gmx, "editconf", "-f", str(em_gro), "-o", str(scaled), "-box", f"{bx1:.6f}", f"{by1:.6f}", f"{bz1:.6f}"], check=True)
        shutil.copy2(scaled, current)

    raise RuntimeError(f"Did not converge within max_iter={args.max_iter}. Last structure: {current}")


if __name__ == "__main__":
    main()

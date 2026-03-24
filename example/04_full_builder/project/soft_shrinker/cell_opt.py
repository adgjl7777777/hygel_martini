#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean

# -----------------------
# utils
# -----------------------
def run(cmd, *, input_str=None, cwd=None, env=None, check=True):
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
            f"Command failed (rc={p.returncode}): {' '.join(cmd)}\n"
            f"--- output ---\n{p.stdout.decode(errors='replace')}"
        )
    return p

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_gro_box(gro_path: Path):
    last = gro_path.read_text().strip().splitlines()[-1].split()
    if len(last) < 3:
        raise ValueError(f"Cannot parse box from {gro_path}")
    return tuple(map(float, last[:3]))

def parse_em_fmax_from_log(log_path: Path):
    text = log_path.read_text(errors="replace")
    for pat in [r"Maximum\s+force\s*=\s*([0-9.Ee+\-]+)", r"Fmax\s*=\s*([0-9.Ee+\-]+)"]:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"Could not parse Fmax from {log_path}. Search for 'Maximum force' in the file.")

# -----------------------
# gmx energy extraction
# -----------------------
def find_energy_indices(gmx_cmd, edr_file: Path, wanted_names):
    probe = run(
        [gmx_cmd, "energy", "-f", str(edr_file), "-o", os.devnull, "-xvg", "none"],
        input_str="0\n",
        check=False,
    )
    text = probe.stdout.decode(errors="replace")

    idx_map = {}
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
        sample = "\n".join([ln for ln in text.splitlines() if re.search(r"\b\d+\s+\S+", ln)][:80])
        raise RuntimeError(
            f"Could not find energy terms in {edr_file}: {missing}\n"
            f"--- energy menu excerpt ---\n{sample}\n"
            f"Tip: run `{gmx_cmd} energy -f {edr_file}` interactively to see exact names."
        )

    return [idx_map[n] for n in wanted_names]

def extract_xvg(gmx_cmd, edr_file: Path, out_xvg: Path, terms, b_ps=None, e_ps=None):
    indices = find_energy_indices(gmx_cmd, edr_file, terms)
    flags = [gmx_cmd, "energy", "-f", str(edr_file), "-o", str(out_xvg), "-xvg", "none"]
    if b_ps is not None:
        flags += ["-b", str(b_ps)]
    if e_ps is not None:
        flags += ["-e", str(e_ps)]
    sel = "\n".join(str(i) for i in indices) + "\n0\n"
    run(flags, input_str=sel, check=True)

def read_xvg_columns(xvg_path: Path):
    rows = []
    for line in xvg_path.read_text().splitlines():
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

def mean_of_term(xvg_path: Path, col_index: int):
    rows = read_xvg_columns(xvg_path)
    return mean([r[col_index] for r in rows])

# -----------------------
# topology softening
# -----------------------
SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$")

def split_comment(line: str):
    """Return (code, comment_with_semicolon_or_empty)."""
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

def scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float):
    """
    Scale bonded force constants in Martini-ish .itp:
      [ bonds ] : scale last numeric field (kb)
      [ angles ]: scale last numeric field (ktheta)
      [ dihedrals ]:
         - funct 1/2/4/9 (periodic/improper style): scale k at position 6 (0-based)
           i j k l funct phi0 k mult
         - funct 3 (RB): scale coefficients c0..c5 (positions 5..end)
           i j k l funct c0 c1 c2 c3 c4 c5
    This is heuristic but works for typical Martini network files.
    """
    cur_section = None
    out_lines = []

    for raw in in_itp.read_text().splitlines(True):
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
        # Keep original indentation style minimal
        prefix_ws = re.match(r"^\s*", code).group(0)

        try:
            if cur_section in {"bonds", "angles"}:
                # scale last numeric token
                # (bonds: ... b0 kb), (angles: ... th0 ktheta)
                # find last float token and scale it
                idx = None
                for i in range(len(toks) - 1, -1, -1):
                    if is_float_token(toks[i]):
                        idx = i
                        break
                if idx is None:
                    out_lines.append(line + "\n")
                    continue
                val = float(toks[idx]) * factor
                toks[idx] = f"{val:.8g}"
                out_lines.append(prefix_ws + " ".join(toks) + ((" " + comment) if comment else "") + "\n")
                continue

            if cur_section == "dihedrals":
                # Need funct
                if len(toks) < 5 or not is_int_token(toks[4]):
                    out_lines.append(line + "\n")
                    continue
                funct = int(toks[4])

                if funct == 3:
                    # RB: scale all coefficients after funct (positions 5..end) that are floats
                    for j in range(5, len(toks)):
                        if is_float_token(toks[j]):
                            toks[j] = f"{float(toks[j]) * factor:.8g}"
                    out_lines.append(prefix_ws + " ".join(toks) + ((" " + comment) if comment else "") + "\n")
                    continue
                else:
                    # periodic-like: i j k l funct phi0 k mult  -> scale k (index 6)
                    # If format differs, fall back to scaling the last float before a final integer multiplicity.
                    if len(toks) >= 8 and is_float_token(toks[6]):
                        toks[6] = f"{float(toks[6]) * factor:.8g}"
                    else:
                        # fallback: scale last float token
                        idx = None
                        for i in range(len(toks) - 1, -1, -1):
                            if is_float_token(toks[i]):
                                idx = i
                                break
                        if idx is not None:
                            toks[idx] = f"{float(toks[idx]) * factor:.8g}"
                    out_lines.append(prefix_ws + " ".join(toks) + ((" " + comment) if comment else "") + "\n")
                    continue

        except Exception:
            # If parsing fails, preserve original line (safer than corrupting)
            out_lines.append(line + "\n")

    out_itp.write_text("".join(out_lines))
def patch_system_top(in_top: Path, out_top: Path, bonded_itp_basename: str, new_local_itp_name: str):
    """
    Replace include line that includes bonded_itp_basename with local include new_local_itp_name.
    Works for absolute/relative includes.
    """
    out = []
    pat = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')

    for raw in in_top.read_text().splitlines(True):
        line = raw.rstrip("\n")
        m = pat.match(line.strip())
        if m:
            inc_path = m.group(1)
            inc_base = Path(inc_path).name
            if inc_base == bonded_itp_basename:
                out.append(f'#include "{new_local_itp_name}"\n')
                continue

        out.append(raw if raw.endswith("\n") else raw + "\n")

    out_top.write_text("".join(out))



# -----------------------
# grompp + mdrun
# -----------------------
def grompp_and_run(gmx_cmd, deffnm: str, mdp: Path, gro: Path, top: Path, outdir: Path, ntomp: int):
    ensure_dir(outdir)
    tpr = outdir / f"{deffnm}.tpr"
    edr = outdir / f"{deffnm}.edr"
    log = outdir / f"{deffnm}.log"
    gro_out = outdir / f"{deffnm}.gro"

    run([gmx_cmd, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", "2"],
        cwd=outdir, check=True)
    run([gmx_cmd, "mdrun", "-deffnm", deffnm, "-ntomp", str(ntomp)],
        cwd=outdir, check=True)

    if not edr.exists() or not log.exists() or not gro_out.exists():
        raise RuntimeError(f"Missing outputs after mdrun in {outdir}")
    return tpr, edr, log, gro_out

# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Loop: (softened topology) EM -> NVT(10ps) -> avg Pressure -> scale box -> repeat. "
                    "Bonded constants ramp from start to end."
    )
    ap.add_argument("--gmx", default=os.environ.get("GMX_CMD", "gmx_mpi"))
    ap.add_argument("--top", required=True)
    ap.add_argument("--start-gro", required=True)
    ap.add_argument("--minim-mdp", default="minim.mdp")
    ap.add_argument("--nvt-mdp", default="nvt_3ps.mdp")
    ap.add_argument("--workdir", default="box_relax_loop")
    ap.add_argument(
    "--bonded-itp",
    required=True,
    help="Path to itp file that contains bonded terms (bonds/angles/dihedrals)"
)
    ap.add_argument("--ntomp", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "64")))

    # pressure scaling
    ap.add_argument("--p-target", type=float, default=1.0)
    ap.add_argument("--p-tol", type=float, default=20.0)
    ap.add_argument("--avg-beg", type=float, default=1.0) ###
    ap.add_argument("--avg-end", type=float, default=3.0) ###
    ap.add_argument("--scale-factor", type=float, default=2e-4, help="dL/L per bar (small!)")
    ap.add_argument("--max-dlen", type=float, default=0.005, help="max |dL/L| per iter, e.g. 0.005 = 0.5%")

    # stop
    ap.add_argument("--e-threshold", type=float, default=100.0)
    ap.add_argument("--f-threshold", type=float, default=200.0)
    ap.add_argument("--min-iter", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=150)

    # bonded ramp
    ap.add_argument("--bonded-start", type=float, default=0.1)
    ap.add_argument("--bonded-end", type=float, default=1.0)
    ap.add_argument("--bonded-ramp-iters", type=int, default=75,
                    help="iterations to reach bonded-end (linear ramp)")

    # safety: if EM Epot explodes, rollback scaling aggressiveness
    ap.add_argument("--epot-spike", type=float, default=1e8,
                    help="If |Epot| exceeds this, treat as spike and reduce max-dlen.")
    args = ap.parse_args()

    gmx = args.gmx
    top = Path(args.top).resolve()
    start_gro = Path(args.start_gro).resolve()
    minim_mdp = Path(args.minim_mdp).resolve()
    nvt_mdp = Path(args.nvt_mdp).resolve()
    workdir = Path(args.workdir).resolve()

    for p in [top, start_gro, minim_mdp, nvt_mdp]:
        if not p.exists():
            raise FileNotFoundError(p)

    # locate the real itp (from include)
    initial_itp = Path(args.bonded_itp).resolve()
    if not initial_itp.exists():
        raise FileNotFoundError(f"--bonded-itp not found: {initial_itp}")


    # Prepare workdir (backup old)
    if workdir.exists():
        bak = workdir.with_name(workdir.name + "_bak")
        if bak.exists():
            shutil.rmtree(bak)
        shutil.move(str(workdir), str(bak))
    ensure_dir(workdir)

    current = workdir / "current.gro"
    shutil.copy2(start_gro, current)

    prev_em_E = None
    local_top = workdir / "system_local.top"
    local_itp = workdir / "initial_hydrogel_scaled.itp"

    print("=== Loop start ===", flush=True)
    print(f"GMX={gmx}, ntomp={args.ntomp}", flush=True)
    print(f"Top: {top}", flush=True)
    print(f"Include itp: {initial_itp}", flush=True)
    print(f"Start: {current}", flush=True)
    print(f"Stop: it>={args.min_iter} AND |ΔE|<{args.e_threshold} AND Fmax<{args.f_threshold} AND |P-Pt|<{args.p_tol}", flush=True)
    print(f"Bonded ramp: {args.bonded_start} -> {args.bonded_end} over {args.bonded_ramp_iters} iters", flush=True)

    max_dlen = args.max_dlen

    for it in range(1, args.max_iter + 1):
        itdir = workdir / f"iter_{it:03d}"
        ensure_dir(itdir)

        # bonded factor (linear ramp)
        if args.bonded_ramp_iters <= 1:
            bonded_factor = args.bonded_end
        else:
            t = clamp((it - 1) / (args.bonded_ramp_iters - 1), 0.0, 1.0)
            bonded_factor = args.bonded_start + t * (args.bonded_end - args.bonded_start)

        # write scaled itp + patched top into iteration dir (keeps artifacts per iter)
        it_top = itdir / "system.top"
        it_itp = itdir / "initial_hydrogel_scaled.itp"
        patch_system_top(top, it_top, Path(args.bonded_itp).name, it_itp.name)
        scale_itp_bonded(initial_itp, it_itp, bonded_factor)

        # 1) EM
        emdir = itdir / "em"
        _, em_edr, em_log, em_gro = grompp_and_run(
            gmx_cmd=gmx, deffnm="em",
            mdp=minim_mdp, gro=current, top=it_top,
            outdir=emdir, ntomp=args.ntomp
        )

        em_xvg = emdir / "em_potential.xvg"
        extract_xvg(gmx, em_edr, em_xvg, terms=["Potential"])
        em_E = mean_of_term(em_xvg, 1)
        em_Fmax = parse_em_fmax_from_log(em_log)

        dE = None if prev_em_E is None else abs(em_E - prev_em_E)
        prev_em_E = em_E

        # spike safety: if EM energy is ridiculous, back off max_dlen for future iterations
        if abs(em_E) > args.epot_spike:
            max_dlen = max(1e-4, max_dlen * 0.5)

        print(f"[iter {it:03d}] bonded_factor={bonded_factor:.4f}  max_dlen={max_dlen:.5f}", flush=True)
        print(f"[iter {it:03d}] EM: Epot={em_E: .3e} kJ/mol, Fmax={em_Fmax: .3e} kJ/mol/nm"
              + (f", |ΔE|={dE: .3e}" if dE is not None else ", |ΔE|=N/A"),
              flush=True)

        # 2) NVT
        nvtdir = itdir / "nvt"
        _, nvt_edr, nvt_log, nvt_gro = grompp_and_run(
            gmx_cmd=gmx, deffnm="nvt",
            mdp=nvt_mdp, gro=em_gro, top=it_top,
            outdir=nvtdir, ntomp=args.ntomp
        )

        # 3) Avg pressure in [avg-beg, avg-end]
        nvt_xvg = nvtdir / "nvt_press.xvg"
        extract_xvg(gmx, nvt_edr, nvt_xvg, terms=["Pressure"], b_ps=args.avg_beg, e_ps=args.avg_end)
        Pavg = mean_of_term(nvt_xvg, 1)

        # 4) Pressure-based isotropic scaling
        # NOTE: use Pt - P (negative feedback)
        dp = Pavg - args.p_target
        dlen = clamp(args.scale_factor * dp, -max_dlen, max_dlen)
        scale = 1.0 + dlen

        bx0, by0, bz0 = parse_gro_box(nvt_gro)
        bx1, by1, bz1 = bx0 * scale, by0 * scale, bz0 * scale

        print(f"[iter {it:03d}] NVT avg({args.avg_beg}-{args.avg_end} ps): P={Pavg: .3f} bar, Box(cur)={bx0:.4f} {by0:.4f} {bz0:.4f} nm", flush=True)
        print(f"[iter {it:03d}] Scale: dp={dp: .3f} bar -> dL/L={dlen: .5f}, new box={bx1:.4f} {by1:.4f} {bz1:.4f} nm", flush=True)

        # 5) Stop criteria
        p_ok = abs(Pavg - args.p_target) < args.p_tol
        f_ok = em_Fmax < args.f_threshold
        e_ok = (dE is not None) and (dE < args.e_threshold)
        min_ok = it >= args.min_iter

        if min_ok and e_ok and f_ok and p_ok and abs(bonded_factor - args.bonded_end) < 1e-6:
            final = workdir / "final.gro"
            shutil.copy2(nvt_gro, final)
            print(f"=== Converged at iter {it}. Wrote {final} ===", flush=True)
            return

        # 6) Apply new box
        scaled = itdir / "scaled.gro"
        run([gmx, "editconf", "-f", str(nvt_gro), "-o", str(scaled),
             "-box", f"{bx1:.6f}", f"{by1:.6f}", f"{bz1:.6f}"], check=True)
        shutil.copy2(scaled, current)

    raise RuntimeError(f"Did not converge within max_iter={args.max_iter}. Last structure: {current}")

if __name__ == "__main__":
    main()

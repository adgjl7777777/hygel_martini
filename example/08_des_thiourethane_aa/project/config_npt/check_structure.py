#!/usr/bin/env python3
"""Is anything actually moving, and has the builder's lattice been forgotten?

``check_convergence.py`` screens the thermodynamics: density, volume and energy
must stop drifting. It says in its own output that structural equilibrium is
not assessed there. This is that second screen, and the PEG pilot cell is why
it exists.

That cell passed every construction check and ran a clean 30 ns NPT, and its
density kept climbing the whole way. The reason was not the protocol. Chloride
moved 1.2 A in 6 ns -- less than its own radius -- and its mean square
displacement grew as t^0.41 rather than t^1. Nothing was exchanging neighbours.
The cell was a glass, the density climb was physical ageing rather than
equilibration, and no amount of extra nanoseconds would have finished it.
Meanwhile the regular lattice the builder starts from had barely decayed, so
every structural average would have reported the builder's arrangement rather
than the material's.

Neither fact is visible in an energy file. Both are cheap to measure. So:

**Mobility.** Fit the log-log slope of the mean square displacement over a
stated lag window. A liquid gives 1. Ballistic motion at very short lags gives
2. A caged particle rattling in place gives something near 0. The screen also
reports the actual root-mean-square displacement at the longest lag in
nanometres, because a slope can look healthy while the particle has moved a
tenth of a bond length.

**Lattice memory.** The builder seeds junctions on a cubic lattice with a known
number of repeats per axis, which puts a Bragg-like peak at the matching
wavevector. Comparing the structure factor there between the first and last
frame says how much of that ordering has decayed. Perfect order gives N, the
number of scatterers; a random arrangement gives about 1.

Exit 0 = mobile and the lattice is forgotten, 1 = not yet, 2 = cannot judge.

    check_structure.py npt.tpr npt.xtc --mobility-sel "resname CL" \\
                       --order-sel "resname HEX" --order-repeats 4

Both screens can be run alone: pass only the selection you want.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple

#: A liquid's mean square displacement grows linearly with lag. Anything below
#: this is subdiffusive enough that neighbour exchange cannot be assumed, and
#: structural averages are reporting whatever the run started from.
DEFAULT_ALPHA_MIN = 0.80

#: The particle must also have gone somewhere. A typical first-neighbour
#: distance in these systems is around 0.4 nm, so a displacement below this has
#: not sampled a single coordination change however clean the slope looks.
DEFAULT_MIN_DISPLACEMENT_NM = 0.5

#: Fraction of the seeded lattice ordering that must have decayed. The builder's
#: arrangement is an initial condition, not a structure; carrying it into an
#: average silently biases every neighbour statistic.
DEFAULT_MAX_ORDER_RETAINED = 0.25


class CannotJudge(RuntimeError):
    """The inputs cannot support a verdict (exit 2)."""


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, **kw)


def _require(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise CannotJudge(f"{what} not found: {path}")


# ---------------------------------------------------------------- mobility
def mean_square_displacement(tpr: str, xtc: str, selection: str, gmx: str,
                             begin_ps: float, trestart_ps: float,
                             workdir: str) -> List[Tuple[float, float]]:
    """(lag_ps, msd_nm2) from ``gmx msd``.

    ``trestart`` must be at least the trajectory's frame spacing or GROMACS
    refuses, so it is passed explicitly rather than left at its default.
    """
    out = os.path.join(workdir, "msd.xvg")
    cmd = [gmx, "msd", "-s", tpr, "-f", xtc, "-o", out,
           "-sel", selection, "-trestart", f"{trestart_ps}"]
    if begin_ps:
        cmd += ["-b", f"{begin_ps}"]
    res = _run(cmd)
    if res.returncode != 0 or not os.path.exists(out):
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-6:]
        raise CannotJudge("gmx msd failed: " + " | ".join(tail))
    series = []
    with open(out) as handle:
        for line in handle:
            if not line or line[0] in "#@":
                continue
            parts = line.split()
            if len(parts) >= 2:
                series.append((float(parts[0]), float(parts[1])))
    if len(series) < 5:
        raise CannotJudge(f"only {len(series)} MSD point(s); need a longer trajectory")
    return series


def loglog_slope(series: Sequence[Tuple[float, float]],
                 lo_ps: float, hi_ps: float) -> Tuple[float, int]:
    """Slope of log(MSD) against log(lag) over [lo, hi], and the point count.

    Fitted in log space on purpose. A linear fit to the raw curve would be
    dominated by the longest lags, which are exactly the noisiest points
    because the fewest time origins contribute to them.
    """
    pts = [(math.log(t), math.log(m)) for t, m in series
           if lo_ps <= t <= hi_ps and t > 0 and m > 0]
    if len(pts) < 3:
        raise CannotJudge(f"only {len(pts)} usable MSD point(s) between "
                          f"{lo_ps} and {hi_ps} ps")
    n = len(pts)
    xbar = sum(x for x, _ in pts) / n
    ybar = sum(y for _, y in pts) / n
    sxx = sum((x - xbar) ** 2 for x, _ in pts)
    if sxx == 0:
        raise CannotJudge("all MSD lags identical; cannot fit a slope")
    slope = sum((x - xbar) * (y - ybar) for x, y in pts) / sxx
    return slope, n


# ------------------------------------------------------------ lattice order
def dump_frame(tpr: str, xtc: str, selection: str, when: str, gmx: str,
               workdir: str, tag: str) -> Tuple[List[Tuple[float, float, float]],
                                                Tuple[float, float, float]]:
    """Coordinates of `selection` in one frame, plus that frame's box."""
    # trjconv takes an index file, not a selection string, so the selection is
    # resolved once by `gmx select` and handed over as a one-group index.
    ndx = os.path.join(workdir, f"sel_{tag}.ndx")
    sel = _run([gmx, "select", "-s", tpr, "-select", selection, "-on", ndx])
    if sel.returncode != 0 or not os.path.exists(ndx):
        tail = (sel.stderr or sel.stdout or "").strip().splitlines()[-6:]
        raise CannotJudge(f"gmx select could not resolve {selection!r}: "
                          + " | ".join(tail))
    out = os.path.join(workdir, f"frame_{tag}.gro")
    cmd = [gmx, "trjconv", "-s", tpr, "-f", xtc, "-n", ndx, "-o", out]
    cmd += ["-b", when, "-e", when] if when != "last" else ["-dump", "999999999"]
    res = _run(cmd, input="0\n")
    if res.returncode != 0 or not os.path.exists(out):
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-6:]
        raise CannotJudge("gmx trjconv failed: " + " | ".join(tail))
    lines = open(out).read().splitlines()
    if len(lines) < 3:
        raise CannotJudge(f"empty frame written for selection {selection!r}")
    count = int(lines[1])
    coords = []
    for line in lines[2:2 + count]:
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))
    box = tuple(float(v) for v in lines[2 + count].split()[:3])
    if not coords:
        raise CannotJudge(f"selection {selection!r} matched no atoms")
    return coords, box  # type: ignore[return-value]


def structure_factor(coords: Sequence[Tuple[float, float, float]],
                     box: Tuple[float, float, float], repeats: int) -> List[float]:
    """S(k) along each axis at the wavevector matching the seeded lattice.

    S(k) = |sum_j exp(i k . r_j)|^2 / N. Perfect ordering at that spacing gives
    N; a random arrangement averages 1. No normalisation choice is hidden here:
    both limits are stated so a number in between can be read directly.
    """
    n = len(coords)
    out = []
    for axis in range(3):
        k = 2.0 * math.pi * repeats / box[axis]
        re = sum(math.cos(k * r[axis]) for r in coords)
        im = sum(math.sin(k * r[axis]) for r in coords)
        out.append((re * re + im * im) / n)
    return out


# -------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tpr")
    ap.add_argument("xtc")
    ap.add_argument("--mobility-sel", default=None,
                    help="selection whose mobility decides the verdict, e.g. 'resname CL'")
    ap.add_argument("--order-sel", default=None,
                    help="selection marking the seeded lattice, e.g. 'resname HEX'")
    ap.add_argument("--order-repeats", type=int, default=None,
                    help="lattice repeats per axis used when the cell was built")
    ap.add_argument("--begin-ps", type=float, default=0.0,
                    help="ignore the trajectory before this time")
    ap.add_argument("--trestart-ps", type=float, default=100.0,
                    help="spacing between MSD time origins; must be >= frame spacing")
    ap.add_argument("--fit-lo-ps", type=float, default=1000.0)
    ap.add_argument("--fit-hi-ps", type=float, default=8000.0)
    ap.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    ap.add_argument("--min-displacement-nm", type=float, default=DEFAULT_MIN_DISPLACEMENT_NM)
    ap.add_argument("--max-order-retained", type=float, default=DEFAULT_MAX_ORDER_RETAINED)
    ap.add_argument("--gmx", default="gmx_mpi")
    args = ap.parse_args(argv)

    if not args.mobility_sel and not args.order_sel:
        print("check_structure: give --mobility-sel, --order-sel, or both",
              file=sys.stderr)
        return 2
    if args.order_sel and not args.order_repeats:
        print("check_structure: --order-sel needs --order-repeats (the lattice "
              "repeats per axis the cell was built with)", file=sys.stderr)
        return 2

    report: dict = {"checks": {}}
    verdicts = []
    workdir = tempfile.mkdtemp(prefix="check_structure_")
    try:
        _require(args.tpr, "run input")
        _require(args.xtc, "trajectory")

        if args.mobility_sel:
            series = mean_square_displacement(args.tpr, args.xtc, args.mobility_sel,
                                              args.gmx, args.begin_ps,
                                              args.trestart_ps, workdir)
            hi = min(args.fit_hi_ps, series[-1][0])
            alpha, npts = loglog_slope(series, args.fit_lo_ps, hi)
            last_lag, last_msd = series[-1]
            rms = math.sqrt(last_msd)
            ok = alpha >= args.alpha_min and rms >= args.min_displacement_nm
            verdicts.append(ok)
            report["checks"]["mobility"] = {
                "selection": args.mobility_sel,
                "loglog_slope_alpha": alpha,
                "alpha_fit_window_ps": [args.fit_lo_ps, hi],
                "alpha_fit_points": npts,
                "longest_lag_ps": last_lag,
                "msd_at_longest_lag_nm2": last_msd,
                "rms_displacement_nm": rms,
                "alpha_min": args.alpha_min,
                "min_displacement_nm": args.min_displacement_nm,
                "ok": ok,
            }
            print(f"mobility [{args.mobility_sel}]")
            print(f"  log-log slope over {args.fit_lo_ps:.0f}-{hi:.0f} ps : "
                  f"{alpha:.3f}   (1.0 = diffusive, needs >= {args.alpha_min})")
            print(f"  rms displacement at {last_lag:.0f} ps lag      : "
                  f"{rms:.3f} nm  (needs >= {args.min_displacement_nm} nm)")
            print(f"  -> {'ok' if ok else 'ARRESTED: neighbours are not exchanging'}")

        if args.order_sel:
            first, box0 = dump_frame(args.tpr, args.xtc, args.order_sel,
                                     f"{args.begin_ps}", args.gmx, workdir, "first")
            last, box1 = dump_frame(args.tpr, args.xtc, args.order_sel,
                                    "last", args.gmx, workdir, "last")
            s0 = structure_factor(first, box0, args.order_repeats)
            s1 = structure_factor(last, box1, args.order_repeats)
            n = len(first)
            # 1 is the random-arrangement floor, so decay is measured from it.
            retained = [max(0.0, (b - 1.0)) / max(1e-12, (a - 1.0))
                        for a, b in zip(s0, s1)]
            worst = max(retained)
            ok = worst <= args.max_order_retained
            verdicts.append(ok)
            report["checks"]["lattice_memory"] = {
                "selection": args.order_sel,
                "repeats_per_axis": args.order_repeats,
                "n_scatterers": n,
                "perfect_order_value": n,
                "random_arrangement_value": 1.0,
                "structure_factor_first": s0,
                "structure_factor_last": s1,
                "fraction_retained_per_axis": retained,
                "worst_fraction_retained": worst,
                "max_order_retained": args.max_order_retained,
                "ok": ok,
            }
            print(f"\nlattice memory [{args.order_sel}], {args.order_repeats} repeats "
                  f"per axis, {n} scatterers")
            print(f"  S(k) first frame : " + "  ".join(f"{v:8.2f}" for v in s0))
            print(f"  S(k) last frame  : " + "  ".join(f"{v:8.2f}" for v in s1))
            print(f"  fraction retained: " + "  ".join(f"{v:8.3f}" for v in retained)
                  + f"   (needs <= {args.max_order_retained})")
            print(f"  -> {'ok' if ok else 'the builder lattice is still imprinted'}")
            print(f"  for scale: perfect order would be {n}, a random arrangement 1")

        passed = bool(verdicts) and all(verdicts)
        verdict = "MOBILE" if passed else "NOT YET"
        print(f"\nverdict: {verdict}")
        if not passed:
            print("  Structural averages taken now describe the starting "
                  "configuration, not the material.")

        with open(args.xtc, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        artifact = {
            "trajectory": os.path.abspath(args.xtc),
            "xtc_sha256": digest,
            "verdict": verdict,
            "checked_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "not_assessed_here": ["thermodynamic plateau (check_convergence.py)",
                                  "chain relaxation time",
                                  "ion pair lifetime"],
            **report,
        }
        with open(args.xtc + ".structure.json", "w") as handle:
            json.dump(artifact, handle, indent=2)
        print(f"artifact: {args.xtc}.structure.json")
        return 0 if passed else 1
    except CannotJudge as exc:
        print(f"check_structure: cannot judge this trajectory: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

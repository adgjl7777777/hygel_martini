#!/usr/bin/env python3
"""Has the NPT reached a plateau? Decide from blocks, not from the clock.

The integrated report (§18C.5) forbids declaring equilibration by elapsed
time; density, volume and energy must show a block plateau. This is the
screen that decides whether production may start. It is deliberately a
*screen*: structural equilibrium (RDF/CN, ion-pair fraction, chain
relaxation, loss of the initial lattice directionality) is the analysis
toolkit's job and is listed in the PASS artifact as "not assessed here".

The independent review (2026-09-05, Major 2) fed the previous version a
series with only Temperature, a skip window past the end of the record, and
a strongly drifting series, and got PLATEAU from all three. So the rules are
now explicit and fail closed:

* Density, Volume, Potential and Temperature are **required**; a missing
  term is exit 2, not a skipped row. Pressure is shown, never gated.
* Every value must be finite and every term must share one time grid.
* After ``--skip-ps``, at least ``--blocks`` blocks of at least
  ``--min-frames`` frames each must remain; otherwise exit 2.
* Three comparisons must all pass for every gated term: the last block
  against the one before it, the last half against the third quarter, and
  **the last block against the first** -- the whole window. The first two
  alone are local: the pilot's 200 ps NPT drifted 1056.6 -> 1073.1 kg/m^3
  monotonically, 0.3% per block step and 1.6% overall, and passed both while
  obviously still densifying. A slow steady march is the drift a plateau
  screen exists to catch.
* Strictly monotonic block means are reported as a trend, because they say
  "still going somewhere" even when every step is small.
* The verdict is written to ``<edr>.plateau.json`` together with the sha256
  of the energy file, so ``run_equilibration.sh production`` can require a
  PASS that belongs to *this* energy file and no other.

Three things the 2026-09-16 cell review found wrong with the version above,
fixed here:

* **A fixed skip makes the whole-window test unsatisfiable.** Blocks are equal
  fractions of whatever is kept, so as a run gets longer block 1 swallows more
  of the initial transient and the first-to-last drift *grows* with run length.
  The PEG pilot's projected whole-window drift asymptotes near 1.6e-2 -- three
  times tolerance -- forever, however long it runs. Setting the skip to half
  the record instead passes a cell still densifying at 0.24 kg/m^3/ns. Neither
  is a judgement about the physics. ``--skip-ps auto`` therefore *finds* the
  discard point: the earliest one after which the retained window has no
  significant trend. When no such point exists the screen says so, which is a
  real answer about the system rather than an artefact of the skip.
* **Relative drift on Potential is not physically meaningful.** Its zero is
  arbitrary, so dividing by its mean makes the tolerance depend on the
  force field's offset. Potential is now judged against its own equilibrium
  fluctuation: the drift must be a small fraction of one standard deviation
  (``--tol-sigma``).
* **The tolerance was never compared with the record's own precision.** Each
  drift is now reported in units of the standard error of the block means, so
  a reader can see whether 0.5% is loose or tight for this particular record.
  It is not gated on -- demanding statistical insignificance would punish long
  runs, which is backwards -- but a drift of 0.5% that is 40 sigma is a
  different statement from one that is 0.4 sigma.

Density and Volume are one measurement counted twice (the mass is fixed), and
Temperature is held by the thermostat, so neither adds independent evidence;
both stay gated because they are free and catch a broken run, and both are
labelled in the report so nobody mistakes four rows for four tests.

Exit 0 = plateau, 1 = not yet, 2 = the input cannot support a verdict.

    check_convergence.py npt_equil.edr [--blocks 5] [--tol 0.005]
                         [--skip-ps auto|500] [--tol-sigma 0.5]
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
from typing import Dict, List, Sequence

REQUIRED = ("Density", "Volume", "Potential", "Temperature")
OPTIONAL = ("Pressure",)

#: Terms whose absolute value carries no physical meaning, so a *relative*
#: drift is meaningless too. These are judged against their own equilibrium
#: fluctuation instead: the systematic change across the window must be a small
#: fraction of one standard deviation.
SIGMA_SCALED = ("Potential",)

#: Terms that are not independent evidence. Density and Volume are the same
#: measurement (the mass is fixed); Temperature is whatever the thermostat was
#: told to hold. They stay gated -- they cost nothing and catch a broken run --
#: but the report says so, so four rows are not mistaken for four tests.
REDUNDANT = {"Volume": "same measurement as Density; the mass is fixed",
             "Temperature": "held by the thermostat, not evidence of a plateau"}


def energy_series(edr: str, terms: Sequence[str] = REQUIRED + OPTIONAL,
                  gmx: str = "gmx_mpi") -> Dict[str, List[float]]:
    """Time series per term via ``gmx energy``. ``_time`` holds each term's grid."""
    binary = shutil.which(gmx) or shutil.which("gmx")
    if binary is None:
        raise RuntimeError("gmx not on PATH (source GMXRC)")
    out: Dict[str, List[float]] = {}
    grids: Dict[str, List[float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for term in terms:
            xvg = os.path.join(tmp, f"{term}.xvg")
            proc = subprocess.run([binary, "energy", "-f", edr, "-o", xvg], input=f"{term}\n",
                                  capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.exists(xvg):
                continue
            times, values = [], []
            with open(xvg) as handle:
                for line in handle:
                    if line.startswith(("#", "@")):
                        continue
                    t, v = line.split()[:2]
                    times.append(float(t)); values.append(float(v))
            out[term] = values
            grids[term] = times
    out["_grids"] = grids  # type: ignore[assignment]
    return out


class InvalidInput(ValueError):
    """The record cannot support a verdict (exit 2)."""


def validate(series: Dict[str, object], skip_ps: float, blocks: int,
             min_frames: int) -> List[int]:
    """Check completeness/finiteness/grid and return the kept frame indices."""
    grids: Dict[str, List[float]] = series.get("_grids", {})  # type: ignore[assignment]
    missing = [t for t in REQUIRED if t not in series]
    if missing:
        raise InvalidInput(f"required energy term(s) missing from the record: {missing}")
    ref = grids.get(REQUIRED[0])
    if not ref:
        raise InvalidInput("no time grid for Density")
    for term in REQUIRED + tuple(t for t in OPTIONAL if t in series):
        if grids.get(term) != ref:
            raise InvalidInput(f"time grid of {term} differs from Density's")
        vals = series[term]  # type: ignore[index]
        if len(vals) != len(ref):
            raise InvalidInput(f"{term}: {len(vals)} values for {len(ref)} times")
        if not all(math.isfinite(v) for v in vals):  # type: ignore[union-attr]
            raise InvalidInput(f"{term} contains NaN/Inf")
    keep = [i for i, tt in enumerate(ref) if tt >= skip_ps]
    if blocks < 2:
        raise InvalidInput("--blocks must be at least 2")
    if len(keep) < blocks * min_frames:
        raise InvalidInput(f"only {len(keep)} frame(s) after skipping {skip_ps} ps; need at least "
                           f"{blocks} blocks x {min_frames} frames")
    return keep


def block_means(values: List[float], blocks: int) -> List[float]:
    n = len(values) // blocks
    return [sum(values[i * n:(i + 1) * n]) / n for i in range(blocks)]


def _rel(a: float, b: float) -> float:
    scale = abs(b) if b else 1.0
    return abs(a - b) / scale


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def _scale_for(term: str, values: Sequence[float]) -> float:
    """What a drift in this term should be measured against.

    For a term with a physical zero, its own magnitude. For one whose zero is
    a convention -- a potential energy -- its equilibrium fluctuation, because
    dividing by an arbitrary offset would make the tolerance meaningless.
    """
    if term in SIGMA_SCALED:
        return _stdev(values) or 1.0
    mean = sum(values) / len(values)
    return abs(mean) or 1.0


def _trend(times: Sequence[float], values: Sequence[float]):
    """Least-squares slope of values against time, and its standard error.

    Returns (slope per time unit, standard error of the slope). The standard
    error assumes independent samples, which energy frames are not, so it
    understates the true uncertainty. It is reported for scale, never gated on.
    """
    n = len(times)
    if n < 3:
        return 0.0, float("inf")
    tbar = sum(times) / n
    vbar = sum(values) / n
    sxx = sum((x - tbar) ** 2 for x in times)
    if sxx == 0:
        return 0.0, float("inf")
    slope = sum((x - tbar) * (y - vbar) for x, y in zip(times, values)) / sxx
    intercept = vbar - slope * tbar
    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(times, values))
    se = ((resid / max(1, n - 2)) / sxx) ** 0.5
    return slope, se


def auto_skip(series: Dict[str, object], blocks: int, min_frames: int,
              tol: float, tol_sigma: float, fractions: Sequence[float] = ()) -> dict:
    """Find the earliest discard point after which nothing is still trending.

    A fixed skip cannot do this. Too small and the retained window keeps the
    transient, so the whole-window test fails however long the run goes; too
    large and the window is short enough that a real drift hides inside it.
    Scanning for the point where the trend stops is the only version of this
    question that has an answer belonging to the system rather than to the
    setting.

    Returns the chosen skip, whether a trend-free window was actually found,
    and the residual drift rate of each gated term at that choice -- in
    physical units per nanosecond, which is what a reader can argue about.
    """
    grids: Dict[str, List[float]] = series["_grids"]  # type: ignore[assignment]
    times = grids[REQUIRED[0]]
    span = times[-1] - times[0]
    candidates = list(fractions) or [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    attempts = []
    for frac in candidates:
        skip = times[0] + frac * span
        keep = [i for i, tt in enumerate(times) if tt >= skip]
        if len(keep) < blocks * min_frames:
            continue
        worst = 0.0
        rates = {}
        for term in REQUIRED:
            vals = [series[term][i] for i in keep]      # type: ignore[index]
            ts = [times[i] for i in keep]
            slope, se = _trend(ts, vals)
            scale = _scale_for(term, vals)
            window = ts[-1] - ts[0]
            limit = tol_sigma if term in SIGMA_SCALED else tol
            # Fractional change the trend implies across the retained window.
            implied = abs(slope) * window / scale
            rates[term] = {"slope_per_ns": slope * 1000.0,
                           "slope_se_per_ns": se * 1000.0,
                           "implied_window_drift": implied,
                           "limit": limit,
                           "ok": implied <= limit}
            worst = max(worst, implied / limit if limit else float("inf"))
        attempts.append({"skip_ps": skip, "retained_frames": len(keep),
                         "retained_ps": ts[-1] - ts[0], "worst_ratio": worst,
                         "terms": rates, "trend_free": worst <= 1.0})
    if not attempts:
        raise InvalidInput("no candidate skip leaves enough frames for the block test")
    trend_free = [a for a in attempts if a["trend_free"]]
    chosen = trend_free[0] if trend_free else min(attempts, key=lambda a: a["worst_ratio"])
    return {"chosen": chosen, "found_trend_free_window": bool(trend_free),
            "attempts": attempts}


def assess(series: Dict[str, object], keep: List[int], blocks: int, tol: float,
           tol_sigma: float = 0.5) -> dict:
    """Every comparison per gated term; returns the full report dict."""
    grids: Dict[str, List[float]] = series.get("_grids", {})  # type: ignore[assignment]
    times = [grids[REQUIRED[0]][i] for i in keep] if grids.get(REQUIRED[0]) else []
    report = {"terms": {}, "gated": list(REQUIRED), "blocks": blocks, "tol": tol,
              "tol_sigma": tol_sigma, "frames": len(keep),
              "redundant_terms": dict(REDUNDANT)}
    verdicts = []
    for term in REQUIRED + tuple(x for x in OPTIONAL if x in series):
        vals = [series[term][i] for i in keep]  # type: ignore[index]
        means = block_means(vals, blocks)
        n = len(vals)
        last_half = sum(vals[n // 2:]) / (n - n // 2)
        third_q = sum(vals[n // 2:3 * n // 4]) / max(1, 3 * n // 4 - n // 2)
        std = _stdev(vals)
        sigma_scaled = term in SIGMA_SCALED
        scale = _scale_for(term, vals)
        limit = tol_sigma if sigma_scaled else tol

        def drift(a: float, b: float) -> float:
            return abs(a - b) / scale

        drift_block = drift(means[-1], means[-2])
        drift_window = drift(last_half, third_q)
        # The whole window: a monotonic march of small steps is still a march.
        drift_total = drift(means[-1], means[0])
        monotonic = (all(b > a for a, b in zip(means, means[1:]))
                     or all(b < a for a, b in zip(means, means[1:])))
        # How precise is this record? The standard error of the block means
        # says whether the tolerance is loose or tight here. Reported, not
        # gated: requiring statistical insignificance would punish long runs.
        block_se = _stdev(means) / (len(means) ** 0.5) if len(means) > 1 else 0.0
        slope, slope_se = _trend(times, vals) if times else (0.0, float("inf"))
        ok = drift_block <= limit and drift_window <= limit and drift_total <= limit
        gated = term in REQUIRED
        if gated:
            verdicts.append(ok)
        report["terms"][term] = {
            "block_means": means,
            "drift_last_block": drift_block,
            "drift_last_half_vs_third_quarter": drift_window,
            "drift_first_to_last_block": drift_total,
            "monotonic_blocks": monotonic,
            "std": std,
            "scale": scale,
            "scaled_by": "stdev" if sigma_scaled else "mean",
            "limit": limit,
            "block_mean_standard_error": block_se,
            "whole_window_drift_in_block_se": (abs(means[-1] - means[0]) / block_se
                                               if block_se else float("inf")),
            "slope_per_ns": slope * 1000.0,
            "slope_se_per_ns": slope_se * 1000.0,
            "ok": ok,
            "gated": gated,
            "redundant": REDUNDANT.get(term),
        }
    report["plateau"] = bool(verdicts) and all(verdicts)
    report["not_assessed_here"] = ["RDF/CN", "AcCh+Cl- pair fraction",
                                   "chain relaxation", "initial pcu directionality",
                                   "particle mobility (is anything diffusing?)"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edr")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.005, help="relative drift tolerance")
    parser.add_argument("--skip-ps", default="0", help="discard this initial span, "
                        "or 'auto' to find the earliest trend-free discard point")
    parser.add_argument("--tol-sigma", type=float, default=0.5,
                        help="drift limit for terms with an arbitrary zero, in units of "
                             "their own standard deviation (default 0.5)")
    parser.add_argument("--min-frames", type=int, default=10, help="frames per block, at least")
    parser.add_argument("--gmx", default="gmx_mpi")
    args = parser.parse_args(argv)

    auto = str(args.skip_ps).strip().lower() == "auto"
    scan = None
    try:
        series = energy_series(args.edr, gmx=args.gmx)
        if auto:
            scan = auto_skip(series, args.blocks, args.min_frames, args.tol, args.tol_sigma)
            skip_ps = scan["chosen"]["skip_ps"]
        else:
            skip_ps = float(args.skip_ps)
        keep = validate(series, skip_ps, args.blocks, args.min_frames)
    except (InvalidInput, RuntimeError) as exc:
        print(f"check_convergence: cannot judge this record: {exc}", file=sys.stderr)
        return 2
    except ValueError:
        print(f"check_convergence: --skip-ps must be a number or 'auto', "
              f"got {args.skip_ps!r}", file=sys.stderr)
        return 2
    report = assess(series, keep, args.blocks, args.tol, args.tol_sigma)
    if scan is not None:
        report["auto_skip"] = scan

    print(f"{'term':<12} " + " ".join(f"{'block %d' % (b + 1):>12}" for b in range(args.blocks))
          + f" {'last blk':>9} {'last half':>9} {'whole':>9}  plateau")
    for term, r in report["terms"].items():
        print(f"{term:<12} " + " ".join(f"{m:12.4f}" for m in r["block_means"])
              + f" {r['drift_last_block']:9.2e} {r['drift_last_half_vs_third_quarter']:9.2e}"
              + f" {r['drift_first_to_last_block']:9.2e}  "
              + ("ok" if r["ok"] else "NO") + ("" if r["gated"] else " (not gated)")
              + ("  [monotonic]" if r["monotonic_blocks"] else ""))
    if scan is not None:
        chosen = scan["chosen"]
        if scan["found_trend_free_window"]:
            print(f"\nauto skip: {chosen['skip_ps']:.0f} ps discarded; the remaining "
                  f"{chosen['retained_ps']:.0f} ps show no trend beyond tolerance.")
        else:
            print(f"\nauto skip: NO trend-free window exists in this record. Best available "
                  f"is {chosen['skip_ps']:.0f} ps discarded, and even then:")
        for term, r in chosen["terms"].items():
            if r["ok"] and scan["found_trend_free_window"]:
                continue
            print(f"    {term:<12} residual drift {r['slope_per_ns']:+.4g} per ns "
                  f"(implies {r['implied_window_drift']:.2e} across the window, "
                  f"limit {r['limit']:.2e})")
        if not scan["found_trend_free_window"]:
            print("  A longer run does not fix this by itself: discarding more of the record "
                  "only shortens the window the trend has to hide in.")

    # The block tests and the trend scan are two views of one question. When
    # the scan could not find any window without a trend, a block test that
    # happens to pass on the shortest candidate is not a plateau -- it is the
    # drift hiding inside a window too short to show it. The scan wins.
    if scan is not None and not scan["found_trend_free_window"] and report["plateau"]:
        report["plateau"] = False
        report["plateau_overridden_by_trend_scan"] = True
        print("  block tests passed on this short window, but no trend-free window "
              "exists in the record; verdict is NOT YET.")
    verdict = "PLATEAU" if report["plateau"] else "NOT YET"
    print(f"\nverdict: {verdict} ({report['frames']} frames after {skip_ps:.0f} ps, "
          f"{args.blocks} blocks, tol {args.tol:.1%} / {args.tol_sigma:.2g} sigma); "
          f"structural equilibrium and particle mobility not assessed here")

    with open(args.edr, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    artifact = {"edr": os.path.abspath(args.edr), "edr_sha256": digest, "verdict": verdict,
                "checked_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "params": {"blocks": args.blocks, "tol": args.tol, "skip_ps": skip_ps,
                           "skip_mode": "auto" if auto else "fixed",
                           "tol_sigma": args.tol_sigma,
                           "min_frames": args.min_frames}, **report}
    with open(args.edr + ".plateau.json", "w") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"artifact: {args.edr}.plateau.json")
    return 0 if report["plateau"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

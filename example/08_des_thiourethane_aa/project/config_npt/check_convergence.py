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
* Two comparisons must both pass for every gated term: the last block against
  the one before it, and the last half against the third quarter (the
  longer-window test this file's docstring promised and did not implement).
* The verdict is written to ``<edr>.plateau.json`` together with the sha256
  of the energy file, so ``run_equilibration.sh production`` can require a
  PASS that belongs to *this* energy file and no other.

Exit 0 = plateau, 1 = not yet, 2 = the input cannot support a verdict.

    check_convergence.py npt_equil.edr [--blocks 5] [--tol 0.005] [--skip-ps 500]
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
    keep = [i for i, t in enumerate(ref) if t >= skip_ps]
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


def assess(series: Dict[str, object], keep: List[int], blocks: int, tol: float) -> dict:
    """Both comparisons per gated term; returns the full report dict."""
    report = {"terms": {}, "gated": list(REQUIRED), "blocks": blocks, "tol": tol,
              "frames": len(keep)}
    verdicts = []
    for term in REQUIRED + tuple(t for t in OPTIONAL if t in series):
        vals = [series[term][i] for i in keep]  # type: ignore[index]
        means = block_means(vals, blocks)
        n = len(vals)
        last_half = sum(vals[n // 2:]) / (n - n // 2)
        third_q = sum(vals[n // 2:3 * n // 4]) / max(1, 3 * n // 4 - n // 2)
        drift_block = _rel(means[-1], means[-2])
        drift_window = _rel(last_half, third_q)
        std = (sum((v - sum(vals) / n) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
        ok = drift_block <= tol and drift_window <= tol
        gated = term in REQUIRED
        if gated:
            verdicts.append(ok)
        report["terms"][term] = {"block_means": means, "drift_last_block": drift_block,
                                 "drift_last_half_vs_third_quarter": drift_window,
                                 "std": std, "ok": ok, "gated": gated}
    report["plateau"] = bool(verdicts) and all(verdicts)
    report["not_assessed_here"] = ["RDF/CN", "AcCh+Cl- pair fraction",
                                   "chain relaxation", "initial pcu directionality"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edr")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.005, help="relative drift tolerance")
    parser.add_argument("--skip-ps", type=float, default=0.0, help="discard this initial span")
    parser.add_argument("--min-frames", type=int, default=10, help="frames per block, at least")
    parser.add_argument("--gmx", default="gmx_mpi")
    args = parser.parse_args(argv)

    try:
        series = energy_series(args.edr, gmx=args.gmx)
        keep = validate(series, args.skip_ps, args.blocks, args.min_frames)
    except (InvalidInput, RuntimeError) as exc:
        print(f"check_convergence: cannot judge this record: {exc}", file=sys.stderr)
        return 2
    report = assess(series, keep, args.blocks, args.tol)

    print(f"{'term':<12} " + " ".join(f"{'block %d' % (b + 1):>12}" for b in range(args.blocks))
          + f" {'last blk':>9} {'last half':>9}  plateau")
    for term, r in report["terms"].items():
        print(f"{term:<12} " + " ".join(f"{m:12.4f}" for m in r["block_means"])
              + f" {r['drift_last_block']:9.2e} {r['drift_last_half_vs_third_quarter']:9.2e}  "
              + ("ok" if r["ok"] else "NO") + ("" if r["gated"] else " (not gated)"))
    verdict = "PLATEAU" if report["plateau"] else "NOT YET"
    print(f"\nverdict: {verdict} ({report['frames']} frames after {args.skip_ps} ps, "
          f"{args.blocks} blocks, tol {args.tol:.1%}); structural equilibrium not assessed here")

    with open(args.edr, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    artifact = {"edr": os.path.abspath(args.edr), "edr_sha256": digest, "verdict": verdict,
                "checked_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "params": {"blocks": args.blocks, "tol": args.tol, "skip_ps": args.skip_ps,
                           "min_frames": args.min_frames}, **report}
    with open(args.edr + ".plateau.json", "w") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"artifact: {args.edr}.plateau.json")
    return 0 if report["plateau"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

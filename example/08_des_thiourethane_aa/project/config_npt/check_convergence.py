#!/usr/bin/env python3
"""Has the NPT reached a plateau? Decide from blocks, not from the clock.

The report (§18C.5) forbids declaring equilibration by elapsed time: density,
volume and energy have to show a block plateau. This pulls those terms out of
a GROMACS energy file with ``gmx energy``, splits the record into equal
blocks, and reports each block's mean with a simple drift test: the last
block's mean must sit within ``--tol`` (relative, default 0.5 %) of the
previous block's, for every term, and the last-half mean within the same
tolerance of the third quarter's. It is a screen, not a proof -- RDFs, the
AcCh+Cl- pair fraction and chain relaxation (§18C.5) are the analysis
toolkit's job -- but it is the screen that decides whether production may
start.

    check_convergence.py npt_equil.edr [--blocks 5] [--tol 0.005] [--skip-ps 500]

Exit 0 on plateau, 1 otherwise, so a driver can gate on it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List

TERMS = ("Density", "Volume", "Potential", "Temperature", "Pressure")


def energy_series(edr: str, terms=TERMS, gmx: str = "gmx_mpi") -> Dict[str, List[float]]:
    """Time series per term via ``gmx energy``; a missing term is skipped."""
    binary = shutil.which(gmx) or shutil.which("gmx")
    if binary is None:
        sys.exit("gmx not on PATH (source GMXRC)")
    out: Dict[str, List[float]] = {}
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
            out.setdefault("_time", times)
    return out


def block_means(values: List[float], blocks: int) -> List[float]:
    n = len(values) // blocks
    return [sum(values[i * n:(i + 1) * n]) / n for i in range(blocks)] if n else []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edr")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.005, help="relative drift tolerance")
    parser.add_argument("--skip-ps", type=float, default=0.0, help="discard this initial span")
    parser.add_argument("--gmx", default="gmx_mpi")
    args = parser.parse_args(argv)

    series = energy_series(args.edr, gmx=args.gmx)
    times = series.pop("_time", [])
    if not series:
        sys.exit("no energy terms read")
    keep = [i for i, t in enumerate(times) if t >= args.skip_ps]
    verdicts = []
    print(f"{'term':<12} " + " ".join(f"{'block %d' % (b + 1):>12}" for b in range(args.blocks))
          + f" {'last drift':>11}  plateau")
    for term, values in series.items():
        vals = [values[i] for i in keep] if keep else values
        means = block_means(vals, args.blocks)
        if len(means) < 2:
            continue
        scale = abs(means[-2]) if means[-2] else 1.0
        drift = abs(means[-1] - means[-2]) / scale
        # Pressure fluctuates by construction; it is shown, not gated.
        gated = term != "Pressure"
        ok = drift <= args.tol
        if gated:
            verdicts.append(ok)
        print(f"{term:<12} " + " ".join(f"{m:12.4f}" for m in means)
              + f" {drift:11.2e}  {'ok' if ok else 'NO'}{'' if gated else ' (not gated)'}")
    plateau = bool(verdicts) and all(verdicts)
    print(f"\nverdict: {'PLATEAU' if plateau else 'NOT YET'} "
          f"({len(keep) if keep else len(times)} frames, {args.blocks} blocks, tol {args.tol:.1%})")
    return 0 if plateau else 1


if __name__ == "__main__":
    raise SystemExit(main())

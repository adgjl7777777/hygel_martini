#!/usr/bin/env python3
"""Reconstruct what a shrink's NVT recoveries ran with, from their own files.

``history.jsonl`` events written before commit ba277fa carry no
``nvt_recovery_settings``; the completed count:32 shrink is one of them. The
history is immutable -- rewriting it to look like the new schema would be
exactly the kind of after-the-fact tidying an audit exists to catch -- so
this reads each recovery attempt's own ``mdout.mdp`` (the parameters grompp
actually used, including the drawn ``gen-seed``) and ``nvt.log`` (whether
mdrun finished) and prints a table beside the history, labelled as
reconstructed provenance.

    recovery_audit.py <shrink_workdir> [--tsv out.tsv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Dict, List

KEYS = ("dt", "nsteps", "ref-t", "gen-temp", "gen-seed", "constraints", "tcoupl")


def _mdout(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in open(path):
        text = line.split(";", 1)[0]
        if "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip().lower().replace("_", "-")
        if key in KEYS:
            out[key] = value.strip()
    return out


def audit(workdir: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for step_dir in sorted(os.listdir(workdir)):
        rec = os.path.join(workdir, step_dir, "nvt_recovery")
        if not os.path.isdir(rec):
            continue
        m = re.match(r"step_(\d+)_scale_([\d.]+)", step_dir)
        mdout = os.path.join(rec, "mdout.mdp")
        log = os.path.join(rec, "nvt.log")
        row: Dict[str, object] = {
            "step": int(m.group(1)) if m else None,
            "scale": float(m.group(2)) if m else None,
            "mdout_sha256": hashlib.sha256(open(mdout, "rb").read()).hexdigest()[:16]
            if os.path.exists(mdout) else None,
            "finished_mdrun": os.path.exists(log) and "Finished mdrun" in open(log, errors="replace").read(),
        }
        row.update(_mdout(mdout) if os.path.exists(mdout) else {})
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir")
    parser.add_argument("--tsv", help="also write the table here")
    args = parser.parse_args(argv)
    rows = audit(args.workdir)
    if not rows:
        print("no recovery attempts found")
        return 0
    history = os.path.join(args.workdir, "history.jsonl")
    events = [json.loads(l) for l in open(history)] if os.path.exists(history) else []
    used = [e for e in events if e.get("nvt_recovery_used")]
    cols = ["step", "scale", "finished_mdrun"] + list(KEYS) + ["mdout_sha256"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    print(f"# reconstructed recovery provenance for {args.workdir}")
    print(f"# history.jsonl: {len(events)} attempts, {len(used)} used the recovery, "
          f"{sum(1 for e in used if e.get('accepted'))} of those accepted; "
          f"{sum(1 for e in events if 'nvt_recovery_settings' in e)} event(s) carry settings inline")
    print("\n".join(lines))
    if args.tsv:
        with open(args.tsv, "w") as handle:
            handle.write("# reconstructed from step_*/nvt_recovery/mdout.mdp and nvt.log; history.jsonl untouched\n")
            handle.write("\n".join(lines) + "\n")
        print(f"wrote {args.tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

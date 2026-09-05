#!/usr/bin/env python3
"""Freeze the force-field files a build used, and notice when they drift.

Parameters here will change in known ways -- LigParGen charges give way to
RESP candidates, the thiourethane torsion gets refit from the new F3/F3x
minimum, the carbonyl improper may be retuned -- and every simulation must be
able to say *which* version it ran. The integrated report asks for exactly
this (§18B.5, §18C.4): a release with provenance, checksums and a status of
``draft``, ``candidate`` or ``production-approved``.

A release is a directory under ``releases/`` holding ``manifest.json``: one
entry per force-field file with its sha256, plus status, author, date and
notes. The files themselves stay where the makers read them (``project/
structure/`` and ``project/config/hydrogel.yaml``); the manifest is the
record of what they were.

::

    release.py freeze v0_ligpargen_draft --status draft --note "LigParGen 1.14*CM1A-LBCC, as built"
    release.py verify v0_ligpargen_draft        # exit 1 if any tracked file changed
    release.py current                          # which release the tree matches, if any

``verify`` is what a run script calls before building: a topology built from
files that match no frozen release has no provenance, and that is the case
worth refusing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import hashlib
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir, "project"))
RELEASES = os.path.join(HERE, "releases")

#: Files whose content defines the force field a build sees. Relative to
#: PROJECT. Coordinates are not parameters and are not tracked here.
TRACKED = (
    "structure/forcefield.itp",
    "structure/HEXU.itp",
    "structure/STR.itp",
    "structure/STR_n33.itp",
    "structure/ACC.itp",
    "structure/CL.itp",
    "structure/hydrogel_stubs_snippet.yaml",
    "config/hydrogel.yaml",
)

STATUSES = ("draft", "candidate", "production-approved")


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def snapshot(project: str = PROJECT) -> Dict[str, str]:
    """``{relative path: sha256}`` for every tracked file that exists."""
    out: Dict[str, str] = {}
    for rel in TRACKED:
        path = os.path.join(project, rel)
        if os.path.exists(path):
            out[rel] = _sha256(path)
    return out


def freeze(name: str, status: str, notes: List[str], project: str = PROJECT,
           releases: str = RELEASES, force: bool = False) -> str:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    target = os.path.join(releases, name)
    manifest_path = os.path.join(target, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        raise FileExistsError(f"{manifest_path} exists; a release is immutable -- "
                              "freeze a new name, or --force to redo this one")
    os.makedirs(target, exist_ok=True)
    manifest = {
        "release": name,
        "status": status,
        "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "frozen_by": getpass.getuser(),
        "notes": notes,
        "files": snapshot(project),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def verify(name: str, project: str = PROJECT, releases: str = RELEASES) -> Dict[str, str]:
    """``{file: 'ok'|'changed'|'missing'|'untracked'}`` against the release."""
    with open(os.path.join(releases, name, "manifest.json")) as handle:
        manifest = json.load(handle)
    now = snapshot(project)
    result: Dict[str, str] = {}
    for rel, digest in manifest["files"].items():
        if rel not in now:
            result[rel] = "missing"
        elif now[rel] != digest:
            result[rel] = "changed"
        else:
            result[rel] = "ok"
    for rel in now:
        if rel not in manifest["files"]:
            result[rel] = "untracked"
    return result


def current(project: str = PROJECT, releases: str = RELEASES) -> List[str]:
    """Names of releases the working tree matches exactly."""
    if not os.path.isdir(releases):
        return []
    matches = []
    for name in sorted(os.listdir(releases)):
        path = os.path.join(releases, name, "manifest.json")
        if os.path.exists(path):
            if all(v == "ok" for v in verify(name, project, releases).values()):
                matches.append(name)
    return matches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("name")
    f.add_argument("--status", default="draft", choices=STATUSES)
    f.add_argument("--note", action="append", default=[])
    f.add_argument("--force", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("name")
    sub.add_parser("current")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        path = freeze(args.name, args.status, args.note, force=args.force)
        print(f"froze {args.name} ({args.status}) -> {path}")
        return 0
    if args.command == "verify":
        result = verify(args.name)
        width = max(len(k) for k in result)
        for rel, state in result.items():
            print(f"  {rel:<{width}}  {state}")
        ok = all(v == "ok" for v in result.values())
        print(f"{args.name}: {'matches' if ok else 'DRIFTED'}")
        return 0 if ok else 1
    names = current()
    print("working tree matches: " + (", ".join(names) if names else "no frozen release"))
    return 0 if names else 1


if __name__ == "__main__":
    raise SystemExit(main())

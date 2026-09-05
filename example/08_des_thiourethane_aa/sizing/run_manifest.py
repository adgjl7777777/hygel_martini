#!/usr/bin/env python3
"""Freeze everything one run reads, and notice if any of it changed.

``release.py`` freezes the base force field. It deliberately does not cover
the rest of what a build consumes -- the maker, the config YAMLs, the
charge-scaled ion variants, the recipe, the mdps, the seeds, the git commit,
the GROMACS binary -- and the independent review (2026-09-05, Major 3) showed
what that gap allows: a scaled ion ITP could be altered by +0.1 e and the
base release still verified. This file is the closure the review asked for.

``write --tag T`` hashes, for the cell ``maker_size_T.yaml`` describes:

* the build and shrink makers and every ``config/*.yaml`` they include;
* **every** ``*.itp`` under ``project/structure`` -- because the builder
  auto-includes all of them, that directory *is* the topology's include
  closure -- plus the ``.gro`` files the maker references;
* the recipe, if one is given, and every mdp under ``config_npt``;
* the git HEAD (and whether the tree is dirty), the force-field release the
  tree matches, the GROMACS and Packmol binaries and their versions, the
  Python version, and the conversion / placement seeds read from the makers.

``verify --tag T`` recomputes and reports ``ok`` / ``changed`` / ``missing``
per file; exit 1 on any drift. ``run_size.sh`` writes the manifest after
emitting and verifies it again before the shrink, so a stale or edited input
cannot carry a build's tag into the next stage.

The manifest lives in ``sizing/manifests/run_manifest_<tag>.json`` (so it
exists before the builder creates the output directory) and is copied into
the output directory once the build has one.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, os.pardir))
PROJECT = os.path.join(EXAMPLE, "project")
MANIFESTS = os.path.join(HERE, "manifests")


def _sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=EXAMPLE, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(binary: str, flag: str) -> str:
    path = shutil.which(binary)
    if not path:
        return "not on PATH"
    try:
        out = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
        text = (out.stdout or out.stderr).strip().splitlines()
        return f"{path} :: {text[0] if text else '?'}"
    except (OSError, subprocess.SubprocessError):
        return f"{path} :: version query failed"


def _release_current() -> List[str]:
    sys.path.insert(0, os.path.join(EXAMPLE, "parameterization"))
    try:
        import release  # type: ignore
        return release.current()
    except Exception:  # pragma: no cover - release tooling missing
        return []


def _maker_paths(maker: str) -> List[str]:
    """Files a maker references: its includes and every ${CONFIG_DIR} path."""
    with open(maker) as handle:
        text = handle.read()
    found: List[str] = []
    for inc in (yaml.safe_load(text) or {}).get("includes", []) or []:
        found.append(os.path.normpath(os.path.join(os.path.dirname(maker), inc)))
    for match in re.finditer(r"\$\{CONFIG_DIR\}/([^\s\"']+)", text):
        found.append(os.path.join(PROJECT, match.group(1)))
    return found


def _seeds(maker: str) -> Dict[str, object]:
    cfg = yaml.safe_load(open(maker)) or {}
    sim = cfg.get("simulation_parameters") or {}
    conv = (sim.get("network_layout") or {}).get("conversion") or {}
    return {"conversion_seed": conv.get("seed"), "conversion": conv or None}


def collect(tag: str, recipe: Optional[str] = None) -> dict:
    build = os.path.join(PROJECT, f"maker_size_{tag}.yaml")
    shrink = os.path.join(PROJECT, f"maker_size_{tag}_shrink.yaml")
    if not os.path.exists(build):
        raise FileNotFoundError(f"no build maker for tag {tag!r}: {build}")

    files: List[str] = [build, shrink]
    for maker in (build, shrink):
        if os.path.exists(maker):
            files += _maker_paths(maker)
    # Included config files reference more ${CONFIG_DIR} paths themselves.
    for cfg in list(files):
        if cfg.endswith(".yaml") and os.path.exists(cfg) and cfg not in (build, shrink):
            files += [p for p in _maker_paths(cfg)]
    structure = os.path.join(PROJECT, "structure")
    files += [os.path.join(structure, f) for f in sorted(os.listdir(structure))
              if f.endswith(".itp")]
    npt = os.path.join(PROJECT, "config_npt")
    if os.path.isdir(npt):
        files += [os.path.join(npt, f) for f in sorted(os.listdir(npt)) if f.endswith(".mdp")]
    if recipe:
        files.append(os.path.abspath(recipe))

    uniq: Dict[str, Optional[str]] = {}
    for path in files:
        path = os.path.normpath(path)
        if path not in uniq and os.path.isfile(path):
            uniq[path] = _sha256(path)

    sim_cfg = yaml.safe_load(open(os.path.join(PROJECT, "config", "simulation.yaml"))) or {}
    placement_seed = (sim_cfg.get("simulation_parameters") or {}).get("random_seed")
    return {
        "tag": tag,
        "written_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "release": _release_current(),
        "gromacs": _version("gmx_mpi", "--version"),
        "packmol": shutil.which("packmol") or "not on PATH",
        "python": sys.version.split()[0],
        "seeds": {**_seeds(build), "placement_random_seed": placement_seed},
        "recipe": os.path.abspath(recipe) if recipe else None,
        "files": {os.path.relpath(k, EXAMPLE): v for k, v in uniq.items()},
    }


def manifest_path(tag: str) -> str:
    return os.path.join(MANIFESTS, f"run_manifest_{tag}.json")


def write(tag: str, recipe: Optional[str] = None) -> str:
    os.makedirs(MANIFESTS, exist_ok=True)
    data = collect(tag, recipe)
    path = manifest_path(tag)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return path


def verify(tag: str) -> Dict[str, str]:
    with open(manifest_path(tag)) as handle:
        data = json.load(handle)
    result: Dict[str, str] = {}
    for rel, digest in data["files"].items():
        path = os.path.join(EXAMPLE, rel)
        now = _sha256(path)
        result[rel] = "missing" if now is None else ("ok" if now == digest else "changed")
    head = _git("rev-parse", "HEAD")
    if head and data.get("git_head") and head != data["git_head"]:
        result["<git HEAD>"] = "changed"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write"); w.add_argument("--tag", required=True); w.add_argument("--recipe")
    v = sub.add_parser("verify"); v.add_argument("--tag", required=True)
    v.add_argument("--ignore-shrink-maker", action="store_true",
                   help="the shrink maker is retargeted after the build; skip it")
    c = sub.add_parser("copy"); c.add_argument("--tag", required=True); c.add_argument("--into", required=True)
    args = parser.parse_args(argv)

    if args.command == "write":
        path = write(args.tag, args.recipe)
        data = json.load(open(path))
        print(f"run manifest: {path}")
        print(f"  {len(data['files'])} files, HEAD {data['git_head'][:10]}"
              f"{' (DIRTY)' if data['git_dirty'] else ''}, release {data['release'] or 'NONE'}")
        if not data["release"]:
            print("  WARNING: the force-field files match no frozen release", file=sys.stderr)
            return 1
        if data["git_dirty"]:
            print("  WARNING: working tree has uncommitted changes", file=sys.stderr)
        return 0
    if args.command == "verify":
        result = verify(args.tag)
        if args.ignore_shrink_maker:
            result = {k: v for k, v in result.items() if not k.endswith("_shrink.yaml")}
        bad = {k: v for k, v in result.items() if v != "ok"}
        for k, v in bad.items():
            print(f"  {v:8s} {k}")
        print(f"run manifest {args.tag}: {'matches' if not bad else 'DRIFTED'} "
              f"({len(result)} entries)")
        return 0 if not bad else 1
    if args.command == "copy":
        os.makedirs(args.into, exist_ok=True)
        dst = os.path.join(args.into, "run_manifest.json")
        shutil.copy2(manifest_path(args.tag), dst)
        print(f"copied manifest to {dst}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

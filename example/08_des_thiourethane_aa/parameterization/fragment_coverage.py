#!/usr/bin/env python3
"""Which template atoms can inherit a charge from which RESP fragment?

Applying fragment charges to the whole-strand templates needs a decision that
the 2026-09-15 DFT-to-MD audit already flagged as the prerequisite: where does
each fragment's authority end, and what happens at the joins? This script does
not make that decision. It makes it concrete, by answering a narrower and
purely mechanical question:

    for every atom in STR / HEXU / ACC / PEG, is there an atom in some RESP
    fragment sitting in the same local chemical environment?

Matching is by environment fingerprint, not by subgraph isomorphism. An atom's
fingerprint is its element plus the elements of its neighbours out to a given
radius, shelled and sorted. That is the right notion for transferring charges:
two atoms should share a charge when they sit in the same chemical
surroundings, and a fingerprint at radius 2-3 captures exactly the range over
which a point charge is meaningfully determined. It also degrades honestly --
raise the radius and the matches get stricter, so the radius at which an atom
stops matching tells you how far its environment really agrees.

The output is three piles per template:

    unique     one fragment environment matches -- charge transfer is mechanical
    ambiguous  several fragments claim it with different charges -- a decision
    orphan     no fragment covers it at all -- needs new QM or interpolation

An orphan pile that is small and confined to cap regions means the fragment set
is adequate and only the joins need a rule. An orphan pile that reaches into
the chemistry the project is measuring means the fragment set is not adequate,
and that is worth knowing before anyone fits anything.

Read the radius column carefully rather than the orphan count alone. What broke
this analysis's first reading was exactly that: the network's thiourethane
nitrogen diverges from its fragment at radius 3, which looked like a missing
aromatic ring. Ring detection says otherwise -- every relevant fragment has one.
The real difference is the ring's substitution: the fragments carry one nitrogen
on a tolyl ring, while the polymer's 2,4-TDI ring carries a urethane nitrogen
and a thiourethane nitrogen at once. A fingerprint tells you *that* two
environments differ and at what distance; it does not tell you *what* differs.
Look at the structure before concluding.

Usage
-----
    PYTHONPATH=<package> python3 fragment_coverage.py [--radius 3]
                                                      [--templates STR HEXU]
                                                      [--json coverage.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE / "validation"))
import ff_pair_benchmark as ffb  # noqa: E402

DFT_ROOT = Path("/nas_3/active/soohki/27.des/dft")
STRUCTURE = EXAMPLE / "project" / "structure"
REFIT = HERE / "raw" / "resp_boltzmann"

#: Per-atom listings are capped so a 17k-atom network stays a readable report.
LISTING_CAP = 400

FRAGMENTS = ["F1_thiol", "F2_urethane_p", "F3_thiouret_p", "F3x_thiouret_ext",
             "F4_ppg1", "F4x_ppg2", "F5_acch"]

#: Mass windows used to name elements from an ITP, which carries masses but not
#: element symbols. Chlorine and sulfur are close enough to each other that the
#: windows are kept tight deliberately.
MASS_TO_ELEMENT = [
    (1.0, 1.2, "H"), (10.8, 11.0, "B"), (12.0, 12.1, "C"), (14.0, 14.1, "N"),
    (15.9, 16.1, "O"), (18.9, 19.1, "F"), (22.9, 23.1, "Na"), (24.2, 24.4, "Mg"),
    (30.9, 31.1, "P"), (32.0, 32.1, "S"), (35.4, 35.5, "Cl"), (39.0, 39.2, "K"),
    (40.0, 40.1, "Ca"), (79.8, 80.0, "Br"), (126.8, 127.0, "I"),
]


def element_from_mass(mass: float) -> str:
    for lo, hi, symbol in MASS_TO_ELEMENT:
        if lo <= mass <= hi:
            return symbol
    return f"?{mass:.2f}"


def read_template(itp: Path):
    """Elements, names, charges and bond adjacency from a GROMACS ITP."""
    section = None
    elements, names, charges = [], [], []
    bonds = []
    for raw in itp.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        parts = line.split()
        if section == "atoms" and len(parts) >= 8 and parts[0].isdigit():
            elements.append(element_from_mass(float(parts[7])))
            names.append(parts[4])
            charges.append(float(parts[6]))
        elif section == "bonds" and len(parts) >= 2 and parts[0].isdigit():
            bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
    adj = defaultdict(set)
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)
    return elements, names, np.array(charges), adj


def read_fragment(name: str, charge_source: Path | None):
    """Elements and adjacency from the fragment's QM geometry, plus its charges."""
    path = DFT_ROOT / "05_resp" / name / "start.xyz"
    elements, xyz = ffb.read_xyz(str(path))
    pairs = ffb.bonds_from_geometry(elements, xyz)
    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    charges = None
    if charge_source is not None and charge_source.exists():
        charges = np.array([float(v) for v in charge_source.read_text().split()])
        if len(charges) != len(elements):
            charges = charges[:len(elements)]
    return list(elements), adj, charges


def fingerprint(index: int, elements, adj, radius: int) -> str:
    """Element plus shelled neighbour elements out to `radius` bonds.

    Shells are sorted within themselves, so the fingerprint is independent of
    atom ordering, and shells are kept separate, so a neighbour two bonds away
    is never confused with one bond away.
    """
    seen = {index}
    frontier = {index}
    shells = [elements[index]]
    for _ in range(radius):
        nxt = set()
        for atom in frontier:
            nxt |= adj[atom] - seen
        if not nxt:
            shells.append("")
            frontier = set()
            continue
        shells.append("".join(sorted(elements[a] for a in nxt)))
        seen |= nxt
        frontier = nxt
    return "|".join(shells)


def build_fragment_index(radius: int, use_refit: bool):
    """environment fingerprint -> list of (fragment, atom index, charge)."""
    index = defaultdict(list)
    loaded = {}
    for name in FRAGMENTS:
        src = (REFIT / f"{name}.qout") if use_refit else (DFT_ROOT / "05_resp" / name / "qout_stage2")
        try:
            elements, adj, charges = read_fragment(name, src)
        except FileNotFoundError:
            print(f"  (skipping {name}: geometry not reachable)", file=sys.stderr)
            continue
        loaded[name] = len(elements)
        for i in range(len(elements)):
            q = None if charges is None else float(charges[i])
            index[fingerprint(i, elements, adj, radius)].append((name, i, q))
    return index, loaded


def resolve_template(template: str) -> Path:
    """A bare name is a component ITP; anything with a separator is a path.

    The assembled network matters more than the standalone components: the
    thiourethane and urethane linkages only exist once the builder has joined a
    strand to a linker, so the junction atoms in STR/HEXU are capped stubs there
    and carry their final chemistry only in the built hydrogel ITP.
    """
    if "/" in template or template.endswith(".itp"):
        return Path(template).expanduser().resolve()
    return STRUCTURE / f"{template}.itp"


def classify(template: str, radius: int, frag_index) -> dict:
    itp = resolve_template(template)
    if not itp.exists():
        return {"error": f"no ITP at {itp}"}
    elements, names, charges, adj = read_template(itp)

    unique, ambiguous, orphan = [], [], []
    for i in range(len(elements)):
        fp = fingerprint(i, elements, adj, radius)
        hits = frag_index.get(fp, [])
        if not hits:
            orphan.append(i)
            continue
        qs = [q for _, _, q in hits if q is not None]
        spread = (max(qs) - min(qs)) if len(qs) > 1 else 0.0
        entry = {
            "atom": i + 1,
            "name": names[i],
            "element": elements[i],
            "current_charge": float(charges[i]),
            "sources": sorted({f for f, _, _ in hits}),
            "candidate_charges": qs,
            "charge_spread": float(spread),
        }
        # One source, or several that agree closely, is a mechanical transfer.
        if len(entry["sources"]) == 1 or spread < 0.02:
            unique.append(entry)
        else:
            ambiguous.append(entry)

    by_element = defaultdict(int)
    for i in orphan:
        by_element[elements[i]] += 1

    return {
        "template": Path(template).name.replace(".itp", ""),
        "n_atoms": len(elements),
        "radius": radius,
        "n_unique": len(unique),
        "n_ambiguous": len(ambiguous),
        "n_orphan": len(orphan),
        "coverage_fraction": (len(unique) + len(ambiguous)) / len(elements),
        "orphan_by_element": dict(by_element),
        "orphan_atoms": [{"atom": i + 1, "name": names[i], "element": elements[i],
                          "current_charge": float(charges[i]),
                          "neighbours": sorted(elements[j] for j in adj[i])}
                         for i in orphan[:LISTING_CAP]],
        "orphan_listing_truncated": len(orphan) > LISTING_CAP,
        "orphan_name_counts": dict(sorted(
            Counter(names[i] for i in orphan).items(), key=lambda kv: -kv[1])),
        "ambiguous_atoms": ambiguous[:LISTING_CAP],
        "unique_atoms": unique[:LISTING_CAP],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, nargs="*", default=[2, 3],
                    help="environment radii to report (default 2 3)")
    ap.add_argument("--templates", nargs="*", default=["STR", "HEXU", "ACC", "PEG"])
    ap.add_argument("--json", default="raw/resp_boltzmann/fragment_coverage.json")
    ap.add_argument("--shipped-charges", action="store_true",
                    help="use the collaborators' qout_stage2 instead of the refit")
    args = ap.parse_args(argv)

    if not DFT_ROOT.exists():
        print(f"DFT data not reachable at {DFT_ROOT}", file=sys.stderr)
        return 2

    report = {"radii": {}, "charge_source": "shipped" if args.shipped_charges else "boltzmann_refit"}
    for radius in args.radius:
        frag_index, loaded = build_fragment_index(radius, use_refit=not args.shipped_charges)
        print(f"\n=== environment radius {radius} bonds "
              f"({len(frag_index)} distinct environments across "
              f"{sum(loaded.values())} fragment atoms)")
        print(f"{'template':<8} {'atoms':>6} {'unique':>7} {'ambig':>6} {'orphan':>7} "
              f"{'covered':>8}   orphan composition")
        per_radius = {}
        for template in args.templates:
            result = classify(template, radius, frag_index)
            if "error" in result:
                print(f"{template:<8} {result['error']}")
                continue
            per_radius[template] = result
            comp = ", ".join(f"{v}{k}" for k, v in sorted(result["orphan_by_element"].items()))
            print(f"{template:<8} {result['n_atoms']:6d} {result['n_unique']:7d} "
                  f"{result['n_ambiguous']:6d} {result['n_orphan']:7d} "
                  f"{result['coverage_fraction']*100:7.1f}%   {comp or '-'}")
        report["radii"][str(radius)] = per_radius

    out = (HERE / args.json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")

    # the decision this analysis exists to frame
    strict = report["radii"].get(str(max(args.radius)), {})
    for template in ("STR", "HEXU"):
        r = strict.get(template)
        if not r:
            continue
        if r["n_orphan"] == 0:
            print(f"{template}: fully covered at radius {r['radius']} -- transfer is mechanical.")
        else:
            print(f"{template}: {r['n_orphan']} of {r['n_atoms']} atoms have no fragment "
                  f"counterpart at radius {r['radius']}. Those are the atoms a boundary "
                  f"rule has to account for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

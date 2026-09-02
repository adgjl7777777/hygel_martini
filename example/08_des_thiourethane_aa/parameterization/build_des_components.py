#!/usr/bin/env python3
"""Turn the DES ions into builder-ready templates: AcCh(+) and Cl(-).

The deep eutectic solvent is acetylcholine chloride at six pairs per Hexakis
junction. Two things make it awkward, and both are handled here rather than
worked around:

* **LigParGen refuses charged species under LBCC.** The 1.14*CM1A-LBCC charge
  model is defined for neutral molecules, so the server rejects an ion with
  ``chargetype=cm1abcc`` and accepts it with plain ``cm1a``. The acetylcholine
  cation in ``raw/ACCH.itp`` is that submission (26 atoms, net +0.9998 e).
* **A monatomic ion has no SMILES geometry to optimize**, so LigParGen cannot
  produce chloride at all. Its parameters are read instead from the OPLS-AA
  force field GROMACS ships -- ``oplsaa.ff/ffnonbonded.itp``, type
  ``opls_401``, sigma 0.441724 nm, epsilon 0.492833 kJ/mol -- quoted here with
  that provenance rather than typed from memory.

Type names are renamed (``ac*`` for the cation, ``desCL`` for the anion) for
the same reason the polymer templates are: LigParGen mints a fresh
``opls_8xx`` namespace per submission, so two submissions collide.

Charges: the cation's LigParGen charges sum to +0.9998 e, not +1. The residual
is spread over its 26 atoms so a pair is exactly neutral -- 384 pairs would
otherwise put -0.077 e on the system. Both molecules' charges are the
provisional part of this file: the collaboration's DFT project has RESP
charges for acetylcholine, and replacing the charge column is the intended
upgrade path.

Order matters: run ``build_templates.py`` first. It owns ``forcefield.itp``
(the single home for ``[ atomtypes ]``, which GROMACS requires before any
``[ moleculetype ]``), and this script appends the DES types into it inside a
marked fence. Re-running ``build_templates.py`` therefore means re-running
this one.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "..", "project", "structure")

FENCE_START = "; --- DES component atomtypes (build_des_components.py) ---"
FENCE_END = "; --- end DES component atomtypes ---"

#: Chloride from the OPLS-AA force field shipped with GROMACS
#: (share/gromacs/top/oplsaa.ff/ffnonbonded.itp, opls_401):
#: name, bonded type, mass (amu), charge, ptype, sigma (nm), epsilon (kJ/mol).
CHLORIDE = ("desCL", "desCL", 35.45300, -1.000, "A", 4.41724e-01, 4.92833e-01)
CHLORIDE_SOURCE = "GROMACS oplsaa.ff/ffnonbonded.itp opls_401"


def parse_itp(path):
    """Section name -> list of whitespace-split rows, comments stripped."""
    section = None
    data = defaultdict(list)
    for line in open(path):
        stripped = line.split(";", 1)[0].strip()
        if not stripped:
            continue
        match = re.match(r"\[\s*(\S+)\s*\]", stripped)
        if match:
            section = match.group(1)
            continue
        data[section].append(stripped.split())
    return data


def parse_gro(path):
    """Atom names and positions (nm) from a GRO file, fixed columns."""
    lines = open(path).read().splitlines()
    count = int(lines[1].split()[0])
    names, coords = [], []
    for line in lines[2 : 2 + count]:
        names.append(line[10:15].strip())
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))
    return names, coords


def build_cation(residue="ACC", molecule="ACC", prefix="ac"):
    """Write the acetylcholine cation template with renamed types.

    Returns the ``[ atomtypes ]`` rows it needs, so the caller can merge them
    into the shared force field file.
    """
    data = parse_itp(os.path.join(RAW, "ACCH.itp"))
    nonbonded = {row[0]: row for row in data["atomtypes"]}
    atoms = []
    for row in data["atoms"]:
        atoms.append({
            "nr": int(row[0]),
            "type": row[1],
            "name": row[4],
            "charge": float(row[6]),
            "mass": float(row[7]),
        })
    _, coords = parse_gro(os.path.join(RAW, "ACCH.gro"))
    if len(coords) != len(atoms):
        raise SystemExit(f"ACCH: {len(coords)} coordinates for {len(atoms)} atoms")

    # Exactly +1: the residual of LigParGen's rounding is spread over every
    # atom, so an AcCh-Cl pair is neutral and 384 pairs add no net charge.
    total = sum(a["charge"] for a in atoms)
    correction = (1.0 - total) / len(atoms)

    renamed = {orig: f"{prefix}{orig.split('_')[-1]}" for orig in nonbonded}
    atomtype_rows = []
    for orig, row in sorted(nonbonded.items()):
        atomtype_rows.append((
            renamed[orig],
            f"{prefix}{row[1]}",
            float(row[2]),
            float(row[3]),
            row[4],
            float(row[5]),
            float(row[6]),
        ))

    itp_path = os.path.join(OUT, f"{molecule}.itp")
    with open(itp_path, "w") as handle:
        handle.write(
            "; Acetylcholine cation, from raw/ACCH.itp (LigParGen OPLS-AA,\n"
            "; plain 1.14*CM1A -- LBCC is defined for neutral molecules only).\n"
            f"; Types renamed with the '{prefix}' prefix to keep LigParGen's\n"
            "; per-submission opls_8xx namespace from colliding with the\n"
            "; polymer templates'. Charges corrected by "
            f"{correction:+.6f} e/atom to sum to exactly +1.\n"
            "; Charges are the provisional part: RESP charges from the DFT\n"
            "; project replace this column when they are handed over.\n"
        )
        handle.write(f"[ moleculetype ]\n  {molecule}     3\n\n[ atoms ]\n")
        written = 0.0
        for index, atom in enumerate(atoms, start=1):
            charge = atom["charge"] + correction
            written += charge
            handle.write(
                f"  {index:4d}  {renamed[atom['type']]:8s}  1  {residue:5s} "
                f"{atom['name']:5s} {index:4d} {charge: .6f}  {atom['mass']:.4f}\n"
            )
        for section, width in (("bonds", 2), ("angles", 3), ("dihedrals", 4), ("pairs", 2)):
            rows = data.get(section) or []
            if not rows:
                continue
            handle.write(f"\n[ {section} ]\n")
            for row in rows:
                ids = "  ".join(row[:width])
                params = "  ".join(row[width:])
                handle.write(f"  {ids}   {params}\n")
    if abs(written - 1.0) > 1e-6:
        raise SystemExit(f"{molecule}: net charge {written:+.6f} != +1")

    gro_path = os.path.join(OUT, f"{molecule}.gro")
    with open(gro_path, "w") as handle:
        handle.write(f"{molecule} (acetylcholine cation)\n{len(atoms)}\n")
        for index, (atom, coord) in enumerate(zip(atoms, coords), start=1):
            handle.write(
                f"{1:5d}{residue:<5s}{atom['name']:>5s}{index:5d}"
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}\n"
            )
        handle.write("   3.00000   3.00000   3.00000\n")
    print(f"wrote {molecule}: {len(atoms)} atoms, net charge {written:+.6f}")
    return atomtype_rows


def build_anion(residue="CL", molecule="CL"):
    """Write the chloride template from the shipped OPLS-AA parameters."""
    name, btype, mass, charge, ptype, sigma, epsilon = CHLORIDE
    itp_path = os.path.join(OUT, f"{molecule}.itp")
    with open(itp_path, "w") as handle:
        handle.write(
            f"; Chloride ion. Parameters: {CHLORIDE_SOURCE}\n"
            "; A monatomic ion has no SMILES geometry to optimize, so\n"
            "; LigParGen cannot produce it; the shipped OPLS-AA force field\n"
            "; is the source instead.\n"
        )
        handle.write(f"[ moleculetype ]\n  {molecule}     3\n\n[ atoms ]\n")
        handle.write(
            f"  {1:4d}  {name:8s}  1  {residue:5s} {'CL':5s} {1:4d} "
            f"{charge: .6f}  {mass:.4f}\n"
        )
    gro_path = os.path.join(OUT, f"{molecule}.gro")
    with open(gro_path, "w") as handle:
        handle.write(f"{molecule} (chloride ion)\n1\n")
        handle.write(
            f"{1:5d}{residue:<5s}{'CL':>5s}{1:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        )
        handle.write("   1.00000   1.00000   1.00000\n")
    print(f"wrote {molecule}: 1 atom, charge {charge:+.3f} ({CHLORIDE_SOURCE})")
    return [(name, btype, mass, charge, ptype, sigma, epsilon)]


def merge_atomtypes(rows):
    """Append the DES ``[ atomtypes ]`` rows into forcefield.itp, idempotently.

    GROMACS wants every ``[ atomtypes ]`` before the first
    ``[ moleculetype ]``, and the include order puts forcefield.itp first, so
    that file is the only correct home for these. The rows live inside a
    marked fence, which is rewritten rather than duplicated on a re-run.
    """
    path = os.path.join(OUT, "forcefield.itp")
    text = open(path).read()
    if FENCE_START in text:
        head, rest = text.split(FENCE_START, 1)
        _, tail = rest.split(FENCE_END, 1)
        text = head + tail.lstrip("\n")
    marker = "\n[ atomtypes ]\n"
    if marker not in text:
        raise SystemExit(f"{path} has no [ atomtypes ] section to extend")
    # Insert immediately after the existing atomtypes block so the file keeps
    # one contiguous atomtypes section.
    lines = text.splitlines()
    end = None
    for index, line in enumerate(lines):
        if line.strip().startswith("[") and "atomtypes" in line:
            end = index + 1
            while end < len(lines) and not lines[end].strip().startswith("["):
                end += 1
            break
    if end is None:
        raise SystemExit(f"{path}: could not locate the atomtypes block")
    block = [FENCE_START]
    for name, btype, mass, charge, ptype, sigma, epsilon in rows:
        block.append(
            f"  {name:10s} {btype:8s} {mass:9.4f} {charge:8.3f} {ptype}  "
            f"{sigma:.5E}  {epsilon:.5E}"
        )
    block.append(FENCE_END)
    merged = lines[:end] + block + lines[end:]
    open(path, "w").write("\n".join(merged) + "\n")
    print(f"merged {len(rows)} DES atomtypes into forcefield.itp")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build_cation()
    rows += build_anion()
    merge_atomtypes(rows)
    print(
        "\nStoichiometry reminder: the formulation is Hexakis : AcChCl = 1 : 6,\n"
        "so a 64-junction network takes 384 cations and 384 anions."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check a built topology against the recipe it was supposed to realize.

A maker asks for a composition; the topology GROMACS reads is what the run
actually has. Between the two sit strand selection, cap removal, Packmol and
every include line, so the report (§18C.1, §18C.4, §18C.10) makes this
comparison a go/no-go gate rather than a courtesy: molecule counts and mass
balance must match the recipe, and charge must match the expected integer per
molecule -- not merely "round to zero".

The recipe is a small YAML that says what should be there::

    name: target_molar_r4
    provisional: true            # composition not yet confirmed by experiment
    network:                     # what the single HYDROGEL molecule is made of
      HEXU: 64
      STR_n33: 32
      reacted_arms: 64           # each loses one cap hydrogen
    species:                     # everything else, by moleculetype
      ACC: {count: 384, charge: 1}
      CL:  {count: 384, charge: -1}

Nothing in the recipe is trusted over the files: the network's expected atom
count, mass and charge are derived from the template ITPs in ``structure/``,
so a re-parameterization changes the expectation automatically. The only
numbers the recipe supplies are counts and formal charges.

Exit status is 0 when every row passes, 1 otherwise, so a driver script can
refuse to shrink or equilibrate a cell that is not what it claims to be.
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itp_inventory import MoleculeType, moleculetypes, system_composition  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir, "project"))

#: Charge is compared as a Decimal of the written values, so the tolerance is
#: about upstream rounding in the *template* files, not about float noise.
CHARGE_TOL = Decimal("0.0001")
MASS_TOL_REL = 1e-6


def _cap_hydrogen_mass(structure: str, junction_itp: str, hydrogel_yaml: str) -> float:
    """Mass of one thiol cap hydrogen (see cell_sizes._cap_hydrogen_mass)."""
    with open(hydrogel_yaml) as handle:
        cfg = yaml.safe_load(handle) or {}
    linkers = (((cfg.get("hydrogel_components") or {}).get("linker_definitions") or {})
               .get("LINKERS") or [])
    caps = [n for e in (linkers[0].get("stub_caps") or []) for n in e.get("cap_atoms", [])]
    masses = {}
    section = None
    with open(os.path.join(structure, junction_itp)) as handle:
        for line in handle:
            text = line.split(";", 1)[0].strip()
            if not text:
                continue
            if text.startswith("["):
                section = text.strip("[] ").strip().lower()
                continue
            if section == "atoms":
                parts = text.split()
                if len(parts) >= 8 and parts[4] in caps:
                    masses[parts[4]] = float(parts[7])
    values = sorted(set(round(v, 6) for v in masses.values()))
    if len(values) != 1:
        raise ValueError(f"cap hydrogens missing or of differing mass: {values}")
    return values[0]


def expected_network(recipe: dict, structure: str, hydrogel_yaml: str) -> MoleculeType:
    """What the HYDROGEL moleculetype should add up to, from the templates."""
    net = recipe["network"]
    reacted = int(net.get("reacted_arms", 0))
    parts = {k: int(v) for k, v in net.items() if k != "reacted_arms"}
    # Recipe keys are ITP file stems (STR_n33), not moleculetype names (that
    # file declares STR33): parse each file on its own and take whatever one
    # moleculetype it declares. Keying by moleculetype name here made the
    # pilot's audit crash with KeyError instead of judging the cell.
    types: Dict[str, MoleculeType] = {}
    for stem in parts:
        found = moleculetypes([os.path.join(structure, f"{stem}.itp")])
        if len(found) != 1:
            raise ValueError(f"{stem}.itp must declare exactly one moleculetype, found {sorted(found)}")
        types[stem] = next(iter(found.values()))
    exp = MoleculeType(name="HYDROGEL", source="recipe")
    for name, count in parts.items():
        exp.atom_count += types[name].atom_count * count
        exp.mass += types[name].mass * count
        # Template charge sums carry the upstream rounding; the expectation
        # for a neutral network is the integer, checked against tolerance.
    junction = next(n for n in parts if n.startswith("HEX"))
    cap = _cap_hydrogen_mass(structure, f"{junction}.itp", hydrogel_yaml)
    exp.atom_count -= reacted
    exp.mass -= reacted * cap
    exp.charge = 0.0
    return exp


def audit(top: str, recipe: dict, structure: str, hydrogel_yaml: str
          ) -> Tuple[List[Tuple[str, str, str, str, bool]], bool]:
    """Return rows ``(check, expected, found, note, ok)`` and the overall verdict."""
    comp = system_composition(top)
    found = dict(comp.molecules)
    rows: List[Tuple[str, str, str, str, bool]] = []

    # --- molecule counts -------------------------------------------------
    expected_counts = {"HYDROGEL": 1}
    expected_counts.update({k: int(v["count"]) for k, v in (recipe.get("species") or {}).items()})
    for name, want in expected_counts.items():
        got = found.get(name, 0)
        rows.append((f"count {name}", str(want), str(got), "", got == want))
    for name in found:
        if name not in expected_counts:
            rows.append((f"count {name}", "0", str(found[name]),
                         "present but not in recipe", False))

    # --- the network molecule ---------------------------------------------
    if "HYDROGEL" in comp.types:
        exp = expected_network(recipe, structure, hydrogel_yaml)
        got = comp.types["HYDROGEL"]
        rows.append(("network atoms", str(exp.atom_count), str(got.atom_count), "", 
                     exp.atom_count == got.atom_count))
        rel = abs(got.mass - exp.mass) / exp.mass if exp.mass else 0.0
        rows.append(("network mass / amu", f"{exp.mass:.3f}", f"{got.mass:.3f}",
                     f"rel diff {rel:.1e}", rel <= MASS_TOL_REL))

    # --- charge: per molecule against its formal integer ---------------------
    formal = {"HYDROGEL": 0}
    formal.update({k: int(v.get("charge", 0)) for k, v in (recipe.get("species") or {}).items()})
    total_expected = Decimal(0)
    for name, count in comp.molecules:
        q = Decimal(repr(comp.types[name].charge)).quantize(Decimal("0.000001"))
        want = Decimal(formal.get(name, 0))
        ok = abs(q - want) <= CHARGE_TOL
        rows.append((f"charge {name} / e", f"{int(want):+d}", f"{q:+.6f}",
                     "" if ok else f"off by {q - want:+.6f}", ok))
        total_expected += want * count
    total = Decimal(repr(comp.charge)).quantize(Decimal("0.000001"))
    ok = abs(total - total_expected) <= CHARGE_TOL
    rows.append(("system charge / e", f"{int(total_expected):+d}", f"{total:+.6f}",
                 "" if ok else "PME will add a neutralizing background", ok))

    return rows, all(r[-1] for r in rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", required=True, help="the build's system.top")
    parser.add_argument("--recipe", required=True, help="recipe YAML")
    parser.add_argument("--structure", default=os.path.join(PROJECT, "structure"))
    parser.add_argument("--hydrogel-yaml", default=os.path.join(PROJECT, "config", "hydrogel.yaml"))
    args = parser.parse_args(argv)

    with open(args.recipe) as handle:
        recipe = yaml.safe_load(handle)
    rows, verdict = audit(args.top, recipe, args.structure, args.hydrogel_yaml)

    tag = " (PROVISIONAL composition -- not confirmed by experiment)" if recipe.get("provisional") else ""
    print(f"recipe {recipe.get('name', '?')}{tag}")
    print(f"topology {args.top}\n")
    width = max(len(r[0]) for r in rows)
    print(f"{'check':<{width}}  {'expected':>14}  {'found':>14}  ok  note")
    for check, want, got, note, ok in rows:
        print(f"{check:<{width}}  {want:>14}  {got:>14}  {'ok' if ok else 'NO'}  {note}")
    print(f"\nverdict: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

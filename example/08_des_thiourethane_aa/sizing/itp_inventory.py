#!/usr/bin/env python3
"""Read atom counts, masses and charges straight out of ITP/TOP files.

Everything downstream of a build -- how big a box the system needs at a
target density, whether the molecule counts match the recipe, whether the
net charge is what it should be -- is a sum over the topology that GROMACS
itself will read. This module does that sum from the *same* files, so a
number quoted in a plan and a number GROMACS sees cannot silently diverge.

Deliberately dependency-free (no MDAnalysis, no ParmEd): the two sections
needed here are simple enough that a parser is shorter than a dependency,
and this must run on a login node with nothing installed.

Two entry points:

``moleculetypes(paths)``
    ``{name: MoleculeType}`` collected from one or more ITP files.
``system_composition(top_path)``
    Follows a ``.top``'s ``#include`` lines, then multiplies its
    ``[ molecules ]`` counts by those per-molecule values.

Both ignore every section that is not ``[ moleculetype ]``/``[ atoms ]``/
``[ molecules ]``, so an ITP full of dihedral tables costs nothing to read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["MoleculeType", "SystemComposition", "moleculetypes", "system_composition"]


@dataclass
class MoleculeType:
    """One ``[ moleculetype ]``, reduced to what a composition sum needs."""

    name: str
    #: Number of rows in the molecule's ``[ atoms ]`` section.
    atom_count: int = 0
    #: Sum of column 8 (mass, amu). Zero if the ITP omits masses -- an ITP may
    #: legally leave mass to the ``[ atomtypes ]`` default, and a silent 0 here
    #: would poison a density target, so the caller is told (see ``missing_mass``).
    mass: float = 0.0
    #: Sum of column 7 (charge, e).
    charge: float = 0.0
    #: Atom rows that carried no explicit mass column.
    missing_mass: int = 0
    #: Source file, for provenance in reports.
    source: str = ""


@dataclass
class SystemComposition:
    """A ``.top``'s ``[ molecules ]`` list resolved against its ITPs."""

    #: ``(molecule_name, count)`` in file order -- order is preserved because it
    #: must match the coordinate file, and a report that reorders it is misleading.
    molecules: List[Tuple[str, int]] = field(default_factory=list)
    #: Per-molecule values, keyed by name.
    types: Dict[str, MoleculeType] = field(default_factory=dict)
    #: ITP files pulled in through ``#include``.
    includes: List[str] = field(default_factory=list)

    @property
    def atom_count(self) -> int:
        return sum(self.types[name].atom_count * count for name, count in self.molecules)

    @property
    def mass(self) -> float:
        """Total mass, amu (= g/mol)."""
        return sum(self.types[name].mass * count for name, count in self.molecules)

    @property
    def charge(self) -> float:
        """Total charge, e. Should be an integer (usually 0) to ~1e-4."""
        return sum(self.types[name].charge * count for name, count in self.molecules)

    def box_for_density(self, density_g_cm3: float,
                        aspect: Sequence[float] = (1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
        """Box edges (nm) holding this mass at ``density_g_cm3``.

        ``rho = M / (602.214 * V)`` with M in g/mol and V in nm^3 -- that is
        Avogadro's number with the cm^3 -> nm^3 factor folded in. ``aspect``
        keeps a non-cubic construction cell's proportions (pass the supercell
        repeats); it is normalized here, so ``(4, 4, 6)`` and ``(2, 2, 3)`` mean
        the same shape.
        """
        if density_g_cm3 <= 0:
            raise ValueError(f"density must be positive, got {density_g_cm3}")
        volume = self.mass / (602.214076 * density_g_cm3)
        ax, ay, az = (float(v) for v in aspect)
        if min(ax, ay, az) <= 0:
            raise ValueError(f"aspect components must be positive, got {aspect!r}")
        # Scale so that (s*ax)(s*ay)(s*az) = volume.
        scale = (volume / (ax * ay * az)) ** (1.0 / 3.0)
        return (scale * ax, scale * ay, scale * az)


def _atom_row_values(parts: Sequence[str]) -> Tuple[float, float, bool]:
    """Return ``(charge, mass, mass_present)`` for one ``[ atoms ]`` row.

    GROMACS column order is
    ``nr type resnr residue atom cgnr charge [mass]``, so charge is index 6 and
    mass index 7, and the mass column is optional. Rows shorter than 7 fields
    are not atom rows at all and raise, because silently skipping them would
    undercount a molecule.
    """
    if len(parts) < 7:
        raise ValueError(f"[ atoms ] row has {len(parts)} fields, need >= 7: {' '.join(parts)!r}")
    charge = float(parts[6])
    if len(parts) >= 8:
        return charge, float(parts[7]), True
    return charge, 0.0, False


def _strip(line: str) -> str:
    """Drop the ``;`` comment and surrounding space. ``#`` lines survive."""
    return line.split(";", 1)[0].strip()


def moleculetypes(paths: Iterable[str]) -> Dict[str, MoleculeType]:
    """Collect ``[ moleculetype ]`` entries from ITP (or TOP) files.

    A name defined twice raises: GROMACS would refuse the same duplicate, and
    a silent last-one-wins here would report a composition the run cannot have.
    """
    found: Dict[str, MoleculeType] = {}
    for path in paths:
        section = None
        pending_name = False
        current: MoleculeType | None = None
        with open(path) as handle:
            for line in handle:
                text = _strip(line)
                if not text or text.startswith("#"):
                    continue
                if text.startswith("["):
                    section = text.strip("[] ").strip().lower()
                    pending_name = section == "moleculetype"
                    if section not in ("moleculetype", "atoms"):
                        current = current if section == "atoms" else current
                    continue
                if pending_name:
                    # First non-comment row after [ moleculetype ] is "name nrexcl".
                    name = text.split()[0]
                    if name in found:
                        raise ValueError(
                            f"moleculetype {name!r} defined twice: "
                            f"{found[name].source} and {path}"
                        )
                    current = MoleculeType(name=name, source=path)
                    found[name] = current
                    pending_name = False
                    continue
                if section == "atoms" and current is not None:
                    charge, mass, has_mass = _atom_row_values(text.split())
                    current.atom_count += 1
                    current.charge += charge
                    current.mass += mass
                    if not has_mass:
                        current.missing_mass += 1
    return found


def system_composition(top_path: str) -> SystemComposition:
    """Resolve a ``.top`` into per-species counts, atoms, mass and charge.

    ``#include`` paths are taken as written when absolute (the builder writes
    absolute paths) and otherwise relative to the ``.top``. A ``[ molecules ]``
    entry with no matching ``[ moleculetype ]`` raises rather than being
    counted as zero mass, which is the failure mode that would quietly shrink
    a density target.
    """
    top_dir = os.path.dirname(os.path.abspath(top_path))
    includes: List[str] = []
    molecules: List[Tuple[str, int]] = []
    section = None
    with open(top_path) as handle:
        for line in handle:
            text = _strip(line)
            if not text:
                continue
            if text.startswith("#include"):
                raw = text.split(None, 1)[1].strip().strip('"').strip("<>")
                includes.append(raw if os.path.isabs(raw) else os.path.join(top_dir, raw))
                continue
            if text.startswith("#"):
                continue
            if text.startswith("["):
                section = text.strip("[] ").strip().lower()
                continue
            if section == "molecules":
                parts = text.split()
                molecules.append((parts[0], int(parts[1])))

    # The .top may also define moleculetypes inline, so it is parsed alongside
    # its includes.
    types = moleculetypes(includes + [top_path])
    missing = [name for name, _ in molecules if name not in types]
    if missing:
        raise ValueError(
            f"{top_path}: [ molecules ] names with no [ moleculetype ] found: "
            f"{', '.join(sorted(set(missing)))}. Included: {len(includes)} file(s)."
        )
    return SystemComposition(molecules=molecules, types=types, includes=includes)


def _main(argv: Sequence[str]) -> int:
    """Print one system's composition: ``itp_inventory.py <system.top>``."""
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("top", help="system.top written by the builder")
    parser.add_argument("--density", type=float, default=1.0,
                        help="target density (g/cm3) for the box estimate")
    args = parser.parse_args(argv)

    comp = system_composition(args.top)
    print(f"{'species':<12} {'count':>7} {'atoms':>9} {'mass/amu':>13} {'charge/e':>10}")
    for name, count in comp.molecules:
        mt = comp.types[name]
        print(f"{name:<12} {count:>7} {mt.atom_count * count:>9} "
              f"{mt.mass * count:>13.3f} {mt.charge * count:>10.4f}")
    print(f"{'TOTAL':<12} {'':>7} {comp.atom_count:>9} {comp.mass:>13.3f} {comp.charge:>10.4f}")
    box = comp.box_for_density(args.density)
    print(f"\ncubic box at {args.density:g} g/cm3: {box[0]:.3f} nm "
          f"(volume {box[0] * box[1] * box[2]:.1f} nm^3)")
    short = [mt.name for mt in comp.types.values() if mt.missing_mass]
    if short:
        print(f"WARNING: no mass column on some atoms of: {', '.join(sorted(short))}; "
              "the mass total is therefore a lower bound")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))

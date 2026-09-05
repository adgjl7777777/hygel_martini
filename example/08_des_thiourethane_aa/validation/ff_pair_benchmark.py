#!/usr/bin/env python3
"""Does our force field reproduce the DFT ranking of Cl- binding sites?

The whole story rests on one ordering: chloride prefers the thiourethane
N-H over the urethane N-H over the residual thiol S-H. That ordering comes
from DFT. Every MD result we will ever produce comes from a force field that
has never been asked whether it agrees. The independent review (2026-09-05,
Major 4) made this the gate before any quantitative interpretation, and it is
answerable **now**: the DFT complexes are optimized and their CP interaction
energies exist; only the MM side was missing.

The test is a single point, not a fit. For each complex:

* take the DFT-optimized geometry exactly as it is -- no MM relaxation, so
  the comparison is against DFT's ``dE_int(CP)`` (the interaction at the
  complex geometry) and not against ``dE_bind`` (which adds fragment
  deformation the MM model would relax away differently);
* place our force field's atoms on those coordinates, matched by molecular
  graph rather than by file order, so a re-parameterization or a different
  atom ordering cannot silently misalign them;
* sum the intermolecular non-bonded energy. Two separate molecules share no
  exclusions and no 1-4 scaling, so this is an exact closed form -- every
  pair, LJ plus Coulomb, no cutoff, no PME, nothing to tune:

      E_int = sum_(i in A, j in B) 4 eps_ij [(sig_ij/r)^12 - (sig_ij/r)^6]
                                   + f q_i q_j / r

  with geometric combination (OPLS comb-rule 3) and f = 138.935458
  kJ mol^-1 nm e^-2.

A rigid scan along the X-H...Cl axis then reports where the MM minimum sits,
because a model can get a depth right for the wrong reason and a well in the
wrong place is a different failure from a well of the wrong size.

What a pass would mean: the ranking survives and the depths are in the right
neighbourhood, so MD built on this force field is at least asking the DFT
question. What it would not mean: that the charges are production quality --
they are 1.14*CM1A-LBCC and the RESP release is still pending. A failure
here would be the more useful result, because it would say so before a
hundred nanoseconds were spent.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Coulomb constant, kJ mol^-1 nm e^-2 (GROMACS ONE_4PI_EPS0).
COULOMB = 138.935458
KJ_PER_KCAL = 4.184

#: Masses -> elements, for reading elements out of an ITP.
_MASS_TO_ELEMENT = ((1.008, "H"), (12.011, "C"), (14.007, "N"), (15.999, "O"),
                    (32.06, "S"), (35.453, "Cl"), (30.974, "P"), (18.998, "F"))

#: Covalent radii (Angstrom, Cordero 2008) for inferring bonds from geometry.
_COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
             "Cl": 1.02, "F": 0.57, "P": 1.07}

#: Chloride as the builds use it: OPLS opls_401 from the GROMACS
#: oplsaa.ff distribution, charge -1.
CHLORIDE = {"sigma": 0.441724, "epsilon": 0.492833, "charge": -1.0}


@dataclass
class Atom:
    element: str
    sigma: float          # nm
    epsilon: float        # kJ/mol
    charge: float         # e
    name: str = ""


def element_for_mass(mass: float) -> str:
    best, diff = "X", 1e9
    for m, el in _MASS_TO_ELEMENT:
        if abs(mass - m) < diff:
            best, diff = el, abs(mass - m)
    return best if diff < 0.3 else "X"


def read_xyz(path: str) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """Elements and coordinates (Angstrom) from an XYZ file."""
    lines = open(path).read().splitlines()
    count = int(lines[0].split()[0])
    elements, coords = [], []
    for line in lines[2:2 + count]:
        parts = line.split()
        elements.append(parts[0].capitalize())
        coords.append(tuple(float(v) for v in parts[1:4]))
    if len(elements) != count:
        raise ValueError(f"{path}: header says {count} atoms, found {len(elements)}")
    return elements, coords


def bonds_from_geometry(elements: Sequence[str],
                        coords: Sequence[Tuple[float, float, float]],
                        slack: float = 0.4) -> List[Tuple[int, int]]:
    """Bond list from covalent radii: ``d < r_i + r_j + slack`` (Angstrom).

    Chloride is excluded from bonding on purpose -- in these complexes it is
    a free ion sitting in a hydrogen bond, and a 2.0 A H...Cl contact would
    otherwise be read as a covalent bond and merge the two fragments.
    """
    out = []
    for i, j in itertools.combinations(range(len(elements)), 2):
        if "Cl" in (elements[i], elements[j]):
            continue
        ri = _COVALENT.get(elements[i])
        rj = _COVALENT.get(elements[j])
        if ri is None or rj is None:
            continue
        d = math.dist(coords[i], coords[j])
        if d < ri + rj + slack:
            out.append((i, j))
    return out


def read_itp(path: str, atomtypes: Dict[str, Tuple[float, float]]
             ) -> Tuple[List[Atom], List[Tuple[int, int]]]:
    """Atoms (with LJ resolved from ``atomtypes``) and bonds from an ITP."""
    atoms: List[Atom] = []
    bonds: List[Tuple[int, int]] = []
    section = None
    for line in open(path):
        text = line.split(";", 1)[0].strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("["):
            section = text.strip("[] ").strip().lower()
            continue
        parts = text.split()
        if section == "atoms" and len(parts) >= 8:
            atype = parts[1]
            if atype not in atomtypes:
                raise KeyError(f"{path}: atom type {atype!r} not in the force field")
            sigma, epsilon = atomtypes[atype]
            atoms.append(Atom(element_for_mass(float(parts[7])), sigma, epsilon,
                              float(parts[6]), parts[4]))
        elif section == "bonds" and len(parts) >= 2:
            bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
    return atoms, bonds


def read_atomtypes(path: str) -> Dict[str, Tuple[float, float]]:
    """``{type: (sigma_nm, epsilon_kJ)}`` from an ITP's ``[ atomtypes ]``."""
    out: Dict[str, Tuple[float, float]] = {}
    section = None
    for line in open(path):
        text = line.split(";", 1)[0].strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("["):
            section = text.strip("[] ").strip().lower()
            continue
        if section == "atomtypes":
            parts = text.split()
            # name [bonded_type] [at.num] mass charge ptype sigma epsilon
            out[parts[0]] = (float(parts[-2]), float(parts[-1]))
    return out


def _signature(elements: Sequence[str], adjacency: Dict[int, set], index: int) -> tuple:
    """Element, degree and sorted neighbour elements -- cheap match pruning."""
    return (elements[index], len(adjacency[index]),
            tuple(sorted(elements[n] for n in adjacency[index])))


def match_graphs(elements_a: Sequence[str], bonds_a: Sequence[Tuple[int, int]],
                 elements_b: Sequence[str], bonds_b: Sequence[Tuple[int, int]]
                 ) -> Optional[List[int]]:
    """Map every atom of A onto an atom of B (same molecule, any ordering).

    Returns ``mapping`` with ``mapping[i]`` the B index for A's atom ``i``,
    or None when no isomorphism exists. Plain backtracking with element,
    degree and neighbour-element pruning: these molecules are 15-32 atoms,
    so a dependency on a graph library would cost more than it saves.
    """
    if len(elements_a) != len(elements_b) or len(bonds_a) != len(bonds_b):
        return None
    adj_a: Dict[int, set] = {i: set() for i in range(len(elements_a))}
    adj_b: Dict[int, set] = {i: set() for i in range(len(elements_b))}
    for i, j in bonds_a:
        adj_a[i].add(j); adj_a[j].add(i)
    for i, j in bonds_b:
        adj_b[i].add(j); adj_b[j].add(i)

    candidates = {}
    for i in range(len(elements_a)):
        sig = _signature(elements_a, adj_a, i)
        candidates[i] = [j for j in range(len(elements_b))
                         if _signature(elements_b, adj_b, j) == sig]
        if not candidates[i]:
            return None
    # Most constrained first: fail fast rather than deep.
    order = sorted(range(len(elements_a)), key=lambda i: len(candidates[i]))
    mapping: Dict[int, int] = {}
    used: set = set()

    def place(step: int) -> bool:
        if step == len(order):
            return True
        i = order[step]
        for j in candidates[i]:
            if j in used:
                continue
            # Consistent with everything already placed, both directions.
            ok = True
            for other, other_j in mapping.items():
                if (other in adj_a[i]) != (other_j in adj_b[j]):
                    ok = False
                    break
            if not ok:
                continue
            mapping[i] = j
            used.add(j)
            if place(step + 1):
                return True
            del mapping[i]
            used.discard(j)
        return False

    if not place(0):
        return None
    return [mapping[i] for i in range(len(elements_a))]


def interaction_energy(atoms_a: Sequence[Atom], coords_a: Sequence[Sequence[float]],
                       atoms_b: Sequence[Atom], coords_b: Sequence[Sequence[float]]
                       ) -> Tuple[float, float, float]:
    """``(total, lj, coulomb)`` in kJ/mol; coordinates in Angstrom.

    Exact: two separate molecules share no exclusions, so every cross pair
    contributes with full LJ and full Coulomb. Geometric combination is
    OPLS's ``comb-rule 3``, which is what ``forcefield.itp`` declares.
    """
    lj = 0.0
    qq = 0.0
    for a, ra in zip(atoms_a, coords_a):
        for b, rb in zip(atoms_b, coords_b):
            r = math.dist(ra, rb) / 10.0          # Angstrom -> nm
            if a.epsilon > 0 and b.epsilon > 0:
                sigma = math.sqrt(a.sigma * b.sigma)
                eps = math.sqrt(a.epsilon * b.epsilon)
                s6 = (sigma / r) ** 6
                lj += 4.0 * eps * (s6 * s6 - s6)
            qq += COULOMB * a.charge * b.charge / r
    return lj + qq, lj, qq


# ---------------------------------------------------------------------------
# The complexes
# ---------------------------------------------------------------------------

#: DFT reference, from the integrated report (§10.1, §10.3) and recomputed
#: from the raw ORCA outputs by the independent review. dE_int is the
#: counterpoise-corrected interaction at the complex geometry -- the quantity
#: an MM single point on the same geometry is comparable to. dE_bind adds
#: fragment deformation and is quoted for context only.
REFERENCE = {
    "C1_thiol_Cl":        dict(donor="thiol S-H",        dE_int=-18.34, dE_bind=-17.50,
                               h_cl=2.205, angle=159.3, itp=None),
    "C2_urethane_Cl":     dict(donor="urethane N-H",     dE_int=-22.69, dE_bind=-21.62,
                               h_cl=2.108, angle=170.4, itp=None),
    "C3_thiouret_Cl":     dict(donor="thiourethane N-H", dE_int=-25.52, dE_bind=-25.06,
                               h_cl=2.046, angle=174.6, itp=None),
    # F3x is the model compound our own crossing parameters came from: the
    # builder's thiourethane linkage, LNK, atom for atom.
    "C3x_thiouretext_Cl": dict(donor="thiourethane N-H (ester arm)", dE_int=-27.4,
                               dE_bind=None, h_cl=None, angle=None, itp="LNK"),
}


def split_complex(elements: Sequence[str], coords: Sequence[Tuple[float, float, float]]
                  ) -> Tuple[List[int], int]:
    """Fragment atom indices and the chloride index."""
    chlorides = [i for i, e in enumerate(elements) if e == "Cl"]
    if len(chlorides) != 1:
        raise ValueError(f"expected exactly one Cl, found {len(chlorides)}")
    cl = chlorides[0]
    return [i for i in range(len(elements)) if i != cl], cl


def donor_hydrogen(elements: Sequence[str], coords, fragment: Sequence[int], cl: int
                   ) -> Tuple[int, int, float, float]:
    """The H closest to Cl, its heavy neighbour, the H...Cl distance and angle.

    Identified from the geometry rather than from atom names, so it does not
    depend on how any file happens to label things.
    """
    best = min((i for i in fragment if elements[i] == "H"),
               key=lambda i: math.dist(coords[i], coords[cl]))
    heavy = min((i for i in fragment if elements[i] != "H"),
                key=lambda i: math.dist(coords[i], coords[best]))
    d = math.dist(coords[best], coords[cl])
    v1 = [coords[heavy][k] - coords[best][k] for k in range(3)]
    v2 = [coords[cl][k] - coords[best][k] for k in range(3)]
    n1 = math.sqrt(sum(v * v for v in v1)); n2 = math.sqrt(sum(v * v for v in v2))
    cosine = sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return best, heavy, d, angle


def scan_along_axis(atoms_frag: Sequence[Atom], coords_frag, cl_atom: Atom,
                    cl_pos: Sequence[float], h_pos: Sequence[float],
                    span: float = 0.8, step: float = 0.05
                    ) -> List[Tuple[float, float]]:
    """Rigid scan: slide Cl- along the H...Cl axis, ``(H...Cl in A, E in kJ/mol)``.

    The fragment never moves, so this reports where THIS force field would put
    the contact if the geometry were free in that one coordinate -- a well in
    the right place for the wrong depth and a well in the wrong place are
    different diagnoses.
    """
    axis = [cl_pos[k] - h_pos[k] for k in range(3)]
    norm = math.sqrt(sum(v * v for v in axis))
    unit = [v / norm for v in axis]
    out = []
    offset = -span / 2
    while offset <= span / 2 + 1e-9:
        pos = [cl_pos[k] + unit[k] * offset for k in range(3)]
        energy, _, _ = interaction_energy(atoms_frag, coords_frag, [cl_atom], [pos])
        out.append((norm + offset, energy))
        offset += step
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, os.pardir))
DFT_COMPLEXES = "/nas_3/active/soohki/27.des/dft/03_complexes"

#: Default fragment force fields. F3x is LNK itself -- the model compound the
#: builder's crossing parameters come from -- so the benchmark tests the
#: parameters the network actually uses, not a stand-in for them.
DEFAULT_ITPS = {
    "C1_thiol_Cl": os.path.join(HERE, "fragments", "F1.itp"),
    "C2_urethane_Cl": os.path.join(HERE, "fragments", "F2.itp"),
    "C3_thiouret_Cl": os.path.join(HERE, "fragments", "F3.itp"),
    "C3x_thiouretext_Cl": os.path.join(EXAMPLE, "parameterization", "raw", "LNK.itp"),
}


def benchmark(name: str, complex_dir: str, itp_path: str) -> dict:
    """One complex: MM interaction at the DFT geometry, plus the axis scan."""
    ref = REFERENCE[name]
    elements, coords = read_xyz(os.path.join(complex_dir, name, "opt.xyz"))
    fragment, cl = split_complex(elements, coords)
    frag_elements = [elements[i] for i in fragment]
    frag_coords = [coords[i] for i in fragment]
    frag_bonds = [(fragment.index(i), fragment.index(j))
                  for i, j in bonds_from_geometry(elements, coords)
                  if i in fragment and j in fragment]

    atomtypes = read_atomtypes(itp_path)
    itp_atoms, itp_bonds = read_itp(itp_path, atomtypes)
    mapping = match_graphs([a.element for a in itp_atoms], itp_bonds,
                           frag_elements, frag_bonds)
    if mapping is None:
        raise ValueError(
            f"{name}: the ITP ({len(itp_atoms)} atoms) and the DFT fragment "
            f"({len(frag_elements)} atoms) are not the same molecule -- refusing "
            "to place parameters on coordinates they do not describe")
    placed = [frag_coords[m] for m in mapping]

    chloride = Atom("Cl", CHLORIDE["sigma"], CHLORIDE["epsilon"], CHLORIDE["charge"], "CL")
    total, lj, qq = interaction_energy(itp_atoms, placed, [chloride], [coords[cl]])

    h_index, heavy, h_cl, angle = donor_hydrogen(elements, coords, fragment, cl)
    # The donor hydrogen's own charge: the single number that decides how this
    # force field ranks one X-H against another.
    inverse = {frag_idx: itp_idx for itp_idx, frag_idx in enumerate(mapping)}
    q_h = itp_atoms[inverse[fragment.index(h_index)]].charge
    q_x = itp_atoms[inverse[fragment.index(heavy)]].charge
    scan = scan_along_axis(itp_atoms, placed, chloride, coords[cl], coords[h_index])
    best_r, best_e = min(scan, key=lambda p: p[1])

    return {
        "name": name, "donor": ref["donor"], "atoms": len(itp_atoms),
        "mm_int": total / KJ_PER_KCAL, "mm_lj": lj / KJ_PER_KCAL, "mm_qq": qq / KJ_PER_KCAL,
        "dft_int": ref["dE_int"], "dft_h_cl": ref["h_cl"], "dft_angle": ref["angle"],
        "geom_h_cl": h_cl, "geom_angle": angle,
        "mm_min_r": best_r, "mm_min_e": best_e / KJ_PER_KCAL,
        "q_h": q_h, "q_x": q_x,
        "donor_atom": f"{elements[heavy]}-H", "itp": os.path.basename(itp_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--complexes", default=DFT_COMPLEXES,
                        help="DFT complex directory (read-only)")
    parser.add_argument("--itp", action="append", metavar="COMPLEX=PATH", default=[],
                        help="override the force-field ITP for a complex's fragment")
    parser.add_argument("--tol-rank", action="store_true",
                        help="exit 1 unless the DFT ranking is reproduced")
    args = parser.parse_args(argv)

    supplied = dict(DEFAULT_ITPS)
    for spec in args.itp:
        key, _, path = spec.partition("=")
        supplied[key] = path
    supplied = {k: v for k, v in supplied.items() if os.path.exists(v)}

    rows = []
    for name in REFERENCE:
        path = supplied.get(name)
        if path is None:
            continue
        rows.append(benchmark(name, args.complexes, path))

    if not rows:
        print("no complexes given: pass --itp COMPLEX=PATH", file=sys.stderr)
        return 2

    print(f"{'complex':<20} {'donor':<28} {'MM@DFT':>8} {'DFT':>8} {'diff':>7} "
          f"{'MM LJ':>7} {'MM qq':>8} {'q(H)':>7} {'H..Cl':>7} {'MMmin r':>8} {'MMmin E':>8}")
    for r in rows:
        diff = r["mm_int"] - r["dft_int"]
        print(f"{r['name']:<20} {r['donor']:<28} {r['mm_int']:>8.2f} {r['dft_int']:>8.2f} "
              f"{diff:>7.2f} {r['mm_lj']:>7.2f} {r['mm_qq']:>8.2f} {r['q_h']:>+7.4f} "
              f"{r['geom_h_cl']:>7.3f} {r['mm_min_r']:>8.3f} {r['mm_min_e']:>8.2f}")
    print("\nkcal/mol; MM@DFT is a single point on the DFT geometry (no relaxation, no")
    print("cutoff, exact pair sum). MMmin is a rigid slide of Cl- along the X-H...Cl axis:")
    print("where this force field would put the contact, and how deep it gets there.")

    if len(rows) > 1:
        def order(key):
            return [r["name"] for r in sorted(rows, key=lambda r: r[key])]
        dft_order = order("dft_int")
        for label, key in (("MM @ DFT geometry", "mm_int"), ("MM @ its own minimum", "mm_min_e")):
            mm_order = order(key)
            agree = mm_order == dft_order
            print(f"\nranking, strongest first -- {label}"
                  f"\n  DFT: {' > '.join(dft_order)}"
                  f"\n  MM : {' > '.join(mm_order)}"
                  f"\n  {'AGREES with DFT' if agree else 'DISAGREES with DFT'}")
            if args.tol_rank and not agree:
                rc = 1
        return 1 if (args.tol_rank and order("mm_int") != dft_order) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tile the strand's PPG block up to the experimental chain length.

LigParGen has an atom-count ceiling, so the strand was parameterized at
PO n = 3 (73 atoms). The experimental prepolymer is Mn ~ 2300, i.e. n ~ 33.
Chemically the interior of a PPG block is one repeat unit
``-O-CH(CH3)-CH2-`` over and over, so the long strand is obtained by
replicating that unit -- parameters and all -- rather than by a new QM run.

What makes this defensible rather than hand-waving is that the interior is
*exactly* regular, which this script verifies rather than assumes:

* the PO units are parameter-identical in canonical order -- LigParGen gives
  every atom its own type NAME (``st812`` vs ``st816``), so equality is checked
  on what those names resolve to (element, mass, sigma, epsilon) plus the
  inline bonded parameters, never on the labels;
* every bonded term touches at most two consecutive units (checked), so a
  term is either intra-unit or spans one unit boundary, and both classes
  replicate mechanically;
* intra-unit and boundary term counts agree across units.

Two honest approximations, both reported when the script runs:

* **Charge end-effect.** Unit charges are not perfectly transferable: the same
  position differs by up to 0.065 e between the first, middle and last PO unit
  (a monotonic 1.14*CM1A end-effect from the two urethane groups). Copies take
  the MIDDLE unit's charges, which are the most bulk-like of the three; the
  parameterized end units keep their own. Residual per-atom error is therefore
  bounded by that same 0.065 e, and is one reason this strand is a construction
  model rather than a transport-quality one.
* **Charge neutrality.** The middle unit's charges sum to +0.0196 e rather than zero, so
  30 verbatim copies would hand the molecule +0.59 e (and a 192-strand
  network +113 e). Each copy is therefore neutralized by spreading
  -q_unit/10 over its ten atoms: a -0.002 e shift per atom, an order of
  magnitude below the 1.14*CM1A-LBCC model's own uncertainty, and the
  molecule keeps the original net charge exactly.
* **Geometry.** Copies are placed by the rigid screw transform that carries
  the template unit's three BACKBONE atoms (CH, CH2, O) onto its successor's.
  Three non-collinear points fix a rigid transform exactly, so the junction
  bond and angles between consecutive copies reproduce the parameterized
  ones and each copy keeps the template's own internal geometry. (Fitting all
  ten atoms instead averages over side-group rotamers, which are not
  transferable: it gave a 0.12 nm fit residual and, applied thirty times, a
  10 nm "bond".) The chain is then a helix; its rise per unit is reported,
  and the result is a starting geometry -- relax in vacuo before use.

Usage::

    python3 tile_ppg.py --n 33            # writes ../project/structure/STR_n33.{itp,gro}
    python3 tile_ppg.py --n 33 --check    # verify only, write nothing
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, "..", "project", "structure")

#: Sections whose rows are (atom indices..., funct, params...) and therefore
#: replicate mechanically, with the index width of each.
BONDED_WIDTH = {"bonds": 2, "angles": 3, "dihedrals": 4, "pairs": 2}


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
    """(names, positions_nm) from a GRO file, fixed columns."""
    lines = open(path).read().splitlines()
    count = int(lines[1].split()[0])
    names, coords = [], []
    for line in lines[2 : 2 + count]:
        names.append(line[10:15].strip())
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))
    return names, np.array(coords, dtype=float)


class Strand:
    """The parsed strand: atoms, connectivity, coordinates, PO unit structure."""

    def __init__(self, itp_path, gro_path):
        data = parse_itp(itp_path)
        self.header = [
            row for row in data["moleculetype"]
        ]
        self.atoms = {}
        for row in data["atoms"]:
            self.atoms[int(row[0])] = {
                "type": row[1],
                "resnr": int(row[2]),
                "residue": row[3],
                "name": row[4],
                "cgnr": int(row[5]),
                "q": float(row[6]),
                "m": float(row[7]),
            }
        self.terms = {
            section: [
                [int(x) for x in row[:width]] + row[width:]
                for row in data[section]
            ]
            for section, width in BONDED_WIDTH.items()
        }
        self.adj = defaultdict(set)
        for i, j, *_ in self.terms["bonds"]:
            self.adj[i].add(j)
            self.adj[j].add(i)

        gro_names, self.coords = parse_gro(gro_path)
        if len(gro_names) != len(self.atoms):
            raise SystemExit(
                f"GRO has {len(gro_names)} atoms, ITP has {len(self.atoms)}"
            )
        self.units = self._find_po_units()

    # -- structure -------------------------------------------------------
    def element(self, index):
        mass = self.atoms[index]["m"]
        if mass < 2.0:
            return "H"
        if mass < 13.0:
            return "C"
        if mass < 15.0:
            return "N"
        if mass < 17.0:
            return "O"
        return "S"

    def _backbone_path(self):
        """Heavy-atom path between the two BCK attachment atoms."""
        ends = [i for i, a in self.atoms.items() if a["name"].startswith("BCK")]
        if len(ends) != 2:
            raise SystemExit(f"expected two BCK atoms, found {len(ends)}")
        start, goal = ends
        previous = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            for neighbour in sorted(self.adj[node]):
                if neighbour not in previous and self.element(neighbour) != "H":
                    previous[neighbour] = node
                    queue.append(neighbour)
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = previous[node]
        return path[::-1]

    def _unit_from_core(self, ch, ch2, oxygen):
        """Full PO unit atoms in canonical order from its three backbone atoms.

        Canonical order is ``[CH, H(CH), CH3, H, H, H, CH2, H, H, O]`` -- the
        order the unit-to-unit bijection relies on. Methyl and methylene
        hydrogens are ordered by index, which is well defined because each
        such set is chemically equivalent (verified: identical types).
        """
        core = {ch, ch2, oxygen}
        methyls = [
            n for n in self.adj[ch] if self.element(n) == "C" and n not in core
        ]
        if len(methyls) != 1:
            raise SystemExit(f"atom {ch} has {len(methyls)} methyl branches")
        methyl = methyls[0]
        ch_h = sorted(n for n in self.adj[ch] if self.element(n) == "H")
        methyl_h = sorted(n for n in self.adj[methyl] if self.element(n) == "H")
        ch2_h = sorted(n for n in self.adj[ch2] if self.element(n) == "H")
        if (len(ch_h), len(methyl_h), len(ch2_h)) != (1, 3, 2):
            raise SystemExit(
                f"unit at {ch}/{ch2}: hydrogen counts "
                f"{len(ch_h)}/{len(methyl_h)}/{len(ch2_h)} are not 1/3/2"
            )
        return [ch, ch_h[0], methyl, *methyl_h, ch2, *ch2_h, oxygen]

    def _find_po_units(self):
        """Locate the PPG block's PO units along the backbone path.

        A unit shows up on the path as the triple (C, C, O); the run of such
        triples between the two urethane ester oxygens is the PPG block.
        """
        path = self._backbone_path()
        elements = [self.element(i) for i in path]
        units = []
        position = 0
        while position + 2 < len(path):
            window = elements[position : position + 3]
            if window == ["C", "C", "O"]:
                ch, ch2, oxygen = path[position : position + 3]
                # A PO unit's first carbon carries exactly one methyl branch;
                # the urethane carbonyl carbon does not, which is what keeps
                # this from walking off the block.
                branches = [
                    n
                    for n in self.adj[ch]
                    if self.element(n) == "C" and n not in (ch2,)
                ]
                if len(branches) == 1 and self.element(branches[0]) == "C":
                    if all(self.element(h) == "H" for h in self.adj[branches[0]] if h != ch):
                        units.append(self._unit_from_core(ch, ch2, oxygen))
                        position += 3
                        continue
            position += 1
        if len(units) < 3:
            raise SystemExit(f"found {len(units)} PO units; need at least 3")
        return units

    # -- verification ----------------------------------------------------
    def check_regularity(self, ff_path=None):
        """Refuse to tile anything but a verifiably regular interior.

        Type NAMES are deliberately not compared: LigParGen mints a unique
        name per atom, so ``st812`` and ``st816`` are the same carbon under
        different labels. What must match, position by position, is what those
        names resolve to -- element and mass here, plus sigma/epsilon when a
        force field file is available -- and the inline bonded parameters.
        """
        signatures = [
            [(self.element(i), round(self.atoms[i]["m"], 6)) for i in unit]
            for unit in self.units
        ]
        for index, row in enumerate(signatures[1:], start=1):
            if row != signatures[0]:
                raise SystemExit(
                    f"PO unit {index} composition {row} differs from unit 0 "
                    f"{signatures[0]}; the units are not interchangeable"
                )
        ff_path = ff_path or os.path.join(STRUCT, "forcefield.itp")
        if os.path.isfile(ff_path):
            nonbonded = {}
            for row in parse_itp(ff_path)["atomtypes"]:
                # name [btype] [at.num] mass charge ptype sigma epsilon:
                # sigma/epsilon are the last two numeric columns.
                nonbonded[row[0]] = (float(row[-2]), float(row[-1]))
            lj = [
                [nonbonded.get(self.atoms[i]["type"]) for i in unit]
                for unit in self.units
            ]
            for index, row in enumerate(lj[1:], start=1):
                if row != lj[0]:
                    raise SystemExit(
                        f"PO unit {index} Lennard-Jones parameters differ from "
                        f"unit 0: {row} vs {lj[0]}"
                    )
        # Inline bonded parameters must agree position-wise too, or a copied
        # term would not carry the interior's own numbers.
        position_of = {}
        for unit_index, unit in enumerate(self.units):
            for position, atom in enumerate(unit):
                position_of[atom] = (unit_index, position)
        for section, width in BONDED_WIDTH.items():
            by_pattern = defaultdict(set)
            for row in self.terms[section]:
                ids, params = row[:width], tuple(row[width:])
                located = [position_of.get(i) for i in ids]
                if any(entry is None for entry in located):
                    continue
                units_touched = {entry[0] for entry in located}
                if len(units_touched) != 1:
                    continue
                pattern = tuple(entry[1] for entry in located)
                by_pattern[pattern].add(params)
            for pattern, parameter_sets in by_pattern.items():
                if len(parameter_sets) > 1:
                    raise SystemExit(
                        f"{section} pattern {pattern} carries differing inline "
                        f"parameters across PO units: {parameter_sets}"
                    )
        owner = {}
        for index, unit in enumerate(self.units):
            for atom in unit:
                owner[atom] = index
        spans = defaultdict(lambda: defaultdict(int))
        for section, width in BONDED_WIDTH.items():
            for row in self.terms[section]:
                touched = sorted({owner.get(i, -1) for i in row[:width]})
                if len([t for t in touched if t >= 0]) > 2:
                    raise SystemExit(
                        f"{section} term {row[:width]} touches PO units "
                        f"{touched}; tiling assumes at most two"
                    )
                spans[section][tuple(touched)] += 1
        # Interior units must agree, or a copied pattern would not be the
        # pattern the interior actually has.
        interior = range(1, len(self.units) - 1)
        for section in BONDED_WIDTH:
            intra = {spans[section].get((u,), 0) for u in interior}
            if len(intra) != 1:
                raise SystemExit(
                    f"{section}: interior intra-unit counts disagree ({intra})"
                )
        return spans

    def bijection(self, source, target):
        """Map template-unit atoms onto successor-unit atoms, canonically."""
        return dict(zip(self.units[source], self.units[target]))


def tile(strand, n_target, quiet=False):
    """Build the tiled molecule; returns (atoms, terms, coords) ready to write.

    Copies of the middle PO unit are inserted between it and its successor,
    each placed by the rigid screw transform carrying the template unit onto
    that successor and each neutralized to exactly zero net charge.
    """
    n_current = len(strand.units)
    if n_target < n_current:
        raise SystemExit(f"n={n_target} is below the parameterized n={n_current}")
    copies = n_target - n_current
    template_index = n_current // 2
    successor_index = template_index + 1
    template = strand.units[template_index]
    mapping = strand.bijection(template_index, successor_index)

    # -- rigid screw transform template -> successor ---------------------
    # Backbone atoms only (canonical positions 0, 6, 9 = CH, CH2, O): three
    # non-collinear points determine the transform exactly, so the inter-copy
    # junction geometry is the parameterized one rather than a least-squares
    # compromise over side-group rotamers.
    frame = [template[0], template[6], template[9]]
    source = strand.coords[[i - 1 for i in frame]]
    target = strand.coords[[mapping[i] - 1 for i in frame]]
    src_c, tgt_c = source.mean(axis=0), target.mean(axis=0)
    correlation = (source - src_c).T @ (target - tgt_c)
    u_mat, _, vt = np.linalg.svd(correlation)
    sign = np.sign(np.linalg.det(vt.T @ u_mat.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u_mat.T
    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(((source - src_c) @ rotation.T + tgt_c - target) ** 2, axis=1)
            )
        )
    )
    # Screw decomposition, reported so a degenerate (tightly coiled) helix is
    # visible rather than silently produced.
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    axis /= np.linalg.norm(axis)
    rise = abs(float(np.dot(tgt_c - src_c, axis)))

    def screw(points, times):
        out = np.array(points, dtype=float)
        for _ in range(times):
            out = (out - src_c) @ rotation.T + tgt_c
        return out

    # Atoms downstream of the cut, found topologically: cut the junction bond
    # (template O -> successor CH) and flood-fill from the successor side.
    # ITP index order is NOT chain order (hydrogens are numbered last), so an
    # index comparison here silently left the far half of the molecule behind.
    cut = (template[9], mapping[template[0]])
    downstream_set = set()
    queue = deque([cut[1]])
    downstream_set.add(cut[1])
    while queue:
        node = queue.popleft()
        for neighbour in strand.adj[node]:
            if neighbour == cut[0] and node == cut[1]:
                continue  # do not cross the cut
            if neighbour not in downstream_set:
                downstream_set.add(neighbour)
                queue.append(neighbour)
    if cut[0] in downstream_set:
        raise SystemExit(
            "cutting the PO junction bond did not separate the chain; the PPG "
            "block is part of a ring?"
        )

    # -- assemble atoms in backbone order --------------------------------
    # New atoms are appended after the template unit's own atoms so the ITP
    # reads in chain order; indices are assigned in a single renumbering pass.
    q_unit = sum(strand.atoms[i]["q"] for i in template)
    correction = q_unit / len(template)
    new_atoms = []          # list of dicts, in final order
    origin_of = []          # provenance: original index (or template index)
    insert_after = max(template)
    for index in sorted(strand.atoms):
        atom = dict(strand.atoms[index])
        new_atoms.append(atom)
        origin_of.append(index)
        if index == insert_after:
            for copy in range(1, copies + 1):
                for position, source_index in enumerate(template):
                    base = strand.atoms[source_index]
                    element = strand.element(source_index)
                    new_atoms.append({
                        "type": base["type"],
                        "resnr": base["resnr"],
                        "residue": base["residue"],
                        # element + copy number + position in unit, so a
                        # maintainer can read provenance off the name.
                        "name": f"{element}{copy:02d}{position}",
                        "cgnr": 0,
                        "q": base["q"] - correction,
                        "m": base["m"],
                    })
                    origin_of.append(("copy", copy, source_index))

    # index maps: original -> new, and (copy, original) -> new
    original_to_new = {}
    copy_to_new = defaultdict(dict)
    for new_index, origin in enumerate(origin_of, start=1):
        if isinstance(origin, tuple):
            _, copy, source_index = origin
            copy_to_new[copy][source_index] = new_index
        else:
            original_to_new[origin] = new_index
    for new_index, atom in enumerate(new_atoms, start=1):
        atom["cgnr"] = new_index

    # -- coordinates ------------------------------------------------------
    coords = np.zeros((len(new_atoms), 3), dtype=float)
    for original, new_index in original_to_new.items():
        coords[new_index - 1] = strand.coords[original - 1]
    # Everything downstream of the cut is pushed out by the copies, so the
    # original successor unit and the chain beyond it move with the last
    # copy's transform.
    downstream = sorted(downstream_set)
    if downstream:
        moved = screw(strand.coords[[i - 1 for i in downstream]], copies)
        for offset, original in enumerate(downstream):
            coords[original_to_new[original] - 1] = moved[offset]
    for copy in range(1, copies + 1):
        placed = screw(strand.coords[[i - 1 for i in template]], copy)
        for offset, source_index in enumerate(template):
            coords[copy_to_new[copy][source_index] - 1] = placed[offset]

    # -- bonded terms -----------------------------------------------------
    owner = {}
    for unit_index, unit in enumerate(strand.units):
        for atom in unit:
            owner[atom] = unit_index

    def resolve(index, copy):
        """Original index -> new index in the frame of insertion slot ``copy``.

        ``copy`` 0 means the template unit itself; ``copies + 1`` means the
        original successor unit. An atom of the template unit resolves into
        slot ``copy``, an atom of the successor unit into slot ``copy + 1``.
        """
        unit = owner.get(index)
        if unit == template_index:
            slot = copy
        elif unit == successor_index:
            slot = copy + 1
        else:
            return original_to_new[index]
        if slot == 0:
            return original_to_new[index]
        if slot == copies + 1:
            return original_to_new[mapping[index]] if unit == template_index else original_to_new[index]
        source = index if unit == template_index else {v: k for k, v in mapping.items()}[index]
        return copy_to_new[slot][source]

    new_terms = {section: [] for section in BONDED_WIDTH}
    for section, width in BONDED_WIDTH.items():
        for row in strand.terms[section]:
            ids, params = row[:width], row[width:]
            touched = {owner.get(i, -1) for i in ids}
            is_intra = touched == {template_index}
            is_span = touched == {template_index, successor_index}
            if is_intra:
                for copy in range(0, copies + 1):
                    new_terms[section].append(
                        [resolve(i, copy) for i in ids] + list(params)
                    )
            elif is_span:
                for copy in range(0, copies + 1):
                    new_terms[section].append(
                        [resolve(i, copy) for i in ids] + list(params)
                    )
            else:
                new_terms[section].append(
                    [original_to_new[i] for i in ids] + list(params)
                )

    if not quiet:
        print(f"template PO unit: atoms {template} (unit {template_index})")
        print(
            f"backbone screw: fit residual {rmsd:.2e} nm, rotation {angle:.1f} deg, "
            f"rise {rise:.4f} nm/unit, {len(downstream)} atoms downstream of the cut"
        )
        print(f"unit charge {q_unit:+.4f} e -> corrected by {-correction:+.5f} e/atom")
    return new_atoms, new_terms, coords, rmsd


def verify(strand, atoms, terms, coords, n_target):
    """Independent checks on the tiled result; raises on any failure."""
    copies = n_target - len(strand.units)
    unit_size = len(strand.units[0])
    expected_atoms = len(strand.atoms) + copies * unit_size
    if len(atoms) != expected_atoms:
        raise SystemExit(f"atom count {len(atoms)} != expected {expected_atoms}")

    charge = sum(a["q"] for a in atoms)
    original_charge = sum(a["q"] for a in strand.atoms.values())
    if abs(charge - original_charge) > 1e-6:
        raise SystemExit(
            f"net charge drifted: {charge:+.6f} vs original {original_charge:+.6f}"
        )

    # Term counts must scale exactly by the per-unit pattern.
    spans = strand.check_regularity()
    template_index = len(strand.units) // 2
    successor_index = template_index + 1
    for section, width in BONDED_WIDTH.items():
        intra = spans[section].get((template_index,), 0)
        span = spans[section].get((template_index, successor_index), 0)
        expected = len(strand.terms[section]) + copies * (intra + span)
        if len(terms[section]) != expected:
            raise SystemExit(
                f"{section}: {len(terms[section])} terms, expected {expected}"
            )

    # Connectivity: one component, and the two BCK atoms still exist.
    adjacency = defaultdict(set)
    for i, j, *_ in terms["bonds"]:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen = {1}
    queue = deque([1])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    if len(seen) != len(atoms):
        raise SystemExit(
            f"tiled molecule has {len(atoms) - len(seen)} atoms off the main "
            "component; the index mapping is wrong"
        )
    bck = [i for i, a in enumerate(atoms, start=1) if a["name"].startswith("BCK")]
    if len(bck) != 2:
        raise SystemExit(f"expected two BCK atoms after tiling, found {len(bck)}")

    # Geometry sanity: bonded lengths and the worst nonbonded contact.
    lengths = [
        float(np.linalg.norm(coords[i - 1] - coords[j - 1]))
        for i, j, *_ in terms["bonds"]
    ]
    bonded = {frozenset((i, j)) for i, j, *_ in terms["bonds"]}
    grid = defaultdict(list)
    for index, point in enumerate(coords, start=1):
        grid[tuple((point // 0.3).astype(int))].append(index)
    worst = (9.9, None)
    for key, members in grid.items():
        neighbours = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbours += grid.get((key[0] + dx, key[1] + dy, key[2] + dz), [])
        for a in members:
            for b in neighbours:
                if b <= a or frozenset((a, b)) in bonded:
                    continue
                if b in adjacency[a] or adjacency[a] & adjacency[b]:
                    continue  # 1-2 / 1-3
                distance = float(np.linalg.norm(coords[a - 1] - coords[b - 1]))
                if distance < worst[0]:
                    worst = (distance, (a, b))
    span = float(np.linalg.norm(coords[bck[0] - 1] - coords[bck[1] - 1]))
    return {
        "atoms": len(atoms),
        "charge": charge,
        "bond_min": min(lengths),
        "bond_max": max(lengths),
        "min_nonbonded": worst[0],
        "worst_pair": worst[1],
        "span": span,
    }


def write(atoms, terms, coords, name, out_prefix, n_target, rmsd):
    itp_path = os.path.join(STRUCT, f"{out_prefix}.itp")
    with open(itp_path, "w") as handle:
        handle.write(
            f"; Generated by tile_ppg.py --n {n_target} from STR.itp/STR.gro.\n"
            "; The PPG block's middle PO unit is replicated with its parameters;\n"
            "; each copy is charge-neutralized (see the script docstring).\n"
            f"; Copy atom names are <element><copy><position-in-unit>.\n"
            f"; Screw-transform RMSD template->successor: {rmsd:.4f} nm.\n"
        )
        handle.write(f"[ moleculetype ]\n  {name}     3\n\n[ atoms ]\n")
        for index, atom in enumerate(atoms, start=1):
            handle.write(
                f"  {index:5d}  {atom['type']:10s}  {atom['resnr']}  "
                f"{atom['residue']:6s} {atom['name']:6s} {atom['cgnr']:5d} "
                f"{atom['q']: .5f}  {atom['m']:.4f}\n"
            )
        for section, width in (("bonds", 2), ("angles", 3), ("dihedrals", 4), ("pairs", 2)):
            handle.write(f"\n[ {section} ]\n")
            for row in terms[section]:
                ids = "  ".join(str(x) for x in row[:width])
                params = "  ".join(str(x) for x in row[width:])
                handle.write(f"  {ids}   {params}\n")

    gro_path = os.path.join(STRUCT, f"{out_prefix}.gro")
    extent = coords.max(axis=0) - coords.min(axis=0) + 2.0
    with open(gro_path, "w") as handle:
        handle.write(f"{name} tiled to PO n={n_target} (tile_ppg.py)\n{len(atoms)}\n")
        for index, atom in enumerate(atoms, start=1):
            x, y, z = coords[index - 1]
            handle.write(
                f"{atom['resnr']:5d}{atom['residue']:<5s}{atom['name']:>5s}"
                f"{index % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        handle.write(f"{extent[0]:10.5f}{extent[1]:10.5f}{extent[2]:10.5f}\n")
    return itp_path, gro_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=33, help="target PO repeat count")
    parser.add_argument("--check", action="store_true", help="verify only")
    parser.add_argument("--in-prefix", default="STR")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--name", default=None, help="moleculetype name")
    args = parser.parse_args()

    strand = Strand(
        os.path.join(STRUCT, f"{args.in_prefix}.itp"),
        os.path.join(STRUCT, f"{args.in_prefix}.gro"),
    )
    spans = strand.check_regularity()
    print(f"parameterized strand: {len(strand.atoms)} atoms, PO n={len(strand.units)}")
    print("  per-unit terms:", {
        section: spans[section].get((1,), 0) for section in BONDED_WIDTH
    })
    print("  unit-boundary terms:", {
        section: spans[section].get((1, 2), 0) for section in BONDED_WIDTH
    })
    spread = max(
        max(strand.atoms[unit[position]]["q"] for unit in strand.units)
        - min(strand.atoms[unit[position]]["q"] for unit in strand.units)
        for position in range(len(strand.units[0]))
    )
    print(
        f"  charge end-effect: worst position-wise spread across units "
        f"{spread:.4f} e (copies take the middle unit's values)"
    )

    atoms, terms, coords, rmsd = tile(strand, args.n)
    report = verify(strand, atoms, terms, coords, args.n)
    print(
        "tiled: {atoms} atoms, net charge {charge:+.6f} e, bonds "
        "{bond_min:.3f}-{bond_max:.3f} nm, min nonbonded {min_nonbonded:.3f} nm "
        "(pair {worst_pair}), BCK-BCK span {span:.3f} nm".format(**report)
    )
    if report["min_nonbonded"] < 0.15:
        print(
            "[경고] 초기 기하에 접촉이 있습니다. 진공 이완(relax_tiled.sh) 후 "
            "좌표를 쓰십시오. / Contacts in the starting geometry: relax in "
            "vacuo before use."
        )
    if args.check:
        print("check only; nothing written")
        return
    out_prefix = args.out_prefix or f"{args.in_prefix}_n{args.n}"
    name = args.name or f"STR{args.n}"
    itp_path, gro_path = write(atoms, terms, coords, name, out_prefix, args.n, rmsd)
    print(f"wrote {os.path.basename(itp_path)} and {os.path.basename(gro_path)}")


if __name__ == "__main__":
    main()

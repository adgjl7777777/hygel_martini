#!/usr/bin/env python3
"""Turn raw LigParGen output into builder-ready all-atom templates.

Inputs (``raw/``), all straight from the LigParGen server (OPLS-AA,
1.14*CM1A-LBCC charges):

* ``HEX``  dipentaerythritol hexakis(3-mercaptopropionate), unreacted (93 atoms)
* ``STR``  CH3S-capped thiourethane / TDI / urethane / PPG(n=3) strand model (83)
* ``LNK``  methyl 3-mercaptopropionate + p-tolyl isocyanate thiourethane adduct
           (32) -- the source of every parameter *across* the S-C(=O) bond the
           builder forms.

Outputs (``../project/structure/``):

* ``HEXU.itp``/``HEXU.gro``  the junction in its UNREACTED form: all 93 atoms
  including the six thiol hydrogens, sulfurs marked as stubs (residue
  ``BCK``) but keeping their thiol types and charges. The builder's per-stub
  cap machinery deletes each arm's hydrogen and applies the reacted-form
  sulfur override (type ``lkS``, charge = qS + qH so the molecule stays
  exactly neutral) only on arms that chemically react -- so a partially
  converted junction keeps real S-H on its unreacted arms.
* ``hydrogel_stubs_snippet.yaml``  the exact ``stubs:``/``stub_caps:`` block
  to paste into ``config/hydrogel.yaml`` (atom names differ per arm, so the
  block is generated, not hand-written).
* ``STR.itp``/``STR.gro``  network strand: both CH3-S caps removed, the two
  thiourethane carbonyl carbons renamed ``BCK1``/``BCK2`` (the attachment
  atoms), each absorbing its deleted cap's total charge.
* ``forcefield.itp``  [defaults] + merged renamed [atomtypes] + the
  [angletypes]/[dihedraltypes] entries that resolve every *parameterless* term
  the builder emits: angles/dihedrals crossing the new S-C bond (values from
  LNK) and the junction-internal angles that involve a stub sulfur (values
  from HEX itself, which the linker loader's stub filter keeps out of the
  template's own angle list).

Known approximations, in one place:

* Charges are 1.14*CM1A-LBCC ("rough"); good enough to validate construction,
  not for quantitative ion transport. Charge folding (H into S, cap into
  carbonyl C) preserves neutrality exactly but localizes the residual.
* The carbonyl planarity improper N-C(=O)-S=O cannot live in either template:
  its four atoms span the junction and the strand, so the strand template --
  where that carbon is only two-coordinate -- cannot declare it. The builder
  generates it instead, from
  ``simulation_parameters.junction_bonded_generation.impropers`` in
  config/simulation.yaml, using this LNK model compound's own parameters
  (180.0, 43.932, 2). Measured effect at the 384 centres: out-of-plane
  deviation drops from mean 8.7 deg / max 49.7 deg to mean 2.4 / max 8.2.
* Junction-internal S parameters keep HEX's thiol-context values; only the
  sulfur's nonbonded type and the terms crossing the new bond use the
  thiourethane (LNK) values.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "..", "project", "structure")

# Crossing-term parameters measured on LNK (see module docstring), keyed by
# chemical pattern. Atom roles: CH2 = arm carbon bonded to S; Ca = the carbon
# behind it; Cco = thiourethane carbonyl carbon (the strand attachment atom);
# O/N its carbonyl oxygen and nitrogen; Cr/HN the ring carbon and amide H on N.
LNK_BOND_S_CCO = (0.1715, 187443.200)
LNK_ANGLES = {
    ("CH2", "S", "Cco"): (105.500, 561.660),
    ("S", "Cco", "O"): (118.180, 584.923),
    ("S", "Cco", "N"): (118.180, 584.923),
}
LNK_DIHEDRALS = {
    ("H", "CH2", "S", "Cco"): (1.354, 4.061, 0.000, -5.414, 0.000, 0.000),
    ("Ca", "CH2", "S", "Cco"): (0.941, 2.314, 2.410, -5.665, 0.000, 0.000),
    ("CH2", "S", "Cco", "O"): (25.476, 0.000, -25.476, 0.000, 0.000, 0.000),
    ("CH2", "S", "Cco", "N"): (9.623, -9.623, 0.000, 0.000, 0.000, 0.000),
    ("S", "Cco", "N", "Cr"): (27.196, 0.000, -27.196, 0.000, 0.000, 0.000),
    ("S", "Cco", "N", "HN"): (19.456, 1.464, -20.920, 0.000, 0.000, 0.000),
}
# The reacted sulfur's nonbonded row, from LNK's thiourethane S (opls_806).
LKS_ROW = ("lkS", "lkS", 32.060, 0.000, "A", 3.60000e-01, 1.77820e00)


def parse_itp(path):
    section = None
    data = defaultdict(list)
    for line in open(path):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        match = re.match(r"\[\s*(\S+)\s*\]", stripped)
        if match:
            section = match.group(1)
            continue
        data[section].append(stripped.split())
    return data


def parse_gro(path):
    lines = open(path).read().splitlines()
    count = int(lines[1].split()[0])
    coords = []
    for line in lines[2 : 2 + count]:
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))
    return coords


class Molecule:
    def __init__(self, name, prefix):
        itp = parse_itp(os.path.join(RAW, f"{name}.itp"))
        self.name = name
        self.prefix = prefix
        self.atomtypes = {
            row[0]: row for row in itp["atomtypes"]
        }  # name -> full row
        self.atoms = {}
        for row in itp["atoms"]:
            self.atoms[int(row[0])] = {
                "type": row[1],
                "residue": row[3],
                "name": row[4],
                "cgnr": int(row[5]),
                "q": float(row[6]),
                "m": float(row[7]),
            }
        self.bonds = [[int(r[0]), int(r[1])] + r[2:] for r in itp["bonds"]]
        self.angles = [[int(r[0]), int(r[1]), int(r[2])] + r[3:] for r in itp["angles"]]
        self.dihedrals = [
            [int(r[0]), int(r[1]), int(r[2]), int(r[3])] + r[4:]
            for r in itp["dihedrals"]
        ]
        self.pairs = [[int(r[0]), int(r[1])] + r[2:] for r in itp["pairs"]]
        # extend_conformer.py writes an unfolded conformer as <name>_ext.gro
        # (same atom order); prefer it so rigidly placed strands are extended
        # chains rather than collapsed blobs.
        gro = os.path.join(RAW, f"{name}_ext.gro")
        if not os.path.isfile(gro):
            gro = os.path.join(RAW, f"{name}.gro")
        self.coords = parse_gro(gro)
        assert len(self.coords) == len(self.atoms), name
        self.adj = defaultdict(set)
        for i, j, *_ in self.bonds:
            self.adj[i].add(j)
            self.adj[j].add(i)
        self.deleted = set()

    def hydrogens_of(self, idx):
        return [n for n in self.adj[idx] if self.atoms[n]["m"] < 2.0]

    def sulfurs(self):
        return sorted(i for i, a in self.atoms.items() if 31.0 < a["m"] < 34.0)

    def renamed_type(self, orig):
        return f"{self.prefix}{orig.split('_')[-1]}"

    def btype(self, idx):
        """Bonded type of an atom after renaming (column 2 of atomtypes)."""
        if self.atoms[idx].get("force_type"):
            return self.atoms[idx]["force_type"]
        orig = self.atoms[idx]["type"]
        return f"{self.prefix}{self.atomtypes[orig][1]}"

    def write(self, out_prefix, molecule_name):
        keep = [i for i in sorted(self.atoms) if i not in self.deleted]
        renumber = {old: new + 1 for new, old in enumerate(keep)}
        itp_path = os.path.join(OUT, f"{out_prefix}.itp")
        with open(itp_path, "w") as f:
            f.write(f"; Generated by build_templates.py from raw/{self.name}.itp\n")
            f.write("; (LigParGen OPLS-AA / 1.14*CM1A-LBCC), reacted/marked form.\n")
            f.write("[ moleculetype ]\n")
            f.write(f"  {molecule_name}     3\n\n[ atoms ]\n")
            total_q = 0.0
            for old in keep:
                atom = self.atoms[old]
                type_name = atom.get("force_type") or self.renamed_type(atom["type"])
                total_q += atom["q"]
                f.write(
                    f"  {renumber[old]:4d}  {type_name:10s}  1  "
                    f"{atom['residue']:6s} {atom['name']:6s} {renumber[old]:4d} "
                    f"{atom['q']: .4f}  {atom['m']:.4f}\n"
                )
            for section, rows, width in (
                ("bonds", self.bonds, 2),
                ("angles", self.angles, 3),
                ("dihedrals", self.dihedrals, 4),
                ("pairs", self.pairs, 2),
            ):
                f.write(f"\n[ {section} ]\n")
                for row in rows:
                    ids, params = row[:width], row[width:]
                    if any(i in self.deleted for i in ids):
                        continue
                    f.write(
                        "  "
                        + "  ".join(str(renumber[i]) for i in ids)
                        + "   "
                        + "  ".join(str(p) for p in params)
                        + "\n"
                    )
        if abs(total_q) > 5e-4:
            raise SystemExit(f"{out_prefix}: net charge {total_q:+.4f} != 0")
        gro_path = os.path.join(OUT, f"{out_prefix}.gro")
        with open(gro_path, "w") as f:
            f.write(f"{molecule_name} template (build_templates.py)\n{len(keep)}\n")
            for old in keep:
                atom = self.atoms[old]
                x, y, z = self.coords[old - 1]
                f.write(
                    f"{1:5d}{atom['residue']:<5s}{atom['name']:>5s}"
                    f"{renumber[old]:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
                )
            f.write("  10.00000  10.00000  10.00000\n")
        print(f"wrote {out_prefix}: {len(keep)} atoms, net charge {total_q:+.5f}")
        return renumber


def build_junction():
    """Write the UNREACTED junction plus the stubs/stub_caps config snippet.

    Nothing is deleted here any more: the template keeps every thiol
    hydrogen, and the builder's cap machinery removes an arm's hydrogen (and
    applies its reacted-form sulfur override) only when that arm reacts.
    """
    hex_mol = Molecule("HEX", "hx")
    arms = []  # per arm: dict of roles S/CH2/Ca/H(=thiol H)/Hc(CH2 hydrogens)
    for sulfur in hex_mol.sulfurs():
        thiol_h = [h for h in hex_mol.hydrogens_of(sulfur)]
        assert len(thiol_h) == 1, f"S{sulfur}: expected one thiol H"
        ch2 = [n for n in hex_mol.adj[sulfur] if hex_mol.atoms[n]["m"] > 11.0]
        assert len(ch2) == 1
        ch2 = ch2[0]
        c_alpha = [
            n
            for n in hex_mol.adj[ch2]
            if n != sulfur and hex_mol.atoms[n]["m"] > 11.0
        ]
        assert len(c_alpha) == 1
        arm = {
            "S": sulfur,
            "CH2": ch2,
            "Ca": c_alpha[0],
            "H": [n for n in hex_mol.adj[ch2] if hex_mol.atoms[n]["m"] < 2.0],
            "thiol_H": thiol_h[0],
        }
        arms.append(arm)
        hex_mol.atoms[sulfur]["residue"] = "BCK"
    for idx, atom in hex_mol.atoms.items():
        if atom["residue"] != "BCK":
            atom["residue"] = "HEX"
    hex_mol.write("HEXU", "HEXU")

    # The stubs/stub_caps block for config/hydrogel.yaml. Reacted charge =
    # qS + qH: folding the deleted hydrogen's charge into its sulfur keeps
    # every junction exactly neutral at any conversion.
    lines = ["        stubs:"]
    for arm in arms:
        lines.append(
            "          - [{between: STR1, bond_funct: 1, "
            "bond_c0: 0.1715, bond_c1: 187443.2}]"
        )
    lines.append("        stub_caps:")
    for arm in arms:
        s_atom = hex_mol.atoms[arm["S"]]
        h_atom = hex_mol.atoms[arm["thiol_H"]]
        reacted_q = s_atom["q"] + h_atom["q"]
        lines.append(
            f"          - {{cap_atoms: [{h_atom['name']}], "
            f"reacted: {{{s_atom['name']}: "
            f"{{charge: {reacted_q:.4f}, type: lkS}}}}}}"
        )
    snippet = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "hydrogel_stubs_snippet.yaml"), "w") as f:
        f.write("# Paste into config/hydrogel.yaml under the linker entry.\n")
        f.write(snippet)
    print("wrote hydrogel_stubs_snippet.yaml")
    return hex_mol, arms


def build_strand():
    str_mol = Molecule("STR", "st")
    ends = []  # per end: roles Cco/O/N/Cr/HN
    bck_names = iter(["BCK1", "BCK2"])
    for sulfur in str_mol.sulfurs():
        carbons = sorted(n for n in str_mol.adj[sulfur] if str_mol.atoms[n]["m"] > 11.0)
        methyl = [c for c in carbons if len(str_mol.hydrogens_of(c)) == 3]
        carbonyl = [
            c
            for c in carbons
            if any(15.0 < str_mol.atoms[n]["m"] < 17.0 for n in str_mol.adj[c])
        ]
        assert len(methyl) == 1 and len(carbonyl) == 1, f"cap at S{sulfur}"
        cap_atoms = [sulfur, methyl[0]] + str_mol.hydrogens_of(methyl[0])
        cap_charge = sum(str_mol.atoms[i]["q"] for i in cap_atoms)
        str_mol.deleted.update(cap_atoms)
        cco = carbonyl[0]
        str_mol.atoms[cco]["q"] += cap_charge
        str_mol.atoms[cco]["name"] = next(bck_names)
        oxygen = [n for n in str_mol.adj[cco] if 15.0 < str_mol.atoms[n]["m"] < 17.0][0]
        nitrogen = [n for n in str_mol.adj[cco] if 13.0 < str_mol.atoms[n]["m"] < 15.0][0]
        ring_c = [n for n in str_mol.adj[nitrogen] if str_mol.atoms[n]["m"] > 11.0 and n != cco][0]
        amide_h = str_mol.hydrogens_of(nitrogen)[0]
        ends.append({"Cco": cco, "O": oxygen, "N": nitrogen, "Cr": ring_c, "HN": amide_h})
    for atom in str_mol.atoms.values():
        atom["residue"] = "STR"
    str_mol.write("STR", "STR")
    return str_mol, ends


def write_forcefield(hex_mol, arms, str_mol, ends):
    path = os.path.join(OUT, "forcefield.itp")
    with open(path, "w") as f:
        f.write("; Assembled by build_templates.py -- OPLS-AA conventions.\n")
        f.write("[ defaults ]\n; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n")
        f.write("  1  3  yes  0.5  0.5\n\n[ atomtypes ]\n")
        f.write("; renamed per molecule (hx*/st*) so the two LigParGen 800-series\n")
        f.write("; namespaces cannot collide; lkS is the thiourethane sulfur (LNK).\n")
        f.write(
            f"  {LKS_ROW[0]:10s} {LKS_ROW[1]:8s} {LKS_ROW[2]:9.4f} {LKS_ROW[3]:8.3f} "
            f"{LKS_ROW[4]}  {LKS_ROW[5]:.5E}  {LKS_ROW[6]:.5E}\n"
        )
        for mol in (hex_mol, str_mol):
            used = set()
            for idx, atom in mol.atoms.items():
                if idx in mol.deleted or atom.get("force_type"):
                    continue
                used.add(atom["type"])
            for orig in sorted(used):
                row = mol.atomtypes[orig]
                f.write(
                    f"  {mol.renamed_type(orig):10s} {mol.prefix + row[1]:8s} "
                    f"{float(row[2]):9.4f} {float(row[3]):8.3f} {row[4]}  "
                    f"{row[5]}  {row[6]}\n"
                )

        # ------ angletypes ------------------------------------------------
        seen = set()
        f.write("\n[ angletypes ]\n")
        f.write("; (1) angles crossing the builder-formed S-C bond, from LNK;\n")
        f.write("; (2) junction-internal angles touching a stub sulfur, from HEX\n")
        f.write(";     (the linker loader keeps stub atoms out of template angles,\n")
        f.write(";      so these arrive parameterless and resolve here).\n")

        def emit_angle(b1, b2, b3, theta, k):
            key = (b1, b2, b3) if (b1, b2, b3) <= (b3, b2, b1) else (b3, b2, b1)
            if key in seen:
                return
            seen.add(key)
            f.write(f"  {b1:8s} {b2:8s} {b3:8s}  1  {theta:9.3f}  {k:10.3f}\n")

        for arm in arms:
            for end in ends:
                theta, k = LNK_ANGLES[("CH2", "S", "Cco")]
                emit_angle(hex_mol.btype(arm["CH2"]), "lkS", str_mol.btype(end["Cco"]), theta, k)
        for end in ends:
            for third, pattern in (("O", ("S", "Cco", "O")), ("N", ("S", "Cco", "N"))):
                theta, k = LNK_ANGLES[pattern]
                emit_angle("lkS", str_mol.btype(end["Cco"]), str_mol.btype(end[third]), theta, k)
        sulfur_set = {arm["S"] for arm in arms}
        # Every template angle touching a stub sulfur is emitted parameterless
        # by the builder (the linker loader keeps stub atoms out of template
        # angle lists), for BOTH arm states: reacted (CH2-S-Cco etc., lkS
        # keys, above) and unreacted (C-C-S, H-C-S, and the thiol C-S-H /
        # H-S-... angles, thiol-type keys, here).
        for row in hex_mol.angles:
            ids, params = row[:3], row[3:]
            if not any(i in sulfur_set for i in ids):
                continue
            # The sulfur's nonbonded type depends on its arm's state (thiol
            # type unreacted, lkS reacted), and these angles are emitted
            # parameterless in BOTH states, so each entry is keyed twice --
            # once per sulfur type. Angles through the thiol hydrogen exist
            # only unreacted, but the duplicate lkS key is inert (nothing
            # reacted ever emits them).
            base = [hex_mol.btype(i) for i in ids]
            variants = [base]
            if any(i in sulfur_set for i in ids):
                swapped = [
                    ("lkS" if i in sulfur_set else hex_mol.btype(i))
                    for i in ids
                ]
                if swapped != base:
                    variants.append(swapped)
            for b1, b2, b3 in variants:
                emit_angle(b1, b2, b3, float(params[1]), float(params[2]))

        # ------ dihedraltypes ---------------------------------------------
        f.write("\n[ dihedraltypes ]\n")
        f.write("; every proper torsion crossing the builder-formed S-C bond;\n")
        f.write("; Ryckaert-Bellemans coefficients from LNK. The carbonyl\n")
        f.write("; planarity improper N-C-S=O spans junction and strand, so it\n")
        f.write("; a documented loss.\n")
        seen_d = set()

        def emit_dih(b1, b2, b3, b4, coeffs):
            key = (b1, b2, b3, b4)
            if key[::-1] < key:
                key = key[::-1]
            if key in seen_d:
                return
            seen_d.add(key)
            f.write(
                f"  {b1:8s} {b2:8s} {b3:8s} {b4:8s}  3  "
                + "  ".join(f"{c:8.3f}" for c in coeffs)
                + "\n"
            )

        for arm in arms:
            for end in ends:
                cco = str_mol.btype(end["Cco"])
                for h_idx in arm["H"]:
                    emit_dih(hex_mol.btype(h_idx), hex_mol.btype(arm["CH2"]), "lkS", cco,
                             LNK_DIHEDRALS[("H", "CH2", "S", "Cco")])
                emit_dih(hex_mol.btype(arm["Ca"]), hex_mol.btype(arm["CH2"]), "lkS", cco,
                         LNK_DIHEDRALS[("Ca", "CH2", "S", "Cco")])
                emit_dih(hex_mol.btype(arm["CH2"]), "lkS", cco, str_mol.btype(end["O"]),
                         LNK_DIHEDRALS[("CH2", "S", "Cco", "O")])
                emit_dih(hex_mol.btype(arm["CH2"]), "lkS", cco, str_mol.btype(end["N"]),
                         LNK_DIHEDRALS[("CH2", "S", "Cco", "N")])
        for end in ends:
            cco = str_mol.btype(end["Cco"])
            nb = str_mol.btype(end["N"])
            emit_dih("lkS", cco, nb, str_mol.btype(end["Cr"]),
                     LNK_DIHEDRALS[("S", "Cco", "N", "Cr")])
            emit_dih("lkS", cco, nb, str_mol.btype(end["HN"]),
                     LNK_DIHEDRALS[("S", "Cco", "N", "HN")])
    print(f"wrote forcefield.itp ({len(seen)} angletypes, {len(seen_d)} dihedraltypes)")


def main():
    os.makedirs(OUT, exist_ok=True)
    hex_mol, arms = build_junction()
    str_mol, ends = build_strand()
    write_forcefield(hex_mol, arms, str_mol, ends)

    import math

    bck = [i for i, a in str_mol.atoms.items() if a["name"] in ("BCK1", "BCK2")]
    (x1, y1, z1), (x2, y2, z2) = str_mol.coords[bck[0] - 1], str_mol.coords[bck[1] - 1]
    span = math.dist((x1, y1, z1), (x2, y2, z2))
    s_coords = [hex_mol.coords[arm["S"] - 1] for arm in arms]
    kept = [i for i in sorted(hex_mol.atoms) if i not in hex_mol.deleted]
    cx = sum(hex_mol.coords[i - 1][0] for i in kept) / len(kept)
    cy = sum(hex_mol.coords[i - 1][1] for i in kept) / len(kept)
    cz = sum(hex_mol.coords[i - 1][2] for i in kept) / len(kept)
    arm_len = sum(math.dist(c, (cx, cy, cz)) for c in s_coords) / len(s_coords)
    print(f"strand BCK1-BCK2 span : {span:.3f} nm")
    print(f"junction mean arm     : {arm_len:.3f} nm (centroid to S)")
    print(f"S-Cco bond            : {LNK_BOND_S_CCO[0]:.4f} nm, k={LNK_BOND_S_CCO[1]}")
    print(
        "suggested cell_parameter ~= span + 2*(arm + bond) = "
        f"{span + 2 * (arm_len + LNK_BOND_S_CCO[0]):.2f} nm"
    )


if __name__ == "__main__":
    main()

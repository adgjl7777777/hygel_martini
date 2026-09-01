#!/usr/bin/env python3
"""Replace the folded LigParGen strand conformer with an extended one.

LigParGen returns a gas-phase-optimized (collapsed) conformer: the two
attachment carbons of STR sit 0.44 nm apart in a ~2.5 nm-contour molecule.
Placed rigidly between junctions, each strand is then a dense blob, and
neighbouring blobs overlap. This script embeds the same molecule (same atom
order -- LigParGen's PDB order equals its ITP order) with RDKit ETKDG,
MMFF-relaxes each conformer, and keeps the one with the largest
BCK-carbonyl-to-BCK-carbonyl distance.

Run with a python that has rdkit; writes raw/STR_ext.gro, which
build_templates.py prefers over raw/STR.gro when present.
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
SMILES = "CSC(=O)Nc1cc(NC(=O)OC(C)COC(C)COC(C)COC(=O)Nc2ccc(C)c(NC(=O)SC)c2)ccc1C"

pdb = Chem.MolFromPDBFile(os.path.join(RAW, "STR.pdb"), removeHs=False)
ref = Chem.MolFromSmiles(SMILES)
mol = AllChem.AssignBondOrdersFromTemplate(ref, pdb)

# The two thiourethane carbonyl carbons: bonded to S, O(double) and N.
targets = []
for atom in mol.GetAtoms():
    if atom.GetSymbol() != "C":
        continue
    symbols = sorted(n.GetSymbol() for n in atom.GetNeighbors())
    if symbols == ["N", "O", "S"]:
        targets.append(atom.GetIdx())
assert len(targets) == 2, targets

params = AllChem.ETKDGv3()
params.randomSeed = 2026
ids = AllChem.EmbedMultipleConfs(mol, numConfs=64, params=params)
best, best_d = None, -1.0
for cid in ids:
    AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
    conf = mol.GetConformer(cid)
    d = conf.GetAtomPosition(targets[0]).Distance(conf.GetAtomPosition(targets[1]))
    if d > best_d:
        best, best_d = cid, d
print(f"best conformer: BCK-BCK {best_d/10:.3f} nm (of {len(ids)} embedded)")

conf = mol.GetConformer(best)
names = [line[12:16].strip() for line in open(os.path.join(RAW, "STR.pdb"))
         if line.startswith(("ATOM", "HETATM"))]
with open(os.path.join(RAW, "STR_ext.gro"), "w") as f:
    f.write(f"STR extended conformer (extend_conformer.py, d={best_d/10:.3f} nm)\n")
    f.write(f"{mol.GetNumAtoms()}\n")
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        f.write(f"{1:5d}{'UNK':<5s}{names[i]:>5s}{i + 1:5d}"
                f"{p.x / 10:8.3f}{p.y / 10:8.3f}{p.z / 10:8.3f}\n")
    f.write("  10.00000  10.00000  10.00000\n")
print("wrote raw/STR_ext.gro")

from __future__ import annotations
from pathlib import Path
from ase.io import read, write

def xyz_to_pdb(xyz_path: str | Path, pdb_path: str | Path) -> None:
    \"\"\"
    Converts an XYZ file to a PDB file using ASE.
    \"\"\"
    atoms = read(xyz_path)
    # We add basic residue info if needed for PDB format compatibility
    write(pdb_path, atoms)
    print(f\"[ASE] Converted {xyz_path} to {pdb_path}\")

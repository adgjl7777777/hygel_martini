"""ASE-based file format helpers for the qm_to_opls workflow.

Owns the single XYZ -> PDB conversion step needed before submitting a
structure to LigParGen (see ``ligpargen_api.run_parameterization_flow``).
Relies entirely on ASE's readers/writers; no coordinates are modified.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write


def xyz_to_pdb(xyz_path: str | Path, pdb_path: str | Path) -> None:
    """Convert an XYZ file to a PDB file using ASE read/write.

    Args:
        xyz_path: Input XYZ file (single frame is read).
        pdb_path: Output PDB path; parent directory must already exist.
    """
    atoms = read(xyz_path)
    write(pdb_path, atoms)
    print(f"[ASE] Converted {xyz_path} to {pdb_path}")

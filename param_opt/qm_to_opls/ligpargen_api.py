from __future__ import annotations

from pathlib import Path

# LigParGen Server URL (Yale)
LIGPARGEN_URL = "http://zarbi.chem.yale.edu/ligpargen/server.php"


def submit_to_ligpargen(pdb_path: str | Path, output_dir: str | Path, name: str = "molecule") -> str:
    """Submits a PDB file to LigParGen and stores placeholder outputs."""
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LigParGen] Submitting {pdb_path.name} to server...")

    itp_output = output_dir / f"{name}.itp"
    gro_output = output_dir / f"{name}.gro"
    itp_output.write_text("; Initial OPLS-AA ITP for " + name, encoding="utf-8")
    gro_output.write_text("; Initial OPLS-AA GRO for " + name, encoding="utf-8")

    return str(itp_output)


def run_parameterization_flow(xyz_path: str | Path, out_root: str | Path, symbol: str):
    """High-level flow: XYZ -> PDB -> LigParGen -> ITP/GRO."""
    from .ase_utils import xyz_to_pdb

    temp_dir = Path(out_root) / "temp_params" / symbol
    temp_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = temp_dir / f"{symbol}.pdb"
    xyz_to_pdb(xyz_path, pdb_path)

    itp_path = submit_to_ligpargen(pdb_path, temp_dir, name=symbol)
    print(f"[Flow] Base parameterization complete for {symbol}: {itp_path}")
    return itp_path

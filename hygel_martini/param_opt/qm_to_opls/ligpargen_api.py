"""LigParGen submission stubs for the qm_to_opls workflow.

Owns the OPLS-AA parameterization step of stage 01: turning an optimized
XYZ into LigParGen-ready inputs and the resulting ITP/GRO files.  Note
that :func:`submit_to_ligpargen` currently performs NO network call: it
only creates placeholder ITP/GRO files (the URL below is kept for the
real submission), so downstream code can wire paths before the actual
server integration exists.
"""

from __future__ import annotations

from pathlib import Path

# LigParGen Server URL (Yale)
LIGPARGEN_URL = "http://zarbi.chem.yale.edu/ligpargen/server.php"


def submit_to_ligpargen(pdb_path: str | Path, output_dir: str | Path, name: str = "molecule") -> str:
    """Create placeholder OPLS-AA outputs for a PDB submission.

    Despite the name, no request is sent to the LigParGen server yet:
    the function only creates ``<name>.itp`` / ``<name>.gro`` files
    containing a one-line comment, so callers get stable output paths.

    Args:
        pdb_path: Structure that would be submitted.
        output_dir: Directory for the ITP/GRO files (created if needed).
        name: Basename used for both output files.

    Returns:
        Path (as str) of the placeholder ITP file.
    """
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
    """High-level flow: XYZ -> PDB -> LigParGen -> ITP/GRO.

    Converts the XYZ to PDB under ``<out_root>/temp_params/<symbol>/``
    and forwards it to :func:`submit_to_ligpargen` (currently the
    placeholder writer, see above).

    Args:
        xyz_path: Optimized monomer/oligomer geometry.
        out_root: Workflow output root; a temp_params subtree is made.
        symbol: Monomer symbol; names the subdirectory and files.

    Returns:
        Path (as str) of the resulting ITP file.
    """
    from .ase_utils import xyz_to_pdb

    temp_dir = Path(out_root) / "temp_params" / symbol
    temp_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = temp_dir / f"{symbol}.pdb"
    xyz_to_pdb(xyz_path, pdb_path)

    itp_path = submit_to_ligpargen(pdb_path, temp_dir, name=symbol)
    print(f"[Flow] Base parameterization complete for {symbol}: {itp_path}")
    return itp_path

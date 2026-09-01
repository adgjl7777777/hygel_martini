"""Extractor adapter: as-built composition from topology files alone.

Registers ``composition.from_top_itp``, which wraps
:meth:`..swelling.SwellingAnalyzer.composition_summary` for the
manifest runner.  Needs only ``top`` + ``itp`` (no MD output), so it is
the one analysis that can run on a freshly built, never-simulated
system.  The wrapped summary carries
``validation_role="composition_check_only"``: it reports the built-in
mass loading, not an equilibrium swelling observable.
"""
from __future__ import annotations
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("composition.from_top_itp")
class CompositionExtractor(BaseExtractor):
    """Compute the count-based loading ratio ``q_m`` from top/itp files.

    This is an as-built bookkeeping check; it cannot be compared
    directly against an experimental equilibrium swelling ratio (the
    wrapped result revokes direct comparison for that reason).
    """
    extractor_name = "composition.from_top_itp"
    required_inputs = ["top", "itp"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Build a SwellingAnalyzer from top/itp and return its summary.

        Args:
            inputs: ``top`` and ``itp`` file paths.
            params: Optional overrides — bead masses (amu), polymer
                bead volume (nm^3), polymer residue/atom names, and
                solvent molecule names (default MARTINI water ``W``).
        """
        from ..swelling import SwellingAnalyzer

        analyzer = SwellingAnalyzer.from_files(
            top_file=str(inputs["top"]),
            itp_file=str(inputs["itp"]),
            polymer_bead_mass=float(params.get("polymer_bead_mass", 45.0)),
            solvent_bead_mass=float(params.get("solvent_bead_mass", 72.0)),
            polymer_bead_vol_nm3=float(params.get("polymer_bead_vol_nm3", 0.065)),
            polymer_residue_name=params.get("polymer_residue_name"),
            polymer_atom_name=params.get("polymer_atom_name"),
            solvent_molecule_names=params.get("solvent_molecule_names", "W"),
        )
        return analyzer.composition_summary()

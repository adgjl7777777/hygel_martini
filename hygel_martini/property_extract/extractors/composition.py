from __future__ import annotations
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("composition.from_top_itp")
class CompositionExtractor(BaseExtractor):
    """
    top/itp로부터 count-based loading_qm을 계산한다.
    실험 equilibrium swelling ratio와 직접 비교할 수 없다.
    """
    extractor_name = "composition.from_top_itp"
    required_inputs = ["top", "itp"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
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

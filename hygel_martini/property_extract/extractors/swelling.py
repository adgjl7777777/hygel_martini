from __future__ import annotations
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("swelling.volume_from_energy")
class SwellingVolumeExtractor(BaseExtractor):
    """
    energy.xvg의 Volume 컬럼으로 polymer_volume_fraction을 계산한다.
    method: bead_volume 만 지원한다.
    """
    extractor_name = "swelling.volume_from_energy"
    required_inputs = ["top", "itp", "energy_xvg"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        method = params.get("method", "bead_volume")
        if method != "bead_volume":
            return PropertyResult.invalid_input(
                "polymer_volume_fraction",
                reason=f"method={method!r}는 지원하지 않습니다. 현재 구현은 bead_volume만 가능합니다.",
                validation_role="direct",
            )

        bead_vol = params.get("bead_volume_nm3")
        if bead_vol is None:
            return PropertyResult.invalid_input(
                "polymer_volume_fraction",
                reason="parameters.bead_volume_nm3가 없습니다.",
                validation_role="direct",
            )

        from ..swelling import SwellingAnalyzer

        analyzer = SwellingAnalyzer.from_files(
            top_file=str(inputs["top"]),
            itp_file=str(inputs["itp"]),
            polymer_bead_mass=float(params.get("polymer_bead_mass", 45.0)),
            solvent_bead_mass=float(params.get("solvent_bead_mass", 72.0)),
            polymer_bead_vol_nm3=float(bead_vol),
            polymer_residue_name=params.get("polymer_residue_name"),
            polymer_atom_name=params.get("polymer_atom_name"),
            solvent_molecule_names=params.get("solvent_molecule_names", "W"),
        )

        start_time_ps = float(params.get("start_time_ps", 0))
        try:
            return analyzer.analyze_trajectory(
                str(inputs["energy_xvg"]),
                start_time_ps=start_time_ps,
            )
        except ValueError as e:
            return PropertyResult.invalid_input(
                "polymer_volume_fraction",
                reason=str(e),
                inputs=[str(inputs["energy_xvg"])],
                validation_role="direct",
            )
        except Exception as e:
            return PropertyResult.analysis_failed(
                "polymer_volume_fraction",
                error=str(e),
                inputs=[str(inputs["energy_xvg"])],
                validation_role="direct",
            )

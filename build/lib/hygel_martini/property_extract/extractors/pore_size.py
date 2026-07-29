from __future__ import annotations
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("pore_size.nearest_surface_grid")
class PoreSizeGridExtractor(BaseExtractor):
    """
    single-frame .gro의 nearest-surface-grid로 peak pore size를 계산한다.
    validation_role=proxy: Poreblazer trajectory 결과와 방법론이 달라 직접 비교 불가.
    """
    extractor_name = "pore_size.nearest_surface_grid"
    required_inputs = ["gro"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        from ..pore_size import parse_gro_coords, get_peak_pore_size

        selection_residues = params.get("selection_residues") or ["PEO", "HYDROGEL"]
        grid_spacing = float(params.get("grid_spacing_nm", 0.2))
        bead_radius_raw = params.get("bead_radius_nm", 0.24)
        if isinstance(bead_radius_raw, dict):
            bead_radius = float(next(iter(bead_radius_raw.values()), 0.24))
        else:
            bead_radius = float(bead_radius_raw)
        bins = int(params.get("bins", 50))

        try:
            coords, box = parse_gro_coords(
                str(inputs["gro"]),
                selection_residues=selection_residues,
            )
        except ValueError as e:
            return PropertyResult.invalid_input(
                "pore_size_single_frame_grid",
                reason=str(e),
                inputs=[str(inputs["gro"])],
                validation_role="proxy",
                metadata={"target_aliases": ["pore_diameter_nm"]},
            )
        except Exception as e:
            return PropertyResult.analysis_failed(
                "pore_size_single_frame_grid",
                error=str(e),
                inputs=[str(inputs["gro"])],
                validation_role="proxy",
                metadata={"target_aliases": ["pore_diameter_nm"]},
            )

        if len(coords) == 0:
            return PropertyResult.invalid_input(
                "pore_size_single_frame_grid",
                reason="polymer atom 0개 선택됨 — selection_residues 확인 필요",
                inputs=[str(inputs["gro"])],
                validation_role="proxy",
                metadata={"target_aliases": ["pore_diameter_nm"]},
            )

        return get_peak_pore_size(
            coords, box,
            grid_spacing=grid_spacing,
            bead_radius=bead_radius,
            bins=bins,
        )

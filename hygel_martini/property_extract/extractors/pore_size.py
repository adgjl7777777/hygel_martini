"""Extractor adapter: single-frame grid pore size from a GRO file.

Registers ``pore_size.nearest_surface_grid``, which wraps
:func:`..pore_size.get_peak_pore_size` (nearest-surface-distance grid
histogram) for the manifest runner.  Gate behavior: GRO parse failures
and an empty polymer selection are reported as ``invalid_input``
instead of raising, and every result carries the
``pore_diameter_nm`` target alias so reporting can map it onto that
target name.  ``validation_role="proxy"``: the method differs from
Poreblazer-style trajectory analyses, so values are not directly
comparable to them or to experiment.
"""
from __future__ import annotations
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("pore_size.nearest_surface_grid")
class PoreSizeGridExtractor(BaseExtractor):
    """Peak pore size of one .gro frame via a nearest-surface grid.

    Proxy observable only — methodology differs from Poreblazer
    trajectory results, so no direct comparison is allowed.
    """
    extractor_name = "pore_size.nearest_surface_grid"
    required_inputs = ["gro"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Parse the frame, then histogram grid-to-surface distances.

        Args:
            inputs: ``gro`` — single-frame coordinate file.
            params: ``selection_residues`` (default PEO/HYDROGEL),
                ``grid_spacing_nm`` (default 0.2 nm),
                ``bead_radius_nm`` (scalar, or a per-type dict whose
                first value is used; default 0.24 nm), and histogram
                ``bins`` (default 50).

        Returns:
            PropertyResult for ``pore_size_single_frame_grid`` —
            computed, or invalid_input / analysis_failed (never
            raises).
        """
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

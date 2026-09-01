"""Extractor adapter: periodic local-clearance distribution on a frame.

Registers ``clearance.periodic_grid``, which wraps
:func:`..pore_size.calculate_periodic_clearance_distribution` for the
manifest runner.  Measures, on a periodic grid over one GRO frame, the
largest probe diameter locally admissible at each grid point, and
reports the median (p50) as the headline value.  Gate behavior: a
selection that matches zero obstacle beads refuses with
``invalid_input``.  ``validation_role="proxy"``: this is a probe-
admissible clearance/volume measure — not a unique pore, a
pore-limiting diameter, or an experimental mesh size (the metadata
carries that claim boundary verbatim).
"""
from __future__ import annotations

from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("clearance.periodic_grid")
class PeriodicClearanceExtractor(BaseExtractor):
    """Calculate definition-explicit local clearance on one periodic GRO frame."""

    extractor_name = "clearance.periodic_grid"
    required_inputs = ["gro"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Compute the clearance distribution and report its median.

        Args:
            inputs: ``gro`` — single-frame coordinate file with box.
            params: ``selection_residues`` (obstacle set, default
                PEO/HYDROGEL), ``bead_radius_nm`` (default 0.24 nm),
                ``grid_spacing_nm`` (default 0.2 nm),
                ``probe_radius_nm`` (default 0.1657 nm), histogram
                ``bins`` (default 50), and ``chunk_size`` (grid points
                per distance batch, memory knob).

        Returns:
            Computed PropertyResult ``local_clearance_diameter_p50_nm``
            (value in nm; 0.0 when the percentile list is empty);
            metadata keeps the full summary, the explicit obstacle
            definition, and the claim boundary.
        """
        from ..pore_size import (
            calculate_periodic_clearance_distribution,
            parse_gro_coords,
        )

        residues = params.get("selection_residues") or ["PEO", "HYDROGEL"]
        coordinates, box = parse_gro_coords(
            str(inputs["gro"]),
            selection_residues=residues,
        )
        if len(coordinates) == 0:
            return PropertyResult.invalid_input(
                "local_clearance_diameter_p50_nm",
                reason="selection_residues selected zero obstacle beads",
                inputs=[str(inputs["gro"])],
                validation_role="proxy",
            )
        _, _, summary = calculate_periodic_clearance_distribution(
            [(coordinates, float(params.get("bead_radius_nm", 0.24)))],
            box,
            grid_spacing=float(params.get("grid_spacing_nm", 0.2)),
            probe_radius=float(params.get("probe_radius_nm", 0.1657)),
            bins=int(params.get("bins", 50)),
            chunk_size=int(params.get("chunk_size", 250_000)),
        )
        percentiles = summary["local_clearance_diameter_percentiles_nm"]
        value = float(percentiles[2]) if percentiles else 0.0
        return PropertyResult(
            property="local_clearance_diameter_p50_nm",
            value=value,
            status="computed",
            direct_experiment_comparison_allowed=False,
            validation_role="proxy",
            metadata={
                **summary,
                "obstacle_definition": {
                    "selection_residues": list(residues),
                    "bead_radius_nm": float(params.get("bead_radius_nm", 0.24)),
                },
                "claim_boundary": (
                    "local clearance/probe-admissible volume; not a unique "
                    "pore, pore-limiting diameter, or experimental mesh size"
                ),
            },
        )

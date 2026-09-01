"""Extractor adapter: viscosity vs shear rate from an NEMD series.

Registers ``rheology.nemd_shear_rate``, which wraps
:func:`..rheology.analyze_shear_rate_viscosity` for the manifest
runner.  Unlike file-path extractors, its single input ``shear_dirs``
is a glob pattern over per-shear-rate run directories, so both gate
methods are overridden to test glob expansion instead of file
existence.  Gate behavior: an unexpanded pattern or any directory
missing its energy XVG refuses with ``missing_required_md``; a missing
``shear_rates_ps_inv`` parameter refuses with ``invalid_input``.
``validation_role="trend_only"``: absolute viscosities are not
comparable to experiment — only shear-thinning trends or formulation
rank order may be interpreted.
"""
from __future__ import annotations
import os
from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("rheology.nemd_shear_rate")
class NEMDShearRateExtractor(BaseExtractor):
    """Viscosity vs shear rate from a series of NEMD shear runs.

    ``shear_dirs`` must be a glob pattern that expands to the actual
    run directories (sorted order pairs them with
    ``shear_rates_ps_inv``).  Trend/rank-order use only; no absolute
    comparison with experimental viscosity.
    """
    extractor_name = "rheology.nemd_shear_rate"
    required_inputs = ["shear_dirs"]

    def can_compute(self, inputs: dict) -> bool:
        """Gate on glob expansion: True only if the pattern matches."""
        shear_dirs = inputs.get("shear_dirs")
        if not shear_dirs:
            return False
        import glob as _glob
        matches = _glob.glob(str(shear_dirs))
        return bool(matches)

    def missing_inputs_list(self, inputs: dict) -> list[str]:
        """Report the input key (unset) or the pattern (no matches)."""
        shear_dirs = inputs.get("shear_dirs")
        if not shear_dirs:
            return ["shear_dirs"]
        import glob as _glob
        if not _glob.glob(str(shear_dirs)):
            return [str(shear_dirs)]
        return []

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Average per-directory NEMD viscosities into one result.

        Args:
            inputs: ``shear_dirs`` glob pattern; matches are sorted so
                directory order must correspond to the declared rates.
            params: ``shear_rates_ps_inv`` (required, ps^-1, one per
                directory) and ``energy_xvg_name`` (default
                ``energy.xvg``) inside each directory.

        Returns:
            PropertyResult ``viscosity_vs_shear_rate`` whose value is
            the mean viscosity (Pa*s); metadata keeps the full curve
            (rates converted to s^-1, per-rate viscosities in Pa*s).
            Never raises — errors map to invalid_input /
            analysis_failed / missing statuses.
        """
        import glob as _glob
        import numpy as np
        from ..rheology import analyze_shear_rate_viscosity

        shear_dirs_pattern = str(inputs["shear_dirs"])
        dirs = sorted(_glob.glob(shear_dirs_pattern))
        if not dirs:
            return PropertyResult.missing(
                "viscosity_vs_shear_rate",
                missing_inputs=[shear_dirs_pattern],
                validation_role="trend_only",
            )

        edr_name = params.get("energy_xvg_name", "energy.xvg")
        shear_rates_ps_inv = params.get("shear_rates_ps_inv")
        if shear_rates_ps_inv is None:
            return PropertyResult.invalid_input(
                "viscosity_vs_shear_rate",
                reason="parameters.shear_rates_ps_inv가 없습니다.",
                validation_role="trend_only",
            )

        xvg_paths = [os.path.join(d, edr_name) for d in dirs]
        missing_xvg = [p for p in xvg_paths if not os.path.exists(p)]
        if missing_xvg:
            return PropertyResult.missing(
                "viscosity_vs_shear_rate",
                missing_inputs=missing_xvg,
                validation_role="trend_only",
            )

        try:
            viscosities = analyze_shear_rate_viscosity(
                xvg_paths,
                list(shear_rates_ps_inv),
            )
            shear_rates_s_inv = [sr * 1e12 for sr in shear_rates_ps_inv]
            return PropertyResult(
                property="viscosity_vs_shear_rate",
                value=float(np.mean(viscosities)),
                status="computed",
                direct_experiment_comparison_allowed=False,
                validation_role="trend_only",
                metadata={
                    "shear_rates_s_inv": shear_rates_s_inv,
                    "viscosities_Pa_s": viscosities.tolist(),
                    "note": "절댓값 비교 불가. shear-thinning trend 또는 formulation rank-order 비교용.",
                },
            )
        except ValueError as e:
            return PropertyResult.invalid_input(
                "viscosity_vs_shear_rate",
                reason=str(e),
                validation_role="trend_only",
            )
        except Exception as e:
            return PropertyResult.analysis_failed(
                "viscosity_vs_shear_rate",
                error=str(e),
                validation_role="trend_only",
            )

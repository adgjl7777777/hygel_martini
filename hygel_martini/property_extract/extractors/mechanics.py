"""Extractor adapter: paired-step apparent shear response from XVGs.

Registers ``mechanics.paired_step_xvg``, which wraps
:func:`..mechanics_analysis.paired_step_xvg_summary` for the manifest
runner.  Consumes three pre-extracted pressure-tensor XVGs (baseline,
+gamma step, -gamma step) and reports the antisymmetric stress
difference as an apparent shear response.  Gate behavior: any missing
mechanics parameter (component, gamma, analysis window) refuses with
``invalid_input`` before files are read.
``validation_role="finite_rate"``: a finite-rate step deformation, not
an equilibrium modulus — no direct experimental comparison.
"""
from __future__ import annotations

from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("mechanics.paired_step_xvg")
class PairedStepXVGExtractor(BaseExtractor):
    """Compute the registered apparent response from aligned +/- step XVGs."""

    extractor_name = "mechanics.paired_step_xvg"
    required_inputs = ["baseline_xvg", "positive_xvg", "negative_xvg"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Summarize the paired step and wrap it as a PropertyResult.

        Args:
            inputs: ``baseline_xvg``, ``positive_xvg``,
                ``negative_xvg`` — time-aligned pressure-tensor XVGs.
            params: Required — ``component`` (tensor column name),
                ``gamma`` (dimensionless applied strain),
                ``window_start_ps``/``window_end_ps`` (averaging
                window, ps).  Missing any of them refuses with
                invalid_input.

        Returns:
            Computed PropertyResult whose value is the mean apparent
            response in MPa; the full summary dict becomes metadata.
        """
        from ..mechanics_analysis import paired_step_xvg_summary

        required = ("component", "gamma", "window_start_ps", "window_end_ps")
        missing = [name for name in required if params.get(name) is None]
        if missing:
            return PropertyResult.invalid_input(
                "paired_step_finite_rate_apparent_shear_response",
                reason=f"missing mechanics parameters: {', '.join(missing)}",
                validation_role="finite_rate",
            )
        summary = paired_step_xvg_summary(
            str(inputs["baseline_xvg"]),
            str(inputs["positive_xvg"]),
            str(inputs["negative_xvg"]),
            component=str(params["component"]),
            gamma=float(params["gamma"]),
            window_start_ps=float(params["window_start_ps"]),
            window_end_ps=float(params["window_end_ps"]),
        )
        return PropertyResult(
            property="paired_step_finite_rate_apparent_shear_response",
            value=summary["apparent_response_mean_mpa"],
            status="computed",
            direct_experiment_comparison_allowed=False,
            validation_role="finite_rate",
            metadata=summary,
        )

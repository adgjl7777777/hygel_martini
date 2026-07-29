from __future__ import annotations

from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("mechanics.paired_step_xvg")
class PairedStepXVGExtractor(BaseExtractor):
    """Compute the registered apparent response from aligned +/- step XVGs."""

    extractor_name = "mechanics.paired_step_xvg"
    required_inputs = ["baseline_xvg", "positive_xvg", "negative_xvg"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
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

from __future__ import annotations

import json
from pathlib import Path

from hygel_martini.property_extract.analysis_jobs import run_analysis


def _write_pressure_xvg(path: Path, pressure: float, legend: str = "Pres-XY") -> None:
    path.write_text(
        f'@ s0 legend "{legend}"\n'
        + "\n".join(f"{time} {pressure}" for time in range(6))
        + "\n"
    )


def test_requirements_gate_and_json_report(tmp_path: Path) -> None:
    _write_pressure_xvg(tmp_path / "base.xvg", 10.0)
    _write_pressure_xvg(tmp_path / "plus.xvg", -210.0)
    _write_pressure_xvg(tmp_path / "minus.xvg", 230.0)
    (tmp_path / "analysis.yaml").write_text(
        """
schema_version: 1
template: false
analysis_jobs:
  mechanics:
    property: paired_step_finite_rate_apparent_shear_response
    extractor: mechanics.paired_step_xvg
    inputs:
      baseline_xvg: base.xvg
      positive_xvg: plus.xvg
      negative_xvg: minus.xvg
    parameters:
      component: Pres-XY
      gamma: 0.01
      window_start_ps: 1
      window_end_ps: 5
    output:
      report: results/mechanics.json
"""
    )
    (tmp_path / "requirements.yaml").write_text(
        """
schema_version: 1
property_requirements:
  paired_step_finite_rate_apparent_shear_response:
    md_required: true
    validation_role: finite_rate
    required_inputs: [baseline_xvg, positive_xvg, negative_xvg]
    required_outputs: []
    required_columns: [Pres-XY]
    required_md_jobs: [paired_step]
"""
    )

    results = run_analysis(
        str(tmp_path / "analysis.yaml"),
        str(tmp_path / "requirements.yaml"),
    )
    result = results["mechanics"]
    assert result.status == "computed"
    assert result.value == 2200.0
    report = json.loads((tmp_path / "results/mechanics.json").read_text())
    assert report["validation_role"] == "finite_rate"
    assert report["value"] == 2200.0

    _write_pressure_xvg(tmp_path / "minus.xvg", 230.0, legend="Pres-XZ")
    blocked = run_analysis(
        str(tmp_path / "analysis.yaml"),
        str(tmp_path / "requirements.yaml"),
    )["mechanics"]
    assert blocked.status == "missing_required_md"
    assert "missing columns" in blocked.missing_required_inputs[0]

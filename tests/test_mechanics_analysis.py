from __future__ import annotations

from pathlib import Path

import numpy as np

from hygel_martini.property_extract.mechanics_analysis import (
    analyze_cycle_blocks,
    analyze_paired_ramp,
    holm_adjust,
    paired_step_window_summary,
    paired_step_xvg_summary,
    summarize_equal_realizations,
)


def test_paired_step_window_and_xvg_have_expected_sign(tmp_path: Path) -> None:
    time = np.arange(0.0, 6.0)
    baseline = np.full_like(time, 10.0)
    positive = np.full_like(time, -210.0)
    negative = np.full_like(time, 230.0)
    summary = paired_step_window_summary(
        time,
        baseline,
        positive,
        negative,
        gamma=0.01,
        window_start_ps=1.0,
        window_end_ps=5.0,
    )
    assert summary["apparent_response_mean_mpa"] == 2200.0
    assert summary["even_to_odd_abs_ratio"] == 0.0

    paths = []
    for name, values in (
        ("base", baseline),
        ("plus", positive),
        ("minus", negative),
    ):
        path = tmp_path / f"{name}.xvg"
        rows = "\n".join(f"{t:g} {value:g}" for t, value in zip(time, values))
        path.write_text(f'@ s0 legend "Pres-XY"\n{rows}\n')
        paths.append(path)
    from_xvg = paired_step_xvg_summary(
        *paths,
        component="Pres-XY",
        gamma=0.01,
        window_start_ps=1.0,
        window_end_ps=5.0,
    )
    assert from_xvg["apparent_response_mean_mpa"] == 2200.0


def test_paired_ramp_and_cycle_block_summaries() -> None:
    time = np.linspace(0.0, 100.0, 101)
    target = 0.04
    strain_fraction = time / time[-1]
    plus = {
        "time_ps": time,
        "temperature_k": np.full_like(time, 310.0),
        "stress_mpa": 100.0 * target * strain_fraction,
    }
    minus = {
        "time_ps": time,
        "temperature_k": np.full_like(time, 310.0),
        "stress_mpa": -100.0 * target * strain_fraction,
    }
    points, blocks, ramp = analyze_paired_ramp(
        plus,
        minus,
        target_amplitude=target,
        ramp_ps=100.0,
        target_temperature_k=310.0,
    )
    assert len(points) == 101
    assert len(blocks) == 5
    assert ramp["ramp_signal_gate_pass"] is True

    gp = np.tile(np.array([-0.5, 0.5, -0.5, 0.5, 0.0]), 6)
    gpp = np.full(30, 20.0)
    block_rows, cycle = analyze_cycle_blocks(
        gp,
        gpp,
        block_size_cycles=5,
        minimum_blocks=6,
        bootstrap_samples=1_000,
    )
    assert len(block_rows) == 6
    assert cycle["g_prime_mean_mpa"] == 0.0
    assert cycle["g_double_prime_mean_mpa"] == 20.0


def test_realization_weighting_and_holm_adjustment() -> None:
    summary = summarize_equal_realizations(
        {
            "seed101": [100.0, 120.0],
            "seed202": [110.0],
            "seed303": [90.0, 110.0, 130.0],
        }
    )
    assert summary["equal_weight_mean"] == 110.0
    assert summary["realization_sample_sd"] == 0.0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

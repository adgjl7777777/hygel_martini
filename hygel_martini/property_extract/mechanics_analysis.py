"""Reusable finite-rate mechanics analysis with explicit claim boundaries.

The functions in this module reproduce the paired-step, paired-ramp, and
block-uncertainty calculations used by the Series-01 validation workflow.
They return finite-rate apparent responses. They do not relabel those
responses as equilibrium, plateau, storage, zero-frequency, or experimental
moduli.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .mechanics import paired_step_shear_response
from .timeseries import read_xvg


DEFAULT_RAMP_FRACTION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def read_labeled_xvg(path: str | Path) -> dict[str, np.ndarray]:
    """Read a GROMACS XVG into named columns, requiring complete legends."""
    legends, values = read_xvg(path)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected time plus at least one value column")
    if len(legends) != values.shape[1] - 1:
        raise ValueError(
            f"{path}: expected {values.shape[1] - 1} legends, found {len(legends)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: nonfinite XVG value")
    return {
        "Time": values[:, 0],
        **{
            legend: values[:, index + 1]
            for index, legend in enumerate(legends)
        },
    }


def paired_step_window_summary(
    time_ps: np.ndarray,
    baseline_pressure_bar: np.ndarray,
    positive_pressure_bar: np.ndarray,
    negative_pressure_bar: np.ndarray,
    *,
    gamma: float,
    window_start_ps: float,
    window_end_ps: float,
) -> dict[str, object]:
    """Summarize a matched +/- step-shear response over a registered window."""
    time = np.asarray(time_ps, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) < 0):
        raise ValueError("time_ps must be a monotonic one-dimensional series")
    if window_end_ps < window_start_ps:
        raise ValueError("window_end_ps must be at least window_start_ps")
    response = paired_step_shear_response(
        baseline_pressure_bar,
        positive_pressure_bar,
        negative_pressure_bar,
        gamma,
    )
    if any(np.asarray(value).shape != time.shape for value in response.values()):
        raise ValueError("all pressure series must match time_ps")
    mask = (time >= window_start_ps) & (time <= window_end_ps)
    if np.count_nonzero(mask) < 1:
        raise ValueError("registered mechanics window contains no samples")
    odd = np.asarray(response["odd_pressure"])[mask]
    even = np.asarray(response["even_residual_pressure"])[mask]
    modulus = np.asarray(response["apparent_modulus"])[mask]
    odd_scale = float(np.mean(np.abs(odd)))
    return {
        "observable": "paired_step_finite_rate_apparent_shear_response",
        "gamma": float(gamma),
        "window_start_ps": float(window_start_ps),
        "window_end_ps": float(window_end_ps),
        "n_samples": int(np.count_nonzero(mask)),
        "apparent_response_mean_mpa": float(np.mean(modulus)),
        "apparent_response_sample_sd_mpa": (
            float(np.std(modulus, ddof=1)) if modulus.size > 1 else 0.0
        ),
        "apparent_response_first_sample_mpa": float(modulus[0]),
        "elastic_first_sample_sign_positive": bool(modulus[0] > 0),
        "odd_pressure_abs_mean_bar": odd_scale,
        "even_residual_mean_bar": float(np.mean(even)),
        "even_to_odd_abs_ratio": (
            float(np.mean(np.abs(even)) / odd_scale)
            if odd_scale > 0
            else math.inf
        ),
        "claim_boundary": (
            "finite-rate apparent response; not equilibrium, plateau, storage, "
            "zero-frequency, or experimental modulus"
        ),
    }


def paired_step_xvg_summary(
    baseline_xvg: str | Path,
    positive_xvg: str | Path,
    negative_xvg: str | Path,
    *,
    component: str,
    gamma: float,
    window_start_ps: float,
    window_end_ps: float,
) -> dict[str, object]:
    """Read three aligned labeled XVGs and summarize one pressure component."""
    sources = [
        read_labeled_xvg(path)
        for path in (baseline_xvg, positive_xvg, negative_xvg)
    ]
    for path, data in zip(
        (baseline_xvg, positive_xvg, negative_xvg), sources, strict=True
    ):
        if component not in data:
            raise ValueError(f"{path}: missing pressure component {component!r}")
    time = sources[0]["Time"]
    if any(
        len(data["Time"]) != len(time)
        or not np.allclose(data["Time"], time, atol=1.0e-8, rtol=0.0)
        for data in sources[1:]
    ):
        raise ValueError("baseline, positive, and negative XVG times do not align")
    result = paired_step_window_summary(
        time,
        sources[0][component],
        sources[1][component],
        sources[2][component],
        gamma=gamma,
        window_start_ps=window_start_ps,
        window_end_ps=window_end_ps,
    )
    result.update(
        {
            "component": component,
            "baseline_xvg": str(Path(baseline_xvg)),
            "positive_xvg": str(Path(positive_xvg)),
            "negative_xvg": str(Path(negative_xvg)),
        }
    )
    return result


def analyze_paired_ramp(
    plus: Mapping[str, np.ndarray],
    minus: Mapping[str, np.ndarray],
    *,
    target_amplitude: float,
    ramp_ps: float,
    target_temperature_k: float,
    fraction_edges: Sequence[float] = DEFAULT_RAMP_FRACTION_EDGES,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, object]]:
    """Analyze aligned +/- finite ramps using block means along the ramp."""
    edges = tuple(float(value) for value in fraction_edges)
    if target_amplitude <= 0 or ramp_ps <= 0:
        raise ValueError("amplitude and ramp duration must be positive")
    if len(edges) < 3 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("fraction edges must span zero to one")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("fraction edges must be strictly increasing")

    for name, series in (("plus", plus), ("minus", minus)):
        for key in ("time_ps", "temperature_k", "stress_mpa"):
            values = np.asarray(series[key], dtype=float)
            if values.ndim != 1 or len(values) < 6:
                raise ValueError(f"{name} {key} is not a usable series")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} {key} contains nonfinite values")
    time = np.asarray(plus["time_ps"], dtype=float)
    minus_time = np.asarray(minus["time_ps"], dtype=float)
    if len(time) != len(minus_time) or not np.allclose(
        time, minus_time, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("positive and negative ramp times do not align")
    if np.any(np.diff(time) <= 0):
        raise ValueError("ramp time must be strictly increasing")
    if not math.isclose(time[-1], ramp_ps, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"observed ramp end {time[-1]} ps differs from {ramp_ps} ps")

    fraction = time / ramp_ps
    plus_stress = np.asarray(plus["stress_mpa"], dtype=float)
    minus_stress = np.asarray(minus["stress_mpa"], dtype=float)
    paired_stress = 0.5 * (plus_stress - minus_stress)
    symmetric_stress = 0.5 * (plus_stress + minus_stress)
    normalized_response = paired_stress / target_amplitude

    points = [
        {
            "time_ps": float(time[index]),
            "ramp_fraction": float(fraction[index]),
            "stress_plus_mpa": float(plus_stress[index]),
            "stress_minus_mpa": float(minus_stress[index]),
            "paired_stress_mpa": float(paired_stress[index]),
            "symmetric_stress_mpa": float(symmetric_stress[index]),
            "paired_stress_per_final_strain_mpa": float(
                normalized_response[index]
            ),
            "temperature_plus_k": float(plus["temperature_k"][index]),
            "temperature_minus_k": float(minus["temperature_k"][index]),
        }
        for index in range(len(time))
    ]

    blocks: list[dict[str, float]] = []
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        mask = (
            (fraction >= start)
            & (
                fraction <= end + 1.0e-12
                if index == len(edges) - 2
                else fraction < end
            )
        )
        if int(mask.sum()) < 3:
            raise ValueError(f"ramp block {start}-{end} has fewer than 3 samples")
        values = normalized_response[mask]
        blocks.append(
            {
                "block": float(index),
                "start_fraction": start,
                "end_fraction": end,
                "start_ps": start * ramp_ps,
                "end_ps": end * ramp_ps,
                "samples": float(mask.sum()),
                "paired_stress_mean_mpa": float(np.mean(paired_stress[mask])),
                "paired_stress_std_mpa": float(
                    np.std(paired_stress[mask], ddof=1)
                ),
                "normalized_mean_mpa": float(np.mean(values)),
                "normalized_std_mpa": float(np.std(values, ddof=1)),
            }
        )

    active = np.asarray(
        [
            row["normalized_mean_mpa"]
            for row in blocks
            if row["start_fraction"] >= 0.2
        ],
        dtype=float,
    )
    active_mean = float(np.mean(active))
    active_sem = float(np.std(active, ddof=1) / math.sqrt(len(active)))
    active_snr = abs(active_mean) / active_sem if active_sem > 0 else math.inf
    temperatures = np.concatenate(
        (
            np.asarray(plus["temperature_k"], dtype=float),
            np.asarray(minus["temperature_k"], dtype=float),
        )
    )
    checks = {
        "ramp_signal_block_snr_at_least_2": active_snr >= 2.0,
        "temperature_mean_error_at_most_1K": (
            abs(float(np.mean(temperatures)) - target_temperature_k) <= 1.0
        ),
        "temperature_max_deviation_at_most_20K": (
            float(np.max(np.abs(temperatures - target_temperature_k))) <= 20.0
        ),
    }
    summary: dict[str, object] = {
        "observable": "paired_ramp_finite_rate_apparent_shear_response",
        "target_amplitude": target_amplitude,
        "ramp_ps": ramp_ps,
        "target_temperature_k": target_temperature_k,
        "samples_per_branch": len(time),
        "fraction_edges": list(edges),
        "active_fraction_window": [0.2, 1.0],
        "active_block_normalized_means_mpa": active.tolist(),
        "active_normalized_mean_mpa": active_mean,
        "active_block_sem_mpa": active_sem,
        "active_block_snr": active_snr,
        "temperature_mean_k": float(np.mean(temperatures)),
        "temperature_min_k": float(np.min(temperatures)),
        "temperature_max_k": float(np.max(temperatures)),
        "checks": checks,
        "ramp_signal_gate_pass": all(checks.values()),
        "claim_boundary": (
            "finite-rate ramp response and resolution gate; not equilibrium "
            "or experimental-frequency rheology"
        ),
    }
    return points, blocks, summary


def analyze_cycle_blocks(
    g_prime_mpa: np.ndarray,
    g_double_prime_mpa: np.ndarray,
    *,
    block_size_cycles: int = 5,
    minimum_blocks: int = 6,
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 20_260_728,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Estimate periodic-response uncertainty from contiguous cycle blocks."""
    gp = np.asarray(g_prime_mpa, dtype=float)
    gpp = np.asarray(g_double_prime_mpa, dtype=float)
    if gp.ndim != 1 or gpp.ndim != 1 or len(gp) != len(gpp):
        raise ValueError("G' and G'' must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(gp)) or not np.all(np.isfinite(gpp)):
        raise ValueError("nonfinite modulus input")
    if block_size_cycles <= 0 or minimum_blocks < 2:
        raise ValueError("invalid block configuration")
    if bootstrap_samples < 1_000:
        raise ValueError("bootstrap_samples must be at least 1000")
    if len(gp) % block_size_cycles:
        raise ValueError("retained cycle count must be divisible by block size")
    n_blocks = len(gp) // block_size_cycles
    if n_blocks < minimum_blocks:
        raise ValueError(
            f"need at least {minimum_blocks} blocks, observed {n_blocks}"
        )

    gp_blocks = gp.reshape(n_blocks, block_size_cycles).mean(axis=1)
    gpp_blocks = gpp.reshape(n_blocks, block_size_cycles).mean(axis=1)
    complex_blocks = gp_blocks + 1j * gpp_blocks
    rows = [
        {
            "block": float(index),
            "first_retained_cycle_offset": float(index * block_size_cycles),
            "last_retained_cycle_offset": float(
                (index + 1) * block_size_cycles - 1
            ),
            "g_prime_mean_mpa": float(block_gp),
            "g_double_prime_mean_mpa": float(block_gpp),
            "g_abs_of_complex_mean_mpa": float(math.hypot(block_gp, block_gpp)),
        }
        for index, (block_gp, block_gpp) in enumerate(
            zip(gp_blocks, gpp_blocks, strict=True)
        )
    ]

    mean_gp = float(np.mean(gp_blocks))
    mean_gpp = float(np.mean(gpp_blocks))
    mean_abs = float(math.hypot(mean_gp, mean_gpp))
    elastic_fraction = abs(mean_gp) / mean_abs if mean_abs > 0 else math.inf
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n_blocks, size=(bootstrap_samples, n_blocks))
    boot_gp = gp_blocks[indices].mean(axis=1)
    boot_gpp = gpp_blocks[indices].mean(axis=1)
    boot_abs = np.hypot(boot_gp, boot_gpp)
    boot_elastic_fraction = np.divide(
        np.abs(boot_gp),
        boot_abs,
        out=np.full_like(boot_abs, np.inf),
        where=boot_abs > 0,
    )
    gp_ci = tuple(float(value) for value in np.percentile(boot_gp, [2.5, 97.5]))
    gpp_ci = tuple(
        float(value) for value in np.percentile(boot_gpp, [2.5, 97.5])
    )
    elastic_fraction_upper = float(np.percentile(boot_elastic_fraction, 95.0))
    half = n_blocks // 2
    first_complex = complex(
        float(np.mean(gp_blocks[:half])), float(np.mean(gpp_blocks[:half]))
    )
    last_complex = complex(
        float(np.mean(gp_blocks[-half:])), float(np.mean(gpp_blocks[-half:]))
    )
    half_drift = (
        abs(last_complex - first_complex) / mean_abs if mean_abs > 0 else math.inf
    )
    gpp_relative_ci_halfwidth = (
        (gpp_ci[1] - gpp_ci[0]) / (2.0 * abs(mean_gpp))
        if abs(mean_gpp) > 0
        else math.inf
    )
    block_abs = np.abs(complex_blocks)
    checks = {
        "enough_blocks": n_blocks >= minimum_blocks,
        "point_elastic_fraction_at_most_0p10": elastic_fraction <= 0.10,
        "g_prime_95ci_contains_zero": gp_ci[0] <= 0.0 <= gp_ci[1],
        "elastic_fraction_95pct_upper_at_most_0p20": (
            elastic_fraction_upper <= 0.20
        ),
        "g_double_prime_95ci_lower_positive": gpp_ci[0] > 0.0,
        "g_double_prime_relative_95ci_halfwidth_at_most_0p30": (
            gpp_relative_ci_halfwidth <= 0.30
        ),
        "first_last_half_complex_drift_at_most_0p30": half_drift <= 0.30,
    }
    summary: dict[str, object] = {
        "observable": "periodic_finite_rate_complex_response",
        "retained_cycles": len(gp),
        "block_size_cycles": block_size_cycles,
        "blocks": n_blocks,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "g_prime_mean_mpa": mean_gp,
        "g_prime_bootstrap_95ci_mpa": list(gp_ci),
        "g_double_prime_mean_mpa": mean_gpp,
        "g_double_prime_bootstrap_95ci_mpa": list(gpp_ci),
        "g_abs_of_complex_mean_mpa": mean_abs,
        "point_elastic_fraction": elastic_fraction,
        "elastic_fraction_bootstrap_95pct_upper": elastic_fraction_upper,
        "g_double_prime_relative_95ci_halfwidth": gpp_relative_ci_halfwidth,
        "block_g_abs_cv": float(
            np.std(block_abs, ddof=1) / np.mean(block_abs)
        ),
        "first_last_half_complex_relative_drift": half_drift,
        "checks": checks,
        "overall_block_gate_pass": all(checks.values()),
        "claim_boundary": (
            "finite-rate periodic response; direct experimental-frequency "
            "comparison requires a method-matched protocol"
        ),
    }
    return rows, summary


def summarize_equal_realizations(
    realization_values: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Give each realization equal weight after averaging within realization."""
    if len(realization_values) < 2:
        raise ValueError("at least two realizations are required")
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, raw in realization_values.items():
        values = np.asarray(raw, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"realization {name!r} must contain finite scalars")
        means[str(name)] = float(np.mean(values))
        counts[str(name)] = int(values.size)
    network_means = np.asarray(list(means.values()), dtype=float)
    return {
        "statistical_unit": "realization",
        "n_realizations": int(len(network_means)),
        "within_realization_counts": counts,
        "realization_means": means,
        "equal_weight_mean": float(np.mean(network_means)),
        "realization_sample_sd": float(np.std(network_means, ddof=1)),
        "realization_sem": float(
            np.std(network_means, ddof=1) / math.sqrt(len(network_means))
        ),
        "claim_boundary": (
            "within-realization samples are not counted as independent "
            "network realizations"
        ),
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be finite values in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()

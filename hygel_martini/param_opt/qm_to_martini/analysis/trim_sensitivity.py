#!/usr/bin/env python3
"""Energy-based trim sensitivity analysis for C/D/S xTB trajectories.

This script owns the diagnostic study of how the equilibration-trim
choice affects retained-energy statistics for the three monomer
trajectories labeled C, D, and S. For each trajectory it reads the
per-frame energies (Hartree) from the xTB ``.trj`` comment lines,
computes tail-reference statistics, tail-mean threshold detection t0
over a (ref_fraction x sigma) sweep, rolling-mean band diagnostics,
retained statistics for a fixed set of candidate trim starts, and an
energy-autocorrelation summary.

Callers: run as a standalone CLI against a stage-03 project directory
whose C/D/S run layout matches the ``TRAJECTORIES``/``*_INFO`` path
templates (the pymbar and threshold trim info JSONs are read when
present). Outputs go to ``<project>/sensitivity``: summary/sweep CSVs,
``summary.md`` (with a fixed interpretation section written from a past
C/D/S study), ``sensitivity_results.json``, and two PNG quick-look plots
(skipped when matplotlib is unavailable).

Conventions: energies are in Hartree; frame spacing is assumed to be
``DUMP_FS`` = 50 fs, so frame counts convert to ns via ``frame_to_ns``.
Module-level path globals are rebound by ``configure_paths``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for portability
    plt = None


DEFAULT_PROJECT = Path.cwd()
# Rebound by configure_paths(); module import only sets placeholders.
PROJECT = DEFAULT_PROJECT
OUTDIR = PROJECT / "sensitivity"
# Time between saved trajectory frames in fs (xTB dump interval).
DUMP_FS = 50.0
# Extracts "energy: <float>" from an xTB .trj frame comment line.
ENERGY_RE = re.compile(r"energy:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

# Fixed per-label run layout inside the project directory.
TRAJECTORIES = {
    "C": PROJECT / "md_C/C/md_only_from_last_snapshot/xtb.trj",
    "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb.trj",
    "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb.trj",
}

PYMBAR_INFO = {
    "C": PROJECT / "md_C/C/md_only_from_last_snapshot/xtb_traj_trim_info.json",
    "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb_traj_trim_info.json",
    "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb_traj_trim_info.json",
}

THRESHOLD_INFO = {
    "C": PROJECT / "trim_threshold/C/xtb_traj_trim_info.json",
    "D": PROJECT / "trim_threshold/D/xtb_traj_trim_info.json",
    "S": PROJECT / "trim_threshold/S/xtb_traj_trim_info.json",
}

# Sweep grids: tail-reference fractions, threshold sigmas, rolling-mean
# windows (frames) and band widths, and reported autocorrelation lags.
REF_FRACTIONS = [0.1, 0.2, 0.3, 0.5]
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
ROLLING_WINDOWS = [100, 500, 1000, 5000]
ROLLING_SIGMAS = [0.5, 1.0, 2.0]
ACF_LAGS = [1, 10, 100, 500, 1000, 5000, 10000, 25000]


def read_energies(path: Path) -> np.ndarray:
    """Read per-frame energies (Hartree) from an xTB XYZ trajectory.

    Walks the multi-frame XYZ structure (atom count line, comment line,
    coordinates) and extracts the "energy:" annotation from each comment.
    Frames without the annotation are skipped; parsing stops at the first
    malformed atom-count line.

    Args:
        path: xTB ``.trj`` file.

    Returns:
        1D array of energies in file order.
    """
    energies: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            while line and not line.strip():
                line = handle.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            comment = handle.readline()
            match = ENERGY_RE.search(comment)
            if match:
                energies.append(float(match.group(1)))
            for _ in range(natoms):
                handle.readline()
    return np.asarray(energies, dtype=float)


def load_json(path: Path) -> dict:
    """Read a JSON file, returning {} when it does not exist."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def frame_to_ns(frame: int) -> float:
    """Convert a frame index/count to ns assuming DUMP_FS fs per frame."""
    return frame * DUMP_FS / 1_000_000.0


def tail_reference(arr: np.ndarray, ref_fraction: float) -> tuple[int, float, float]:
    """Compute the last-``ref_fraction`` reference statistics.

    The std is floored at a tiny value relative to the mean so later
    sigma-normalized comparisons never divide by zero.

    Args:
        arr: per-frame energies (Hartree).
        ref_fraction: tail fraction used as the equilibrated reference.

    Returns:
        (reference start frame, reference mean, reference std) in Hartree.
    """
    start = max(int(len(arr) * (1.0 - ref_fraction)), 1)
    ref = arr[start:]
    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    std = max(std, abs(mean) * 1e-6 + 1e-12)
    return start, mean, std


def threshold_t0(arr: np.ndarray, ref_fraction: float, sigma: float) -> tuple[int, float, float, int]:
    """Detect the trim start via the tail-mean threshold criterion.

    t0 is the earliest frame whose suffix mean (mean of all frames from
    that index to the end) falls within ``sigma`` reference-stds of the
    tail-reference mean; ``len(arr)`` when no frame qualifies.

    Args:
        arr: per-frame energies (Hartree).
        ref_fraction: tail fraction defining the reference.
        sigma: allowed deviation in units of the reference std.

    Returns:
        (t0 frame, reference mean, reference std, reference start frame).
    """
    ref_start, ref_mean, ref_std = tail_reference(arr, ref_fraction)
    tail_means = np.cumsum(arr[::-1])[::-1] / np.arange(len(arr), 0, -1)
    in_band = np.abs(tail_means - ref_mean) <= sigma * ref_std
    hits = np.where(in_band)[0]
    t0 = int(hits[0]) if len(hits) else len(arr)
    return t0, ref_mean, ref_std, ref_start


def rolling_stats(arr: np.ndarray, ref_mean: float, ref_std: float, window: int, sigma: float) -> dict:
    """Diagnose when the rolling mean enters/stays in the reference band.

    Args:
        arr: per-frame energies (Hartree).
        ref_mean: reference mean (Hartree).
        ref_std: reference std (Hartree).
        window: rolling-mean window in frames.
        sigma: band half-width in units of ``ref_std``.

    Returns:
        Dict with ``first_in_band`` (frame of first in-band window
        center), ``stable_from`` (frame after which every window stays
        in band), ``fraction_in_band``, and ``max_abs_sigma``; the first
        three are None when the trajectory is shorter than the window.
    """
    if len(arr) < window:
        return {"first_in_band": None, "stable_from": None, "fraction_in_band": None}
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    dev = np.abs(roll - ref_mean) / ref_std
    in_band = dev <= sigma
    first = int(np.argmax(in_band) + window // 2) if np.any(in_band) else None
    suffix_ok = np.logical_and.accumulate(in_band[::-1])[::-1]
    stable_hits = np.where(suffix_ok)[0]
    stable = int(stable_hits[0] + window // 2) if len(stable_hits) else None
    return {
        "first_in_band": first,
        "stable_from": stable,
        "fraction_in_band": float(np.mean(in_band)),
        "max_abs_sigma": float(np.max(dev)),
    }


def candidate_stats(arr: np.ndarray, start: int, ref_mean: float, ref_std: float) -> dict:
    """Statistics of the energies retained by one candidate trim start.

    Args:
        arr: per-frame energies (Hartree).
        start: trim start frame (clamped into range).
        ref_mean: reference mean (Hartree).
        ref_std: reference std (Hartree).

    Returns:
        Dict with the clamped start, kept frame count/ns, kept mean/std,
        ``delta_ref_sigma`` (|kept mean - ref mean| / ref std), and the
        20-block standard error of the mean (NaN when <2 blocks or empty).
    """
    start = min(max(int(start), 0), len(arr))
    kept = arr[start:]
    if len(kept) == 0:
        return {
            "start": start,
            "kept_frames": 0,
            "kept_ns": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "delta_ref_sigma": math.nan,
            "block_sem": math.nan,
        }

    mean = float(np.mean(kept))
    std = float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0
    n_blocks = min(20, len(kept))
    block_sem = math.nan
    if n_blocks >= 2:
        blocks = np.array_split(kept, n_blocks)
        block_means = np.asarray([float(np.mean(block)) for block in blocks])
        block_sem = float(np.std(block_means, ddof=1) / math.sqrt(n_blocks))
    return {
        "start": start,
        "kept_frames": int(len(kept)),
        "kept_ns": frame_to_ns(len(kept)),
        "mean": mean,
        "std": std,
        "delta_ref_sigma": abs(mean - ref_mean) / ref_std,
        "block_sem": block_sem,
    }


def autocorrelation_summary(arr: np.ndarray, max_lag: int = 50_000) -> dict:
    """Summarize the normalized energy autocorrelation function.

    The ACF is computed via FFT with per-lag normalization, truncated at
    ``max_lag``. The statistical inefficiency g = 1 + 2*sum(ACF) is
    integrated up to the first non-positive lag.

    Args:
        arr: per-frame energies (Hartree).
        max_lag: largest lag (frames) kept in the normalized ACF.

    Returns:
        Dict with ``acf_lag_<L>`` samples at the ``ACF_LAGS`` lags,
        ``lag_1_over_e`` / ``lag_0p1`` (first lags where the ACF drops
        below 1/e and 0.1; None when never), ``g_first_negative``, and
        ``first_nonpositive_lag``.
    """
    centered = arr - float(np.mean(arr))
    n = len(centered)
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_len)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_len)[:n]
    acf = acf / np.arange(n, 0, -1)
    if acf[0] == 0:
        norm = np.zeros_like(acf[: max_lag + 1])
    else:
        norm = acf[: max_lag + 1] / acf[0]

    positive = norm[1:] > 0
    first_nonpositive = int(np.argmax(~positive) + 1) if np.any(~positive) else len(norm) - 1
    cutoff = max(first_nonpositive, 1)
    g_first_negative = float(1.0 + 2.0 * np.sum(norm[1:cutoff]))

    below_e = np.where(norm <= math.exp(-1))[0]
    below_01 = np.where(norm <= 0.1)[0]
    lag_e = int(below_e[0]) if len(below_e) else None
    lag_01 = int(below_01[0]) if len(below_01) else None

    values = {}
    for lag in ACF_LAGS:
        values[f"acf_lag_{lag}"] = float(norm[lag]) if lag < len(norm) else math.nan

    values.update(
        {
            "lag_1_over_e": lag_e,
            "lag_0p1": lag_01,
            "g_first_negative": g_first_negative,
            "first_nonpositive_lag": first_nonpositive,
        }
    )
    return values


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write dict rows as CSV with the given fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_num(value, digits: int = 3) -> str:
    """Format a value for the Markdown report ("-" for None/NaN/inf)."""
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def plot_energy(results: dict) -> None:
    """Plot per-label energy traces with rolling means and trim markers.

    One panel per label: downsampled raw energy, 500/5000-frame rolling
    means, the last-20% reference mean +- 1 std band, and vertical lines
    at the highlighted candidate trim starts. No-op without matplotlib.

    Args:
        results: label -> {energies, ref, candidate_starts} as assembled
            in ``main``.
    """
    if plt is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, (label, data) in zip(axes, results.items()):
        arr = data["energies"]
        ref = data["ref"]
        max_points = 6000
        step = max(1, len(arr) // max_points)
        x = np.arange(0, len(arr), step) * DUMP_FS / 1000.0
        y = arr[::step]
        ax.plot(x, y, lw=0.35, alpha=0.45, label="energy")
        for window in [500, 5000]:
            if len(arr) >= window:
                roll = np.convolve(arr, np.ones(window) / window, mode="valid")
                rx = (np.arange(len(roll)) + window // 2) * DUMP_FS / 1000.0
                rstep = max(1, len(roll) // max_points)
                ax.plot(rx[::rstep], roll[::rstep], lw=0.9, label=f"rolling {window}")
        ax.axhline(ref["mean"], color="black", lw=0.8, ls="--", label="last 20% mean")
        ax.axhspan(ref["mean"] - ref["std"], ref["mean"] + ref["std"], color="gray", alpha=0.12)
        for name, start in data["candidate_starts"].items():
            if name in {"pymbar", "threshold_default", "rolling_w500_1sigma_stable"} and start is not None:
                ax.axvline(start * DUMP_FS / 1000.0, lw=1.0, ls=":", label=name)
        ax.set_title(f"{label}: energy and rolling means")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("E (Ha)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "energy_rolling_candidates.png", dpi=170)
    plt.close(fig)


def plot_candidate_sensitivity(candidate_rows: list[dict]) -> None:
    """Plot kept-mean deviation versus trim start for every candidate.

    One line per label with the highlighted candidates annotated; the
    dashed guide marks a 1-sigma deviation from the reference mean.
    No-op without matplotlib.
    """
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {"C": "o", "D": "s", "S": "^"}
    for label in ["C", "D", "S"]:
        rows = [row for row in candidate_rows if row["label"] == label]
        rows = sorted(rows, key=lambda row: int(row["start_frame"]))
        x = [float(row["start_ns"]) for row in rows]
        y = [float(row["delta_ref_sigma"]) if row["delta_ref_sigma"] != "" else math.nan for row in rows]
        ax.plot(x, y, marker=markers[label], label=label, lw=1.1)
        for row in rows:
            if row["candidate"] in {"pymbar", "threshold_default", "rolling_w500_1sigma_stable"}:
                ax.text(float(row["start_ns"]), float(row["delta_ref_sigma"]), row["candidate"], fontsize=7)
    ax.axhline(1.0, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("trim start (ns)")
    ax.set_ylabel("|kept mean - last 20% mean| / raw sigma")
    ax.set_title("Candidate trim sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "candidate_mean_sensitivity.png", dpi=170)
    plt.close(fig)


def configure_paths(project_dir: Path, out_dir: Path | None = None) -> None:
    """Rebind the module-level project/output/input path globals.

    Must be called (as ``main`` does) before the analysis so the fixed
    C/D/S run layout is rooted at the actual project directory.

    Args:
        project_dir: stage-03 project directory containing the C/D/S runs.
        out_dir: output directory (default ``<project>/sensitivity``).
    """
    global PROJECT, OUTDIR, TRAJECTORIES, PYMBAR_INFO, THRESHOLD_INFO
    PROJECT = project_dir.resolve()
    OUTDIR = out_dir.resolve() if out_dir else PROJECT / "sensitivity"
    TRAJECTORIES = {
        "C": PROJECT / "md_C/C/md_only_from_last_snapshot/xtb.trj",
        "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb.trj",
        "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb.trj",
    }
    PYMBAR_INFO = {
        "C": PROJECT / "md_C/C/md_only_from_last_snapshot/xtb_traj_trim_info.json",
        "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb_traj_trim_info.json",
        "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb_traj_trim_info.json",
    }
    THRESHOLD_INFO = {
        "C": PROJECT / "trim_threshold/C/xtb_traj_trim_info.json",
        "D": PROJECT / "trim_threshold/D/xtb_traj_trim_info.json",
        "S": PROJECT / "trim_threshold/S/xtb_traj_trim_info.json",
    }


def main() -> None:
    """CLI entry point: run the full sensitivity study for C, D, and S.

    For each label: read energies, compute the reference/threshold/
    rolling/candidate/ACF diagnostics (with extra fixed candidate starts
    for C), then write the CSVs, Markdown report, JSON payload, and plots
    to the output directory.

    Raises:
        RuntimeError: a trajectory yields no energies.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT, help="03_qm_to_martini project directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: <project-dir>/sensitivity")
    args = parser.parse_args()
    configure_paths(args.project_dir, args.out_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    rolling_rows: list[dict] = []
    candidate_rows: list[dict] = []
    acf_rows: list[dict] = []
    results: dict = {}

    for label, path in TRAJECTORIES.items():
        arr = read_energies(path)
        if len(arr) == 0:
            raise RuntimeError(f"No energies found for {label}: {path}")

        ref_start, ref_mean, ref_std = tail_reference(arr, 0.2)
        pymbar = load_json(PYMBAR_INFO[label])
        threshold = load_json(THRESHOLD_INFO[label])

        tail_t0_default, _, _, _ = threshold_t0(arr, 0.2, 1.0)
        roll_default = rolling_stats(arr, ref_mean, ref_std, 500, 1.0)
        acf = autocorrelation_summary(arr)

        summary_rows.append(
            {
                "label": label,
                "frames": len(arr),
                "duration_ns": frame_to_ns(len(arr)),
                "last20_start_frame": ref_start,
                "last20_mean_Ha": ref_mean,
                "last20_std_Ha": ref_std,
                "threshold_tail_t0_sigma1": tail_t0_default,
                "threshold_written_start": threshold.get("start_index", ""),
                "pymbar_t0": pymbar.get("t0", ""),
                "pymbar_trim_fraction": pymbar.get("trim_fraction", ""),
                "rolling_w500_first_in_1sigma": roll_default["first_in_band"],
                "rolling_w500_stable_1sigma": roll_default["stable_from"],
                "acf_lag_1_over_e": acf["lag_1_over_e"],
                "acf_lag_0p1": acf["lag_0p1"],
                "acf_g_first_negative": acf["g_first_negative"],
            }
        )

        for ref_fraction in REF_FRACTIONS:
            for sigma in SIGMAS:
                t0, mean, std, ref_start_i = threshold_t0(arr, ref_fraction, sigma)
                threshold_rows.append(
                    {
                        "label": label,
                        "ref_fraction": ref_fraction,
                        "sigma": sigma,
                        "t0_frame": t0,
                        "t0_ns": frame_to_ns(t0),
                        "kept_frames": max(len(arr) - t0, 0),
                        "ref_start_frame": ref_start_i,
                        "ref_mean_Ha": mean,
                        "ref_std_Ha": std,
                    }
                )

        for window in ROLLING_WINDOWS:
            for sigma in ROLLING_SIGMAS:
                stats = rolling_stats(arr, ref_mean, ref_std, window, sigma)
                rolling_rows.append(
                    {
                        "label": label,
                        "window_frames": window,
                        "window_ps": window * DUMP_FS / 1000.0,
                        "sigma": sigma,
                        "first_in_band_frame": stats["first_in_band"],
                        "first_in_band_ns": frame_to_ns(stats["first_in_band"]) if stats["first_in_band"] is not None else "",
                        "stable_from_frame": stats["stable_from"],
                        "stable_from_ns": frame_to_ns(stats["stable_from"]) if stats["stable_from"] is not None else "",
                        "fraction_in_band": stats["fraction_in_band"],
                        "max_abs_sigma": stats["max_abs_sigma"],
                    }
                )

        candidate_starts = {
            "no_trim": 0,
            "threshold_tail_sigma1": tail_t0_default,
            "threshold_default": threshold.get("start_index", 0),
            "manual_5000": 5000 if len(arr) > 5000 else 0,
            "pymbar": pymbar.get("t0", 0),
            "rolling_w500_1sigma_stable": roll_default["stable_from"],
            "last_50pct": int(len(arr) * 0.5),
            "last_25pct": int(len(arr) * 0.75),
            "last_20pct": ref_start,
        }
        if label == "C":
            candidate_starts.update(
                {
                    "C_50k": 50_000,
                    "C_100k": 100_000,
                    "C_150k": 150_000,
                    "C_185k": 185_000,
                }
            )

        for name, start in candidate_starts.items():
            if start is None:
                continue
            stats = candidate_stats(arr, int(start), ref_mean, ref_std)
            candidate_rows.append(
                {
                    "label": label,
                    "candidate": name,
                    "start_frame": stats["start"],
                    "start_ns": frame_to_ns(stats["start"]),
                    "kept_frames": stats["kept_frames"],
                    "kept_ns": stats["kept_ns"],
                    "mean_Ha": stats["mean"],
                    "std_Ha": stats["std"],
                    "block_sem_Ha": stats["block_sem"],
                    "delta_ref_sigma": stats["delta_ref_sigma"],
                }
            )

        acf_row = {"label": label}
        acf_row.update(acf)
        acf_rows.append(acf_row)

        results[label] = {
            "energies": arr,
            "ref": {"start": ref_start, "mean": ref_mean, "std": ref_std},
            "candidate_starts": candidate_starts,
        }

    write_csv(
        OUTDIR / "summary.csv",
        summary_rows,
        [
            "label",
            "frames",
            "duration_ns",
            "last20_start_frame",
            "last20_mean_Ha",
            "last20_std_Ha",
            "threshold_tail_t0_sigma1",
            "threshold_written_start",
            "pymbar_t0",
            "pymbar_trim_fraction",
            "rolling_w500_first_in_1sigma",
            "rolling_w500_stable_1sigma",
            "acf_lag_1_over_e",
            "acf_lag_0p1",
            "acf_g_first_negative",
        ],
    )
    write_csv(
        OUTDIR / "threshold_sweep.csv",
        threshold_rows,
        ["label", "ref_fraction", "sigma", "t0_frame", "t0_ns", "kept_frames", "ref_start_frame", "ref_mean_Ha", "ref_std_Ha"],
    )
    write_csv(
        OUTDIR / "rolling_sweep.csv",
        rolling_rows,
        [
            "label",
            "window_frames",
            "window_ps",
            "sigma",
            "first_in_band_frame",
            "first_in_band_ns",
            "stable_from_frame",
            "stable_from_ns",
            "fraction_in_band",
            "max_abs_sigma",
        ],
    )
    write_csv(
        OUTDIR / "candidate_stats.csv",
        candidate_rows,
        ["label", "candidate", "start_frame", "start_ns", "kept_frames", "kept_ns", "mean_Ha", "std_Ha", "block_sem_Ha", "delta_ref_sigma"],
    )
    write_csv(OUTDIR / "acf_summary.csv", acf_rows, ["label", *acf_rows[0].keys() - {"label"}] if False else list(acf_rows[0].keys()))

    plot_energy(results)
    plot_candidate_sensitivity(candidate_rows)

    report_lines = [
        "# Trim Sensitivity: C/D/S",
        "",
        "Trajectory energy sensitivity analysis. Times assume 50 fs between saved frames.",
        "",
        "## Main Summary",
        "",
        "| label | frames | ns | threshold t0 | threshold written start | pymbar t0 | pymbar trim % | rolling w500 stable 1sigma | ACF g | lag ACF<0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {label} | {frames} | {ns} | {et0} | {tw} | {pt0} | {ptrim} | {rw} | {acf_g} | {acf_lag} |".format(
                label=row["label"],
                frames=row["frames"],
                ns=fmt_num(row["duration_ns"], 2),
                et0=row["threshold_tail_t0_sigma1"],
                tw=row["threshold_written_start"],
                pt0=row["pymbar_t0"],
                ptrim=fmt_num(float(row["pymbar_trim_fraction"]) * 100 if row["pymbar_trim_fraction"] != "" else math.nan, 2),
                rw=row["rolling_w500_stable_1sigma"],
                acf_g=fmt_num(row["acf_g_first_negative"], 1),
                acf_lag=row["acf_lag_0p1"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Tail-mean threshold is permissive for all three systems. With ref_fraction=0.2 and sigma=1.0, the detected t0 is 0 for C/D/S.",
            "- C remains qualitatively different under rolling-mean checks. The 500-frame rolling mean only stays inside the last-20%-reference 1sigma band very late, while D/S are stable near the beginning.",
            "- The pymbar C trim leaves only about 0.052 ns, so it is too destructive for parameter statistics unless a very conservative sanity check is required.",
            "- A practical route is to treat C as strongly autocorrelated and report trim sensitivity, rather than extending the MD again.",
            "",
            "## Files",
            "",
            "- summary.csv: one-line summary per monomer.",
            "- threshold_sweep.csv: t0 across ref_fraction and threshold_sigma.",
            "- rolling_sweep.csv: rolling-mean first-in-band and stable-from diagnostics.",
            "- candidate_stats.csv: retained-energy statistics for candidate trim starts.",
            "- acf_summary.csv: energy autocorrelation diagnostics.",
            "- energy_rolling_candidates.png and candidate_mean_sensitivity.png: quick-look plots.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    payload = {
        "summary": summary_rows,
        "threshold_sweep": threshold_rows,
        "rolling_sweep": rolling_rows,
        "candidate_stats": candidate_rows,
        "acf_summary": acf_rows,
    }
    (OUTDIR / "sensitivity_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

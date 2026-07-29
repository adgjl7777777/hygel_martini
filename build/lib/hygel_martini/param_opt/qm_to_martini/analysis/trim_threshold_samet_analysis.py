#!/usr/bin/env python3
"""Analyze the trim_threshold_samet trajectory set.

This set uses C from md_C_old and D/S from the current D/S MD trajectories,
all with the same threshold_default rule:
energy_threshold(ref_fraction=0.2, sigma=1.0) plus skip_frames=5000.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_PROJECT = Path.cwd()
PROJECT = DEFAULT_PROJECT
OUTDIR = PROJECT / "trim_threshold_samet"
DUMP_FS = 50.0
ENERGY_RE = re.compile(r"energy:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

TRAJECTORIES = {
    "C": PROJECT / "md_C_old/C/relax_xtb_geoopt/xtb.trj",
    "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb.trj",
    "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb.trj",
}

SOURCE_NOTE = {
    "C": "md_C_old",
    "D": "md_D",
    "S": "md_S",
}


def read_energies(path: Path) -> np.ndarray:
    energies = []
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


def load_trim_info(label: str) -> dict:
    path = OUTDIR / label / "xtb_traj_trim_info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def frame_to_ns(frame: int) -> float:
    return frame * DUMP_FS / 1_000_000.0


def tail_reference(arr: np.ndarray, ref_fraction: float = 0.2) -> tuple[int, float, float]:
    start = max(int(len(arr) * (1.0 - ref_fraction)), 1)
    ref = arr[start:]
    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    std = max(std, abs(mean) * 1e-6 + 1e-12)
    return start, mean, std


def threshold_t0(arr: np.ndarray, ref_fraction: float = 0.2, sigma: float = 1.0) -> int:
    _, mean, std = tail_reference(arr, ref_fraction)
    tail_means = np.cumsum(arr[::-1])[::-1] / np.arange(len(arr), 0, -1)
    in_band = np.abs(tail_means - mean) <= sigma * std
    hits = np.where(in_band)[0]
    return int(hits[0]) if len(hits) else len(arr)


def rolling_stable_from(arr: np.ndarray, ref_mean: float, ref_std: float, window: int = 500, sigma: float = 1.0):
    if len(arr) < window:
        return None, None
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    norm = np.abs(roll - ref_mean) / ref_std
    in_band = norm <= sigma
    first = int(np.argmax(in_band) + window // 2) if np.any(in_band) else None
    suffix_ok = np.logical_and.accumulate(in_band[::-1])[::-1]
    stable_hits = np.where(suffix_ok)[0]
    stable = int(stable_hits[0] + window // 2) if len(stable_hits) else None
    return first, stable


def acf_g(arr: np.ndarray, max_lag: int = 50_000) -> tuple[float, int]:
    centered = arr - float(np.mean(arr))
    n = len(centered)
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_len)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_len)[:n]
    acf = acf / np.arange(n, 0, -1)
    norm = acf[: max_lag + 1] / acf[0] if acf[0] else np.zeros(max_lag + 1)
    positive = norm[1:] > 0
    first_nonpositive = int(np.argmax(~positive) + 1) if np.any(~positive) else len(norm) - 1
    cutoff = max(first_nonpositive, 1)
    return float(1.0 + 2.0 * np.sum(norm[1:cutoff])), first_nonpositive


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(results: dict[str, dict]) -> None:
    if plt is None:
        return

    labels = ["C", "D", "S"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, label in zip(axes, labels):
        data = results[label]
        arr = data["energies"]
        step = max(1, len(arr) // 6000)
        x = np.arange(0, len(arr), step) * DUMP_FS / 1000.0
        ax.plot(x, arr[::step], lw=0.35, alpha=0.55, label="energy")
        for window in [500, 5000]:
            if len(arr) >= window:
                roll = np.convolve(arr, np.ones(window) / window, mode="valid")
                rx = (np.arange(len(roll)) + window // 2) * DUMP_FS / 1000.0
                rstep = max(1, len(roll) // 6000)
                ax.plot(rx[::rstep], roll[::rstep], lw=0.9, label=f"rolling {window}")
        ref_mean = data["ref_mean"]
        ref_std = data["ref_std"]
        start = data["trim_info"]["start_index"]
        ax.axhline(ref_mean, color="black", lw=0.8, ls="--", label="last 20% mean")
        ax.axhspan(ref_mean - ref_std, ref_mean + ref_std, color="gray", alpha=0.12)
        ax.axvline(start * DUMP_FS / 1000.0, color="red", lw=1.1, ls=":", label=f"start={start}")
        ax.set_title(f"{label} ({SOURCE_NOTE[label]}): threshold_default")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("E (Ha)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparison_energy_series.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, label in zip(axes, labels):
        data = results[label]
        arr = data["energies"]
        ref_mean = data["ref_mean"]
        ref_std = data["ref_std"]
        tail_means = np.cumsum(arr[::-1])[::-1] / np.arange(len(arr), 0, -1)
        norm = (tail_means - ref_mean) / ref_std
        step = max(1, len(arr) // 6000)
        x = np.arange(0, len(arr), step) * DUMP_FS / 1000.0
        ax.plot(x, norm[::step], lw=0.75)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axhspan(-1, 1, color="gray", alpha=0.12)
        ax.axvline(data["trim_info"]["start_index"] * DUMP_FS / 1000.0, color="red", lw=1.1, ls=":")
        ax.set_title(f"{label}: tail mean deviation from final 20% reference")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("sigma units")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparison_tail_mean.png", dpi=170)
    plt.close(fig)


def configure_paths(project_dir: Path, out_dir: Path | None = None) -> None:
    global PROJECT, OUTDIR, TRAJECTORIES
    PROJECT = project_dir.resolve()
    OUTDIR = out_dir.resolve() if out_dir else PROJECT / "trim_threshold_samet"
    TRAJECTORIES = {
        "C": PROJECT / "md_C_old/C/relax_xtb_geoopt/xtb.trj",
        "D": PROJECT / "md_D/D/relax_xtb_geoopt/xtb.trj",
        "S": PROJECT / "md_S/S/relax_xtb_geoopt/xtb.trj",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT, help="03_qm_to_martini project directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="trim_threshold_samet output directory")
    args = parser.parse_args()
    configure_paths(args.project_dir, args.out_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}
    for label, path in TRAJECTORIES.items():
        arr = read_energies(path)
        if len(arr) == 0:
            raise RuntimeError(f"No energies found for {label}: {path}")
        trim_info = load_trim_info(label)
        ref_start, ref_mean, ref_std = tail_reference(arr)
        first, stable = rolling_stable_from(arr, ref_mean, ref_std)
        g, first_nonpositive = acf_g(arr)
        t0 = threshold_t0(arr)
        start = int(trim_info["start_index"])
        kept = arr[start:]
        row = {
            "label": label,
            "source": SOURCE_NOTE[label],
            "frames": len(arr),
            "duration_ns": frame_to_ns(len(arr)),
            "threshold_t0": t0,
            "skip_frames": trim_info["skip_frames"],
            "start_index": start,
            "written_frames": trim_info["written_frames"],
            "trim_fraction": trim_info["trim_fraction"],
            "last20_mean_Ha": ref_mean,
            "last20_std_Ha": ref_std,
            "kept_mean_Ha": float(np.mean(kept)),
            "kept_delta_ref_sigma": abs(float(np.mean(kept)) - ref_mean) / ref_std,
            "rolling_w500_first_in_1sigma": first,
            "rolling_w500_stable_1sigma": stable,
            "acf_g_first_negative": g,
            "acf_first_nonpositive_lag": first_nonpositive,
        }
        rows.append(row)
        results[label] = {
            "energies": arr,
            "trim_info": trim_info,
            "ref_mean": ref_mean,
            "ref_std": ref_std,
        }

    write_csv(OUTDIR / "summary.csv", rows)
    (OUTDIR / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_plots(results)

    lines = [
        "# trim_threshold_samet",
        "",
        "This set applies the same threshold_default trim to C/D/S, with C taken from md_C_old.",
        "",
        "Rule: energy_threshold(ref_fraction=0.2, threshold_sigma=1.0), then `start_index=max(t0, 5000)`.",
        "",
        "| label | source | frames | ns | t0 | start | written | trim % | kept delta sigma | rolling stable | ACF g |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {source} | {frames} | {ns:.2f} | {t0} | {start} | {written} | {trim:.2f} | {delta:.3f} | {rolling} | {acf:.1f} |".format(
                label=row["label"],
                source=row["source"],
                frames=row["frames"],
                ns=row["duration_ns"],
                t0=row["threshold_t0"],
                start=row["start_index"],
                written=row["written_frames"],
                trim=100.0 * row["trim_fraction"],
                delta=row["kept_delta_ref_sigma"],
                rolling=row["rolling_w500_stable_1sigma"],
                acf=row["acf_g_first_negative"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- All three systems use the identical 5000-frame initial buffer.",
            "- In this same-length set, C_old/D/S all have 104167 raw frames and 99167 written frames.",
            "- C_old is still more correlated than D/S by ACF g, but it is much less pathological than current C under pymbar.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

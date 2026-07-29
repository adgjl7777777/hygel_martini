from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:
    from pymbar import timeseries
except ImportError:  # pragma: no cover - optional runtime dependency
    timeseries = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def _parse_xvg(path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    values: List[float] = []
    if not path.exists():
        return times, values
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue
    return times, values


def _iter_pdb_frames(path: Path) -> Iterable[List[str]]:
    current: List[str] = []
    saw_model = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if saw_model and current:
                    yield current
                saw_model = True
                current.append(line)
            elif line.startswith("ENDMDL"):
                if saw_model:
                    current.append(line)
                    yield current
                    current = []
                else:
                    current.append(line)
            else:
                current.append(line)
    if current:
        if saw_model:
            yield current
        else:
            yield current


def _detect_t0_pymbar(values: List[float], nskip: int, fast: bool) -> tuple[int, float, float]:
    if np is None or timeseries is None or not values:
        return 0, 1.0, float(len(values))
    arr = np.asarray(values, dtype=float)
    t0, g, neff = timeseries.detect_equilibration(arr, nskip=max(1, int(nskip)), fast=fast)
    return int(t0), float(g), float(neff)


def _detect_t0_energy_threshold(
    values: List[float],
    ref_fraction: float,
    threshold_sigma: float,
) -> int:
    if np is None or not values:
        return 0
    arr = np.asarray(values, dtype=float)
    total = len(arr)
    ref_start = max(1, int(total * (1.0 - ref_fraction)))
    ref = arr[ref_start:]
    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    ref_std = max(ref_std, abs(ref_mean) * 1e-6 + 1e-12)
    threshold = threshold_sigma * ref_std
    cumsum_rev = np.cumsum(arr[::-1])[::-1]
    counts = np.arange(total, 0, -1, dtype=float)
    tail_means = cumsum_rev / counts
    candidates = np.where(np.abs(tail_means - ref_mean) <= threshold)[0]
    return int(candidates[0]) if len(candidates) else 0


def _write_plots(values: List[float], t0: int, start_index: int, out_pdb: Path) -> None:
    if plt is None or np is None or not values:
        return
    arr = np.asarray(values, dtype=float)
    frames = np.arange(len(arr))
    stem = out_pdb.stem

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(frames, arr, lw=0.7, color="steelblue")
    axes[0].axvline(start_index, color="red", ls="--", lw=1.0, label=f"start={start_index}")
    if t0 != start_index:
        axes[0].axvline(t0, color="orange", ls=":", lw=1.0, label=f"t0={t0}")
    axes[0].set_ylabel("Energy")
    axes[0].legend(fontsize=8)

    running = np.cumsum(arr) / (frames + 1)
    axes[1].plot(frames, running, lw=0.8, color="navy")
    axes[1].axvline(start_index, color="red", ls="--", lw=1.0)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Cumulative mean")
    plt.tight_layout()
    fig.savefig(out_pdb.with_name(stem + "_energy_convergence.png"), dpi=150)
    plt.close(fig)


def trim_pdb(
    input_pdb: Path,
    output_pdb: Path,
    *,
    energy_xvg: Path | None,
    auto_trim: bool,
    skip_frames: int,
    nskip: int,
    max_trim_fraction: float,
    trim_method: str,
    ref_fraction: float,
    threshold_sigma: float,
    fast: bool,
    write_plots: bool,
) -> dict:
    frames = list(_iter_pdb_frames(input_pdb))
    total_frames = len(frames)
    _, energies = _parse_xvg(energy_xvg) if energy_xvg else ([], [])

    t0 = 0
    g = None
    neff = None
    note = ""
    if auto_trim and energies:
        if trim_method == "energy_threshold":
            t0 = _detect_t0_energy_threshold(energies, ref_fraction, threshold_sigma)
        else:
            t0, g, neff = _detect_t0_pymbar(energies, nskip, fast)
    elif auto_trim:
        note = "No energy XVG was available; only skip_frames was applied."

    max_t0 = int(total_frames * max(0.0, min(1.0, max_trim_fraction)))
    if t0 > max_t0:
        t0 = max_t0
    start_index = max(int(skip_frames), int(t0))
    start_index = min(start_index, total_frames)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        for frame in frames[start_index:]:
            handle.writelines(frame)

    written = max(0, total_frames - start_index)
    info = {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "energy_xvg": str(energy_xvg) if energy_xvg else None,
        "total_frames": total_frames,
        "energy_points": len(energies),
        "t0": int(t0),
        "skip_frames": int(skip_frames),
        "start_index": int(start_index),
        "written_frames": int(written),
        "trim_fraction": round((total_frames - written) / total_frames, 4) if total_frames else 0.0,
        "trim_method": trim_method if auto_trim else "none",
        "g": g,
        "neff": neff,
        "note": note,
    }
    info_path = output_pdb.with_name(output_pdb.stem + "_trim_info.json")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    if write_plots and energies:
        _write_plots(energies, int(t0), int(start_index), output_pdb)

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim a GROMACS-derived multi-model PDB using optional energy XVG data."
    )
    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("--energy-xvg", default="")
    parser.add_argument("--auto-trim", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--nskip", type=int, default=1)
    parser.add_argument("--max-trim-fraction", type=float, default=1.0)
    parser.add_argument("--trim-method", choices=["pymbar", "energy_threshold"], default="pymbar")
    parser.add_argument("--ref-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-sigma", type=float, default=1.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    energy_xvg = Path(args.energy_xvg) if args.energy_xvg else None
    info = trim_pdb(
        Path(args.input_pdb),
        Path(args.output_pdb),
        energy_xvg=energy_xvg,
        auto_trim=args.auto_trim,
        skip_frames=args.skip_frames,
        nskip=args.nskip,
        max_trim_fraction=args.max_trim_fraction,
        trim_method=args.trim_method,
        ref_fraction=args.ref_fraction,
        threshold_sigma=args.threshold_sigma,
        fast=args.fast,
        write_plots=not args.no_plots,
    )
    print(json.dumps(info, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

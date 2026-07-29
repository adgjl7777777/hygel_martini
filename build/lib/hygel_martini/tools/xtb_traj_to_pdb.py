from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    from pymbar import timeseries
except ImportError:
    timeseries = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# xTB trajectory comment example: " energy: -123.45678 gnorm: 0.001 ..."
ENERGY_RE = re.compile(r"energy:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def extract_energy(comment: str) -> float | None:
    match = ENERGY_RE.search(comment)
    if match:
        return float(match.group(1))
    return None


def parse_frames_streaming(path: Path):
    """Yields (comment_line, atom_lines) for each frame in an XYZ trajectory."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            comment = f.readline().strip()
            atom_lines = []
            for _ in range(natoms):
                atom_line = f.readline()
                if not atom_line:
                    break
                atom_lines.append(atom_line.strip())
            if len(atom_lines) == natoms:
                yield comment, atom_lines
            else:
                break


def parse_pdb_frame_count(path: Path) -> int:
    """Count MODEL records in a PDB file (proxy for frame count)."""
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("MODEL"):
                count += 1
    return count


def pdb_atom_line(atom_index: int, symbol: str, x: float, y: float, z: float) -> str:
    atom_name = symbol[:2].upper().rjust(2)
    return (
        f"ATOM  {atom_index:5d} {atom_name:<4} MOL A{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol[:2].upper():>2}\n"
    )


def detect_t0(
    path: Path,
    nskip: int = 1,
    max_trim_fraction: float = 1.0,
    detrend: bool = False,
    fast: bool = False,
) -> tuple[int, int, list]:
    """Pass 1: Extract energies, detect equilibration start.

    Args:
        nskip: pymbar nskip — check every Nth frame as a candidate t0.
               Higher values are less sensitive to slow monotonic drift (faster too).
        max_trim_fraction: cap on how much of the trajectory can be discarded.
               E.g. 0.5 means never trim more than 50% regardless of pymbar result.
        detrend: subtract a linear fit from the energy before passing to pymbar.
                 Prevents slow monotonic Epot drift from being flagged as non-equilibrated.

    Returns (t0, total_frames, energies).
    """
    if np is None or timeseries is None:
        print("[WARN] numpy or pymbar not found. Skipping auto-trimming.", file=sys.stderr)
        total = sum(1 for _ in parse_frames_streaming(path))
        return 0, total, []

    energies = []
    frame_iter = parse_frames_streaming(path)
    if _tqdm is not None:
        frame_iter = _tqdm(frame_iter, unit="frame", desc="Reading energies", leave=False)
    for comment, _ in frame_iter:
        e = extract_energy(comment)
        if e is not None:
            energies.append(e)

    total_frames = len(energies)
    print(f"[INFO] Total frames in trajectory: {total_frames}", file=sys.stderr)

    if not energies:
        print("[WARN] No energy data found in trajectory comments.", file=sys.stderr)
        return 0, total_frames, energies

    arr = np.array(energies)
    pymbar_input = arr
    if detrend and len(arr) > 2:
        x = np.arange(len(arr), dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        pymbar_input = arr - (slope * x + intercept)
        print(f"[INFO] Detrend: removed linear slope {slope:.3e} Hartree/frame before pymbar.", file=sys.stderr)

    max_t0 = int(total_frames * max_trim_fraction)
    try:
        t0, g, neff = timeseries.detect_equilibration(pymbar_input, nskip=nskip, fast=fast)
        if t0 > max_t0:
            print(
                f"[INFO] pymbar t0={t0} exceeds max_trim_fraction={max_trim_fraction} "
                f"(max_t0={max_t0}). Capping at {max_t0}.",
                file=sys.stderr,
            )
            t0 = max_t0
        print(f"[INFO] Equilibration detected: t0={t0}, g={g:.2f}, Neff={neff:.1f}", file=sys.stderr)
        return int(t0), total_frames, energies
    except Exception as e:
        print(f"[WARN] Pymbar failed: {e}. Using full trajectory.", file=sys.stderr)
        return 0, total_frames, energies


def detect_t0_energy_threshold(
    path: Path,
    ref_fraction: float = 0.2,
    threshold_sigma: float = 1.0,
    max_trim_fraction: float = 1.0,
) -> tuple[int, int, list]:
    """Pass 1: Extract energies, detect t0 by tail-mean convergence.

    t0 = earliest frame t where mean(energy[t:]) is within threshold_sigma * std
    of the mean of the last ref_fraction of the trajectory. Uses numpy cumsum
    for O(N) computation.

    Returns (t0, total_frames, energies).
    """
    if np is None:
        print("[WARN] numpy not found. Skipping auto-trimming.", file=sys.stderr)
        total = sum(1 for _ in parse_frames_streaming(path))
        return 0, total, []

    energies = []
    frame_iter = parse_frames_streaming(path)
    if _tqdm is not None:
        frame_iter = _tqdm(frame_iter, unit="frame", desc="Reading energies", leave=False)
    for comment, _ in frame_iter:
        e = extract_energy(comment)
        if e is not None:
            energies.append(e)

    total_frames = len(energies)
    print(f"[INFO] Total frames in trajectory: {total_frames}", file=sys.stderr)

    if not energies:
        print("[WARN] No energy data found in trajectory comments.", file=sys.stderr)
        return 0, total_frames, energies

    arr = np.array(energies)

    # Reference region: last ref_fraction of trajectory
    ref_start = int(total_frames * (1.0 - ref_fraction))
    ref_start = max(ref_start, 1)
    ref = arr[ref_start:]
    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    # Floor to avoid degenerate zero-std bands
    ref_std = max(ref_std, abs(ref_mean) * 1e-6 + 1e-12)
    threshold = threshold_sigma * ref_std

    print(
        f"[INFO] Energy-threshold ref: mean={ref_mean:.6f} Ha, std={ref_std:.2e} Ha, "
        f"band=±{threshold:.2e} Ha ({threshold_sigma}σ), ref_start={ref_start}",
        file=sys.stderr,
    )

    # Tail means: mean(arr[t:]) for each t, computed in O(N) via reverse cumsum
    cumsum_rev = np.cumsum(arr[::-1])[::-1]
    counts = np.arange(total_frames, 0, -1, dtype=float)
    tail_means = cumsum_rev / counts

    in_band = np.abs(tail_means - ref_mean) <= threshold

    max_t0 = int(total_frames * max_trim_fraction)
    candidates = np.where(in_band[: max_t0 + 1])[0]
    if len(candidates) > 0:
        t0 = int(candidates[0])
        print(
            f"[INFO] Energy-threshold t0={t0} "
            f"(tail mean first entered ±{threshold_sigma}σ band at frame {t0})",
            file=sys.stderr,
        )
    else:
        t0 = max_t0
        print(
            f"[INFO] Tail mean never entered ±{threshold_sigma}σ band within "
            f"max_trim_fraction={max_trim_fraction}. Capping at t0={t0}.",
            file=sys.stderr,
        )

    return t0, total_frames, energies


def _block_standard_error(data, max_block_size=None):
    """Compute BSE vs block size for block averaging convergence check."""
    N = len(data)
    if max_block_size is None:
        max_block_size = N // 4
    block_sizes = []
    bse = []
    for n in range(1, max_block_size + 1):
        M = N // n
        if M < 4:
            break
        blocks = [np.mean(data[i * n:(i + 1) * n]) for i in range(M)]
        block_sizes.append(n)
        bse.append(np.std(blocks, ddof=1) / np.sqrt(M))
    return block_sizes, bse


def save_convergence_plots(energies: list, t0: int, start_index: int, output_path: Path) -> None:
    """Save energy time-series and block-averaging plots alongside the PDB."""
    if plt is None or np is None or not energies:
        return

    arr = np.array(energies)
    frames = np.arange(len(arr))
    stem = output_path.stem

    # --- Plot 1: energy time series with t0 / start_index markers ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(frames, arr, lw=0.6, color="steelblue", alpha=0.8)
    if start_index > 0:
        axes[0].axvline(start_index, color="red", ls="--", lw=1.2, label=f"start_index={start_index}")
    if t0 > 0 and t0 != start_index:
        axes[0].axvline(t0, color="orange", ls=":", lw=1.2, label=f"t0={t0}")
    axes[0].set_ylabel("Energy (Hartree)")
    axes[0].set_title(f"{stem}: xTB MD energy time series")
    axes[0].legend(fontsize=8)

    # Running mean
    running_mean = np.cumsum(arr) / (frames + 1)
    axes[1].plot(frames, running_mean, lw=0.8, color="navy")
    axes[1].axhline(running_mean[-1], color="red", ls="--", lw=0.8, label="final mean")
    if start_index > 0:
        axes[1].axvline(start_index, color="red", ls="--", lw=1.0)
    axes[1].set_ylabel("Cumulative mean")
    axes[1].legend(fontsize=8)

    # Production region histogram
    prod = arr[start_index:]
    if len(prod) > 1:
        axes[2].hist(prod, bins=40, density=True, color="steelblue", alpha=0.7, label="production")
    if start_index > 0 and start_index < len(arr):
        eq_region = arr[:start_index]
        if len(eq_region) > 1:
            axes[2].hist(eq_region, bins=40, density=True, color="salmon", alpha=0.5, label="equilibration")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Energy distribution: equilibration vs production")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_path.with_name(stem + "_energy_convergence.png")
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Energy convergence plot saved to {plot_path}", file=sys.stderr)

    # --- Plot 2: Block averaging on production region ---
    if len(prod) >= 8:
        block_sizes, bse = _block_standard_error(prod)
        if block_sizes:
            fig2, ax = plt.subplots(figsize=(7, 4))
            ax.plot(block_sizes, bse, lw=1.0, color="steelblue")
            ax.axhline(bse[-1], color="red", ls="--", lw=0.8, label=f"asymptote ≈ {bse[-1]:.2e}")
            ax.set_xlabel("Block size (frames)")
            ax.set_ylabel("BSE")
            ax.set_title(f"{stem}: Block averaging (production region, N={len(prod)} frames)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            bse_path = output_path.with_name(stem + "_block_avg.png")
            fig2.savefig(str(bse_path), dpi=150)
            plt.close(fig2)
            print(f"[INFO] Block-averaging plot saved to {bse_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Convert xTB XYZ trajectory to PDB with auto-trimming.")
    parser.add_argument("input", help="Input .trj (XYZ) file")
    parser.add_argument("output", help="Output .pdb file")
    parser.add_argument("--auto-trim", action="store_true", help="Automatically detect t0 using energy convergence")
    parser.add_argument("--skip-frames", type=int, default=0, help="Manually skip first N frames (stacks with auto-trim)")
    parser.add_argument("--nskip", type=int, default=1, help="pymbar nskip: check every Nth frame as t0 candidate. Higher = less sensitive to slow drift")
    parser.add_argument("--max-trim-fraction", type=float, default=1.0, help="Cap: never discard more than this fraction of frames (e.g. 0.5)")
    parser.add_argument("--no-plots", action="store_true", help="Skip convergence plot generation")
    parser.add_argument("--pdb-info", action="store_true", help="Count MODEL frames in a PDB and write trim_info.json; no PDB is written")
    parser.add_argument("--detrend", action="store_true", help="Subtract linear energy trend before pymbar (suppresses slow monotonic Epot drift)")
    parser.add_argument("--fast", action="store_true", help="Use pymbar fast=True mode (O(N log N), much faster but slightly less accurate)")
    parser.add_argument("--trim-method", choices=["pymbar", "energy_threshold"], default="pymbar",
                        help="Equilibration detection method: pymbar (default) or energy_threshold")
    parser.add_argument("--ref-fraction", type=float, default=0.2,
                        help="energy_threshold: fraction of trajectory tail used as reference (default 0.2)")
    parser.add_argument("--threshold-sigma", type=float, default=1.0,
                        help="energy_threshold: band half-width in units of reference std (default 1.0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.pdb_info:
        total = parse_pdb_frame_count(input_path)
        print(f"[INFO] PDB frame count: {total}", file=sys.stderr)
        trim_info = {
            "total_frames": total,
            "t0": 0,
            "skip_frames": 0,
            "start_index": 0,
            "written_frames": total,
            "trim_fraction": 0.0,
            "note": "PDB input — energy-based equilibration detection not available",
        }
        trim_info_path = output_path.with_name(output_path.stem + "_trim_info.json")
        trim_info_path.write_text(json.dumps(trim_info, indent=2), encoding="utf-8")
        print(f"[INFO] Trim info written to {trim_info_path}", file=sys.stderr)
        return

    t0 = 0
    total_frames = 0
    energies = []
    if args.auto_trim:
        if args.trim_method == "energy_threshold":
            t0, total_frames, energies = detect_t0_energy_threshold(
                input_path,
                ref_fraction=args.ref_fraction,
                threshold_sigma=args.threshold_sigma,
                max_trim_fraction=args.max_trim_fraction,
            )
        else:
            t0, total_frames, energies = detect_t0(
                input_path,
                nskip=args.nskip,
                max_trim_fraction=args.max_trim_fraction,
                detrend=args.detrend,
                fast=args.fast,
            )

    start_index = max(t0, args.skip_frames)
    if start_index > 0:
        print(f"[INFO] Skipping first {start_index} frames.", file=sys.stderr)

    # Pass 2: Write PDB
    with output_path.open("w", encoding="utf-8") as out_f:
        count = 0
        written = 0
        frame_iter2 = parse_frames_streaming(input_path)
        if _tqdm is not None:
            desc = f"Writing PDB (skip={start_index})" if start_index > 0 else "Writing PDB"
            frame_iter2 = _tqdm(frame_iter2, total=total_frames or None, unit="frame", desc=desc)
        for comment, atom_lines in frame_iter2:
            if count >= start_index:
                model_index = written + 1
                out_f.write(f"MODEL     {model_index}\n")
                for atom_index, raw in enumerate(atom_lines, start=1):
                    parts = raw.split()
                    if len(parts) < 4:
                        continue
                    symbol = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        out_f.write(pdb_atom_line(atom_index, symbol, x, y, z))
                    except ValueError:
                        continue
                out_f.write("ENDMDL\n")
                written += 1
            count += 1

    if not args.auto_trim:
        total_frames = count

    print(f"[INFO] Successfully wrote {written} frames to {output_path}", file=sys.stderr)

    trim_info = {
        "total_frames": total_frames,
        "t0": t0,
        "skip_frames": args.skip_frames,
        "start_index": start_index,
        "written_frames": written,
        "trim_fraction": round((total_frames - written) / total_frames, 4) if total_frames > 0 else 0.0,
        "trim_method": args.trim_method if args.auto_trim else "none",
    }
    trim_info_path = output_path.with_name(output_path.stem + "_trim_info.json")
    trim_info_path.write_text(json.dumps(trim_info, indent=2), encoding="utf-8")
    print(f"[INFO] Trim info written to {trim_info_path}", file=sys.stderr)

    if args.auto_trim and not args.no_plots and energies:
        save_convergence_plots(energies, t0, start_index, output_path)


if __name__ == "__main__":
    main()

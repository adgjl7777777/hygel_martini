#!/usr/bin/env python3
"""Create visual diagnostics for a postprocess summary sweep."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


VARIANT_RE = re.compile(r"^(?P<potential>.+?)_R(?P<rmsd>[^_]+)_(?P<profile>.+)$")
SECTION_COLORS = {
    "accepted_bonds": "#3b6ea8",
    "accepted_constraints": "#6f9e59",
    "accepted_angles": "#007c89",
    "accepted_dihedrals": "#cc7a29",
    "accepted_impropers": "#8f6ab8",
}
ACCEPTED_SECTIONS = ["bonds", "constraints", "angles", "dihedrals", "impropers"]
POTENTIAL_PALETTE = [
    "#27647b",
    "#ad5d2d",
    "#4f7d4a",
    "#8d5a95",
    "#7c6d38",
    "#4e708c",
]
MODE_ORDER = [
    "init_only",
    "topology_n0",
    "topology_n1",
    "topology_n2",
    "topology_swap_n0",
    "topology_swap_n1",
    "topology_swap_n2",
    "all_unique",
]
LABEL_ORDER = ["C", "D", "S"]
SUPTITLE_Y = 0.965


def natural_key(text: str) -> tuple[int, str]:
    match = re.match(r"^[A-Za-z]+(\d+)", str(text))
    if match:
        return int(match.group(1)), str(text)
    return 10_000, str(text)


def clean_number(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return f"{number:g}"


def rmsd_label(value: object) -> str:
    return f"R{clean_number(value)}"


def compact_force(value: str) -> str:
    return str(value).replace("_", " ")


def compact_variant_label(row: pd.Series) -> str:
    return f"{row['potential_set']}\n{rmsd_label(row['rmsd_max_cutoff'])} {compact_force(row['force_profile'])}"


def one_line_variant_label(row: pd.Series) -> str:
    return f"{row['potential_set']} | {rmsd_label(row['rmsd_max_cutoff'])} {compact_force(row['force_profile'])}"


def potential_label(row: pd.Series) -> str:
    potential = str(row["potential_set"])
    angle = clean_number(row.get("angles_funct", ""))
    dihedral = clean_number(row.get("dihedrals_funct", ""))
    improper = clean_number(row.get("impropers_funct", ""))
    if angle == dihedral == improper == "bartender":
        return f"{potential}\nBartender"
    return f"{potential}\nA{angle} D{dihedral} I{improper}"


def enrich_variants(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for idx, variant_id in df["variant_id"].astype(str).items():
        match = VARIANT_RE.match(variant_id)
        if not match:
            records.append({"potential_set": variant_id, "force_profile": "unknown"})
            continue
        profile = match.group("profile")
        metric = str(df.at[idx, "multi_constant_metric"]) if "multi_constant_metric" in df else ""
        metric_suffix = f"_M{metric}" if metric else ""
        if metric_suffix and profile.endswith(metric_suffix):
            profile = profile[: -len(metric_suffix)]
        records.append(
            {
                "potential_set": match.group("potential"),
                "force_profile": profile,
            }
        )

    parsed = pd.DataFrame.from_records(records, index=df.index)
    out = pd.concat([df.copy(), parsed], axis=1)
    numeric_cols = [
        "case_count",
        "rmsd_max_cutoff",
        "accepted_total",
        "accepted_bonds",
        "accepted_constraints",
        "accepted_angles",
        "accepted_dihedrals",
        "accepted_impropers",
        "zero_angles_cases",
        "zero_dihedrals_cases",
        "all_rmse_max",
        "all_rmse_p90",
        "angles_rmse_max",
        "dihedrals_rmse_max",
    ]
    for section in ACCEPTED_SECTIONS:
        numeric_cols.append(f"force_min_cutoff_{section}")
    for prefix in ["all"] + ACCEPTED_SECTIONS:
        for kind in ["rmse", "force"]:
            for stat in ["n", "min", "max", "mean", "median", "p90"]:
                numeric_cols.append(f"{prefix}_{kind}_{stat}")
    for col in numeric_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "case_count" in out:
        denom = out["case_count"].replace(0, np.nan)
        avg_cols: dict[str, pd.Series] = {}
        for section in ["total"] + ACCEPTED_SECTIONS:
            col = f"accepted_{section}"
            if col in out:
                avg_cols[f"{col}_avg_per_case"] = out[col] / denom
        if avg_cols:
            out = pd.concat([out, pd.DataFrame(avg_cols, index=out.index)], axis=1)
    margin_cols: dict[str, pd.Series] = {}
    for section in ACCEPTED_SECTIONS:
        force_col = f"{section}_force_min"
        cutoff_col = f"force_min_cutoff_{section}"
        if force_col in out and cutoff_col in out:
            denom = out[cutoff_col].replace(0, np.nan)
            margin_cols[f"{section}_force_min_over_cutoff"] = out[force_col] / denom
    if margin_cols:
        out = pd.concat([out, pd.DataFrame(margin_cols, index=out.index)], axis=1)
    return out.copy()


def coverage_col(section: str, average: bool) -> str:
    base = f"accepted_{section}"
    return f"{base}_avg_per_case" if average else base


def count_label(average: bool) -> str:
    return "average per label/mode case" if average else "sum over label/mode cases"


def count_stem_suffix(average: bool) -> str:
    return "avg_per_case" if average else "sum"


def count_fmt(average: bool) -> str:
    return ".2g" if average else ".0f"


def grid_shape(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    return math.ceil(count / 2), 2


def potential_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["potential_set"].dropna().unique(), key=natural_key)


def force_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["force_profile"].dropna().unique(), key=natural_key)


def rmsd_order(df: pd.DataFrame) -> list[float]:
    values = pd.to_numeric(df["rmsd_max_cutoff"], errors="coerce").dropna().unique()
    return sorted(values, reverse=True)


def potential_label_map(df: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for pot in potential_order(df):
        row = df[df["potential_set"] == pot].iloc[0]
        labels[pot] = potential_label(row)
    return labels


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the explicit canvas so centered suptitles stay visually centered in
    # both PDF and PNG. Tight cropping can shift titles when colorbars or long
    # y tick labels make the bounding box asymmetric.
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    plt.close(fig)


def cleanup_stale_plot_outputs(out_dir: Path) -> None:
    stale_stems = [
        "01_loose_screen_coverage",
        "02_accepted_angles_heatmap",
        "03_accepted_dihedrals_heatmap",
        "04_rmse_p90_heatmap",
        "04b_rmse_max_heatmap",
        "05_coverage_error_tradeoff",
        "05_coverage_error_tradeoff_sum",
        "05_coverage_error_tradeoff_avg_per_case",
        "06_recommended_plot_review_candidates",
    ]
    stale_patterns = [
        "07_case_heatmap_*.pdf",
        "07_case_heatmap_*.png",
        "08_threshold_curves_*.pdf",
        "08_threshold_curves_*.png",
    ]
    for stem in stale_stems:
        for suffix in (".pdf", ".png"):
            path = out_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
    for pattern in stale_patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    generated_subdirs = [
        out_dir / "case_heatmaps",
        out_dir / "threshold_curves",
        out_dir / "rmse_metrics" / "threshold_curves",
        out_dir / "rmse_metrics" / "recommended_candidates",
        out_dir / "force_metrics" / "threshold_curves",
        out_dir / "force_metrics" / "recommended_candidates",
    ]
    for subdir in generated_subdirs:
        if not subdir.exists():
            continue
        for pattern in ("*.pdf", "*.png"):
            for path in subdir.glob(pattern):
                if path.is_file():
                    path.unlink()


def annotate_heatmap(ax: plt.Axes, data: np.ndarray, fmt: str, threshold: float) -> None:
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = data[y, x]
            if not np.isfinite(value):
                continue
            color = "white" if value > threshold else "#1f2933"
            ax.text(x, y, format(value, fmt), ha="center", va="center", fontsize=8, color=color)


def annotation_box() -> dict[str, object]:
    return {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
    }


def plot_heatmap_grid(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    title: str,
    colorbar_label: str,
    stem: str,
    cmap_name: str,
    fmt: str,
    log_scale: bool = False,
) -> None:
    pots = potential_order(df)
    forces = force_order(df)
    rmsds = rmsd_order(df)
    labels = potential_label_map(df)
    values = pd.to_numeric(df[metric], errors="coerce")
    if log_scale:
        values = values.where(values > 0)
    vmin = float(np.nanmin(values)) if values.notna().any() else 0.0
    vmax = float(np.nanmax(values)) if values.notna().any() else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin * 10.0 if log_scale and vmin > 0 else vmin + 1.0
    norm = LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=vmax) if log_scale else None

    nrows, ncols = grid_shape(len(pots))
    fig_height = 4.15 * nrows + 0.85
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.25 * ncols, fig_height), squeeze=False)

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad("#eeeeee")
    image = None
    threshold = vmin + (vmax - vmin) * 0.62

    for plot_idx, (ax, pot) in enumerate(zip(axes.ravel(), pots)):
        col_idx = plot_idx % ncols
        sub = df[df["potential_set"] == pot]
        pivot = sub.pivot_table(index="force_profile", columns="rmsd_max_cutoff", values=metric, aggfunc="first")
        pivot = pivot.reindex(index=forces, columns=rmsds)
        data = pivot.to_numpy(dtype=float)
        if log_scale:
            masked = np.ma.masked_where((~np.isfinite(data)) | (data <= 0), data)
        else:
            masked = np.ma.masked_invalid(data)
        if norm is not None:
            image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
        else:
            image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if log_scale:
            contrast = math.sqrt(max(vmin, np.finfo(float).tiny) * vmax)
        else:
            contrast = threshold
        annotate_heatmap(ax, data, fmt, contrast)
        ax.set_title(labels[pot], fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(range(len(rmsds)))
        ax.set_xticklabels([clean_number(x) for x in rmsds], fontsize=8)
        ax.set_yticks(range(len(forces)))
        ax.set_yticklabels([compact_force(x) for x in forces], fontsize=8)
        ax.set_xlabel("RMSD/RMSE cutoff")
        ax.set_ylabel("force profile" if col_idx == 0 else "")
        ax.set_xticks(np.arange(-0.5, len(rmsds), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(forces), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)

    for ax in axes.ravel()[len(pots) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.5, y=SUPTITLE_Y, ha="center")
    if nrows == 1:
        top = 0.78
        bottom = 0.105
    elif nrows == 2:
        top = 0.87
        bottom = 0.075
    else:
        top = 0.89
        bottom = 0.065
    fig.subplots_adjust(left=0.105, right=0.89, top=top, bottom=bottom, wspace=0.28, hspace=0.84)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.025)
        cbar.set_label(colorbar_label)
    save_figure(fig, out_dir, stem)


def plot_loose_screen_coverage(df: pd.DataFrame, out_dir: Path, average: bool = False) -> None:
    max_rmsd = df["rmsd_max_cutoff"].max()
    loose_force = force_order(df)[0]
    sub = df[(df["rmsd_max_cutoff"] == max_rmsd) & (df["force_profile"] == loose_force)].copy()
    sub = sub.sort_values("potential_set", key=lambda s: s.map(natural_key))
    labels = [potential_label(row) for _, row in sub.iterrows()]

    fig_width = max(16.2, len(sub) * 2.45)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 7.25), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    sections = [coverage_col(section, average) for section in ACCEPTED_SECTIONS]
    for section in sections:
        values = sub[section].fillna(0).to_numpy(dtype=float)
        base_section = section.replace("_avg_per_case", "")
        ax.bar(x, values, bottom=bottom, label=base_section.replace("accepted_", ""), color=SECTION_COLORS[base_section])
        bottom += values
    for idx, row in enumerate(sub.itertuples()):
        angle_value = getattr(row, coverage_col("angles", average))
        dihedral_value = getattr(row, coverage_col("dihedrals", average))
        ax.text(
            idx,
            bottom[idx] + max(bottom) * 0.038,
            f"A {angle_value:.2g}\nD {dihedral_value:.2g}" if average else f"A {int(row.accepted_angles)}\nD {int(row.accepted_dihedrals)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#263238",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(f"accepted term count ({count_label(average)})")
    ax.set_ylim(0, max(bottom) * 1.23)
    ax.set_title(f"Loosest screen coverage ({rmsd_label(max_rmsd)}, {compact_force(loose_force)})", fontweight="bold", pad=26)
    ax.legend(ncols=5, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee2", linewidth=0.8, alpha=0.8)

    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, sub["zero_angles_cases"].fillna(0), width=width, color="#007c89", label="zero-angle cases")
    ax.bar(x + width / 2, sub["zero_dihedrals_cases"].fillna(0), width=width, color="#cc7a29", label="zero-dihedral cases")
    ax.axhline(24, color="#5f6b73", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(sub) - 0.45, 24.35, "all 24 cases", ha="right", va="bottom", fontsize=8, color="#5f6b73")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("number of label/mode cases")
    ax.set_ylim(0, 27)
    ax.set_title("Coverage failures", fontweight="bold", pad=26)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee2", linewidth=0.8, alpha=0.8)

    fig.suptitle(
        f"Function-family coverage under the loosest filter ({count_label(average)})",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out_dir, f"01_loose_screen_coverage_{count_stem_suffix(average)}")


def color_map_for_potentials(df: pd.DataFrame) -> dict[str, str]:
    return {pot: POTENTIAL_PALETTE[i % len(POTENTIAL_PALETTE)] for i, pot in enumerate(potential_order(df))}


def plot_tradeoff_scatter(df: pd.DataFrame, out_dir: Path, average: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    colors = color_map_for_potentials(df)
    angle_col = coverage_col("angles", average)
    dihedral_col = coverage_col("dihedrals", average)
    for pot in potential_order(df):
        sub = df[df["potential_set"] == pot]
        size_scale = 22.0 if average else 1.55
        sizes = 28 + sub[dihedral_col].fillna(0).to_numpy(dtype=float) * size_scale
        ax.scatter(
            sub[angle_col],
            sub["all_rmse_p90"],
            s=sizes,
            c=colors[pot],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.72,
            label=potential_label(sub.iloc[0]).replace("\n", " "),
        )

    ax.axhline(6.0, color="#5d6770", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(1.0, 6.08, "RMSE/RMSD 6 guide", fontsize=8, color="#5d6770", va="bottom")

    annotations = {
        "P2_a10_d3_i1_R6_F2_medium": ("P2 R6 F2\nA10/D3 candidate", (54, -34)),
        "P2_a10_d3_i1_R7p5_F2_medium": ("P2 R7.5 F2\nvisual check", (-58, 28)),
        "P5_a10_d1_i1_R6_F2_medium": ("P5 R6 F2\ndihedral comparator", (-18, 42)),
        "P0_bartender_R6_F2_medium": ("P0 R6 F2\nBartender baseline", (42, 10)),
        "P1_a2_d3_i1_R10_F0_loose": ("P1 R10 F0\nA2/D3 loose", (42, 42)),
        "P4_a2_d1_i1_R10_F0_loose": ("P4 R10 F0\nA2/D1 loose", (42, -8)),
    }
    for variant_id, (label, offset) in annotations.items():
        hit = df[df["variant_id"] == variant_id]
        if hit.empty:
            continue
        row = hit.iloc[0]
        if average and float(row.get(angle_col, 0) or 0) <= 0:
            label = label.replace("accepted angles", "accepted angles/case")
        ax.annotate(
            label,
            xy=(row[angle_col], row["all_rmse_p90"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            bbox=annotation_box(),
            arrowprops={"arrowstyle": "-", "color": "#6b737b", "linewidth": 0.8},
        )

    ax.set_xlabel(f"accepted angle terms ({count_label(average)})")
    ax.set_ylabel("P90 fit RMSE/RMSD of accepted terms")
    ax.set_title(f"Coverage versus fit-error tradeoff ({count_label(average)})", fontsize=15, fontweight="bold", pad=18)
    ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="potential set")
    ax.text(
        0.62,
        0.02,
        f"marker size = accepted dihedral terms ({count_label(average)})",
        transform=ax.transAxes,
        fontsize=8,
        color="#56616a",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    save_figure(fig, out_dir, f"05_coverage_error_tradeoff_{count_stem_suffix(average)}")


def read_recommended_variants(result_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    recommended = result_dir / "tables" / "recommended_plot_review_variants.tsv"
    if recommended.exists():
        ids = [line.strip() for line in recommended.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
        selected = df[df["variant_id"].isin(ids)].copy()
        selected["_order"] = selected["variant_id"].map({variant_id: i for i, variant_id in enumerate(ids)})
        return selected.sort_values("_order")

    # When no manually curated list exists, keep every potential family for the
    # three middle cutoffs/profiles. This is important for all-candidate sweeps,
    # where commented funct-2 candidates may become strong competitors.
    keep_pots = potential_order(df)
    keep_forces = ["F1_gentle", "F2_medium", "F3_angle_strict"]
    selected = df[
        df["potential_set"].isin(keep_pots)
        & df["force_profile"].isin(keep_forces)
        & df["rmsd_max_cutoff"].isin([7.5, 6.0, 5.0])
    ].copy()
    selected["_pot_order"] = selected["potential_set"].map({pot: i for i, pot in enumerate(keep_pots)})
    selected["_force_order"] = selected["force_profile"].map({force: i for i, force in enumerate(keep_forces)})
    selected["_rmsd_order"] = selected["rmsd_max_cutoff"].map({7.5: 0, 6.0: 1, 5.0: 2})
    return selected.sort_values(["_pot_order", "_rmsd_order", "_force_order"])


def plot_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path, average: bool = False) -> None:
    selected = read_recommended_variants(result_dir, df)
    if selected.empty:
        return
    labels = [compact_variant_label(row) for _, row in selected.iterrows()]
    y = np.arange(len(selected))
    colors = [color_map_for_potentials(df)[pot] for pot in selected["potential_set"]]

    metrics = [
        (coverage_col("angles", average), f"accepted angles\n({count_label(average)})"),
        (coverage_col("dihedrals", average), f"accepted dihedrals\n({count_label(average)})"),
        ("all_rmse_p90", "P90 RMSE/RMSD"),
        ("all_rmse_max", "max RMSE/RMSD"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17.5, max(8.5, len(selected) * 0.34)), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        values = pd.to_numeric(selected[metric], errors="coerce").fillna(0)
        ax.barh(y, values, color=colors, alpha=0.9)
        ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
        ax.grid(axis="x", color="#d9dee2", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if metric.startswith("all_rmse"):
            ax.axvline(6.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
        for idx, value in enumerate(values):
            ax.text(float(value) + max(values.max() * 0.012, 0.08), idx, f"{value:g}", va="center", fontsize=7)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    sweep_title = result_dir.name.replace("_", " ").title()
    fig.suptitle(
        f"Recommended plot-review candidates from {sweep_title} ({count_label(average)})",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, out_dir, f"06_recommended_plot_review_candidates_{count_stem_suffix(average)}")


def metric_available(df: pd.DataFrame, metric: str) -> bool:
    if metric not in df:
        return False
    return pd.to_numeric(df[metric], errors="coerce").notna().any()


def plot_rmse_metric_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        (
            "all_rmse_p90",
            "All accepted terms: P90 fit RMSE/RMSD across threshold grid",
            "P90 RMSE/RMSD",
            "04_all_rmse_p90_heatmap",
            "magma_r",
        ),
        (
            "all_rmse_max",
            "All accepted terms: maximum fit RMSE/RMSD across threshold grid",
            "max RMSE/RMSD",
            "04b_all_rmse_max_heatmap",
            "magma_r",
        ),
        (
            "angles_rmse_p90",
            "Angles: P90 fit RMSE/RMSD across threshold grid",
            "P90 angle RMSE/RMSD",
            "04c_angles_rmse_p90_heatmap",
            "magma_r",
        ),
        (
            "angles_rmse_max",
            "Angles: maximum fit RMSE/RMSD across threshold grid",
            "max angle RMSE/RMSD",
            "04d_angles_rmse_max_heatmap",
            "magma_r",
        ),
        (
            "dihedrals_rmse_p90",
            "Dihedrals: P90 fit RMSE/RMSD across threshold grid",
            "P90 dihedral RMSE/RMSD",
            "04e_dihedrals_rmse_p90_heatmap",
            "magma_r",
        ),
        (
            "dihedrals_rmse_max",
            "Dihedrals: maximum fit RMSE/RMSD across threshold grid",
            "max dihedral RMSE/RMSD",
            "04f_dihedrals_rmse_max_heatmap",
            "magma_r",
        ),
    ]
    for metric, title, colorbar_label, stem, cmap_name in specs:
        if not metric_available(df, metric):
            continue
        plot_heatmap_grid(
            df,
            out_dir,
            metric,
            title,
            colorbar_label,
            stem,
            cmap_name,
            ".2g",
        )


def plot_rmse_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path) -> None:
    selected = read_recommended_variants(result_dir, df)
    if selected.empty:
        return
    metrics = [
        ("all_rmse_p90", "all-term P90\nRMSE/RMSD"),
        ("all_rmse_max", "all-term max\nRMSE/RMSD"),
        ("angles_rmse_p90", "angle P90\nRMSE/RMSD"),
        ("angles_rmse_max", "angle max\nRMSE/RMSD"),
        ("dihedrals_rmse_p90", "dihedral P90\nRMSE/RMSD"),
        ("dihedrals_rmse_max", "dihedral max\nRMSE/RMSD"),
    ]
    metrics = [spec for spec in metrics if metric_available(selected, spec[0])]
    if not metrics:
        return

    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    chunk_size = 18
    chunks = [selected.iloc[start : start + chunk_size].copy() for start in range(0, len(selected), chunk_size)]
    chunk_dir = out_dir / "recommended_candidates"
    if chunk_dir.exists():
        for stale in chunk_dir.glob("06_rmse_recommended_candidates_part*.*"):
            if stale.is_file():
                stale.unlink()

    for chunk_index, chunk in enumerate(chunks, start=1):
        labels = [one_line_variant_label(row) for _, row in chunk.iterrows()]
        y = np.arange(len(chunk))
        colors = [color_map_for_potentials(df)[pot] for pot in chunk["potential_set"]]
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.35 * ncols, max(7.5, len(chunk) * 0.58)),
            sharey=True,
            squeeze=False,
        )

        for ax, (metric, title) in zip(axes.ravel(), metrics):
            values = pd.to_numeric(chunk[metric], errors="coerce")
            valid = values.notna()
            ax.barh(y[valid.to_numpy()], values[valid], color=np.array(colors, dtype=object)[valid.to_numpy()], alpha=0.9)
            ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
            ax.grid(axis="x", color="#d9dee2", linewidth=0.8, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axvline(6.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
            for idx, value in enumerate(values):
                if pd.isna(value):
                    continue
                ax.text(float(value) + max(float(values.max()) * 0.012, 0.08), idx, f"{value:.2g}", va="center", fontsize=7)

        for ax in axes.ravel()[len(metrics) :]:
            ax.axis("off")
        axes.ravel()[0].set_yticks(y)
        axes.ravel()[0].set_yticklabels(labels, fontsize=7.2)
        axes.ravel()[0].invert_yaxis()
        fig.suptitle(
            f"RMSE/RMSD comparison for plot-review candidates (part {chunk_index}/{len(chunks)})",
            fontsize=16,
            fontweight="bold",
            x=0.5,
            y=SUPTITLE_Y,
            ha="center",
        )
        fig.subplots_adjust(left=0.25, right=0.985, top=0.855, bottom=0.055, wspace=0.3, hspace=0.68)
        save_figure(fig, chunk_dir, f"06_rmse_recommended_candidates_part{chunk_index:02d}")


def plot_rmse_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str) -> None:
    selected = df[df["potential_set"] == potential_set].copy()
    if selected.empty:
        return
    metrics = [
        ("all_rmse_p90", "all-term P90 RMSE/RMSD"),
        ("all_rmse_max", "all-term max RMSE/RMSD"),
        ("angles_rmse_p90", "angle P90 RMSE/RMSD"),
        ("angles_rmse_max", "angle max RMSE/RMSD"),
        ("dihedrals_rmse_p90", "dihedral P90 RMSE/RMSD"),
        ("dihedrals_rmse_max", "dihedral max RMSE/RMSD"),
    ]
    metrics = [spec for spec in metrics if metric_available(selected, spec[0])]
    if not metrics:
        return

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.7, 4.75 * nrows), squeeze=False)
    colors = plt.colormaps["tab10"]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for idx, force in enumerate(force_order(selected)):
            sub = selected[selected["force_profile"] == force].sort_values("rmsd_max_cutoff", ascending=False)
            values = pd.to_numeric(sub[metric], errors="coerce")
            ax.plot(
                sub["rmsd_max_cutoff"],
                values,
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=colors(idx),
                label=compact_force(force),
            )
        ax.invert_xaxis()
        ax.axhline(6.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_xlabel("RMSD/RMSE cutoff: loose to strict")
        ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
        ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes.ravel()[min(len(metrics), len(axes.ravel())) - 1].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(
        f"RMSE/RMSD threshold curves for {potential_set}",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, out_dir, f"07_rmse_threshold_curves_{potential_set}")


def plot_rmse_metric_suite(df: pd.DataFrame, result_dir: Path, out_dir: Path) -> None:
    rmse_dir = out_dir / "rmse_metrics"
    plot_rmse_metric_heatmaps(df, rmse_dir)
    for average in (False, True):
        plot_tradeoff_scatter(df, rmse_dir, average=average)
    plot_rmse_recommended_candidates(df, result_dir, rmse_dir)

    curve_dir = rmse_dir / "threshold_curves"
    for potential_set in potential_order(df):
        plot_rmse_threshold_curves(df, curve_dir, potential_set)


def plot_force_metric_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        (
            "all_force_p90",
            "All accepted terms: P90 force metric across threshold grid",
            "P90 force metric",
            "09_all_force_p90_heatmap",
            "cividis",
        ),
        (
            "all_force_max",
            "All accepted terms: maximum force metric across threshold grid",
            "max force metric",
            "09b_all_force_max_heatmap",
            "cividis",
        ),
        (
            "angles_force_min_over_cutoff",
            "Angles: minimum force metric divided by cutoff",
            "min angle force / cutoff",
            "10_angles_force_min_over_cutoff_heatmap",
            "viridis",
        ),
        (
            "angles_force_p90",
            "Angles: P90 force metric across threshold grid",
            "P90 angle force metric",
            "10b_angles_force_p90_heatmap",
            "viridis",
        ),
        (
            "dihedrals_force_min_over_cutoff",
            "Dihedrals: minimum force metric divided by cutoff",
            "min dihedral force / cutoff",
            "11_dihedrals_force_min_over_cutoff_heatmap",
            "YlOrBr",
        ),
        (
            "dihedrals_force_p90",
            "Dihedrals: P90 force metric across threshold grid",
            "P90 dihedral force metric",
            "11b_dihedrals_force_p90_heatmap",
            "YlOrBr",
        ),
    ]
    for metric, title, colorbar_label, stem, cmap_name in specs:
        if not metric_available(df, metric):
            continue
        plot_heatmap_grid(
            df,
            out_dir,
            metric,
            title,
            colorbar_label,
            stem,
            cmap_name,
            ".2g",
            log_scale=True,
        )


def plot_force_error_scatter(df: pd.DataFrame, out_dir: Path, section: str) -> None:
    margin_col = f"{section}_force_min_over_cutoff"
    p90_col = f"{section}_force_p90"
    count_col = f"accepted_{section}"
    if not (metric_available(df, margin_col) and metric_available(df, p90_col) and count_col in df):
        return

    selected = df.copy()
    selected[margin_col] = pd.to_numeric(selected[margin_col], errors="coerce")
    selected[p90_col] = pd.to_numeric(selected[p90_col], errors="coerce")
    selected[count_col] = pd.to_numeric(selected[count_col], errors="coerce")
    colors = color_map_for_potentials(selected)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    ax = axes[0]
    for pot in potential_order(selected):
        sub = selected[(selected["potential_set"] == pot) & selected[margin_col].notna() & selected["all_rmse_p90"].notna()]
        if sub.empty:
            continue
        sizes = 28 + sub[count_col].fillna(0).to_numpy(dtype=float) * 1.2
        ax.scatter(
            sub[margin_col],
            sub["all_rmse_p90"],
            s=sizes,
            c=colors[pot],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.72,
            label=potential_label(sub.iloc[0]).replace("\n", " "),
        )
    ax.set_xscale("log")
    ax.axvline(1.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(6.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(1.05, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, "force cutoff", fontsize=8, color="#5d6770")
    ax.text(ax.get_xlim()[0] * 1.08, 6.08, "RMSE/RMSD 6 guide", fontsize=8, color="#5d6770", va="bottom")
    ax.set_xlabel(f"{section} minimum force metric / cutoff")
    ax.set_ylabel("P90 fit RMSE/RMSD of accepted terms")
    ax.set_title("Force threshold margin versus fit error", fontweight="bold", fontsize=10, pad=18)
    ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for pot in potential_order(selected):
        sub = selected[(selected["potential_set"] == pot) & selected[p90_col].notna() & selected[count_col].notna()]
        if sub.empty:
            continue
        ax.scatter(
            sub[count_col],
            sub[p90_col],
            s=58,
            c=colors[pot],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.72,
            label=potential_label(sub.iloc[0]).replace("\n", " "),
        )
    ax.set_yscale("log")
    ax.set_xlabel(f"accepted {section} count (sum over label/mode cases)")
    ax.set_ylabel(f"P90 {section} force metric")
    ax.set_title("Coverage versus force magnitude", fontweight="bold", fontsize=10, pad=18)
    ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="potential set")
    fig.suptitle(
        f"{section.capitalize()} force-metric diagnostics",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 0.84, 0.88))
    save_figure(fig, out_dir, f"12_{section}_force_metric_tradeoff")


def plot_force_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path) -> None:
    selected = read_recommended_variants(result_dir, df)
    if selected.empty:
        return
    metrics = [
        ("angles_force_min_over_cutoff", "angle min force\n/ cutoff", True),
        ("angles_force_p90", "angle P90\nforce metric", True),
        ("dihedrals_force_min_over_cutoff", "dihedral min force\n/ cutoff", True),
        ("dihedrals_force_p90", "dihedral P90\nforce metric", True),
        ("all_force_p90", "all-term P90\nforce metric", True),
        ("all_force_max", "all-term max\nforce metric", True),
    ]
    metrics = [spec for spec in metrics if metric_available(selected, spec[0])]
    if not metrics:
        return

    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    chunk_size = 18
    chunks = [selected.iloc[start : start + chunk_size].copy() for start in range(0, len(selected), chunk_size)]
    chunk_dir = out_dir / "recommended_candidates"
    for stale in out_dir.glob("13_recommended_force_metric_candidates.*"):
        if stale.is_file():
            stale.unlink()
    if chunk_dir.exists():
        for stale in chunk_dir.glob("13_recommended_force_metric_candidates_part*.*"):
            if stale.is_file():
                stale.unlink()

    for chunk_index, chunk in enumerate(chunks, start=1):
        labels = [one_line_variant_label(row) for _, row in chunk.iterrows()]
        y = np.arange(len(chunk))
        colors = [color_map_for_potentials(df)[pot] for pot in chunk["potential_set"]]
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.35 * ncols, max(7.5, len(chunk) * 0.58)),
            sharey=True,
            squeeze=False,
        )

        for ax, (metric, title, log_axis) in zip(axes.ravel(), metrics):
            values = pd.to_numeric(chunk[metric], errors="coerce")
            valid = values.notna() & (values > 0 if log_axis else True)
            ax.barh(y[valid.to_numpy()], values[valid], color=np.array(colors, dtype=object)[valid.to_numpy()], alpha=0.9)
            ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
            ax.grid(axis="x", color="#d9dee2", linewidth=0.8, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if log_axis:
                ax.set_xscale("log")
            if metric.endswith("_over_cutoff"):
                ax.axvline(1.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
            for idx, value in enumerate(values):
                if pd.isna(value) or (log_axis and value <= 0):
                    continue
                ax.text(float(value) * (1.06 if log_axis else 1.01), idx, f"{value:.2g}", va="center", fontsize=7)

        for ax in axes.ravel()[len(metrics) :]:
            ax.axis("off")
        axes.ravel()[0].set_yticks(y)
        axes.ravel()[0].set_yticklabels(labels, fontsize=7.2)
        axes.ravel()[0].invert_yaxis()
        fig.suptitle(
            f"Force-metric comparison for plot-review candidates (part {chunk_index}/{len(chunks)})",
            fontsize=16,
            fontweight="bold",
            x=0.5,
            y=SUPTITLE_Y,
            ha="center",
        )
        fig.subplots_adjust(left=0.25, right=0.985, top=0.855, bottom=0.055, wspace=0.3, hspace=0.68)
        save_figure(fig, chunk_dir, f"13_recommended_force_metric_candidates_part{chunk_index:02d}")


def plot_force_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str) -> None:
    selected = df[df["potential_set"] == potential_set].copy()
    if selected.empty:
        return
    metrics = [
        ("angles_force_min_over_cutoff", "angle min force / cutoff"),
        ("angles_force_p90", "angle P90 force metric"),
        ("dihedrals_force_min_over_cutoff", "dihedral min force / cutoff"),
        ("dihedrals_force_p90", "dihedral P90 force metric"),
    ]
    metrics = [spec for spec in metrics if metric_available(selected, spec[0])]
    if not metrics:
        return

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.7, 4.75 * nrows), squeeze=False)
    colors = plt.colormaps["tab10"]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for idx, force in enumerate(force_order(selected)):
            sub = selected[selected["force_profile"] == force].sort_values("rmsd_max_cutoff", ascending=False)
            values = pd.to_numeric(sub[metric], errors="coerce")
            ax.plot(
                sub["rmsd_max_cutoff"],
                values,
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=colors(idx),
                label=compact_force(force),
            )
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.set_xlabel("RMSD/RMSE cutoff: loose to strict")
        ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
        ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if metric.endswith("_over_cutoff"):
            ax.axhline(1.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes.ravel()[min(len(metrics), len(axes.ravel())) - 1].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(
        f"Force-metric threshold curves for {potential_set}",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, out_dir, f"14_force_threshold_curves_{potential_set}")


def plot_force_metric_suite(df: pd.DataFrame, result_dir: Path, out_dir: Path) -> None:
    force_dir = out_dir / "force_metrics"
    plot_force_metric_heatmaps(df, force_dir)
    for section in ["angles", "dihedrals"]:
        plot_force_error_scatter(df, force_dir, section)
    plot_force_recommended_candidates(df, result_dir, force_dir)

    curve_dir = force_dir / "threshold_curves"
    for potential_set in potential_order(df):
        plot_force_threshold_curves(df, curve_dir, potential_set)


def plot_case_heatmap(case_df: pd.DataFrame, out_dir: Path, variant_id: str) -> None:
    selected = case_df[case_df["variant_id"] == variant_id].copy()
    if selected.empty:
        return
    for col in ["accepted_angles", "accepted_dihedrals", "all_rmse_max"]:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")

    metrics = [
        ("accepted_angles", "accepted angles", "YlGnBu", ".0f"),
        ("accepted_dihedrals", "accepted dihedrals", "YlOrBr", ".0f"),
        ("all_rmse_max", "max RMSE/RMSD", "magma_r", ".2g"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17.5, 5.15), squeeze=False)
    axes_list = axes.ravel()

    for ax, (metric, title, cmap_name, fmt) in zip(axes_list, metrics):
        pivot = selected.pivot_table(index="label", columns="mode", values=metric, aggfunc="first")
        modes = [mode for mode in MODE_ORDER if mode in pivot.columns]
        labels = [label for label in LABEL_ORDER if label in pivot.index]
        pivot = pivot.reindex(index=labels, columns=modes)
        data = pivot.to_numpy(dtype=float)
        vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#eeeeee")
        image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(ax, data, fmt, vmin + (vmax - vmin) * 0.62)
        ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(modes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(
        f"Case-level behavior for {variant_id}",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out_dir, f"07_case_heatmap_{variant_id}")


def case_heatmap_variant_ids(result_dir: Path, variant_df: pd.DataFrame, requested_variant: str | None) -> list[str]:
    selected = read_recommended_variants(result_dir, variant_df)
    ids = list(dict.fromkeys(str(value) for value in selected["variant_id"].tolist()))
    if requested_variant and requested_variant not in ids:
        ids.append(requested_variant)
    return ids


def plot_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str, average: bool = False) -> None:
    selected = df[df["potential_set"] == potential_set].copy()
    if selected.empty:
        return
    metrics = [
        (coverage_col("angles", average), f"accepted angles\n({count_label(average)})"),
        (coverage_col("dihedrals", average), f"accepted dihedrals\n({count_label(average)})"),
        ("all_rmse_p90", "P90 RMSE/RMSD"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.8, 5.15))
    colors = plt.colormaps["tab10"]
    for ax, (metric, title) in zip(axes, metrics):
        for idx, force in enumerate(force_order(selected)):
            sub = selected[selected["force_profile"] == force].sort_values("rmsd_max_cutoff", ascending=False)
            ax.plot(
                sub["rmsd_max_cutoff"],
                sub[metric],
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=colors(idx),
                label=compact_force(force),
            )
        ax.invert_xaxis()
        ax.set_xlabel("RMSD/RMSE cutoff: loose to strict")
        ax.set_title(title, fontweight="bold", fontsize=10, pad=18)
        ax.grid(color="#d9dee2", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "all_rmse_p90":
            ax.axhline(6.0, color="#5d6770", linestyle="--", linewidth=1.0, alpha=0.75)
    axes[-1].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(
        f"Threshold curves for {potential_set} ({count_label(average)})",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=SUPTITLE_Y,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out_dir, f"08_threshold_curves_{potential_set}_{count_stem_suffix(average)}")


def write_plot_readme(out_dir: Path, result_dir: Path) -> None:
    title = result_dir.name.replace("_", " ").title()
    lines = [
        f"# {title} Plots",
        "",
        "Coverage-count plots are written in two forms:",
        "",
        "- `*_sum.*`: accepted counts summed over all label/mode cases in the variant.",
        "- `*_avg_per_case.*`: accepted counts divided by `case_count`, i.e. average accepted terms per label/mode case.",
        "",
        "Suggested reading order:",
        "",
        "1. `01_loose_screen_coverage_sum.pdf` and `01_loose_screen_coverage_avg_per_case.pdf`: function-family coverage at the loosest filter.",
        "2. `02_accepted_angles_heatmap_sum.pdf` and `02_accepted_angles_heatmap_avg_per_case.pdf`: accepted angle counts across RMSD/RMSE cutoffs and force profiles.",
        "3. `03_accepted_dihedrals_heatmap_sum.pdf` and `03_accepted_dihedrals_heatmap_avg_per_case.pdf`: accepted dihedral counts across the same grid.",
        "4. `rmse_metrics/`: RMSE/RMSD diagnostics. This folder contains all-term, angle, and dihedral RMSE/RMSD heatmaps, coverage-error scatter plots, candidate bar plots, and RMSE/RMSD threshold curves.",
        "5. `force_metrics/`: force-metric diagnostics. These use log-scaled axes or color maps because force constants span several orders of magnitude. The `*_over_cutoff` plots report the minimum accepted force metric divided by the configured cutoff, so values above 1 passed the force threshold. Candidate bar plots are split under `force_metrics/recommended_candidates/` so labels stay readable.",
        "6. `06_recommended_plot_review_candidates_sum.pdf` and `06_recommended_plot_review_candidates_avg_per_case.pdf`: compact mixed coverage/RMSE comparison of the focused plot-review candidates. If no manual candidate list exists, this includes all potential families at RMSD/RMSE 7.5, 6.0, and 5.0 with F1-F3 force profiles.",
        "7. `case_heatmaps/`: label/mode-level heatmaps for the candidate variants selected by this script.",
        "8. `threshold_curves/`: count/RMSE threshold-response curves for each potential family selected by this script.",
        "",
        "PNG copies are also written for quick image preview.",
        "",
        f"Source result directory: `{result_dir}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Analyze result directory, e.g. analyze/results/01_summary_sweep")
    parser.add_argument("--out-dir", type=Path, default=None, help="Plot output directory. Default: result_dir/plots_summary")
    parser.add_argument(
        "--case-variant",
        default="P2_a10_d3_i1_R6_F2_medium",
        help="Variant id for the case-level heatmap.",
    )
    parser.add_argument(
        "--curve-potential",
        default="P2_a10_d3_i1",
        help="Potential set id for threshold-curve plots.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    table_dir = result_dir / "tables"
    out_dir = args.out_dir.resolve() if args.out_dir else result_dir / "plots_summary"

    variant_path = table_dir / "variant_summary.csv"
    case_path = table_dir / "case_summary.csv"
    if not variant_path.exists():
        raise SystemExit(f"Missing variant summary: {variant_path}")
    if not case_path.exists():
        raise SystemExit(f"Missing case summary: {case_path}")

    variants = enrich_variants(pd.read_csv(variant_path))
    cases = pd.read_csv(case_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_plot_outputs(out_dir)

    for average in (False, True):
        suffix = count_stem_suffix(average)
        label_suffix = count_label(average)
        plot_loose_screen_coverage(variants, out_dir, average=average)
        plot_heatmap_grid(
            variants,
            out_dir,
            coverage_col("angles", average),
            f"Accepted angle counts across threshold grid ({label_suffix})",
            f"accepted angles ({label_suffix})",
            f"02_accepted_angles_heatmap_{suffix}",
            "YlGnBu",
            count_fmt(average),
        )
        plot_heatmap_grid(
            variants,
            out_dir,
            coverage_col("dihedrals", average),
            f"Accepted dihedral counts across threshold grid ({label_suffix})",
            f"accepted dihedrals ({label_suffix})",
            f"03_accepted_dihedrals_heatmap_{suffix}",
            "YlOrBr",
            count_fmt(average),
        )
    plot_rmse_metric_suite(variants, result_dir, out_dir)
    plot_force_metric_suite(variants, result_dir, out_dir)
    for average in (False, True):
        plot_recommended_candidates(variants, result_dir, out_dir, average=average)

    case_out_dir = out_dir / "case_heatmaps"
    for variant_id in case_heatmap_variant_ids(result_dir, variants, args.case_variant):
        plot_case_heatmap(cases, case_out_dir, variant_id)

    curve_out_dir = out_dir / "threshold_curves"
    curve_potentials = list(dict.fromkeys(potential_order(variants) + [args.curve_potential]))
    for potential_set in curve_potentials:
        for average in (False, True):
            plot_threshold_curves(variants, curve_out_dir, potential_set, average=average)
    write_plot_readme(out_dir, result_dir)

    print(f"[plots] wrote {out_dir}")


if __name__ == "__main__":
    main()

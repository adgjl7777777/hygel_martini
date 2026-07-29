#!/usr/bin/env python3
"""Audit whether a sparse high-level reference can qualify an xTB ensemble.

The routines in this module deliberately keep four questions separate:

1. Does the reference preserve the xTB relative-energy ordering?
2. Is a reported optimized geometry stationary at the reference level?
3. Do independently optimized endpoints form one structural family?
4. Is xTB-to-reference reweighting supported by sufficient overlap?

These checks do not fit Martini parameters and do not turn xTB into a DFT
ground truth.  They produce explicit, machine-readable gates that can be used
before a bounded parameter-refinement step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


R_KJ_MOL_K = 0.00831446261815324


def _finite_vector(values: Sequence[float], name: str, minimum_size: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} value(s)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _pairwise_sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "pass", "passed"}:
            return True
        if normalized in {"0", "false", "no", "fail", "failed"}:
            return False
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"cannot parse boolean value: {value!r}")


def audit_relative_energies(
    xtb_energy_kj_mol: Sequence[float],
    reference_energy_kj_mol: Sequence[float],
    *,
    max_abs_error_kj_mol: float = 8.4,
    ordering_tolerance_kj_mol: float = 1.0e-8,
) -> dict[str, Any]:
    """Compare xTB and reference relative-energy landscapes.

    Both energy vectors are shifted to their own minimum.  The default gate
    requires the same minimum index, complete pairwise ordering agreement, and
    a maximum relative-energy error no larger than 8.4 kJ/mol.
    """

    xtb = _finite_vector(xtb_energy_kj_mol, "xtb_energy_kj_mol", minimum_size=2)
    reference = _finite_vector(
        reference_energy_kj_mol,
        "reference_energy_kj_mol",
        minimum_size=2,
    )
    if xtb.shape != reference.shape:
        raise ValueError("xTB and reference energy vectors must have the same length")
    if max_abs_error_kj_mol < 0 or ordering_tolerance_kj_mol < 0:
        raise ValueError("energy thresholds must be non-negative")

    xtb_relative = xtb - float(np.min(xtb))
    reference_relative = reference - float(np.min(reference))
    error = xtb_relative - reference_relative

    agreeing_pairs = 0
    total_pairs = 0
    disagreeing_pairs: list[list[int]] = []
    for left in range(xtb.size - 1):
        for right in range(left + 1, xtb.size):
            xtb_sign = _pairwise_sign(
                float(xtb_relative[left] - xtb_relative[right]),
                ordering_tolerance_kj_mol,
            )
            reference_sign = _pairwise_sign(
                float(reference_relative[left] - reference_relative[right]),
                ordering_tolerance_kj_mol,
            )
            total_pairs += 1
            if xtb_sign == reference_sign:
                agreeing_pairs += 1
            else:
                disagreeing_pairs.append([left, right])

    xtb_minimum_index = int(np.argmin(xtb_relative))
    reference_minimum_index = int(np.argmin(reference_relative))
    same_minimum = xtb_minimum_index == reference_minimum_index
    ordering_agreement = agreeing_pairs / total_pairs
    max_abs_error = float(np.max(np.abs(error)))
    passed = (
        same_minimum
        and not disagreeing_pairs
        and max_abs_error <= max_abs_error_kj_mol
    )

    return {
        "decision": "PASS" if passed else "FAIL",
        "same_minimum": same_minimum,
        "xtb_minimum_index": xtb_minimum_index,
        "reference_minimum_index": reference_minimum_index,
        "ordering_agreement_fraction": ordering_agreement,
        "agreeing_pairs": agreeing_pairs,
        "total_pairs": total_pairs,
        "disagreeing_pairs": disagreeing_pairs,
        "mae_kj_mol": float(np.mean(np.abs(error))),
        "rmse_kj_mol": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error_kj_mol": max_abs_error,
        "max_abs_error_gate_kj_mol": float(max_abs_error_kj_mol),
        "xtb_relative_energy_kj_mol": xtb_relative.tolist(),
        "reference_relative_energy_kj_mol": reference_relative.tolist(),
        "relative_energy_error_kj_mol": error.tolist(),
    }


def audit_gradient(
    gradient: Sequence[float] | Sequence[Sequence[float]],
    *,
    rms_threshold: float = 3.0e-5,
    max_threshold: float = 1.0e-4,
) -> dict[str, Any]:
    """Audit reference-level stationarity from Cartesian gradient components."""

    components = _finite_vector(np.asarray(gradient, dtype=float), "gradient")
    if rms_threshold < 0 or max_threshold < 0:
        raise ValueError("gradient thresholds must be non-negative")

    rms = float(np.sqrt(np.mean(np.square(components))))
    max_abs = float(np.max(np.abs(components)))
    stationary = rms <= rms_threshold and max_abs <= max_threshold
    return {
        "decision": "STATIONARY" if stationary else "NON_STATIONARY",
        "stationary": stationary,
        "component_count": int(components.size),
        "rms_gradient": rms,
        "max_abs_gradient": max_abs,
        "rms_threshold": float(rms_threshold),
        "max_threshold": float(max_threshold),
        "rms_ratio_to_threshold": (
            rms / rms_threshold if rms_threshold > 0 else math.inf
        ),
        "max_ratio_to_threshold": (
            max_abs / max_threshold if max_threshold > 0 else math.inf
        ),
    }


def audit_endpoint_family(
    entries: Iterable[Mapping[str, Any]],
    *,
    rmsd_threshold_nm: float = 0.05,
    energy_threshold_kj_mol: float = 2.0,
) -> dict[str, Any]:
    """Classify optimized endpoints as a single or multiple DFT family.

    Each entry must provide ``id``, ``rmsd_nm``, ``delta_energy_kj_mol``, and
    ``integrity``.  RMSD and energy values are distances from a declared
    representative endpoint, not arbitrary pairwise values.
    """

    if rmsd_threshold_nm < 0 or energy_threshold_kj_mol < 0:
        raise ValueError("endpoint thresholds must be non-negative")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        endpoint_id = str(entry.get("id", index))
        rmsd = float(entry["rmsd_nm"])
        delta_energy = float(entry["delta_energy_kj_mol"])
        integrity = _coerce_bool(entry["integrity"])
        if not math.isfinite(rmsd) or not math.isfinite(delta_energy):
            raise ValueError(f"endpoint {endpoint_id!r} contains a non-finite value")
        if rmsd < 0:
            raise ValueError(f"endpoint {endpoint_id!r} has a negative RMSD")
        normalized.append(
            {
                "id": endpoint_id,
                "rmsd_nm": rmsd,
                "delta_energy_kj_mol": delta_energy,
                "integrity": integrity,
            }
        )
    if not normalized:
        raise ValueError("entries must contain at least one endpoint")

    integrity_failures = [row["id"] for row in normalized if not row["integrity"]]
    rmsd_failures = [
        row["id"]
        for row in normalized
        if abs(row["rmsd_nm"]) > rmsd_threshold_nm
    ]
    energy_failures = [
        row["id"]
        for row in normalized
        if abs(row["delta_energy_kj_mol"]) > energy_threshold_kj_mol
    ]

    if integrity_failures:
        decision = "STRUCTURAL_INTEGRITY_FAILURE"
    elif rmsd_failures or energy_failures:
        decision = "MULTIPLE_DFT_ENDPOINTS"
    else:
        decision = "SINGLE_DFT_ENDPOINT_FAMILY"

    return {
        "decision": decision,
        "endpoint_count": len(normalized),
        "rmsd_threshold_nm": float(rmsd_threshold_nm),
        "energy_threshold_kj_mol": float(energy_threshold_kj_mol),
        "integrity_failures": integrity_failures,
        "rmsd_failures": rmsd_failures,
        "energy_failures": energy_failures,
        "endpoints": normalized,
    }


def audit_reweighting(
    delta_energy_kj_mol: Sequence[float],
    *,
    temperature_k: float,
    min_ess_fraction: float = 0.20,
    max_normalized_weight: float = 0.20,
) -> dict[str, Any]:
    """Audit xTB-to-reference importance-weight overlap.

    ``delta_energy_kj_mol`` is ``E_reference - E_xTB`` for xTB-sampled
    structures.  Normalized weights are evaluated with a stable log-sum-exp
    shift.  A pass requires both the ESS and maximum-weight gates.
    """

    delta = _finite_vector(delta_energy_kj_mol, "delta_energy_kj_mol")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if not 0 < min_ess_fraction <= 1:
        raise ValueError("min_ess_fraction must be in (0, 1]")
    if not 0 < max_normalized_weight <= 1:
        raise ValueError("max_normalized_weight must be in (0, 1]")

    log_weights = -delta / (R_KJ_MOL_K * temperature_k)
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    weights /= float(np.sum(weights))
    ess = float(1.0 / np.sum(np.square(weights)))
    ess_fraction = ess / delta.size
    largest_weight = float(np.max(weights))
    sufficient_overlap = (
        ess_fraction >= min_ess_fraction
        and largest_weight <= max_normalized_weight
    )

    return {
        "decision": "SUFFICIENT_OVERLAP" if sufficient_overlap else "INSUFFICIENT_OVERLAP",
        "sufficient_overlap": sufficient_overlap,
        "sample_count": int(delta.size),
        "temperature_k": float(temperature_k),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess_fraction,
        "minimum_ess_fraction": float(min_ess_fraction),
        "maximum_normalized_weight": largest_weight,
        "maximum_normalized_weight_gate": float(max_normalized_weight),
        "normalized_weights": weights.tolist(),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_column(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    try:
        return [float(row[column]) for row in rows]
    except KeyError as exc:
        raise ValueError(f"CSV column not found: {column}") from exc


def _parse_bool(value: str) -> bool:
    return _coerce_bool(value)


def _write_result(result: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


def _energy_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.csv)
    if not rows:
        raise ValueError("energy CSV contains no data rows")
    groups: dict[str, list[dict[str, str]]] = {}
    if args.group_column:
        for row in rows:
            groups.setdefault(row[args.group_column], []).append(row)
    else:
        groups["all"] = rows

    results: dict[str, Any] = {}
    for group, group_rows in groups.items():
        audit = audit_relative_energies(
            _float_column(group_rows, args.xtb_column),
            _float_column(group_rows, args.reference_column),
            max_abs_error_kj_mol=args.max_error_kj,
            ordering_tolerance_kj_mol=args.ordering_tolerance_kj,
        )
        if args.id_column:
            audit["structure_ids"] = [row[args.id_column] for row in group_rows]
        results[group] = audit
    return {
        "analysis": "relative_energy_qualification",
        "decision": (
            "PASS"
            if all(result["decision"] == "PASS" for result in results.values())
            else "FAIL"
        ),
        "groups": results,
    }


def _gradient_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.csv)
    if not rows:
        raise ValueError("gradient CSV contains no data rows")
    columns = [args.gx_column, args.gy_column, args.gz_column]
    components = [[float(row[column]) for column in columns] for row in rows]
    result = audit_gradient(
        components,
        rms_threshold=args.rms_threshold,
        max_threshold=args.max_threshold,
    )
    result["analysis"] = "gradient_stationarity"
    result["units"] = args.units
    return result


def _endpoint_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.csv)
    entries = [
        {
            "id": row[args.id_column],
            "rmsd_nm": float(row[args.rmsd_column]),
            "delta_energy_kj_mol": float(row[args.energy_column]),
            "integrity": _parse_bool(row[args.integrity_column]),
        }
        for row in rows
    ]
    result = audit_endpoint_family(
        entries,
        rmsd_threshold_nm=args.rmsd_threshold_nm,
        energy_threshold_kj_mol=args.energy_threshold_kj,
    )
    result["analysis"] = "endpoint_family_classification"
    return result


def _overlap_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.csv)
    result = audit_reweighting(
        _float_column(rows, args.delta_energy_column),
        temperature_k=args.temperature_k,
        min_ess_fraction=args.min_ess_fraction,
        max_normalized_weight=args.max_normalized_weight,
    )
    result["analysis"] = "importance_reweighting_overlap"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    energy = subparsers.add_parser(
        "energy",
        help="compare xTB and high-level relative energies",
    )
    energy.add_argument("csv", type=Path)
    energy.add_argument("--group-column")
    energy.add_argument("--id-column", default="structure_id")
    energy.add_argument("--xtb-column", default="xtb_energy_kj_mol")
    energy.add_argument("--reference-column", default="reference_energy_kj_mol")
    energy.add_argument("--max-error-kj", type=float, default=8.4)
    energy.add_argument("--ordering-tolerance-kj", type=float, default=1.0e-8)
    energy.add_argument("--output", type=Path)
    energy.set_defaults(handler=_energy_command)

    gradient = subparsers.add_parser(
        "gradient",
        help="check whether a reference geometry is stationary",
    )
    gradient.add_argument("csv", type=Path)
    gradient.add_argument("--gx-column", default="gx")
    gradient.add_argument("--gy-column", default="gy")
    gradient.add_argument("--gz-column", default="gz")
    gradient.add_argument("--rms-threshold", type=float, default=3.0e-5)
    gradient.add_argument("--max-threshold", type=float, default=1.0e-4)
    gradient.add_argument("--units", default="Eh/bohr")
    gradient.add_argument("--output", type=Path)
    gradient.set_defaults(handler=_gradient_command)

    endpoint = subparsers.add_parser(
        "endpoint",
        help="classify independently optimized reference endpoints",
    )
    endpoint.add_argument("csv", type=Path)
    endpoint.add_argument("--id-column", default="endpoint_id")
    endpoint.add_argument("--rmsd-column", default="rmsd_nm")
    endpoint.add_argument("--energy-column", default="delta_energy_kj_mol")
    endpoint.add_argument("--integrity-column", default="integrity")
    endpoint.add_argument("--rmsd-threshold-nm", type=float, default=0.05)
    endpoint.add_argument("--energy-threshold-kj", type=float, default=2.0)
    endpoint.add_argument("--output", type=Path)
    endpoint.set_defaults(handler=_endpoint_command)

    overlap = subparsers.add_parser(
        "overlap",
        help="audit xTB-to-reference importance-weight overlap",
    )
    overlap.add_argument("csv", type=Path)
    overlap.add_argument("--delta-energy-column", default="delta_energy_kj_mol")
    overlap.add_argument("--temperature-k", type=float, required=True)
    overlap.add_argument("--min-ess-fraction", type=float, default=0.20)
    overlap.add_argument("--max-normalized-weight", type=float, default=0.20)
    overlap.add_argument("--output", type=Path)
    overlap.set_defaults(handler=_overlap_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        _write_result(result, args.output)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import json

import numpy as np
import pytest

from hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification import (
    audit_endpoint_family,
    audit_gradient,
    audit_relative_energies,
    audit_reweighting,
    main,
)


def test_relative_energy_audit_passes_consistent_landscape() -> None:
    result = audit_relative_energies(
        [0.0, 2.0, 7.0, 4.0],
        [10.0, 12.5, 16.0, 14.5],
    )

    assert result["decision"] == "PASS"
    assert result["same_minimum"] is True
    assert result["ordering_agreement_fraction"] == pytest.approx(1.0)
    assert result["max_abs_error_kj_mol"] == pytest.approx(1.0)


def test_relative_energy_audit_detects_ordering_failure() -> None:
    result = audit_relative_energies(
        [0.0, 2.0, 7.0],
        [0.0, 8.0, 3.0],
    )

    assert result["decision"] == "FAIL"
    assert result["same_minimum"] is True
    assert result["disagreeing_pairs"] == [[1, 2]]


def test_gradient_audit_separates_stationary_and_nonstationary() -> None:
    stationary = audit_gradient(np.full((4, 3), 1.0e-5))
    nonstationary = audit_gradient([[0.0, 0.0, 2.0e-4]])

    assert stationary["decision"] == "STATIONARY"
    assert stationary["stationary"] is True
    assert nonstationary["decision"] == "NON_STATIONARY"
    assert nonstationary["max_abs_gradient"] == pytest.approx(2.0e-4)


def test_endpoint_family_audit_has_three_explicit_states() -> None:
    single = audit_endpoint_family(
        [
            {"id": "a", "rmsd_nm": 0.0, "delta_energy_kj_mol": 0.0, "integrity": True},
            {"id": "b", "rmsd_nm": 0.03, "delta_energy_kj_mol": 1.0, "integrity": True},
        ]
    )
    multiple = audit_endpoint_family(
        [
            {"id": "a", "rmsd_nm": 0.0, "delta_energy_kj_mol": 0.0, "integrity": True},
            {"id": "b", "rmsd_nm": 0.08, "delta_energy_kj_mol": 1.0, "integrity": True},
        ]
    )
    broken = audit_endpoint_family(
        [
            {"id": "a", "rmsd_nm": 0.0, "delta_energy_kj_mol": 0.0, "integrity": False},
        ]
    )

    assert single["decision"] == "SINGLE_DFT_ENDPOINT_FAMILY"
    assert multiple["decision"] == "MULTIPLE_DFT_ENDPOINTS"
    assert multiple["rmsd_failures"] == ["b"]
    assert broken["decision"] == "STRUCTURAL_INTEGRITY_FAILURE"


def test_endpoint_family_does_not_treat_false_string_as_true() -> None:
    result = audit_endpoint_family(
        [
            {
                "id": "bad",
                "rmsd_nm": 0.0,
                "delta_energy_kj_mol": 0.0,
                "integrity": "false",
            }
        ]
    )

    assert result["decision"] == "STRUCTURAL_INTEGRITY_FAILURE"


def test_reweighting_audit_detects_overlap_collapse() -> None:
    uniform = audit_reweighting([0.0] * 10, temperature_k=310.0)
    collapsed = audit_reweighting(
        [-100.0] + [100.0] * 9,
        temperature_k=310.0,
    )

    assert uniform["decision"] == "SUFFICIENT_OVERLAP"
    assert uniform["effective_sample_size"] == pytest.approx(10.0)
    assert collapsed["decision"] == "INSUFFICIENT_OVERLAP"
    assert collapsed["maximum_normalized_weight"] == pytest.approx(1.0)


def test_energy_cli_writes_grouped_json(tmp_path) -> None:
    input_csv = tmp_path / "energies.csv"
    output_json = tmp_path / "audit.json"
    input_csv.write_text(
        "chemistry,structure_id,xtb_energy_kj_mol,reference_energy_kj_mol\n"
        "C,c0,0,0\n"
        "C,c1,2,2.5\n"
        "D,d0,4,10\n"
        "D,d1,5,11\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "energy",
            str(input_csv),
            "--group-column",
            "chemistry",
            "--output",
            str(output_json),
        ]
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["decision"] == "PASS"
    assert set(payload["groups"]) == {"C", "D"}

"""Gate A: does a candidate charge set reproduce the N-H donor ordering?

The project's measurement is a site preference -- chloride binding to the
thiourethane N-H versus the urethane N-H. A charge set that inverts or flattens
that ordering cannot answer it, no matter how well the MD is equilibrated, so
the ordering is gated here rather than discovered later in a production run.

Why these criteria and not "match the DFT interaction energy"
------------------------------------------------------------
The absolute depth of the chloride interaction (14-17 kcal/mol by QM) carries
polarization and charge transfer that fixed point charges on nuclei do not
represent. Chasing it would mean over-polarising the charges to compensate.
What a fixed-charge model *can* carry is the *difference* between the two
donors, and it can carry it because the omitted physics is near common-mode:
point charges are systematically too shallow on the N-H axis for both donors,
so the error cancels in the difference -- but only if the two errors are
actually similar. That is why the error-match criterion is gated alongside the
difference itself. A charge set that gets the right difference from two large
and unequal errors has got it by luck.

The numbers come from the ESP-cone analysis (cowork/INDEPENDENT_PARAMETERIZATION_ADVICE.md)
and the 2026-09-16 conformer-weighting investigation (cowork/REVIEW_FORCEFIELD_STRATEGY_20260916.md).

These tests read a summary JSON produced by
``example/08_des_thiourethane_aa/parameterization/resp_boltzmann_refit.py``.
That file is committed, so the gate runs without reaching into the
collaborators' DFT directory. Regenerate it with the script when the fit
changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SUMMARY = (Path(__file__).resolve().parents[1] / "example" / "08_des_thiourethane_aa"
           / "parameterization" / "raw" / "resp_boltzmann" / "refit_summary.json")

# --- Gate A thresholds -----------------------------------------------------
#: The QM axial ESP difference F3 - F2 is -1.77 kcal/mol. A model is allowed to
#: land between these bounds. The lower bound rejects a flat or inverted
#: ordering; the upper bound rejects overshoot, which would mean the ordering
#: was bought by over-polarising rather than by fitting.
AXIAL_DELTA_MIN = -2.4
AXIAL_DELTA_MAX = -1.2

#: The two donors' on-axis errors must be close enough that they cancel in the
#: difference. 0.5 kcal/mol is well inside the ~1.8 kcal/mol difference being
#: resolved.
MAX_ERROR_MISMATCH = 0.5

#: Fitting fewer, better-weighted conformers is allowed to cost a little
#: whole-grid accuracy on the held-out conformer, but not much.
MAX_HELDOUT_DEGRADATION = 0.02

#: Chloride at each model's own axial minimum. DFT gives -2.83 kcal/mol for the
#: difference; a fixed-charge model is not expected to reach that, but it must
#: resolve the two donors by more than thermal noise at 300 K (~0.6 kcal/mol).
MIN_CL_SEPARATION = 1.0


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY.exists():
        pytest.skip(f"no refit summary at {SUMMARY}; run resp_boltzmann_refit.py")
    return json.loads(SUMMARY.read_text())


@pytest.fixture(scope="module")
def ordering(summary):
    if "ordering" not in summary:
        pytest.skip("summary has no ordering block (F2/F3 not both refitted)")
    return summary["ordering"]


# --- the gate itself -------------------------------------------------------
def test_refit_reproduces_the_dft_ordering(ordering):
    """F3 (thiourethane) must be the stronger donor, by a believable margin."""
    delta = ordering["axial_delta_refit"]
    assert AXIAL_DELTA_MIN <= delta <= AXIAL_DELTA_MAX, (
        f"axial ESP difference F3-F2 = {delta:+.2f} kcal/mol is outside "
        f"[{AXIAL_DELTA_MIN}, {AXIAL_DELTA_MAX}]; QM reference is "
        f"{ordering['axial_delta_qm']:+.2f}")


def test_refit_errors_are_common_mode(ordering):
    """The ordering must survive because the errors cancel, not by luck."""
    mismatch = ordering["axial_error_match_refit"]
    assert mismatch <= MAX_ERROR_MISMATCH, (
        f"the two donors' on-axis errors differ by {mismatch:.2f} kcal/mol; "
        f"a difference this large is comparable to the effect being measured")


def test_refit_separates_chloride_binding(ordering):
    """At each model's own minimum the two donors must be distinguishable."""
    if "cl_own_min_delta_refit" not in ordering:
        pytest.skip("no chloride complex scores in the summary")
    delta = ordering["cl_own_min_delta_refit"]
    assert delta <= -MIN_CL_SEPARATION, (
        f"chloride prefers thiourethane by only {abs(delta):.2f} kcal/mol, "
        f"which is not enough to resolve above thermal noise at 300 K")


def test_heldout_conformer_does_not_degrade(summary):
    """Reweighting must not be overfitting: check the conformer held out of every fit."""
    offenders = {
        name: rec["heldout_rrms_change"]
        for name, rec in summary["fragments"].items()
        if rec.get("heldout_rrms_change", 0.0) > MAX_HELDOUT_DEGRADATION
    }
    assert not offenders, (
        f"held-out RRMS degrades past {MAX_HELDOUT_DEGRADATION} for: {offenders}")


# --- the gate must actually discriminate -----------------------------------
# A gate that everything passes is not a gate. The shipped charges are the
# known-bad case that motivated all of this, so they are asserted to fail.
def test_shipped_charges_fail_the_ordering_gate(ordering):
    delta = ordering["axial_delta_shipped"]
    assert not (AXIAL_DELTA_MIN <= delta <= AXIAL_DELTA_MAX), (
        f"the shipped charges gave {delta:+.2f} kcal/mol and now pass the gate; "
        f"either the gate was loosened or the inputs changed -- investigate "
        f"before trusting a pass")


def test_shipped_charges_fail_the_error_match_gate(ordering):
    mismatch = ordering["axial_error_match_shipped"]
    assert mismatch > MAX_ERROR_MISMATCH, (
        f"the shipped charges' error mismatch is {mismatch:.2f}, inside the "
        f"tolerance; the gate no longer discriminates")


# --- provenance ------------------------------------------------------------
def test_unweighted_control_reproduces_the_shipped_fit(summary):
    """Our RESP solver must reproduce the collaborators' fit when unweighted.

    Without this the comparison is between two different solvers, not between
    two weightings, and nothing above means anything.
    """
    failures = {
        name: rec["max_abs_dev_control_vs_shipped"]
        for name, rec in summary["fragments"].items()
        if not rec.get("reproduces_shipped", False)
    }
    assert not failures, (
        f"unweighted refit does not reproduce the shipped charges for: {failures} "
        f"(max abs deviation in e)")


def test_f3_is_the_fragment_with_the_weighting_problem(summary):
    """Document the diagnosis in an executable form.

    The effective conformer count is the inverse participation ratio of the
    Boltzmann populations: how many conformers actually carry the fit. F3 sits
    at 1.0 -- one conformer matters and the fit gave half its weight to the
    other -- while every other fragment is well above 2.
    """
    eff = {name: rec["effective_conformers"] for name, rec in summary["fragments"].items()}
    assert eff["F3_thiouret_p"] < 1.1, (
        f"F3 effective conformer count is {eff['F3_thiouret_p']:.2f}; the "
        f"diagnosis that its fit rests on a single populated conformer no "
        f"longer holds")
    others = {k: v for k, v in eff.items() if k != "F3_thiouret_p"}
    assert min(others.values()) > 2.0, (
        f"another fragment now has a degenerate conformer set too: {others}")

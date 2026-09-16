"""The structural screen must catch an arrested cell and a remembered lattice.

The thermodynamic screen cannot see either. The PEG pilot cell passed every
construction check, ran a clean 30 ns NPT, and was a glass the whole time:
chloride moved 0.15 nm, its mean square displacement grew as t^0.38, and the
builder's seeded lattice still held 84-96% of its ordering at the end. Every
structural average from that run would have described the starting
configuration. These tests pin the arithmetic that says so.

The GROMACS calls are not exercised here -- they need a trajectory. What is
exercised is every decision the screen makes once it has numbers.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

MODULE = (Path(__file__).resolve().parents[1] / "example" / "08_des_thiourethane_aa"
          / "project" / "config_npt" / "check_structure.py")


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("check_structure", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_msd(alpha: float, prefactor: float = 1e-4, n: int = 40):
    """MSD = prefactor * t^alpha, sampled on a log-spaced lag grid."""
    return [(t, prefactor * t ** alpha)
            for t in (100.0 * 1.15 ** i for i in range(n))]


# --- mobility --------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.35, 0.5, 0.8, 1.0, 1.4])
def test_the_slope_fit_recovers_the_exponent(cs, alpha):
    series = synthetic_msd(alpha)
    got, npts = cs.loglog_slope(series, 1000.0, 8000.0)
    assert npts >= 3
    assert got == pytest.approx(alpha, abs=1e-6)


def test_a_fit_window_with_too_few_points_cannot_be_judged(cs):
    series = synthetic_msd(1.0, n=40)
    with pytest.raises(cs.CannotJudge):
        cs.loglog_slope(series, 1e9, 2e9)


def test_zero_and_negative_msd_points_are_dropped_not_logged(cs):
    """A zero at lag 0 must not take log(0); it must simply not be fitted."""
    series = [(0.0, 0.0)] + synthetic_msd(1.0)
    got, _ = cs.loglog_slope(series, 1000.0, 8000.0)
    assert got == pytest.approx(1.0, abs=1e-6)


# --- lattice memory --------------------------------------------------------
def test_a_perfect_lattice_gives_the_scatterer_count(cs):
    """Points exactly on the seeded spacing scatter in phase: S(k) = N."""
    repeats, per_axis, box = 4, 4, (8.0, 8.0, 8.0)
    spacing = box[0] / per_axis
    coords = [(i * spacing, j * spacing, k * spacing)
              for i in range(per_axis) for j in range(per_axis) for k in range(per_axis)]
    s = cs.structure_factor(coords, box, repeats)
    for value in s:
        assert value == pytest.approx(len(coords), rel=1e-9)


def test_a_random_arrangement_sits_near_one(cs):
    """The floor the screen measures decay against."""
    import random
    rng = random.Random(20260916)
    box = (8.0, 8.0, 8.0)
    coords = [(rng.uniform(0, 8), rng.uniform(0, 8), rng.uniform(0, 8))
              for _ in range(4000)]
    s = cs.structure_factor(coords, box, 4)
    for value in s:
        assert value < 10.0, f"random arrangement gave S(k)={value}, expected order 1"


def test_structure_factor_uses_the_box_not_a_fixed_spacing(cs):
    """The wavevector must follow the box, or an NPT cell drifts off the peak."""
    per_axis = 4
    for length in (6.0, 8.0, 11.5):
        spacing = length / per_axis
        coords = [(i * spacing, j * spacing, k * spacing)
                  for i in range(per_axis) for j in range(per_axis) for k in range(per_axis)]
        s = cs.structure_factor(coords, (length, length, length), per_axis)
        assert s[0] == pytest.approx(len(coords), rel=1e-9)


# --- the screen must discriminate -----------------------------------------
def test_the_pilot_cells_numbers_would_fail_both_checks(cs):
    """Regression against the run that motivated this screen.

    Chloride: log-log slope 0.377, rms displacement 0.151 nm at a 15.5 ns lag.
    Lattice: 0.885 / 0.956 / 0.841 of the seeded ordering still present.
    """
    alpha, rms = 0.377, 0.151
    assert alpha < cs.DEFAULT_ALPHA_MIN
    assert rms < cs.DEFAULT_MIN_DISPLACEMENT_NM
    retained = [0.885, 0.956, 0.841]
    assert max(retained) > cs.DEFAULT_MAX_ORDER_RETAINED


def test_a_healthy_liquid_would_pass_both_checks(cs):
    """A gate nothing passes is as useless as one nothing fails."""
    series = synthetic_msd(1.0, prefactor=2e-4)
    alpha, _ = cs.loglog_slope(series, 1000.0, 8000.0)
    rms = math.sqrt(series[-1][1])
    assert alpha >= cs.DEFAULT_ALPHA_MIN
    assert rms >= cs.DEFAULT_MIN_DISPLACEMENT_NM
    import random
    rng = random.Random(7)
    box = (8.0, 8.0, 8.0)
    coords = [(rng.uniform(0, 8), rng.uniform(0, 8), rng.uniform(0, 8))
              for _ in range(4000)]
    ordered = [(i * 2.0, j * 2.0, k * 2.0)
               for i in range(4) for j in range(4) for k in range(4)]
    first = cs.structure_factor(ordered, box, 4)
    last = cs.structure_factor(coords, box, 4)
    retained = [max(0.0, b - 1.0) / max(1e-12, a - 1.0) for a, b in zip(first, last)]
    assert max(retained) <= cs.DEFAULT_MAX_ORDER_RETAINED


# --- fail closed -----------------------------------------------------------
def test_no_selection_at_all_is_refused(cs):
    assert cs.main(["some.tpr", "some.xtc"]) == 2


def test_lattice_check_without_the_repeat_count_is_refused(cs):
    """The repeats define the wavevector; guessing one would invent a result."""
    assert cs.main(["some.tpr", "some.xtc", "--order-sel", "resname HEX"]) == 2


def test_missing_input_files_are_refused(cs, tmp_path):
    assert cs.main([str(tmp_path / "absent.tpr"), str(tmp_path / "absent.xtc"),
                    "--mobility-sel", "resname CL"]) == 2

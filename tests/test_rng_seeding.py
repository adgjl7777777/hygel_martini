"""A run seed must reach the compiled geometry RNG, not only Python and NumPy.

``random_normal_vector`` is Numba-compiled and Numba keeps its own per-thread
RNG state. Seeding only ``random`` and ``np.random`` therefore left every
side-chain direction nondeterministic between CLI processes that shared a
``random_seed``. Ported from the Series-01 reliability copy (0.1.1.dev2).
"""
import random

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.common.utility import (
    random_normal_vector,
    seed_numba_random,
)
from hygel_martini.hydrogel_builder.config_params import build_hydrogel

A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.0])
C = np.array([2.0, 0.3, 0.0])
L = 10.0  # rij() takes a scalar box length: an array fails Numba typing


def _vectors(seed, n=5):
    seed_numba_random(seed)
    return np.array([random_normal_vector(A, B, C, 1.0, L) for _ in range(n)])


def test_same_seed_repeats_the_compiled_geometry_stream():
    first = _vectors(2020)
    second = _vectors(2020)
    assert np.array_equal(first, second)


def test_different_seeds_diverge():
    assert not np.array_equal(_vectors(2020), _vectors(2021))


def test_run_seed_reaches_all_three_generators_without_shifting_python_or_numpy():
    """The first Python/NumPy draws after seeding must be what they were before
    the Numba call was added, or every seeded backbone would silently change."""
    random.seed(7); np.random.seed(7)
    expected_py = [random.random() for _ in range(5)]
    expected_np = np.random.random(5)

    build_hydrogel._seed_random_generators(7)
    got_py = [random.random() for _ in range(5)]
    got_np = np.random.random(5)
    assert got_py == expected_py
    assert np.array_equal(got_np, expected_np)

    # and the compiled stream is now reproducible under that same call
    build_hydrogel._seed_random_generators(7)
    v1 = random_normal_vector(A, B, C, 1.0, L)
    build_hydrogel._seed_random_generators(7)
    v2 = random_normal_vector(A, B, C, 1.0, L)
    assert np.array_equal(v1, v2)


@pytest.mark.parametrize("bad", [None, "not-a-number", object()])
def test_unusable_seed_is_a_no_op(bad):
    build_hydrogel._seed_random_generators(bad)  # must not raise

"""Check the exact subset expectations of the E24 analysis against brute-force enumeration."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from seedext_v1_eval import expected_top, expected_top_given_first, k_star, reading, subset_weights  # noqa: E402


def brute_top(keys, values, subset):
    best = max(subset, key=lambda i: (keys[i], -i))  # largest key, ties to the lower index
    return values[best]


@pytest.mark.parametrize("n,k", [(5, 1), (5, 3), (6, 6), (7, 4)])
def test_subset_weights_sum_to_one(n, k):
    assert subset_weights(n, k).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_expected_top_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = 7
    keys = rng.integers(0, 4, size=n).astype(float)  # ties included
    values = rng.random(n)
    for k in range(1, n + 1):
        subsets = list(itertools.combinations(range(n), k))
        brute = np.mean([brute_top(keys, values, s) for s in subsets])
        assert expected_top(keys, values, k) == pytest.approx(brute)


@pytest.mark.parametrize("seed", range(5))
def test_expected_top_given_first_matches_enumeration(seed):
    rng = np.random.default_rng(100 + seed)
    n = 6
    keys = rng.integers(0, 3, size=n).astype(float)
    values = rng.random(n)
    for first in range(n):
        others = [i for i in range(n) if i != first]
        for k in range(1, n + 1):
            subsets = [(first, *c) for c in itertools.combinations(others, k - 1)]
            brute = np.mean([brute_top(keys, values, s) for s in subsets])
            assert expected_top_given_first(keys, values, first, k) == pytest.approx(brute)


def test_first_seed_average_equals_unconditional():
    rng = np.random.default_rng(7)
    n = 8
    keys, values = rng.random(n), rng.random(n)
    for k in range(1, n + 1):
        avg = np.mean([expected_top_given_first(keys, values, j, k) for j in range(n)])
        assert avg == pytest.approx(expected_top(keys, values, k))


def test_k_star_and_reading():
    assert k_star(np.array([0.004, 0.003]), 0.005) == (1, True)
    assert k_star(np.array([0.010, 0.006, 0.004]), 0.005) == (3, True)
    assert k_star(np.array([0.010, 0.009]), 0.005) == (3, False)
    assert reading(0.003, -0.005, 0.012) == "A"
    assert reading(0.030, 0.010, 0.050) == "B"
    assert reading(0.015, -0.002, 0.032) == "C"

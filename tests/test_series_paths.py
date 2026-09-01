"""Tests for the best-of-N series scoreline enumeration (roadmap M30).

Covers ``utils/series_paths.py``: the ``_validate_best_of`` /
``series_win_threshold`` validation pair, the core
``enumerate_series_paths`` recursion (hand-computed exact values for
Bo1/Bo3/Bo5/Bo7 and a full error matrix), and the two convenience
reshapers ``series_outcome_order`` and ``series_probabilities_in_order``.
Everything is pure in-memory math — no cache, no Parquet, no network —
and every "sums to 1" assertion uses the stated float tolerance
``_SUM_TOL = 1e-9`` rather than exact equality.
"""

import math

import numpy as np
import pytest

from utils import series_paths

# Tolerance for "the returned scoreline probabilities sum to 1" test
# assertions. The enumeration distributes exactly one unit of
# probability mass across the leaves, so the only error is float
# accumulation; 1e-9 matches utils/scoring.py's _PROB_SUM_TOL and is
# generous enough to absorb it while still catching a genuine bug that
# drops or double-counts mass.
_SUM_TOL = 1e-9


def _assert_distribution_sum(distribution):
    """Assert a scoreline-probability mapping sums to 1 within tolerance.

    Args:
        distribution: A ``dict`` mapping terminal ``(a_wins, b_wins)``
            scorelines to ``float`` probabilities.

    Returns:
        Nothing.

    Raises:
        AssertionError: If the sum of the probabilities differs from 1
            by more than ``_SUM_TOL``.
    """
    total = math.fsum(distribution.values())
    assert abs(total - 1.0) <= _SUM_TOL, (
        f"scoreline probabilities must sum to 1 within {_SUM_TOL:g}, "
        f"got {total!r}"
    )


# --------------------------------------------------------------------------
# _validate_best_of / series_win_threshold
# --------------------------------------------------------------------------


def test_series_win_threshold_bo1_bo3_bo5_bo7():
    # The majority win count is (best_of + 1) // 2: 1/2/3/4 maps needed
    # to win a Bo1/Bo3/Bo5/Bo7.
    assert series_paths.series_win_threshold(1) == 1
    assert series_paths.series_win_threshold(3) == 2
    assert series_paths.series_win_threshold(5) == 3
    assert series_paths.series_win_threshold(7) == 4


def test_series_win_threshold_accepts_numpy_integer():
    # operator.index-style coercion must accept numpy integer scalars
    # (the house style inherited from utils/scoring.py), not only
    # plain Python ints.
    assert series_paths.series_win_threshold(np.int64(5)) == 3


@pytest.mark.parametrize("bad", [0, -1, -7, 4, 6, 2, 3.5, "3", None])
def test_validate_best_of_rejects_invalid_values(bad):
    # Every invalid best_of from plan decision 7 — zero, negative,
    # even, and non-integer-like — must raise ValueError, never be
    # silently coerced or truncated.
    with pytest.raises(ValueError):
        series_paths.series_win_threshold(bad)


def test_validate_best_of_error_messages_name_the_value():
    # The error must name the offending value so a caller can see which
    # argument was rejected without re-parsing the stack.
    with pytest.raises(ValueError, match="4"):
        series_paths.series_win_threshold(4)
    with pytest.raises(ValueError, match="3.5"):
        series_paths.series_win_threshold(3.5)


# --------------------------------------------------------------------------
# enumerate_series_paths: hand-computed exact values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.3, 0.5, 0.8])
def test_enumerate_series_paths_bo1_exact(p):
    # A Bo1 is a single map: the result must be exactly {(1, 0): p,
    # (0, 1): 1 - p}. Includes a non-symmetric p (0.3) to prove the
    # two leaves are not hard-coded symmetrically.
    assert series_paths.enumerate_series_paths([p], 1) == {
        (1, 0): p,
        (0, 1): 1.0 - p,
    }


def test_enumerate_series_paths_bo3_constant_p_closed_form():
    # Bo3 with every map at p=0.6: the closed forms are P(2,0)=p^2,
    # P(2,1)=2p^2(1-p), P(1,2)=2p(1-p)^2, P(0,2)=(1-p)^2. Check all
    # four leaves against the formulas and the sum against tolerance.
    p = 0.6
    result = series_paths.enumerate_series_paths([p, p, p], 3)
    assert result[(2, 0)] == pytest.approx(p * p)
    assert result[(2, 1)] == pytest.approx(2 * p * p * (1 - p))
    assert result[(1, 2)] == pytest.approx(2 * p * (1 - p) * (1 - p))
    assert result[(0, 2)] == pytest.approx((1 - p) * (1 - p))
    _assert_distribution_sum(result)


def test_enumerate_series_paths_bo3_varying_p_hand_enumerated():
    # Bo3 with map_win_probs=[0.7, 0.4, 0.9], all reachable paths
    # hand-enumerated here (not via the function under test) to catch
    # an off-by-one in which map_win_probs index is consulted at which
    # recursion depth:
    #   (2,0): A,A                                   -> 0.7*0.4
    #   (2,1): A,B,A and B,A,A                       -> 0.7*0.6*0.9
    #                                                  + 0.3*0.4*0.9
    #   (1,2): A,B,B and B,A,B                       -> 0.7*0.6*0.1
    #                                                  + 0.3*0.4*0.1
    #   (0,2): B,B                                   -> 0.3*0.6
    probs = [0.7, 0.4, 0.9]
    result = series_paths.enumerate_series_paths(probs, 3)
    assert result[(2, 0)] == pytest.approx(0.7 * 0.4)
    assert result[(2, 1)] == pytest.approx(0.7 * 0.6 * 0.9 + 0.3 * 0.4 * 0.9)
    assert result[(1, 2)] == pytest.approx(0.7 * 0.6 * 0.1 + 0.3 * 0.4 * 0.1)
    assert result[(0, 2)] == pytest.approx(0.3 * 0.6)
    _assert_distribution_sum(result)


def test_enumerate_series_paths_bo5_corner_leaf_and_sum():
    # Bo5 corner leaf: P(3,0) is A winning the first three maps,
    # p0*p1*p2. Also check the full six-leaf distribution sums to 1.
    probs = [0.7, 0.4, 0.9, 0.2, 0.8]
    result = series_paths.enumerate_series_paths(probs, 5)
    assert result[(3, 0)] == pytest.approx(0.7 * 0.4 * 0.9)
    assert len(result) == 6
    _assert_distribution_sum(result)


def test_enumerate_series_paths_bo7_corner_leaf_and_sum():
    # Bo7 is the roadmap's explicit requirement even though no
    # ACTION_SEQUENCES table elsewhere in the repo has a "Bo7" entry:
    # P(4,0) is A winning the first four maps, and the full eight-leaf
    # distribution must sum to 1 — proving the single implementation
    # really covers a length absent everywhere else in the codebase.
    probs = [0.3, 0.6, 0.2, 0.8, 0.4, 0.7, 0.9]
    result = series_paths.enumerate_series_paths(probs, 7)
    assert result[(4, 0)] == pytest.approx(0.3 * 0.6 * 0.2 * 0.8)
    assert len(result) == 8
    _assert_distribution_sum(result)


def test_enumerate_series_paths_bo5_symmetric_uniform():
    # With every map at p=0.5 the six Bo5 leaves are 0.125, 0.1875,
    # 0.1875, 0.1875, 0.1875, 0.125 in canonical order (binomial
    # weights: C(3,0)/8, C(3,1)/8, C(3,2)/8, C(2,2)/8-style — the
    # two 3-0 leaves carry 1/8 each, the four 3-1/2-3 leaves 3/16).
    result = series_paths.series_probabilities_in_order([0.5] * 5, 5)
    expected = [0.125, 0.1875, 0.1875, 0.1875, 0.1875, 0.125]
    assert result == pytest.approx(expected)


# --------------------------------------------------------------------------
# enumerate_series_paths: error matrix
# --------------------------------------------------------------------------


def test_enumerate_series_paths_rejects_wrong_length():
    # map_win_probs must have exactly best_of entries: a shorter
    # vector leaves a reachable map index without a probability, a
    # longer one carries mass for maps that can never be played.
    with pytest.raises(ValueError, match="exactly 3"):
        series_paths.enumerate_series_paths([0.5, 0.5], 3)
    with pytest.raises(ValueError, match="exactly 3"):
        series_paths.enumerate_series_paths([0.5, 0.5, 0.5, 0.5], 3)


@pytest.mark.parametrize("bad_entry", [-0.1, 1.1, math.nan, math.inf])
def test_enumerate_series_paths_rejects_out_of_range_or_non_finite(bad_entry):
    # A negative, >1, or non-finite per-map probability is invalid
    # regardless of where it sits in the vector.
    with pytest.raises(ValueError):
        series_paths.enumerate_series_paths([bad_entry, 0.5, 0.5], 3)


def test_enumerate_series_paths_rejects_non_numeric_entry():
    # A non-numeric entry must raise ValueError uniformly with the
    # numeric failures rather than blowing up with an unrelated
    # TypeError.
    with pytest.raises(ValueError):
        series_paths.enumerate_series_paths(["x", 0.5, 0.5], 3)


@pytest.mark.parametrize("bad", [0, -3, 4, 3.5, "3"])
def test_enumerate_series_paths_rejects_invalid_best_of(bad):
    # The invalid best_of cases from plan decision 7 must propagate as
    # ValueError through the public entry point, not only through
    # series_win_threshold.
    with pytest.raises(ValueError):
        series_paths.enumerate_series_paths([0.5] * 3, bad)


def test_enumerate_series_paths_accepts_endpoint_probabilities():
    # p=0.0 and p=1.0 are valid closed-interval endpoints: with a
    # guaranteed sweep, all probability mass must land on the single
    # deterministic scoreline (the other leaves are present with
    # exactly 0.0, since every terminal scoreline is returned with its
    # exact probability).
    sweep_a = series_paths.enumerate_series_paths([1.0, 1.0, 1.0], 3)
    assert sweep_a[(2, 0)] == 1.0
    assert all(
        prob == 0.0 for scoreline, prob in sweep_a.items() if scoreline != (2, 0)
    )
    sweep_b = series_paths.enumerate_series_paths([0.0, 0.0, 0.0], 3)
    assert sweep_b[(0, 2)] == 1.0
    assert all(
        prob == 0.0 for scoreline, prob in sweep_b.items() if scoreline != (0, 2)
    )


# --------------------------------------------------------------------------
# series_outcome_order
# --------------------------------------------------------------------------


def test_series_outcome_order_bo3_exact():
    # Canonical ordinal order for Bo3: A's wins most-dominant-first,
    # then B's wins most-dominant-first.
    assert series_paths.series_outcome_order(3) == (
        (2, 0),
        (2, 1),
        (1, 2),
        (0, 2),
    )


def test_series_outcome_order_bo5_exact():
    # Canonical ordinal order for Bo5: six scorelines, A's three wins
    # first then B's three wins.
    assert series_paths.series_outcome_order(5) == (
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 3),
        (1, 3),
        (0, 3),
    )


@pytest.mark.parametrize("best_of", [1, 3, 5, 7])
def test_series_outcome_order_length_matches_scoreline_count(best_of):
    # A best-of-N series has exactly N+1 terminal scorelines.
    assert len(series_paths.series_outcome_order(best_of)) == best_of + 1


@pytest.mark.parametrize("best_of", [1, 3, 5, 7])
def test_series_outcome_order_vocabulary_matches_enumeration(best_of):
    # Cross-check that series_outcome_order and enumerate_series_paths
    # agree on the outcome vocabulary: the set of scorelines in the
    # canonical order must equal the key set of the enumeration dict
    # for the same best_of (run with an arbitrary fixed probability
    # vector since the vocabulary is probability-independent).
    probs = [0.5] * best_of
    order = series_paths.series_outcome_order(best_of)
    keys = set(series_paths.enumerate_series_paths(probs, best_of))
    assert set(order) == keys
    assert len(order) == len(keys)


# --------------------------------------------------------------------------
# series_probabilities_in_order
# --------------------------------------------------------------------------


def test_series_probabilities_in_order_matches_manual_reindex():
    # For the Bo3 varying-p case, the ordered vector must equal
    # enumerate_series_paths reindexed by hand via
    # series_outcome_order — catching a mismatch between the two
    # functions' scoreline vocabularies.
    probs = [0.7, 0.4, 0.9]
    paths = series_paths.enumerate_series_paths(probs, 3)
    expected = [paths[scoreline] for scoreline in series_paths.series_outcome_order(3)]
    assert series_paths.series_probabilities_in_order(probs, 3) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("best_of", [1, 3, 5, 7])
def test_series_probabilities_in_order_sums_to_one(best_of):
    # The convenience vector must sum to 1 within tolerance and have
    # exactly best_of + 1 entries for every supported series length.
    probs = [0.3 + 0.05 * i for i in range(best_of)]
    vector = series_paths.series_probabilities_in_order(probs, best_of)
    assert len(vector) == best_of + 1
    assert abs(math.fsum(vector) - 1.0) <= _SUM_TOL


# --------------------------------------------------------------------------
# Determinism / purity
# --------------------------------------------------------------------------


def test_enumerate_series_paths_is_deterministic():
    # Two calls with identical inputs must return dict-equal results —
    # the memo dict is per-call and must not leak state between calls.
    probs = [0.7, 0.4, 0.9]
    first = series_paths.enumerate_series_paths(probs, 3)
    second = series_paths.enumerate_series_paths(probs, 3)
    assert first == second
    assert first is not second


def test_series_probabilities_in_order_is_deterministic():
    # Same determinism contract for the convenience vector.
    probs = [0.7, 0.4, 0.9]
    first = series_paths.series_probabilities_in_order(probs, 3)
    second = series_paths.series_probabilities_in_order(probs, 3)
    assert first == second
    assert first is not second

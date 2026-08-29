"""Tests for the proper scoring rules library (roadmap M11).

Covers the four metrics in ``utils/scoring.py`` — RPS, multi-class log
loss, multi-class Brier score, and marginal binary accuracy — plus
their batch-mean wrappers and the shared input validation. Everything
is pure in-memory math: no cache, no Parquet, no network. The four-way
outcome vocabulary (``drivers.labels.OUTCOME_LABELS``: 0 = A-regulation,
1 = A-OT, 2 = B-OT, 3 = B-regulation) is used as the concrete ``K = 4``
example throughout.
"""

import math

import pytest

from utils import scoring


def _concentrated_probs(k, mass_index, confidence):
    """Build a K-way probability vector concentrated at one category.

    Places ``confidence`` on ``mass_index`` and spreads the remaining
    ``1 - confidence`` uniformly over the other ``k - 1`` categories, so
    the result always sums to exactly 1 and is a valid distribution.

    Args:
        k: The number of categories.
        mass_index: The category index receiving ``confidence``.
        confidence: The probability mass on ``mass_index``; the leftover
            ``1 - confidence`` is divided equally among the rest.

    Returns:
        A ``list`` of ``k`` floats summing to 1.

    Raises:
        Nothing.
    """
    rest = (1.0 - confidence) / (k - 1)
    return [confidence if i == mass_index else rest for i in range(k)]


# --------------------------------------------------------------------------
# RPS known values
# --------------------------------------------------------------------------


def test_rps_known_value_k4_spread():
    # Hand-computed for true index 0 and probs [0.1, 0.6, 0.2, 0.1]:
    # CDF_pred at cuts 1,2,3 is 0.1, 0.7, 0.9 vs true CDF all 1, so
    # RPS = 0.9^2 + 0.3^2 + 0.1^2 = 0.81 + 0.09 + 0.01 = 0.91.
    assert scoring.rps([0.1, 0.6, 0.2, 0.1], 0) == pytest.approx(0.91)


def test_rps_perfect_and_worst():
    # Perfect: all mass on the true category -> 0.
    assert scoring.rps([1.0, 0.0, 0.0, 0.0], 0) == pytest.approx(0.0)
    # Worst: all mass on the furthest category from the truth -> K-1.
    assert scoring.rps([1.0, 0.0, 0.0, 0.0], 3) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# RPS ordering property (the roadmap's explicit requirement)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [1.0, 0.8, 0.6])
@pytest.mark.parametrize(
    ("true_index", "adjacent", "distant"),
    [
        (0, 1, 3),
        (1, 0, 3),
        (2, 1, 0),
        (3, 2, 0),
    ],
)
def test_rps_adjacent_miss_scores_better_than_distant(
    confidence, true_index, adjacent, distant
):
    # For a fixed true category, a prediction concentrated one category
    # away must score strictly better than an equally concentrated
    # prediction two or three categories away — across several
    # confidence levels and true positions, not just one example.
    k = 4
    adjacent_probs = _concentrated_probs(k, adjacent, confidence)
    distant_probs = _concentrated_probs(k, distant, confidence)
    assert scoring.rps(adjacent_probs, true_index) < scoring.rps(
        distant_probs, true_index
    )


# --------------------------------------------------------------------------
# RPS vs Brier contrast
# --------------------------------------------------------------------------


def test_rps_distinguishes_ordinal_distance_where_brier_does_not():
    # True category is 0. The adjacent-miss prediction puts its big mass
    # on category 1; the distant-miss prediction is a permutation with
    # the identical multiset of probabilities but the big mass on
    # category 3. Brier (unordered) scores them identically; RPS
    # (ordinal-aware) prefers the adjacent miss.
    adjacent = [0.1, 0.8, 0.05, 0.05]
    distant = [0.1, 0.05, 0.05, 0.8]
    assert scoring.brier_score(adjacent, 0) == pytest.approx(
        scoring.brier_score(distant, 0)
    )
    assert scoring.rps(adjacent, 0) == pytest.approx(0.8225)
    assert scoring.rps(distant, 0) == pytest.approx(2.1725)
    assert scoring.rps(adjacent, 0) < scoring.rps(distant, 0)


# --------------------------------------------------------------------------
# Log loss known values
# --------------------------------------------------------------------------


def test_log_loss_known_values():
    # Confident and correct: -ln(0.9).
    assert scoring.log_loss([0.9, 0.1, 0.0, 0.0], 0) == pytest.approx(
        -math.log(0.9)
    )
    # Confident and wrong: -ln(0.1) — far larger than the correct case.
    assert scoring.log_loss([0.1, 0.9, 0.0, 0.0], 0) == pytest.approx(
        -math.log(0.1)
    )
    # Uniform over K=4: -ln(0.25) == ln(4) == log(K).
    assert scoring.log_loss([0.25, 0.25, 0.25, 0.25], 0) == pytest.approx(
        math.log(4)
    )
    # The ordering of the three cases: correct < uniform < wrong.
    assert -math.log(0.9) < math.log(4) < -math.log(0.1)


def test_log_loss_zero_probability_raises():
    # A zero probability on the true category is a hard error (the true
    # loss is +inf), not a silently clipped finite value.
    with pytest.raises(ValueError, match="zero probability"):
        scoring.log_loss([0.0, 1.0, 0.0, 0.0], 0)


# --------------------------------------------------------------------------
# Brier score known values
# --------------------------------------------------------------------------


def test_brier_score_known_values():
    # Perfect prediction -> 0.
    assert scoring.brier_score([1.0, 0.0, 0.0, 0.0], 0) == pytest.approx(0.0)
    # Uniform over K=4 -> (K-1)/K = 0.75.
    assert scoring.brier_score([0.25, 0.25, 0.25, 0.25], 0) == pytest.approx(
        0.75
    )
    # Split across categories 0 and 1 with true 0 -> 0.25 + 0.25 = 0.5.
    assert scoring.brier_score([0.5, 0.5, 0.0, 0.0], 0) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Marginal binary accuracy
# --------------------------------------------------------------------------


def test_marginal_binary_accuracy_default_grouping_k4():
    # Default grouping: first half (indices 0, 1) is side A, second half
    # (2, 3) is side B — exactly the four-way vocabulary's A/B split.
    # A-heavy prediction with true A -> correct.
    assert scoring.marginal_binary_accuracy([0.7, 0.2, 0.05, 0.05], 0) is True
    # B-heavy prediction with true A -> incorrect.
    assert scoring.marginal_binary_accuracy([0.2, 0.1, 0.6, 0.1], 1) is False
    # B-heavy prediction with true B -> correct.
    assert scoring.marginal_binary_accuracy([0.2, 0.1, 0.6, 0.1], 2) is True


def test_marginal_binary_accuracy_explicit_grouping():
    # Non-contiguous grouping: categories 0 and 2 form side A, 1 and 3
    # form side B. p_a = 0.1 + 0.6 = 0.7 >= p_b = 0.3 -> predict A.
    group_a = (0, 2)
    probs = [0.1, 0.2, 0.6, 0.1]
    assert scoring.marginal_binary_accuracy(probs, 2, group_a) is True
    assert scoring.marginal_binary_accuracy(probs, 1, group_a) is False


def test_marginal_binary_accuracy_tie_goes_to_a():
    # Equal collapsed mass (0.5 vs 0.5): the tie is resolved to side A.
    probs = [0.5, 0.0, 0.5, 0.0]
    assert scoring.marginal_binary_accuracy(probs, 0) is True
    assert scoring.marginal_binary_accuracy(probs, 2) is False


def test_mean_marginal_binary_accuracy_fraction():
    # Three of four collapsed predictions are correct.
    rows = [
        [0.7, 0.2, 0.05, 0.05],  # predict A, true A -> correct
        [0.2, 0.1, 0.6, 0.1],  # predict B, true A -> incorrect
        [0.2, 0.1, 0.6, 0.1],  # predict B, true B -> correct
        [0.9, 0.05, 0.03, 0.02],  # predict A, true A -> correct
    ]
    indices = [0, 1, 2, 0]
    assert scoring.mean_marginal_binary_accuracy(rows, indices) == pytest.approx(
        0.75
    )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        scoring.rps,
        scoring.log_loss,
        scoring.brier_score,
        scoring.marginal_binary_accuracy,
    ],
)
def test_validation_bad_probabilities_raise(metric):
    # Does not sum to 1.
    with pytest.raises(ValueError):
        metric([0.5, 0.5, 0.5], 0)
    # Negative entry (sums to 1 but not a valid distribution).
    with pytest.raises(ValueError):
        metric([-0.1, 0.6, 0.5], 0)
    # Only one category.
    with pytest.raises(ValueError):
        metric([1.0], 0)


@pytest.mark.parametrize(
    "metric",
    [
        scoring.rps,
        scoring.log_loss,
        scoring.brier_score,
        scoring.marginal_binary_accuracy,
    ],
)
def test_validation_out_of_range_true_index_raises(metric):
    # K=3 valid probabilities; index 3 and -1 are both out of [0, 3).
    with pytest.raises(ValueError):
        metric([0.3, 0.3, 0.4], 3)
    with pytest.raises(ValueError):
        metric([0.3, 0.3, 0.4], -1)


def test_validation_non_integer_true_index_raises():
    # A float true_index is rejected rather than silently truncated.
    with pytest.raises(ValueError):
        scoring.rps([0.5, 0.5], 0.5)


def test_group_validation_raises():
    # Empty side A.
    with pytest.raises(ValueError):
        scoring.marginal_binary_accuracy([0.5, 0.5], 0, ())
    # Side A = all categories (no side B left).
    with pytest.raises(ValueError):
        scoring.marginal_binary_accuracy([0.5, 0.5], 0, (0, 1))
    # Out-of-range side-A index.
    with pytest.raises(ValueError):
        scoring.marginal_binary_accuracy([0.5, 0.5], 0, (2,))


# --------------------------------------------------------------------------
# Batch-mean wrapper invariants
# --------------------------------------------------------------------------

_ROWS = [
    [0.9, 0.05, 0.03, 0.02],
    [0.1, 0.6, 0.2, 0.1],
    [0.25, 0.25, 0.25, 0.25],
    [0.0, 0.0, 0.7, 0.3],
]
_INDICES = [0, 1, 2, 2]


def test_mean_rps_equals_mean_of_per_observation():
    # The batch wrapper is the arithmetic mean of the per-observation
    # scores, not a separately hand-tuned formula.
    expected = sum(scoring.rps(p, i) for p, i in zip(_ROWS, _INDICES)) / len(
        _ROWS
    )
    assert scoring.mean_rps(_ROWS, _INDICES) == pytest.approx(expected)


def test_mean_log_loss_equals_mean_of_per_observation():
    expected = sum(
        scoring.log_loss(p, i) for p, i in zip(_ROWS, _INDICES)
    ) / len(_ROWS)
    assert scoring.mean_log_loss(_ROWS, _INDICES) == pytest.approx(expected)


def test_mean_brier_score_equals_mean_of_per_observation():
    expected = sum(
        scoring.brier_score(p, i) for p, i in zip(_ROWS, _INDICES)
    ) / len(_ROWS)
    assert scoring.mean_brier_score(_ROWS, _INDICES) == pytest.approx(expected)


def test_batch_validation_raises():
    # Empty batch: a mean over zero observations is undefined.
    with pytest.raises(ValueError):
        scoring.mean_rps([], [])
    # Length mismatch: pairing the wrong labels would be silent.
    with pytest.raises(ValueError):
        scoring.mean_rps([[0.5, 0.5]], [0, 1])

"""Tests for the pure bootstrap-interval helpers (evaluation/bootstrap_intervals.py).

Covers the three model-free helpers of M36's statistics half with
hand-computed bands on small synthetic arrays: ``percentile_interval``
(odd and even replicate counts, since numpy's linear percentile
interpolation differs between them; ``ci_level`` boundary validation;
empty-sample rejection), ``replicate_matrix_intervals`` (per-column
bands generic over K; ragged-row and empty-matrix rejection;
``ci_level`` validation), and ``n_games_backing`` (the ``min``
convention on hand-picked pairs including ``a == 0``, symmetry, numpy-
integer inputs, and negative-count rejection). No real fitted artifacts
are required by any of these tests — the helpers are pure.
"""

import numpy as np
import pytest

from evaluation import bootstrap_intervals

# --------------------------------------------------------------------------
# plan#4: percentile_interval hand-computed bands and validation
# --------------------------------------------------------------------------


def test_percentile_interval_hand_computed_odd_count():
    # Five values, ci_level=0.90: numpy linear interpolation puts the
    # 5th percentile at index (5-1)*0.05 = 0.2 and the 95th at 3.8, so
    # lo = 0.1 + 0.2*(0.2-0.1) = 0.12 and hi = 0.4 + 0.8*(0.5-0.4) =
    # 0.48 — the hand-computed band.
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    lo, hi = bootstrap_intervals.percentile_interval(values, ci_level=0.90)
    assert lo == pytest.approx(0.12)
    assert hi == pytest.approx(0.48)
    assert lo <= hi


def test_percentile_interval_hand_computed_even_count():
    # Four values: the 5th percentile sits at index (4-1)*0.05 = 0.15
    # (lo = 0.1 + 0.15*(0.2-0.1) = 0.115) and the 95th at 2.85
    # (hi = 0.3 + 0.85*(0.4-0.3) = 0.385) — the interpolation differs
    # from the odd-count case, which is why both are pinned here.
    values = [0.1, 0.2, 0.3, 0.4]
    lo, hi = bootstrap_intervals.percentile_interval(values, ci_level=0.90)
    assert lo == pytest.approx(0.115)
    assert hi == pytest.approx(0.385)
    assert lo <= hi


def test_percentile_interval_matches_numpy_percentile():
    # A cross-check against numpy's own percentile for a mid-size
    # synthetic sample (the implementation is documented as a plain
    # numpy percentile call, so the two must agree exactly).
    rng = np.random.default_rng(42)
    values = rng.random(23)
    lo, hi = bootstrap_intervals.percentile_interval(values, ci_level=0.80)
    expected_lo, expected_hi = np.percentile(values, [10.0, 90.0])
    assert lo == pytest.approx(float(expected_lo))
    assert hi == pytest.approx(float(expected_hi))


def test_percentile_interval_single_value():
    # A one-element sample: both percentiles collapse to that value
    # (lo == hi == the value) rather than erroring.
    lo, hi = bootstrap_intervals.percentile_interval([0.42], ci_level=0.90)
    assert lo == hi == pytest.approx(0.42)


def test_percentile_interval_rejects_empty():
    # A percentile band over zero replicates is undefined: hard error,
    # not a silent NaN.
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_intervals.percentile_interval([])


def test_percentile_interval_rejects_non_1d():
    # The helper takes a 1-D sequence of values, not a matrix.
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_intervals.percentile_interval([[0.1, 0.2], [0.3, 0.4]])


def test_percentile_interval_ci_level_boundaries():
    # ci_level must be strictly inside (0, 1): 0.0 and 1.0 (degenerate
    # bands) as well as negative and >1 values all raise; non-numeric
    # inputs raise too.
    values = [0.1, 0.2, 0.3]
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ci_level"):
            bootstrap_intervals.percentile_interval(values, ci_level=bad)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_intervals.percentile_interval(values, ci_level="wide")


# --------------------------------------------------------------------------
# plan#4: replicate_matrix_intervals per-column bands and validation
# --------------------------------------------------------------------------


def test_replicate_matrix_intervals_per_column_generic_over_k():
    # A (3, 4) matrix at ci_level=0.5 (25th/75th): column 0 values
    # [0.1, 0.2, 0.3] band to (0.15, 0.25) and column 1 values
    # [0.2, 0.3, 0.4] band to (0.25, 0.35) — hand-computed from numpy's
    # linear interpolation (index positions 0.5 and 1.5 of the
    # sorted 3-element column).
    matrix = [
        [0.1, 0.2, 0.5, 0.2],
        [0.2, 0.3, 0.4, 0.1],
        [0.3, 0.4, 0.3, 0.0],
    ]
    bands = bootstrap_intervals.replicate_matrix_intervals(
        matrix, ci_level=0.5
    )
    assert len(bands) == 4
    assert bands[0] == pytest.approx((0.15, 0.25))
    assert bands[1] == pytest.approx((0.25, 0.35))
    # Every band is ordered lo <= hi.
    for lo, hi in bands:
        assert lo <= hi


def test_replicate_matrix_intervals_six_category_case():
    # The same helper handles the K-way per-series scoreline case (K=6
    # for Bo5) with no OUTCOME_LABELS coupling: two replicates of a
    # 6-vector produce 6 per-column bands, each lo == hi (only one
    # distinct value per column at n_replicates=2 with identical rows).
    matrix = [[0.1, 0.2, 0.2, 0.1, 0.2, 0.2]] * 2
    bands = bootstrap_intervals.replicate_matrix_intervals(
        matrix, ci_level=0.90
    )
    assert len(bands) == 6
    assert bands == pytest.approx(
        [(0.1, 0.1), (0.2, 0.2), (0.2, 0.2), (0.1, 0.1), (0.2, 0.2), (0.2, 0.2)]
    )


def test_replicate_matrix_intervals_accepts_2d_array():
    # A numpy 2-D array input works identically to a list-of-lists.
    matrix = np.asarray(
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=float
    )
    bands = bootstrap_intervals.replicate_matrix_intervals(
        matrix, ci_level=0.5
    )
    assert bands[0] == pytest.approx((0.15, 0.25))
    assert bands[1] == pytest.approx((0.25, 0.35))


def test_replicate_matrix_intervals_rejects_ragged_rows():
    # Rows of differing lengths cannot form a per-column band: hard
    # error naming the offending row.
    with pytest.raises(ValueError, match="ragged"):
        bootstrap_intervals.replicate_matrix_intervals(
            [[0.1, 0.2, 0.3], [0.1, 0.2]]
        )


def test_replicate_matrix_intervals_rejects_empty():
    # Zero replicate rows: no distribution to band.
    with pytest.raises(ValueError, match="at least one replicate"):
        bootstrap_intervals.replicate_matrix_intervals([])


def test_replicate_matrix_intervals_rejects_zero_categories():
    # A zero-category row has no columns to band.
    with pytest.raises(ValueError, match="at least one category"):
        bootstrap_intervals.replicate_matrix_intervals([[ ], [ ]])


def test_replicate_matrix_intervals_ci_level_boundaries():
    # ci_level validation propagates to the matrix form.
    matrix = [[0.1, 0.2], [0.2, 0.3]]
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="ci_level"):
            bootstrap_intervals.replicate_matrix_intervals(
                matrix, ci_level=bad
            )


# --------------------------------------------------------------------------
# plan#4: n_games_backing min convention, symmetry, and validation
# --------------------------------------------------------------------------


def test_n_games_backing_min_convention():
    # The weaker side's sample size, hand-picked pairs: (5, 3) -> 3,
    # (3, 5) -> 3 (symmetry), (0, 10) -> 0 (a brand-new side backs
    # nothing), and (0, 0) -> 0.
    assert bootstrap_intervals.n_games_backing(5, 3) == 3
    assert bootstrap_intervals.n_games_backing(3, 5) == 3
    assert bootstrap_intervals.n_games_backing(0, 10) == 0
    assert bootstrap_intervals.n_games_backing(0, 0) == 0


def test_n_games_backing_accepts_numpy_integers():
    # team_map_win_rate returns numpy integer games counts; they must
    # coerce cleanly.
    assert bootstrap_intervals.n_games_backing(
        np.int64(7), np.int64(4)
    ) == 4


def test_n_games_backing_rejects_negative():
    # A negative game count is malformed data.
    with pytest.raises(ValueError, match="non-negative"):
        bootstrap_intervals.n_games_backing(-1, 5)
    with pytest.raises(ValueError, match="non-negative"):
        bootstrap_intervals.n_games_backing(5, -1)

"""Tests for the pure veto-conditional spread helpers
(evaluation/veto_conditional_variance.py).

Covers the four model-free helpers of M37's statistics half with
hand-computed bands on small synthetic arrays:
``unweighted_scoreline_spread`` (odd and even sample counts, since
numpy's linear percentile interpolation differs between them;
``ci_level`` boundary validation; empty-matrix / ragged-row /
zero-category rejection), ``band_widths`` / ``mean_band_width`` (the
``hi - lo`` and mean-of-widths conventions, including the zero-width
degenerate case), and ``weighted_mean_and_variance`` (hand-computed
weighted first/second moments on a tiny 3-row example against an
independently-derived arithmetic result, plus the weights-length-
mismatch / negative-weight / non-positive-total-weight
``ValueError``s). No real fitted artifacts are required by any of these
tests — the helpers are pure.
"""

import numpy as np
import pytest

from evaluation import veto_conditional_variance as vcv

# --------------------------------------------------------------------------
# plan#3: unweighted_scoreline_spread hand-computed bands and validation
# --------------------------------------------------------------------------


def test_unweighted_scoreline_spread_hand_computed_odd_count():
    # Five rows, ci_level=0.90: numpy linear interpolation puts the
    # 5th percentile of column 0 ([0.1, 0.2, 0.3, 0.4, 0.5]) at index
    # (5-1)*0.05 = 0.2 (lo = 0.1 + 0.2*(0.2-0.1) = 0.12) and the 95th
    # at 3.8 (hi = 0.4 + 0.8*(0.5-0.4) = 0.48); column 1 is constant so
    # its band collapses to a point.
    rows = [
        [0.1, 0.2],
        [0.2, 0.2],
        [0.3, 0.2],
        [0.4, 0.2],
        [0.5, 0.2],
    ]
    bands = vcv.unweighted_scoreline_spread(rows, ci_level=0.90)
    assert bands[0] == pytest.approx((0.12, 0.48))
    assert bands[1] == pytest.approx((0.2, 0.2))
    assert len(bands) == 2
    for lo, hi in bands:
        assert lo <= hi


def test_unweighted_scoreline_spread_hand_computed_even_count():
    # Four rows: the 5th percentile of column 0 ([0.1, 0.2, 0.3, 0.4])
    # sits at index (4-1)*0.05 = 0.15 (lo = 0.1 + 0.15*(0.2-0.1) =
    # 0.115) and the 95th at 2.85 (hi = 0.3 + 0.85*(0.4-0.3) = 0.385) —
    # the interpolation differs from the odd-count case, which is why
    # both are pinned here.
    rows = [
        [0.1, 0.5],
        [0.2, 0.5],
        [0.3, 0.5],
        [0.4, 0.5],
    ]
    bands = vcv.unweighted_scoreline_spread(rows, ci_level=0.90)
    assert bands[0] == pytest.approx((0.115, 0.385))
    assert bands[1] == pytest.approx((0.5, 0.5))


def test_unweighted_scoreline_spread_matches_numpy_percentile():
    # A cross-check against numpy's own per-column percentile for a
    # mid-size synthetic matrix (the implementation is documented as a
    # plain per-column numpy percentile call, so the two must agree
    # exactly).
    rng = np.random.default_rng(42)
    matrix = rng.random((23, 4))
    bands = vcv.unweighted_scoreline_spread(matrix, ci_level=0.80)
    lo = np.percentile(matrix, 10.0, axis=0)
    hi = np.percentile(matrix, 90.0, axis=0)
    for j, (lo_j, hi_j) in enumerate(bands):
        assert lo_j == pytest.approx(float(lo[j]))
        assert hi_j == pytest.approx(float(hi[j]))


def test_unweighted_scoreline_spread_zero_spread_collapses():
    # Every sample row identical: every band collapses to a single
    # point, so every width is exactly 0 per category — the "resolves
    # the moment the veto happens" boundary case where there is
    # effectively no veto-sequence ambiguity.
    rows = [
        [0.6, 0.3, 0.1],
        [0.6, 0.3, 0.1],
        [0.6, 0.3, 0.1],
        [0.6, 0.3, 0.1],
    ]
    bands = vcv.unweighted_scoreline_spread(rows, ci_level=0.90)
    assert all(lo == hi for lo, hi in bands)
    assert vcv.band_widths(bands) == pytest.approx((0.0, 0.0, 0.0))
    assert vcv.mean_band_width(bands) == pytest.approx(0.0)


def test_unweighted_scoreline_spread_single_row():
    # A one-row sample: both percentiles collapse to that row's values
    # (lo == hi per category) rather than erroring.
    bands = vcv.unweighted_scoreline_spread(
        [[0.42, 0.33, 0.25]], ci_level=0.90
    )
    assert len(bands) == 3
    for j, (lo, hi) in enumerate(bands):
        assert lo == hi == pytest.approx([0.42, 0.33, 0.25][j])


def test_unweighted_scoreline_spread_rejects_empty():
    # A percentile band over zero draws is undefined: hard error, not a
    # silent NaN.
    with pytest.raises(ValueError, match="at least one sample"):
        vcv.unweighted_scoreline_spread([])


def test_unweighted_scoreline_spread_rejects_ragged_rows():
    # Rows of differing lengths cannot form a per-column band: hard
    # error naming the offending row.
    with pytest.raises(ValueError, match="ragged"):
        vcv.unweighted_scoreline_spread(
            [[0.1, 0.2, 0.3], [0.1, 0.2]]
        )


def test_unweighted_scoreline_spread_rejects_non_1d_row():
    # Each row must be a 1-D scoreline vector, not a nested matrix.
    with pytest.raises(ValueError, match="1-D"):
        vcv.unweighted_scoreline_spread([[0.1, 0.2], [[0.3], [0.4]]])


def test_unweighted_scoreline_spread_rejects_zero_categories():
    # A zero-category row has no columns to band.
    with pytest.raises(ValueError, match="at least one category"):
        vcv.unweighted_scoreline_spread([[], []])


def test_unweighted_scoreline_spread_ci_level_boundaries():
    # ci_level must be strictly inside (0, 1): 0.0 and 1.0 (degenerate
    # bands) as well as negative and >1 values all raise; non-numeric
    # inputs raise too.
    rows = [[0.1, 0.2], [0.2, 0.3]]
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ci_level"):
            vcv.unweighted_scoreline_spread(rows, ci_level=bad)
    with pytest.raises(ValueError, match="ci_level"):
        vcv.unweighted_scoreline_spread(rows, ci_level="wide")


# --------------------------------------------------------------------------
# plan#3: band_widths / mean_band_width conventions
# --------------------------------------------------------------------------


def test_band_widths_and_mean_band_width_hand_computed():
    # Hand-computed widths from a bands tuple: (0.1, 0.5) -> 0.4,
    # (0.2, 0.3) -> 0.1, (0.0, 0.0) -> 0.0; the mean is
    # (0.4 + 0.1 + 0.0) / 3 = 0.1666... — the single per-series scalar
    # headline number.
    bands = ((0.1, 0.5), (0.2, 0.3), (0.0, 0.0))
    assert vcv.band_widths(bands) == pytest.approx((0.4, 0.1, 0.0))
    assert vcv.mean_band_width(bands) == pytest.approx(0.5 / 3.0)


def test_band_widths_rejects_empty():
    # A zero-category series has no widths to report.
    with pytest.raises(ValueError, match="at least one band"):
        vcv.band_widths([])


def test_band_widths_rejects_malformed_band():
    # A band entry that is not a (lo, hi) pair cannot be widened.
    with pytest.raises(ValueError, match="(lo, hi)"):
        vcv.band_widths([(0.1, 0.5, 0.2)])


# --------------------------------------------------------------------------
# plan#3: weighted_mean_and_variance hand-computed moments and validation
# --------------------------------------------------------------------------


def test_weighted_mean_and_variance_hand_computed():
    # A tiny 3-row, 4-category example with weights [0.5, 0.3, 0.2]
    # (summing to 1): the weighted means are 0.5*0.50 + 0.3*0.45 +
    # 0.2*0.40 = 0.465 for category 0 and 0.5*0.30 + 0.3*0.35 +
    # 0.2*0.40 = 0.335 for category 1, with categories 2/3 constant at
    # 0.10. The weighted variances are
    # 0.5*0.035^2 + 0.3*0.015^2 + 0.2*0.065^2 = 0.001525 for both
    # varying categories and exactly 0.0 for the constant ones — all
    # independently re-derived by hand from the weighted-mean-and-
    # variance-about-it definition.
    rows = [
        [0.50, 0.30, 0.10, 0.10],
        [0.45, 0.35, 0.10, 0.10],
        [0.40, 0.40, 0.10, 0.10],
    ]
    weights = [0.5, 0.3, 0.2]
    means, variances = vcv.weighted_mean_and_variance(rows, weights)
    assert means == pytest.approx((0.465, 0.335, 0.10, 0.10))
    assert variances == pytest.approx((0.001525, 0.001525, 0.0, 0.0))
    # The means themselves form a valid simplex (they are a convex
    # combination of simplex rows with weights summing to 1).
    assert sum(means) == pytest.approx(1.0)


def test_weighted_mean_and_variance_matches_numpy_weighted_arithmetic():
    # A cross-check against numpy's own weighted-average primitive and a
    # directly-written weighted variance loop, on a mid-size synthetic
    # matrix with non-normalized weights (the W-normalized definition
    # must hold for any positive weight vector).
    rng = np.random.default_rng(7)
    matrix = rng.random((11, 5))
    matrix /= matrix.sum(axis=1, keepdims=True)
    weights = [0.5, 0.5, 1.0, 1.5, 0.25, 0.75, 1.0, 1.0, 0.1, 0.9, 2.0]
    means, variances = vcv.weighted_mean_and_variance(matrix, weights)
    expected_means = np.average(matrix, axis=0, weights=weights)
    total = sum(weights)
    expected_variances = np.average(
        (matrix - expected_means) ** 2, axis=0, weights=weights
    )
    assert means == pytest.approx(tuple(float(m) for m in expected_means))
    assert variances == pytest.approx(
        tuple(float(v) for v in expected_variances)
    )
    assert total > 0.0


def test_weighted_mean_and_variance_accepts_normalized_weights():
    # M31's own normalized weights (summing to 1 within float error):
    # the W-normalized definition coincides with the plain sum-w_i=1
    # population variance.
    rows = [[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]
    weights = [0.5, 0.3, 0.2]
    means, variances = vcv.weighted_mean_and_variance(rows, weights)
    assert means == pytest.approx((0.43, 0.57))
    assert variances == pytest.approx(
        (0.5 * (0.5 - 0.43) ** 2 + 0.3 * (0.4 - 0.43) ** 2 + 0.2 * (0.3 - 0.43) ** 2,
         0.5 * (0.5 - 0.57) ** 2 + 0.3 * (0.6 - 0.57) ** 2 + 0.2 * (0.7 - 0.57) ** 2)
    )


def test_weighted_mean_and_variance_weights_length_mismatch():
    # Exactly one weight per sample row is required.
    rows = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    with pytest.raises(ValueError, match="exactly one weight"):
        vcv.weighted_mean_and_variance(rows, [0.5, 0.5])


def test_weighted_mean_and_variance_negative_weight():
    # A negative weight is malformed probability mass.
    rows = [[0.1, 0.2], [0.2, 0.3]]
    with pytest.raises(ValueError, match="non-negative"):
        vcv.weighted_mean_and_variance(rows, [0.5, -0.1])


def test_weighted_mean_and_variance_non_positive_total_weight():
    # An all-zero weight set leaves both moments undefined.
    rows = [[0.1, 0.2], [0.2, 0.3]]
    with pytest.raises(ValueError, match="strictly positive"):
        vcv.weighted_mean_and_variance(rows, [0.0, 0.0])


def test_weighted_mean_and_variance_rejects_empty_and_ragged():
    # The same malformed-matrix guards as the band helper: no rows and
    # ragged rows both raise.
    with pytest.raises(ValueError, match="at least one sample"):
        vcv.weighted_mean_and_variance([], [])
    with pytest.raises(ValueError, match="ragged"):
        vcv.weighted_mean_and_variance(
            [[0.1, 0.2, 0.3], [0.1, 0.2]], [0.5, 0.5]
        )

"""Tests for the pure per-category reliability-diagram helpers
(evaluation/reliability_diagrams.py).

Covers the two public functions of M38's statistics half with
hand-computed quantile bins on small synthetic arrays:
``category_reliability_bins`` (the one-vs-rest quantile binning
routine: odd-count and even-count splits via numpy.array_split's own
as-equal-as-possible semantics, the stable-sort tie-handling rule —
ties keep original row order so a run of equal predicted probabilities
is split deterministically — and the per-bin mean/observed/gap
values) and ``build_reliability_report`` (multi-category K=4 and K=6
hand-computed examples verifying the per-category count-weighted ECE
and full genericity over K, plus the consistency of the report's
per-category bins with a direct ``category_reliability_bins`` call).
The full ValueError contract from the plan's design decision 5 is
pinned: empty input, ragged rows, non-1-D rows, zero-category rows,
category_labels length mismatch, true_indices length mismatch,
true_indices out of range, n_bins < 1, n_bins > n_eval, and non-integer
n_bins. No real fitted artifacts are required — the helpers are pure.
"""

import json

import numpy as np
import pytest

from evaluation import reliability_diagrams as rd

# --------------------------------------------------------------------------
# plan#3: category_reliability_bins hand-computed quantile bins
# --------------------------------------------------------------------------


def test_category_reliability_bins_odd_count_hand_computed():
    # Seven observations split into 3 bins: numpy.array_split puts the
    # extra observation in the FIRST bins (sizes 3, 2, 2). Ascending
    # predicted probs [0.1..0.7] with indicators [1,0,1,0,0,1,0] give
    # bin0 = (0.1, 0.2, 0.3) mean 0.2, observed 2/3, gap 0.4666...;
    # bin1 = (0.4, 0.5) mean 0.45, observed 0, gap 0.45; bin2 = (0.6,
    # 0.7) mean 0.65, observed 0.5, gap 0.15.
    bins = rd.category_reliability_bins(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [1, 0, 1, 0, 0, 1, 0],
        3,
    )
    assert [b["bin_index"] for b in bins] == [0, 1, 2]
    assert [b["count"] for b in bins] == [3, 2, 2]
    assert sum(b["count"] for b in bins) == 7
    assert [b["mean_predicted_prob"] for b in bins] == pytest.approx(
        [0.2, 0.45, 0.65]
    )
    assert [b["observed_frequency"] for b in bins] == pytest.approx(
        [2.0 / 3.0, 0.0, 0.5]
    )
    assert [b["gap"] for b in bins] == pytest.approx(
        [abs(0.2 - 2.0 / 3.0), 0.45, 0.15]
    )
    # Bins are ordered by ascending mean predicted probability.
    means = [b["mean_predicted_prob"] for b in bins]
    assert means == sorted(means)


def test_category_reliability_bins_even_count_hand_computed():
    # Six observations split into 3 bins of equal size 2 (no remainder):
    # probs [0.1..0.6] with indicators [1,0,0,1,0,0] give bin0 = (0.1,
    # 0.2) mean 0.15 observed 0.5 gap 0.35; bin1 = (0.3, 0.4) mean 0.35
    # observed 0.5 gap 0.15; bin2 = (0.5, 0.6) mean 0.55 observed 0.0
    # gap 0.55.
    bins = rd.category_reliability_bins(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1, 0, 0, 1, 0, 0],
        3,
    )
    assert [b["count"] for b in bins] == [2, 2, 2]
    assert [b["mean_predicted_prob"] for b in bins] == pytest.approx(
        [0.15, 0.35, 0.55]
    )
    assert [b["observed_frequency"] for b in bins] == pytest.approx(
        [0.5, 0.5, 0.0]
    )
    assert [b["gap"] for b in bins] == pytest.approx([0.35, 0.15, 0.55])


def test_category_reliability_bins_accepts_boolean_indicators():
    # The is_true_category vector may carry numpy bools or plain Python
    # booleans as well as 0/1 ints — the one-vs-rest indicator is a
    # binary flag either way and the results are identical.
    bins_ints = rd.category_reliability_bins(
        [0.25, 0.5, 0.75], [1, 0, 1], 3
    )
    bins_bools = rd.category_reliability_bins(
        [0.25, 0.5, 0.75], [True, False, True], 3
    )
    bins_np = rd.category_reliability_bins(
        [0.25, 0.5, 0.75], np.array([True, False, True]), 3
    )
    assert bins_ints == bins_bools == bins_np
    assert [b["observed_frequency"] for b in bins_ints] == pytest.approx(
        [1.0, 0.0, 1.0]
    )


def test_category_reliability_bins_stable_sort_splits_ties_deterministically():
    # Every predicted probability is 0.5, so the ascending sort is a
    # pure tie-break: the STABLE sort keeps original row order, and
    # array_split (5 rows into 3 bins: sizes 2, 2, 1) then gives bin0
    # rows 0-1 (true 1, 0 -> observed 0.5), bin1 rows 2-3 (true 1, 0 ->
    # observed 0.5) and bin2 row 4 (true 1 -> observed 1.0). A
    # non-stable sort could permute the tied rows and change bin2's
    # observed frequency, so this pins the deterministic rule.
    bins = rd.category_reliability_bins(
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [1, 0, 1, 0, 1],
        3,
    )
    assert [b["count"] for b in bins] == [2, 2, 1]
    assert [b["observed_frequency"] for b in bins] == pytest.approx(
        [0.5, 0.5, 1.0]
    )
    assert [b["gap"] for b in bins] == pytest.approx([0.0, 0.0, 0.5])


def test_category_reliability_bins_perfect_calibration_zero_ece():
    # A toy perfectly calibrated case engineered so each bin's mean
    # predicted probability exactly equals its observed frequency: 6
    # rows split into 3 bins of 2, with each pair carrying a constant
    # predicted value whose true fraction equals that value (0.0 -> 0/2
    # true, 0.5 -> 1/2 true, 1.0 -> 2/2 true). Every gap is therefore
    # 0 and the count-weighted ECE over the bins is 0 — the
    # "reliability curve on the diagonal" reference.
    bins = rd.category_reliability_bins(
        [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        [0, 0, 1, 0, 1, 1],
        3,
    )
    assert [b["gap"] for b in bins] == pytest.approx([0.0, 0.0, 0.0])
    assert sum(b["count"] * b["gap"] for b in bins) == pytest.approx(
        0.0
    )


def test_category_reliability_bins_matches_manual_reimplementation():
    # A cross-check against a directly written reference implementation
    # (sort, array_split, per-bin means) on a mid-size synthetic vector
    # with ties, so the numpy wiring is pinned independently of the
    # hand-computed small cases above.
    rng = np.random.default_rng(11)
    probs = rng.integers(0, 5, size=29).astype(float) / 4.0
    truth = rng.integers(0, 2, size=29)
    bins = rd.category_reliability_bins(probs, truth, 4)
    order = np.argsort(np.asarray(probs), kind="stable")
    sorted_probs = np.asarray(probs)[order]
    sorted_truth = np.asarray(truth)[order]
    for bin_index, group in enumerate(np.array_split(np.arange(29), 4)):
        expected_mean = float(np.mean(sorted_probs[group]))
        expected_obs = float(np.mean(sorted_truth[group]))
        assert bins[bin_index]["count"] == len(group)
        assert bins[bin_index]["mean_predicted_prob"] == pytest.approx(
            expected_mean
        )
        assert bins[bin_index]["observed_frequency"] == pytest.approx(
            expected_obs
        )
        assert bins[bin_index]["gap"] == pytest.approx(
            abs(expected_mean - expected_obs)
        )


# --------------------------------------------------------------------------
# plan#3: category_reliability_bins ValueError contract (design decision 5)
# --------------------------------------------------------------------------


def test_category_reliability_bins_rejects_empty():
    # A reliability curve over zero observations is undefined.
    with pytest.raises(ValueError, match="at least one observation"):
        rd.category_reliability_bins([], [], 3)


def test_category_reliability_bins_rejects_indicator_length_mismatch():
    # Exactly one binary indicator per observation is required.
    with pytest.raises(ValueError, match="one binary indicator"):
        rd.category_reliability_bins([0.1, 0.2, 0.3], [1, 0], 3)


def test_category_reliability_bins_rejects_non_1d_probs():
    # A per-category column must be a flat vector, not a nested matrix.
    with pytest.raises(ValueError, match="1-D"):
        rd.category_reliability_bins([[0.1, 0.2], [0.3, 0.4]], [1, 0], 2)


def test_category_reliability_bins_rejects_non_1d_indicators():
    # The indicator vector must be a flat vector too.
    with pytest.raises(ValueError, match="1-D"):
        rd.category_reliability_bins([0.1, 0.2], [[1], [0]], 2)


def test_category_reliability_bins_n_bins_bounds():
    # n_bins < 1 and n_bins > n_eval both raise (the latter fails loud
    # rather than silently under-filling bins — decision 5's
    # no-silent-clamping doctrine); a non-integer n_bins raises too.
    probs = [0.1, 0.2, 0.3, 0.4]
    truth = [1, 0, 1, 0]
    with pytest.raises(ValueError, match="at least 1"):
        rd.category_reliability_bins(probs, truth, 0)
    with pytest.raises(ValueError, match="at least 1"):
        rd.category_reliability_bins(probs, truth, -2)
    with pytest.raises(ValueError, match="must not exceed"):
        rd.category_reliability_bins(probs, truth, 5)
    with pytest.raises(ValueError, match="must be an integer"):
        rd.category_reliability_bins(probs, truth, 2.5)


# --------------------------------------------------------------------------
# plan#3: build_reliability_report multi-category hand-computed ECE
# --------------------------------------------------------------------------

# The K=4 hand-computed example: 6 rows, each a valid simplex over four
# categories, true ordinals [0, 0, 1, 1, 2, 3]. Per category the
# ascending-sorted predicted column and its true indicators produce
# three 2-row bins; the count-weighted ECEs below were re-derived by
# hand (see each test's comment).
_K4_ROWS = [
    [0.6, 0.2, 0.1, 0.1],
    [0.5, 0.3, 0.1, 0.1],
    [0.4, 0.3, 0.2, 0.1],
    [0.3, 0.5, 0.1, 0.1],
    [0.2, 0.2, 0.4, 0.2],
    [0.1, 0.1, 0.3, 0.5],
]
_K4_TRUE = [0, 0, 1, 1, 2, 3]
_K4_LABELS = ("cat0", "cat1", "cat2", "cat3")


def test_build_reliability_report_k4_hand_computed_ece():
    # K=4, n_eval=6, n_bins=3 (bins of 2). Category 0: sorted column
    # [0.1,0.2,0.3,0.4,0.5,0.6] with indicators [0,0,0,0,1,1] gives
    # bins (0.15,0),(0.35,0),(0.55,1.0) -> ECE =
    # 2*(0.15+0.35+0.45)/6 = 1.9/6 = 0.31666.... Category 1: sorted
    # [0.1,0.2,0.2,0.3,0.3,0.5] with indicators [0,0,0,0,1,1] gives
    # (0.15,0),(0.25,0),(0.4,1.0) -> ECE = 2*(0.15+0.25+0.6)/6 = 2/6 =
    # 0.33333.... Category 2: sorted [0.1,0.1,0.1,0.2,0.3,0.4] with
    # indicators [0,0,0,0,0,1] gives (0.1,0),(0.15,0),(0.35,0.5) ->
    # ECE = 2*(0.1+0.15+0.15)/6 = 0.8/6 = 0.13333.... Category 3:
    # sorted [0.1,0.1,0.1,0.1,0.2,0.5] with indicators [0,0,0,0,0,1]
    # gives (0.1,0),(0.1,0),(0.35,0.5) -> ECE = 2*(0.1+0.1+0.15)/6 =
    # 0.7/6 = 0.11666....
    report = rd.build_reliability_report(
        _K4_ROWS, _K4_TRUE, _K4_LABELS, n_bins=3
    )
    assert report["n_eval"] == 6
    assert report["n_bins"] == 3
    assert [c["category"] for c in report["categories"]] == list(
        _K4_LABELS
    )
    eces = [
        c["expected_calibration_error"] for c in report["categories"]
    ]
    assert eces == pytest.approx(
        [1.9 / 6.0, 2.0 / 6.0, 0.8 / 6.0, 0.7 / 6.0]
    )
    # Every ECE is non-negative and every category's bin counts sum to
    # n_eval (the report's structural sanity the BUILD note records).
    for category in report["categories"]:
        assert category["expected_calibration_error"] >= 0.0
        assert sum(b["count"] for b in category["bins"]) == 6
        assert len(category["bins"]) == 3


def test_build_reliability_report_bins_match_direct_call():
    # The report's per-category bins must be byte-identical to a direct
    # category_reliability_bins call on that category's extracted
    # column and indicator vector — the wiring consistency check.
    report = rd.build_reliability_report(
        _K4_ROWS, _K4_TRUE, _K4_LABELS, n_bins=3
    )
    matrix = np.asarray(_K4_ROWS, dtype=float)
    true_array = np.asarray(_K4_TRUE)
    for c, category in enumerate(report["categories"]):
        direct = rd.category_reliability_bins(
            matrix[:, c], true_array == c, 3
        )
        assert category["bins"] == direct


def test_build_reliability_report_k6_generic_over_k():
    # K=6 genericity: 6 rows where row i puts 0.5 on category i and 0.1
    # on the other five (a valid simplex), true ordinals 0..5. For
    # every category j the ascending-sorted predicted column is
    # [0.1 x5, 0.5] (the 0.5 row being the one truly in j), so with
    # n_bins=3 (bins of 2) every category has identical bins (0.1,0),
    # (0.1,0), (0.3,0.5) and ECE = 2*(0.1+0.1+0.2)/6 = 0.8/6 =
    # 0.13333... — proving the same code scores a K=6 matrix with no
    # K-specific knowledge anywhere.
    rows = [
        [0.5 if j == i else 0.1 for j in range(6)] for i in range(6)
    ]
    labels = [f"scoreline-{j}" for j in range(6)]
    report = rd.build_reliability_report(rows, list(range(6)), labels, 3)
    assert report["n_eval"] == 6
    assert report["n_bins"] == 3
    assert len(report["categories"]) == 6
    for category in report["categories"]:
        assert category["expected_calibration_error"] == pytest.approx(
            0.8 / 6.0
        )
        assert [b["observed_frequency"] for b in category["bins"]] == (
            pytest.approx([0.0, 0.0, 0.5])
        )
        assert [b["mean_predicted_prob"] for b in category["bins"]] == (
            pytest.approx([0.1, 0.1, 0.3])
        )


def test_build_reliability_report_json_serializable():
    # The whole report round-trips through json.dumps (every value is a
    # plain str/int/float/list/dict).
    report = rd.build_reliability_report(
        _K4_ROWS, _K4_TRUE, _K4_LABELS, n_bins=3
    )
    parsed = json.loads(json.dumps(report))
    assert parsed == report


# --------------------------------------------------------------------------
# plan#3: build_reliability_report ValueError contract (design decision 5)
# --------------------------------------------------------------------------


def test_build_reliability_report_rejects_empty():
    # A report over zero observation rows is undefined.
    with pytest.raises(ValueError, match="at least one observation"):
        rd.build_reliability_report([], [], _K4_LABELS, 3)


def test_build_reliability_report_rejects_ragged_rows():
    # Rows of differing lengths cannot form a per-column report: hard
    # error naming the offending row.
    with pytest.raises(ValueError, match="ragged"):
        rd.build_reliability_report(
            [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3]],
            [0, 1],
            _K4_LABELS,
            3,
        )


def test_build_reliability_report_rejects_non_1d_row():
    # Each observation row must be a 1-D probability vector.
    with pytest.raises(ValueError, match="1-D"):
        rd.build_reliability_report(
            [[0.1, 0.2], [[0.3], [0.4]]], [0, 1], ("a", "b"), 2
        )


def test_build_reliability_report_rejects_zero_categories():
    # A zero-category row has no columns to calibrate.
    with pytest.raises(ValueError, match="at least one category"):
        rd.build_reliability_report([[], []], [0, 0], [], 1)


def test_build_reliability_report_rejects_category_labels_length_mismatch():
    # One label per category column is required.
    with pytest.raises(ValueError, match="category_labels"):
        rd.build_reliability_report(
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            [0, 1],
            ("only-three", "labels", "here"),
            2,
        )


def test_build_reliability_report_rejects_true_indices_length_mismatch():
    # Exactly one true index per observation row is required.
    with pytest.raises(ValueError, match="exactly one true index"):
        rd.build_reliability_report(
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            [0],
            _K4_LABELS,
            2,
        )


def test_build_reliability_report_rejects_true_indices_out_of_range():
    # A true index must name a real category column: -1 and K are both
    # outside [0, K) and raise (a silent all-False indicator for the
    # true category would corrupt every observed frequency).
    rows = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        rd.build_reliability_report(rows, [-1, 1], _K4_LABELS, 2)
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        rd.build_reliability_report(rows, [0, 4], _K4_LABELS, 2)


def test_build_reliability_report_n_bins_bounds():
    # The same no-silent-clamping n_bins contract as the per-category
    # routine: < 1, > n_eval and non-integer values all raise.
    rows = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    with pytest.raises(ValueError, match="at least 1"):
        rd.build_reliability_report(rows, [0, 1], _K4_LABELS, 0)
    with pytest.raises(ValueError, match="must not exceed"):
        rd.build_reliability_report(rows, [0, 1], _K4_LABELS, 3)
    with pytest.raises(ValueError, match="must be an integer"):
        rd.build_reliability_report(rows, [0, 1], _K4_LABELS, 1.5)

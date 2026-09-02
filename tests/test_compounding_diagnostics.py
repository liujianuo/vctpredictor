"""Tests for the M35 compounding-diagnostics analysis module
(evaluation/compounding_diagnostics.py).

Covers the two pure report functions:

- ``sweep_rate_report``: a hand-computed Bo3-only fixture (exact
  predicted mean sweep probability, observed sweep rate, calibration
  gap, and both directional breakdowns by hand arithmetic); a mixed
  Bo3+Bo5 fixture proving two separate groups are reported with no
  cross-``K`` blending; the degenerate Bo1 tautology case (both
  outcomes are trivially sweeps); and the empty-input / length-mismatch
  ``ValueError`` matrix.
- ``map1_predicts_map2_report``: a small fully-deterministic fixture
  hand-verifying the group-mean residuals and ``observed_diff``
  arithmetic exactly; a brute-force-enumerable 4-match case whose
  empirical permutation p-value is cross-checked against the exact
  enumerated permutation distribution (statistical tolerance, since
  the report samples rather than enumerates); same-seed/different-seed
  determinism of the permutation draw; and the ``ValueError`` matrix
  (too-few-matches, empty map-1-outcome subgroup, invalid
  ``n_permutations``).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from evaluation.compounding_diagnostics import (
    map1_predicts_map2_report,
    sweep_rate_report,
)

# The schema-contract columns the two report functions read (referenced
# through the module constants so the tests never hardcode a stale
# name).
SWEEP_COLUMNS = (
    "match_id",
    "best_of",
    "best_of_int",
    "probabilities",
    "outcome_index",
)
MAP_COLUMNS = (
    "match_id",
    "map_index",
    "outcome_ordinal",
    "p_a_regulation",
    "p_a_ot",
)


def _scored_series_df(rows):
    """Build a scored-series-shaped DataFrame from hand-written rows.

    Args:
        rows: An iterable of dicts, each with keys ``match_id``,
            ``best_of``, ``best_of_int``, ``probabilities`` (a
            sequence of ``best_of_int + 1`` floats) and
            ``outcome_index``.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SWEEP_COLUMNS`
            columns, one row per input dict in order.

    Raises:
        Nothing.
    """
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def _scored_maps_df(rows):
    """Build a scored-maps-shaped DataFrame from hand-written rows.

    Args:
        rows: An iterable of dicts, each with keys ``match_id``,
            ``map_index``, ``outcome_ordinal``, ``p_a_regulation`` and
            ``p_a_ot``.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`MAP_COLUMNS`
            columns, one row per input dict in order.

    Raises:
        Nothing.
    """
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def _exact_permutation_p(residuals, labels, observed_diff):
    """Compute the exact two-sided permutation p-value by enumeration.

    Enumerates every distinct relabeling of the boolean ``labels``
    array (all ``C(n, k)`` combinations of which ``k`` positions are
    "A") holding ``residuals`` fixed, recomputes the group-mean
    difference for each, and returns the fraction whose absolute value
    is at least ``abs(observed_diff)`` — the exact value the report's
    sampled permutation test approximates (for a tiny ``n`` the whole
    relabeling space is enumerable, so the exact p-value is a ground
    truth the sampled p-value can be compared against with statistical
    tolerance).

    Args:
        residuals: A 1-D numeric array of map-2 residuals, one per
            eligible match.
        labels: A 1-D boolean array of map-1-A-won flags, one per
            eligible match; ``k = labels.sum()`` positions are "A".
        observed_diff: The observed group-mean difference
            ``mean(residuals | labels) - mean(residuals | ~labels)``
            to compare each relabeling against (two-sided).

    Returns:
        The exact p-value as a ``float`` in ``(0.0, 1.0]`` (the
            identity relabeling always qualifies, so it is never 0.0).

    Raises:
        Nothing.
    """
    residuals = np.asarray(residuals, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n = len(residuals)
    k = int(labels.sum())
    count = 0
    total = 0
    for combo in itertools.combinations(range(n), k):
        a_mask = np.zeros(n, dtype=bool)
        a_mask[list(combo)] = True
        diff = residuals[a_mask].mean() - residuals[~a_mask].mean()
        total += 1
        if abs(diff) >= abs(observed_diff):
            count += 1
    return count / total


# --------------------------------------------------------------------------
# sweep_rate_report: hand-computed Bo3-only fixture
# --------------------------------------------------------------------------


def test_sweep_rate_report_hand_computed_bo3():
    # Three Bo3 rows with hand-picked probabilities/outcome indices:
    #   m1: probs [0.5, 0.2, 0.2, 0.1], idx 1 (2-1, not a sweep)
    #       -> combined sweep prob 0.5+0.1 = 0.6, a-sweep 0.5, b-sweep 0.1
    #   m2: probs [0.4, 0.1, 0.3, 0.2], idx 0 (2-0, A sweep)
    #       -> combined 0.4+0.2 = 0.6, a-sweep 0.4, b-sweep 0.2
    #   m3: probs [0.6, 0.1, 0.1, 0.2], idx 3 (0-2, B sweep)
    #       -> combined 0.6+0.2 = 0.8, a-sweep 0.6, b-sweep 0.2
    # Predicted combined mean = (0.6+0.6+0.8)/3 = 2/3; observed sweep
    # rate = 2/3 (m2, m3); gap = 0.0 exactly. Directional: a predicted
    # (0.5+0.4+0.6)/3 = 0.5 vs observed 1/3 (m2 only) -> gap +1/6; b
    # predicted (0.1+0.2+0.2)/3 = 1/6 vs observed 1/3 (m3 only) ->
    # gap -1/6 (the nonzero directional gaps show the asymmetry
    # detection works even when the combined gap is exactly zero).
    scored = _scored_series_df(
        [
            {"match_id": "m1", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.5, 0.2, 0.2, 0.1], "outcome_index": 1},
            {"match_id": "m2", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.4, 0.1, 0.3, 0.2], "outcome_index": 0},
            {"match_id": "m3", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.6, 0.1, 0.1, 0.2], "outcome_index": 3},
        ]
    )
    report = sweep_rate_report(scored)

    assert report["n_eval_total"] == 3
    assert set(report) == {"n_eval_total", "Bo3"}
    block = report["Bo3"]
    assert block["n_eval"] == 3
    assert block["predicted_mean_sweep_prob"] == pytest.approx(2.0 / 3.0)
    assert block["observed_sweep_rate"] == pytest.approx(2.0 / 3.0)
    assert block["sweep_calibration_gap"] == pytest.approx(0.0)
    assert block["predicted_mean_a_sweep_prob"] == pytest.approx(0.5)
    assert block["observed_a_sweep_rate"] == pytest.approx(1.0 / 3.0)
    assert block["a_sweep_calibration_gap"] == pytest.approx(1.0 / 6.0)
    assert block["predicted_mean_b_sweep_prob"] == pytest.approx(1.0 / 6.0)
    assert block["observed_b_sweep_rate"] == pytest.approx(1.0 / 3.0)
    assert block["b_sweep_calibration_gap"] == pytest.approx(-1.0 / 6.0)


# --------------------------------------------------------------------------
# sweep_rate_report: mixed Bo3+Bo5 fixture (no cross-K blending)
# --------------------------------------------------------------------------


def test_sweep_rate_report_mixed_bo3_bo5_no_cross_k_blending():
    # Two Bo3 rows (K=4, sweep indices 0 and 3) and two Bo5 rows
    # (K=6, sweep indices 0 and 5) in one table: the report must
    # produce two separate groups with no cross-K blending.
    #   m1 Bo3: probs [0.5, 0.2, 0.2, 0.1], idx 0 -> pred 0.6, sweep
    #   m2 Bo3: probs [0.4, 0.3, 0.2, 0.1], idx 1 -> pred 0.5
    #   Bo3 group: predicted (0.6+0.5)/2 = 0.55, observed 1/2 = 0.5,
    #   gap +0.05; a predicted (0.5+0.4)/2 = 0.45, b predicted
    #   (0.1+0.1)/2 = 0.1.
    #   m3 Bo5: probs [0.3, 0.2, 0.1, 0.1, 0.1, 0.2], idx 0 ->
    #   pred 0.3+0.2 = 0.5, sweep
    #   m4 Bo5: probs [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], idx 3 ->
    #   pred 0.2+0.1 = 0.3
    #   Bo5 group: predicted (0.5+0.3)/2 = 0.4, observed 1/2 = 0.5,
    #   gap -0.1; a predicted (0.3+0.2)/2 = 0.25, b predicted
    #   (0.2+0.1)/2 = 0.15.
    scored = _scored_series_df(
        [
            {"match_id": "m1", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.5, 0.2, 0.2, 0.1], "outcome_index": 0},
            {"match_id": "m2", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.4, 0.3, 0.2, 0.1], "outcome_index": 1},
            {"match_id": "m3", "best_of": "Bo5", "best_of_int": 5,
             "probabilities": [0.3, 0.2, 0.1, 0.1, 0.1, 0.2],
             "outcome_index": 0},
            {"match_id": "m4", "best_of": "Bo5", "best_of_int": 5,
             "probabilities": [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
             "outcome_index": 3},
        ]
    )
    report = sweep_rate_report(scored)

    assert report["n_eval_total"] == 4
    assert set(report) == {"n_eval_total", "Bo3", "Bo5"}

    bo3 = report["Bo3"]
    assert bo3["n_eval"] == 2
    assert bo3["predicted_mean_sweep_prob"] == pytest.approx(0.55)
    assert bo3["observed_sweep_rate"] == pytest.approx(0.5)
    assert bo3["sweep_calibration_gap"] == pytest.approx(0.05)
    assert bo3["predicted_mean_a_sweep_prob"] == pytest.approx(0.45)
    assert bo3["observed_a_sweep_rate"] == pytest.approx(0.5)
    assert bo3["a_sweep_calibration_gap"] == pytest.approx(-0.05)
    assert bo3["predicted_mean_b_sweep_prob"] == pytest.approx(0.1)
    assert bo3["observed_b_sweep_rate"] == pytest.approx(0.0)
    assert bo3["b_sweep_calibration_gap"] == pytest.approx(0.1)

    bo5 = report["Bo5"]
    assert bo5["n_eval"] == 2
    assert bo5["predicted_mean_sweep_prob"] == pytest.approx(0.4)
    assert bo5["observed_sweep_rate"] == pytest.approx(0.5)
    assert bo5["sweep_calibration_gap"] == pytest.approx(-0.1)
    assert bo5["predicted_mean_a_sweep_prob"] == pytest.approx(0.25)
    assert bo5["observed_a_sweep_rate"] == pytest.approx(0.5)
    assert bo5["a_sweep_calibration_gap"] == pytest.approx(-0.25)
    assert bo5["predicted_mean_b_sweep_prob"] == pytest.approx(0.15)
    assert bo5["observed_b_sweep_rate"] == pytest.approx(0.0)
    assert bo5["b_sweep_calibration_gap"] == pytest.approx(0.15)


# --------------------------------------------------------------------------
# sweep_rate_report: degenerate Bo1 tautology (defensive correctness)
# --------------------------------------------------------------------------


def test_sweep_rate_report_bo1_degenerate_tautology():
    # Bo1 has K=2 and both outcome indices (0 and 1) are trivially
    # "sweeps": predicted and observed sweep rate are both 1.0 (the
    # marginal match-win rate), so the combined gap is exactly zero —
    # informationally meaningless, but the report still computes it
    # (defensive correctness per the plan, not exclusion).
    scored = _scored_series_df(
        [
            {"match_id": "m1", "best_of": "Bo1", "best_of_int": 1,
             "probabilities": [0.3, 0.7], "outcome_index": 0},
            {"match_id": "m2", "best_of": "Bo1", "best_of_int": 1,
             "probabilities": [0.6, 0.4], "outcome_index": 1},
        ]
    )
    report = sweep_rate_report(scored)

    assert report["n_eval_total"] == 2
    block = report["Bo1"]
    assert block["n_eval"] == 2
    assert block["predicted_mean_sweep_prob"] == pytest.approx(1.0)
    assert block["observed_sweep_rate"] == pytest.approx(1.0)
    assert block["sweep_calibration_gap"] == pytest.approx(0.0)
    # Directional fields still decompose per side (a predicted
    # (0.3+0.6)/2 = 0.45 vs observed 1/2, b predicted (0.7+0.4)/2 =
    # 0.55 vs observed 1/2).
    assert block["predicted_mean_a_sweep_prob"] == pytest.approx(0.45)
    assert block["observed_a_sweep_rate"] == pytest.approx(0.5)
    assert block["a_sweep_calibration_gap"] == pytest.approx(-0.05)
    assert block["predicted_mean_b_sweep_prob"] == pytest.approx(0.55)
    assert block["observed_b_sweep_rate"] == pytest.approx(0.5)
    assert block["b_sweep_calibration_gap"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# sweep_rate_report: degenerate-input ValueError matrix
# --------------------------------------------------------------------------


def test_sweep_rate_report_empty_input_raises():
    # A mean over zero series is undefined: an empty input table must
    # raise ValueError rather than return a NaN-laden report.
    scored = _scored_series_df([])
    with pytest.raises(ValueError, match="zero scored series"):
        sweep_rate_report(scored)


def test_sweep_rate_report_length_mismatch_raises():
    # A Bo3 row whose probabilities vector is not exactly
    # 2 * series_win_threshold(3) = 4 entries is an internal desync
    # with the M33a contract: rejected loudly, naming the series,
    # rather than silently misindexing probabilities[3].
    scored = _scored_series_df(
        [
            {"match_id": "m1", "best_of": "Bo3", "best_of_int": 3,
             "probabilities": [0.5, 0.2, 0.2], "outcome_index": 0},
        ]
    )
    with pytest.raises(ValueError, match="m1"):
        sweep_rate_report(scored)


# --------------------------------------------------------------------------
# map1_predicts_map2_report: deterministic fixture, hand-verified
# --------------------------------------------------------------------------


def test_map1_predicts_map2_report_hand_computed_observed_diff():
    # Five matches, each with a map-1 and a map-2 row. Map-1 outcomes:
    # m1 A (ord 0), m2 A (ord 1), m3 B (ord 3), m4 A (ord 0), m5 B
    # (ord 2). Map-2 predictions p_a_reg + p_a_ot and true ordinals:
    #   m1: pred 0.4+0.2 = 0.6, ord 0 (A won)   -> residual +0.4
    #   m2: pred 0.2+0.1 = 0.3, ord 3 (B won)   -> residual -0.3
    #   m3: pred 0.5+0.2 = 0.7, ord 1 (A won)   -> residual +0.3
    #   m4: pred 0.3+0.2 = 0.5, ord 0 (A won)   -> residual +0.5
    #   m5: pred 0.3+0.1 = 0.4, ord 2 (B won)   -> residual -0.4
    # map1_a_won: m1 T, m2 T, m3 F, m4 T, m5 F -> n_a 3, n_b 2.
    # mean residual given A = (0.4-0.3+0.5)/3 = 0.2; given B =
    # (0.3-0.4)/2 = -0.05; observed_diff = 0.25 exactly.
    rows = []
    for match_id, map1_ord, pred, map2_ord in [
        ("m1", 0, (0.4, 0.2), 0),
        ("m2", 1, (0.2, 0.1), 3),
        ("m3", 3, (0.5, 0.2), 1),
        ("m4", 0, (0.3, 0.2), 0),
        ("m5", 2, (0.3, 0.1), 2),
    ]:
        rows.append(
            {"match_id": match_id, "map_index": 0,
             "outcome_ordinal": map1_ord, "p_a_regulation": 0.0,
             "p_a_ot": 0.0}
        )
        rows.append(
            {"match_id": match_id, "map_index": 1,
             "outcome_ordinal": map2_ord, "p_a_regulation": pred[0],
             "p_a_ot": pred[1]}
        )
    report = map1_predicts_map2_report(
        _scored_maps_df(rows),
        np.random.default_rng(0),
        n_permutations=10000,
    )

    assert report["n_eligible_matches"] == 5
    assert report["n_map1_a_won"] == 3
    assert report["n_map1_b_won"] == 2
    assert report["mean_residual_given_map1_a_won"] == pytest.approx(0.2)
    assert report["mean_residual_given_map1_b_won"] == pytest.approx(-0.05)
    assert report["observed_diff"] == pytest.approx(0.25)
    assert report["n_permutations"] == 10000
    assert 0.0 < report["p_value_empirical"] <= 1.0


# --------------------------------------------------------------------------
# map1_predicts_map2_report: brute-force-enumerable permutation p-value
# --------------------------------------------------------------------------


def test_map1_predicts_map2_report_permutation_p_matches_exact_enumeration():
    # Four matches; map-1 outcomes A, A, B, B (n_a = n_b = 2) and
    # map-2 residuals [0.8, -0.1, 0.5, -0.6]:
    #   m1: map1 ord 0, map2 ord 0 (A), pred 0.1+0.1 = 0.2 -> +0.8
    #   m2: map1 ord 0, map2 ord 3 (B), pred 0.05+0.05 = 0.1 -> -0.1
    #   m3: map1 ord 3, map2 ord 0 (A), pred 0.3+0.2 = 0.5 -> +0.5
    #   m4: map1 ord 3, map2 ord 3 (B), pred 0.4+0.2 = 0.6 -> -0.6
    # Observed diff = (0.8-0.1)/2 - (0.5-0.6)/2 = 0.35 - (-0.05) = 0.4.
    # All C(4,2) = 6 relabelings are enumerable exactly; 4 of the 6
    # (A={0,1}, A={0,2}, A={1,3}, A={2,3}) have |diff| >= 0.4, so the
    # exact two-sided p-value is 4/6 = 2/3. The sampled permutation
    # p-value at large n_permutations must approximate it within
    # statistical tolerance (sampling, not enumeration).
    rows = []
    for match_id, map1_ord, pred, map2_ord in [
        ("m1", 0, (0.1, 0.1), 0),
        ("m2", 0, (0.05, 0.05), 3),
        ("m3", 3, (0.3, 0.2), 0),
        ("m4", 3, (0.4, 0.2), 3),
    ]:
        rows.append(
            {"match_id": match_id, "map_index": 0,
             "outcome_ordinal": map1_ord, "p_a_regulation": 0.0,
             "p_a_ot": 0.0}
        )
        rows.append(
            {"match_id": match_id, "map_index": 1,
             "outcome_ordinal": map2_ord, "p_a_regulation": pred[0],
             "p_a_ot": pred[1]}
        )
    report = map1_predicts_map2_report(
        _scored_maps_df(rows),
        np.random.default_rng(12345),
        n_permutations=20000,
    )

    assert report["n_eligible_matches"] == 4
    assert report["observed_diff"] == pytest.approx(0.4)
    exact_p = _exact_permutation_p(
        [0.8, -0.1, 0.5, -0.6], [True, True, False, False], 0.4
    )
    assert exact_p == pytest.approx(2.0 / 3.0)
    assert report["p_value_empirical"] == pytest.approx(
        exact_p, abs=0.03
    )


# --------------------------------------------------------------------------
# map1_predicts_map2_report: permutation-draw determinism
# --------------------------------------------------------------------------


def test_map1_predicts_map2_report_same_seed_identical_different_seed_valid():
    # Same seed -> byte-identical report (the permutation draw is
    # driven purely by the caller-supplied rng); a different seed
    # produces a valid p-value in (0, 1] (no assertion on inequality:
    # with a tiny fixture two different draws could in principle
    # coincide, and the contract is validity + reproducibility, not
    # guaranteed divergence).
    rows = []
    for match_id, map1_ord, pred, map2_ord in [
        ("m1", 0, (0.1, 0.1), 0),
        ("m2", 0, (0.05, 0.05), 3),
        ("m3", 3, (0.3, 0.2), 0),
        ("m4", 3, (0.4, 0.2), 3),
    ]:
        rows.append(
            {"match_id": match_id, "map_index": 0,
             "outcome_ordinal": map1_ord, "p_a_regulation": 0.0,
             "p_a_ot": 0.0}
        )
        rows.append(
            {"match_id": match_id, "map_index": 1,
             "outcome_ordinal": map2_ord, "p_a_regulation": pred[0],
             "p_a_ot": pred[1]}
        )
    scored = _scored_maps_df(rows)

    first = map1_predicts_map2_report(
        scored, np.random.default_rng(7), n_permutations=5000
    )
    second = map1_predicts_map2_report(
        scored, np.random.default_rng(7), n_permutations=5000
    )
    assert first == second

    other = map1_predicts_map2_report(
        scored, np.random.default_rng(8), n_permutations=5000
    )
    assert 0.0 < other["p_value_empirical"] <= 1.0


# --------------------------------------------------------------------------
# map1_predicts_map2_report: degenerate-input ValueError matrix
# --------------------------------------------------------------------------


def test_map1_predicts_map2_report_too_few_matches_raises():
    # One match with both map rows -> 1 eligible match < minimum 2.
    scored = _scored_maps_df(
        [
            {"match_id": "m1", "map_index": 0, "outcome_ordinal": 0,
             "p_a_regulation": 0.2, "p_a_ot": 0.1},
            {"match_id": "m1", "map_index": 1, "outcome_ordinal": 0,
             "p_a_regulation": 0.2, "p_a_ot": 0.1},
        ]
    )
    with pytest.raises(ValueError, match="at least 2 matches"):
        map1_predicts_map2_report(
            scored, np.random.default_rng(0), n_permutations=100
        )


def test_map1_predicts_map2_report_no_map2_rows_raises():
    # Two matches with only map-index-0 rows: the inner join yields
    # zero eligible matches (Bo1-style data has no map 2 at all).
    scored = _scored_maps_df(
        [
            {"match_id": "m1", "map_index": 0, "outcome_ordinal": 0,
             "p_a_regulation": 0.2, "p_a_ot": 0.1},
            {"match_id": "m2", "map_index": 0, "outcome_ordinal": 3,
             "p_a_regulation": 0.2, "p_a_ot": 0.1},
        ]
    )
    with pytest.raises(ValueError, match="at least 2 matches"):
        map1_predicts_map2_report(
            scored, np.random.default_rng(0), n_permutations=100
        )


def test_map1_predicts_map2_report_empty_subgroup_raises():
    # Three eligible matches whose map-1 outcomes are all side-A wins:
    # the map-1-B-won subgroup is empty, so the B group mean is
    # undefined — a real, expected possibility at v1's n=15 scale,
    # not defensive theater.
    rows = []
    for match_id in ("m1", "m2", "m3"):
        rows.append(
            {"match_id": match_id, "map_index": 0, "outcome_ordinal": 0,
             "p_a_regulation": 0.0, "p_a_ot": 0.0}
        )
        rows.append(
            {"match_id": match_id, "map_index": 1, "outcome_ordinal": 3,
             "p_a_regulation": 0.3, "p_a_ot": 0.2}
        )
    with pytest.raises(ValueError, match="each direction"):
        map1_predicts_map2_report(
            _scored_maps_df(rows),
            np.random.default_rng(0),
            n_permutations=100,
        )


def test_map1_predicts_map2_report_invalid_n_permutations_raises():
    # n_permutations must be a positive integer-like value: zero,
    # negative, and non-integer values all raise ValueError.
    rows = [
        {"match_id": "m1", "map_index": 0, "outcome_ordinal": 0,
         "p_a_regulation": 0.2, "p_a_ot": 0.1},
        {"match_id": "m1", "map_index": 1, "outcome_ordinal": 0,
         "p_a_regulation": 0.2, "p_a_ot": 0.1},
        {"match_id": "m2", "map_index": 0, "outcome_ordinal": 3,
         "p_a_regulation": 0.2, "p_a_ot": 0.1},
        {"match_id": "m2", "map_index": 1, "outcome_ordinal": 3,
         "p_a_regulation": 0.2, "p_a_ot": 0.1},
    ]
    scored = _scored_maps_df(rows)
    for bad in (0, -5, "many", 2.5):
        with pytest.raises(ValueError, match="n_permutations"):
            map1_predicts_map2_report(
                scored, np.random.default_rng(0), n_permutations=bad
            )
    # A numpy integer scalar is accepted (operator.index coercion).
    report = map1_predicts_map2_report(
        scored, np.random.default_rng(0), n_permutations=np.int64(100)
    )
    assert report["n_permutations"] == 100

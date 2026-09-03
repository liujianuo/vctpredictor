"""Tests for the flat series-scoreline baseline model (M32).

Covers the ``"Bo<N>"`` parser's exact mappings and its ``ValueError``
matrix, hand-computed Bo3/Bo5 scoreline distributions against a
synthetic league (pinned by closed-form corners and cross-checked
against ``utils.series_paths.enumerate_series_paths`` reindexed by
``series_outcome_order``), the ``p == 0.5`` pure-binomial degeneracy for
both the zero-games and zero-wins cases, ``as_tuple()`` parity, simplex
summation across three distinct team-strength combinations, a
leakage-safety regression (a future match must not change the
prediction), and a skip-guarded end-to-end sanity check at real
``data/v1`` scale that cross-checks the prediction's ``p_win_a``
against an independently recomputed feature call.
"""

import math

import pandas as pd
import pytest

from features import map_win_rate
from models.flat_series_baseline import (
    FlatSeriesPrediction,
    _parse_best_of,
    predict_series_outcome,
)
from tests._shared import _real_v1_available
from utils import asof, series_paths

# The as-of cutoff used by the synthetic fixtures: one hour after the
# last fixture match, so every fixture row is strictly before it.
QUERY_DATE = "2026-01-06T00:00:00"

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every test fixture produces the same
    column order/dtypes regardless of which subset of columns a given
    fixture actually needs.

    Args:
        rows: A list of dicts, one per match; each must carry the keys
            in :data:`_MATCHES_COLS` (extra keys are ignored by the
            explicit ``columns=`` ordering).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MATCHES_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_MATCHES_COLS)


def _maps_df(rows):
    """Build a maps table with the fixed M8 column set.

    Mirrors :func:`_matches_df` for the maps side so every fixture
    shares one column order/dtype.

    Args:
        rows: A list of dicts, one per map; each must carry the keys in
            :data:`_MAPS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MAPS_COLS` columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_MAPS_COLS)


def _build(match_rows, map_rows):
    """Build a ``(matches_df, maps_df)`` pair from parallel row lists.

    The shared constructor behind every synthetic league fixture so the
    column-order/dtype convention lives in one place. Each entry of
    ``match_rows``/``map_rows`` must carry the keys of
    :data:`_MATCHES_COLS`/:data:`_MAPS_COLS` respectively.

    Args:
        match_rows: A list of match dicts, one per completed match.
        map_rows: A list of map dicts, one per finished map.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df`
        / :func:`_maps_df`.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return _matches_df(match_rows), _maps_df(map_rows)


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    The single row-writing helper for the synthetic league fixtures.
    The map's ``winner`` is derived from the scores (never a
    display-name string), matching the existing test fixtures'
    convention; a null-score row is never produced because ``winner``
    is always set.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        mid: The shared ``match_id`` for the new match and map.
        date: The match's ISO date string.
        team1_id: The match's team1 stable id.
        team2_id: The match's team2 stable id.
        map_name: The finished map's name.
        t1s: Rounds team1 won (the map's ``team1_score``).
        t2s: Rounds team2 won (the map's ``team2_score``).

    Returns:
        Nothing (appends in place).

    Raises:
        Nothing.
    """
    match_rows.append(
        {
            "match_id": mid,
            "date": date,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "status": "completed",
        }
    )
    map_rows.append(
        {
            "match_id": mid,
            "map_index": 0,
            "map_name": map_name,
            "team1_score": t1s,
            "team2_score": t2s,
            "winner": team1_id if t1s > t2s else team2_id,
        }
    )


def _stamp(i):
    """Return the ISO timestamp ``i`` hours after a fixed 2026-01-01 base.

    The shared clock for the synthetic leagues: every fixture's dates
    are one hour apart so chronological ordering is unambiguous and the
    as-of query date (an hour after the last map) is strictly after
    everything.

    Args:
        i: The hour offset from the base.

    Returns:
        An ISO-8601 datetime string.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    return (base + pd.Timedelta(hours=i)).isoformat()


def _flat_league(team_a_record, team_b_record, filler_count=0):
    """Build a synthetic league with exactly two useful teams.

    Team ``A`` and team ``B`` each play their ``(wins, losses)`` record
    of finished maps against distinct filler opponents (never each
    other), so each team's overall as-of rate is exactly
    ``wins / (wins + losses)`` — the value :func:`_flat_league`
    consumers hand-derive — with no cross-contamination. An optional
    batch of regulation filler maps (unrelated teams) rounds out the
    league for tests that only need extra noise.

    Args:
        team_a_record: A ``(wins, losses)`` pair for team ``A``.
        team_b_record: A ``(wins, losses)`` pair for team ``B``.
        filler_count: How many extra filler matches (unrelated teams,
            regulation 13-8 scorelines) to append after A and B's
            matches; default 0.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_build` whose
        teams ``"A"`` and ``"B"`` have exactly the requested records.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    hour = 0
    for record, team in ((team_a_record, "A"), (team_b_record, "B")):
        wins, losses = record
        for _ in range(wins):
            _add(
                match_rows, map_rows, f"{team}-w{hour}", _stamp(hour),
                team, f"op{hour}", "Bind", 13, 8,
            )
            hour += 1
        for _ in range(losses):
            _add(
                match_rows, map_rows, f"{team}-l{hour}", _stamp(hour),
                team, f"op{hour}", "Bind", 8, 13,
            )
            hour += 1
    for i in range(filler_count):
        _add(
            match_rows, map_rows, f"f{i}", _stamp(hour),
            f"f{i}", f"g{i}", "Haven", 13, 8,
        )
        hour += 1
    return _build(match_rows, map_rows)


def _expected_in_order(p_win_a, best_of):
    """Return the exact expected scoreline vector for a flat baseline.

    Re-enumerates the terminal distribution via
    :func:`utils.series_paths.series_probabilities_in_order` with the
    single per-map probability applied to every map — the same code
    path ``predict_series_outcome`` uses, so the test asserts the
    wiring (parsing, normalization, vector construction) without
    re-deriving the enumeration's internals; closed-form corners are
    pinned separately in the hand-computed tests.

    Args:
        p_win_a: The single per-map probability applied to every map.
        best_of: The map count (a positive odd int).

    Returns:
        A ``list`` of ``best_of + 1`` floats in
        :func:`utils.series_paths.series_outcome_order` order.

    Raises:
        Nothing (delegates validation to ``series_paths``).
    """
    return series_paths.series_probabilities_in_order(
        [p_win_a] * best_of, best_of
    )


# --------------------------------------------------------------------------
# plan#4a: _parse_best_of exact mappings and ValueError matrix
# --------------------------------------------------------------------------


def test_parse_best_of_exact_mappings():
    # The three values observed in data/v1 (and Bo1, which the real
    # dataset lacks but the parser must still handle) convert to the
    # plain odd ints utils.series_paths expects.
    assert _parse_best_of("Bo1") == 1
    assert _parse_best_of("Bo3") == 3
    assert _parse_best_of("Bo5") == 5
    # Any other "Bo<N>" with a positive odd N is accepted too.
    assert _parse_best_of("Bo7") == 7
    assert _parse_best_of("Bo9") == 9


def test_parse_best_of_rejects_malformed():
    # Every malformed *string* must raise ValueError, never silently
    # coerce: even map counts, non-numeric suffixes, empty strings,
    # unrelated strings, and a non-positive count. Non-string inputs
    # violate the annotated str contract and raise TypeError instead.
    for bad in (
        "Bo2", "Bo4", "BestOf3", "bo3", "BO3", "Bo", "BoX", "",
        "3", "Bo0", "Bo-1", "Bo3 ",
    ):
        with pytest.raises(ValueError):
            _parse_best_of(bad)
    for bad in (3, None):
        with pytest.raises(TypeError):
            _parse_best_of(bad)


# --------------------------------------------------------------------------
# plan#4b: hand-computed Bo3/Bo5 distributions
# --------------------------------------------------------------------------


def test_bo3_hand_computed_distribution():
    # Team A goes 3W-1L (rate 0.75), team B goes 1W-3L (rate 0.25), so
    # p_win_a = 0.75/(0.75+0.25) = 0.75 and the Bo3 vector is the
    # closed form at p = 3/4 in series_outcome_order:
    #   P(2,0) = p^2 = 9/16, P(2,1) = 2p^2(1-p) = 9/32,
    #   P(1,2) = 2p(1-p)^2 = 3/32, P(0,2) = (1-p)^2 = 1/16.
    matches_df, maps_df = _flat_league((3, 1), (1, 3))
    pred = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)
    assert pred.overall_a.rate == 0.75
    assert pred.overall_b.rate == 0.25
    assert pred.p_win_a == pytest.approx(0.75)
    assert pred.best_of == 3
    assert pred.outcome_order == ((2, 0), (2, 1), (1, 2), (0, 2))
    assert pred.probabilities == pytest.approx((0.5625, 0.28125, 0.09375, 0.0625))
    assert pred.probabilities == pytest.approx(
        tuple(_expected_in_order(0.75, 3))
    )
    assert sum(pred.probabilities) == pytest.approx(1.0)


def test_bo5_hand_computed_distribution():
    # Same 3W-1L vs 1W-3L league, but the series spans five maps: the
    # closed form at p = 3/4 in series_outcome_order is
    #   P(3,0) = p^3, P(3,1) = 3p^3(1-p), P(3,2) = 6p^3(1-p)^2,
    #   P(2,3) = 6p^2(1-p)^3, P(1,3) = 3p(1-p)^3, P(0,3) = (1-p)^3,
    # i.e. (27/64, 81/256, 405/2560, ...) pinned to tight tolerance.
    matches_df, maps_df = _flat_league((3, 1), (1, 3))
    pred = predict_series_outcome("A", "B", "Bo5", QUERY_DATE, matches_df, maps_df)
    assert pred.best_of == 5
    assert pred.outcome_order == (
        (3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3),
    )
    assert pred.probabilities == pytest.approx(
        (0.421875, 0.31640625, 0.158203125, 0.052734375, 0.03515625, 0.015625)
    )
    assert pred.probabilities == pytest.approx(
        tuple(_expected_in_order(0.75, 5))
    )
    assert sum(pred.probabilities) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#4c: p = 0.5 degeneracy (pure binomial) in both zero-rate cases
# --------------------------------------------------------------------------


def _assert_binomial_at_half(pred, best_of):
    """Assert a prediction equals the pure-binomial p = 0.5 closed form.

    Shared by the zero-games and zero-wins cases: at ``p_win_a == 0.5``
    every scoreline probability is a half-coin binomial, so Bo3 is
    ``(1/4, 1/4, 1/4, 1/4)`` and Bo5 is
    ``(1/8, 3/16, 3/16, 3/16, 3/16, 1/8)``, and the whole vector is a
    valid simplex in ordinal order.

    Args:
        pred: The :class:`FlatSeriesPrediction` under test.
        best_of: The expected parsed map count (3 or 5).

    Returns:
        Nothing (asserts in place).

    Raises:
        Nothing.
    """
    assert pred.best_of == best_of
    assert pred.p_win_a == 0.5
    if best_of == 3:
        expected = (0.25, 0.25, 0.25, 0.25)
    else:
        expected = (0.125, 0.1875, 0.1875, 0.1875, 0.1875, 0.125)
    assert pred.probabilities == pytest.approx(expected)
    assert sum(pred.probabilities) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in pred.probabilities)


def test_zero_games_both_sides_is_binomial_at_half():
    # Neither team has played a single map before the cutoff: both
    # OverallWinRate.rate values are exactly 0.5 (the estimator's
    # "no evidence" stand-in), so the normalization yields p_win_a =
    # 0.5/(0.5+0.5) = 0.5 naturally and every scoreline probability is
    # the pure-binomial closed form at p = 0.5.
    matches_df, maps_df = _build([], [])
    for bo, expected_bo in (("Bo3", 3), ("Bo5", 5)):
        pred = predict_series_outcome(
            "UNSEEN_A", "UNSEEN_B", bo, "2026-01-01T00:00:00", matches_df, maps_df
        )
        _assert_binomial_at_half(pred, expected_bo)
        assert pred.overall_a.games == 0 and pred.overall_b.games == 0
        assert pred.overall_a.rate == 0.5 and pred.overall_b.rate == 0.5


def test_zero_wins_both_sides_falls_back_to_half():
    # Both teams have games but zero wins (every map lost 8-13), so
    # both rates are exactly 0.0 and rate_a + rate_b == 0.0: the
    # normalization is undefined and must fall back to p_win_a = 0.5
    # instead of dividing by zero — the same both-zero-means fallback
    # M18's four-way baseline established.
    matches_df, maps_df = _flat_league((0, 2), (0, 2))
    pred3 = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)
    assert pred3.overall_a.rate == 0.0 and pred3.overall_b.rate == 0.0
    _assert_binomial_at_half(pred3, 3)
    pred5 = predict_series_outcome("A", "B", "Bo5", QUERY_DATE, matches_df, maps_df)
    _assert_binomial_at_half(pred5, 5)
    assert all(math.isfinite(p) for p in pred3.probabilities)
    assert all(math.isfinite(p) for p in pred5.probabilities)


# --------------------------------------------------------------------------
# plan#4d: as_tuple parity and simplex summation across strength combos
# --------------------------------------------------------------------------


def test_as_tuple_returns_probabilities_unchanged():
    # as_tuple() exists for parity with FourWayPrediction.as_tuple and
    # must return exactly the stored probability tuple (the SeriesModelFn
    # vector), not a copy-with-a-twist.
    matches_df, maps_df = _flat_league((3, 1), (1, 3))
    pred = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)
    assert pred.as_tuple() is pred.probabilities
    assert pred.as_tuple() == pred.probabilities
    assert isinstance(pred, FlatSeriesPrediction)


def test_simplex_sums_to_one_for_three_strength_combos():
    # Three genuinely distinct (rate_a, rate_b) combinations — 0.75/0.25
    # (p = 0.75), 0.4/0.8 (p = 1/3, not a mirror of the first) and
    # 0.0/0.0 (the fallback) — must all produce a valid unit simplex in
    # ordinal order whose entries are all finite and whose p_win_a
    # matches the hand-derived normalization.
    combos = (
        ((3, 1), (1, 3), 0.75),
        ((2, 3), (4, 1), 1 / 3),
        ((0, 2), (0, 2), 0.5),
    )
    for record_a, record_b, expected_p in combos:
        matches_df, maps_df = _flat_league(record_a, record_b)
        pred = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)
        assert pred.p_win_a == pytest.approx(expected_p)
        assert sum(pred.probabilities) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in pred.probabilities)
        assert all(math.isfinite(p) for p in pred.probabilities)
        assert pred.probabilities == pytest.approx(
            tuple(_expected_in_order(pred.p_win_a, 3))
        )


def test_team_swap_flips_scoreline_orientation():
    # Predicting with team1/team2 swapped must swap the A-side
    # scorelines with the mirror B-side ones (same crude non-head-to-head
    # symmetry the M18 baseline documents): p_win_a maps to 1 - p_win_a
    # and each (a, b) probability moves to (b, a), so the reversed
    # vector equals the original.
    matches_df, maps_df = _flat_league((3, 1), (1, 3))
    ab = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)
    ba = predict_series_outcome("B", "A", "Bo3", QUERY_DATE, matches_df, maps_df)
    assert ba.p_win_a == pytest.approx(1 - ab.p_win_a)
    assert ba.probabilities == pytest.approx(tuple(reversed(ab.probabilities)))


# --------------------------------------------------------------------------
# plan#4e: leakage-safety regression
# --------------------------------------------------------------------------


def test_future_match_does_not_change_prediction():
    # A match dated strictly after the query cutoff must not influence
    # the prediction: both teams' as-of histories end at the cutoff, so
    # two leagues identical except for one extra future match produce
    # byte-identical predictions (the as-of layer's strict < boundary
    # is the guarantee under test, exercised through the real feature
    # estimator rather than stubbed).
    matches_df, maps_df = _flat_league((3, 1), (1, 3))
    before = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, matches_df, maps_df)

    # Rebuild the identical league plus one extra match dated after the
    # query cutoff; the prediction must not change.
    match_rows = [dict(r._asdict()) for r in matches_df.itertuples(index=False)]
    map_rows = [dict(r._asdict()) for r in maps_df.itertuples(index=False)]
    _add(
        match_rows, map_rows, "future-1", "2026-01-07T00:00:00",
        "A", "B", "Bind", 13, 8,
    )
    future_matches, future_maps = _build(match_rows, map_rows)
    after = predict_series_outcome("A", "B", "Bo3", QUERY_DATE, future_matches, future_maps)
    assert after.probabilities == before.probabilities
    assert after.p_win_a == before.p_win_a
    assert after.overall_a == before.overall_a
    assert after.overall_b == before.overall_b


# --------------------------------------------------------------------------
# plan#4f: real v1 end-to-end (mirrors test_four_way_baseline.py)
# --------------------------------------------------------------------------


def _real_v1_series_target():
    """Pick a real finished v1 match whose teams both have prior history.

    Scans ``data/v1`` matches from the latest date backwards and returns
    the first completed match that (a) has at least one finished map
    (both scores present, ``winner`` non-null) and (b) has both teams
    holding at least one strictly-earlier finished map (so the as-of
    prediction at the match's own date is not the degenerate empty
    history case). Mirrors ``test_four_way_baseline._real_v1_target``
    but keeps the whole match as the series target.

    Returns:
        A 6-tuple ``(matches_df, maps_df, team1_id, team2_id,
        best_of, date)`` where ``date`` is the target match's own
        timestamp and ``best_of`` is the match's ``"Bo<N>"`` string.

    Raises:
        AssertionError: If no suitable match exists in the loaded
            ``data/v1`` tables (both teams with prior history).
    """
    matches_df, maps_df = asof.load_asof_tables("v1")
    finished = maps_df[
        maps_df["winner"].notna()
        & maps_df["team1_score"].notna()
        & maps_df["team2_score"].notna()
    ]
    finished_ids = set(finished["match_id"])
    for row in matches_df.sort_values("date", ascending=False).itertuples(index=False):
        if row.match_id not in finished_ids:
            continue
        date = row.date
        if len(asof.maps_as_of(row.team1_id, date, matches_df, maps_df)) == 0:
            continue
        if len(asof.maps_as_of(row.team2_id, date, matches_df, maps_df)) == 0:
            continue
        return (
            matches_df,
            maps_df,
            row.team1_id,
            row.team2_id,
            row.best_of,
            date,
        )
    raise AssertionError(
        "no real v1 match has a finished map with both teams holding "
        "prior history; data/v1 is unexpectedly small"
    )


@pytest.mark.skipif(
    not _real_v1_available(("matches", "maps")),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_simplex_and_cross_checks_feature_calls():
    # plan#4f: predict as of a real finished match's own date (both
    # teams with prior history) and cross-check every output field
    # against independently recomputed feature calls at the same
    # cutoff: the scoreline vector is a valid simplex in ordinal order,
    # as_tuple() equals probabilities, and p_win_a matches
    # rate_a/(rate_a+rate_b) computed from fresh team_overall_win_rate
    # calls.
    matches_df, maps_df, t1, t2, best_of, date = _real_v1_series_target()
    pred = predict_series_outcome(t1, t2, best_of, date, matches_df, maps_df)
    parsed = int(best_of[2:])
    assert pred.best_of == parsed
    assert pred.outcome_order == series_paths.series_outcome_order(parsed)
    assert len(pred.probabilities) == parsed + 1
    assert sum(pred.probabilities) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in pred.probabilities)
    assert pred.as_tuple() == pred.probabilities
    a = map_win_rate.team_overall_win_rate(t1, date, matches_df, maps_df)
    b = map_win_rate.team_overall_win_rate(t2, date, matches_df, maps_df)
    exp_win = a.rate / (a.rate + b.rate) if a.rate + b.rate != 0.0 else 0.5
    assert pred.p_win_a == pytest.approx(exp_win)
    assert pred.overall_a == a and pred.overall_b == b

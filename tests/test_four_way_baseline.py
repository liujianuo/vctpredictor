"""Tests for the four-way baseline model (M18).

Covers the unit-simplex/scoring-usability contract, the normalization
arithmetic (including the both-zero-means fallback), the symmetric OT
split's edge cases (``p_ot == 0`` and empty league-wide history), the
team-swap symmetry regression, and a skip-guarded end-to-end sanity
check at real ``data/v1`` scale that cross-checks the prediction's
``p_win_a``/``p_ot`` against independently recomputed feature calls.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from features import closeness, map_win_rate
from models.four_way_baseline import OUTCOME_LABELS, predict_map_outcome
from utils import asof, scoring

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


def _league_tables():
    """Build the 12-map arithmetic/symmetry league.

    Team ``A`` plays three Haven maps (2 wins, 1 loss, all OT scorelines
    14-12/12-14/14-12); team ``B`` plays three Haven maps (1 win, 2
    losses, all OT). Six filler matches (unique teams, regulation Bind
    13-8) complete the league-wide pool at 12 maps with exactly 6 OT
    maps, so ``global_ot_rate`` is exactly ``0.5``. Hand-derived
    expectation: ``mean_a = 2/3`` (prior 2/3, full shrinkage to it
    because the overall history is the same three maps),
    ``mean_b = 1/3``, ``p_win_a = 2/3``, ``p_ot = 0.5``, and the four
    probabilities ``(1/3, 1/3, 1/6, 1/6)``.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 12 matches and 12 maps
        built by :func:`_build`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", _stamp(0), "A", "op1", "Haven", 14, 12)
    _add(match_rows, map_rows, "m2", _stamp(1), "A", "op2", "Haven", 12, 14)
    _add(match_rows, map_rows, "m3", _stamp(2), "A", "op3", "Haven", 14, 12)
    _add(match_rows, map_rows, "m4", _stamp(3), "B", "op4", "Haven", 14, 12)
    _add(match_rows, map_rows, "m5", _stamp(4), "B", "op5", "Haven", 12, 14)
    _add(match_rows, map_rows, "m6", _stamp(5), "B", "op6", "Haven", 12, 14)
    for i in range(6):
        _add(match_rows, map_rows, f"f{i}", _stamp(6 + i), f"f{i}", f"g{i}", "Bind", 13, 8)
    return _build(match_rows, map_rows)


def _degenerate_tables():
    """Build the both-zero-means fallback league.

    Teams ``A`` and ``B`` each lose every map they play (Bind, 8-13), so
    each has overall ``prior == 0.0`` and zero games on the queried
    map (Haven): both ``ShrunkWinRate.mean`` values are exactly ``0.0``
    and the normalization would divide by zero without the ``0.5``
    fallback. One filler OT map (Haven 14-12) makes the league-wide OT
    pool nonzero: 5 maps with 1 OT, so ``p_ot = 0.2`` and the expected
    output is ``(0.4, 0.1, 0.1, 0.4)``.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 5 matches and 5 maps built
        by :func:`_build`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", _stamp(0), "A", "o1", "Bind", 8, 13)
    _add(match_rows, map_rows, "m2", _stamp(1), "A", "o2", "Bind", 8, 13)
    _add(match_rows, map_rows, "m3", _stamp(2), "B", "o3", "Bind", 8, 13)
    _add(match_rows, map_rows, "m4", _stamp(3), "B", "o4", "Bind", 8, 13)
    _add(match_rows, map_rows, "m5", _stamp(4), "f1", "f2", "Haven", 14, 12)
    return _build(match_rows, map_rows)


def _regulation_only_tables():
    """Build the zero-global-OT-rate league.

    Four regulation scorelines (13-8 / 8-13, all ``min(score) == 8``),
    so the league-wide OT pool has zero OT maps and
    ``global_ot_rate`` is exactly ``0.0``. Team ``A`` is 1W-1L on
    Haven (prior 0.5, ``mean_a = 0.5``); team ``B`` is 2W-0L on Haven
    (prior 1.0, ``mean_b = 1.0``), so ``p_win_a = 0.5/1.5 = 1/3`` and
    the expected output is ``(1/3, 0, 0, 2/3)`` — all probability mass
    in the two regulation categories.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 4 matches and 4 maps built
        by :func:`_build`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", _stamp(0), "A", "o1", "Haven", 13, 8)
    _add(match_rows, map_rows, "m2", _stamp(1), "A", "o2", "Haven", 8, 13)
    _add(match_rows, map_rows, "m3", _stamp(2), "B", "o3", "Haven", 13, 8)
    _add(match_rows, map_rows, "m4", _stamp(3), "B", "o4", "Haven", 13, 8)
    return _build(match_rows, map_rows)


def _real_v1_target():
    """Pick a real finished v1 match whose teams both have prior history.

    Scans ``data/v1`` matches from the latest date backwards and returns
    the first completed match that (a) has at least one finished map
    (both scores present, ``winner`` non-null) and (b) has both teams
    holding at least one strictly-earlier finished map (so the as-of
    prediction at the match's own date is not the degenerate empty
    history case). The match's own first finished map is the prediction
    target.

    Returns:
        A 8-tuple ``(matches_df, maps_df, team1_id, team2_id, map_name,
        date, team1_score, team2_score)`` where ``date`` is the target
        match's own timestamp and the two scores are the target map's.

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
        target = finished[finished["match_id"] == row.match_id].iloc[0]
        return (
            matches_df,
            maps_df,
            row.team1_id,
            row.team2_id,
            target["map_name"],
            date,
            int(target["team1_score"]),
            int(target["team2_score"]),
        )
    raise AssertionError(
        "no real v1 match has a finished map with both teams holding "
        "prior history; data/v1 is unexpectedly small"
    )


# --------------------------------------------------------------------------
# plan#4b: arithmetic correctness (stubbed feature calls)
# --------------------------------------------------------------------------


class _StubShrunk:
    """Minimal stand-in for features.map_win_rate.ShrunkWinRate.

    Only the ``mean`` attribute is needed by the arithmetic under test;
    the real dataclass's other fields are irrelevant to the four-way
    formulas.

    Args:
        mean: The stub's ``mean`` value.

    Returns:
        Nothing (attribute holder).

    Raises:
        Nothing.
    """

    def __init__(self, mean):
        self.mean = mean


def test_arithmetic_matches_hand_computed_formulas(monkeypatch):
    # With stubbed means mean_a=0.6, mean_b=0.3 and a stubbed global OT
    # rate of 0.2, the formulas give p_win_a = 0.6/0.9 = 2/3 and the
    # four categories (8/15, 2/15, 1/15, 4/15), summing to 1. This pins
    # the arithmetic (plan#4b) independently of the feature estimators.
    def fake_team(team_id, _map_name, _date, _matches_df, _maps_df, _k):
        return _StubShrunk({"A": 0.6, "B": 0.3}[team_id])

    class _StubRate:
        rate = 0.2

    monkeypatch.setattr(map_win_rate, "team_map_win_rate", fake_team)
    monkeypatch.setattr(
        closeness, "global_ot_rate", lambda _d, _m, _p: _StubRate()
    )
    pred = predict_map_outcome(
        "A", "B", "Haven", QUERY_DATE, _matches_df([]), _maps_df([])
    )
    assert pred.p_win_a == pytest.approx(2 / 3)
    assert pred.p_ot == pytest.approx(0.2)
    assert pred.p_a_regulation == pytest.approx(2 / 3 * 0.8)
    assert pred.p_a_ot == pytest.approx(2 / 3 * 0.2)
    assert pred.p_b_ot == pytest.approx(1 / 3 * 0.2)
    assert pred.p_b_regulation == pytest.approx(1 / 3 * 0.8)
    assert sum(pred.as_tuple()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#4c / 4d / 4e: edge cases through real as-of data
# --------------------------------------------------------------------------


def test_both_zero_means_falls_back_to_half():
    # Both teams have prior == 0.0 (they lost everything) and zero games
    # on the queried map, so both ShrunkWinRate.mean values are exactly
    # 0.0 and mean_a + mean_b == 0.0: the normalization must fall back
    # to p_win_a = 0.5 instead of dividing by zero. The league's one OT
    # filler map makes p_ot = 1/5 = 0.2, so the expected output is
    # (0.4, 0.1, 0.1, 0.4) — exercising the fallback composed with a
    # nonzero OT split.
    matches_df, maps_df = _degenerate_tables()
    pred = predict_map_outcome("A", "B", "Haven", QUERY_DATE, matches_df, maps_df)
    assert pred.shrunk_a.mean == 0.0
    assert pred.shrunk_b.mean == 0.0
    assert pred.p_win_a == 0.5
    assert pred.p_ot == pytest.approx(0.2)
    assert pred.p_a_regulation == pytest.approx(0.4)
    assert pred.p_a_ot == pytest.approx(0.1)
    assert pred.p_b_ot == pytest.approx(0.1)
    assert pred.p_b_regulation == pytest.approx(0.4)
    # No NaN anywhere in the output.
    assert all(math.isfinite(p) for p in pred.as_tuple())
    assert math.isfinite(pred.p_win_a) and math.isfinite(pred.p_ot)


def test_zero_global_ot_rate_puts_all_mass_in_regulation():
    # No OT map has occurred before the cutoff, so global_ot_rate is
    # exactly 0.0: the two OT categories must be exactly 0.0 and all
    # probability mass sits in A-regulation/B-regulation (1/3 and 2/3
    # respectively from means 0.5 and 1.0).
    matches_df, maps_df = _regulation_only_tables()
    pred = predict_map_outcome("A", "B", "Haven", QUERY_DATE, matches_df, maps_df)
    assert pred.p_ot == 0.0
    assert pred.p_a_ot == 0.0
    assert pred.p_b_ot == 0.0
    assert pred.p_a_regulation == pytest.approx(pred.p_win_a)
    assert pred.p_b_regulation == pytest.approx(1 - pred.p_win_a)
    assert pred.p_a_regulation == pytest.approx(1 / 3)
    assert pred.p_b_regulation == pytest.approx(2 / 3)
    assert sum(pred.as_tuple()) == pytest.approx(1.0)


def test_empty_history_for_both_teams_is_valid_simplex():
    # An as-of cutoff before either team's first match (both teams
    # unseen): each ShrunkWinRate.mean degrades to its 0.5 prior, the
    # league-wide OT pool is empty (rate 0.0), and the output is the
    # non-NaN simplex (0.5, 0, 0, 0.5) rather than a raise.
    matches_df, maps_df = _build([], [])
    pred = predict_map_outcome(
        "UNSEEN_A", "UNSEEN_B", "Haven", "2026-01-01T00:00:00", matches_df, maps_df
    )
    assert pred.p_win_a == 0.5
    assert pred.p_ot == 0.0
    assert pred.as_tuple() == pytest.approx((0.5, 0.0, 0.0, 0.5))
    assert all(math.isfinite(p) for p in pred.as_tuple())
    assert sum(pred.as_tuple()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#4g: symmetry regression
# --------------------------------------------------------------------------


def test_swapping_teams_swaps_a_b_probabilities():
    # Under the crude non-head-to-head normalization, swapping team1/
    # team2 must swap the A-side probabilities with the B-side ones:
    # predict(B, A, ...) == swap of predict(A, B, ...). Asserted with
    # pytest.approx because the complementary (1 - p_win_a) term is a
    # separate float subtraction that differs from mean_b/(mean_a+mean_b)
    # in the last ulp (~1e-16); a future head-to-head-aware model would
    # break this symmetry by orders of magnitude more and fail here.
    matches_df, maps_df = _league_tables()
    ab = predict_map_outcome("A", "B", "Haven", QUERY_DATE, matches_df, maps_df)
    ba = predict_map_outcome("B", "A", "Haven", QUERY_DATE, matches_df, maps_df)
    assert ba.p_a_regulation == pytest.approx(ab.p_b_regulation)
    assert ba.p_a_ot == pytest.approx(ab.p_b_ot)
    assert ba.p_b_ot == pytest.approx(ab.p_a_ot)
    assert ba.p_b_regulation == pytest.approx(ab.p_a_regulation)


# --------------------------------------------------------------------------
# plan#4a / 4f: real v1 end-to-end
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the materialised v1 tables exist on disk.

    The skip guard for the real-data tests, matching the convention in
    ``test_map_win_rate.py`` / ``test_closeness.py``: both Parquet
    files must exist (i.e. ``materialize.py`` has been run).

    Returns:
        A bool: ``True`` iff ``data/v1/matches.parquet`` and
        ``data/v1/maps.parquet`` both exist.

    Raises:
        Nothing.
    """
    return Path("data/v1/matches.parquet").exists() and Path(
        "data/v1/maps.parquet"
    ).exists()


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_simplex_and_scoring_usability():
    # plan#4a: as of a real v1 match's own date (match 731406,
    # 2026-08-23T12:15:00, team1 397 vs team2 20085 on Abyss), the
    # predicted tuple is a valid unit simplex and plugs directly into
    # utils.scoring.rps/log_loss against the target map's own true
    # ordinal (3 = B-regulation for the 11-13 scoreline) without
    # raising.
    matches_df, maps_df, t1, t2, map_name, date, s1, s2 = _real_v1_target()
    pred = predict_map_outcome(t1, t2, map_name, date, matches_df, maps_df)
    probs = pred.as_tuple()
    assert sum(probs) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in probs)
    # True ordinal from the target map's own scores (drivers.labels'
    # A/B convention: A = team1).
    if s1 > s2:
        ordinal = 1 if min(s1, s2) >= 12 else 0
    else:
        ordinal = 2 if min(s1, s2) >= 12 else 3
    assert 0 <= ordinal < 4
    assert OUTCOME_LABELS[ordinal] in OUTCOME_LABELS
    assert math.isfinite(scoring.rps(probs, ordinal))
    assert math.isfinite(scoring.log_loss(probs, ordinal))


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_end_to_end_cross_checks_feature_calls():
    # plan#4f: predict as of a real finished match's own date (match
    # 731406, 2026-08-23T12:15:00, team1 397 vs team2 20085 on Abyss)
    # and cross-check every output field against independently
    # recomputed feature calls at the same cutoff:
    #   mean_a = 0.4838709677419355, mean_b = 0.606060606060606,
    #   global OT rate = 0.11983471074380166 (29 OT / 242 maps),
    #   p_win_a = 0.44394618834080724,
    #   tuple = (0.39074602527517327, 0.053200163065633924,
    #            0.06663454767816773, 0.48941926398102503).
    matches_df, maps_df, t1, t2, map_name, date, _s1, _s2 = _real_v1_target()
    pred = predict_map_outcome(t1, t2, map_name, date, matches_df, maps_df)
    a = map_win_rate.team_map_win_rate(
        t1, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    )
    b = map_win_rate.team_map_win_rate(
        t2, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    )
    glob = closeness.global_ot_rate(date, matches_df, maps_df)
    exp_win = (
        a.mean / (a.mean + b.mean) if a.mean + b.mean != 0.0 else 0.5
    )
    assert pred.p_win_a == pytest.approx(exp_win)
    assert pred.p_ot == pytest.approx(glob.rate)
    assert pred.p_a_regulation == pytest.approx(exp_win * (1 - glob.rate))
    assert pred.p_a_ot == pytest.approx(exp_win * glob.rate)
    assert pred.p_b_ot == pytest.approx((1 - exp_win) * glob.rate)
    assert pred.p_b_regulation == pytest.approx((1 - exp_win) * (1 - glob.rate))

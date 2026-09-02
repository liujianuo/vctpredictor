"""Tests for signed round-margin strength (M38.3).

Covers the two estimators (``team_overall_signed_margin`` /
``team_map_signed_margin``) and their plain linear shrunk-mean
dataclasses (no Beta posterior / variance — signed margin is continuous
and signed, not a rate), the symmetric-zero identity (every map's two
seats' derived ``signed_margin`` cancel exactly, and the inner prior is
the literal :data:`LEAGUE_MEAN_SIGNED_MARGIN` constant, not a pool
statistic), the shrinkage-behaviour contracts (zero history -> full
shrinkage, few maps -> pulled toward the prior, many maps -> close to
raw, monotonic in ``k``), the two-level hierarchy's distinguishing
behaviour (the map-level ``prior`` is the *shrunk overall* mean margin
from ``team_overall_signed_margin``, not ``0.0`` and not the raw
team-overall mean), the seat-resolution correctness (a team queried from
both the ``team1`` and the ``team2`` seat gets the correctly-signed
margins — the test that would catch a sign-flip bug), map-name
normalization, the leakage-safety proof (maps dated at/after the cutoff
never enter any estimator), invalid-``k`` rejection from both public
functions, and a skip-guarded smoke test recording the real ``data/v1``
numbers. Test fixtures build internally-consistent regulation map rows
(the opposing-side-pairing and case-split invariants of
``features.round_detail`` hold by construction); a non-test helper raises
``ValueError`` on an invalid row so a mis-typed fixture fails at
construction, never silently mid-test.
"""

from pathlib import Path

import pandas as pd
import pytest

from features import round_detail as rd
from features import signed_margin as sm
from utils import asof

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    rd.MATCH_ID_COL,
    rd.MAP_INDEX_COL,
    "map_name",
    rd.TEAM1_SCORE_COL,
    rd.TEAM2_SCORE_COL,
    "winner",
    rd.TEAM1_ATK_COL,
    rd.TEAM1_DEF_COL,
    rd.TEAM2_ATK_COL,
    rd.TEAM2_DEF_COL,
    rd.TEAM1_FIRST_HALF_COL,
    rd.TEAM1_SECOND_HALF_COL,
    rd.TEAM2_FIRST_HALF_COL,
    rd.TEAM2_SECOND_HALF_COL,
]

# The four half-split columns are present and non-null (0.0) on every
# fixture row: round_detail reads them only for null-row identification,
# never for derivation, so a non-null dummy keeps a constructed row usable.
_HALF_COLS = {
    rd.TEAM1_FIRST_HALF_COL: 0.0,
    rd.TEAM1_SECOND_HALF_COL: 0.0,
    rd.TEAM2_FIRST_HALF_COL: 0.0,
    rd.TEAM2_SECOND_HALF_COL: 0.0,
}


def _stamp(index):
    """Return the ISO timestamp of fixture position ``index``.

    Every mass-produced fixture places its matches one hour apart from a
    fixed 2026 base date, so chronological order == list order and the
    query date for "after everything" is trivially derivable.

    Args:
        index: The 0-based fixture position.

    Returns:
        An ISO-8601 timestamp string one hour after the previous
        position (``2026-01-01T00:00:00`` + ``index`` hours).

    Raises:
        Nothing.
    """
    return (pd.Timestamp("2026-01-01T00:00:00") + pd.Timedelta(hours=index)).isoformat()


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every test fixture produces the same column
    order regardless of which subset of columns a fixture needs.

    Args:
        rows: A list of dicts, one per match; each must carry the keys
            in :data:`_MATCHES_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MATCHES_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_MATCHES_COLS)


def _maps_df(rows):
    """Build a maps-shaped table with the fixed column set.

    Mirrors :func:`_matches_df` for the maps side (maps rows carry the
    ``winner``/``map_name`` columns the as-of layer and map filter read
    plus the full ``round_detail`` required column set).

    Args:
        rows: A list of dicts, one per map; each must carry the keys in
            :data:`_MAPS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MAPS_COLS` columns.

    Raises:
        Nothing (pandas ``ValueError`` on a malformed dict surfaces as
        is).
    """
    return pd.DataFrame(rows, columns=_MAPS_COLS)


def _regulation_map(
    match_id,
    index,
    map_name,
    team1_id,
    team2_id,
    t1_score,
    t2_score,
    t1_atk,
    t1_def,
    t2_atk,
    t2_def,
    winner,
):
    """Build one internally-valid regulation map fixture row.

    Constructs a maps-shaped row dict (round columns plus the match-side
    columns ``asof`` reads) whose raw atk/def round wins satisfy
    ``round_detail``'s validation invariants by construction: per-side
    ``atk + def == score``, each opposing-side pairing
    (``t1_atk + t2_def`` / ``t1_def + t2_atk``) inside ``[0, 12]`` with
    the two summing to the map total, and a finished regulation scoreline
    (winner on 13, loser on at most 11 — OT maps are out of scope for
    these fixtures; every constructed signed margin is therefore in
    ``[-13, -2]`` / ``[+2, +13]``). The four half-split columns default
    to non-null ``0.0`` (unused by derivation, see :data:`_HALF_COLS`).
    An invalid combination raises immediately so a mis-typed fixture
    never reaches the module.

    Args:
        match_id: The fixture match id.
        index: The 0-based map index within the match.
        map_name: The map name (title-cased, e.g. ``"Haven"``).
        team1_id: The team1 side's stable id.
        team2_id: The team2 side's stable id.
        t1_score: Rounds won by team1 on the finished map.
        t2_score: Rounds won by team2 on the finished map.
        t1_atk: Team1's regulation attack-round wins.
        t1_def: Team1's regulation defence-round wins.
        t2_atk: Team2's regulation attack-round wins.
        t2_def: Team2's regulation defence-round wins.
        winner: The finished-map winner marker (non-null; the as-of
            layer only checks presence).

    Returns:
        A dict keyed by the maps/match columns of :data:`_MAPS_COLS` plus
        the match-side columns, one map row.

    Raises:
        ValueError: If the per-side sums, the pairing bounds/partition,
            or the finished-regulation scoreline constraint do not hold.
    """
    if t1_atk + t1_def != t1_score:
        raise ValueError(
            f"map {match_id} team1 atk+def {t1_atk}+{t1_def} != score {t1_score}"
        )
    if t2_atk + t2_def != t2_score:
        raise ValueError(
            f"map {match_id} team2 atk+def {t2_atk}+{t2_def} != score {t2_score}"
        )
    pairing1 = t1_atk + t2_def
    pairing2 = t1_def + t2_atk
    if not (0 <= pairing1 <= 12 and 0 <= pairing2 <= 12):
        raise ValueError(
            f"map {match_id} opposing-side pairings {pairing1}/{pairing2} "
            "must each lie in [0, 12]"
        )
    if pairing1 + pairing2 != t1_score + t2_score:
        raise ValueError(
            f"map {match_id} pairings {pairing1}+{pairing2} do not "
            "partition the map's rounds"
        )
    if max(t1_score, t2_score) != 13 or min(t1_score, t2_score) > 11:
        raise ValueError(
            f"map {match_id} scoreline {t1_score}-{t2_score} is not a "
            "finished regulation result (winner must reach 13, loser at "
            "most 11)"
        )
    row = {
        rd.MATCH_ID_COL: match_id,
        rd.MAP_INDEX_COL: index,
        "map_name": map_name,
        "team1_id": team1_id,
        "team2_id": team2_id,
        rd.TEAM1_SCORE_COL: t1_score,
        rd.TEAM2_SCORE_COL: t2_score,
        "winner": winner,
        rd.TEAM1_ATK_COL: t1_atk,
        rd.TEAM1_DEF_COL: t1_def,
        rd.TEAM2_ATK_COL: t2_atk,
        rd.TEAM2_DEF_COL: t2_def,
    }
    row.update(_HALF_COLS)
    return row


def _opposing_side_wins(loser_score):
    """Distribute a loser's ``score`` into valid atk/def round wins.

    The complementary-side split used by the margin builders. Given a
    winning team1 fixed at 7 attack-round wins and 6 defence-round wins
    (a valid partition of its 13), the loser's ``score`` (0-11) must be
    split into its own attack/defence wins so that the opposing-side
    pairing invariants of ``round_detail`` hold: pairing1
    (``7 + loser_def``) and pairing2 (``6 + loser_atk``) must each stay
    inside ``[0, 12]``. The returned split
    ``loser_atk = clamp(score - 5, 0, 6)`` / ``loser_def = score -
    loser_atk`` is the unique largest-attack split satisfying both
    bounds (``loser_def <= 5`` from pairing1, ``loser_atk <= 6`` from
    pairing2, and ``loser_atk >= score - 5`` from pairing2's upper
    bound).

    Args:
        loser_score: The losing side's regulation rounds (0-11).

    Returns:
        A ``(loser_atk, loser_def)`` tuple of ints summing to
        ``loser_score``, each non-negative.

    Raises:
        ValueError: If ``loser_score`` lies outside ``[0, 11]``.
    """
    if not 0 <= loser_score <= 11:
        raise ValueError(
            f"loser_score {loser_score} must lie in [0, 11] for a "
            "finished regulation map"
        )
    loser_atk = min(6, max(0, loser_score - 5))
    return loser_atk, loser_score - loser_atk


def _q_win_map(match_id, ts, map_name, q_id, opp_id, margin):
    """Build a queried-team win map with a controlled signed margin.

    The mass-fixture builder for wins: the queried team (always on the
    ``team1`` seat) wins ``13``-``(13 - margin)``, so its derived
    ``signed_margin`` is exactly ``+margin`` and its opponent's exactly
    ``-margin``. The winner's 13 rounds split 7 attack / 6 defence; the
    loser's ``13 - margin`` rounds are distributed by
    :func:`_opposing_side_wins` so ``round_detail``'s pairing invariants
    hold by construction.

    Args:
        match_id: The fixture match id.
        ts: The ISO match timestamp.
        map_name: The map name.
        q_id: The queried (winning) team's id.
        opp_id: The opponent's id.
        margin: The map's signed margin for the queried team
            (``2..13``; the loser scores ``13 - margin``).

    Returns:
        A map row dict via :func:`_regulation_map` (winner = ``q_id``)
        with the ``date`` key added.

    Raises:
        ValueError: If ``margin`` is outside ``[2, 13]`` (propagated
            from :func:`_regulation_map` / :func:`_opposing_side_wins`).
    """
    loser_score = 13 - margin
    loser_atk, loser_def = _opposing_side_wins(loser_score)
    row = _regulation_map(
        match_id, 0, map_name, q_id, opp_id, 13, loser_score,
        7, 6, loser_atk, loser_def, q_id,
    )
    row["date"] = ts
    return row


def _q_loss_map(match_id, ts, map_name, q_id, opp_id, margin):
    """Build a queried-team loss map with a controlled signed margin.

    The loss analogue of :func:`_q_win_map`: the queried team (always on
    the ``team1`` seat) loses ``(13 - margin)``-``13``, so its derived
    ``signed_margin`` is exactly ``-margin`` and its opponent's exactly
    ``+margin``. The *opponent's* 13 rounds split 7 attack / 6 defence
    and the queried team's ``13 - margin`` rounds are distributed by
    :func:`_opposing_side_wins` (the same split, with the winning and
    losing seats exchanged).

    Args:
        match_id: The fixture match id.
        ts: The ISO match timestamp.
        map_name: The map name.
        q_id: The queried (losing) team's id.
        opp_id: The opponent's id.
        margin: The map's signed margin for the queried team
            (``2..13``; the queried team scores ``13 - margin``).

    Returns:
        A map row dict via :func:`_regulation_map` (winner = ``opp_id``)
        with the ``date`` key added.

    Raises:
        ValueError: If ``margin`` is outside ``[2, 13]`` (propagated
            from :func:`_regulation_map` / :func:`_opposing_side_wins`).
    """
    loser_score = 13 - margin
    q_atk, q_def = _opposing_side_wins(loser_score)
    row = _regulation_map(
        match_id, 0, map_name, q_id, opp_id, loser_score, 13,
        q_atk, q_def, 7, 6, opp_id,
    )
    row["date"] = ts
    return row


def _row_to_matches(map_row):
    """Split a map row dict into its matches-table half.

    The map rows produced by the builders above carry the match-side
    columns (``match_id``/``date``/``team1_id``/``team2_id``/``status``)
    alongside the map columns; this helper extracts the former so a
    scenario can be materialised as both tables from one list of rows.

    Args:
        map_row: A row dict from :func:`_q_win_map` /
            :func:`_q_loss_map` / :func:`_regulation_map`.

    Returns:
        A dict with exactly the :data:`_MATCHES_COLS` keys.

    Raises:
        Nothing.
    """
    return {col: map_row[col] for col in _MATCHES_COLS}


def _core_tables():
    """Build the shared multi-team fixture with both seats and leakage rows.

    A five-map fixture exercising (a) both seats for the queried team,
    (b) two map names, (c) an unrelated-match row whose presence keeps
    the queried team's history a strict subset of the frame, and (d) the
    equal/after leakage rows:

    - ``m0``: U1 (team1) beats U2 on Haven 13-7 (margins +6/-6); neither
      side is the queried team T1.
    - ``m1``: T1 (team1) beats T2 on Haven 13-8 (T1 margin +5).
    - ``m2``: T3 (team1) beats T1 (team2) on Haven 13-9 (T1 margin -4;
      T1's seat is team2 here).
    - ``m3``: T1 (team1) beats T4 on Bind 13-2 (T1 margin +11).
    - ``m4``: a Haven map dated exactly at the query cutoff (excluded).
    - ``m5``: a Haven map dated after the cutoff (excluded).

    T1's own as-of history (m1-m3): 3 maps summing to +12 (Haven
    +5-4 = +1 over 2 maps, Bind +11 over 1), raw overall mean 4.0 —
    a strongly positive team whose pool proves the inner prior is the
    literal ``0.0`` constant, not T1's own mean.

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple; ``query_date`` is
        exactly the date of ``m4`` (so ``m0``-``m3`` are strictly before
        it and ``m4``/``m5`` are the leakage rows).

    Raises:
        Nothing.
    """
    t0 = _stamp(0)
    t1 = _stamp(1)
    t2 = _stamp(2)
    t3 = _stamp(3)
    query = _stamp(4)
    t5 = _stamp(5)
    map_rows = [
        # m0: unrelated teams U1/U2 on Haven 13-7 (margins +6/-6).
        _regulation_map(
            "m0", 0, "Haven", "U1", "U2", 13, 7, 7, 6, 2, 5, "U1",
        ),
        # m1: T1 as team1 wins Haven 13-8 (T1 margin +5).
        _regulation_map(
            "m1", 0, "Haven", "T1", "T2", 13, 8, 7, 6, 3, 5, "T1",
        ),
        # m2: T1 as team2 loses Haven 13-9 (T1 margin -4).
        _regulation_map(
            "m2", 0, "Haven", "T3", "T1", 13, 9, 7, 6, 4, 5, "T3",
        ),
        # m3: T1 as team1 wins Bind 13-2 (T1 margin +11).
        _regulation_map(
            "m3", 0, "Bind", "T1", "T4", 13, 2, 7, 6, 0, 2, "T1",
        ),
        # m4/m5: leakage rows (equal to / after the query date).
        _regulation_map(
            "m4", 0, "Haven", "T1", "T5", 13, 6, 7, 6, 1, 5, "T1",
        ),
        _regulation_map(
            "m5", 0, "Haven", "T1", "T6", 13, 6, 7, 6, 1, 5, "T1",
        ),
    ]
    dates = {mid: ts for mid, ts in zip(
        ("m0", "m1", "m2", "m3", "m4", "m5"),
        (t0, t1, t2, t3, query, t5),
    )}
    for row in map_rows:
        row["date"] = dates[row[rd.MATCH_ID_COL]]
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows), query


def _team_tables(segments, team_id="T"):
    """Build a recurring-team win history from ``(map_name, games, margin)`` segments.

    Produces one completed win map per game, in segment order, with the
    queried team always ``team1`` and winning with the segment's margin
    (orientation variation is covered by :func:`_core_tables`). Dates are
    one hour apart; the returned query date is one hour after the last
    map so every row is strictly before the cutoff.

    Args:
        segments: An iterable of ``(map_name, games, margin)`` triples;
            each expands into ``games`` maps on that map name, each won
            by the queried team with signed margin ``margin``.
        team_id: The queried team's id on the ``team1`` side (default
            ``"T"``).

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple; ``query_date`` is
        an ISO string one hour after the final map.

    Raises:
        ValueError: If any ``margin`` lies outside ``[2, 13]``
            (propagated from :func:`_q_win_map`).
    """
    map_rows = []
    position = 0
    for map_name, games, margin in segments:
        for _ in range(games):
            ts = _stamp(position)
            mid = f"t{position:04d}"
            map_rows.append(
                _q_win_map(mid, ts, map_name, team_id, f"o{position}", margin)
            )
            position += 1
    query = _stamp(position)
    for row in map_rows:
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows), query


def _strong_tables():
    """Build the strong-team fixture the hierarchy tests use.

    Team ``S`` (always ``team1``) wins 30 Haven maps, each 13-8 (signed
    margin +5): its as-of overall sample sums to +150 over 30 maps (raw
    mean 5.0, shrunk mean with :data:`DEFAULT_OVERALL_K` = 10 exactly
    150/40 = 3.75), and it has *no* Ascent maps — the fixture the
    zero-map-history full-shrinkage and prior-chain tests assert against.

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple.

    Raises:
        Nothing.
    """
    return _team_tables([("Haven", 30, 5)], team_id="S")


def _ascent_strong_tables():
    """Build the strong-team fixture with a thin sampled second map.

    Team ``S`` wins 30 Haven maps at margin +5 (sum +150) *and* 2 Ascent
    maps at margin +2 (sum +4): the Ascent sample (n=2, raw mean 2.0) is
    the thin map sample the roadmap's "strength" framing targets, while
    the overall sample sums to +154 over 32 maps (raw mean 4.8125, shrunk
    mean with :data:`DEFAULT_OVERALL_K` = 10 exactly 154/42 ~ 3.6667).
    With the map-level prior that shrunk overall mean (3.6667) well above
    Ascent's raw 2.0, the map-level shrinkage must pull the estimate
    *up* toward the team's general strength.

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple.

    Raises:
        Nothing.
    """
    return _team_tables([("Haven", 30, 5), ("Ascent", 2, 2)], team_id="S")


# --------------------------------------------------------------------------
# symmetric-zero identity + the inner prior
# --------------------------------------------------------------------------


def test_symmetric_zero_identity_per_map_and_exact_global_sum():
    # On every fixture map the two seats' derived signed margins cancel
    # exactly (team2's value is always the negation of team1's on the
    # same map — integer arithmetic, no float residual), and the global
    # sum over the whole derived frame is exactly 0.
    _matches, maps_df, _query = _core_tables()
    derived = rd.derive_map_round_details(maps_df).records
    assert derived["signed_margin"].sum() == 0
    pair_sums = derived.groupby([rd.MATCH_ID_COL, rd.MAP_INDEX_COL])["signed_margin"].sum()
    assert (pair_sums == 0).all()


def test_overall_prior_is_literal_zero_constant_not_own_or_pool_mean():
    # The inner-level prior is the module constant 0.0, never re-derived:
    # T1's own as-of history sums to +12 over 3 maps (raw mean 4.0, far
    # from zero), yet result.prior == LEAGUE_MEAN_SIGNED_MARGIN == 0.0
    # exactly while result.raw_mean == 4.0 — a self-shrinking
    # implementation (prior = own raw mean) or a pool-mean one would both
    # disagree here.
    matches_df, maps_df, query = _core_tables()
    result = sm.team_overall_signed_margin("T1", query, matches_df, maps_df)
    assert result.n_maps == 3
    assert result.sum_margin == 12
    assert result.raw_mean == pytest.approx(4.0)
    assert sm.LEAGUE_MEAN_SIGNED_MARGIN == 0.0
    assert result.prior == sm.LEAGUE_MEAN_SIGNED_MARGIN == 0.0
    assert result.raw_mean != result.prior


# --------------------------------------------------------------------------
# team_overall_signed_margin: shrinkage behaviour
# --------------------------------------------------------------------------


def test_overall_zero_history_full_shrinkage_to_league_prior():
    # An unseen team has no maps; the inner posterior degrades to
    # mean == raw_mean == the 0.0 structural prior exactly.
    matches_df, maps_df, query = _core_tables()
    result = sm.team_overall_signed_margin("UNSEEN", query, matches_df, maps_df)
    assert result.n_maps == 0
    assert result.sum_margin == 0
    assert result.prior == 0.0
    assert result.raw_mean == pytest.approx(0.0)
    assert result.mean == pytest.approx(0.0)


def test_overall_few_maps_shrinks_toward_league_prior():
    # A team whose only history is one 13-8 win (raw mean +5.0): with
    # k=10 the shrunk mean 5/11 sits strictly between the 0.0 prior and
    # the raw rate (exact-formula check).
    matches_df, maps_df, query = _team_tables([("Haven", 1, 5)])
    result = sm.team_overall_signed_margin("T", query, matches_df, maps_df, k=10.0)
    assert result.n_maps == 1
    assert result.sum_margin == 5
    assert result.raw_mean == pytest.approx(5.0)
    assert result.mean == pytest.approx(5.0 / 11.0)
    assert 0.0 < result.mean < result.raw_mean


def test_overall_many_maps_close_to_raw_mean():
    # 100 maps at margin +4 (raw 4.0) swamp the fixed k, so the shrunk
    # mean is within a small tolerance of the raw mean even though the
    # 0.0 prior differs from it.
    matches_df, maps_df, query = _team_tables([("Haven", 100, 4)])
    result = sm.team_overall_signed_margin("T", query, matches_df, maps_df)
    assert result.n_maps == 100
    assert result.sum_margin == 400
    assert result.raw_mean == pytest.approx(4.0)
    assert abs(result.mean - result.raw_mean) < 0.5


def test_overall_monotonic_in_k_toward_prior():
    # Holding the team sample fixed, mean moves monotonically toward the
    # 0.0 prior as k grows (raw +5.0 > prior, so it decreases), and at
    # heavy k it lands close to the prior.
    matches_df, maps_df, query = _team_tables([("Haven", 40, 5)])
    means = [
        sm.team_overall_signed_margin("T", query, matches_df, maps_df, k=k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[0] < 5.0
    assert abs(means[-1] - 0.0) < 0.2


# --------------------------------------------------------------------------
# team_map_signed_margin: the two-level hierarchy
# --------------------------------------------------------------------------


def test_map_zero_history_full_shrinkage_to_shrunk_overall():
    # Zero maps on the queried map: mean == prior exactly, where prior is
    # the team's *shrunk overall* mean margin (inner-level output), not
    # the 0.0 league constant and not the raw team-overall mean.
    matches_df, maps_df, query = _strong_tables()
    overall = sm.team_overall_signed_margin("S", query, matches_df, maps_df)
    assert overall.raw_mean == pytest.approx(5.0)
    assert overall.mean == pytest.approx(150.0 / 40.0)  # shrunk, below raw
    result = sm.team_map_signed_margin("S", "Ascent", query, matches_df, maps_df)
    assert result.n_maps == 0
    assert result.sum_margin == 0
    assert result.prior == pytest.approx(overall.mean)
    assert result.prior == pytest.approx(3.75)
    assert result.prior > 1.0  # clearly not the 0.0 league constant
    assert result.raw_mean == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)


def test_map_prior_is_shrunk_overall_not_raw_or_league():
    # The hierarchy's distinguishing behaviour asserted against the raw
    # and league alternatives at once: with k=5 the map-level mean must
    # interpolate between Ascent's raw 2.0 and the shrunk overall prior
    # (~3.6667), and the prior must equal team_overall_signed_margin
    # (same date). A flat single-level shrinkage toward 0.0 would pull
    # the estimate the wrong way entirely (raw 2.0 > 0.0 would shrink
    # *down*, toward no signal); a prior of the raw overall 4.8125 would
    # also disagree.
    matches_df, maps_df, query = _ascent_strong_tables()
    overall = sm.team_overall_signed_margin("S", query, matches_df, maps_df)
    assert overall.raw_mean == pytest.approx(154.0 / 32.0)
    assert overall.mean == pytest.approx(154.0 / 42.0)
    result = sm.team_map_signed_margin("S", "Ascent", query, matches_df, maps_df, k=5.0)
    assert result.n_maps == 2
    assert result.sum_margin == 4
    assert result.raw_mean == pytest.approx(2.0)
    assert result.prior == pytest.approx(overall.mean)  # shrunk overall
    assert result.prior != pytest.approx(0.0)
    assert result.prior != pytest.approx(overall.raw_mean)
    assert result.raw_mean < result.mean < result.prior  # pulled up toward strength


def test_map_few_maps_shrinks_toward_prior():
    # The map-level sample (30 Haven maps at margin +5, raw mean 5.0)
    # against a shrunk overall prior of 3.75: the k*prior pseudo-maps
    # pull the mean strictly below the raw rate toward the prior
    # (exact-formula check with k=5).
    matches_df, maps_df, query = _strong_tables()
    result = sm.team_map_signed_margin("S", "Haven", query, matches_df, maps_df, k=5.0)
    assert result.n_maps == 30
    assert result.sum_margin == 150
    assert result.raw_mean == pytest.approx(5.0)
    assert result.prior == pytest.approx(3.75)
    assert result.mean == pytest.approx((150.0 + 5.0 * 3.75) / 35.0)
    assert result.prior < result.mean < result.raw_mean


def test_map_monotonic_in_k_toward_prior():
    # Holding the map sample fixed, mean moves monotonically toward the
    # shrunk-overall prior as k grows (raw 5.0 > prior 3.75, so it
    # decreases), bounded below by the prior.
    matches_df, maps_df, query = _strong_tables()
    means = [
        sm.team_map_signed_margin("S", "Haven", query, matches_df, maps_df, k=k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[0] < 5.0
    assert abs(means[-1] - 3.75) < 0.1


def test_map_normalizes_map_name():
    # A caller passing " breeze " must match the stored "Breeze" via
    # normalize_map_name rather than depending on exact case/whitespace.
    matches_df, maps_df, query = _team_tables([("Breeze", 5, 5), ("Haven", 5, 2)])
    result = sm.team_map_signed_margin("T", " breeze ", query, matches_df, maps_df)
    assert result.n_maps == 5
    assert result.sum_margin == 25
    assert result.raw_mean == pytest.approx(5.0)


# --------------------------------------------------------------------------
# seat resolution + leakage
# --------------------------------------------------------------------------


def test_seat_resolution_correct_for_both_orientations():
    # T1 is team1 on m1/m3 and team2 on m2; the seat resolution must pick
    # T1's own derived record per map or the margins would be wrong (the
    # continuous-quantity analogue of M38.2's seat/phase mix-up test). On
    # Haven (m1 as team1 with +5, m2 as team2 with -4) T1's sample is 2
    # maps summing to +1; over all maps (adding m3's +11 Bind) it is 3
    # maps summing to +12. A seat bug taking team1's record on m2 would
    # give +5 + (+4) = +9 on Haven; a sign-flip bug (taking the
    # opponent's) would give -1 on Haven.
    matches_df, maps_df, query = _core_tables()
    haven = sm.team_map_signed_margin("T1", "Haven", query, matches_df, maps_df)
    assert (haven.n_maps, haven.sum_margin) == (2, 1)
    assert haven.raw_mean == pytest.approx(0.5)
    overall = sm.team_overall_signed_margin("T1", query, matches_df, maps_df)
    assert (overall.n_maps, overall.sum_margin) == (3, 12)
    # The private helper exposes the raw sums for both filters.
    assert sm._team_signed_margins("T1", query, matches_df, maps_df) == (3, 12)
    assert sm._team_signed_margins(
        "T1", query, matches_df, maps_df, map_name="Haven"
    ) == (2, 1)
    assert sm._team_signed_margins(
        "T1", query, matches_df, maps_df, map_name="Bind"
    ) == (1, 11)
    assert sm._team_signed_margins(
        "T1", query, matches_df, maps_df, map_name="Ascent"
    ) == (0, 0)


def test_equal_and_after_maps_never_enter_any_estimator():
    # The leakage proof across both estimators: m4 (exactly at the
    # cutoff) and m5 (after it) must not contribute, and trimming them
    # leaves every estimate byte-for-byte unchanged. (T1's Haven maps
    # m4/m5 both carry margin +7, so an inclusive-<= bug would change
    # the Haven sample from (2, 1) to (4, 15).)
    matches_df, maps_df, query = _core_tables()
    before = (
        sm.team_overall_signed_margin("T1", query, matches_df, maps_df),
        sm.team_map_signed_margin("T1", "Haven", query, matches_df, maps_df),
    )
    trimmed_m = matches_df[~matches_df["match_id"].isin(["m4", "m5"])]
    trimmed_p = maps_df[~maps_df[rd.MATCH_ID_COL].isin(["m4", "m5"])]
    after = (
        sm.team_overall_signed_margin("T1", query, trimmed_m, trimmed_p),
        sm.team_map_signed_margin("T1", "Haven", query, trimmed_m, trimmed_p),
    )
    for x, y in zip(before, after):
        assert (x.n_maps, x.sum_margin, x.raw_mean, x.prior, x.mean) == (
            y.n_maps, y.sum_margin, y.raw_mean, y.prior, y.mean,
        )


def test_strictly_earlier_map_changes_the_estimate():
    # Flip side of the leakage proof: a strictly-earlier completed Haven
    # map must change the map-level counts.
    matches_df, maps_df, query = _core_tables()
    base = sm.team_map_signed_margin("T1", "Haven", query, matches_df, maps_df)
    earlier = _regulation_map(
        "m0b", 0, "Haven", "T1", "T9", 13, 8, 7, 6, 3, 5, "T1",
    )
    earlier["date"] = _stamp(-1)
    earlier["status"] = "completed"
    grown_m = pd.concat([matches_df, _matches_df([_row_to_matches(earlier)])], ignore_index=True)
    grown_p = pd.concat([maps_df, _maps_df([earlier])], ignore_index=True)
    grown = sm.team_map_signed_margin("T1", "Haven", query, grown_m, grown_p)
    assert (grown.n_maps, grown.sum_margin) == (base.n_maps + 1, base.sum_margin + 5)


@pytest.mark.parametrize("k", [0, -1, float("nan"), float("inf"), "abc", None])
def test_invalid_k_raises_from_both_public_functions(k):
    # k <= 0 (or NaN/inf/non-numeric) is rejected up front by both
    # estimators through the shared features._shared._validate_k choke
    # point.
    matches_df, maps_df, query = _core_tables()
    with pytest.raises(ValueError, match="k must be"):
        sm.team_overall_signed_margin("T1", query, matches_df, maps_df, k=k)
    with pytest.raises(ValueError, match="k must be"):
        sm.team_map_signed_margin("T1", "Haven", query, matches_df, maps_df, k=k)


# --------------------------------------------------------------------------
# real-data smoke test
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_sane_numbers_and_exact_zero_identity():
    # Real v1 sanity (query = latest match + 1h, most frequent team 1001,
    # most frequent map Lotus; DEFAULT_OVERALL_K=10, DEFAULT_MAP_K=5):
    #   team 1001 overall: n_maps=50, sum_margin=+67, raw_mean=1.34,
    #     mean=1.1167 (prior 0.0).
    #   team 1001 Lotus: n_maps=12, sum_margin=+14, raw_mean=1.1667,
    #     prior=1.1167 (= overall.mean), mean=1.1520.
    # The per-team sum (+67) is NOT zero — a single seat does not cancel —
    # but the whole-dataset symmetric-zero identity holds exactly: global
    # sum over all 484 derived records == 0 and every per-(match_id,
    # map_index) pair sums to exactly 0.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()
    map_name = maps_df["map_name"].value_counts().idxmax()

    overall = sm.team_overall_signed_margin(team_id, query, matches_df, maps_df)
    assert overall.n_maps > 0
    assert overall.prior == sm.LEAGUE_MEAN_SIGNED_MARGIN == 0.0
    assert overall.mean == pytest.approx(overall.raw_mean, abs=1.0)  # finite & sane
    assert float(overall.raw_mean) == overall.raw_mean  # finite float
    assert float(overall.mean) == overall.mean

    shrunk = sm.team_map_signed_margin(
        team_id, map_name, query, matches_df, maps_df
    )
    assert shrunk.n_maps >= 0
    assert shrunk.n_maps <= overall.n_maps
    assert shrunk.prior == pytest.approx(overall.mean)
    assert float(shrunk.raw_mean) == shrunk.raw_mean
    assert float(shrunk.mean) == shrunk.mean

    # The symmetric-zero identity, independently recomputed against the
    # full real maps table (no as-of filter): exact integer cancellation.
    result = rd.derive_map_round_details(maps_df)
    assert result.records["signed_margin"].sum() == 0
    pair_sums = result.records.groupby(
        [rd.MATCH_ID_COL, rd.MAP_INDEX_COL]
    )["signed_margin"].sum()
    assert (pair_sums == 0).all()

"""Tests for side-specific (attack/defence phase) shrunk round win rates (M38.2).

Covers the three estimators (``league_side_rate`` /
``team_overall_side_rate`` / ``team_map_side_rate``) and their Beta
posterior dataclasses, the shrinkage-behaviour contracts (zero history ->
full shrinkage, few rounds -> pulled toward the prior, many rounds ->
close to raw, monotonic in ``k``, posterior sanity), the two-level
hierarchy's distinguishing behaviour (the map-level ``prior`` is the
*shrunk* overall-phase rate, not the raw league rate), the seat-vs-phase
resolution (a team queried from both the ``team1`` and the ``team2`` seat
gets the correct per-phase round counts), the leakage-safety proof (maps
dated at/after the cutoff never enter any estimator), the round-level
``select_k`` CV harness (low round data prefers a large ``k``, high round
data prefers a small ``k``, phase-scoping via a phase-swap identity and a
pooling guard), and a skip-guarded smoke test recording the real
``data/v1`` numbers. Test fixtures build internally-consistent regulation
map rows (the opposing-side-pairing and case-split invariants of
``features.round_detail`` hold by construction); a non-test helper raises
``ValueError`` on an invalid row so a mis-typed fixture fails at
construction, never silently mid-test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import round_detail as rd
from features import side_win_rate as swr
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
    plus the full ``round_detail`` required column set). Null round
    columns may be passed (``None``) to construct the round_detail
    exclusion case; the caller is responsible for such a row being
    otherwise well-formed.

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
    these fixtures). The four half-split columns default to non-null
    ``0.0`` (unused by derivation, see :data:`_HALF_COLS`). An invalid
    combination raises immediately so a mis-typed fixture never reaches
    the module.

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


def _q_win_map(match_id, ts, map_name, q_id, opp_id, atk_won, atk_played=12, def_played=12):
    """Build a queried-team win map with controlled derived phase samples.

    The mass-fixture builder for scenarios where one recurring team (the
    "queried" team, always placed on the ``team1`` seat) wins every map
    13-X. The team's derived attack half contains ``atk_won`` wins in
    ``atk_played`` rounds and its derived defence half the complementary
    ``13 - atk_won`` wins in ``def_played`` rounds, so a caller controls
    the per-phase sample sizes (``atk_played``/``def_played``) and success
    counts (``atk_won``; defence wins are forced by the 13-round score)
    independently per map.

    Args:
        match_id: The fixture match id.
        ts: The ISO match timestamp.
        map_name: The map name.
        q_id: The queried (winning) team's id.
        opp_id: The opponent's id (a unique team per match in the CV
            scenarios, making it k-neutral filler).
        atk_won: The queried team's derived attack-round wins
            (``1..12``; defence wins are ``13 - atk_won``).
        atk_played: The queried team's derived attack-half size
            (default 12, the full half).
        def_played: The queried team's derived defence-half size
            (default 12; pass 3 to build a 13-2 blowout whose defence
            sample is a quarter the size of its attack sample).

    Returns:
        A map row dict via :func:`_regulation_map` (winner = ``q_id``).

    Raises:
        ValueError: If ``atk_won``/``def_played`` are inconsistent with a
            valid finished regulation win (propagated from
            :func:`_regulation_map`).
    """
    def_won = 13 - atk_won
    t2_score = atk_played + def_played - 13
    t2_atk = def_played - def_won
    t2_def = atk_played - atk_won
    row = _regulation_map(
        match_id, 0, map_name, q_id, opp_id, 13, t2_score,
        atk_won, def_won, t2_atk, t2_def, q_id,
    )
    row["date"] = ts
    return row


def _q_loss_map(match_id, ts, map_name, q_id, opp_id, atk_won, atk_played=12, def_played=12):
    """Build a queried-team loss map with controlled derived phase samples.

    The loss analogue of :func:`_q_win_map`: the queried team (always on
    the ``team1`` seat) loses 11-13, so its derived attack half contains
    ``atk_won`` wins in ``atk_played`` rounds and its defence half the
    complementary ``11 - atk_won`` wins in ``def_played`` rounds, with the
    halves summing to the full 24 rounds of a close loss (a loss cannot be
    a blowout without breaking the pairing partition, so both halves must
    fill out the map). Losses are what decouple the attack and defence
    per-phase rates from each other (a win's two phase counts always sum
    to 13; a loss's to 11), which the phase-scoping fixture relies on.

    Args:
        match_id: The fixture match id.
        ts: The ISO match timestamp.
        map_name: The map name.
        q_id: The queried (losing) team's id.
        opp_id: The opponent's id.
        atk_won: The queried team's derived attack-round wins.
        atk_played: The queried team's derived attack-half size.
        def_played: The queried team's derived defence-half size
            (``atk_played + def_played`` must equal the full 24 rounds).

    Returns:
        A map row dict via :func:`_regulation_map` (winner = ``opp_id``).

    Raises:
        ValueError: If ``atk_won``/``def_played`` are inconsistent with a
            valid finished regulation loss (propagated from
            :func:`_regulation_map`).
    """
    def_won = 11 - atk_won
    t2_score = atk_played + def_played - 11
    t2_atk = def_played - def_won
    t2_def = atk_played - atk_won
    row = _regulation_map(
        match_id, 0, map_name, q_id, opp_id, 11, t2_score,
        atk_won, def_won, t2_atk, t2_def, opp_id,
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

    A four-map fixture exercising (a) both seats for the queried team,
    (b) both phases, (c) a second map name, and (d) the equal/after
    leakage rows:

    - ``m1``: T1 (team1) beats T2 on Haven 13-8, T1's derived attack
      7 of 12 rounds and defence 6 of 9.
    - ``m2``: T3 (team1) beats T1 (team2) on Haven 13-11, T1's derived
      attack 6 of 12 and defence 5 of 12 (T1's seat is team2 here).
    - ``m3``: T1 (team1) beats T4 on Bind 13-8, T1's derived attack 7 of
      12 and defence 6 of 9.
    - ``m4``: a Haven map dated exactly at the query cutoff (excluded).
    - ``m5``: a Haven map dated after the cutoff (excluded).

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple; ``query_date`` is
        exactly the date of ``m4`` (so ``m1``-``m3`` are strictly before
        it and ``m4``/``m5`` are the leakage rows).

    Raises:
        Nothing.
    """
    t1 = _stamp(0)
    t2 = _stamp(1)
    t3 = _stamp(2)
    query = _stamp(3)
    t5 = _stamp(4)
    map_rows = [
        # m1: T1 as team1 wins Haven 13-8 (atk 7/12, def 6/9).
        _regulation_map(
            "m1", 0, "Haven", "T1", "T2", 13, 8, 7, 6, 3, 5, "T1",
        ),
        # m2: T1 as team2 loses Haven 13-11 (atk 6/12, def 5/12).
        _regulation_map(
            "m2", 0, "Haven", "T3", "T1", 13, 11, 7, 6, 6, 5, "T3",
        ),
        # m3: T1 as team1 wins Bind 13-8 (atk 7/12, def 6/9).
        _regulation_map(
            "m3", 0, "Bind", "T1", "T4", 13, 8, 7, 6, 3, 5, "T1",
        ),
        # m4/m5: leakage rows (equal to / after the query date).
        _regulation_map(
            "m4", 0, "Haven", "T1", "T5", 13, 8, 7, 6, 3, 5, "T1",
        ),
        _regulation_map(
            "m5", 0, "Haven", "T1", "T6", 13, 8, 7, 6, 3, 5, "T1",
        ),
    ]
    dates = {mid: ts for mid, ts in zip(("m1", "m2", "m3", "m4", "m5"), (t1, t2, t3, query, t5))}
    for row in map_rows:
        row["date"] = dates[row[rd.MATCH_ID_COL]]
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows), query


def _team_tables(segments, team_id="T"):
    """Build a recurring-team history from ``(map_name, games, attack_wins)`` segments.

    Produces one completed 13-11 win (and one map) per game, in segment
    order, with the queried team always ``team1`` (orientation variation
    is covered by :func:`_core_tables`). Every map is a 13-11 regulation
    win so each side's derived attack half is exactly 12 rounds, with
    ``attack_wins`` of them won by the queried team (its derived defence
    half is also 12 rounds with ``13 - attack_wins`` wins). Dates are one
    hour apart; the returned query date is one hour after the last map so
    every row is strictly before the cutoff.

    Args:
        segments: An iterable of ``(map_name, games, attack_wins)``
            triples; each expands into ``games`` maps on that map name
            with ``attack_wins`` attack-round wins per map.
        team_id: The queried team's id on the ``team1`` side (default
            ``"T"``).

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple; ``query_date`` is
        an ISO string one hour after the final map.

    Raises:
        ValueError: If any ``attack_wins`` lies outside ``[1, 12]``
            (propagated from :func:`_q_win_map`).
    """
    map_rows = []
    position = 0
    for map_name, games, attack_wins in segments:
        for _ in range(games):
            ts = _stamp(position)
            mid = f"t{position:04d}"
            map_rows.append(
                _q_win_map(
                    mid, ts, map_name, team_id, f"o{position}",
                    attack_wins, 12, 12,
                )
            )
            position += 1
    query = _stamp(position)
    for row in map_rows:
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows), query


def _pool_tables():
    """Build the two-map, four-team league-pool fixture.

    ``m1`` (13-8: T1 as team1 atk 7/def 6, T2 atk 3/def 5) and ``m2``
    (13-11: T3 atk 7/def 6, T4 atk 6/def 5), both strictly before the
    returned query date. Pooled over both seats: attack 23 won of 45
    played, defence 22 of 45 (hand-computed; see the test that consumes
    it).

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple.

    Raises:
        Nothing.
    """
    rows = [
        _regulation_map("p1", 0, "Haven", "T1", "T2", 13, 8, 7, 6, 3, 5, "T1"),
        _regulation_map("p2", 0, "Haven", "T3", "T4", 13, 11, 7, 6, 6, 5, "T3"),
    ]
    for row, position in zip(rows, (0, 1)):
        row["date"] = _stamp(position)
        row["status"] = "completed"
    query = _stamp(2)
    return _matches_df([_row_to_matches(row) for row in rows]), _maps_df(rows), query


# --------------------------------------------------------------------------
# league_side_rate
# --------------------------------------------------------------------------


def test_league_side_rate_hand_computed_pool():
    # A two-map, four-team pool: attack rounds pool to 23 won of 45
    # played and defence to 22 of 45 (each map contributes both seats'
    # derived regulation halves; the pool does not care which seat a team
    # sat in).
    matches_df, maps_df, query = _pool_tables()
    attack = swr.league_side_rate("attack", query, matches_df, maps_df)
    defense = swr.league_side_rate("defense", query, matches_df, maps_df)
    assert (attack.rounds_won, attack.rounds_played) == (23, 45)
    assert attack.rate == pytest.approx(23 / 45)
    assert (defense.rounds_won, defense.rounds_played) == (22, 45)
    assert defense.rate == pytest.approx(22 / 45)


def test_league_side_rate_zero_history_defaults_to_half():
    # A cutoff before any match yields an empty pool; the documented
    # uninformative default is 0.5 (a side round win is a genuine ~50/50
    # quantity, unlike a rare event's 0.0).
    matches_df, maps_df, _query = _pool_tables()
    dawn = _stamp(-2)
    attack = swr.league_side_rate("attack", dawn, matches_df, maps_df)
    defense = swr.league_side_rate("defense", dawn, matches_df, maps_df)
    assert attack == swr.LeagueSideRate(rounds_won=0, rounds_played=0, rate=0.5)
    assert defense == swr.LeagueSideRate(rounds_won=0, rounds_played=0, rate=0.5)


# --------------------------------------------------------------------------
# team_overall_side_rate: shrinkage behaviour
# --------------------------------------------------------------------------


def test_team_overall_side_rate_zero_history_full_shrinkage():
    # An unseen team has no rounds; the inner posterior degrades to
    # mean == raw_rate == the league prior exactly.
    matches_df, maps_df, query = _core_tables()
    result = swr.team_overall_side_rate("UNSEEN", "attack", query, matches_df, maps_df)
    league = swr.league_side_rate("attack", query, matches_df, maps_df)
    assert result.rounds_won == 0
    assert result.rounds_played == 0
    assert result.prior == league.rate
    assert result.raw_rate == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)
    assert result.alpha > 0.0 and result.beta > 0.0


def test_team_overall_side_rate_few_rounds_shrinks_toward_prior():
    # A team whose only history is one 13-11 win (10 of 12 attack rounds,
    # raw 0.8333) against a league pool of 19 of 24 (0.7917): with
    # prior_k=10 the posterior mean 17.9167/22 sits strictly between the
    # league prior and the raw rate (exact-formula check).
    matches_df, maps_df, query = _team_tables([("Haven", 1, 10)])
    result = swr.team_overall_side_rate(
        "T", "attack", query, matches_df, maps_df, prior_k=10.0
    )
    assert result.rounds_won == 10
    assert result.rounds_played == 12
    assert result.prior == pytest.approx(19 / 24)
    assert result.raw_rate == pytest.approx(10 / 12)
    assert result.mean == pytest.approx((10 + 10.0 * (19 / 24)) / (12 + 10.0))
    assert result.prior < result.mean < result.raw_rate


def test_team_overall_side_rate_many_rounds_close_to_raw():
    # 100 maps (1200 attack rounds at 0.8333) swamp the fixed prior_k, so
    # mean is within a small tolerance of the raw rate even though the
    # league prior differs from it.
    matches_df, maps_df, query = _team_tables([("Haven", 100, 10)])
    result = swr.team_overall_side_rate("T", "attack", query, matches_df, maps_df)
    assert result.rounds_won == 1000
    assert result.rounds_played == 1200
    assert result.raw_rate == pytest.approx(1000 / 1200)
    assert abs(result.mean - result.raw_rate) < 0.01


def test_team_overall_side_rate_posterior_sanity_and_variance_shrinks():
    # alpha/beta/variance are always positive and mean lies strictly
    # inside (0, 1); a larger sample yields a tighter posterior.
    small_m, small_p, q1 = _team_tables([("Haven", 10, 10)])
    large_m, large_p, q2 = _team_tables([("Haven", 100, 10)])
    small = swr.team_overall_side_rate("T", "attack", q1, small_m, small_p)
    large = swr.team_overall_side_rate("T", "attack", q2, large_m, large_p)
    for result in (small, large):
        assert result.alpha > 0.0
        assert result.beta > 0.0
        assert result.variance > 0.0
        assert 0.0 < result.mean < 1.0
    assert large.variance < small.variance


def test_team_overall_side_rate_rounds_are_phase_scoped():
    # The same team's attack and defence totals come from the same maps
    # but different halves: on the core fixture T1's attack rounds are
    # 20 of 36 and its defence rounds 17 of 30 (m2's T1-seat defence
    # played is 12, not the mirror of its attack played).
    matches_df, maps_df, query = _core_tables()
    attack = swr.team_overall_side_rate("T1", "attack", query, matches_df, maps_df)
    defense = swr.team_overall_side_rate("T1", "defense", query, matches_df, maps_df)
    assert (attack.rounds_won, attack.rounds_played) == (20, 36)
    assert (defense.rounds_won, defense.rounds_played) == (17, 30)


def test_team_overall_side_rate_invalid_prior_k_raises():
    # prior_k <= 0 (or NaN/inf) is rejected like any shrinkage strength.
    matches_df, maps_df, query = _core_tables()
    with pytest.raises(ValueError, match="k must be"):
        swr.team_overall_side_rate("T1", "attack", query, matches_df, maps_df, prior_k=0)


# --------------------------------------------------------------------------
# team_map_side_rate: the two-level hierarchy
# --------------------------------------------------------------------------


def test_team_map_side_rate_zero_history_full_shrinkage_to_shrunk_overall():
    # Zero rounds on the queried map: mean == prior exactly, where prior
    # is the team's *shrunk overall* phase rate (inner level output), not
    # the raw team rate or the league rate.
    matches_df, maps_df, query = _core_tables()
    overall = swr.team_overall_side_rate("T1", "attack", query, matches_df, maps_df)
    result = swr.team_map_side_rate(
        "T1", "Ascent", "attack", query, matches_df, maps_df, 10.0
    )
    assert result.rounds_won == 0
    assert result.rounds_played == 0
    assert result.prior == pytest.approx(overall.mean)
    assert result.raw_rate == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)


def test_team_map_side_rate_prior_is_shrunk_overall_not_raw_league_rate():
    # The hierarchy's distinguishing behaviour: a team whose only map is a
    # 13-2 attack sweep (raw map attack 1.0) has a shrunk overall prior
    # far above the league rate, and the map-level prior must equal that
    # shrunk overall mean — a flat single-level shrinkage toward the
    # league would be wrong by a wide margin.
    map_rows = []
    position = 0
    for _ in range(30):
        ts = _stamp(position)
        map_rows.append(_q_win_map(f"b{position}", ts, "Haven", "S", f"so{position}", 12, 12, 3))
        position += 1
    for _ in range(200):
        ts = _stamp(position)
        map_rows.append(_q_win_map(f"f{position}", ts, "Bind", f"fa{position}", f"fb{position}", 6, 12, 12))
        position += 1
    for row in map_rows:
        row["status"] = "completed"
    matches_df = _matches_df([_row_to_matches(row) for row in map_rows])
    maps_df = _maps_df(map_rows)
    query = _stamp(position)

    league = swr.league_side_rate("attack", query, matches_df, maps_df)
    overall = swr.team_overall_side_rate("S", "attack", query, matches_df, maps_df)
    assert overall.mean > league.rate + 0.2  # the shrunk overall is not the league rate
    result = swr.team_map_side_rate(
        "S", "Haven", "attack", query, matches_df, maps_df, 10.0
    )
    assert result.prior == pytest.approx(overall.mean)
    assert result.prior > league.rate + 0.2


def test_team_map_side_rate_few_rounds_shrinks_toward_prior():
    # One Haven map with 12 of 12 attack rounds won (raw 1.0): the
    # k*prior pseudo-counts pull the mean strictly below the raw rate
    # toward the (shrunk overall) prior.
    matches_df, maps_df, query = _team_tables([("Haven", 1, 12), ("Bind", 1, 6)])
    result = swr.team_map_side_rate(
        "T", "Haven", "attack", query, matches_df, maps_df, 10.0
    )
    assert result.rounds_won == 12
    assert result.rounds_played == 12
    assert result.raw_rate == 1.0
    assert 0.0 < result.prior < 1.0
    assert result.prior < result.mean < result.raw_rate


def test_team_map_side_rate_monotonic_toward_prior():
    # Holding the map sample fixed, mean moves monotonically toward the
    # prior as k grows (raw 1.0 > prior, so it decreases).
    matches_df, maps_df, query = _team_tables([("Haven", 1, 12), ("Bind", 1, 6)])
    means = [
        swr.team_map_side_rate("T", "Haven", "attack", query, matches_df, maps_df, k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[3] > 0.0
    assert means[0] < 1.0


def test_team_map_side_rate_normalizes_map_name():
    # A caller passing " breeze " must match the stored "Breeze" via
    # normalize_map_name rather than depending on exact case/whitespace.
    matches_df, maps_df, query = _team_tables([("Breeze", 5, 10), ("Haven", 5, 2)])
    result = swr.team_map_side_rate(
        "T", " breeze ", "attack", query, matches_df, maps_df, 10.0
    )
    assert result.rounds_won == 50
    assert result.rounds_played == 60


def test_team_map_side_rate_invalid_k_raises():
    # k <= 0 (or NaN/inf) is rejected up front.
    matches_df, maps_df, query = _core_tables()
    with pytest.raises(ValueError, match="k must be"):
        swr.team_map_side_rate("T1", "Haven", "attack", query, matches_df, maps_df, 0)


# --------------------------------------------------------------------------
# seat resolution + leakage
# --------------------------------------------------------------------------


def test_seat_resolution_correct_for_both_orientations_and_phases():
    # T1 is team1 on m1/m3 and team2 on m2; the seat resolution must pick
    # T1's own derived record per map or the counts would be wrong. Haven
    # (m1 as team1 + m2 as team2): attack 13 won of 24, defence 11 won of
    # 21. The defence played differs from the attack played on m1 (9 vs
    # 12) because m1's scoreline truncates T1's defence half — a phase
    # mix-up or a seat mix-up both change every count here.
    matches_df, maps_df, query = _core_tables()

    haven_attack = swr.team_map_side_rate(
        "T1", "Haven", "attack", query, matches_df, maps_df, 10.0
    )
    haven_defense = swr.team_map_side_rate(
        "T1", "Haven", "defense", query, matches_df, maps_df, 10.0
    )
    assert (haven_attack.rounds_won, haven_attack.rounds_played) == (13, 24)
    assert (haven_defense.rounds_won, haven_defense.rounds_played) == (11, 21)

    # Over all maps the counts change again (m3 is Bind): attack 20/36,
    # defence 17/30 — cross-checked against team_overall_side_rate.
    overall_attack = swr.team_overall_side_rate("T1", "attack", query, matches_df, maps_df)
    overall_defense = swr.team_overall_side_rate("T1", "defense", query, matches_df, maps_df)
    assert (overall_attack.rounds_won, overall_attack.rounds_played) == (20, 36)
    assert (overall_defense.rounds_won, overall_defense.rounds_played) == (17, 30)


def test_equal_and_after_maps_never_enter_any_estimator():
    # The leakage proof across all three estimators: m4 (exactly at the
    # cutoff) and m5 (after it) must not contribute, and trimming them
    # leaves every estimate byte-for-byte unchanged.
    matches_df, maps_df, query = _core_tables()
    before = (
        swr.league_side_rate("attack", query, matches_df, maps_df),
        swr.team_overall_side_rate("T1", "attack", query, matches_df, maps_df),
        swr.team_map_side_rate("T1", "Haven", "attack", query, matches_df, maps_df, 10.0),
    )
    trimmed_m = matches_df[~matches_df["match_id"].isin(["m4", "m5"])]
    trimmed_p = maps_df[~maps_df[rd.MATCH_ID_COL].isin(["m4", "m5"])]
    after = (
        swr.league_side_rate("attack", query, trimmed_m, trimmed_p),
        swr.team_overall_side_rate("T1", "attack", query, trimmed_m, trimmed_p),
        swr.team_map_side_rate("T1", "Haven", "attack", query, trimmed_m, trimmed_p, 10.0),
    )
    for x, y in zip(before, after):
        assert (x.rounds_won, x.rounds_played, x.mean if hasattr(x, "mean") else x.rate) == (
            y.rounds_won, y.rounds_played, y.mean if hasattr(y, "mean") else y.rate,
        )


def test_strictly_earlier_map_changes_the_estimate():
    # Flip side of the leakage proof: a strictly-earlier completed Haven
    # map must change the map-level counts.
    matches_df, maps_df, query = _core_tables()
    base = swr.team_map_side_rate("T1", "Haven", "attack", query, matches_df, maps_df, 10.0)
    earlier = _regulation_map(
        "m0", 0, "Haven", "T1", "T9", 13, 8, 7, 6, 3, 5, "T1",
    )
    earlier["date"] = _stamp(-1)
    earlier["status"] = "completed"
    grown_m = pd.concat([matches_df, _matches_df([_row_to_matches(earlier)])], ignore_index=True)
    grown_p = pd.concat([maps_df, _maps_df([earlier])], ignore_index=True)
    grown = swr.team_map_side_rate("T1", "Haven", "attack", query, grown_m, grown_p, 10.0)
    assert grown.rounds_won == base.rounds_won + 7
    assert grown.rounds_played == base.rounds_played + 12


# --------------------------------------------------------------------------
# _collect_validation_instances + select_k
# --------------------------------------------------------------------------


def test_collect_validation_instances_round_counts_and_phase():
    # A held-out finished map yields exactly two instances (one per side),
    # each carrying that side's derived regulation rounds for the queried
    # phase and resolved to the match's team ids.
    matches_df, maps_df, _query = _core_tables()
    folds = [(0, ["m2", "m3"], ["m1"])]
    attack_instances = swr._collect_validation_instances(
        matches_df, maps_df, "attack", folds
    )
    assert len(attack_instances) == 2
    by_team = {inst[0]: (inst[3], inst[4]) for inst in attack_instances}
    # m1: team1 seat (T1) attack 7 of 12; team2 seat (T2) attack 3 of 9.
    assert by_team["T1"] == (7, 12)
    assert by_team["T2"] == (3, 9)
    assert all(inst[2] == matches_df.loc[matches_df["match_id"] == "m1", "date"].iloc[0] for inst in attack_instances)
    assert all(inst[1] == "Haven" for inst in attack_instances)

    defense_instances = swr._collect_validation_instances(
        matches_df, maps_df, "defense", folds
    )
    by_team_d = {inst[0]: (inst[3], inst[4]) for inst in defense_instances}
    assert by_team_d["T1"] == (6, 9)
    assert by_team_d["T2"] == (5, 12)


def test_collect_validation_instances_skips_null_round_maps():
    # A validation map whose round-detail columns are null has no derived
    # round detail to validate against; it contributes no instance for
    # either side (skipped, not raised).
    rows = [
        _regulation_map("m1", 0, "Haven", "T1", "T2", 13, 8, 7, 6, 3, 5, "T1"),
        _regulation_map("m2", 0, "Bind", "T3", "T4", 13, 11, 7, 6, 6, 5, "T3"),
    ]
    for row, position in zip(rows, (0, 1)):
        row["date"] = _stamp(position)
        row["status"] = "completed"
    null_maps = rows[1].copy()
    for col in rd.ROUND_COLS:
        null_maps[col] = None
    maps_df = _maps_df([rows[0], null_maps])
    matches_df = _matches_df([_row_to_matches(rows[0]), _row_to_matches(null_maps)])
    # m2 is the validation match and its map has null round columns.
    folds = [(0, ["m1"], ["m2"])]
    instances = swr._collect_validation_instances(matches_df, maps_df, "attack", folds)
    assert instances == []

    # The same fixture with the valid map as the validation match yields
    # its two instances.
    folds2 = [(0, ["m2"], ["m1"])]
    instances2 = swr._collect_validation_instances(matches_df, maps_df, "attack", folds2)
    assert len(instances2) == 2


def _low_round_scenario():
    """Build the low-round CV scenario (a large ``k`` should win).

    Team ``L`` seeds its as-of history with two Lotus 13-11 wins at full
    attack efficiency (12 of 12 attack rounds each — a misleading small
    sample of 24 attack rounds at raw 1.0) and a balanced Haven pair,
    then plays one validation Lotus map won 13-11 with only 6 of its 12
    attack rounds (the "true" rate the misleading 1.0 sample contradicts).
    Its overall attack prior (~0.7 after inner shrinkage) is far closer
    to that 0.5 truth than the map raw 1.0 is, so a large ``k`` (heavy
    shrinkage) must score best. Every other team appears exactly once, so
    its estimate is the league rate for every ``k`` — k-neutral filler
    that satisfies the split/fold machinery's size floor without
    affecting the argmin.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 30 chronological matches
        (and maps) built by :func:`_matches_df` / :func:`_maps_df`.

    Raises:
        Nothing.
    """
    map_rows = []
    position = 0

    def add(row):
        nonlocal position
        row["date"] = _stamp(position)
        row["status"] = "completed"
        map_rows.append(row)
        position += 1

    for opp in ("ho0", "ho1"):
        add(_q_win_map(f"h{position}", _stamp(position), "Lotus", "L", opp, 12))
    for opp in ("ho2", "ho3"):
        add(_q_win_map(f"h{position}", _stamp(position), "Haven", "L", opp, 6))
    for i in range(4, 10):
        add(_q_win_map(f"w{i}", _stamp(position), "Haven", f"wa{i}", f"wb{i}", 6))
    # The signal: L loses 6 of its 12 attack rounds on a validation Lotus
    # map, contradicting its misleading 24/24 Lotus history.
    add(_q_win_map("sig", _stamp(position), "Lotus", "L", "so", 6))
    for i in range(11, 26):
        add(_q_win_map(f"v{i}", _stamp(position), "Haven", f"va{i}", f"vb{i}", 6))
    for i in range(26, 30):
        add(_q_win_map(f"t{i}", _stamp(position), "Haven", f"ta{i}", f"tb{i}", 6))
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows)


def _high_round_scenario():
    """Build the high-round CV scenario (a small ``k`` should win).

    Team ``H`` plays 200 completed matches against one-off opponents:
    100 Haven wins (10 of 12 attack rounds each, raw attack 0.8333) and
    100 Bind wins (2 of 12, raw 0.1667), interleaved chronologically.
    Both per-map attack samples are therefore backed by ~1200 as-of
    rounds — far above the ~10-round scale where shrinkage matters — while
    the overall attack prior (~0.5 after the inner shrinkage) is a muddle
    of the two very different map rates, so a small ``k`` (trusting the
    reliable per-map raw rate) must score strictly better than a large one
    (pulling every map toward the muddled prior). Opponents are unique per
    match and thus k-neutral.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 230 chronological matches
        (and maps) built by :func:`_matches_df` / :func:`_maps_df`.

    Raises:
        Nothing.
    """
    map_rows = []

    def add(map_name, attack_wins):
        position = len(map_rows)
        map_rows.append(
            _q_win_map(f"m{position:04d}", _stamp(position), map_name, "H", f"o{position}", attack_wins)
        )

    for _ in range(30):
        add("Haven", 10)
    for _ in range(30):
        add("Bind", 2)
    for _ in range(70):
        add("Haven", 10)
    for _ in range(70):
        add("Bind", 2)
    for _ in range(15):
        add("Haven", 10)
    for _ in range(15):
        add("Bind", 2)
    for row in map_rows:
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows)


def test_select_k_low_round_data_prefers_large_k():
    # A low-round (misleading) map sample should make heavy shrinkage
    # score best, so the argmin lands on the large end of the grid.
    matches_df, maps_df = _low_round_scenario()
    grid = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    best_k, scores = swr.select_k(matches_df, maps_df, "attack", grid)
    assert best_k == max(grid)
    assert scores[best_k] < scores[min(grid)]


def test_select_k_high_round_data_prefers_small_k():
    # A high-round (reliable) per-map sample should make trusting the raw
    # rate score best, so the argmin lands on the small end of the grid.
    matches_df, maps_df = _high_round_scenario()
    grid = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    best_k, scores = swr.select_k(matches_df, maps_df, "attack", grid)
    assert best_k == min(grid)
    assert scores[best_k] < scores[max(grid)]


def test_select_k_scores_dict_invariant():
    # scores_by_k has exactly one entry per grid value, best_k is a key,
    # and every score is positive (a rounds-weighted mean log loss).
    matches_df, maps_df = _low_round_scenario()
    grid = [1.0, 5.0, 50.0, 500.0]
    best_k, scores = swr.select_k(matches_df, maps_df, "attack", grid)
    assert list(scores.keys()) == grid
    assert best_k in scores
    assert all(value > 0.0 for value in scores.values())


def test_select_k_empty_k_grid_raises():
    # An empty grid cannot produce an argmin.
    matches_df, maps_df = _low_round_scenario()
    with pytest.raises(ValueError, match="k_grid"):
        swr.select_k(matches_df, maps_df, "attack", [])


def test_select_k_invalid_k_in_grid_raises():
    # A non-positive candidate in the grid is rejected up front.
    matches_df, maps_df = _low_round_scenario()
    with pytest.raises(ValueError, match="k must be"):
        swr.select_k(matches_df, maps_df, "attack", [1.0, 0.0])


def test_select_k_zero_scoreable_round_instances_raises():
    # Enough matches for the fold machinery but no usable round detail in
    # the held-out maps: select_k must raise, not return a best_k over an
    # empty held-out set.
    matches_df, _ = _low_round_scenario()
    empty_maps = _maps_df([])
    with pytest.raises(ValueError, match="zero scoreable validation instances"):
        swr.select_k(matches_df, empty_maps, "attack")


def _phase_scoping_tables():
    """Build the asymmetric fixture the phase-scoping tests use.

    Team ``X`` plays a mixed schedule on Haven and Bind with both wins
    (13-11, 10 of 12 attack rounds) and losses (11-13, 2 of 12 attack
    rounds), in an imbalance (10/6 Haven, 6/10 Bind) that leaves the two
    phases' per-map rate structures genuinely different (a win's
    attack+defence counts sum to 13, a loss's to 11 — the losses are what
    decouple the phases). The profile repeats once (as-of history +
    validation region) followed by a short test chunk. Every opponent is a
    unique team, hence k-neutral. The exact scale is a judgement call
    between fold-machinery floor and runtime: the phase-swap identity the
    test asserts is exact at any scale, so the fixture is kept small
    enough to keep the four ``select_k`` calls cheap.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 70 chronological matches
        (and maps) built by :func:`_matches_df` / :func:`_maps_df`.

    Raises:
        Nothing.
    """
    map_rows = []

    def win(map_name):
        position = len(map_rows)
        map_rows.append(_q_win_map(f"m{position:04d}", _stamp(position), map_name, "X", f"o{position}", 10))

    def loss(map_name):
        position = len(map_rows)
        map_rows.append(_q_loss_map(f"m{position:04d}", _stamp(position), map_name, "X", f"o{position}", 2))

    for _ in range(2):  # warm-up + validation profile
        for _ in range(10):
            win("Haven")
        for _ in range(6):
            loss("Haven")
        for _ in range(6):
            win("Bind")
        for _ in range(10):
            loss("Bind")
    for _ in range(6):
        win("Haven")
    for row in map_rows:
        row["status"] = "completed"
    return _matches_df([_row_to_matches(row) for row in map_rows]), _maps_df(map_rows)


def _swap_atk_def(maps_df):
    """Return ``maps_df`` with every map's attack/defence columns exchanged.

    Swapping ``team*_atk_rounds`` with ``team*_def_rounds`` per row is an
    exact phase relabel: the per-side sums and the opposing-side pairings
    are unchanged (atk and def merely exchange names), so the derived
    records of the swapped frame are the original records with the
    attack/defence round columns exchanged per seat. A phase-scoped
    estimator must therefore satisfy
    ``select_k(attack, swap(F)) == select_k(defense, F)`` — the identity
    this helper feeds the phase-scoping test.

    Args:
        maps_df: A maps-shaped frame with the four atk/def round columns.

    Returns:
        A copy of ``maps_df`` with ``team1_atk_rounds`` <-> ``team1_def_rounds``
        and ``team2_atk_rounds`` <-> ``team2_def_rounds`` value-swapped
        per row (all other columns unchanged).

    Raises:
        Nothing.
    """
    cols = [rd.TEAM1_ATK_COL, rd.TEAM1_DEF_COL, rd.TEAM2_ATK_COL, rd.TEAM2_DEF_COL]
    out = maps_df.copy()
    out.loc[:, cols] = maps_df[cols].to_numpy()[:, [1, 0, 3, 2]]
    return out


def test_select_k_phase_scoped_pooling_guard_and_swap_identity():
    # (a) Pooling guard: on an asymmetric fixture the attack and defence
    # curves must genuinely differ — an implementation that accidentally
    # pooled both phases' rounds into one shared curve would return
    # identical dicts for both calls. (b) Phase routing: relabelling every
    # map's attack/defence halves and re-running the attack CV must
    # reproduce the defence CV of the original frame exactly, and
    # vice versa. (Observed on this fixture: attack best_k 500.0, defence
    # best_k 50.0 — the two phases' argmins may disagree, though the score
    # valley near each optimum is flat to ~1e-5, so only the curve
    # difference and the swap identity are asserted as exact properties.)
    matches_df, maps_df = _phase_scoping_tables()
    grid = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    results = {}
    for phase in (swr.PHASE_ATTACK, swr.PHASE_DEFENSE):
        best_k, scores = swr.select_k(matches_df, maps_df, phase, grid)
        results[phase] = (best_k, scores)
    assert results["attack"][1] != results["defense"][1]

    swapped_maps = _swap_atk_def(maps_df)
    for phase in (swr.PHASE_ATTACK, swr.PHASE_DEFENSE):
        other = swr.PHASE_DEFENSE if phase == swr.PHASE_ATTACK else swr.PHASE_ATTACK
        best_k, scores = swr.select_k(matches_df, swapped_maps, phase, grid)
        assert (best_k, scores) == results[other]


@pytest.mark.parametrize(
    "phase", ["side", "atk", "defense-phase", "", None, 0],
)
def test_invalid_phase_raises_from_every_public_function(phase):
    # Every public phase-taking function rejects an invalid phase with
    # ValueError before any computation.
    matches_df, maps_df, query = _core_tables()
    with pytest.raises(ValueError, match="invalid phase"):
        swr.league_side_rate(phase, query, matches_df, maps_df)
    with pytest.raises(ValueError, match="invalid phase"):
        swr.team_overall_side_rate("T1", phase, query, matches_df, maps_df)
    with pytest.raises(ValueError, match="invalid phase"):
        swr.team_map_side_rate("T1", "Haven", phase, query, matches_df, maps_df, 10.0)
    with pytest.raises(ValueError, match="invalid phase"):
        swr.select_k(matches_df, maps_df, phase)


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
def test_real_data_smoke_sane_numbers_and_select_k_curves():
    # Real v1 sanity + the per-phase CV curves. Discovered at BUILD time
    # (query = latest match + 1h, most frequent team 1001):
    #   league attack 2583/5075 rate 0.5090; league defence 2492/5075
    #   rate 0.4910 (matches the module docstring's whole-dataset pool).
    #   team 1001 overall: attack 267/496 raw 0.5383 mean 0.5356;
    #   defence 278/533 raw 0.5216 mean 0.5190.
    #   team 1001 Haven at DEFAULT_K=10: attack 39/70 mean 0.5545;
    #   defence 35/73 mean 0.4842.
    #   select_k (default grid): attack best_k=1000.0 with scores
    #   {1:0.7315, 2:0.7241, 5:0.7137, 10:0.7065, 20:0.7007, 50:0.6959,
    #    100:0.6943, 200:0.6936, 500:0.6933, 1000:0.6932}; defence
    #   best_k=500.0 with scores {1:0.7331, 2:0.7251, 5:0.7142,
    #    10:0.7067, 20:0.7009, 50:0.6966, 100:0.6954, 200:0.6950,
    #    500:0.6950, 1000:0.6950}.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    for phase in (swr.PHASE_ATTACK, swr.PHASE_DEFENSE):
        league = swr.league_side_rate(phase, query, matches_df, maps_df)
        assert league.rounds_played > 0
        assert 0.0 < league.rate < 1.0

    map_name = maps_df["map_name"].value_counts().idxmax()
    for phase in (swr.PHASE_ATTACK, swr.PHASE_DEFENSE):
        overall = swr.team_overall_side_rate(team_id, phase, query, matches_df, maps_df)
        assert overall.rounds_played > 0
        assert 0.0 < overall.mean < 1.0
        shrunk = swr.team_map_side_rate(
            team_id, map_name, phase, query, matches_df, maps_df, swr.DEFAULT_K
        )
        assert shrunk.rounds_played <= overall.rounds_played
        assert 0.0 < shrunk.mean < 1.0
        assert shrunk.alpha > 0.0 and shrunk.beta > 0.0 and shrunk.variance > 0.0

    best_attack, _scores_attack = swr.select_k(matches_df, maps_df, "attack")
    best_defense, _scores_defense = swr.select_k(matches_df, maps_df, "defense")
    assert best_attack in swr.DEFAULT_K_GRID
    assert best_defense in swr.DEFAULT_K_GRID
    assert best_attack == 1000.0
    assert best_defense == 500.0


# --------------------------------------------------------------------------
# batched attack-side diff parity (task 052)
# --------------------------------------------------------------------------


def test_batched_attack_side_win_rate_diff_bit_exact_parity():
    # The batched attack path reproduces, element-for-element, the
    # looped single-row team_map_side_rate means per row (no
    # tolerance), including a wholly unseen side (Z) whose whole
    # hierarchy degrades through the league/inner priors exactly as the
    # single-row path degrades it.
    matches_df, maps_df, query = _core_tables()
    d1, d2, d3 = _stamp(0), _stamp(1), _stamp(2)
    rows_df = pd.DataFrame(
        [
            {"team1_id": "T1", "team2_id": "T2", "map_name": "Haven", "date": d1},
            {"team1_id": "T1", "team2_id": "T2", "map_name": "Haven", "date": d2},
            {"team1_id": "T1", "team2_id": "T3", "map_name": "Haven", "date": d3},
            {"team1_id": "T1", "team2_id": "T4", "map_name": "Bind", "date": d3},
            {"team1_id": "T1", "team2_id": "T4", "map_name": "Bind", "date": query},
            {"team1_id": "Z", "team2_id": "T1", "map_name": "Haven", "date": query},
        ]
    )
    expected = np.zeros(len(rows_df))
    for i, row in enumerate(rows_df.itertuples(index=False)):
        mean_a = swr.team_map_side_rate(
            row.team1_id,
            row.map_name,
            swr.PHASE_ATTACK,
            row.date,
            matches_df,
            maps_df,
            swr.BEST_K_ATTACK,
        ).mean
        mean_b = swr.team_map_side_rate(
            row.team2_id,
            row.map_name,
            swr.PHASE_ATTACK,
            row.date,
            matches_df,
            maps_df,
            swr.BEST_K_ATTACK,
        ).mean
        expected[i] = mean_a - mean_b
    got = swr.batched_attack_side_win_rate_diff(rows_df, matches_df, maps_df)
    assert got.shape == (len(rows_df),)
    assert np.array_equal(got, expected)

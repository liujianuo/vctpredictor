"""Tests for the team-map first-blood differential (M38.4).

Covers the two estimators (``team_overall_first_blood_rate`` /
``team_map_first_blood_rate``) and their Beta-posterior dataclasses, the
two hard invariants (a null ``first_kills``/``first_deaths`` in a
resolved roster raises; a per-map ``sum(first_kills) !=
sum(first_deaths)`` over the full group raises), the exact-``0.5``
identity (the inner prior is the literal
:data:`LEAGUE_FIRST_BLOOD_RATE` constant, never re-derived from data),
the shrinkage-behaviour contracts (zero history -> full shrinkage, small
sample -> pulled toward the prior, large sample -> close to raw,
monotonic in ``k``), the two-level hierarchy's distinguishing behaviour
(the map-level ``prior`` is the *shrunk overall* mean from
``team_overall_first_blood_rate``, not ``0.5`` and not the raw
team-overall rate), the seat-orientation correctness (a team queried
from both the ``team1`` and the ``team2`` seat gets the right ``(FK,
FD)`` pairs — the test that would catch a seat-swap bug), the team-name
resolution contract (a ``team_name`` matching neither side raises; a
missing ``player_map_stats`` group is skipped and counted in
``maps_skipped``), map-name normalization, the leakage-safety proof
(maps dated at/after the cutoff never enter any estimator), ``select_k``
(a ``best_k`` from the grid with non-negative finite trial-weighted
scores; empty grid / invalid candidate ``k`` / training region too small
/ zero scoreable instances all raise), invalid-``k`` rejection from both
public functions, and skip-guarded real ``data/v1`` smoke tests
recording the conservation identity, the 2-excluded-maps fact and the
real CV argmin. Test fixtures build internally-valid first-blood groups
(the per-map conservation invariant holds by construction: a side's pair
``(fk, fd)`` is mirrored as the opponent's ``(fd, fk)``, so the two
teams' sums always conserve); a non-test helper raises ``ValueError`` on
an invalid row so a mis-typed fixture fails at construction, never
silently mid-test.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import first_blood as fb
from utils import asof

_MATCHES_COLS = [
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "team1_name",
    "team2_name",
    "status",
]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]
_PMS_COLS = [
    "match_id",
    "map_index",
    "player_name",
    "team_name",
    "first_kills",
    "first_deaths",
]


def _stamp(index):
    """Return the ISO timestamp of fixture position ``index``.

    Every mass-produced fixture places its matches one hour apart from a
    fixed 2026 base date, so chronological order == list order and the
    query date for "after everything" is trivially derivable. Negative
    indices produce earlier timestamps, so a "strictly before the first
    real event" map can be added without renumbering.

    Args:
        index: The 0-based fixture position (may be negative).

    Returns:
        An ISO-8601 timestamp string one hour after the previous
        position (``2026-01-01T00:00:00`` + ``index`` hours).

    Raises:
        Nothing.
    """
    return (
        pd.Timestamp("2026-01-01T00:00:00") + pd.Timedelta(hours=index)
    ).isoformat()


def _matches_df(rows):
    """Build a matches table with the fixed M8 + names column set.

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
    ``winner``/``map_name`` columns the as-of layer and the map filter
    read).

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


def _pms_df(rows):
    """Build a player_map_stats table with the column set this module reads.

    Mirrors :func:`_matches_df` for the player-map stats side; a
    ``None`` (or ``float("nan")``) ``first_kills``/``first_deaths``
    value becomes a null cell in the resulting frame, which is exactly
    how the no-nulls invariant tests inject a null.

    Args:
        rows: A list of dicts, one per player-map row; each must carry
            the keys in :data:`_PMS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_PMS_COLS` columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_PMS_COLS)


def _split(total, parts):
    """Split a non-negative total into ``parts`` non-negative ints summing to it.

    The per-player distribution helper behind :func:`_roster_rows`: a
    team's roster-summed ``(first_kills, first_deaths)`` totals are
    spread across the roster rows with the remainder absorbed by the
    last row, so any integer total can be carried by any positive row
    count.

    Args:
        total: The non-negative integer to distribute.
        parts: The number of parts (must be ``>= 1``).

    Returns:
        A list of ``parts`` non-negative ints summing to ``total``.

    Raises:
        ValueError: If ``parts`` is not a positive integer or ``total``
            is negative.
    """
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    base, remainder = divmod(total, parts)
    return [base + 1] * remainder + [base] * (parts - remainder)


def _roster_rows(match_id, map_index, team_name, fk, fd, players=5, prefix="p"):
    """Build ``players`` player_map_stats row dicts for one team's map.

    Each row is one player's line for ``(match_id, map_index)`` on
    ``team_name``; the roster's ``first_kills`` sum to ``fk`` and its
    ``first_deaths`` sum to ``fd`` (the level-0 aggregation shape the
    module sums). The default is the v1-universal 5-player roster; a
    smaller/larger count is allowed (the module never hardcodes five).

    Args:
        match_id: The fixture match id.
        map_index: The map index within the match.
        team_name: The team's display name (the ``team_name`` column).
        fk: The roster-summed ``first_kills`` total to distribute.
        fd: The roster-summed ``first_deaths`` total to distribute.
        players: The number of roster rows (default 5).
        prefix: A per-call prefix making player names unique across
            teams sharing one name set (default ``"p"``).

    Returns:
        A list of ``players`` row dicts keyed by :data:`_PMS_COLS`.

    Raises:
        ValueError: If ``players`` is ``< 1`` or ``fk``/``fd`` are
            negative (propagated from :func:`_split`).
    """
    fk_splits = _split(fk, players)
    fd_splits = _split(fd, players)
    return [
        {
            "match_id": match_id,
            "map_index": map_index,
            "player_name": f"{prefix}{i}",
            "team_name": team_name,
            "first_kills": fk_splits[i],
            "first_deaths": fd_splits[i],
        }
        for i in range(players)
    ]


def _event(event):
    """Expand one map-event dict into its matches/maps/player_map_stats parts.

    A single fixture map: ``event`` carries the match identity and side
    names plus each side's roster-summed ``(fk, fd)`` pair (e.g.
    ``"t1_fk"``/``"t1_fd"`` for team1). The two sides' pairs are
    expected to conserve (``t1_fk + t2_fk == t1_fd + t2_fd``) — the
    caller's responsibility; the mirror convention (team2 gets
    ``(fd, fk)`` of team1) satisfies it automatically. Both teams
    receive ``players`` (default 5) roster rows each via
    :func:`_roster_rows`.

    Args:
        event: A dict with keys ``match_id``, ``date``, ``team1_id``,
            ``team2_id``, ``team1_name``, ``team2_name``, ``t1_fk``,
            ``t1_fd``, ``t2_fk``, ``t2_fd``; optional ``map_name``
            (default ``"Haven"``), ``map_index`` (default 0),
            ``players`` (default 5), ``winner`` (default team1 id).

    Returns:
        A ``(match_row, map_row, pms_rows)`` triple ready for
        :func:`_matches_df` / :func:`_maps_df` / :func:`_pms_df`.

    Raises:
        Nothing.
    """
    match_id = event["match_id"]
    map_index = event.get("map_index", 0)
    map_name = event.get("map_name", "Haven")
    players = event.get("players", 5)
    winner = event.get("winner", event["team1_id"])
    match_row = {
        "match_id": match_id,
        "date": event["date"],
        "team1_id": event["team1_id"],
        "team2_id": event["team2_id"],
        "team1_name": event["team1_name"],
        "team2_name": event["team2_name"],
        "status": "completed",
    }
    map_row = {
        "match_id": match_id,
        "map_index": map_index,
        "map_name": map_name,
        "team1_score": 13,
        "team2_score": 8,
        "winner": winner,
    }
    pms_rows = _roster_rows(
        match_id, map_index, event["team1_name"], event["t1_fk"],
        event["t1_fd"], players=players, prefix=f"{event['team1_name']}p",
    )
    pms_rows += _roster_rows(
        match_id, map_index, event["team2_name"], event["t2_fk"],
        event["t2_fd"], players=players, prefix=f"{event['team2_name']}p",
    )
    return match_row, map_row, pms_rows


def _scenario(events):
    """Materialise an event list into the three fixture DataFrames.

    Runs every event through :func:`_event` and concatenates the
    matches/maps/player-map-stats parts into the three frames the module
    consumes.

    Args:
        events: A list of event dicts in :func:`_event`'s shape.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple built by
        :func:`_matches_df` / :func:`_maps_df` / :func:`_pms_df`.

    Raises:
        Nothing (pandas ``ValueError`` on a malformed event surfaces as
        is).
    """
    match_rows = []
    map_rows = []
    pms_rows = []
    for event in events:
        match_row, map_row, rows = _event(event)
        match_rows.append(match_row)
        map_rows.append(map_row)
        pms_rows.extend(rows)
    return (
        _matches_df(match_rows),
        _maps_df(map_rows),
        _pms_df(pms_rows),
    )


def _core_tables():
    """Build the shared multi-team fixture with both seats and leakage rows.

    A six-map fixture exercising (a) both seats for the queried team,
    (b) two map names, (c) an unrelated-match row whose presence keeps
    the queried team's history a strict subset of the frame, and (d) the
    equal/after leakage rows. Every side's pair is mirrored by its
    opponent's ``(fd, fk)`` so per-map conservation holds by
    construction:

    - ``m0``: U1 (team1) beats U2 on Haven, neither side queried.
    - ``m1``: T1 (team1, name Alpha) vs T2 (Beta) on Haven; T1 ``(11,
      10)`` (raw 11/21 ~ 0.524).
    - ``m2``: T3 (team1, Gamma) vs T1 (team2, Alpha) on Haven; T1 ``(7,
      13)`` (raw 7/20 = 0.35; T1's seat is team2 here).
    - ``m3``: T1 (team1, Alpha) vs T4 (Delta) on Bind; T1 ``(12, 8)``.
    - ``m4``: a Haven map dated exactly at the query cutoff, T1 ``(30,
      0)`` (excluded).
    - ``m5``: a Haven map dated after the cutoff, T1 ``(30, 0)``
      (excluded).

    T1's own as-of history (m1-m3): 3 maps, ``first_kills = 11 + 7 + 12
    = 30`` and ``first_deaths = 10 + 13 + 8 = 31`` overall; on Haven
    ``(18, 23)`` over 2 maps; on Bind ``(12, 8)`` over 1 map.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df, query_date)``
        tuple; ``query_date`` is exactly the date of ``m4`` (so
        ``m0``-``m3`` are strictly before it and ``m4``/``m5`` are the
        leakage rows).

    Raises:
        Nothing.
    """
    t0 = _stamp(0)
    t1 = _stamp(1)
    t2 = _stamp(2)
    t3 = _stamp(3)
    query = _stamp(4)
    t5 = _stamp(5)
    events = [
        # m0: unrelated teams U1/U2 on Haven (T1 untouched).
        {
            "match_id": "m0", "date": t0, "team1_id": "U1", "team2_id": "U2",
            "team1_name": "Uno", "team2_name": "Dos",
            "t1_fk": 6, "t1_fd": 7, "t2_fk": 7, "t2_fd": 6,
        },
        # m1: T1 as team1 wins Haven, T1 pair (11, 10).
        {
            "match_id": "m1", "date": t1, "team1_id": "T1", "team2_id": "T2",
            "team1_name": "Alpha", "team2_name": "Beta",
            "t1_fk": 11, "t1_fd": 10, "t2_fk": 10, "t2_fd": 11,
        },
        # m2: T1 as team2 loses Haven, T1 pair (7, 13).
        {
            "match_id": "m2", "date": t2, "team1_id": "T3", "team2_id": "T1",
            "team1_name": "Gamma", "team2_name": "Alpha",
            "t1_fk": 13, "t1_fd": 7, "t2_fk": 7, "t2_fd": 13,
        },
        # m3: T1 as team1 wins Bind, T1 pair (12, 8).
        {
            "match_id": "m3", "date": t3, "team1_id": "T1", "team2_id": "T4",
            "team1_name": "Alpha", "team2_name": "Delta",
            "map_name": "Bind",
            "t1_fk": 12, "t1_fd": 8, "t2_fk": 8, "t2_fd": 12,
        },
        # m4/m5: leakage rows (equal to / after the query date).
        {
            "match_id": "m4", "date": query, "team1_id": "T1", "team2_id": "T5",
            "team1_name": "Alpha", "team2_name": "Echo",
            "t1_fk": 30, "t1_fd": 0, "t2_fk": 0, "t2_fd": 30,
        },
        {
            "match_id": "m5", "date": t5, "team1_id": "T1", "team2_id": "T6",
            "team1_name": "Alpha", "team2_name": "Foxtrot",
            "t1_fk": 30, "t1_fd": 0, "t2_fk": 0, "t2_fd": 30,
        },
    ]
    matches_df, maps_df, pms_df = _scenario(events)
    return matches_df, maps_df, pms_df, query


def _team_tables(segments, team_id="T", team_name="Alpha"):
    """Build a recurring-team history from ``(map_name, games, fk, fd)`` segments.

    Produces one completed map per game, in segment order, with the
    queried team always ``team1`` (orientation variation is covered by
    :func:`_core_tables`) against a unique one-off opponent whose pair
    mirrors the queried team's (so per-map conservation holds). Dates
    are one hour apart; the returned query date is one hour after the
    last map so every row is strictly before the cutoff.

    Args:
        segments: An iterable of ``(map_name, games, fk, fd)`` tuples;
            each expands into ``games`` maps on that map name, each with
            the queried team's roster-summed ``(fk, fd)`` pair.
        team_id: The queried team's id on the ``team1`` side (default
            ``"T"``).
        team_name: The queried team's display name (default
            ``"Alpha"``).

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df, query_date)``
        tuple; ``query_date`` is an ISO string one hour after the final
        map.

    Raises:
        Nothing.
    """
    events = []
    position = 0
    for map_name, games, fk, fd in segments:
        for _ in range(games):
            events.append(
                {
                    "match_id": f"t{position:04d}",
                    "date": _stamp(position),
                    "team1_id": team_id,
                    "team2_id": f"o{position:04d}",
                    "team1_name": team_name,
                    "team2_name": f"Opp{position:04d}",
                    "map_name": map_name,
                    "t1_fk": fk,
                    "t1_fd": fd,
                    "t2_fk": fd,
                    "t2_fd": fk,
                }
            )
            position += 1
    matches_df, maps_df, pms_df = _scenario(events)
    return matches_df, maps_df, pms_df, _stamp(position)


def _strong_tables():
    """Build the strong-team fixture the hierarchy tests use.

    Team ``S`` (always ``team1``) plays 30 Haven maps, each with pair
    ``(12, 8)`` (raw Haven rate 0.6): overall ``first_kills = 360``,
    ``first_deaths = 240`` (raw 0.6; shrunk mean with
    :data:`DEFAULT_OVERALL_K` = 50 exactly ``(360 + 25) / 650 =
    0.5923077...``), and it has *no* Ascent maps — the fixture the
    zero-map-history full-shrinkage and prior-chain tests assert
    against.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df, query_date)``
        tuple.

    Raises:
        Nothing.
    """
    return _team_tables([("Haven", 30, 12, 8)], team_id="S", team_name="Strong")


def _ascent_strong_tables():
    """Build the strong-team fixture with a thin sampled second map.

    Team ``S`` plays 30 Haven maps at pair ``(12, 8)`` *and* 2 Ascent
    maps at pair ``(2, 8)`` (raw Ascent rate 0.2): the Ascent sample
    (n = 20 trials) is the thin map sample the roadmap's differential
    framing targets, while the overall sample sums to
    ``first_kills = 360 + 4 = 364`` and ``first_deaths = 240 + 16 =
    256`` over 32 maps (raw 364/620 = 0.5871; shrunk mean with
    :data:`DEFAULT_OVERALL_K` = 50 exactly ``(364 + 25) / 670 =
    0.580597...``). With the map-level prior that shrunk overall mean
    (0.5806) well above Ascent's raw 0.2, the map-level shrinkage must
    pull the estimate *up* toward the team's general first-blood
    tendency.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df, query_date)``
        tuple.

    Raises:
        Nothing.
    """
    return _team_tables(
        [("Haven", 30, 12, 8), ("Ascent", 2, 2, 8)],
        team_id="S",
        team_name="Strong",
    )


def _cv_scenario(n_matches=70):
    """Build the walk-forward CV scenario.

    Team ``L`` (always ``team1``) plays ``n_matches`` completed matches
    against unique one-off opponents, alternating Haven maps at pair
    ``(12, 8)`` (raw 0.6) and Bind maps at pair ``(5, 15)`` (raw 0.25).
    Every opponent appears exactly once, so its own as-of history is
    empty and its estimate is the league prior for every ``k`` —
    k-neutral filler that satisfies the split/fold machinery's size
    floor without steering the argmin (the same convention
    ``test_side_win_rate.py``'s CV scenarios use). Each map carries a
    full valid group for both sides, so every held-out validation map
    yields both sides' instances.

    Args:
        n_matches: The number of chronological matches to build (default
            70; the default fold defaults need ``>= ~30`` for
            ``split_matches`` and a few dozen more for a real fold
            sweep).

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple.

    Raises:
        Nothing.
    """
    events = []
    for i in range(n_matches):
        if i % 2 == 0:
            fk, fd, map_name = 12, 8, "Haven"
        else:
            fk, fd, map_name = 5, 15, "Bind"
        events.append(
            {
                "match_id": f"m{i:04d}",
                "date": _stamp(i),
                "team1_id": "L",
                "team2_id": f"o{i:04d}",
                "team1_name": "Lima",
                "team2_name": f"Oscar{i:04d}",
                "map_name": map_name,
                "t1_fk": fk,
                "t1_fd": fd,
                "t2_fk": fd,
                "t2_fd": fk,
            }
        )
    matches_df, maps_df, pms_df = _scenario(events)
    return matches_df, maps_df, pms_df


# --------------------------------------------------------------------------
# the two hard invariants
# --------------------------------------------------------------------------


def test_null_first_kills_in_roster_raises_value_error():
    # A null first_kills value in the resolved roster must fail loudly
    # (the no-nulls hard invariant), not be silently summed as zero.
    matches, maps, pms, _query = _core_tables()
    pms.loc[
        (pms["match_id"] == "m1") & (pms["team_name"] == "Alpha"),
        "first_kills",
    ] = None
    with pytest.raises(ValueError, match="null"):
        fb.team_overall_first_blood_rate("T1", _query, matches, maps, pms)


def test_conservation_violation_raises_value_error():
    # A map whose full-group sums violate sum(first_kills) ==
    # sum(first_deaths) (here 10 + 5 = 15 vs 5 + 5 = 10) must raise the
    # conservation ValueError naming the map, not silently proceed.
    match = {
        "match_id": "m1",
        "date": _stamp(0),
        "team1_id": "T1",
        "team2_id": "T2",
        "team1_name": "Alpha",
        "team2_name": "Beta",
        "t1_fk": 10,
        "t1_fd": 5,
        "t2_fk": 5,
        "t2_fd": 5,
    }
    matches, maps, pms = _scenario([match])
    query = _stamp(1)
    with pytest.raises(ValueError, match="conservation") as excinfo:
        fb.team_overall_first_blood_rate("T1", query, matches, maps, pms)
    assert "m1" in str(excinfo.value)


# --------------------------------------------------------------------------
# the exact-0.5 identity + the inner prior
# --------------------------------------------------------------------------


def test_overall_prior_is_literal_half_constant_not_own_or_pool_rate():
    # The inner-level prior is the module constant 0.5, never re-derived:
    # a team whose whole history is one 8-2 map (raw 0.8, far from 0.5)
    # must still report prior == LEAGUE_FIRST_BLOOD_RATE == 0.5 exactly
    # while raw_rate == 0.8 — a self-shrinking implementation (prior =
    # own raw rate) or a pool-mean one would both disagree here.
    matches, maps, pms, query = _team_tables([("Haven", 1, 8, 2)])
    result = fb.team_overall_first_blood_rate("T", query, matches, maps, pms)
    assert result.maps_used == 1
    assert (result.first_kills, result.first_deaths) == (8, 2)
    assert fb.LEAGUE_FIRST_BLOOD_RATE == 0.5
    assert result.prior == fb.LEAGUE_FIRST_BLOOD_RATE == 0.5
    assert result.raw_rate == pytest.approx(0.8)
    assert result.raw_rate != result.prior


# --------------------------------------------------------------------------
# team_overall_first_blood_rate: shrinkage behaviour
# --------------------------------------------------------------------------


def test_overall_zero_history_full_shrinkage_to_league_prior():
    # An unseen team has no maps; the inner posterior degrades to
    # mean == raw_rate == the 0.5 structural prior exactly, with empty
    # maps_used/maps_skipped counts.
    matches, maps, pms, query = _core_tables()
    result = fb.team_overall_first_blood_rate("UNSEEN", query, matches, maps, pms)
    assert (result.first_kills, result.first_deaths) == (0, 0)
    assert (result.maps_used, result.maps_skipped) == (0, 0)
    assert result.prior == 0.5
    assert result.raw_rate == pytest.approx(0.5)
    assert result.mean == pytest.approx(0.5)


def test_overall_few_trials_shrinks_toward_league_prior():
    # A team whose only history is one 8-2 map (raw 0.8): with k=10 the
    # shrunk mean (8 + 5) / 20 = 0.65 sits strictly between the 0.5
    # prior and the raw rate (exact-formula check).
    matches, maps, pms, query = _team_tables([("Haven", 1, 8, 2)])
    result = fb.team_overall_first_blood_rate("T", query, matches, maps, pms, k=10.0)
    assert (result.first_kills, result.first_deaths) == (8, 2)
    assert result.raw_rate == pytest.approx(0.8)
    assert result.mean == pytest.approx(0.65)
    assert 0.5 < result.mean < result.raw_rate


def test_overall_many_trials_close_to_raw_rate():
    # 100 maps at pair (12, 8) (raw 0.6, 2000 trials) swamp the fixed k,
    # so the shrunk mean is within a small tolerance of the raw rate
    # even though the 0.5 prior differs from it.
    matches, maps, pms, query = _team_tables([("Haven", 100, 12, 8)])
    result = fb.team_overall_first_blood_rate("T", query, matches, maps, pms)
    assert result.maps_used == 100
    assert (result.first_kills, result.first_deaths) == (1200, 800)
    assert result.raw_rate == pytest.approx(0.6)
    assert abs(result.mean - result.raw_rate) < 0.01
    assert result.variance > 0.0


def test_overall_monotonic_in_k_toward_prior():
    # Holding the team sample fixed, mean moves monotonically toward the
    # 0.5 prior as k grows (raw 0.8 > prior, so it decreases), and at
    # heavy k it lands close to the prior.
    matches, maps, pms, query = _team_tables([("Haven", 4, 8, 2)])
    means = [
        fb.team_overall_first_blood_rate("T", query, matches, maps, pms, k=k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[0] < 0.8
    assert abs(means[-1] - 0.5) < 0.02


# --------------------------------------------------------------------------
# team_map_first_blood_rate: the two-level hierarchy
# --------------------------------------------------------------------------


def test_map_zero_history_full_shrinkage_to_shrunk_overall():
    # Zero maps on the queried map: mean == prior exactly, where prior is
    # the team's *shrunk overall* rate (inner-level output), not the 0.5
    # league constant and not the raw team-overall rate.
    matches, maps, pms, query = _strong_tables()
    overall = fb.team_overall_first_blood_rate("S", query, matches, maps, pms)
    assert overall.raw_rate == pytest.approx(0.6)
    assert overall.mean == pytest.approx((360.0 + 25.0) / 650.0)  # shrunk below raw
    result = fb.team_map_first_blood_rate("S", "Ascent", query, matches, maps, pms, k=5.0)
    assert (result.first_kills, result.first_deaths) == (0, 0)
    assert (result.maps_used, result.maps_skipped) == (0, 0)
    assert result.prior == pytest.approx(overall.mean)
    assert result.prior != pytest.approx(0.5)  # clearly not the league constant
    assert result.prior != pytest.approx(overall.raw_rate)
    assert result.raw_rate == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)


def test_map_prior_is_shrunk_overall_not_raw_or_league():
    # The hierarchy's distinguishing behaviour asserted against the raw
    # and league alternatives at once: with k=5 the map-level mean must
    # interpolate between Ascent's raw 0.2 and the shrunk overall prior
    # (~0.5806), and the prior must equal team_overall_first_blood_rate
    # at the same date. A flat single-level shrinkage toward the 0.5
    # league constant would also pull the estimate up, but to a
    # different value; a prior of the raw overall 0.5871 would disagree
    # exactly as asserted.
    matches, maps, pms, query = _ascent_strong_tables()
    overall = fb.team_overall_first_blood_rate("S", query, matches, maps, pms)
    assert overall.first_kills == 364
    assert overall.first_deaths == 256
    assert overall.raw_rate == pytest.approx(364.0 / 620.0)
    assert overall.mean == pytest.approx((364.0 + 25.0) / 670.0)
    result = fb.team_map_first_blood_rate("S", "Ascent", query, matches, maps, pms, k=5.0)
    assert result.maps_used == 2
    assert (result.first_kills, result.first_deaths) == (4, 16)
    assert result.raw_rate == pytest.approx(0.2)
    assert result.prior == pytest.approx(overall.mean)  # shrunk overall
    assert result.prior != pytest.approx(0.5)
    assert result.prior != pytest.approx(overall.raw_rate)
    assert result.raw_rate < result.mean < result.prior  # pulled up toward strength


def test_map_few_maps_shrinks_toward_prior():
    # The map-level sample (30 Haven maps at pair (12, 8), raw 0.6)
    # against a shrunk overall prior of exactly (360 + 25)/650: the
    # k*prior pseudo-trials pull the mean strictly below the raw rate
    # toward the prior (exact-formula check with k=5).
    matches, maps, pms, query = _strong_tables()
    result = fb.team_map_first_blood_rate("S", "Haven", query, matches, maps, pms, k=5.0)
    assert result.maps_used == 30
    assert (result.first_kills, result.first_deaths) == (360, 240)
    assert result.raw_rate == pytest.approx(0.6)
    prior = (360.0 + 25.0) / 650.0
    assert result.prior == pytest.approx(prior)
    assert result.mean == pytest.approx((360.0 + 5.0 * prior) / (600.0 + 5.0))
    assert prior < result.mean < result.raw_rate


def test_map_monotonic_in_k_toward_prior():
    # Holding the map sample fixed, mean moves monotonically toward the
    # shrunk-overall prior as k grows (raw 0.6 > prior ~0.5923, so it
    # decreases), bounded below by the prior.
    matches, maps, pms, query = _strong_tables()
    means = [
        fb.team_map_first_blood_rate("S", "Haven", query, matches, maps, pms, k=k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[0] < 0.6
    assert abs(means[-1] - (360.0 + 25.0) / 650.0) < 0.02


def test_map_normalizes_map_name():
    # A caller passing " breeze " must match the stored "Breeze" via
    # normalize_map_name rather than depending on exact case/whitespace.
    matches, maps, pms, query = _team_tables(
        [("Breeze", 5, 12, 8), ("Haven", 5, 5, 15)]
    )
    result = fb.team_map_first_blood_rate("T", " breeze ", query, matches, maps, pms, k=5.0)
    assert result.maps_used == 5
    assert (result.first_kills, result.first_deaths) == (60, 40)
    assert result.raw_rate == pytest.approx(0.6)


# --------------------------------------------------------------------------
# name resolution: fail-loud mismatch + skip-and-count
# --------------------------------------------------------------------------


def test_team_name_mismatch_raises_value_error():
    # A player_map_stats team_name that matches neither side of its match
    # must fail loudly (the roadmap's ambiguity-4 reconciliation guard),
    # even when the group's own sums conserve. The base group's two sides
    # carry mirror sums (5 + 3 == 3 + 5 == 8), and the extra GHOST row is
    # given equal sums (4 == 4) so the group total conserves (12 == 12)
    # and the name guard is what fires.
    match = {
        "match_id": "m1",
        "date": _stamp(0),
        "team1_id": "T1",
        "team2_id": "T2",
        "team1_name": "Alpha",
        "team2_name": "Beta",
        "t1_fk": 5,
        "t1_fd": 3,
        "t2_fk": 3,
        "t2_fd": 5,
        "players": 1,
    }
    matches, maps, pms = _scenario([match])
    query = _stamp(1)
    ghost = {
        "match_id": "m1",
        "map_index": 0,
        "player_name": "g",
        "team_name": "GHOST",
        "first_kills": 4,
        "first_deaths": 4,
    }
    pms = pd.concat([pms, _pms_df([ghost])], ignore_index=True)
    with pytest.raises(ValueError, match="matching neither"):
        fb.team_overall_first_blood_rate("T1", query, matches, maps, pms)


def test_map_without_player_rows_skipped_and_counted():
    # m1 has a full roster, m2 has none (the real 242/244 gap case): m2
    # is skipped and counted in maps_skipped, not an error, and only
    # m1's pair feeds the estimate.
    events = [
        {
            "match_id": "m1", "date": _stamp(0), "team1_id": "T", "team2_id": "B",
            "team1_name": "Alpha", "team2_name": "Beta",
            "t1_fk": 12, "t1_fd": 8, "t2_fk": 8, "t2_fd": 12,
        },
        {
            "match_id": "m2", "date": _stamp(1), "team1_id": "T", "team2_id": "C",
            "team1_name": "Alpha", "team2_name": "Gamma",
            "t1_fk": 1, "t1_fd": 1, "t2_fk": 1, "t2_fd": 1,
        },
    ]
    matches, maps, pms = _scenario(events)
    # Drop m2's player rows entirely (no pms group for its key).
    pms = pms[pms["match_id"] != "m2"]
    query = _stamp(2)
    result = fb.team_overall_first_blood_rate("T", query, matches, maps, pms)
    assert result.maps_used == 1
    assert result.maps_skipped == 1
    assert (result.first_kills, result.first_deaths) == (12, 8)
    assert result.raw_rate == pytest.approx(0.6)


def test_map_with_only_opponent_rows_skipped():
    # The map's group exists but holds rows only for the opponent (with
    # conserving equal sums 10 == 10): the queried team's roster is empty
    # -> skip-and-count, not an error, and the overall result degrades
    # to full shrinkage toward the 0.5 prior.
    events = [
        {
            "match_id": "m1", "date": _stamp(0), "team1_id": "T", "team2_id": "B",
            "team1_name": "Alpha", "team2_name": "Beta",
            "t1_fk": 1, "t1_fd": 1, "t2_fk": 10, "t2_fd": 10,
        }
    ]
    matches, maps, pms = _scenario(events)
    pms = pms[pms["team_name"] == "Beta"]
    query = _stamp(1)
    result = fb.team_overall_first_blood_rate("T", query, matches, maps, pms)
    assert (result.first_kills, result.first_deaths) == (0, 0)
    assert (result.maps_used, result.maps_skipped) == (0, 1)
    assert result.mean == pytest.approx(0.5)
    assert result.raw_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# seat resolution + leakage
# --------------------------------------------------------------------------


def test_seat_resolution_correct_for_both_orientations():
    # T1 is team1 on m1/m3 and team2 on m2; the seat resolution must pick
    # T1's own roster rows per map or the sums would be wrong. On Haven
    # (m1 as team1 with pair (11, 10), m2 as team2 with pair (7, 13))
    # T1's sample is 2 maps summing to (18, 23); over all maps (adding
    # m3's Bind (12, 8)) it is 3 maps summing to (30, 31). A seat bug
    # always taking the team1-side rows would give Haven (11 + 13, 10 +
    # 7) = (24, 17); a mirror-flip bug would swap each pair.
    matches, maps, pms, query = _core_tables()
    haven = fb.team_map_first_blood_rate("T1", "Haven", query, matches, maps, pms, k=5.0)
    assert (haven.first_kills, haven.first_deaths) == (18, 23)
    assert haven.maps_used == 2
    overall = fb.team_overall_first_blood_rate("T1", query, matches, maps, pms)
    assert (overall.first_kills, overall.first_deaths) == (30, 31)
    assert overall.maps_used == 3
    # The private aggregator exposes the raw sums for both filters.
    assert fb._team_first_bloods("T1", query, matches, maps, pms) == (30, 31, 3, 0)
    assert fb._team_first_bloods(
        "T1", query, matches, maps, pms, map_name="Haven"
    ) == (18, 23, 2, 0)
    assert fb._team_first_bloods(
        "T1", query, matches, maps, pms, map_name="Bind"
    ) == (12, 8, 1, 0)
    assert fb._team_first_bloods(
        "T1", query, matches, maps, pms, map_name="Ascent"
    ) == (0, 0, 0, 0)


def test_equal_and_after_maps_never_enter_any_estimator():
    # The leakage proof across both estimators: m4 (exactly at the
    # cutoff) and m5 (after it) must not contribute, and trimming them
    # leaves every estimate byte-for-byte unchanged. (T1's Haven maps
    # m4/m5 both carry pair (30, 0), so an inclusive-<= bug would change
    # the Haven sample from (18, 23) to (78, 23).)
    matches, maps, pms, query = _core_tables()
    before = (
        fb.team_overall_first_blood_rate("T1", query, matches, maps, pms),
        fb.team_map_first_blood_rate("T1", "Haven", query, matches, maps, pms, k=5.0),
    )
    trimmed_m = matches[~matches["match_id"].isin(["m4", "m5"])]
    trimmed_p = maps[~maps["match_id"].isin(["m4", "m5"])]
    trimmed_s = pms[~pms["match_id"].isin(["m4", "m5"])]
    after = (
        fb.team_overall_first_blood_rate("T1", query, trimmed_m, trimmed_p, trimmed_s),
        fb.team_map_first_blood_rate(
            "T1", "Haven", query, trimmed_m, trimmed_p, trimmed_s, k=5.0
        ),
    )
    for x, y in zip(before, after):
        assert (
            x.first_kills, x.first_deaths, x.maps_used, x.maps_skipped,
            x.raw_rate, x.prior, x.mean, x.alpha, x.beta, x.variance,
        ) == (
            y.first_kills, y.first_deaths, y.maps_used, y.maps_skipped,
            y.raw_rate, y.prior, y.mean, y.alpha, y.beta, y.variance,
        )


def test_strictly_earlier_map_changes_the_estimate():
    # Flip side of the leakage proof: a strictly-earlier completed Haven
    # map must change the map-level counts.
    matches, maps, pms, query = _core_tables()
    base = fb.team_map_first_blood_rate("T1", "Haven", query, matches, maps, pms, k=5.0)
    earlier = {
        "match_id": "m_early",
        "date": _stamp(-1),
        "team1_id": "T1",
        "team2_id": "T9",
        "team1_name": "Alpha",
        "team2_name": "Zulu",
        "t1_fk": 9,
        "t1_fd": 12,
        "t2_fk": 12,
        "t2_fd": 9,
    }
    em, ep, es = _scenario([earlier])
    grown_m = pd.concat([matches, em], ignore_index=True)
    grown_p = pd.concat([maps, ep], ignore_index=True)
    grown_s = pd.concat([pms, es], ignore_index=True)
    grown = fb.team_map_first_blood_rate(
        "T1", "Haven", query, grown_m, grown_p, grown_s, k=5.0
    )
    assert (grown.first_kills, grown.first_deaths) == (
        base.first_kills + 9,
        base.first_deaths + 12,
    )
    assert grown.maps_used == base.maps_used + 1


# --------------------------------------------------------------------------
# select_k
# --------------------------------------------------------------------------


def test_select_k_returns_best_k_from_grid_with_finite_scores():
    # select_k returns a best_k that is a grid key, scores_by_k has
    # exactly one entry per grid value, and every score is a positive
    # finite float (a trial-weighted mean binomial log loss — positive
    # because per-trial entropy of a first-blood event is > 0).
    matches, maps, pms = _cv_scenario()
    grid = [1.0, 5.0, 50.0, 500.0]
    best_k, scores = fb.select_k(matches, maps, pms, k_grid=grid)
    assert list(scores.keys()) == grid
    assert best_k in scores
    assert all(math.isfinite(value) and value > 0.0 for value in scores.values())


def test_select_k_scores_are_trial_weighted_binomial():
    # Sanity on the score's meaning: the aggregate is a mean per first-
    # blood trial, so with roughly balanced held-out truths its scale
    # must sit near ln 2 ~ 0.693 (the entropy of a fair coin) — not near
    # 0.693 * 2 and not near 0 — i.e. the denominator counts every
    # (fk + fd) trial, not only one side of the pair. (The old
    # divide-by-fd-only bug produced exactly 2 * ln 2 ~ 1.386.)
    matches, maps, pms = _cv_scenario()
    grid = [1.0, 50.0, 1000.0]
    _best_k, scores = fb.select_k(matches, maps, pms, k_grid=grid)
    for value in scores.values():
        assert 0.5 < value < 0.9


def test_select_k_empty_k_grid_raises():
    # An empty grid cannot produce an argmin.
    matches, maps, pms = _cv_scenario()
    with pytest.raises(ValueError, match="k_grid"):
        fb.select_k(matches, maps, pms, k_grid=[])


def test_select_k_invalid_k_in_grid_raises():
    # A non-positive candidate in the grid is rejected up front.
    matches, maps, pms = _cv_scenario()
    with pytest.raises(ValueError, match="k must be"):
        fb.select_k(matches, maps, pms, k_grid=[1.0, 0.0])


def test_select_k_training_region_too_small_raises():
    # A training region below the split/fold machinery's floor must
    # raise (here the completed table is too small for split_matches'
    # MIN_TRAIN_MATCHES=20, so the error names the training region).
    matches, maps, pms = _cv_scenario(n_matches=12)
    with pytest.raises(ValueError, match="training"):
        fb.select_k(matches, maps, pms, k_grid=[1.0, 5.0])


def test_select_k_zero_scoreable_instances_raises():
    # Enough matches for the fold machinery but no player_map_stats
    # groups in the held-out maps: select_k must raise, not return a
    # best_k over an empty held-out set.
    matches, maps, _pms = _cv_scenario()
    empty_pms = _pms_df([])
    with pytest.raises(ValueError, match="zero scoreable validation instances"):
        fb.select_k(matches, maps, empty_pms, k_grid=[1.0, 5.0])


# --------------------------------------------------------------------------
# invalid k
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, float("nan"), float("inf"), "abc", None])
def test_invalid_k_raises_from_both_public_functions(k):
    # k <= 0 (or NaN/inf/non-numeric) is rejected up front by both
    # estimators through the shared features._shared._validate_k choke
    # point.
    matches, maps, pms, query = _core_tables()
    with pytest.raises(ValueError, match="k must be"):
        fb.team_overall_first_blood_rate("T1", query, matches, maps, pms, k=k)
    with pytest.raises(ValueError, match="k must be"):
        fb.team_map_first_blood_rate("T1", "Haven", query, matches, maps, pms, k)


# --------------------------------------------------------------------------
# real-data smoke tests
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
        and Path("data/v1/player_map_stats.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_sane_numbers_and_conservation():
    # Real v1 sanity (query = latest match + 1h, most frequent team
    # 1001, most frequent map Lotus; DEFAULT_OVERALL_K=50, DEFAULT_K=10,
    # observed at BUILD time):
    #   team 1001 overall: first_kills=546, first_deaths=507,
    #     maps_used=50, maps_skipped=0, raw_rate=0.51852,
    #     mean=0.51768 (prior 0.5).
    #   team 1001 Lotus: first_kills=133, first_deaths=115,
    #     maps_used=12, prior=0.51768 (= overall.mean), mean=0.53557.
    # The whole-dataset conservation identity holds exactly
    # (sum(first_kills) == sum(first_deaths) == 5178 over all 2420
    # rows, 0 nulls), and the 2 maps of match 712803 are the only maps
    # of 244 with no player_map_stats group — a team that played that
    # match must report maps_skipped == 2.
    matches_df, maps_df = asof.load_asof_tables("v1")
    pms_df = pd.read_parquet("data/v1/player_map_stats.parquet")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()
    map_name = maps_df["map_name"].value_counts().idxmax()

    overall = fb.team_overall_first_blood_rate(
        team_id, query, matches_df, maps_df, pms_df
    )
    assert overall.maps_used > 0
    assert overall.prior == fb.LEAGUE_FIRST_BLOOD_RATE == 0.5
    assert 0.0 < overall.raw_rate < 1.0
    assert 0.0 < overall.mean < 1.0
    assert float(overall.raw_rate) == overall.raw_rate  # finite float
    assert overall.alpha > 0.0 and overall.beta > 0.0 and overall.variance > 0.0

    shrunk = fb.team_map_first_blood_rate(
        team_id, map_name, query, matches_df, maps_df, pms_df, fb.DEFAULT_K
    )
    assert shrunk.maps_used >= 0
    assert shrunk.maps_used <= overall.maps_used
    assert shrunk.prior == pytest.approx(overall.mean)
    assert 0.0 < shrunk.raw_rate < 1.0
    assert 0.0 < shrunk.mean < 1.0
    assert shrunk.alpha > 0.0 and shrunk.beta > 0.0 and shrunk.variance > 0.0

    # The conservation identity, independently recomputed against the
    # full real player_map_stats table (no as-of filter): 0 nulls, the
    # global sums equal exactly, and every per-map group conserves.
    assert pms_df["first_kills"].isna().sum() == 0
    assert pms_df["first_deaths"].isna().sum() == 0
    assert int(pms_df["first_kills"].sum()) == int(pms_df["first_deaths"].sum())
    group_fk = pms_df.groupby(["match_id", "map_index"])["first_kills"].sum()
    group_fd = pms_df.groupby(["match_id", "map_index"])["first_deaths"].sum()
    assert len(group_fk) == 242
    assert (group_fk == group_fd).all()

    # Exactly 2 of the 244 maps have no player_map_stats group; a team
    # that played match 712803 skips exactly those 2 maps.
    keys = set(zip(pms_df["match_id"], pms_df["map_index"]))
    n_without = sum(
        1
        for mid, mi in zip(maps_df["match_id"], maps_df["map_index"])
        if (mid, mi) not in keys
    )
    assert n_without == 2
    gap_row = matches_df[matches_df["match_id"] == "712803"].iloc[0]
    gap_team = gap_row["team1_id"]
    gap_overall = fb.team_overall_first_blood_rate(
        gap_team, query, matches_df, maps_df, pms_df
    )
    assert gap_overall.maps_used > 0
    assert gap_overall.maps_skipped == 2


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
        and Path("data/v1/player_map_stats.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_select_k():
    # Real-v1 CV smoke: select_k over the default grid runs to
    # completion and returns a best_k from the grid with positive finite
    # trial-weighted scores. Observed at BUILD time on data/v1:
    # best_k == 1000.0 (the grid's top edge) with scores falling
    # monotonically from 0.705236 (k=1) to 0.693158 (k=1000), an
    # asymptote near ln 2 ~ 0.6931 (see DEFAULT_K_GRID's comment).
    matches_df, maps_df = asof.load_asof_tables("v1")
    pms_df = pd.read_parquet("data/v1/player_map_stats.parquet")
    best_k, scores = fb.select_k(matches_df, maps_df, pms_df)
    assert best_k in fb.DEFAULT_K_GRID
    assert all(math.isfinite(value) and value > 0.0 for value in scores.values())


# --------------------------------------------------------------------------
# batched first-blood diff parity (task 052)
# --------------------------------------------------------------------------


def test_batched_first_blood_diff_bit_exact_parity():
    # The batched path reproduces, element-for-element, the looped
    # single-row team_map_first_blood_rate means per row (no
    # tolerance), across several as-of cutoffs, both seat orientations
    # for T1, and a wholly unseen side (Z) whose whole hierarchy
    # degrades exactly as the single-row path degrades it.
    matches_df, maps_df, pms_df, query = _core_tables()
    d0, d1, d2, d3 = _stamp(0), _stamp(1), _stamp(2), _stamp(3)
    rows_df = pd.DataFrame(
        [
            {"team1_id": "T1", "team2_id": "T2", "map_name": "Haven", "date": d0},
            {"team1_id": "T1", "team2_id": "T2", "map_name": "Haven", "date": d1},
            {"team1_id": "T1", "team2_id": "T3", "map_name": "Haven", "date": d2},
            {"team1_id": "T1", "team2_id": "T4", "map_name": "Bind", "date": d3},
            {"team1_id": "T1", "team2_id": "T4", "map_name": "Bind", "date": query},
            {"team1_id": "Z", "team2_id": "T1", "map_name": "Haven", "date": query},
        ]
    )
    expected = np.zeros(len(rows_df))
    for i, row in enumerate(rows_df.itertuples(index=False)):
        mean_a = fb.team_map_first_blood_rate(
            row.team1_id, row.map_name, row.date, matches_df, maps_df, pms_df,
            fb.BEST_K,
        ).mean
        mean_b = fb.team_map_first_blood_rate(
            row.team2_id, row.map_name, row.date, matches_df, maps_df, pms_df,
            fb.BEST_K,
        ).mean
        expected[i] = mean_a - mean_b
    got = fb.batched_first_blood_diff(rows_df, matches_df, maps_df, pms_df)
    assert got.shape == (len(rows_df),)
    assert np.array_equal(got, expected)


def test_batched_first_blood_conservation_violation_still_raises():
    # The batched path's one-time group validation must still raise the
    # conservation ValueError when a corrupt fixture group violates the
    # per-map invariant, exactly as the single-row path does.
    matches_df, maps_df, pms_df, _query = _core_tables()
    pms_bad = pms_df.copy()
    # Break conservation on m1's team1 side (Alpha: fk 11 -> 20; the
    # mirrored fd on the other side no longer balances the group).
    mask = (pms_bad["match_id"] == "m1") & (pms_bad["team_name"] == "Alpha")
    pms_bad.loc[mask, "first_kills"] = 20
    rows_df = pd.DataFrame(
        [{"team1_id": "T1", "team2_id": "T2", "map_name": "Haven",
          "date": _stamp(4)}]
    )
    with pytest.raises(ValueError, match="conservation"):
        fb.batched_first_blood_diff(rows_df, matches_df, maps_df, pms_bad)

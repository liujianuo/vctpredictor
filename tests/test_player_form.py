"""Tests for recency-weighted player-form features (M16).

Covers the per-match name resolution + join, the three-way failure split
(name mismatch fails loud, missing player rows skip-and-count, per-row
null stats skip-and-count), the two-stage aggregation (per-map unweighted
roster mean, then recency-weighted last-N window), validation of ``n``
and ``decay_rate``, the leakage-safety proof (a poison map dated at or
after the query never changes the value — asserted by *value*, not just
row count), and the ``utils/`` -> no-``drivers/`` layering rule. A
skip-guarded smoke test repeats a basic sanity assertion at real
``data/v1`` scale.
"""

from pathlib import Path

import pandas as pd
import pytest

from utils import player_form

D1 = "2026-01-01T10:00:00"
D2 = "2026-01-02T10:00:00"
D3 = "2026-01-03T10:00:00"
D4 = "2026-01-04T10:00:00"
QUERY = "2026-01-05T00:00:00"

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
_PMS_COLS = ["match_id", "map_index", "player_name", "team_name", "rating", "acs"]


def _matches_df(rows):
    """Build a matches table with the fixed M8 + names column set.

    Wraps ``pd.DataFrame`` so every fixture produces the same column
    order/dtypes regardless of which subset of columns a given fixture
    actually needs.

    Args:
        rows: A list of dicts, one per match; each must carry the keys in
            :data:`_MATCHES_COLS`.

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


def _pms_df(rows):
    """Build a player_map_stats table with the column set this module reads.

    Mirrors :func:`_matches_df` for the player-map stats side; a ``None``
    (or ``float("nan")``) ``acs``/``rating`` value becomes a null cell in
    the resulting frame, which is exactly how the null-skip tests inject
    missing stats.

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


def _build(events):
    """Build matches/maps/player_map_stats frames from a list of map events.

    Each event is a single completed match with a single finished map
    (``winner`` non-null, scores 13-8 so the map is finished regardless
    of which side is queried). The optional ``team1_acs``/``team1_rating``
    (and their ``team2_*`` counterparts) add one single-player roster row
    for that side; an event *without* those keys produces a map that has
    no ``player_map_stats`` rows at all — the real 242/244 gap case.

    Args:
        events: A list of dicts, each with keys ``match_id``, ``date``,
            ``team1_id``, ``team2_id``, ``team1_name``, ``team2_name``,
            and optionally ``map_index`` (default 0), ``team1_acs``,
            ``team1_rating``, ``team2_acs``, ``team2_rating``.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple built by
        :func:`_matches_df` / :func:`_maps_df` / :func:`_pms_df`.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if an event dict is malformed).
    """
    match_rows = []
    map_rows = []
    pms_rows = []
    for e in events:
        map_index = e.get("map_index", 0)
        match_rows.append(
            {
                "match_id": e["match_id"],
                "date": e["date"],
                "team1_id": e["team1_id"],
                "team2_id": e["team2_id"],
                "team1_name": e["team1_name"],
                "team2_name": e["team2_name"],
                "status": "completed",
            }
        )
        map_rows.append(
            {
                "match_id": e["match_id"],
                "map_index": map_index,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": e["team1_id"],
            }
        )
        if "team1_acs" in e:
            pms_rows.append(
                {
                    "match_id": e["match_id"],
                    "map_index": map_index,
                    "player_name": "p1",
                    "team_name": e["team1_name"],
                    "rating": e["team1_rating"],
                    "acs": e["team1_acs"],
                }
            )
        if "team2_acs" in e:
            pms_rows.append(
                {
                    "match_id": e["match_id"],
                    "map_index": map_index,
                    "player_name": "p2",
                    "team_name": e["team2_name"],
                    "rating": e["team2_rating"],
                    "acs": e["team2_acs"],
                }
            )
    return _matches_df(match_rows), _maps_df(map_rows), _pms_df(pms_rows)


def _alpha_events(acs_values, start_index=0):
    """Build team A (team1) single-map events with the given ACS history.

    Produces one completed match per ACS value, team ``A`` ("Alpha") as
    team1 vs team ``B`` ("Beta"), dated :data:`D1` + ``start_index`` days
    in order, with a single-player Alpha roster carrying that ACS (rating
    fixed at 1.0) and a fixed Beta roster (acs 50.0, rating 0.5).

    Args:
        acs_values: The ACS values in chronological (oldest-first) order;
            one event per value.
        start_index: The date offset (in days after :data:`D1`) for the
            first event's date.

    Returns:
        A list of event dicts in the shape :func:`_build` expects.

    Raises:
        Nothing.
    """
    dates = [
        f"2026-01-{1 + start_index + i:02d}T10:00:00" for i in range(len(acs_values))
    ]
    return [
        {
            "match_id": f"m{start_index + i}",
            "date": dates[i],
            "team1_id": "A",
            "team2_id": "B",
            "team1_name": "Alpha",
            "team2_name": "Beta",
            "team1_acs": acs,
            "team1_rating": 1.0,
            "team2_acs": 50.0,
            "team2_rating": 0.5,
        }
        for i, acs in enumerate(acs_values)
    ]


# --------------------------------------------------------------------------
# leakage safety
# --------------------------------------------------------------------------


def test_leakage_poison_map_at_or_after_query_excluded():
    # Non-tautological leakage proof: team A has one strictly-before map
    # (acs 100). A poison map dated exactly at the query carries acs 1000
    # (10x). The value must stay 100.0 both with and without the poison —
    # asserted on the *value*, not just the maps_used count — while a
    # genuinely earlier map (acs 300) *does* move the value, proving the
    # feature responds to real history and ignores at/after-query rows.
    base_m, base_p, base_s = _build(
        _alpha_events([100.0])
    )
    base = player_form.team_player_form("A", D2, base_m, base_p, base_s)
    assert base.acs.mean == pytest.approx(100.0)

    poison_m, poison_p, poison_s = _build(
        [
            {
                "match_id": "poison",
                "date": D2,  # exactly at the query date -> strict < excludes it
                "team1_id": "A",
                "team2_id": "C",
                "team1_name": "Alpha",
                "team2_name": "Gamma",
                "team1_acs": 1000.0,
                "team1_rating": 1.0,
                "team2_acs": 5.0,
                "team2_rating": 0.1,
            }
        ]
    )
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)
    grown_s = pd.concat([base_s, poison_s], ignore_index=True)
    grown = player_form.team_player_form("A", D2, grown_m, grown_p, grown_s)
    assert grown.acs.mean == pytest.approx(100.0)
    assert grown.acs.maps_used == 1

    earlier_m, earlier_p, earlier_s = _build(
        [
            {
                "match_id": "m_early",
                "date": "2025-12-31T10:00:00",
                "team1_id": "A",
                "team2_id": "C",
                "team1_name": "Alpha",
                "team2_name": "Gamma",
                "team1_acs": 300.0,
                "team1_rating": 1.0,
            }
        ]
    )
    real_m = pd.concat([base_m, earlier_m], ignore_index=True)
    real_p = pd.concat([base_p, earlier_p], ignore_index=True)
    real_s = pd.concat([base_s, earlier_s], ignore_index=True)
    real = player_form.team_player_form(
        "A", D2, real_m, real_p, real_s, n=10, decay_rate=0.9
    )
    # m1 (100, most recent, weight 1.0) + m0 (300, weight 0.9):
    assert real.acs.mean == pytest.approx((100.0 + 300.0 * 0.9) / 1.9)


# --------------------------------------------------------------------------
# recency weighting
# --------------------------------------------------------------------------


def test_recency_weighting_outlier_position():
    # Two histories differing only in where the outlier (100) sits. With
    # n=3 and decay 0.5 the hand-computed weighted means are 61.4285...
    # (outlier most recent) and 22.8571... (outlier least recent); the
    # most-recent-outlier value must be strictly closer to 100.
    m, p, s = _build(_alpha_events([10.0, 10.0, 100.0]))
    recent = player_form.team_player_form(
        "A", QUERY, m, p, s, n=3, decay_rate=0.5
    )
    assert recent.acs.maps_used == 3
    assert recent.acs.per_map_means == (100.0, 10.0, 10.0)
    assert recent.acs.weights == (1.0, 0.5, 0.25)
    assert recent.acs.mean == pytest.approx((100.0 + 10.0 * 0.5 + 10.0 * 0.25) / 1.75)

    m2, p2, s2 = _build(_alpha_events([100.0, 10.0, 10.0]))
    distant = player_form.team_player_form(
        "A", QUERY, m2, p2, s2, n=3, decay_rate=0.5
    )
    assert distant.acs.per_map_means == (10.0, 10.0, 100.0)
    assert distant.acs.mean == pytest.approx((10.0 + 10.0 * 0.5 + 100.0 * 0.25) / 1.75)

    assert abs(100.0 - recent.acs.mean) < abs(100.0 - distant.acs.mean)


def test_partial_history_uses_what_exists_no_padding():
    # 2 maps with n=5: both are used with no padding/reweighting; the
    # weighted mean is the hand-computed (200*1 + 100*0.9)/1.9.
    m, p, s = _build(_alpha_events([100.0, 200.0]))
    res = player_form.team_player_form(
        "A", QUERY, m, p, s, n=5, decay_rate=0.9
    )
    assert res.acs.maps_used == 2
    assert res.acs.per_map_means == (200.0, 100.0)
    assert res.acs.weights == (1.0, 0.9)
    assert res.acs.mean == pytest.approx((200.0 + 100.0 * 0.9) / 1.9)


def test_window_uses_last_n_only():
    # 4 maps with n=2: only the two most recent survive, most-recent first.
    m, p, s = _build(_alpha_events([100.0, 200.0, 300.0, 400.0]))
    res = player_form.team_player_form(
        "A", QUERY, m, p, s, n=2, decay_rate=0.5
    )
    assert res.acs.maps_used == 2
    assert res.acs.per_map_means == (400.0, 300.0)
    assert res.acs.weights == (1.0, 0.5)
    assert res.acs.mean == pytest.approx((400.0 + 300.0 * 0.5) / 1.5)


def test_decay_one_is_uniform_and_zero_is_hard_window():
    # decay_rate=1.0 -> plain window mean; 0.0 -> only the most recent map.
    m, p, s = _build(_alpha_events([100.0, 200.0]))
    uniform = player_form.team_player_form(
        "A", QUERY, m, p, s, n=2, decay_rate=1.0
    )
    assert uniform.acs.mean == pytest.approx(150.0)

    hard = player_form.team_player_form(
        "A", QUERY, m, p, s, n=2, decay_rate=0.0
    )
    assert hard.acs.mean == pytest.approx(200.0)


def test_exposed_fields_recompute_weighted_mean():
    # The exposed per_map_means/weights must independently reproduce the
    # reported mean (most-recent-first alignment of the two tuples).
    m, p, s = _build(_alpha_events([10.0, 20.0, 30.0]))
    res = player_form.team_player_form(
        "A", QUERY, m, p, s, n=3, decay_rate=0.5
    )
    recomputed = sum(
        w * v for w, v in zip(res.acs.weights, res.acs.per_map_means)
    ) / sum(res.acs.weights)
    assert res.acs.mean == pytest.approx(recomputed)


# --------------------------------------------------------------------------
# name resolution + join
# --------------------------------------------------------------------------


def test_orientation_resolves_team2_name():
    # A is team2 in this match; its roster rows carry team2_name, and the
    # resolved name must pick those (acs 123), not team1's (acs 45).
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "B",
                "team2_id": "A",
                "team1_name": "Beta",
                "team2_name": "Alpha",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "B",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "pa",
             "team_name": "Alpha", "rating": 1.0, "acs": 123.0},
            {"match_id": "m1", "map_index": 0, "player_name": "pb",
             "team_name": "Beta", "rating": 0.5, "acs": 45.0},
        ]
    )
    res = player_form.team_player_form("A", QUERY, matches, maps, pms)
    assert res.acs.mean == pytest.approx(123.0)
    assert res.acs.maps_used == 1


def test_per_map_mean_is_unweighted_across_roster_rows():
    # Three roster rows (acs 100/200/300, rating 1/2/3) must average to
    # 200.0 / 2.0 — the divisor is the number of joined rows, not a
    # hardcoded 5, and within-map weighting never applies.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "p1",
             "team_name": "Alpha", "rating": 1.0, "acs": 100.0},
            {"match_id": "m1", "map_index": 0, "player_name": "p2",
             "team_name": "Alpha", "rating": 2.0, "acs": 200.0},
            {"match_id": "m1", "map_index": 0, "player_name": "p3",
             "team_name": "Alpha", "rating": 3.0, "acs": 300.0},
        ]
    )
    res = player_form.team_player_form("A", QUERY, matches, maps, pms)
    assert res.acs.mean == pytest.approx(200.0)
    assert res.rating.mean == pytest.approx(2.0)
    assert res.acs.null_rows_skipped == 0


def test_map_index_orders_maps_within_a_match():
    # One match, three maps listed out of order (2, 0, 1) in the frames:
    # the (date, match_id, map_index) sort must replay them in play order,
    # so the most-recent (map_index 2) is the 300 map.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {"match_id": "m1", "map_index": 2, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m1", "map_index": 1, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "p",
             "team_name": "Alpha", "rating": 1.0, "acs": 100.0},
            {"match_id": "m1", "map_index": 1, "player_name": "p",
             "team_name": "Alpha", "rating": 1.0, "acs": 200.0},
            {"match_id": "m1", "map_index": 2, "player_name": "p",
             "team_name": "Alpha", "rating": 1.0, "acs": 300.0},
        ]
    )
    res = player_form.team_player_form(
        "A", QUERY, matches, maps, pms, n=3, decay_rate=0.5
    )
    assert res.acs.per_map_means == (300.0, 200.0, 100.0)


# --------------------------------------------------------------------------
# three-way failure split
# --------------------------------------------------------------------------


def test_map_without_player_rows_skipped_and_counted():
    # m1 has a roster, m2 has none (the 242/244 gap case): m2 is skipped
    # and counted, not an error, and only m1's acs 100 feeds the mean.
    m, p, s = _build(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "team1_acs": 100.0,
                "team1_rating": 1.0,
                "team2_acs": 50.0,
                "team2_rating": 0.5,
            },
            {
                "match_id": "m2",
                "date": D2,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
            },
        ]
    )
    res = player_form.team_player_form("A", QUERY, m, p, s)
    assert res.as_of_maps == 2
    assert res.skipped_maps == 1
    assert res.acs.maps_used == 1
    assert res.acs.mean == pytest.approx(100.0)


def test_map_with_only_opponent_rows_skipped():
    # The map's group exists but holds only the opponent's rows: the
    # queried team's roster is empty -> skip-and-count, not an error.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "pb",
             "team_name": "Beta", "rating": 0.5, "acs": 45.0},
        ]
    )
    res = player_form.team_player_form("A", QUERY, matches, maps, pms)
    assert res.as_of_maps == 1
    assert res.skipped_maps == 1
    assert res.acs.maps_used == 0
    assert res.acs.mean is None


def test_null_acs_row_skipped_other_stat_unaffected():
    # Two Alpha rows: acs (100, None) and rating (1.0, 2.0). The null acs
    # row is excluded from the acs mean (100.0) but rating still averages
    # both rows (1.5), independently.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "p1",
             "team_name": "Alpha", "rating": 1.0, "acs": 100.0},
            {"match_id": "m1", "map_index": 0, "player_name": "p2",
             "team_name": "Alpha", "rating": 2.0, "acs": None},
            {"match_id": "m1", "map_index": 0, "player_name": "p3",
             "team_name": "Beta", "rating": 0.5, "acs": 40.0},
        ]
    )
    res = player_form.team_player_form("A", QUERY, matches, maps, pms)
    assert res.acs.mean == pytest.approx(100.0)
    assert res.acs.maps_used == 1
    assert res.acs.null_rows_skipped == 1
    assert res.rating.mean == pytest.approx(1.5)
    assert res.rating.null_rows_skipped == 0


def test_all_null_acs_does_not_knock_out_rating():
    # Every Alpha row has a null acs but a real rating: acs contributes
    # nothing (maps_used 0, mean None) while rating still computes.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "p1",
             "team_name": "Alpha", "rating": 1.0, "acs": None},
            {"match_id": "m1", "map_index": 0, "player_name": "p2",
             "team_name": "Alpha", "rating": 2.0, "acs": None},
        ]
    )
    res = player_form.team_player_form("A", QUERY, matches, maps, pms)
    assert res.acs.maps_used == 0
    assert res.acs.mean is None
    assert res.acs.null_rows_skipped == 2
    assert res.rating.maps_used == 1
    assert res.rating.mean == pytest.approx(1.5)


def test_name_mismatch_raises_value_error():
    # A player_map_stats team_name that matches neither side of its match
    # must fail loudly, not be silently dropped or guessed at.
    matches = _matches_df(
        [
            {
                "match_id": "m1",
                "date": D1,
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "status": "completed",
            }
        ]
    )
    maps = _maps_df(
        [
            {
                "match_id": "m1",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        ]
    )
    pms = _pms_df(
        [
            {"match_id": "m1", "map_index": 0, "player_name": "x",
             "team_name": "GHOST", "rating": 1.0, "acs": 100.0},
        ]
    )
    with pytest.raises(ValueError, match="matching neither"):
        player_form.team_player_form("A", QUERY, matches, maps, pms)


# --------------------------------------------------------------------------
# empty history + validation
# --------------------------------------------------------------------------


def test_empty_history_sentinel_none():
    # An unseen team has zero as-of maps and therefore None means (not a
    # fabricated numeric default).
    m, p, s = _build(_alpha_events([100.0]))
    res = player_form.team_player_form("Z", QUERY, m, p, s)
    assert res.as_of_maps == 0
    assert res.skipped_maps == 0
    assert res.acs.maps_used == 0
    assert res.acs.mean is None
    assert res.acs.per_map_means == ()
    assert res.acs.weights == ()
    assert res.rating.maps_used == 0
    assert res.rating.mean is None


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, True, "3", None])
def test_invalid_n_raises(bad_n):
    # n must be a positive integer (bools, non-integers and <= 0 rejected).
    m, p, s = _build(_alpha_events([100.0]))
    with pytest.raises(ValueError, match="n must be"):
        player_form.team_player_form("A", QUERY, m, p, s, n=bad_n)


@pytest.mark.parametrize("bad_d", [-0.1, 1.1, float("nan"), float("inf"), "x"])
def test_invalid_decay_rate_raises(bad_d):
    # decay_rate must be a finite real in [0, 1].
    m, p, s = _build(_alpha_events([100.0]))
    with pytest.raises(ValueError, match="decay_rate must be"):
        player_form.team_player_form("A", QUERY, m, p, s, decay_rate=bad_d)


# --------------------------------------------------------------------------
# layering + real-data smoke
# --------------------------------------------------------------------------


def test_no_drivers_import_in_utils_player_form():
    # The utils/ layering rule: utils modules must not import from
    # drivers/. Assert the literal import is absent from the module source.
    source = Path(player_form.__file__).read_text(encoding="utf-8")
    assert "from drivers" not in source
    assert "import drivers" not in source


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
        and Path("data/v1/player_map_stats.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_sane_numbers():
    # Sanity at real v1 scale: the most-appearing team queried just after
    # the dataset's latest date has maps_used > 0, means within plausible
    # ACS/rating ranges, and its reported mean reproduces from the exposed
    # per-map means/weights. A team that played the known player-stat-gap
    # match must report skipped_maps > 0 without erroring.
    matches, maps, pms = player_form.load_player_form_tables("v1")
    appearances = pd.concat([matches["team1_id"], matches["team2_id"]]).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    res = player_form.team_player_form(team_id, query, matches, maps, pms)
    assert res.acs.maps_used > 0
    assert res.rating.maps_used > 0
    assert res.acs.mean is not None and res.rating.mean is not None
    assert 0.0 < res.acs.mean < 500.0
    assert 0.0 < res.rating.mean < 3.0
    recomputed = sum(
        w * v for w, v in zip(res.acs.weights, res.acs.per_map_means)
    ) / sum(res.acs.weights)
    assert res.acs.mean == pytest.approx(recomputed)

    gap_match = matches[matches["match_id"] == "712803"]
    if not gap_match.empty:
        gap_team = gap_match.iloc[0]["team1_id"]
        gap_res = player_form.team_player_form(gap_team, query, matches, maps, pms)
        assert gap_res.skipped_maps > 0

"""Tests for closeness and overtime features (M15).

Covers the league-wide as-of filter, the shared margin/OT/null-score
helper, the three public features (``team_close_map_frequency`` /
``global_ot_rate`` / ``team_ot_rate`` / ``map_round_margin_variance``),
their shrinkage and variance contracts, the null-score fail-loud guard
exercised from every public entry point, and the leakage-safety proof
(a map dated exactly at or after the cutoff never enters any estimate;
strict ``<`` boundary). A skip-guarded smoke test repeats the
no-leakage assertion at real ``data/v1`` scale.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import closeness
from utils import asof

QUERY = "2026-01-03T00:00:00"
D1 = "2026-01-01T10:00:00"
D2 = "2026-01-02T10:00:00"

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

    Wraps ``pd.DataFrame`` so every fixture produces the same column
    order/dtypes regardless of which subset of columns a given fixture
    actually needs.

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


def _build(events):
    """Build a matches+maps pair from a list of event dicts.

    Each event becomes one completed match and one finished map, with
    the ``winner`` derived from the scores (never a display-name string),
    so a null score still yields a non-null ``winner`` — exactly the
    shape the null-score guard must catch.

    Args:
        events: A list of dicts, each with keys ``match_id``, ``date``,
            ``team1_id``, ``team2_id``, ``map_name``, ``team1_score``,
            ``team2_score``.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df` /
        :func:`_maps_df`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    for e in events:
        t1 = e["team1_score"]
        t2 = e["team2_score"]
        match_rows.append(
            {
                "match_id": e["match_id"],
                "date": e["date"],
                "team1_id": e["team1_id"],
                "team2_id": e["team2_id"],
                "status": "completed",
            }
        )
        map_rows.append(
            {
                "match_id": e["match_id"],
                "map_index": 0,
                "map_name": e["map_name"],
                "team1_score": t1,
                "team2_score": t2,
                "winner": e["team1_id"] if t1 > t2 else e["team2_id"],
            }
        )
    return _matches_df(match_rows), _maps_df(map_rows)


def _base_events():
    """Return the shared strictly-before history for the leakage tests.

    Team ``A`` plays m1 (Haven, 13-11: close, not OT) and m2 (Bind,
    15-13 as team2: close *and* OT, exercising both orientation states
    while keeping the symmetric close/OT derivation orientation-free);
    team ``C``/``D`` add m5 (Ascent, 13-8) and m6 (Haven, 13-7) so the
    league pool has more than one team's maps and "Haven" has two maps
    (a real, non-NaN sample variance). All four are strictly before
    :data:`QUERY`.

    Returns:
        A list of event dicts in the shape :func:`_build` expects.

    Raises:
        Nothing.
    """
    return [
        {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
         "map_name": "Haven", "team1_score": 13, "team2_score": 11},
        {"match_id": "m2", "date": D2, "team1_id": "B", "team2_id": "A",
         "map_name": "Bind", "team1_score": 15, "team2_score": 13},
        {"match_id": "m5", "date": D1, "team1_id": "C", "team2_id": "D",
         "map_name": "Ascent", "team1_score": 13, "team2_score": 8},
        {"match_id": "m6", "date": D2, "team1_id": "C", "team2_id": "D",
         "map_name": "Haven", "team1_score": 13, "team2_score": 7},
    ]


def _poison_events():
    """Return the exactly-at and after :data:`QUERY` poison maps.

    One map dated exactly at the cutoff and one dated after it, both
    involving team ``A`` on "Haven" with a close scoreline, so if either
    leaked into any estimate it would change the result.

    Returns:
        A list of event dicts in the shape :func:`_build` expects.

    Raises:
        Nothing.
    """
    return [
        {"match_id": "m3", "date": QUERY, "team1_id": "A", "team2_id": "C",
         "map_name": "Haven", "team1_score": 13, "team2_score": 5},
        {"match_id": "m4", "date": "2026-01-04T10:00:00", "team1_id": "A",
         "team2_id": "C", "map_name": "Haven", "team1_score": 13,
         "team2_score": 5},
    ]


# --------------------------------------------------------------------------
# shared helper
# --------------------------------------------------------------------------


def test_margin_and_ot_derives_flags_and_rejects_null():
    # The shared helper produces abs_margin, is_close and is_ot from the
    # two scores, and a NaN in either column raises before arithmetic.
    abs_margin, is_close, is_ot = closeness._margin_and_ot(
        pd.Series([13, 15, 13]), pd.Series([11, 13, 8])
    )
    assert abs_margin.tolist() == [2, 2, 5]
    assert is_close.tolist() == [True, True, False]
    assert is_ot.tolist() == [False, True, False]
    with pytest.raises(ValueError, match="null/NaN"):
        closeness._margin_and_ot(pd.Series([float("nan")]), pd.Series([8]))


# --------------------------------------------------------------------------
# team_close_map_frequency
# --------------------------------------------------------------------------


def test_close_frequency_exact_rate():
    # Margins 2, 1, 5, 5, 3 -> two close maps out of five -> rate 0.4.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 13, "team2_score": 11},
            {"match_id": "m2", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Bind", "team1_score": 13, "team2_score": 12},
            {"match_id": "m3", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Lotus", "team1_score": 13, "team2_score": 8},
            {"match_id": "m4", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Pearl", "team1_score": 8, "team2_score": 13},
            {"match_id": "m5", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Split", "team1_score": 13, "team2_score": 10},
        ]
    )
    result = closeness.team_close_map_frequency("A", QUERY, matches_df, maps_df)
    assert result.close_maps == 2
    assert result.total_maps == 5
    assert result.rate == pytest.approx(0.4)


def test_close_frequency_zero_history_is_zero():
    # An unseen team yields 0 close maps / 0 total maps and rate 0.0
    # ("no evidence of closeness"), not an error.
    matches_df, maps_df = _build(_base_events())
    result = closeness.team_close_map_frequency("UNSEEN", QUERY, matches_df, maps_df)
    assert result.close_maps == 0
    assert result.total_maps == 0
    assert result.rate == 0.0


# --------------------------------------------------------------------------
# global_ot_rate
# --------------------------------------------------------------------------


def test_global_ot_rate_exact_rate():
    # Pool: 15-13 (OT), 13-11 (not OT), 13-12 (OT), 13-8 (not OT) ->
    # 2 OT events / 4 games -> 0.5, pooled across all teams.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 15, "team2_score": 13},
            {"match_id": "m2", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Bind", "team1_score": 13, "team2_score": 11},
            {"match_id": "m3", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Lotus", "team1_score": 13, "team2_score": 12},
            {"match_id": "m4", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Pearl", "team1_score": 13, "team2_score": 8},
        ]
    )
    result = closeness.global_ot_rate(QUERY, matches_df, maps_df)
    assert result.events == 2
    assert result.games == 4
    assert result.rate == pytest.approx(0.5)


def test_global_ot_rate_zero_history_is_zero():
    # A cutoff before every match yields 0 events / 0 games and rate 0.0.
    matches_df, maps_df = _build(_base_events())
    result = closeness.global_ot_rate("2025-12-31T00:00:00", matches_df, maps_df)
    assert result.events == 0
    assert result.games == 0
    assert result.rate == 0.0


# --------------------------------------------------------------------------
# team_ot_rate
# --------------------------------------------------------------------------


def test_team_ot_rate_zero_games_full_shrinkage():
    # Zero games: mean == prior exactly (full shrinkage), raw_rate is
    # reported as the prior too.
    matches_df, maps_df = _build(_base_events())
    result = closeness.team_ot_rate("UNSEEN", QUERY, matches_df, maps_df)
    assert result.events == 0
    assert result.games == 0
    assert result.raw_rate == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)


def test_team_ot_rate_heavy_k_pulls_harder_than_m13_default():
    # Team A: 1 OT event / 2 games (raw 0.5); league pool prior is 0.2
    # (team C adds 1 OT event over 8 games). The heavy DEFAULT_OT_K
    # (1000) must pull the mean toward the prior *strictly harder* than
    # M13's lighter win-rate DEFAULT_K (10), making the "heavily-shrunk"
    # claim concrete rather than asserted in prose.
    events = [
        {"match_id": "a1", "date": D1, "team1_id": "A", "team2_id": "B",
         "map_name": "Haven", "team1_score": 15, "team2_score": 13},
        {"match_id": "a2", "date": D1, "team1_id": "A", "team2_id": "B",
         "map_name": "Bind", "team1_score": 13, "team2_score": 8},
    ]
    # Team C: 1 OT event out of 8 games, so the pooled prior is
    # (1 + 1) / (2 + 8) = 0.2.
    for i in range(8):
        ot = i == 0
        events.append(
            {"match_id": f"c{i}", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Lotus", "team1_score": 15 if ot else 13,
             "team2_score": 13 if ot else 8}
        )
    matches_df, maps_df = _build(events)

    prior = closeness.global_ot_rate(QUERY, matches_df, maps_df).rate
    assert prior == pytest.approx(0.2)

    heavy = closeness.team_ot_rate("A", QUERY, matches_df, maps_df)
    light = closeness.team_ot_rate("A", QUERY, matches_df, maps_df, k=10.0)
    assert heavy.events == 1
    assert heavy.games == 2
    assert heavy.raw_rate == 0.5
    assert heavy.prior == pytest.approx(prior)
    # Both shrink toward the prior (raw 0.5 > prior 0.2), but the heavy
    # k lands strictly closer.
    assert prior < heavy.mean < light.mean < heavy.raw_rate
    assert (heavy.mean - prior) < (light.mean - prior)


def test_team_ot_rate_monotonic_in_k():
    # Holding events/games/prior fixed, mean moves monotonically toward
    # the prior as k grows (raw 0.5 > prior 0.2, so it decreases).
    matches_df, maps_df = _build(
        [
            {"match_id": "a1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 15, "team2_score": 13},
            {"match_id": "a2", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Bind", "team1_score": 13, "team2_score": 8},
            {"match_id": "c1", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Lotus", "team1_score": 15, "team2_score": 13},
            {"match_id": "c2", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Lotus", "team1_score": 13, "team2_score": 8},
            {"match_id": "c3", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Lotus", "team1_score": 13, "team2_score": 8},
        ]
    )
    means = [
        closeness.team_ot_rate("A", QUERY, matches_df, maps_df, k=k).mean
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert means[0] > means[1] > means[2] > means[3]
    assert means[3] > 0.0


@pytest.mark.parametrize("bad_k", [0, -1, 0.0, float("nan"), float("inf")])
def test_team_ot_rate_invalid_k_raises(bad_k):
    # k must be a positive finite real number.
    matches_df, maps_df = _build(_base_events())
    with pytest.raises(ValueError, match="k must be"):
        closeness.team_ot_rate("A", QUERY, matches_df, maps_df, k=bad_k)


# --------------------------------------------------------------------------
# map_round_margin_variance
# --------------------------------------------------------------------------


def test_map_variance_matches_numpy_ddof1():
    # Margins [2, 6, 4] on "Haven" -> sample variance 4.0; a "Bind" map
    # with a different margin must not leak into Haven's estimate.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 13, "team2_score": 11},
            {"match_id": "m2", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Haven", "team1_score": 13, "team2_score": 7},
            {"match_id": "m3", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Haven", "team1_score": 13, "team2_score": 9},
            {"match_id": "m4", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Bind", "team1_score": 13, "team2_score": 1},
        ]
    )
    result = closeness.map_round_margin_variance("Haven", QUERY, matches_df, maps_df)
    assert result.n == 3
    assert result.variance == pytest.approx(np.var([2.0, 6.0, 4.0], ddof=1))


def test_map_variance_degenerate_n_is_nan():
    # n=0 (no maps on the name) and n=1 (a single map) both yield NaN,
    # asserted via math.isnan (NaN != NaN).
    matches_df, maps_df = _build(_base_events())
    zero = closeness.map_round_margin_variance("Fracture", QUERY, matches_df, maps_df)
    one = closeness.map_round_margin_variance("Ascent", QUERY, matches_df, maps_df)
    assert zero.n == 0
    assert math.isnan(zero.variance)
    assert one.n == 1
    assert math.isnan(one.variance)


def test_map_variance_normalizes_map_name():
    # " breeze " must match the stored "Breeze" via normalize_map_name.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Breeze", "team1_score": 13, "team2_score": 11},
            {"match_id": "m2", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Breeze", "team1_score": 13, "team2_score": 7},
        ]
    )
    result = closeness.map_round_margin_variance(" breeze ", QUERY, matches_df, maps_df)
    assert result.n == 2
    assert result.variance == pytest.approx(np.var([2.0, 6.0], ddof=1))


# --------------------------------------------------------------------------
# map_ot_rate (M27)
# --------------------------------------------------------------------------


def test_map_ot_rate_shrinkage_arithmetic():
    # Haven: 4 as-of maps, 3 OT events (15-13, 14-12, 15-13) and one
    # regulation (13-8); Bind: 2 as-of maps, both regulation. The
    # league-wide prior pools all 6 maps -> 3/6 = 0.5. With k = 2:
    # mean = (3 + 2*0.5) / (4 + 2) = 4/6, alpha = 4, beta = 2,
    # variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)) =
    # 4*2 / (6^2 * 7). The Bind estimate must be different (0 OT / 2
    # games), proving the per-map filter isolates the named map.
    matches_df, maps_df = _build(
        [
            {"match_id": "h1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 15, "team2_score": 13},
            {"match_id": "h2", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Haven", "team1_score": 14, "team2_score": 12},
            {"match_id": "h3", "date": D2, "team1_id": "A", "team2_id": "C",
             "map_name": "Haven", "team1_score": 15, "team2_score": 13},
            {"match_id": "h4", "date": D2, "team1_id": "B", "team2_id": "D",
             "map_name": "Haven", "team1_score": 13, "team2_score": 8},
            {"match_id": "b1", "date": D1, "team1_id": "A", "team2_id": "D",
             "map_name": "Bind", "team1_score": 13, "team2_score": 11},
            {"match_id": "b2", "date": D2, "team1_id": "B", "team2_id": "C",
             "map_name": "Bind", "team1_score": 13, "team2_score": 7},
        ]
    )
    result = closeness.map_ot_rate("Haven", QUERY, matches_df, maps_df, k=2.0)
    prior = closeness.global_ot_rate(QUERY, matches_df, maps_df).rate
    assert prior == pytest.approx(0.5)
    assert result.events == 3
    assert result.games == 4
    assert result.raw_rate == pytest.approx(0.75)
    assert result.mean == pytest.approx((3 + 2.0 * 0.5) / (4 + 2.0))
    assert result.alpha == pytest.approx(4.0)
    assert result.beta == pytest.approx(2.0)
    assert result.variance == pytest.approx(4.0 * 2.0 / (6.0**2 * 7.0))

    bind = closeness.map_ot_rate("Bind", QUERY, matches_df, maps_df, k=2.0)
    assert bind.events == 0
    assert bind.games == 2
    assert bind.mean == pytest.approx((0 + 2.0 * 0.5) / (2 + 2.0))


def test_map_ot_rate_zero_games_full_shrinkage():
    # No as-of map on the name: mean == prior exactly (full shrinkage)
    # and raw_rate is reported as the prior too.
    matches_df, maps_df = _build(_base_events())
    result = closeness.map_ot_rate("Fracture", QUERY, matches_df, maps_df)
    assert result.events == 0
    assert result.games == 0
    assert result.raw_rate == pytest.approx(result.prior)
    assert result.mean == pytest.approx(result.prior)


def test_map_ot_rate_normalizes_map_name():
    # " haven " must match the stored "Haven" via normalize_map_name.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 15, "team2_score": 13},
            {"match_id": "m2", "date": D1, "team1_id": "C", "team2_id": "D",
             "map_name": "Haven", "team1_score": 13, "team2_score": 8},
        ]
    )
    result = closeness.map_ot_rate(" haven ", QUERY, matches_df, maps_df, k=2.0)
    assert result.events == 1
    assert result.games == 2


@pytest.mark.parametrize("bad_k", [0, -1, 0.0, float("nan"), float("inf")])
def test_map_ot_rate_invalid_k_raises(bad_k):
    # k must be a positive finite real number.
    matches_df, maps_df = _build(_base_events())
    with pytest.raises(ValueError, match="k must be"):
        closeness.map_ot_rate("Haven", QUERY, matches_df, maps_df, k=bad_k)


def test_map_ot_rate_leakage_strict_boundary():
    # The exactly-at and after poison maps (both "Haven") must never
    # change the per-map OT estimate, which stays over the two
    # strictly-before Haven maps (0 OT events / 2 games, prior 0.25).
    base_m, base_p = _build(_base_events())
    poison_m, poison_p = _build(_poison_events())
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)

    base = closeness.map_ot_rate("Haven", QUERY, base_m, base_p, k=2.0)
    grown = closeness.map_ot_rate("Haven", QUERY, grown_m, grown_p, k=2.0)
    assert base.events == 0
    assert base.games == 2
    assert (base.events, base.games, base.mean) == (
        grown.events,
        grown.games,
        grown.mean,
    )


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_map_ot_rate_sane_numbers():
    # Real v1 scale: a real map's per-map OT estimate is non-degenerate
    # (mean strictly inside (0, 1), not all-zero and not all-one), the
    # counts are consistent, and the prior is the global pooled OT rate
    # at the same cutoff.
    matches_df, maps_df = asof.load_asof_tables("v1")
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()
    a_map = maps_df["map_name"].iloc[0]

    result = closeness.map_ot_rate(a_map, query, matches_df, maps_df)
    assert result.games > 0
    assert result.events <= result.games
    assert 0.0 <= result.raw_rate <= 1.0
    assert 0.0 < result.mean < 1.0
    assert result.prior == pytest.approx(
        closeness.global_ot_rate(query, matches_df, maps_df).rate
    )


# --------------------------------------------------------------------------
# null-score guard (deliverable B's fail-loud convention, M15 side)
# --------------------------------------------------------------------------


def test_null_score_raises_from_all_public_functions():
    # A finished map (winner non-null) with a NaN team1_score must raise
    # ValueError from every public function that would touch it — not
    # just the shared helper in isolation.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": float("nan"),
             "team2_score": 8},
        ]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.team_close_map_frequency("A", QUERY, matches_df, maps_df)
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.global_ot_rate(QUERY, matches_df, maps_df)
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.team_ot_rate("A", QUERY, matches_df, maps_df)
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.map_round_margin_variance("Haven", QUERY, matches_df, maps_df)
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.map_ot_rate("Haven", QUERY, matches_df, maps_df)


def test_null_score_team2_raises():
    # The symmetric guard fires when the *other* score column is NaN.
    matches_df, maps_df = _build(
        [
            {"match_id": "m1", "date": D1, "team1_id": "A", "team2_id": "B",
             "map_name": "Haven", "team1_score": 13,
             "team2_score": float("nan")},
        ]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        closeness.team_close_map_frequency("A", QUERY, matches_df, maps_df)


# --------------------------------------------------------------------------
# leakage safety
# --------------------------------------------------------------------------


def test_close_frequency_leakage_strict_boundary():
    # The exactly-at and after poison maps must never change team A's
    # close-map frequency (only the strictly-before m1/m2 count).
    base_m, base_p = _build(_base_events())
    poison_m, poison_p = _build(_poison_events())
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)

    base = closeness.team_close_map_frequency("A", QUERY, base_m, base_p)
    grown = closeness.team_close_map_frequency("A", QUERY, grown_m, grown_p)
    assert (base.close_maps, base.total_maps, base.rate) == (
        grown.close_maps, grown.total_maps, grown.rate,
    )
    assert base.close_maps == 2
    assert base.total_maps == 2


def test_global_ot_rate_leakage_strict_boundary():
    # The poison maps must never change the league-wide OT pool.
    base_m, base_p = _build(_base_events())
    poison_m, poison_p = _build(_poison_events())
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)

    base = closeness.global_ot_rate(QUERY, base_m, base_p)
    grown = closeness.global_ot_rate(QUERY, grown_m, grown_p)
    assert (base.events, base.games, base.rate) == (grown.events, grown.games, grown.rate)


def test_team_ot_rate_leakage_strict_boundary():
    # The poison maps must never change team A's shrunk OT rate.
    base_m, base_p = _build(_base_events())
    poison_m, poison_p = _build(_poison_events())
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)

    base = closeness.team_ot_rate("A", QUERY, base_m, base_p)
    grown = closeness.team_ot_rate("A", QUERY, grown_m, grown_p)
    assert (base.events, base.games, base.mean) == (grown.events, grown.games, grown.mean)


def test_map_variance_leakage_strict_boundary():
    # The poison maps (both "Haven", at/after the cutoff) must never
    # change Haven's margin variance, which stays over the two
    # strictly-before Haven maps (margins 2 and 6 -> 8.0).
    base_m, base_p = _build(_base_events())
    poison_m, poison_p = _build(_poison_events())
    grown_m = pd.concat([base_m, poison_m], ignore_index=True)
    grown_p = pd.concat([base_p, poison_p], ignore_index=True)

    base = closeness.map_round_margin_variance("Haven", QUERY, base_m, base_p)
    grown = closeness.map_round_margin_variance("Haven", QUERY, grown_m, grown_p)
    assert base.n == 2
    assert base.variance == pytest.approx(8.0)
    assert (base.n, base.variance) == (grown.n, grown.variance)


def test_strictly_earlier_row_changes_the_estimate():
    # The flip side of the leakage proof: adding a strictly-earlier map
    # does change the estimate (proving the earlier row is the only
    # thing that ever moves it).
    base_m, base_p = _build(_base_events())
    base = closeness.team_close_map_frequency("A", QUERY, base_m, base_p)

    earlier_m, earlier_p = _build(
        [
            {"match_id": "m0", "date": "2025-12-31T10:00:00", "team1_id": "A",
             "team2_id": "C", "map_name": "Haven", "team1_score": 13,
             "team2_score": 11},
        ]
    )
    grown_m = pd.concat([base_m, earlier_m], ignore_index=True)
    grown_p = pd.concat([base_p, earlier_p], ignore_index=True)
    grown = closeness.team_close_map_frequency("A", QUERY, grown_m, grown_p)
    assert grown.total_maps == base.total_maps + 1
    assert grown.close_maps == base.close_maps + 1


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
def test_real_data_smoke_sane_numbers():
    # The no-leakage + sanity check at real v1 scale: query the most
    # frequently appearing team just after the dataset's latest date and
    # verify counts agree with a direct maps_as_of tally, the global OT
    # rate is strictly inside (0, 1), the shrunk team OT mean is also
    # inside (0, 1), and a real map's variance has n >= 0 with a
    # non-NaN variance whenever n > 1.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    direct = asof.maps_as_of(team_id, query, matches_df, maps_df)

    freq = closeness.team_close_map_frequency(team_id, query, matches_df, maps_df)
    assert freq.total_maps == len(direct)
    assert freq.close_maps <= freq.total_maps

    ot = closeness.team_ot_rate(team_id, query, matches_df, maps_df)
    assert ot.games == len(direct)
    assert 0.0 < ot.mean < 1.0

    glob = closeness.global_ot_rate(query, matches_df, maps_df)
    assert glob.games > 0
    assert 0.0 < glob.rate < 1.0

    a_map = maps_df["map_name"].iloc[0]
    var = closeness.map_round_margin_variance(a_map, query, matches_df, maps_df)
    assert var.n >= 0
    if var.n > 1:
        assert not math.isnan(var.variance)


# --------------------------------------------------------------------------
# batched closeness parity (task 052)
# --------------------------------------------------------------------------


def _closeness_parity_fixture():
    """Build a small multi-team, multi-map fixture plus a row table.

    Two teams (``A``/``B``) over four completed matches/maps across
    four dates, including an OT map (``13-11`` — wait, an OT map needs
    ``min(score) >= 12``; ``13-11`` has min 11, so the OT maps here are
    the ``13-12`` ones) and a close map (``13-12`` has abs margin 1,
    OT by the ``min >= 12`` rule). Dates deliberately repeat across
    maps of different matches to exercise same-date as-of resolution.

    Returns:
        A ``(matches_df, maps_df, rows_df)`` tuple; ``rows_df`` has
        ``team1_id, team2_id, map_name, date`` columns.

    Raises:
        Nothing.
    """
    matches_rows = [
        {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "A",
         "team2_id": "B", "status": "completed"},
        {"match_id": "m2", "date": "2026-01-02T10:00:00", "team1_id": "B",
         "team2_id": "A", "status": "completed"},
        {"match_id": "m3", "date": "2026-01-03T10:00:00", "team1_id": "A",
         "team2_id": "B", "status": "completed"},
        {"match_id": "m4", "date": "2026-01-05T10:00:00", "team1_id": "B",
         "team2_id": "A", "status": "completed"},
    ]
    maps_rows = [
        {"match_id": "m1", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m2", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 12, "winner": "B"},
        {"match_id": "m3", "map_index": 0, "map_name": "haven",
         "team1_score": 13, "team2_score": 11, "winner": "A"},
        {"match_id": "m3", "map_index": 1, "map_name": "Haven",
         "team1_score": 13, "team2_score": 5, "winner": "B"},
        {"match_id": "m4", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 6, "winner": "A"},
    ]
    matches_df = _matches_df(matches_rows)
    maps_df = _maps_df(maps_rows)
    rows_df = pd.DataFrame(
        [
            {"team1_id": "A", "team2_id": "B", "map_name": "Haven",
             "date": "2026-01-02T10:00:00"},
            {"team1_id": "A", "team2_id": "B", "map_name": "Bind",
             "date": "2026-01-03T10:00:00"},
            {"team1_id": "A", "team2_id": "B", "map_name": "haven",
             "date": "2026-01-04T10:00:00"},
            {"team1_id": "A", "team2_id": "B", "map_name": "Bind",
             "date": "2026-01-05T10:00:00"},
            {"team1_id": "A", "team2_id": "B", "map_name": "Haven",
             "date": "2026-01-06T10:00:00"},
        ]
    )
    return matches_df, maps_df, rows_df


def test_batched_ot_rate_diff_bit_exact_parity():
    # batched_ot_rate_diff reproduces, element-for-element, the looped
    # single-row team_ot_rate means per row (no tolerance), including
    # rows whose teams have zero prior maps (full shrinkage to the
    # global pooled prior at that row's own date).
    matches_df, maps_df, rows_df = _closeness_parity_fixture()
    expected = np.zeros(len(rows_df))
    for i, row in enumerate(rows_df.itertuples(index=False)):
        mean_a = closeness.team_ot_rate(
            row.team1_id, row.date, matches_df, maps_df,
            k=closeness.DEFAULT_OT_K,
        ).mean
        mean_b = closeness.team_ot_rate(
            row.team2_id, row.date, matches_df, maps_df,
            k=closeness.DEFAULT_OT_K,
        ).mean
        expected[i] = mean_a - mean_b
    got = closeness.batched_ot_rate_diff(rows_df, matches_df, maps_df)
    assert got.shape == (len(rows_df),)
    assert np.array_equal(got, expected)


def test_batched_map_round_margin_variance_bit_exact_parity():
    # The variance path must reproduce the single-row estimator's
    # np.var(ddof=1) output bit-for-bit — same values, same order,
    # same call — including the NaN (n <= 1) degenerate rows.
    matches_df, maps_df, rows_df = _closeness_parity_fixture()
    expected = np.zeros(len(rows_df))
    for i, row in enumerate(rows_df.itertuples(index=False)):
        v = closeness.map_round_margin_variance(
            row.map_name, row.date, matches_df, maps_df
        ).variance
        expected[i] = v
    got = closeness.batched_map_round_margin_variance(rows_df, matches_df, maps_df)
    assert got.shape == (len(rows_df),)
    # NaN-vs-NaN counts as equal here (array_equal's equal_nan), so the
    # degenerate n <= 1 rows align too; every non-NaN value is exact.
    assert np.array_equal(got, expected, equal_nan=True)


def test_batched_map_round_margin_variance_nan_rows_match_single_row():
    # The NaN degenerate rows (n <= 1 prior maps on that map) must line
    # up positionally with the single-row path's NaN values.
    matches_df, maps_df, rows_df = _closeness_parity_fixture()
    got = closeness.batched_map_round_margin_variance(rows_df, matches_df, maps_df)
    # Row 0 queries Haven at m1's own date: no strictly-prior Haven map
    # exists anywhere -> n == 0 -> NaN.
    assert np.isnan(got[0])
    # Row 2 queries Haven at 2026-01-04: exactly m1's Haven is prior
    # (m3's two Havens are dated 01-03 and excluded? no -- 01-03 < 01-04
    # so both are prior) -> n == 3.
    assert not np.isnan(got[2])

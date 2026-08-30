"""Tests for head-to-head and match-context features (M17).

Covers the shrunk H2H estimator (zero-history full shrinkage, a
hand-computed 3-2 record, the a/b symmetry invariant, the per-map
filter), the fail-loud event-stage parser, days-since-last-match
(including the empty-history ``None`` and the strict-``<`` same-day
exclusion), roster change (identical / 1-swap / 3-swap / insufficient
history, plus the decay at ``days_since_change == 0`` and ``== half_life``),
the leakage-safety proof across all as-of features, the ``features/`` ->
no-``drivers/`` layering rule, and a skip-guarded real-``data/v1`` sanity
pass.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from features import h2h_context

QUERY = "2026-01-05T10:00:00"

_MATCHES_COLS = [
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "team1_name",
    "team2_name",
    "event_name",
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

_ROSTER_A = ["p1", "p2", "p3", "p4", "p5"]


def _matches_df(rows):
    """Build a matches table with the fixed M8 + names + event_name columns.

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

    Mirrors :func:`_matches_df` for the player-map stats side.

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


def _five_match_pair():
    """Build the shared A-vs-B H2H fixture: A is 3-2 across 5 maps.

    Team ``A`` ("Alpha") plays team ``B`` ("Beta") five times, one map per
    match: A wins ``m1`` (Haven), ``m3`` (Haven) and ``m5`` (Bind), loses
    ``m2`` (Haven) and ``m4`` (Bind) — an overall 3-2 record, with 2 wins
    / 1 loss on Haven and 1 win / 1 loss on Bind. Orientations alternate
    so both ``team_is_team1`` values are exercised (A is team1 in m1/m3/m4,
    team2 in m2/m5).

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df` /
        :func:`_maps_df`.

    Raises:
        Nothing.
    """
    matches_rows = [
        {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "A",
         "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m2", "date": "2026-01-02T10:00:00", "team1_id": "B",
         "team2_id": "A", "team1_name": "Beta", "team2_name": "Alpha",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m3", "date": "2026-01-03T10:00:00", "team1_id": "A",
         "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m4", "date": "2026-01-04T10:00:00", "team1_id": "A",
         "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m5", "date": "2026-01-04T12:00:00", "team1_id": "B",
         "team2_id": "A", "team1_name": "Beta", "team2_name": "Alpha",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
    ]
    maps_rows = [
        {"match_id": "m1", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m2", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 9, "winner": "B"},
        {"match_id": "m3", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 7, "winner": "A"},
        {"match_id": "m4", "map_index": 0, "map_name": "Bind",
         "team1_score": 8, "team2_score": 13, "winner": "B"},
        {"match_id": "m5", "map_index": 0, "map_name": "Bind",
         "team1_score": 11, "team2_score": 13, "winner": "A"},
    ]
    return _matches_df(matches_rows), _maps_df(maps_rows)


def _roster_fixture(events):
    """Build matches/maps/pms frames for A-vs-B roster-change fixtures.

    Each event is one completed match with one finished map (team A =
    ``"Alpha"`` as team1, team B = ``"Beta"`` as team2, A wins 13-8) and
    exactly the given 5-player Alpha roster plus a fixed 5-player Beta
    roster. The Alpha roster is what changes between events.

    Args:
        events: A list of dicts, each with ``match_id``, ``date`` and
            ``roster`` (an iterable of 5 Alpha player names).

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple built by
        :func:`_matches_df` / :func:`_maps_df` / :func:`_pms_df`.

    Raises:
        Nothing (pandas ``ValueError`` on a malformed event surfaces
        as-is).
    """
    matches_rows = []
    maps_rows = []
    pms_rows = []
    for e in events:
        mid = e["match_id"]
        matches_rows.append(
            {
                "match_id": mid,
                "date": e["date"],
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "event_name": "VCT 2026: EMEA Stage 1",
                "status": "completed",
            }
        )
        maps_rows.append(
            {
                "match_id": mid,
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "A",
            }
        )
        for p in e["roster"]:
            pms_rows.append(
                {
                    "match_id": mid,
                    "map_index": 0,
                    "player_name": p,
                    "team_name": "Alpha",
                    "rating": 1.0,
                    "acs": 200.0,
                }
            )
        for i in range(5):
            pms_rows.append(
                {
                    "match_id": mid,
                    "map_index": 0,
                    "player_name": f"q{i}",
                    "team_name": "Beta",
                    "rating": 0.5,
                    "acs": 50.0,
                }
            )
    return _matches_df(matches_rows), _maps_df(maps_rows), _pms_df(pms_rows)


# --------------------------------------------------------------------------
# H2H shrinkage
# --------------------------------------------------------------------------


def test_h2h_zero_history_full_shrinkage():
    # The pair has never played: games == 0 collapses the posterior fully
    # to the flat 0.5 prior, and raw_rate == prior exactly.
    empty_m = _matches_df([])
    empty_p = _maps_df([])
    res = h2h_context.team_pair_h2h("A", "B", QUERY, empty_m, empty_p)
    assert res.wins == 0
    assert res.games == 0
    assert res.prior == pytest.approx(0.5)
    assert res.raw_rate == pytest.approx(0.5)
    assert res.mean == pytest.approx(0.5)
    assert res.alpha == pytest.approx(0.0 + 20.0 * 0.5)
    assert res.beta == pytest.approx(0.0 + 20.0 * 0.5)


def test_h2h_three_two_record_hand_computed():
    # A is 3-2 against B. With k=20 and prior 0.5 the hand-computed
    # posterior is (3 + 10) / (5 + 20) = 13/25 = 0.52.
    m, p = _five_match_pair()
    res = h2h_context.team_pair_h2h("A", "B", QUERY, m, p, k=20.0)
    assert res.wins == 3
    assert res.games == 5
    assert res.raw_rate == pytest.approx(3 / 5)
    assert res.alpha == pytest.approx(3 + 20.0 * 0.5)
    assert res.beta == pytest.approx(2 + 20.0 * 0.5)
    assert res.mean == pytest.approx((3 + 20.0 * 0.5) / (5 + 20.0))
    assert res.variance == pytest.approx(
        (res.alpha * res.beta) / ((res.alpha + res.beta) ** 2 * (res.alpha + res.beta + 1.0))
    )


def test_h2h_symmetry_complementary_counts_and_means():
    # h2h(b, a) must be the exact mirror of h2h(a, b): same games,
    # complementary wins, and — because the prior is the flat 0.5 — the
    # two shrunk means sum to exactly 1.0 in floating point.
    m, p = _five_match_pair()
    ab = h2h_context.team_pair_h2h("A", "B", QUERY, m, p, k=20.0)
    ba = h2h_context.team_pair_h2h("B", "A", QUERY, m, p, k=20.0)
    assert ba.games == ab.games == 5
    assert ab.wins + ba.wins == 5
    assert ab.mean + ba.mean == pytest.approx(1.0)


def test_h2h_per_map_filter_excludes_other_maps():
    # Overall A is 3-2; on Haven it is 2-1, on Bind 1-1. The per-map
    # filter (with case/whitespace normalisation) must restrict the
    # history to only the named map.
    m, p = _five_match_pair()
    overall = h2h_context.team_pair_h2h("A", "B", QUERY, m, p, k=20.0)
    haven = h2h_context.team_pair_h2h("A", "B", QUERY, m, p, map_name=" haven ", k=20.0)
    bind = h2h_context.team_pair_h2h("A", "B", QUERY, m, p, map_name="Bind", k=20.0)
    assert overall.games == 5 and overall.wins == 3
    assert haven.games == 3 and haven.wins == 2
    assert bind.games == 2 and bind.wins == 1


# --------------------------------------------------------------------------
# event stage
# --------------------------------------------------------------------------


def test_event_stage_parses_expected_tokens():
    # The two real v1 event-name shapes parse to their integer stages.
    assert h2h_context.event_stage("VCT 2026: EMEA Stage 1") == 1
    assert h2h_context.event_stage("VCT 2026: EMEA Stage 2") == 2


@pytest.mark.parametrize(
    "bad",
    [
        "VCT 2026: EMEA Grand Final",
        "VCT 2026: EMEA Stage",  # token present but no number
        "VCT 2026: EMEA stage 1",  # wrong casing
        "VCT 2026: EMEA Stages 1",  # extra letter after "Stage"
    ],
)
def test_event_stage_unparseable_raises(bad):
    # Any string without a parseable "Stage N" token must fail loudly.
    with pytest.raises(ValueError):
        h2h_context.event_stage(bad)


def test_event_stage_non_string_raises():
    # A non-string event_name is rejected with TypeError, never coerced.
    with pytest.raises(TypeError):
        h2h_context.event_stage(None)


def test_match_event_stage_lookup_and_missing_match():
    # match_event_stage composes the lookup with event_stage; a match id
    # absent from the table fails loudly rather than defaulting.
    m = _matches_df(
        [
            {
                "match_id": "m1",
                "date": "2026-01-01T10:00:00",
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "event_name": "VCT 2026: EMEA Stage 2",
                "status": "completed",
            }
        ]
    )
    assert h2h_context.match_event_stage("m1", m) == 2
    with pytest.raises(ValueError, match="not present"):
        h2h_context.match_event_stage("nope", m)


# --------------------------------------------------------------------------
# days since last match
# --------------------------------------------------------------------------


def test_days_since_last_match_hand_computed():
    # Two prior matches (Jan 1 and Jan 3); queried Jan 5 10:00, so the
    # most recent prior match is exactly 2 days earlier.
    m = _matches_df(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "A",
             "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
             "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
            {"match_id": "m2", "date": "2026-01-03T10:00:00", "team1_id": "A",
             "team2_id": "C", "team1_name": "Alpha", "team2_name": "Gamma",
             "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        ]
    )
    assert h2h_context.days_since_last_match("A", QUERY, m) == 2


def test_days_since_last_match_empty_history_none():
    # An unseen team has no prior match -> None (honest sentinel).
    m = _matches_df(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "A",
             "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
             "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        ]
    )
    assert h2h_context.days_since_last_match("Z", QUERY, m) is None


def test_days_since_last_match_same_day_cutoff_excluded():
    # A match dated exactly at the query is excluded by the strict <
    # boundary, so the team has empty history and the result is None.
    m = _matches_df(
        [
            {"match_id": "m1", "date": QUERY, "team1_id": "A",
             "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
             "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        ]
    )
    assert h2h_context.days_since_last_match("A", QUERY, m) is None


# --------------------------------------------------------------------------
# roster change
# --------------------------------------------------------------------------


def test_roster_change_identical_rosters():
    # Two most-recent maps with the exact same 5-player roster: no change,
    # similarity exactly 1.0, and no decay (populated only on a change).
    m, p, s = _roster_fixture(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "roster": _ROSTER_A},
            {"match_id": "m2", "date": "2026-01-02T10:00:00", "roster": _ROSTER_A},
        ]
    )
    res = h2h_context.team_roster_change("A", QUERY, m, p, s)
    assert res.changed is False
    assert res.similarity == pytest.approx(1.0)
    assert res.decay_multiplier is None
    assert res.changed_as_of_date is None


def test_roster_change_single_player_swap_not_a_change():
    # One of five players differs: Jaccard 4/6 ~= 0.67 >= 0.6, so this is
    # a stand-in sub, not a roster change.
    m, p, s = _roster_fixture(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "roster": ["p1", "p2", "p3", "p4", "p5"]},
            {"match_id": "m2", "date": "2026-01-02T10:00:00", "roster": ["p1", "p2", "p3", "p4", "p6"]},
        ]
    )
    res = h2h_context.team_roster_change("A", QUERY, m, p, s)
    assert res.changed is False
    assert res.similarity == pytest.approx(4 / 6)


def test_roster_change_three_player_swap_is_change():
    # Three of five players differ: Jaccard 2/8 = 0.25 < 0.6 -> change.
    m, p, s = _roster_fixture(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "roster": ["p1", "p2", "p3", "p4", "p5"]},
            {"match_id": "m2", "date": "2026-01-02T10:00:00", "roster": ["p1", "p2", "p6", "p7", "p8"]},
        ]
    )
    res = h2h_context.team_roster_change("A", QUERY, m, p, s)
    assert res.changed is True
    assert res.similarity == pytest.approx(2 / 8)
    assert res.changed_as_of_date == "2026-01-02T10:00:00"


def test_roster_change_insufficient_history_unknown():
    # Fewer than two evaluable maps -> changed is None (unknown), not False.
    m, p, s = _roster_fixture(
        [{"match_id": "m1", "date": "2026-01-01T10:00:00", "roster": _ROSTER_A}]
    )
    res = h2h_context.team_roster_change("A", QUERY, m, p, s)
    assert res.changed is None
    assert res.similarity is None
    assert res.decay_multiplier is None
    assert res.changed_as_of_date is None


def test_roster_change_decay_at_zero_and_half_life():
    # days_since_change == 0 (same calendar day, query 1h later) -> decay
    # 0.5**0 == 1.0; days_since_change == 14 (default half-life) -> 0.5.
    m0, p0, s0 = _roster_fixture(
        [
            {"match_id": "m1", "date": "2026-01-04T09:00:00", "roster": ["p1", "p2", "p3", "p4", "p5"]},
            {"match_id": "m2", "date": "2026-01-04T10:00:00", "roster": ["p1", "p2", "p6", "p7", "p8"]},
        ]
    )
    res0 = h2h_context.team_roster_change("A", "2026-01-04T11:00:00", m0, p0, s0)
    assert res0.changed is True
    assert res0.decay_multiplier == pytest.approx(1.0)

    m14, p14, s14 = _roster_fixture(
        [
            {"match_id": "m1", "date": "2025-12-31T10:00:00", "roster": ["p1", "p2", "p3", "p4", "p5"]},
            {"match_id": "m2", "date": "2026-01-01T10:00:00", "roster": ["p1", "p2", "p6", "p7", "p8"]},
        ]
    )
    res14 = h2h_context.team_roster_change("A", "2026-01-15T10:00:00", m14, p14, s14)
    assert res14.changed is True
    assert res14.decay_multiplier == pytest.approx(0.5)


# --------------------------------------------------------------------------
# leakage safety
# --------------------------------------------------------------------------


def test_leakage_future_rows_excluded_across_features():
    # Non-tautological leakage proof across the three as-of features
    # (H2H mean, days-since, roster-change): a poison A-vs-B map dated
    # exactly at the query must leave every value identical to the
    # poison-free fixture, while moving the cutoff past it changes all
    # three values (proving the features do respond to real history and
    # ignore only at/after-query rows). event_stage is not tested here:
    # it is a pure str -> int parse with no history surface.
    base_matches = [
        {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
         "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "B",
         "team2_id": "A", "team1_name": "Beta", "team2_name": "Alpha",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
    ]
    base_maps = [
        {"match_id": "m1", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m2", "map_index": 0, "map_name": "Bind",
         "team1_score": 8, "team2_score": 13, "winner": "A"},
    ]
    base_pms = []
    for mid in ("m1", "m2"):
        for p in _ROSTER_A:
            base_pms.append({"match_id": mid, "map_index": 0, "player_name": p,
                             "team_name": "Alpha", "rating": 1.0, "acs": 200.0})
        for i in range(5):
            base_pms.append({"match_id": mid, "map_index": 0, "player_name": f"q{i}",
                             "team_name": "Beta", "rating": 0.5, "acs": 50.0})

    base_m = _matches_df(base_matches)
    base_p = _maps_df(base_maps)
    base_s = _pms_df(base_pms)

    poison_matches = [
        {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "A",
         "team2_id": "B", "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 2", "status": "completed"},
    ]
    poison_maps = [
        {"match_id": "m3", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 0, "winner": "A"},
    ]
    poison_pms = []
    for p in ["x1", "x2", "x3", "x4", "x5"]:
        poison_pms.append({"match_id": "m3", "map_index": 0, "player_name": p,
                           "team_name": "Alpha", "rating": 1.0, "acs": 999.0})
    for i in range(5):
        poison_pms.append({"match_id": "m3", "map_index": 0, "player_name": f"q{i}",
                           "team_name": "Beta", "rating": 0.5, "acs": 50.0})

    grown_m = pd.concat([base_m, _matches_df(poison_matches)], ignore_index=True)
    grown_p = pd.concat([base_p, _maps_df(poison_maps)], ignore_index=True)
    grown_s = pd.concat([base_s, _pms_df(poison_pms)], ignore_index=True)

    query = "2026-01-03T00:00:00"
    base_h2h = h2h_context.team_pair_h2h("A", "B", query, base_m, base_p, k=20.0)
    grown_h2h = h2h_context.team_pair_h2h("A", "B", query, grown_m, grown_p, k=20.0)
    assert (grown_h2h.wins, grown_h2h.games, grown_h2h.mean) == (
        base_h2h.wins, base_h2h.games, base_h2h.mean
    )
    assert base_h2h.wins == 2 and base_h2h.games == 2

    assert h2h_context.days_since_last_match("A", query, base_m) == 1
    assert h2h_context.days_since_last_match("A", query, grown_m) == 1

    base_rc = h2h_context.team_roster_change("A", query, base_m, base_p, base_s)
    grown_rc = h2h_context.team_roster_change("A", query, grown_m, grown_p, grown_s)
    assert (grown_rc.changed, grown_rc.similarity) == (base_rc.changed, base_rc.similarity)
    assert base_rc.changed is False and base_rc.similarity == pytest.approx(1.0)

    # Moving the cutoff past the poison row changes all three values.
    later = "2026-01-05T00:00:00"
    later_h2h = h2h_context.team_pair_h2h("A", "B", later, grown_m, grown_p, k=20.0)
    assert (later_h2h.wins, later_h2h.games) == (3, 3)
    assert later_h2h.mean != pytest.approx(base_h2h.mean)

    assert h2h_context.days_since_last_match("A", later, grown_m) == 2
    later_rc = h2h_context.team_roster_change("A", later, grown_m, grown_p, grown_s)
    assert later_rc.changed is True
    assert later_rc.similarity == pytest.approx(0.0)


# --------------------------------------------------------------------------
# layering + real-data smoke
# --------------------------------------------------------------------------


def test_no_drivers_import_in_h2h_context():
    # The features/ layering rule: feature modules must not import from
    # drivers/. Assert the literal import is absent from the module source.
    source = Path(h2h_context.__file__).read_text(encoding="utf-8")
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
def test_real_data_sanity_sane_numbers():
    # Sanity at real v1 scale: every real event_name parses to stage 1 or
    # 2; a real frequently-meeting pair yields a finite in-(0,1) shrunk
    # H2H mean with the a/b symmetry exact; days_since_last_match is None
    # at a team's own first match timestamp and non-negative later; and
    # roster change is well-typed with an in-[0,1] similarity when known.
    matches, maps, pms = h2h_context.load_h2h_context_tables("v1")

    for name in matches["event_name"].unique():
        assert h2h_context.event_stage(name) in (1, 2)

    joined = maps.merge(matches[["match_id", "team1_id", "team2_id", "date"]], on="match_id")
    pair = joined.apply(
        lambda r: tuple(sorted((r["team1_id"], r["team2_id"]))), axis=1
    )
    a, b = pair.value_counts().idxmax()
    latest = pd.to_datetime(matches["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    ab = h2h_context.team_pair_h2h(a, b, query, matches, maps)
    ba = h2h_context.team_pair_h2h(b, a, query, matches, maps)
    assert math.isfinite(ab.mean) and 0.0 < ab.mean < 1.0
    assert ab.games > 0
    assert ba.games == ab.games
    assert ab.wins + ba.wins == ab.games
    assert ab.mean + ba.mean == pytest.approx(1.0)

    appearances = pd.concat([matches["team1_id"], matches["team2_id"]]).dropna()
    team = appearances.value_counts().idxmax()
    team_matches = matches[
        (matches["team1_id"] == team) | (matches["team2_id"] == team)
    ]
    first = pd.to_datetime(team_matches["date"]).min().isoformat()
    assert h2h_context.days_since_last_match(team, first, matches) is None
    later = h2h_context.days_since_last_match(team, query, matches)
    assert later is not None and later >= 0

    rc = h2h_context.team_roster_change(team, query, matches, maps, pms)
    assert rc.changed in (True, False, None)
    if rc.changed is not None:
        assert rc.similarity is not None and 0.0 <= rc.similarity <= 1.0

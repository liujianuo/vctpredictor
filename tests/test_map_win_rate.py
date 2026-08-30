"""Tests for Bayesian-shrunk map win rates (M13).

Covers the estimator pair (``team_overall_win_rate`` /
``team_map_win_rate``), the Beta-posterior dataclasses, the
shrinkage-behaviour contracts (zero games -> full shrinkage, few games
-> pulled toward the prior, many games -> close to raw, monotonic in k,
posterior sanity), the ``select_k`` cross-validation harness, and the
leakage-safety proof (a map dated exactly at / after the cutoff never
enters the estimate; only strictly-earlier rows do). A skip-guarded
smoke test repeats the no-leakage assertion at real ``data/v1`` scale.
"""

from pathlib import Path

import pandas as pd
import pytest

from utils import asof
from utils import map_win_rate as mwr

QUERY_DATE = "2026-01-03T00:00:00"

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


def _core_tables():
    """Build the shared estimator fixture with equal/after cutoff rows.

    Team ``T1`` plays: ``m1`` (Haven, as team1, a win) and ``m2`` (Bind,
    as team2, a loss) strictly before :data:`QUERY_DATE`; ``m3`` (Haven)
    exactly at :data:`QUERY_DATE`; and ``m4`` (Haven) after it. This
    exercises both orientations of the ``team_is_team1`` flag (T1 is
    team1 in m1, team2 in m2) and gives the leakage tests their
    exactly-equal and after rows to exclude.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df` /
        :func:`_maps_df`.

    Raises:
        Nothing.
    """
    matches_rows = [
        {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "T1",
         "team2_id": "T2", "status": "completed"},
        {"match_id": "m2", "date": "2026-01-02T10:00:00", "team1_id": "T2",
         "team2_id": "T1", "status": "completed"},
        {"match_id": "m3", "date": QUERY_DATE, "team1_id": "T1",
         "team2_id": "T3", "status": "completed"},
        {"match_id": "m4", "date": "2026-01-04T10:00:00", "team1_id": "T1",
         "team2_id": "T2", "status": "completed"},
    ]
    maps_rows = [
        {"match_id": "m1", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "T1"},
        {"match_id": "m2", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 9, "winner": "T2"},
        {"match_id": "m3", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "T1"},
        {"match_id": "m4", "map_index": 0, "map_name": "Haven",
         "team1_score": 8, "team2_score": 13, "winner": "T2"},
    ]
    return _matches_df(matches_rows), _maps_df(maps_rows)


def _team_tables(segments, team_id="T"):
    """Build a single-team history from ``(map_name, wins, games)`` segments.

    Produces one completed match (and one finished map) per game, in
    segment order, with the queried team always ``team1`` so the
    orientation flag is uniformly ``True`` (the orientation *variation*
    is covered by :func:`_core_tables`). Dates are one hour apart from a
    fixed base, and the returned query date is one hour after the last
    map, so every row is strictly before the cutoff.

    Args:
        segments: An iterable of ``(map_name, wins, games)`` triples;
            within a segment the first ``wins`` maps are wins and the
            remaining ``games - wins`` are losses (13-8 / 8-13 scores).
        team_id: The queried team's id to place on the ``team1`` side
            (default ``"T"``).

    Returns:
        A ``(matches_df, maps_df, query_date)`` tuple; ``query_date`` is
        an ISO string one hour after the final map, so calling the
        estimator at it sees the whole history.

    Raises:
        ValueError: If ``wins > games`` or ``wins < 0`` for any segment
            (the ``range(wins)`` slicing would silently mis-shape the
            fixture otherwise).
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    match_rows = []
    map_rows = []
    i = 0
    for map_name, wins, games in segments:
        if wins < 0 or wins > games:
            raise ValueError(
                f"segment ({map_name!r}, {wins}, {games}) needs "
                f"0 <= wins <= games"
            )
        for j in range(games):
            mid = f"m{i:04d}"
            ts = base + pd.Timedelta(hours=i)
            won = j < wins
            match_rows.append(
                {
                    "match_id": mid,
                    "date": ts.isoformat(),
                    "team1_id": team_id,
                    "team2_id": f"opp{i}",
                    "status": "completed",
                }
            )
            t1s, t2s = (13, 8) if won else (8, 13)
            map_rows.append(
                {
                    "match_id": mid,
                    "map_index": 0,
                    "map_name": map_name,
                    "team1_score": t1s,
                    "team2_score": t2s,
                    "winner": team_id if won else f"opp{i}",
                }
            )
            i += 1
    query = (base + pd.Timedelta(hours=i)).isoformat()
    return _matches_df(match_rows), _maps_df(map_rows), query


def _low_data_scenario():
    """Build the low-data CV scenario (a large ``k`` should win).

    Team ``L`` seeds its warm-up history with two Lotus wins and a
    balanced Haven pair (so its overall prior is a sensible ~0.75), then
    plays one validation Lotus map that it *loses* — a small, misleading
    Lotus sample (raw 1.0) that shrinkage should pull back toward the
    prior. Every other team appears exactly once, so its estimate is
    exactly 0.5 for every ``k`` (zero as-of history -> full shrinkage),
    making those instances ``k``-neutral filler that satisfies the
    split/fold machinery's size floor without affecting the argmin.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 30 chronological matches
        (and 30 maps) built by :func:`_matches_df` / :func:`_maps_df`.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")

    def stamp(i):
        return (base + pd.Timedelta(hours=i)).isoformat()

    match_rows = []
    map_rows = []

    # Positions 0-3: L's history (warm-up, never validated).
    history = [
        ("Lotus", 13, 8),   # L wins
        ("Lotus", 13, 8),   # L wins
        ("Haven", 13, 8),   # L wins
        ("Haven", 8, 13),   # L loses
    ]
    for i, (map_name, t1s, t2s) in enumerate(history):
        mid = f"h{i}"
        match_rows.append(
            {"match_id": mid, "date": stamp(i), "team1_id": "L",
             "team2_id": f"ho{i}", "status": "completed"}
        )
        map_rows.append(
            {"match_id": mid, "map_index": 0, "map_name": map_name,
             "team1_score": t1s, "team2_score": t2s,
             "winner": "L" if t1s > t2s else f"ho{i}"}
        )

    # Positions 4-9: warm-up throwaway filler (unique teams, k-neutral).
    for i in range(4, 10):
        mid = f"w{i}"
        match_rows.append(
            {"match_id": mid, "date": stamp(i), "team1_id": f"wa{i}",
             "team2_id": f"wb{i}", "status": "completed"}
        )
        map_rows.append(
            {"match_id": mid, "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": f"wa{i}"}
        )

    # Position 10: the signal — L loses a Lotus validation map.
    match_rows.append(
        {"match_id": "sig", "date": stamp(10), "team1_id": "L",
         "team2_id": "so", "status": "completed"}
    )
    map_rows.append(
        {"match_id": "sig", "map_index": 0, "map_name": "Lotus",
         "team1_score": 8, "team2_score": 13, "winner": "so"}
    )

    # Positions 11-25: validation throwaway filler.
    for i in range(11, 26):
        mid = f"v{i}"
        match_rows.append(
            {"match_id": mid, "date": stamp(i), "team1_id": f"va{i}",
             "team2_id": f"vb{i}", "status": "completed"}
        )
        map_rows.append(
            {"match_id": mid, "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": f"va{i}"}
        )

    # Positions 26-29: test-region throwaway filler (never scored).
    for i in range(26, 30):
        mid = f"t{i}"
        match_rows.append(
            {"match_id": mid, "date": stamp(i), "team1_id": f"ta{i}",
             "team2_id": f"tb{i}", "status": "completed"}
        )
        map_rows.append(
            {"match_id": mid, "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": f"ta{i}"}
        )

    return _matches_df(match_rows), _maps_df(map_rows)


def _high_data_scenario():
    """Build the high-data CV scenario (a small ``k`` should win).

    Team ``H`` plays 200 matches against one-off opponents: 100 Haven
    (80 wins) and 100 Lotus (0 wins), interleaved chronologically. Both
    per-map rates are therefore backed by large, reliable samples while
    the overall prior (~0.4) is a muddle of the two, so a small ``k``
    (trusting the per-map raw rate) scores strictly better than a large
    one (pulling every map toward the muddled prior). Opponents are
    unique per match and thus ``k``-neutral (their estimate is 0.5 for
    every ``k``).

    Returns:
        A ``(matches_df, maps_df)`` tuple of 200 chronological matches
        (and 200 maps) built by :func:`_matches_df` / :func:`_maps_df`.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    match_rows = []
    map_rows = []
    i = 0

    def add(map_name, won):
        nonlocal i
        mid = f"m{i:04d}"
        ts = (base + pd.Timedelta(hours=i)).isoformat()
        i += 1
        match_rows.append(
            {"match_id": mid, "date": ts, "team1_id": "H",
             "team2_id": f"o{i}", "status": "completed"}
        )
        t1s, t2s = (13, 8) if won else (8, 13)
        map_rows.append(
            {"match_id": mid, "map_index": 0, "map_name": map_name,
             "team1_score": t1s, "team2_score": t2s,
             "winner": "H" if won else f"o{i}"}
        )

    # Warm-up 30: 15 Haven (12W/3L) then 15 Lotus (0W).
    for _ in range(12):
        add("Haven", True)
    for _ in range(3):
        add("Haven", False)
    for _ in range(15):
        add("Lotus", False)
    # Validation 140: 70 Haven (56W/14L) then 70 Lotus (0W).
    for _ in range(56):
        add("Haven", True)
    for _ in range(14):
        add("Haven", False)
    for _ in range(70):
        add("Lotus", False)
    # Test 30: 15 Haven (12W/3L) then 15 Lotus (0W).
    for _ in range(12):
        add("Haven", True)
    for _ in range(3):
        add("Haven", False)
    for _ in range(15):
        add("Lotus", False)

    return _matches_df(match_rows), _maps_df(map_rows)


# --------------------------------------------------------------------------
# team_overall_win_rate
# --------------------------------------------------------------------------


def test_team_overall_win_rate_uses_scores_and_orientation():
    # T1 wins m1 as team1 (score 13) and loses m2 as team2 (score 9);
    # the equal/after rows (m3/m4) are excluded by the as-of boundary.
    # So the overall record is 1 win in 2 games, derived purely from
    # scores + the orientation flag, never from the winner text.
    matches_df, maps_df = _core_tables()
    result = mwr.team_overall_win_rate("T1", QUERY_DATE, matches_df, maps_df)
    assert result.wins == 1
    assert result.games == 2
    assert result.rate == 0.5


def test_team_overall_win_rate_zero_games_is_half():
    # An unseen team (or a cutoff before the team's first match) has no
    # observable rate; the documented default is the maximally
    # uninformative 0.5 rather than an error or a silent 0/1.
    matches_df, maps_df = _core_tables()
    result = mwr.team_overall_win_rate("UNSEEN", QUERY_DATE, matches_df, maps_df)
    assert result.wins == 0
    assert result.games == 0
    assert result.rate == 0.5


def test_team_overall_win_rate_tie_raises():
    # A finished map with equal scores is impossible; the fail-loudly
    # contract raises instead of counting it as a loss.
    matches_df, _ = _core_tables()
    tie_maps = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": 10, "team2_score": 10, "winner": "T1"},
        ]
    )
    with pytest.raises(ValueError, match="tied scores"):
        mwr.team_overall_win_rate("T1", QUERY_DATE, matches_df, tie_maps)


# --------------------------------------------------------------------------
# team_map_win_rate: shrinkage behaviour
# --------------------------------------------------------------------------


def test_team_map_win_rate_zero_games_full_shrinkage():
    # (item 6a) Zero games on the queried map: mean == prior exactly
    # (full shrinkage), and raw_rate is reported as the prior too.
    matches_df, maps_df = _core_tables()
    result = mwr.team_map_win_rate(
        "T1", "Ascent", QUERY_DATE, matches_df, maps_df, 10.0
    )
    assert result.wins == 0
    assert result.games == 0
    assert result.prior == 0.5
    assert result.mean == pytest.approx(result.prior)
    assert result.raw_rate == pytest.approx(result.prior)


def test_team_map_win_rate_few_games_shrinks_toward_prior():
    # (item 6b) 1 win / 1 game (raw 1.0) with prior 0.5 at k=10 lands
    # strictly between prior and raw, closer to the prior.
    matches_df, maps_df = _core_tables()
    result = mwr.team_map_win_rate(
        "T1", "Haven", QUERY_DATE, matches_df, maps_df, 10.0
    )
    assert result.wins == 1
    assert result.games == 1
    assert result.raw_rate == 1.0
    assert result.prior == 0.5
    assert result.prior < result.mean < result.raw_rate
    assert (result.mean - result.prior) < (result.raw_rate - result.mean)


def test_team_map_win_rate_many_games_close_to_raw():
    # (item 6c) 200 wins / 250 games with a differing prior (0.4): the
    # k*prior term is swamped by games, so mean is within a small
    # tolerance of the raw rate 0.8.
    matches_df, maps_df, query = _team_tables(
        [("Haven", 200, 250), ("Bind", 0, 250)]
    )
    result = mwr.team_map_win_rate("T", "Haven", query, matches_df, maps_df, 10.0)
    assert result.wins == 200
    assert result.games == 250
    assert result.raw_rate == 0.8
    assert result.prior == 0.4
    assert abs(result.mean - result.raw_rate) < 0.02


def test_team_map_win_rate_monotonic_toward_prior():
    # (item 6d) Holding wins/games/prior fixed, mean moves monotonically
    # toward the prior as k grows (raw 1.0 > prior 0.5, so it decreases).
    matches_df, maps_df = _core_tables()
    means = [
        mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, maps_df, k).mean
        for k in (1.0, 10.0, 100.0)
    ]
    assert means[0] > means[1] > means[2]
    assert means[2] > 0.5
    assert means[0] < 1.0


def test_team_map_win_rate_posterior_sanity_and_variance_shrinks():
    # (item 6e) alpha/beta/variance are always positive, and variance
    # shrinks as games grows holding prior and k fixed (more data -> a
    # more informative posterior).
    small_m, small_p, q1 = _team_tables([("Haven", 1, 2)])
    large_m, large_p, q2 = _team_tables([("Haven", 100, 200)])
    small = mwr.team_map_win_rate("T", "Haven", q1, small_m, small_p, 10.0)
    large = mwr.team_map_win_rate("T", "Haven", q2, large_m, large_p, 10.0)
    for r in (small, large):
        assert r.alpha > 0.0
        assert r.beta > 0.0
        assert r.variance > 0.0
        assert 0.0 < r.mean < 1.0
    assert large.variance < small.variance


def test_team_map_win_rate_normalizes_map_name():
    # A caller passing " breeze " must match the stored "Breeze" via
    # normalize_map_name, rather than depending on exact case/whitespace.
    matches_df, maps_df, query = _team_tables(
        [("Breeze", 1, 1), ("Haven", 0, 1)]
    )
    result = mwr.team_map_win_rate("T", " breeze ", query, matches_df, maps_df, 10.0)
    assert result.wins == 1
    assert result.games == 1


def test_team_map_win_rate_tie_raises():
    # The map-level tie guard also fires through the shrink estimator.
    matches_df, _ = _core_tables()
    tie_maps = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": 10, "team2_score": 10, "winner": "T1"},
        ]
    )
    with pytest.raises(ValueError, match="tied scores"):
        mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, tie_maps, 10.0)


def test_team_overall_win_rate_null_score_raises():
    # A finished map (winner non-null) with a NaN team1_score must raise
    # ValueError before the tie check, not silently count the row as a
    # loss (NaN compares neither equal nor greater to anything).
    matches_df, _ = _core_tables()
    null_maps = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": float("nan"), "team2_score": 8, "winner": "T1"},
        ]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        mwr.team_overall_win_rate("T1", QUERY_DATE, matches_df, null_maps)


def test_team_map_win_rate_null_score_raises():
    # The same null-score guard fires through the shrink estimator path.
    matches_df, _ = _core_tables()
    null_maps = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": float("nan"), "team2_score": 8, "winner": "T1"},
        ]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, null_maps, 10.0)


def test_collect_validation_instances_null_score_raises():
    # The CV ground-truth collection reads scores from finished maps
    # directly; a NaN score must raise rather than be silently
    # mislabelled as a team2 win (the same bug class the estimator-side
    # guard fixes).
    matches_df = _matches_df(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "T1",
             "team2_id": "T2", "status": "completed"},
        ]
    )
    maps_df = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": float("nan"), "team2_score": 8, "winner": "T1"},
        ]
    )
    folds = [(0, [], ["m1"])]
    with pytest.raises(ValueError, match="null/NaN"):
        mwr._collect_validation_instances(matches_df, maps_df, folds)


@pytest.mark.parametrize("bad_k", [0, -1, 0.0, float("nan"), float("inf")])
def test_team_map_win_rate_invalid_k_raises(bad_k):
    # k <= 0 (or NaN/inf) is rejected: k=0 would drop the prior term
    # entirely and negative k would produce negative pseudo-counts.
    matches_df, maps_df = _core_tables()
    with pytest.raises(ValueError, match="k must be"):
        mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, maps_df, bad_k)


# --------------------------------------------------------------------------
# leakage safety
# --------------------------------------------------------------------------


def test_equal_and_after_maps_never_enter_the_estimate():
    # The leakage proof on the synthetic fixture: m3 (exactly at the
    # cutoff) and m4 (after it) must not contribute to wins/games/prior,
    # and removing them leaves the estimate byte-for-byte unchanged.
    matches_df, maps_df = _core_tables()
    overall = mwr.team_overall_win_rate("T1", QUERY_DATE, matches_df, maps_df)
    haven = mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, maps_df, 10.0)
    assert overall.wins == 1
    assert overall.games == 2
    assert haven.wins == 1
    assert haven.games == 1

    trimmed_m = matches_df[~matches_df["match_id"].isin(["m3", "m4"])]
    trimmed_p = maps_df[~maps_df["match_id"].isin(["m3", "m4"])]
    overall2 = mwr.team_overall_win_rate("T1", QUERY_DATE, trimmed_m, trimmed_p)
    haven2 = mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, trimmed_m, trimmed_p, 10.0)
    assert (overall2.wins, overall2.games, overall2.rate) == (
        overall.wins, overall.games, overall.rate,
    )
    assert (haven2.wins, haven2.games, haven2.mean) == (haven.wins, haven.games, haven.mean)


def test_strictly_earlier_row_changes_the_estimate():
    # The flip side of the leakage proof: adding a strictly-earlier
    # completed map must change the estimate (proving the earlier row is
    # the only thing that ever moves it).
    matches_df, maps_df = _core_tables()
    base = mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, matches_df, maps_df, 10.0)

    earlier_m = _matches_df(
        [
            {"match_id": "m0", "date": "2025-12-31T10:00:00", "team1_id": "T1",
             "team2_id": "T9", "status": "completed"},
        ]
    )
    earlier_p = _maps_df(
        [
            {"match_id": "m0", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 5, "winner": "T1"},
        ]
    )
    grown_m = pd.concat([matches_df, earlier_m], ignore_index=True)
    grown_p = pd.concat([maps_df, earlier_p], ignore_index=True)
    grown = mwr.team_map_win_rate("T1", "Haven", QUERY_DATE, grown_m, grown_p, 10.0)
    assert grown.wins == base.wins + 1
    assert grown.games == base.games + 1


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_no_leakage_and_sane_numbers():
    # The no-leakage + sanity check at real v1 scale: query the most
    # frequently appearing team just after the dataset's latest date and
    # verify (a) overall games matches a direct maps_as_of count, (b) no
    # as-of date is >= the cutoff, (c) a known real map yields
    # games <= overall games and a mean strictly inside (0, 1), and
    # (d) v1's winner column is display names (never team1/team2), which
    # is exactly why the estimator uses scores instead of winner text.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    overall = mwr.team_overall_win_rate(team_id, query, matches_df, maps_df)
    direct = asof.maps_as_of(team_id, query, matches_df, maps_df)
    assert overall.games == len(direct)
    assert overall.games > 0

    query_ts = pd.to_datetime(query)
    assert (pd.to_datetime(direct["date"]) < query_ts).all()

    a_map = maps_df["map_name"].iloc[0]
    shrunk = mwr.team_map_win_rate(team_id, a_map, query, matches_df, maps_df, 10.0)
    assert shrunk.games <= overall.games
    assert 0.0 < shrunk.mean < 1.0
    assert shrunk.alpha > 0.0 and shrunk.beta > 0.0 and shrunk.variance > 0.0

    winners = set(maps_df["winner"].dropna().unique())
    assert "team1" not in winners and "team2" not in winners


# --------------------------------------------------------------------------
# select_k
# --------------------------------------------------------------------------

K_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def test_select_k_low_data_prefers_large_k():
    # A low-data team's misleading 1-0 map sample should make large k
    # (heavy shrinkage toward the prior) score better, so the argmin
    # lands on the large end of the grid.
    matches_df, maps_df = _low_data_scenario()
    best_k, scores = mwr.select_k(matches_df, maps_df, K_GRID)
    assert best_k == max(K_GRID)
    assert scores[best_k] < scores[min(K_GRID)]


def test_select_k_high_data_prefers_small_k():
    # A high-data team's reliable per-map rates should make small k
    # (trust the raw rate) score better, so the argmin lands on the
    # small end of the grid.
    matches_df, maps_df = _high_data_scenario()
    best_k, scores = mwr.select_k(matches_df, maps_df, K_GRID)
    assert best_k == min(K_GRID)
    assert scores[best_k] < scores[max(K_GRID)]


def test_select_k_scores_dict_invariant():
    # scores_by_k has exactly one entry per grid value and best_k is a
    # key of that dict (the argmin invariant).
    matches_df, maps_df = _low_data_scenario()
    best_k, scores = mwr.select_k(matches_df, maps_df, K_GRID)
    assert list(scores.keys()) == K_GRID
    assert best_k in scores
    assert all(v > 0.0 for v in scores.values())


def test_select_k_empty_k_grid_raises():
    # An empty grid cannot produce an argmin; it must raise, not return
    # a meaningless best_k.
    matches_df, maps_df = _low_data_scenario()
    with pytest.raises(ValueError, match="k_grid"):
        mwr.select_k(matches_df, maps_df, [])


def test_select_k_invalid_k_in_grid_raises():
    # A non-positive candidate in the grid is rejected up front.
    matches_df, maps_df = _low_data_scenario()
    with pytest.raises(ValueError, match="k must be"):
        mwr.select_k(matches_df, maps_df, [1.0, 0.0])


def test_select_k_zero_validation_instances_raises():
    # Enough matches for the fold machinery but zero finished maps means
    # there is nothing to score; select_k must raise rather than return
    # a best_k over an empty held-out set.
    matches_df, _ = _low_data_scenario()
    empty_maps = _maps_df([])
    with pytest.raises(ValueError, match="zero scoreable validation instances"):
        mwr.select_k(matches_df, empty_maps, K_GRID)

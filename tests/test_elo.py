"""Tests for sequential Elo ratings and team-parity differentials (M14).

Covers the update-rule primitives (``_expected_score`` / ``_update_pair``),
the league-wide as-of filter, the deterministic full replay
(``_replay_ratings_as_of``), the public ``elo_rating`` /
``elo_differential`` queries, the leakage-safety proof (a map dated
exactly at or after the cutoff never enters the replay; strict ``<``
boundary, order respected), and ordering/determinism guarantees
(``(date, match_id, map_index)`` tie-breaks applied, input row-order
independence). A skip-guarded smoke test repeats the strict-before
assertion at real ``data/v1`` scale.
"""

from pathlib import Path

import pandas as pd
import pytest

from features import elo
from utils import asof

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]

D1 = "2026-01-01T10:00:00"
D2 = "2026-01-03T10:00:00"

# Hand-computed reference ratings (K=32, initial=1500) for the core
# "A beats B then loses to B" timeline, verified independently against
# the Elo logistic formula rather than against the module under test.
WIN_THEN_LOSS_A = 1498.5304984710244
WIN_THEN_LOSS_B = 1501.4695015289756
THREE_WINS_A = 1543.747133633611
THREE_WINS_B = 1456.252866366389


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every fixture produces the same column
    order/dtypes regardless of which subset of columns a given fixture
    actually needs.

    Args:
        rows: A list of dicts, one per match; each must carry the keys in
            :data:`_MATCHES_COLS` (extra keys are ignored by the
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


def _core_timeline():
    """Build the A-beats-B-then-loses-to-B timeline.

    Two completed matches: ``m1`` on :data:`D1` where team ``A`` (team1)
    beats ``B`` 13-8, and ``m2`` on :data:`D2` where team ``B`` (team1)
    beats ``A`` 13-8 (so ``A`` loses as team2, exercising both
    orientation states). This is the shared fixture for the leakage and
    ordering tests.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df` /
        :func:`_maps_df`.

    Raises:
        Nothing.
    """
    matches_rows = [
        {"match_id": "m1", "date": D1, "team1_id": "A",
         "team2_id": "B", "status": "completed"},
        {"match_id": "m2", "date": D2, "team1_id": "B",
         "team2_id": "A", "status": "completed"},
    ]
    maps_rows = [
        {"match_id": "m1", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m2", "map_index": 0, "map_name": "Haven",
         "team1_score": 13, "team2_score": 8, "winner": "B"},
    ]
    return _matches_df(matches_rows), _maps_df(maps_rows)


# --------------------------------------------------------------------------
# expected score + update pair
# --------------------------------------------------------------------------


def test_expected_score_symmetry_sums_to_one():
    # E_a + E_b == 1 for any pair (E_b is E(a,b) with arguments swapped).
    for ra, rb in [(1500.0, 1500.0), (1600.0, 1400.0), (1200.0, 1800.0)]:
        assert elo._expected_score(ra, rb) + elo._expected_score(rb, ra) == pytest.approx(1.0)


def test_expected_score_spot_values():
    # Equal ratings -> 0.5; 400 points above -> 10/11 ~= 0.909 (the
    # well-known Elo reference point), symmetric below -> 1/11.
    assert elo._expected_score(1500.0, 1500.0) == pytest.approx(0.5)
    assert elo._expected_score(1900.0, 1500.0) == pytest.approx(10.0 / 11.0)
    assert elo._expected_score(1500.0, 1900.0) == pytest.approx(1.0 / 11.0)


def test_update_pair_winner_rises_loser_falls():
    # For any starting ratings, the winner strictly gains and the loser
    # strictly loses, including a huge underdog winning and a huge
    # favorite winning.
    assert elo._update_pair(1500.0, 1500.0, True, 32.0) == pytest.approx((1516.0, 1484.0))
    big_underdog = elo._update_pair(1200.0, 1800.0, True, 32.0)
    assert big_underdog[0] > 1200.0
    assert big_underdog[1] < 1800.0
    big_favorite = elo._update_pair(1800.0, 1200.0, True, 32.0)
    assert big_favorite[0] > 1800.0
    assert big_favorite[1] < 1200.0


def test_update_pair_zero_sum():
    # (R_a' - R_a) == -(R_b' - R_b) because S_a + S_b == 1 and
    # E_a + E_b == 1 in a two-player update.
    for ra, rb in [(1500.0, 1500.0), (1400.0, 1600.0), (1900.0, 1100.0)]:
        new_a, new_b = elo._update_pair(ra, rb, True, 32.0)
        assert (new_a - ra) == pytest.approx(-(new_b - rb))


def test_update_pair_bigger_upset_bigger_swing():
    # A lower-rated winner produces a bigger rating swing than a
    # higher-rated winner, holding K fixed.
    upset_swing = elo._update_pair(1400.0, 1500.0, True, 32.0)[0] - 1400.0
    favorite_swing = elo._update_pair(1600.0, 1500.0, True, 32.0)[0] - 1600.0
    assert upset_swing > favorite_swing


# --------------------------------------------------------------------------
# league-wide as-of filter
# --------------------------------------------------------------------------


def test_league_filter_excludes_exact_and_after():
    # The core leakage pattern: a map dated exactly at the cutoff and a
    # map dated after it must not appear; only strictly-earlier rows do.
    matches_df, maps_df = _core_timeline()

    at_d1 = elo._league_maps_as_of(D1, matches_df, maps_df)
    assert len(at_d1) == 0  # m1 is dated exactly D1 -> excluded

    at_d2 = elo._league_maps_as_of(D2, matches_df, maps_df)
    assert set(at_d2["match_id"]) == {"m1"}  # m2 dated exactly D2 -> excluded

    after = elo._league_maps_as_of("2026-01-04T00:00:00", matches_df, maps_df)
    assert set(after["match_id"]) == {"m1", "m2"}


def test_league_filter_includes_all_teams_not_one_team():
    # The league filter is all-teams: two unrelated matches (A-B and C-D)
    # both strictly before the cutoff must both contribute maps.
    matches_df = _matches_df(
        [
            {"match_id": "m1", "date": "2026-01-01T10:00:00", "team1_id": "A",
             "team2_id": "B", "status": "completed"},
            {"match_id": "m2", "date": "2026-01-01T11:00:00", "team1_id": "C",
             "team2_id": "D", "status": "completed"},
        ]
    )
    maps_df = _maps_df(
        [
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m2", "map_index": 0, "map_name": "Bind",
             "team1_score": 13, "team2_score": 8, "winner": "C"},
        ]
    )
    rows = elo._league_maps_as_of("2026-01-02T00:00:00", matches_df, maps_df)
    assert set(rows["match_id"]) == {"m1", "m2"}
    assert set(rows["team1_id"]) == {"A", "C"}


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_replay_deterministic():
    # The exact same input replayed twice yields bit-identical ratings
    # (guards float-accumulation nondeterminism from unordered iteration).
    matches_df, maps_df = _core_timeline()
    r1 = elo._replay_ratings_as_of("2026-01-04T00:00:00", matches_df, maps_df)
    r2 = elo._replay_ratings_as_of("2026-01-04T00:00:00", matches_df, maps_df)
    assert r1 == r2


def test_replay_unseen_team_absent_from_dict():
    # A team never referenced in any as-of map is simply absent from the
    # returned dict; callers (elo_rating) default it to initial_rating.
    matches_df, maps_df = _core_timeline()
    ratings = elo._replay_ratings_as_of("2026-01-04T00:00:00", matches_df, maps_df)
    assert "A" in ratings and "B" in ratings
    assert "Z" not in ratings


def test_same_date_match_id_tiebreak_order():
    # Two matches with the identical date but different match_ids:
    # "m_a" (A wins) and "m_b" (A loses). The (date, match_id) tie-break
    # must replay m_a first, so A ends at the win-then-loss value.
    matches_df = _matches_df(
        [
            {"match_id": "m_a", "date": D1, "team1_id": "A",
             "team2_id": "B", "status": "completed"},
            {"match_id": "m_b", "date": D1, "team1_id": "B",
             "team2_id": "A", "status": "completed"},
        ]
    )
    maps_df = _maps_df(
        [
            {"match_id": "m_a", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m_b", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "B"},
        ]
    )
    ratings = elo._replay_ratings_as_of("2026-01-02T00:00:00", matches_df, maps_df)
    assert ratings["A"] == pytest.approx(WIN_THEN_LOSS_A)
    assert ratings["B"] == pytest.approx(WIN_THEN_LOSS_B)


def test_map_index_tiebreak_order_within_match():
    # One match with 3 maps at map_index 0/1/2, deliberately listed out
    # of order (2, 0, 1): the (date, match_id, map_index) sort must
    # replay them 0,1,2, matching the hand-computed three-wins value.
    matches_df = _matches_df(
        [{"match_id": "m1", "date": D1, "team1_id": "A",
          "team2_id": "B", "status": "completed"}]
    )
    maps_df = _maps_df(
        [
            {"match_id": "m1", "map_index": 2, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m1", "map_index": 0, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
            {"match_id": "m1", "map_index": 1, "map_name": "Haven",
             "team1_score": 13, "team2_score": 8, "winner": "A"},
        ]
    )
    ratings = elo._replay_ratings_as_of("2026-01-02T00:00:00", matches_df, maps_df)
    assert ratings["A"] == pytest.approx(THREE_WINS_A)
    assert ratings["B"] == pytest.approx(THREE_WINS_B)


def test_input_row_order_independent():
    # Shuffling the row order of both input frames (but not the logical
    # event order) leaves the replay result unchanged, proving the sort
    # is independent of input row order.
    matches_df, maps_df = _core_timeline()
    expected = elo._replay_ratings_as_of("2026-01-04T00:00:00", matches_df, maps_df)
    shuffled_m = matches_df.sample(frac=1.0, random_state=0)
    shuffled_p = maps_df.sample(frac=1.0, random_state=0)
    got = elo._replay_ratings_as_of("2026-01-04T00:00:00", shuffled_m, shuffled_p)
    assert got == expected


def test_tied_map_raises():
    # A finished map with tied scores is impossible; the replay rejects
    # it loudly rather than silently defining a draw score.
    matches_df = _matches_df(
        [{"match_id": "m1", "date": D1, "team1_id": "A",
          "team2_id": "B", "status": "completed"}]
    )
    maps_df = _maps_df(
        [{"match_id": "m1", "map_index": 0, "map_name": "Haven",
          "team1_score": 12, "team2_score": 12, "winner": "A"}]
    )
    with pytest.raises(ValueError, match="tied scores"):
        elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df)


def test_null_score_on_finished_map_raises_not_silently_misattributed():
    # A "finished" map (winner non-null) with a NaN team1_score must
    # raise ValueError before the tie comparison: IEEE-754 NaN compares
    # neither equal nor greater to anything, so the unguarded code path
    # would silently record team2 as the winner instead of failing.
    matches_df = _matches_df(
        [{"match_id": "m1", "date": D1, "team1_id": "A",
          "team2_id": "B", "status": "completed"}]
    )
    maps_df = _maps_df(
        [{"match_id": "m1", "map_index": 0, "map_name": "Haven",
          "team1_score": float("nan"), "team2_score": 8, "winner": "A"}]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df)


def test_null_score_team2_raises():
    # The same guard fires when the *other* score column is NaN.
    matches_df = _matches_df(
        [{"match_id": "m1", "date": D1, "team1_id": "A",
          "team2_id": "B", "status": "completed"}]
    )
    maps_df = _maps_df(
        [{"match_id": "m1", "map_index": 0, "map_name": "Haven",
          "team1_score": 13, "team2_score": float("nan"), "winner": "A"}]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df)


def test_tie_still_raises_tie_message_not_null_message():
    # Check ordering: a genuine tie (both scores non-null) hits the tie
    # branch, whose message mentions "tied scores", never the
    # null-score branch added before it (a null score is not a tie and
    # must not be reported as one).
    matches_df = _matches_df(
        [{"match_id": "m1", "date": D1, "team1_id": "A",
          "team2_id": "B", "status": "completed"}]
    )
    maps_df = _maps_df(
        [{"match_id": "m1", "map_index": 0, "map_name": "Haven",
          "team1_score": 12, "team2_score": 12, "winner": "A"}]
    )
    with pytest.raises(ValueError, match="tied scores") as excinfo:
        elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df)
    assert "null" not in str(excinfo.value)


# --------------------------------------------------------------------------
# elo_rating
# --------------------------------------------------------------------------


def test_elo_rating_unseen_team_returns_initial():
    # A team with no as-of maps returns exactly initial_rating, and an
    # overridden initial_rating is returned verbatim for the unseen case.
    matches_df, maps_df = _core_timeline()
    assert elo.elo_rating("Z", "2026-01-04T00:00:00", matches_df, maps_df) == 1500.0
    assert elo.elo_rating("Z", "2026-01-04T00:00:00", matches_df, maps_df,
                          initial_rating=1600.0) == 1600.0


def test_elo_rating_one_win_above_one_loss_below_initial():
    # Between D1 and D2 only m1 has happened: the winner (A) is above
    # initial and the loser (B) is below initial.
    matches_df, maps_df = _core_timeline()
    assert elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df) == pytest.approx(1516.0)
    assert elo.elo_rating("B", "2026-01-02T00:00:00", matches_df, maps_df) == pytest.approx(1484.0)
    assert elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df) > 1500.0
    assert elo.elo_rating("B", "2026-01-02T00:00:00", matches_df, maps_df) < 1500.0


@pytest.mark.parametrize("bad_k", [0.0, -1.0, float("nan"), float("inf")])
def test_elo_rating_invalid_k_raises(bad_k):
    # k must be a positive finite real number.
    matches_df, maps_df = _core_timeline()
    with pytest.raises(ValueError, match="k must be"):
        elo.elo_rating("A", "2026-01-04T00:00:00", matches_df, maps_df, k=bad_k)


@pytest.mark.parametrize("bad_init", [float("nan"), float("inf")])
def test_elo_rating_invalid_initial_rating_raises(bad_init):
    # initial_rating must be a finite real number.
    matches_df, maps_df = _core_timeline()
    with pytest.raises(ValueError, match="initial_rating must be"):
        elo.elo_rating("A", "2026-01-04T00:00:00", matches_df, maps_df,
                       initial_rating=bad_init)


# --------------------------------------------------------------------------
# elo_differential
# --------------------------------------------------------------------------


def test_elo_differential_signed_and_abs():
    # differential == rating_a - rating_b exactly; abs_differential ==
    # abs(differential) exactly; and the after-D2 values match the
    # hand-computed win-then-loss replay.
    matches_df, maps_df = _core_timeline()
    d = elo.elo_differential("A", "B", "2026-01-04T00:00:00", matches_df, maps_df)
    assert d.rating_a == pytest.approx(WIN_THEN_LOSS_A)
    assert d.rating_b == pytest.approx(WIN_THEN_LOSS_B)
    assert d.differential == pytest.approx(d.rating_a - d.rating_b)
    assert d.abs_differential == pytest.approx(abs(d.differential))
    assert d.abs_differential > 0.0


def test_elo_differential_swap_negates_signed_keeps_abs():
    # Swapping the two team ids negates differential but leaves
    # abs_differential unchanged (the core signed+absolute parity
    # contract roadmap M14 asks for).
    matches_df, maps_df = _core_timeline()
    d_ab = elo.elo_differential("A", "B", "2026-01-04T00:00:00", matches_df, maps_df)
    d_ba = elo.elo_differential("B", "A", "2026-01-04T00:00:00", matches_df, maps_df)
    assert d_ba.differential == pytest.approx(-d_ab.differential)
    assert d_ba.abs_differential == pytest.approx(d_ab.abs_differential)


def test_elo_differential_two_unseen_zero():
    # Two unseen teams yield differential == 0.0 and abs_differential ==
    # 0.0 (both default to initial_rating).
    matches_df, maps_df = _core_timeline()
    d = elo.elo_differential("X", "Y", "2026-01-04T00:00:00", matches_df, maps_df)
    assert d.rating_a == 1500.0 and d.rating_b == 1500.0
    assert d.differential == 0.0
    assert d.abs_differential == 0.0


def test_elo_differential_uses_single_replay(monkeypatch):
    # elo_differential must compute both ratings from one shared replay,
    # not two independent elo_rating calls (which would replay twice).
    matches_df, maps_df = _core_timeline()
    calls = []
    original = elo._replay_ratings_as_of

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(elo, "_replay_ratings_as_of", counting)
    elo.elo_differential("A", "B", "2026-01-04T00:00:00", matches_df, maps_df)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# leakage safety
# --------------------------------------------------------------------------


def test_leakage_strict_boundary_and_order():
    # The load-bearing leakage proof: (a) at D1 (== the first map's date)
    # the rating is still initial — the exactly-equal map is excluded;
    # (b) between D1 and D2 it reflects only the D1 win; (c) after D2 it
    # reflects both updates in chronological order (win then loss).
    matches_df, maps_df = _core_timeline()

    at_d1 = elo.elo_rating("A", D1, matches_df, maps_df)
    assert at_d1 == pytest.approx(1500.0)  # (a) strict < excludes m1

    between = elo.elo_rating("A", "2026-01-02T00:00:00", matches_df, maps_df)
    assert between == pytest.approx(1516.0)  # (b) only the D1 win

    after = elo.elo_rating("A", "2026-01-04T00:00:00", matches_df, maps_df)
    assert after == pytest.approx(WIN_THEN_LOSS_A)  # (c) both, in order


def test_leakage_order_matters_not_reversed():
    # Applying the two updates in reverse order (loss first, then win)
    # yields a different rating, proving the replay respects the true
    # chronological order rather than any other permutation.
    matches_df, maps_df = _core_timeline()
    actual = elo.elo_rating("A", "2026-01-04T00:00:00", matches_df, maps_df)
    reversed_rating = 1501.4695015289756  # loss-then-win result for A
    assert actual == pytest.approx(WIN_THEN_LOSS_A)
    assert actual != pytest.approx(reversed_rating)
    assert abs(actual - reversed_rating) > 1e-6


@pytest.mark.parametrize(
    "query",
    [D1, "2026-01-02T00:00:00", "2026-01-04T00:00:00"],
)
def test_map_dated_at_or_after_query_never_changes_rating(query):
    # A poison map (A wins) dated exactly at the query date must never
    # change the result of a query at that date, for several query dates
    # sprinkled through the timeline.
    matches_df, maps_df = _core_timeline()
    base = elo.elo_rating("A", query, matches_df, maps_df)

    poison_m = _matches_df(
        [{"match_id": "poison", "date": query, "team1_id": "A",
          "team2_id": "C", "status": "completed"}]
    )
    poison_p = _maps_df(
        [{"match_id": "poison", "map_index": 0, "map_name": "Haven",
          "team1_score": 13, "team2_score": 8, "winner": "A"}]
    )
    grown_m = pd.concat([matches_df, poison_m], ignore_index=True)
    grown_p = pd.concat([maps_df, poison_p], ignore_index=True)
    assert elo.elo_rating("A", query, grown_m, grown_p) == base


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_scale_replay_strictly_before():
    # The strict-before assertion at real v1 scale: the league filter's
    # surviving maps are exactly the finished maps of completed matches
    # dated < a mid-dataset cutoff (cross-checked against a manual
    # matches_df[match.date < cutoff] count).
    matches_df, maps_df = asof.load_asof_tables("v1")

    dates = pd.to_datetime(matches_df["date"]).sort_values()
    cutoff = dates.iloc[len(dates) // 2]
    cutoff_str = cutoff.isoformat()

    rows = elo._league_maps_as_of(cutoff_str, matches_df, maps_df)

    expected_matches = matches_df[pd.to_datetime(matches_df["date"]) < cutoff]
    expected_maps = maps_df[
        maps_df["match_id"].isin(set(expected_matches["match_id"]))
        & maps_df["winner"].notna()
    ]
    assert len(rows) == len(expected_maps)
    assert set(rows["match_id"]) == set(expected_maps["match_id"])
    assert len(rows) > 0
    assert (pd.to_datetime(rows["date"]) < cutoff).all()

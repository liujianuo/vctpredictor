"""Tests for the veto evaluation harness (M26).

Covers the abbreviation table + verification (real v1 reconciliation
with zero mismatches, plus injected-unknown/mismatched/non-two-
abbreviation ValueErrors), the held-out builder (15 real test matches,
all Bo3, 105 rows), both per-step predictors (greedy softmax
cross-checked against team_map_scores; frequency baseline share,
leakage, and uniform fallback), the teacher-forced scorer (hand-
computed cross-entropy/top-1/top-3 on a uniform stub, real-sequence
bookkeeping under a wrong-favorite stub, decider exclusion, length and
pool-mismatch validation), both report builders (aggregate math,
row-alignment guard, JSON-serializability), and a real v1 end-to-end
run via ``drivers/evaluate_veto.py`` asserting **internal consistency
only** — no predetermined direction is asserted for the
greedy-vs-baseline deltas.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation import veto_evaluation as ve
from models.greedy_veto_simulator import team_map_scores

_QUERY_DATE = "2026-01-06T00:00:00"

# The 7-map pool matching config.json's 2026-abyss era, in ascending
# name order.
POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]

_REAL_V1_TABLES = ("matches", "maps", "splits", "veto_actions")


def _real_v1_available():
    """Report whether the real v1 tables needed by the veto harness exist.

    The skip guard for the end-to-end tests: matches, maps, splits and
    veto_actions must all be materialised under ``data/v1``, i.e.
    ``materialize.py`` and ``splits.py`` have been run.

    Returns:
        A bool: ``True`` iff all four ``data/v1/*.parquet`` files
            exist.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}.parquet").exists() for name in _REAL_V1_TABLES
    )


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Args:
        rows: A list of dicts, one per match, each carrying the keys
            in :data:`_MATCHES_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MATCHES_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is).
    """
    return pd.DataFrame(rows, columns=_MATCHES_COLS)


def _maps_df(rows):
    """Build a maps table with the fixed M8 column set.

    Args:
        rows: A list of dicts, one per map, each carrying the keys in
            :data:`_MAPS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MAPS_COLS` columns.

    Raises:
        Nothing.
    """
    return pd.DataFrame(rows, columns=_MAPS_COLS)


def _build(match_rows, map_rows):
    """Build a ``(matches_df, maps_df)`` pair from parallel row lists.

    Args:
        match_rows: A list of match dicts.
        map_rows: A list of map dicts.

    Returns:
        A ``(matches_df, maps_df)`` tuple.

    Raises:
        Nothing.
    """
    return _matches_df(match_rows), _maps_df(map_rows)


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        mid: The shared ``match_id`` for the new match and map.
        date: The match's ISO date string.
        team1_id: The match's team1 stable id.
        team2_id: The match's team2 stable id.
        map_name: The finished map's name.
        t1s: Rounds team1 won.
        t2s: Rounds team2 won.

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
    """Build the 16-map greedy-choice league (mirrors M25's fixture).

    Team ``A`` is 0W-4L on Split (weakest) and 4W-0L on Haven
    (strongest); team ``B`` mirrors that with Sunset/Ascent. Every
    opponent id is unique and plays once, all dated before
    :data:`_QUERY_DATE`, so the greedy choices are deterministic.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 16 matches and 16 maps.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    for i in range(4):
        _add(match_rows, map_rows, f"a_split_{i}", _stamp(i), "A", f"op{i}", "Split", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"a_haven_{i}", _stamp(4 + i), "A", f"op{10 + i}", "Haven", 13, 8)
    for i in range(4):
        _add(match_rows, map_rows, f"b_sunset_{i}", _stamp(8 + i), "B", f"op{20 + i}", "Sunset", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"b_ascent_{i}", _stamp(12 + i), "B", f"op{30 + i}", "Ascent", 13, 8)
    return _build(match_rows, map_rows)


def _frequency_tables():
    """Build the play-frequency league for the baseline tests.

    Team ``T`` (and its mirror) plays Haven three times and Ascent
    once before :data:`_QUERY_DATE`; a fifth Haven match is dated
    *exactly at* the query cutoff (excluded by the strict ``<``
    boundary) and a sixth is dated after it. All wins by score.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 6 matches and 6 maps.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "f1", _stamp(0), "T", "o1", "Haven", 13, 8)
    _add(match_rows, map_rows, "f2", _stamp(1), "T", "o2", "Haven", 13, 8)
    _add(match_rows, map_rows, "f3", _stamp(2), "T", "o3", "Haven", 13, 8)
    _add(match_rows, map_rows, "f4", _stamp(3), "T", "o4", "Ascent", 13, 8)
    # Exactly at the query timestamp (strictly < excludes it) and after.
    _add(match_rows, map_rows, "f5", _stamp(6), "T", "o5", "Haven", 13, 8)
    _add(match_rows, map_rows, "f6", _stamp(7), "T", "o6", "Haven", 13, 8)
    return _build(match_rows, map_rows)


def _held_out_frame(rows):
    """Build a held-out veto-step table from per-step row dicts.

    Args:
        rows: A list of dicts, each carrying ``match_id``,
            ``step_index``, ``team_id``, ``action``, ``map_name`` and
            ``date`` (the columns :func:`ve.score_veto_steps` reads;
            any extra keys are ignored by the explicit column order).

    Returns:
        A ``pandas.DataFrame`` with exactly those six columns, in that
        order, unsorted (the scorer sorts per match).

    Raises:
        Nothing.
    """
    columns = ["match_id", "step_index", "team_id", "action", "map_name", "date"]
    return pd.DataFrame(rows, columns=columns)


# A complete real Bo3 veto sequence over POOL: ban Abyss, ban Split,
# pick Ascent, pick Lotus, ban Haven, ban Summit, decider Sunset. With
# a uniform stub predictor the per-step hand-computed numbers are:
#   step 0: 7 maps, true Abyss rank 1 -> ce log(7), top1 T, top3 T
#   step 1: 6 maps, true Split rank 4 -> ce log(6), top1 F, top3 F
#   step 2: 5 maps, true Ascent rank 1 -> ce log(5), top1 T, top3 T
#   step 3: 4 maps, true Lotus rank 2 -> ce log(4), top1 F, top3 T
#   step 4: 3 maps, true Haven rank 1 -> ce log(3), top1 T, top3 T
#   step 5: 2 maps, true Summit rank 1 -> ce log(2), top1 T, top3 T
_SYN_BO3_ROWS = [
    ("syn1", 0, "A", "ban", "Abyss", "2026-08-23T12:15:00"),
    ("syn1", 1, "B", "ban", "Split", "2026-08-23T12:15:00"),
    ("syn1", 2, "A", "pick", "Ascent", "2026-08-23T12:15:00"),
    ("syn1", 3, "B", "pick", "Lotus", "2026-08-23T12:15:00"),
    ("syn1", 4, "A", "ban", "Haven", "2026-08-23T12:15:00"),
    ("syn1", 5, "B", "ban", "Summit", "2026-08-23T12:15:00"),
    ("syn1", 6, None, "decider", "Sunset", "2026-08-23T12:15:00"),
]


def _uniform_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """The uniform-stub predictor used for hand-computed scoring.

    Test function (stub), not a scored model: returns the uniform
    distribution over ``remaining_maps`` so every per-step metric is
    hand-computable (rank = alphabetical position of the true map).

    Args:
        acting_team_id / action / date / matches_df / maps_df:
            Ignored (the uniform distribution depends only on the
            candidate count).
        remaining_maps: The sorted list of maps still in play.

    Returns:
        A ``list`` of ``1 / len(remaining_maps)`` probabilities.

    Raises:
        Nothing.
    """
    del acting_team_id, action, date, matches_df, maps_df
    k = len(remaining_maps)
    return [1.0 / k] * k


def _wrong_favorite_stub(recorded):
    """Build a stub predictor that concentrates on the wrong map.

    Returns a closure over ``recorded`` (a list the closure appends
    every ``remaining_maps`` argument to) that puts 0.9 on the
    alphabetically *last* map of the passed sorted list and spreads the
    rest uniformly — so whenever the true choice is not the last map
    (all six real steps of the synthetic Bo3), the predictor's favorite
    differs from the real map and the scorer's teacher forcing is what
    keeps the bookkeeping correct.

    Args:
        recorded: A mutable list the returned closure appends each
            ``list(remaining_maps)`` argument to, in call order.

    Returns:
        The stub predictor callable.

    Raises:
        Nothing.
    """
    def stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        del acting_team_id, action, date, matches_df, maps_df
        recorded.append(list(remaining_maps))
        k = len(remaining_maps)
        probs = [0.1 / (k - 1)] * k
        probs[-1] = 0.9
        return probs

    return stub


# --------------------------------------------------------------------------
# plan#15a: abbreviation table + verify_team_abbreviation_map
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py and "
    "splits.py first)",
)
def test_real_v1_abbreviation_reconciliation_zero_mismatches():
    # The decision-1 verification on real data: every one of the 96
    # veto matches' two abbreviations maps to exactly {team1_id,
    # team2_id}. The table is exactly the 16 abbreviations present in
    # the data (no more, no less), and every abbreviation resolves.
    import pandas as pd

    veto_df = pd.read_parquet("data/v1/veto_actions.parquet")
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    ve.verify_team_abbreviation_map(veto_df, matches_df)  # must not raise
    abbrevs = set(veto_df["team"].dropna().unique())
    assert len(abbrevs) == 16
    assert set(ve.TEAM_ABBREVIATION_TO_ID) == abbrevs


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py and "
    "splits.py first)",
)
def test_real_v1_resolve_veto_team_ids_on_known_match():
    # resolve_veto_team_ids attaches the mapped stable ids, with None
    # on decider rows. Spot-check the known real Bo3 731406 (FF -> 20085,
    # BBL -> 397): the first ban is FF's, the decider row carries None.
    import pandas as pd

    veto_df = pd.read_parquet("data/v1/veto_actions.parquet")
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    resolved = ve.resolve_veto_team_ids(veto_df, matches_df)
    assert "team_id" in resolved.columns
    assert len(resolved) == len(veto_df)
    row = resolved[
        (resolved["match_id"] == "731406") & (resolved["step_index"] == 0)
    ].iloc[0]
    assert row["team"] == "FF"
    assert row["team_id"] == "20085"
    decider = resolved[
        (resolved["match_id"] == "731406") & (resolved["step_index"] == 6)
    ].iloc[0]
    assert decider["team"] is None
    assert decider["team_id"] is None


def test_verify_raises_on_unknown_abbreviation():
    # A 17th abbreviation absent from the closed table must fail loudly
    # (decision 1: never silently mis-map or drop a match).
    veto_df = pd.DataFrame(
        [
            {"match_id": "x1", "step_index": 0, "team": "FNC", "action": "ban", "map_name": "Haven"},
            {"match_id": "x1", "step_index": 1, "team": "XXX", "action": "ban", "map_name": "Split"},
        ]
    )
    matches_df = _matches_df(
        [
            {"match_id": "x1", "date": "2026-08-23T12:00:00", "team1_id": "2593", "team2_id": "9999", "status": "completed"}
        ]
    )
    with pytest.raises(ValueError, match="XXX"):
        ve.verify_team_abbreviation_map(veto_df, matches_df)


def test_verify_raises_on_mismatched_mapping():
    # Two known abbreviations whose mapped ids do not equal the match's
    # {team1_id, team2_id} pair: a stale matches/veto dataset must be
    # caught, not silently trusted.
    veto_df = pd.DataFrame(
        [
            {"match_id": "x1", "step_index": 0, "team": "FNC", "action": "ban", "map_name": "Haven"},
            {"match_id": "x1", "step_index": 1, "team": "TL", "action": "ban", "map_name": "Split"},
        ]
    )
    matches_df = _matches_df(
        [
            {"match_id": "x1", "date": "2026-08-23T12:00:00", "team1_id": "2593", "team2_id": "9999", "status": "completed"}
        ]
    )
    with pytest.raises(ValueError, match="x1"):
        ve.verify_team_abbreviation_map(veto_df, matches_df)


def test_verify_raises_on_non_two_abbreviations():
    # A match with a single distinct acting team (or three) cannot be
    # reconciled against a two-sided match; fail loudly.
    veto_df = pd.DataFrame(
        [
            {"match_id": "x1", "step_index": 0, "team": "FNC", "action": "ban", "map_name": "Haven"},
            {"match_id": "x1", "step_index": 1, "team": "FNC", "action": "pick", "map_name": "Split"},
        ]
    )
    matches_df = _matches_df(
        [
            {"match_id": "x1", "date": "2026-08-23T12:00:00", "team1_id": "2593", "team2_id": "474", "status": "completed"}
        ]
    )
    with pytest.raises(ValueError, match="exactly 2"):
        ve.verify_team_abbreviation_map(veto_df, matches_df)


def test_verify_raises_on_veto_match_absent_from_matches():
    # A veto row whose match has no matches_df row cannot be checked.
    veto_df = pd.DataFrame(
        [
            {"match_id": "ghost", "step_index": 0, "team": "FNC", "action": "ban", "map_name": "Haven"},
        ]
    )
    matches_df = _matches_df([])
    with pytest.raises(ValueError, match="absent from matches_df"):
        ve.verify_team_abbreviation_map(veto_df, matches_df)


# --------------------------------------------------------------------------
# plan#15b: build_held_out_veto_matches on real v1
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py and "
    "splits.py first)",
)
def test_real_v1_held_out_builder_shape():
    # The held-out builder on real v1: exactly the 15 test-split
    # matches (all Bo3), one row per veto step = 105 rows total,
    # columns exactly HELD_OUT_VETO_COLUMNS, sorted by
    # (match_id, step_index), with team_id None exactly on decider
    # rows and the split column all "test".
    import pandas as pd

    veto_df = pd.read_parquet("data/v1/veto_actions.parquet")
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    held = ve.build_held_out_veto_matches(veto_df, matches_df, splits_df, split="test")
    assert list(held.columns) == list(ve.HELD_OUT_VETO_COLUMNS)
    assert len(held) == 105
    assert held["match_id"].nunique() == 15
    assert set(held["best_of"].unique()) == {"Bo3"}
    assert set(held["split"].unique()) == {"test"}
    # Sorted by (match_id, step_index).
    keys = list(zip(held["match_id"], held["step_index"]))
    assert keys == sorted(keys)
    # Decider rows are the only team_id-None rows, and there is one per
    # match (15).
    decider_rows = held[held["action"] == "decider"]
    assert len(decider_rows) == 15
    assert decider_rows["team_id"].isna().all()
    assert held[held["action"] != "decider"]["team_id"].notna().all()


# --------------------------------------------------------------------------
# plan#15c: greedy_veto_step_model
# --------------------------------------------------------------------------


def test_greedy_step_model_matches_team_map_scores_softmax():
    # The greedy arm must equal softmax(team_map_scores) for a pick and
    # softmax(-team_map_scores) for a ban, over the sorted remaining
    # maps, with probabilities summing to 1. The argmax of the pick
    # distribution must be Haven (A's strongest map) and the argmax of
    # the ban distribution must be Split (A's weakest) — the maps
    # simulate_veto's deterministic rule would choose at those steps.
    matches_df, maps_df = _league_tables()
    sorted_pool = sorted(POOL)
    scores = team_map_scores(
        "A", sorted_pool, _QUERY_DATE, matches_df, maps_df, 10.0
    )

    pick_probs = ve.greedy_veto_step_model(
        "A", "pick", sorted_pool, _QUERY_DATE, matches_df, maps_df, k=10.0
    )
    expected_pick = ve._softmax([scores[name] for name in sorted_pool])
    assert pick_probs == pytest.approx(expected_pick)
    assert sum(pick_probs) == pytest.approx(1.0)
    assert pick_probs[sorted_pool.index("Haven")] == pytest.approx(
        max(pick_probs)
    )

    ban_probs = ve.greedy_veto_step_model(
        "A", "ban", sorted_pool, _QUERY_DATE, matches_df, maps_df, k=10.0
    )
    expected_ban = ve._softmax([-scores[name] for name in sorted_pool])
    assert ban_probs == pytest.approx(expected_ban)
    assert sum(ban_probs) == pytest.approx(1.0)
    assert ban_probs[sorted_pool.index("Split")] == pytest.approx(
        max(ban_probs)
    )


def test_greedy_step_model_unknown_action_raises():
    # The sign flip is defined only for bans and picks; anything else
    # is a hard error rather than a silent assumption.
    matches_df, maps_df = _league_tables()
    with pytest.raises(ValueError, match="ban.*pick|pick.*ban"):
        ve.greedy_veto_step_model(
            "A", "decider", sorted(POOL), _QUERY_DATE, matches_df, maps_df
        )


# --------------------------------------------------------------------------
# plan#15d: most_frequent_map_baseline_model
# --------------------------------------------------------------------------


def test_frequency_baseline_shares_and_leakage():
    # Haven was played 3 times strictly before the query, Ascent once:
    # the baseline over the full pool must put ~3/4 on Haven, ~1/4 on
    # Ascent, and the floor (1e-12) on the five zero-count maps, with
    # the two future Haven matches (dated at/after the cutoff) excluded
    # by the strict-< boundary — the leakage check.
    matches_df, maps_df = _frequency_tables()
    sorted_pool = sorted(POOL)
    probs = ve.most_frequent_map_baseline_model(
        "T", "ban", sorted_pool, _stamp(6), matches_df, maps_df
    )
    assert sum(probs) == pytest.approx(1.0)
    assert probs[sorted_pool.index("Haven")] == pytest.approx(0.75)
    assert probs[sorted_pool.index("Ascent")] == pytest.approx(0.25)
    for name in ("Abyss", "Lotus", "Split", "Summit", "Sunset"):
        assert probs[sorted_pool.index(name)] == pytest.approx(
            ve._BASELINE_PROB_FLOOR
        )
    # The baseline is action-agnostic: ban and pick give identical
    # distributions, and acting_team_id is ignored.
    pick_probs = ve.most_frequent_map_baseline_model(
        "someone-else", "pick", sorted_pool, _stamp(6), matches_df, maps_df
    )
    assert pick_probs == pytest.approx(probs)


def test_frequency_baseline_uniform_fallback_on_empty_history():
    # No map has any strictly-prior play: the documented fallback is
    # the uniform distribution over the remaining set.
    matches_df, maps_df = _build([], [])
    probs = ve.most_frequent_map_baseline_model(
        "T", "ban", sorted(POOL), _QUERY_DATE, matches_df, maps_df
    )
    assert probs == pytest.approx([1.0 / 7] * 7)


# --------------------------------------------------------------------------
# plan#15e: score_veto_steps (teacher-forced, hand-computed)
# --------------------------------------------------------------------------


def test_score_veto_steps_uniform_stub_hand_computed():
    # With the uniform stub every metric is closed-form: ce = log(k) at
    # a k-map step, top1 = (the true map is alphabetically first), top3
    # = (rank <= min(3, k)). The output has exactly 6 rows (the decider
    # is excluded), in (match_id, step_index) order, with the real
    # sequence's n_remaining.
    held = _held_out_frame([dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row)) for row in _SYN_BO3_ROWS])
    empty_matches = _matches_df([])
    empty_maps = _maps_df([])
    scored = ve.score_veto_steps(
        _uniform_stub, held, empty_matches, empty_maps, map_pool=POOL
    )
    assert list(scored.columns) == list(ve.SCORED_STEP_COLUMNS)
    assert len(scored) == 6
    assert list(scored["n_remaining"]) == [7, 6, 5, 4, 3, 2]
    assert list(scored["action"]) == ["ban", "ban", "pick", "pick", "ban", "ban"]
    expected_ce = [math.log(7), math.log(6), math.log(5), math.log(4), math.log(3), math.log(2)]
    assert list(scored["cross_entropy"]) == pytest.approx(expected_ce)
    assert list(scored["top1_correct"]) == [True, False, True, False, True, True]
    assert list(scored["top3_correct"]) == [True, False, True, True, True, True]


def test_score_veto_steps_teacher_forcing_tracks_real_sequence():
    # The teacher-forcing contract (decision 4): the scorer's remaining
    # set must track the *real* sequence, not the predictor's favorites.
    # The stub concentrates 0.9 on the alphabetically last map (never
    # the real choice on any of the six steps), so the only way the
    # recorded remaining lists can match the expected real-sequence
    # sets is if the scorer removed the real maps, not the stub's picks.
    held = _held_out_frame([dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row)) for row in _SYN_BO3_ROWS])
    recorded: list[list[str]] = []
    stub = _wrong_favorite_stub(recorded)
    empty_matches = _matches_df([])
    empty_maps = _maps_df([])
    scored = ve.score_veto_steps(stub, held, empty_matches, empty_maps, map_pool=POOL)
    assert recorded == [
        ["Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"],
        ["Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"],
        ["Ascent", "Haven", "Lotus", "Summit", "Sunset"],
        ["Haven", "Lotus", "Summit", "Sunset"],
        ["Haven", "Summit", "Sunset"],
        ["Summit", "Sunset"],
    ]
    assert list(scored["n_remaining"]) == [7, 6, 5, 4, 3, 2]
    # With the wrong-favorite stub the true map never has the top
    # probability, so no step is top-1 correct (a sanity cross-check of
    # the ranking under a deliberately bad predictor).
    assert not scored["top1_correct"].any()


def test_score_veto_steps_rejects_wrong_length_vector():
    # A predictor returning the wrong number of probabilities must be
    # rejected with a message naming the offending match and step.
    held = _held_out_frame([dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row)) for row in _SYN_BO3_ROWS])

    def bad_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        del acting_team_id, action, date, matches_df, maps_df
        return [0.5, 0.5]  # wrong length unless exactly 2 maps remain

    with pytest.raises(ValueError, match="syn1.*step 0"):
        ve.score_veto_steps(
            bad_stub, held, _matches_df([]), _maps_df([]), map_pool=POOL
        )


def test_score_veto_steps_rejects_pool_sequence_mismatch():
    # Decision 2's fail-loud clause: if a real veto's map-name set does
    # not exactly equal the resolved pool, teacher forcing cannot be
    # trusted and the scorer must raise rather than silently mis-track
    # remaining.
    rows = [
        dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row))
        for row in _SYN_BO3_ROWS
    ]
    # Swap the decider for a map outside the pool.
    rows[-1]["map_name"] = "Breeze"
    held = _held_out_frame(rows)
    with pytest.raises(ValueError, match="does not exactly equal"):
        ve.score_veto_steps(
            _uniform_stub, held, _matches_df([]), _maps_df([]), map_pool=POOL
        )


def test_score_veto_steps_config_era_pool_resolution():
    # map_pool=None resolves the pool from config.json's era covering
    # the match date (2026-08-23 falls in the 2026-abyss era, whose
    # pool is exactly POOL), so the default-path run equals the
    # explicit-pool run on the synthetic Bo3.
    held = _held_out_frame([dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row)) for row in _SYN_BO3_ROWS])
    empty_matches = _matches_df([])
    empty_maps = _maps_df([])
    default = ve.score_veto_steps(
        _uniform_stub, held, empty_matches, empty_maps
    )
    explicit = ve.score_veto_steps(
        _uniform_stub, held, empty_matches, empty_maps, map_pool=POOL
    )
    assert default.equals(explicit)


# --------------------------------------------------------------------------
# plan#15f: report builders
# --------------------------------------------------------------------------


def _scored_frame(rows):
    """Build a score_veto_steps-shaped DataFrame from per-step rows.

    Args:
        rows: A list of ``(match_id, step_index, action, n_remaining,
            cross_entropy, top1_correct, top3_correct)`` tuples.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`ve.SCORED_STEP_COLUMNS`
        columns, one row per input.

    Raises:
        Nothing.
    """
    columns = list(ve.SCORED_STEP_COLUMNS)
    return pd.DataFrame(
        [dict(zip(columns, row)) for row in rows], columns=columns
    )


# Six scored steps: steps 0/1/2 each twice (two matches), with greedy
# cross-entropies 1/2/3 and baseline cross-entropies 2/2/2, greedy
# top1 [T,F,T,T,F,T] and baseline all False, greedy top3
# [T,T,F,T,T,F] and baseline all True.
_GREEDY_ROWS = [
    ("m1", 0, "ban", 7, 1.0, True, True),
    ("m1", 1, "pick", 6, 2.0, False, True),
    ("m1", 2, "ban", 5, 3.0, True, False),
    ("m2", 0, "ban", 7, 1.0, True, True),
    ("m2", 1, "pick", 6, 2.0, False, True),
    ("m2", 2, "ban", 5, 3.0, True, False),
]
_BASELINE_ROWS = [
    ("m1", 0, "ban", 7, 2.0, False, True),
    ("m1", 1, "pick", 6, 2.0, False, True),
    ("m1", 2, "ban", 5, 2.0, False, True),
    ("m2", 0, "ban", 7, 2.0, False, True),
    ("m2", 1, "pick", 6, 2.0, False, True),
    ("m2", 2, "ban", 5, 2.0, False, True),
]


def test_evaluation_report_aggregate_math():
    # The aggregate block is the plain mean of the per-step columns, and
    # the by_step_index breakdown recomputes the same four fields per
    # step index (each index has exactly 2 rows here, so the means are
    # the midpoints). All hand-computed below.
    scored = _scored_frame(_GREEDY_ROWS)
    report = ve.build_veto_evaluation_report(scored)
    assert report["n_steps"] == 6
    assert report["mean_cross_entropy"] == pytest.approx(2.0)
    assert report["top1_accuracy"] == pytest.approx(4 / 6)
    assert report["top3_accuracy"] == pytest.approx(4 / 6)
    assert [entry["step_index"] for entry in report["by_step_index"]] == [0, 1, 2]
    by_index = {entry["step_index"]: entry for entry in report["by_step_index"]}
    assert by_index[0]["n_steps"] == 2
    assert by_index[0]["mean_cross_entropy"] == pytest.approx(1.0)
    assert by_index[0]["top1_accuracy"] == pytest.approx(1.0)
    assert by_index[0]["top3_accuracy"] == pytest.approx(1.0)
    assert by_index[1]["mean_cross_entropy"] == pytest.approx(2.0)
    assert by_index[1]["top1_accuracy"] == pytest.approx(0.0)
    assert by_index[1]["top3_accuracy"] == pytest.approx(1.0)
    assert by_index[2]["mean_cross_entropy"] == pytest.approx(3.0)
    assert by_index[2]["top1_accuracy"] == pytest.approx(1.0)
    assert by_index[2]["top3_accuracy"] == pytest.approx(0.0)


def test_evaluation_report_rejects_empty_table():
    # A report over zero scored steps is undefined; fail loudly.
    scored = _scored_frame([])
    with pytest.raises(ValueError, match="zero scored steps"):
        ve.build_veto_evaluation_report(scored)


def test_comparison_report_shape_deltas_and_json():
    # The head-to-head report holds both arms plus greedy-minus-baseline
    # deltas, and the whole dict round-trips through json.
    greedy = _scored_frame(_GREEDY_ROWS)
    baseline = _scored_frame(_BASELINE_ROWS)
    report = ve.build_veto_comparison_report(greedy, baseline)
    assert report["greedy"]["n_steps"] == 6
    assert report["baseline"]["n_steps"] == 6
    assert report["delta"]["mean_cross_entropy"] == pytest.approx(0.0)
    assert report["delta"]["top1_accuracy"] == pytest.approx(4 / 6)
    assert report["delta"]["top3_accuracy"] == pytest.approx(-2 / 6)
    serialized = json.dumps(report, sort_keys=True)
    assert json.loads(serialized) == report


def test_comparison_report_rejects_misaligned_rows():
    # Different row counts, or the same rows with a different id at the
    # same position, are misalignments: a positional comparison would
    # silently pair two different steps' scores.
    greedy = _scored_frame(_GREEDY_ROWS)
    with pytest.raises(ValueError, match="row counts"):
        ve.build_veto_comparison_report(greedy, _scored_frame(_BASELINE_ROWS[:-1]))
    swapped = [
        ("m9", 0, action, n, ce, t1, t3) if mid == "m1" else (mid, i, action, n, ce, t1, t3)
        for (mid, i, action, n, ce, t1, t3) in _BASELINE_ROWS
    ]
    with pytest.raises(ValueError, match="not row-aligned"):
        ve.build_veto_comparison_report(greedy, _scored_frame(swapped))


# --------------------------------------------------------------------------
# plan#12-13 (M27): actions_to_score filter, ban training examples,
# multi-arm report
# --------------------------------------------------------------------------


def _full_held_out_frame(rows):
    """Build a held-out frame with team1_id/team2_id for opponent resolution.

    Extends :func:`_held_out_frame` with the ``team1_id``/``team2_id``
    columns of :data:`ve.HELD_OUT_VETO_COLUMNS`, which the shared
    teacher-forced walk needs to resolve each step's opponent.

    Args:
        rows: A list of ``(match_id, step_index, team_id, action,
            map_name, date, team1_id, team2_id)`` tuples.

    Returns:
        A ``pandas.DataFrame`` with exactly those eight columns, in
        that order.

    Raises:
        Nothing.
    """
    columns = [
        "match_id",
        "step_index",
        "team_id",
        "action",
        "map_name",
        "date",
        "team1_id",
        "team2_id",
    ]
    return pd.DataFrame(rows, columns=columns)


# The synthetic Bo3 (same as _SYN_BO3_ROWS) with team1_id/team2_id
# appended so opponent resolution is exercised. Expected ban examples:
#   step 0: A bans Abyss over the 7-map pool      -> idx 0, opponent B
#   step 1: B bans Split over 6 maps              -> idx 3, opponent A
#   step 2/3: picks (skipped)
#   step 4: A bans Haven over [Haven, Summit, Sunset] -> idx 0, opp B
#   step 5: B bans Summit over [Summit, Sunset]   -> idx 0, opponent A
_SYN_BO3_FULL_ROWS = [
    ("syn1", 0, "A", "ban", "Abyss", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 1, "B", "ban", "Split", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 2, "A", "pick", "Ascent", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 3, "B", "pick", "Lotus", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 4, "A", "ban", "Haven", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 5, "B", "ban", "Summit", "2026-08-23T12:15:00", "A", "B"),
    ("syn1", 6, None, "decider", "Sunset", "2026-08-23T12:15:00", "A", "B"),
]


def test_score_veto_steps_actions_to_score_filters_but_keeps_bookkeeping():
    # The decision-11 filter: with actions_to_score={"ban"} only the
    # four ban steps are scored (in the same remaining order as the
    # full replay — the picks in between are skipped from scoring but
    # still consumed for remaining bookkeeping), so the step-4 ban sees
    # 3 maps and the step-5 ban sees 2, exactly as the full run does.
    held = _held_out_frame(
        [dict(zip(("match_id", "step_index", "team_id", "action", "map_name", "date"), row)) for row in _SYN_BO3_ROWS]
    )
    scored = ve.score_veto_steps(
        _uniform_stub,
        held,
        _matches_df([]),
        _maps_df([]),
        map_pool=POOL,
        actions_to_score={"ban"},
    )
    assert list(scored["action"]) == ["ban", "ban", "ban", "ban"]
    assert list(scored["n_remaining"]) == [7, 6, 3, 2]
    assert list(scored["cross_entropy"]) == pytest.approx(
        [math.log(7), math.log(6), math.log(3), math.log(2)]
    )
    assert list(scored["top1_correct"]) == [True, False, True, True]


def test_build_ban_training_examples_hand_checked_labels_and_opponents():
    # The decision-12 builder on the hand-checked synthetic Bo3: four
    # examples (one per ban step), each with the sorted remaining
    # candidate list, the true banned map's index within it, and the
    # opponent = the other id of {team1_id, team2_id}.
    held = _full_held_out_frame(_SYN_BO3_FULL_ROWS)
    examples = ve.build_ban_training_examples(
        held, _matches_df([]), _maps_df([]), map_pool=POOL
    )
    assert len(examples) == 4
    expected = [
        ("A", "B", ["Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"], 0),
        ("B", "A", ["Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"], 3),
        ("A", "B", ["Haven", "Summit", "Sunset"], 0),
        ("B", "A", ["Summit", "Sunset"], 0),
    ]
    for example, (acting, opponent, remaining, true_index) in zip(examples, expected):
        assert example.acting_team_id == acting
        assert example.opponent_team_id == opponent
        assert list(example.remaining_maps) == remaining
        assert example.true_map_index == true_index
        assert example.date == "2026-08-23T12:15:00"


def test_build_ban_training_examples_rejects_no_ban_rows():
    # A held-out table with zero ban actions has nothing to train on;
    # fail loudly rather than returning an empty example list.
    rows = [
        ("syn1", 0, "A", "pick", "Abyss", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 1, "B", "pick", "Split", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 2, "A", "pick", "Ascent", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 3, "B", "pick", "Lotus", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 4, "A", "pick", "Haven", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 5, "B", "pick", "Summit", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 6, None, "decider", "Sunset", "2026-08-23T12:15:00", "A", "B"),
    ]
    held = _full_held_out_frame(rows)
    with pytest.raises(ValueError, match="no ban veto actions"):
        ve.build_ban_training_examples(
            held, _matches_df([]), _maps_df([]), map_pool=POOL
        )


def test_iter_teacher_forced_steps_rejects_unknown_acting_team():
    # A non-decider row whose acting team is neither of the match's
    # {team1_id, team2_id} pair cannot have its opponent resolved; the
    # shared walk must fail loudly instead of emitting a wrong label.
    rows = [
        ("syn1", 0, "GHOST", "ban", "Abyss", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 1, "B", "ban", "Split", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 2, "A", "pick", "Ascent", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 3, "B", "pick", "Lotus", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 4, "A", "ban", "Haven", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 5, "B", "ban", "Summit", "2026-08-23T12:15:00", "A", "B"),
        ("syn1", 6, None, "decider", "Sunset", "2026-08-23T12:15:00", "A", "B"),
    ]
    held = _full_held_out_frame(rows)
    with pytest.raises(ValueError, match="neither of"):
        ve.build_ban_training_examples(
            held, _matches_df([]), _maps_df([]), map_pool=POOL
        )


# A third scored arm for the multi-arm report tests: every step has
# cross-entropy 0.5, top-1 True and top-3 True, so its aggregates are
# mean_ce 0.5, top1 1.0, top3 1.0.
_CL_ROWS = [
    ("m1", 0, "ban", 7, 0.5, True, True),
    ("m1", 1, "pick", 6, 0.5, True, True),
    ("m1", 2, "ban", 5, 0.5, True, True),
    ("m2", 0, "ban", 7, 0.5, True, True),
    ("m2", 1, "pick", 6, 0.5, True, True),
    ("m2", 2, "ban", 5, 0.5, True, True),
]


def test_multi_arm_report_shape_deltas_and_json():
    # The N-arm report holds one block per arm plus a per-non-baseline-
    # arm delta block (arm minus baseline), and the whole dict
    # round-trips through json. Hand-computed: conditional_logit
    # mean_ce 0.5 vs baseline 2.0 -> -1.5; greedy deltas match the
    # 2-arm report's (0.0 / +4/6 / -2/6).
    greedy = _scored_frame(_GREEDY_ROWS)
    baseline = _scored_frame(_BASELINE_ROWS)
    cl = _scored_frame(_CL_ROWS)
    report = ve.build_veto_multi_arm_report(
        {"conditional_logit": cl, "greedy": greedy, "baseline": baseline},
        baseline_arm="baseline",
    )
    assert set(report) == {"conditional_logit", "greedy", "baseline", "deltas_vs_baseline"}
    assert report["conditional_logit"]["n_steps"] == 6
    assert report["conditional_logit"]["mean_cross_entropy"] == pytest.approx(0.5)
    assert report["conditional_logit"]["top1_accuracy"] == pytest.approx(1.0)
    deltas = report["deltas_vs_baseline"]
    assert set(deltas) == {"conditional_logit", "greedy"}
    assert deltas["conditional_logit"]["mean_cross_entropy"] == pytest.approx(-1.5)
    assert deltas["conditional_logit"]["top1_accuracy"] == pytest.approx(1.0)
    assert deltas["conditional_logit"]["top3_accuracy"] == pytest.approx(0.0)
    assert deltas["greedy"]["mean_cross_entropy"] == pytest.approx(0.0)
    assert deltas["greedy"]["top1_accuracy"] == pytest.approx(4 / 6)
    assert deltas["greedy"]["top3_accuracy"] == pytest.approx(-2 / 6)
    serialized = json.dumps(report, sort_keys=True)
    assert json.loads(serialized) == report


def test_multi_arm_report_rejects_misaligned_rows():
    # Different row counts, or the same rows with a different id at the
    # same position, are misalignments across every arm pair; a
    # baseline arm that is not scored, or a single-arm dict, is also
    # rejected.
    greedy = _scored_frame(_GREEDY_ROWS)
    baseline = _scored_frame(_BASELINE_ROWS)
    cl = _scored_frame(_CL_ROWS)
    with pytest.raises(ValueError, match="row counts"):
        ve.build_veto_multi_arm_report(
            {"conditional_logit": cl, "greedy": greedy, "baseline": baseline[:-1]},
            baseline_arm="baseline",
        )
    swapped = [
        ("m9", 0, action, n, ce, t1, t3) if mid == "m1" else (mid, i, action, n, ce, t1, t3)
        for (mid, i, action, n, ce, t1, t3) in _BASELINE_ROWS
    ]
    with pytest.raises(ValueError, match="not row-aligned"):
        ve.build_veto_multi_arm_report(
            {"conditional_logit": cl, "greedy": greedy, "baseline": _scored_frame(swapped)},
            baseline_arm="baseline",
        )
    with pytest.raises(ValueError, match="baseline_arm"):
        ve.build_veto_multi_arm_report(
            {"conditional_logit": cl, "greedy": greedy},
            baseline_arm="not_an_arm",
        )
    with pytest.raises(ValueError, match="at least two arms"):
        ve.build_veto_multi_arm_report(
            {"greedy": greedy},
            baseline_arm="greedy",
        )


# --------------------------------------------------------------------------
# plan#15g: real v1 end-to-end via the CLI
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py and "
    "splits.py first)",
)
def test_real_v1_veto_evaluation_end_to_end():
    # The M26 finding on real data/v1: run the comparison CLI (which
    # scores the 90 non-decider steps of the 15-match Bo3 test-split
    # veto logs with both arms and writes
    # data/v1/veto_evaluation_report.json), reload the artifact, and
    # assert **internal consistency only** — no predetermined direction
    # is asserted for the greedy-vs-baseline deltas: n_steps is the
    # known 90, cross-entropies are finite and >= 0, accuracies are in
    # [0, 1], the deltas equal the per-arm metric differences, and the
    # report equals an independent in-process recomputation through the
    # pure functions (so a wiring bug in the CLI would be caught). The
    # measured values are printed for the record.
    import pandas as pd

    from drivers.evaluate_veto import main as cli_main

    rc = cli_main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/veto_evaluation_report.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["greedy"]["n_steps"] == 90
    assert artifact["baseline"]["n_steps"] == 90
    for arm in ("greedy", "baseline"):
        assert math.isfinite(artifact[arm]["mean_cross_entropy"])
        assert artifact[arm]["mean_cross_entropy"] >= 0.0
        assert 0.0 <= artifact[arm]["top1_accuracy"] <= 1.0
        assert 0.0 <= artifact[arm]["top3_accuracy"] <= 1.0
        # The by_step_index breakdown recomputes the same aggregates on
        # disjoint subsets whose row counts sum to n_steps.
        by_index = artifact[arm]["by_step_index"]
        assert sum(entry["n_steps"] for entry in by_index) == artifact[arm]["n_steps"]
    # Delta consistency: greedy minus baseline per metric.
    for metric in ("mean_cross_entropy", "top1_accuracy", "top3_accuracy"):
        assert artifact["delta"][metric] == pytest.approx(
            artifact["greedy"][metric] - artifact["baseline"][metric]
        )

    # Independent in-process recomputation through the pure functions
    # (not through the CLI's own scored tables), so the CLI's wiring is
    # double-checked.
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    veto_df = pd.read_parquet("data/v1/veto_actions.parquet")
    held = ve.build_held_out_veto_matches(veto_df, matches_df, splits_df, split="test")
    scored_greedy = ve.score_veto_steps(
        ve.greedy_veto_step_model, held, matches_df, maps_df
    )
    scored_baseline = ve.score_veto_steps(
        ve.most_frequent_map_baseline_model, held, matches_df, maps_df
    )
    report = ve.build_veto_comparison_report(scored_greedy, scored_baseline)
    assert report == artifact

    print(
        "M26 veto evaluation on real v1 test split "
        f"(n_steps={artifact['greedy']['n_steps']}): "
        f"greedy mean_cross_entropy={artifact['greedy']['mean_cross_entropy']!r} "
        f"top1={artifact['greedy']['top1_accuracy']!r} "
        f"top3={artifact['greedy']['top3_accuracy']!r} | "
        f"baseline mean_cross_entropy={artifact['baseline']['mean_cross_entropy']!r} "
        f"top1={artifact['baseline']['top1_accuracy']!r} "
        f"top3={artifact['baseline']['top3_accuracy']!r} | "
        f"delta mean_cross_entropy={artifact['delta']['mean_cross_entropy']!r} "
        f"top1={artifact['delta']['top1_accuracy']!r} "
        f"top3={artifact['delta']['top3_accuracy']!r}"
    )


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py and "
    "splits.py first)",
)
def test_real_v1_ban_training_examples_on_train_split():
    # The decision-12 builder on real v1 train split: 81 matches have
    # veto rows, of which 79 are Bo3 (4 ban steps each = 316) and 2 are
    # Bo5 (2 ban steps each = 4), so exactly 320 ban training examples,
    # each with a resolved opponent (never None), a sorted remaining
    # list whose true-map index is in range, and a date equal to its
    # match's date.
    import pandas as pd

    veto_df = pd.read_parquet("data/v1/veto_actions.parquet")
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    held = ve.build_held_out_veto_matches(veto_df, matches_df, splits_df, split="train")
    examples = ve.build_ban_training_examples(held, matches_df, maps_df)
    assert len(examples) == 320
    assert len({e.acting_team_id for e in examples}) > 1
    for example in examples:
        assert example.opponent_team_id is not None
        assert example.opponent_team_id != example.acting_team_id
        assert 0 <= example.true_map_index < len(example.remaining_maps)
        assert sorted(example.remaining_maps) == list(example.remaining_maps)

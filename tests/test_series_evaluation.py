"""Tests for the series-scoreline evaluation harness core (M33a).

Covers the local ``"Bo<N>"`` parser's exact mappings and its
``ValueError``/``TypeError`` matrix, held-out-series derivation from a
small hand-built matches/maps/splits fixture (exact ``(a_wins, b_wins,
outcome_index)`` for hand-worked Bo3 and Bo5 examples plus the
tied-map / empty-split / zero-maps / null-score ``ValueError``s),
per-series scoring against synthetic stub ``SeriesModelFn``
implementations (a uniform stub and a perfect-oracle stub deriving the
observed scoreline from the fixture tables) with hand-computed RPS /
log-loss / marginal-accuracy values, the per-``best_of`` report
builder against a hand-built mixed-K scored table, the N-arm comparison
report against three hand-computed stub arms (row-alignment /
too-few-arms / unknown-baseline guards), and a skip-guarded end-to-end
run of the whole pipeline against the real M32 flat baseline on
``data/v1``.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation.series_evaluation import (
    _parse_best_of,
    build_held_out_series,
    build_series_evaluation_report,
    build_series_multi_arm_report,
    flat_series_baseline_model,
    score_held_out_series,
)
from utils import series_paths

# --------------------------------------------------------------------------
# Fixtures: a small hand-built league with one Bo3 (m1) and one Bo5 (m2)
# held-out match plus one train-split Bo3 match (m3), so every test sees
# the same deterministic tables.
# --------------------------------------------------------------------------

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "best_of", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]
_SPLITS_COLS = ["match_id", "split"]

_MATCHES = [
    # m1: Bo3 held out, A wins 2-1 -> scoreline (2,1) -> Bo3 index 1.
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    # m2: Bo5 held out, A wins 3-2 -> scoreline (3,2) -> Bo5 index 2.
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo5", "status": "completed"},
    # m3: Bo3 train split, C wins 2-0 -> scoreline (2,0) -> Bo3 index 0.
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
]

_MAPS = [
    # m1: A wins maps 0 and 1, B wins map 2.
    {"match_id": "m1", "map_index": 0, "map_name": "Bind", "team1_score": 13,
     "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven", "team1_score": 13,
     "team2_score": 11, "winner": "A"},
    {"match_id": "m1", "map_index": 2, "map_name": "Split", "team1_score": 8,
     "team2_score": 13, "winner": "B"},
    # m2: A wins maps 0-2, B wins maps 3-4.
    {"match_id": "m2", "map_index": 0, "map_name": "Bind", "team1_score": 13,
     "team2_score": 8, "winner": "A"},
    {"match_id": "m2", "map_index": 1, "map_name": "Haven", "team1_score": 13,
     "team2_score": 9, "winner": "A"},
    {"match_id": "m2", "map_index": 2, "map_name": "Split", "team1_score": 13,
     "team2_score": 11, "winner": "A"},
    {"match_id": "m2", "map_index": 3, "map_name": "Ascent", "team1_score": 7,
     "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 4, "map_name": "Icebox", "team1_score": 9,
     "team2_score": 13, "winner": "B"},
    # m3: C wins both maps.
    {"match_id": "m3", "map_index": 0, "map_name": "Bind", "team1_score": 13,
     "team2_score": 5, "winner": "C"},
    {"match_id": "m3", "map_index": 1, "map_name": "Haven", "team1_score": 13,
     "team2_score": 6, "winner": "C"},
]

_SPLITS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "train"},
]


def _matches_df(rows):
    """Build a matches table with the fixed fixture column set.

    Args:
        rows: A list of dicts, one per match, each carrying the keys in
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
    """Build a maps table with the fixed fixture column set.

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


def _splits_df(rows):
    """Build a splits table with the fixed fixture column set.

    Args:
        rows: A list of dicts, one per match, each carrying
            ``match_id`` and ``split``.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_SPLITS_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_SPLITS_COLS)


def _series_tables():
    """Return the canonical ``(matches, maps, splits)`` fixture triple.

    The shared constructor behind every fixture-based test so the
    column-order/dtype convention lives in one place.

    Returns:
        A ``(matches_df, maps_df, splits_df)`` tuple built from
        :data:`_MATCHES` / :data:`_MAPS` / :data:`_SPLITS`.

    Raises:
        Nothing.
    """
    return _matches_df(_MATCHES), _maps_df(_MAPS), _splits_df(_SPLITS)


def _add_match_and_maps(match_rows, map_rows, split_rows, match, maps):
    """Append one match, its maps, and its split row to mutable lists.

    The single row-writing helper for the failure-case fixtures (tied
    map, null score, zero maps): appends ``match`` to ``match_rows``,
    every dict in ``maps`` to ``map_rows``, and a ``{"match_id",
    "split": "test"}`` row to ``split_rows`` so the new match lands in
    the held-out split.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        split_rows: The mutable split-row list to append to.
        match: The new match dict (``_MATCHES_COLS`` keys).
        maps: A list of map dicts (``_MAPS_COLS`` keys) for the match;
            may be empty for the zero-maps case.

    Returns:
        Nothing (appends in place).

    Raises:
        Nothing.
    """
    match_rows.append(match)
    map_rows.extend(maps)
    split_rows.append({"match_id": match["match_id"], "split": "test"})


# --------------------------------------------------------------------------
# plan#3: the local best-of parser's exact mappings and error matrix
# --------------------------------------------------------------------------


def test_parse_best_of_exact_mappings():
    # The three values observed in data/v1 (and Bo1, which the real
    # dataset lacks but the parser must still handle) convert to the
    # plain odd ints utils.series_paths expects; any other "Bo<N>" with
    # a positive odd N is accepted too.
    assert _parse_best_of("Bo1") == 1
    assert _parse_best_of("Bo3") == 3
    assert _parse_best_of("Bo5") == 5
    assert _parse_best_of("Bo7") == 7
    assert _parse_best_of("Bo9") == 9


def test_parse_best_of_rejects_malformed():
    # Every malformed *string* must raise ValueError, never silently
    # coerce: even map counts, non-numeric suffixes, empty strings,
    # unrelated strings, and a non-positive count. Non-string inputs
    # violate the annotated str contract and raise TypeError instead.
    for bad in (
        "Bo2", "Bo4", "BestOf3", "bo3", "BO3", "Bo", "BoX", "",
        "3", "Bo0", "Bo-1", "Bo3 ",
    ):
        with pytest.raises(ValueError):
            _parse_best_of(bad)
    for bad in (3, None):
        with pytest.raises(TypeError):
            _parse_best_of(bad)


# --------------------------------------------------------------------------
# plan#9a: build_held_out_series derivation and fail-loud guards
# --------------------------------------------------------------------------


def test_build_held_out_series_derives_observed_scorelines():
    # m1 (Bo3, A wins 2-1) and m2 (Bo5, A wins 3-2) are held out, m3 is
    # train; the derived (a_wins, b_wins) pairs and their ordinal
    # indices within series_outcome_order must match the hand-worked
    # values, and the fixed column order must be exactly
    # HELD_OUT_SERIES_COLUMNS.
    matches_df, maps_df, splits_df = _series_tables()
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    assert held_out.columns.tolist() == [
        "match_id", "date", "team1_id", "team2_id", "best_of",
        "best_of_int", "a_wins", "b_wins", "outcome_index",
    ]
    assert len(held_out) == 2
    by_id = {row.match_id: row for row in held_out.itertuples(index=False)}
    m1 = by_id["m1"]
    assert m1.best_of == "Bo3" and m1.best_of_int == 3
    assert (m1.a_wins, m1.b_wins) == (2, 1)
    assert m1.outcome_index == 1  # (2,1) is the 2nd Bo3 ordinal
    m2 = by_id["m2"]
    assert m2.best_of == "Bo5" and m2.best_of_int == 5
    assert (m2.a_wins, m2.b_wins) == (3, 2)
    assert m2.outcome_index == 2  # (3,2) is the 3rd Bo5 ordinal


def test_build_held_out_series_respects_split_argument():
    # The train split holds out m3 only, whose hand-worked scoreline is
    # (2,0) -> Bo3 index 0; the split restriction must reuse the shared
    # helper's semantics (test = everything not train here).
    matches_df, maps_df, splits_df = _series_tables()
    train = build_held_out_series(matches_df, maps_df, splits_df, split="train")
    assert len(train) == 1
    row = train.itertuples(index=False).__next__()
    assert row.match_id == "m3"
    assert (row.a_wins, row.b_wins) == (2, 0)
    assert row.outcome_index == 0


def test_build_held_out_series_raises_on_tied_map():
    # A held-out match with a tied finished map (13-13) is invalid data
    # and must raise loudly rather than silently skip the map.
    match_rows = list(_MATCHES)
    map_rows = list(_MAPS)
    split_rows = list(_SPLITS)
    _add_match_and_maps(
        match_rows,
        map_rows,
        split_rows,
        {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
         "team2_id": "F", "best_of": "Bo3", "status": "completed"},
        [{"match_id": "m4", "map_index": 0, "map_name": "Bind",
          "team1_score": 13, "team2_score": 13, "winner": "E"}],
    )
    with pytest.raises(ValueError, match="tied"):
        build_held_out_series(
            _matches_df(match_rows), _maps_df(map_rows), _splits_df(split_rows)
        )


def test_build_held_out_series_raises_on_null_score():
    # A map whose team2 score is null resolves to neither side's win;
    # the derivation must raise loudly (naming the match) rather than
    # silently undercount the series.
    match_rows = list(_MATCHES)
    map_rows = list(_MAPS)
    split_rows = list(_SPLITS)
    _add_match_and_maps(
        match_rows,
        map_rows,
        split_rows,
        {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
         "team2_id": "F", "best_of": "Bo3", "status": "completed"},
        [{"match_id": "m4", "map_index": 0, "map_name": "Bind",
          "team1_score": 13, "team2_score": None, "winner": "E"}],
    )
    with pytest.raises(ValueError, match="m4"):
        build_held_out_series(
            _matches_df(match_rows), _maps_df(map_rows), _splits_df(split_rows)
        )


def test_build_held_out_series_raises_on_zero_maps():
    # A held-out match with no maps in maps_df has no observable
    # scoreline and must raise naming the match.
    match_rows = list(_MATCHES)
    map_rows = list(_MAPS)
    split_rows = list(_SPLITS)
    _add_match_and_maps(
        match_rows,
        map_rows,
        split_rows,
        {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
         "team2_id": "F", "best_of": "Bo3", "status": "completed"},
        [],
    )
    with pytest.raises(ValueError, match="zero maps"):
        build_held_out_series(
            _matches_df(match_rows), _maps_df(map_rows), _splits_df(split_rows)
        )


def test_build_held_out_series_raises_on_empty_split():
    # A splits table with no rows of the requested value must raise
    # rather than return an empty held-out table (mirrors
    # build_held_out_maps's empty-result guard).
    matches_df, maps_df, _ = _series_tables()
    splits_df = _splits_df(
        [{"match_id": row["match_id"], "split": "train"} for row in _MATCHES]
    )
    with pytest.raises(ValueError, match="no held-out series"):
        build_held_out_series(matches_df, maps_df, splits_df)


def test_build_held_out_series_raises_on_malformed_best_of():
    # A held-out match whose best_of string is malformed must propagate
    # the local parser's ValueError rather than guess a series length.
    match_rows = list(_MATCHES)
    map_rows = list(_MAPS)
    split_rows = list(_SPLITS)
    _add_match_and_maps(
        match_rows,
        map_rows,
        split_rows,
        {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
         "team2_id": "F", "best_of": "BestOf3", "status": "completed"},
        [{"match_id": "m4", "map_index": 0, "map_name": "Bind",
          "team1_score": 13, "team2_score": 8, "winner": "E"}],
    )
    with pytest.raises(ValueError, match="Bo<N>"):
        build_held_out_series(
            _matches_df(match_rows), _maps_df(map_rows), _splits_df(split_rows)
        )


# --------------------------------------------------------------------------
# plan#9b: score_held_out_series against synthetic stub SeriesModelFn
# --------------------------------------------------------------------------


def _uniform_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """A stub SeriesModelFn returning the uniform distribution.

    Returns ``1 / K`` for each of the ``K = best_of + 1`` scoreline
    categories, in series_outcome_order order, for every queried match
    — the least-committal distribution whose per-row RPS / log loss /
    marginal correctness are closed-form hand-computable.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The :data:`SeriesModelFn` arguments; deliberately unused
            by this stub (it ignores the queried series entirely).

    Returns:
        A ``list`` of ``best_of + 1`` equal probabilities summing to
        ``1.0``.

    Raises:
        ValueError: If ``best_of`` is malformed (propagated from
            :func:`evaluation.series_evaluation._parse_best_of`).
    """
    k = _parse_best_of(best_of) + 1
    return [1.0 / k] * k


def _oracle_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """A stub SeriesModelFn placing all mass on the observed scoreline.

    Derives the queried match's observed scoreline from the fixture
    tables (locating the match by its unique ``date``, then counting
    map wins by the same ``team1_score > team2_score`` convention the
    harness itself uses) and returns a one-hot vector over that
    scoreline's ordinal — a perfect predictor whose RPS / log loss are
    exactly ``0.0`` and whose marginal correctness is always ``True``.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The :data:`SeriesModelFn` arguments; ``date`` must be
            unique across ``matches_df`` (it is in the fixture), and
            ``matches_df``/``maps_df`` must carry the M8 columns.

    Returns:
        A ``list`` of ``best_of + 1`` probabilities with a single
        ``1.0`` at the observed scoreline's ordinal and zeros
        elsewhere.

    Raises:
        ValueError: If ``best_of`` is malformed, or if the derived
            scoreline is not terminal (propagated from
            ``series_outcome_order(...).index(...)``).
    """
    match_id = matches_df.loc[
        matches_df["date"] == date, "match_id"
    ].iloc[0]
    match_maps = maps_df[maps_df["match_id"] == match_id]
    a_wins = int((match_maps["team1_score"] > match_maps["team2_score"]).sum())
    b_wins = int((match_maps["team2_score"] > match_maps["team1_score"]).sum())
    best_of_int = _parse_best_of(best_of)
    outcome_index = series_paths.series_outcome_order(best_of_int).index(
        (a_wins, b_wins)
    )
    vector = [0.0] * (best_of_int + 1)
    vector[outcome_index] = 1.0
    return vector


def _short_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """A stub SeriesModelFn returning a wrong-length vector.

    Always returns ``[0.5, 0.5]`` regardless of the queried series, to
    exercise the scorer's per-row length validation.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The :data:`SeriesModelFn` arguments; deliberately unused.

    Returns:
        A 2-element list — never the ``best_of + 1`` length the scorer
        requires for Bo3/Bo5 rows.

    Raises:
        Nothing.
    """
    return [0.5, 0.5]


def _bad_simplex_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """A stub SeriesModelFn returning a non-simplex 4-vector.

    Returns ``[0.25, 0.25, 0.25, 0.5]`` (sums to 1.25) so the metric
    validation in ``utils.scoring`` must reject it with ``ValueError``.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The :data:`SeriesModelFn` arguments; deliberately unused.

    Returns:
        A 4-element vector that does not sum to 1.

    Raises:
        Nothing.
    """
    return [0.25, 0.25, 0.25, 0.5]


def test_score_held_out_series_uniform_stub_hand_computed():
    # Uniform over K categories: RPS at ordinal i is the closed form
    # sum over cuts, log loss is log(K), and marginal correctness is
    # True iff the collapsed side matches. m1 (Bo3, idx 1):
    #   cuts 1..3 -> cum = k/4, true_cdf = 1 for k > 1 ->
    #   (1/4)^2 + (2/4-1)^2 + (3/4-1)^2 = 1/16+4/16+1/16 = 6/16 = 0.375.
    # m2 (Bo5, idx 2): cuts 1..5 with true_cdf flipping at cut 3 ->
    #   gaps (1/6, 2/6, 3/6, 2/6, 1/6) -> 19/36.
    matches_df, maps_df, splits_df = _series_tables()
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    scored = score_held_out_series(
        _uniform_series_model_fn, held_out, matches_df, maps_df
    )
    assert scored.columns.tolist() == [
        "match_id", "date", "team1_id", "team2_id", "best_of",
        "best_of_int", "a_wins", "b_wins", "outcome_index",
        "probabilities", "rps", "log_loss", "marginal_correct",
    ]
    by_id = {row.match_id: row for row in scored.itertuples(index=False)}
    m1 = by_id["m1"]
    assert m1.probabilities == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert m1.rps == pytest.approx(0.375)
    assert m1.log_loss == pytest.approx(math.log(4))
    assert m1.marginal_correct is True  # idx 1 is an A-win scoreline
    m2 = by_id["m2"]
    assert m2.probabilities == pytest.approx([1 / 6] * 6)
    assert m2.rps == pytest.approx(19 / 36)
    assert m2.log_loss == pytest.approx(math.log(6))
    assert m2.marginal_correct is True  # idx 2 is an A-win scoreline


def test_score_held_out_series_oracle_stub_perfect_scores():
    # The perfect-oracle stub must place all mass on each row's true
    # scoreline, yielding rps == log_loss == 0.0 and marginal
    # correctness True for every row, with vector lengths matching
    # best_of_int + 1 per row.
    matches_df, maps_df, splits_df = _series_tables()
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    scored = score_held_out_series(
        _oracle_series_model_fn, held_out, matches_df, maps_df
    )
    by_id = {row.match_id: row for row in scored.itertuples(index=False)}
    m1 = by_id["m1"]
    assert m1.probabilities == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert m1.rps == pytest.approx(0.0)
    assert m1.log_loss == pytest.approx(0.0)
    assert m1.marginal_correct is True
    m2 = by_id["m2"]
    assert len(m2.probabilities) == 6
    assert m2.probabilities[2] == pytest.approx(1.0)
    assert m2.rps == pytest.approx(0.0)
    assert m2.log_loss == pytest.approx(0.0)
    assert m2.marginal_correct is True


def test_score_held_out_series_rejects_wrong_length_vector():
    # A Bo3 row requires a 4-vector; a stub returning 2 probabilities
    # must raise a per-series error naming the match, before any metric
    # computation.
    matches_df, maps_df, splits_df = _series_tables()
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    with pytest.raises(ValueError, match="expected exactly 4"):
        score_held_out_series(
            _short_series_model_fn, held_out, matches_df, maps_df
        )


def test_score_held_out_series_propagates_scoring_validation():
    # A vector that does not sum to 1 is not the scorer's job to
    # pre-check; utils.scoring's simplex validation must propagate
    # unchanged (the "sums to 1" failure, not a length failure).
    matches_df, maps_df, splits_df = _series_tables()
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    with pytest.raises(ValueError, match="sum to 1"):
        score_held_out_series(
            _bad_simplex_series_model_fn, held_out, matches_df, maps_df
        )


# --------------------------------------------------------------------------
# plan#9c: build_series_evaluation_report grouping and JSON contract
# --------------------------------------------------------------------------


def _scored_frame(rows):
    """Build a scored table with just the report's required columns.

    The minimal frame :func:`build_series_evaluation_report` and
    :func:`build_series_multi_arm_report` read (``match_id``,
    ``best_of``, ``outcome_index``, ``probabilities``); the report
    recomputes metrics from the probabilities, so the per-row
    ``rps``/``log_loss``/``marginal_correct`` columns are omitted here.

    Args:
        rows: A list of dicts, each with ``match_id``, ``best_of``,
            ``outcome_index`` and ``probabilities``.

    Returns:
        A ``pandas.DataFrame`` with exactly those four columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(
        rows,
        columns=["match_id", "best_of", "outcome_index", "probabilities"],
    )


def test_build_series_evaluation_report_groups_by_best_of():
    # Four hand-computed rows (two Bo3, two Bo5, all uniform) yield a
    # per-best_of report whose values match the closed forms:
    # Bo3 uniform at idx 0 -> rps 0.875, at idx 3 -> rps 0.875 (both
    #   ll log4; marginal True at idx 0, False at idx 3 -> acc 0.5);
    # Bo5 uniform at idx 5 -> rps 55/36, at idx 1 -> rps 31/36 (both
    #   ll log6; marginal False at idx 5, True at idx 1 -> acc 0.5).
    scored = _scored_frame(
        [
            {"match_id": "x1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "x2", "best_of": "Bo3", "outcome_index": 3,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "x3", "best_of": "Bo5", "outcome_index": 5,
             "probabilities": [1 / 6] * 6},
            {"match_id": "x4", "best_of": "Bo5", "outcome_index": 1,
             "probabilities": [1 / 6] * 6},
        ]
    )
    report = build_series_evaluation_report(scored)
    assert set(report) == {"Bo3", "Bo5", "n_eval_total"}
    assert report["n_eval_total"] == 4
    bo3 = report["Bo3"]
    assert set(bo3) == {
        "n_eval", "mean_rps", "mean_log_loss", "marginal_binary_accuracy",
    }
    assert bo3["n_eval"] == 2
    assert bo3["mean_rps"] == pytest.approx(0.875)
    assert bo3["mean_log_loss"] == pytest.approx(math.log(4))
    assert bo3["marginal_binary_accuracy"] == pytest.approx(0.5)
    bo5 = report["Bo5"]
    assert bo5["n_eval"] == 2
    assert bo5["mean_rps"] == pytest.approx(43 / 36)
    assert bo5["mean_log_loss"] == pytest.approx(math.log(6))
    assert bo5["marginal_binary_accuracy"] == pytest.approx(0.5)
    # Every value is a plain str/int/float/dict: directly dumpable.
    json.dumps(report)


def test_build_series_evaluation_report_omits_zero_row_group():
    # A scored table with only Bo3 rows must report that group plus
    # n_eval_total and omit "Bo5" entirely (a zero-row best_of group is
    # not an error — assumption 5), remaining JSON-serializable.
    scored = _scored_frame(
        [
            {"match_id": "x1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [1.0, 0.0, 0.0, 0.0]},
            {"match_id": "x2", "best_of": "Bo3", "outcome_index": 1,
             "probabilities": [0.0, 1.0, 0.0, 0.0]},
        ]
    )
    report = build_series_evaluation_report(scored)
    assert set(report) == {"Bo3", "n_eval_total"}
    assert report["Bo3"]["n_eval"] == 2
    assert report["Bo3"]["mean_rps"] == pytest.approx(0.0)
    assert report["n_eval_total"] == 2
    json.dumps(report)


def test_build_series_evaluation_report_raises_on_empty():
    # A mean over zero series is undefined: the empty table must raise
    # rather than produce NaN-laden numbers.
    scored = _scored_frame([])
    with pytest.raises(ValueError, match="zero scored series"):
        build_series_evaluation_report(scored)


# --------------------------------------------------------------------------
# plan#9d: build_series_multi_arm_report on hand-computed stub arms
# --------------------------------------------------------------------------


def test_build_series_multi_arm_report_hand_computed():
    # Three arms over two identical Bo3 rows (idx 0 and idx 3):
    #   "uniform" (baseline): [0.25]*4 both -> rps 0.875, ll log4,
    #     acc 0.5 (tie-break A loses row 2);
    #   "oracle": [1,0,0,0] / [0,0,0,1] -> rps 0, ll 0, acc 1.0;
    #   "half": [0.5,0.5,0,0] / [0,0,0.5,0.5] -> rps 0.25, ll log2,
    #     acc 1.0.
    # Deltas vs uniform are arm-minus-baseline per best_of group:
    #   oracle: rps -0.875, ll -log4, acc +0.5;
    #   half: rps -0.625, ll -log2, acc +0.5.
    uniform = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 3,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    oracle = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [1.0, 0.0, 0.0, 0.0]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 3,
             "probabilities": [0.0, 0.0, 0.0, 1.0]},
        ]
    )
    half = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.5, 0.5, 0.0, 0.0]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 3,
             "probabilities": [0.0, 0.0, 0.5, 0.5]},
        ]
    )
    report = build_series_multi_arm_report(
        {"uniform": uniform, "oracle": oracle, "half": half},
        baseline_arm="uniform",
    )
    assert set(report) == {"uniform", "oracle", "half", "deltas_vs_uniform"}
    for name, expected in (
        ("uniform", (0.875, math.log(4), 0.5)),
        ("oracle", (0.0, 0.0, 1.0)),
        ("half", (0.25, math.log(2), 1.0)),
    ):
        block = report[name]["Bo3"]
        assert block["mean_rps"] == pytest.approx(expected[0])
        assert block["mean_log_loss"] == pytest.approx(expected[1])
        assert block["marginal_binary_accuracy"] == pytest.approx(expected[2])
        assert report[name]["n_eval_total"] == 2
    deltas = report["deltas_vs_uniform"]
    assert set(deltas) == {"oracle", "half"}
    assert deltas["oracle"]["Bo3"] == pytest.approx(
        {
            "mean_rps_delta": -0.875,
            "mean_log_loss_delta": -math.log(4),
            "marginal_binary_accuracy_delta": 0.5,
        }
    )
    assert deltas["half"]["Bo3"] == pytest.approx(
        {
            "mean_rps_delta": -0.625,
            "mean_log_loss_delta": -math.log(2),
            "marginal_binary_accuracy_delta": 0.5,
        }
    )
    json.dumps(report)


def test_build_series_multi_arm_report_omits_group_missing_from_baseline():
    # A best_of group present in an arm but absent from the baseline
    # must be omitted from that arm's delta block (assumption 5) while
    # still appearing in its own report block. Both arms are row-aligned
    # (identical match_id values at identical positions — the alignment
    # contract only compares the identifying key), but one row carries a
    # different best_of in the arm's table, so the baseline's report has
    # no Bo5 group while the arm's does.
    baseline = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    arm = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m2", "best_of": "Bo5", "outcome_index": 0,
             "probabilities": [1 / 6] * 6},
        ]
    )
    report = build_series_multi_arm_report(
        {"baseline": baseline, "arm": arm}, baseline_arm="baseline"
    )
    assert "Bo5" in report["arm"]
    assert "Bo5" not in report["baseline"]
    assert report["deltas_vs_baseline"]["arm"] == {"Bo3": pytest.approx(
        {
            "mean_rps_delta": 0.0,
            "mean_log_loss_delta": 0.0,
            "marginal_binary_accuracy_delta": 0.0,
        }
    )}


def test_build_series_multi_arm_report_raises_on_row_misalignment():
    # Arms scored on different held-out rows (same ids, different
    # order) must raise rather than silently pair two different
    # series' scores.
    a = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 1,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    b = _scored_frame(
        [
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 1,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    with pytest.raises(ValueError, match="not row-aligned"):
        build_series_multi_arm_report({"a": a, "b": b}, baseline_arm="a")


def test_build_series_multi_arm_report_raises_on_row_count_mismatch():
    # Arms with different row counts describe different held-out sets
    # and must raise before any delta is computed.
    a = _scored_frame(
        [{"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
          "probabilities": [0.25, 0.25, 0.25, 0.25]}]
    )
    b = _scored_frame(
        [
            {"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"match_id": "m2", "best_of": "Bo3", "outcome_index": 1,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    with pytest.raises(ValueError, match="different row counts"):
        build_series_multi_arm_report({"a": a, "b": b}, baseline_arm="a")


def test_build_series_multi_arm_report_raises_on_too_few_arms():
    # A one-arm "comparison" is meaningless and must raise.
    scored = _scored_frame(
        [{"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
          "probabilities": [0.25, 0.25, 0.25, 0.25]}]
    )
    with pytest.raises(ValueError, match="at least two arms"):
        build_series_multi_arm_report({"a": scored}, baseline_arm="a")


def test_build_series_multi_arm_report_raises_on_unknown_baseline():
    # A baseline_arm that is not a scored arm must raise naming the
    # offending key rather than silently skipping the delta block.
    scored = _scored_frame(
        [{"match_id": "m1", "best_of": "Bo3", "outcome_index": 0,
          "probabilities": [0.25, 0.25, 0.25, 0.25]}]
    )
    with pytest.raises(ValueError, match="is not a scored arm"):
        build_series_multi_arm_report(
            {"a": scored, "b": scored}, baseline_arm="nope"
        )


# --------------------------------------------------------------------------
# plan#9e: real v1 end-to-end through the M32 flat baseline adapter
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the materialised v1 tables exist on disk.

    The skip guard for the real-data test, matching the convention in
    ``test_flat_series_baseline.py`` / ``test_map_win_rate.py``: all
    three Parquet files must exist (i.e. ``materialize.py`` /
    ``splits.py`` have been run).

    Returns:
        A bool: ``True`` iff ``data/v1/matches.parquet``,
        ``data/v1/maps.parquet`` and ``data/v1/splits.parquet`` all
        exist.

    Raises:
        Nothing.
    """
    return (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
        and Path("data/v1/splits.parquet").exists()
    )


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py/splits.py first)",
)
def test_real_v1_flat_baseline_end_to_end():
    # plan#9e: run the whole harness against the real M32 flat baseline
    # on the real v1 test split. The held-out table is non-empty, every
    # derived scoreline is terminal, the scored probabilities are valid
    # simplexes of the per-row length, and the report's Bo3 group (the
    # only group v1's all-Bo3 test split produces) has finite mean_rps
    # within [0, best_of] (the K-1 RPS ceiling for K=4), a finite
    # non-negative mean_log_loss, and a marginal accuracy in [0, 1].
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    assert len(held_out) > 0
    scored = score_held_out_series(
        flat_series_baseline_model, held_out, matches_df, maps_df
    )
    assert len(scored) == len(held_out)
    for row in scored.itertuples(index=False):
        probs = row.probabilities
        assert len(probs) == row.best_of_int + 1
        assert sum(probs) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs)

    report = build_series_evaluation_report(scored)
    assert report["n_eval_total"] == len(scored)
    # Every distinct best_of present is a report key (v1's test split is
    # all Bo3 today, so "Bo5" may legitimately be absent — assumption 5).
    for best_of in scored["best_of"].unique():
        assert best_of in report
    assert "Bo3" in report
    bo3 = report["Bo3"]
    assert bo3["n_eval"] == len(scored[scored["best_of"] == "Bo3"])
    assert math.isfinite(bo3["mean_rps"])
    assert 0.0 <= bo3["mean_rps"] <= 3.0  # max RPS for K=4 is K-1 = best_of
    assert math.isfinite(bo3["mean_log_loss"])
    assert bo3["mean_log_loss"] >= 0.0
    assert 0.0 <= bo3["marginal_binary_accuracy"] <= 1.0
    json.dumps(report)

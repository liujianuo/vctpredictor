"""Tests for the M38 per-category reliability-diagrams CLI driver
(drivers/evaluate_reliability_diagrams.py).

Covers the CLI/IO glue only — the pure binned-calibration math is
already tested in tests/test_reliability_diagrams.py, the M19 map
harness in tests/test_evaluation_harness.py and the M31 series
pipeline in tests/test_veto_marginalized_series.py. The tests here
are: ``parse_args`` defaults and flag overrides; a synthetic
end-to-end ``main()`` run with the table loaders, the ``MODEL_REGISTRY
["ordinal_logit"]`` factory and the M31 ``make_series_model_fn``
factory monkeypatched to deterministic one-hot-perfect stubs (so every
per-category ECE in both arms is exactly 0 — a clean wiring assertion:
any misalignment between prediction columns, true indices, labels or
the per-``best_of`` grouping breaks it), covering both arms and a
multi-``best_of`` case where the Bo5 group (1 series) is too sparse
for ``--n-bins-series 3`` and is omitted with the decision-9
skip-and-warn (asserted via caplog), plus a second run at
``--n-bins-series 1`` proving the Bo5-included branch end to end; the
missing-prerequisite-artifact ``FileNotFoundError`` contract; invalid
flag values; and a ``skipif``-guarded real-v1 integration smoke test
asserting finite, non-negative ECEs, bin counts summing to ``n_eval``
per category, the expected Bo5-absence in the series block, and a
``json.dumps``-serializable report. No real fitted artifacts are
required by the non-smoke tests.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate_reliability_diagrams as erd
from evaluation import harness

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_BINS_MAP = erd.DEFAULT_N_BINS_MAP
DEFAULT_N_BINS_SERIES = erd.DEFAULT_N_BINS_SERIES
DEFAULT_N_SAMPLES = erd.DEFAULT_N_SAMPLES
DEFAULT_SEED = erd.DEFAULT_SEED

# A tiny hand-built league exercising both arms: m1/m2/m3 Bo3 test
# matches, m4 a Bo5 test match, m5 a Bo3 train match (held out of both
# evaluations). Every scoreline is decisive (never tied, never null) so
# build_held_out_series can derive each observed series scoreline: m1,
# m2 and m3 all finish (2, 1) (Bo3 outcome index 1); m4 finishes
# (3, 2) (Bo5 outcome index 2). Team ids/dates are arbitrary — every
# stub model keys its one-hot prediction off the match date.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo5", "status": "completed"},
    {"match_id": "m5", "date": "2026-01-05T00:00:00", "team1_id": "E",
     "team2_id": "F", "best_of": "Bo3", "status": "completed"},
]

# Map rows: every test match's scoreline is 2-1 (A wins maps 0 and 1)
# for the Bo3s and 3-2 for the Bo5; m5 (train) is 2-1 too.
_MAP_ROWS = [
    {"match_id": "m1", "map_index": i, "map_name": name,
     "team1_score": 8 if i == 2 else 13, "team2_score": 13 if i == 2 else 8,
     "winner": "B" if i == 2 else "A"}
    for i, name in enumerate(("Bind", "Haven", "Split"))
] + [
    {"match_id": "m2", "map_index": i, "map_name": name,
     "team1_score": 8 if i == 2 else 13, "team2_score": 13 if i == 2 else 8,
     "winner": "B" if i == 2 else "A"}
    for i, name in enumerate(("Bind", "Haven", "Split"))
] + [
    {"match_id": "m3", "map_index": i, "map_name": name,
     "team1_score": 8 if i == 1 else 13, "team2_score": 13 if i == 1 else 8,
     "winner": "B" if i == 1 else "A"}
    for i, name in enumerate(("Bind", "Haven", "Split"))
] + [
    {"match_id": "m4", "map_index": i, "map_name": name,
     "team1_score": 8 if i in (2, 3) else 13,
     "team2_score": 13 if i in (2, 3) else 8,
     "winner": "D" if i in (2, 3) else "C"}
    for i, name in enumerate(("Bind", "Haven", "Split", "Ascent", "Icebox"))
] + [
    {"match_id": "m5", "map_index": i, "map_name": name,
     "team1_score": 8 if i == 2 else 13, "team2_score": 13 if i == 2 else 8,
     "winner": "B" if i == 2 else "A"}
    for i, name in enumerate(("Bind", "Haven", "Split"))
]

_SPLIT_ROWS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "test"},
    {"match_id": "m4", "split": "test"},
    {"match_id": "m5", "split": "train"},
]

# The map-arm four-way outcome ordinal every held-out map of a match
# carries (the map-arm stub returns the one-hot of this ordinal, and
# the labels table encodes exactly this mapping per map — so the
# predictions are perfectly calibrated by construction). m1 -> 0,
# m2 -> 1, m3 -> 2, m4 -> 3; m5 is train-only (never scored by the
# map arm), its value is irrelevant.
_MAP_ORDINAL_BY_DATE = {
    "2026-01-01T00:00:00": 0,  # m1
    "2026-01-02T00:00:00": 1,  # m2
    "2026-01-03T00:00:00": 2,  # m3
    "2026-01-04T00:00:00": 3,  # m4
    "2026-01-05T00:00:00": 3,  # m5 (train)
}

# The series-arm outcome index of each held-out match's derived
# scoreline (the series stub returns the one-hot of this index):
# (2, 1) is Bo3 outcome index 1, (3, 2) is Bo5 outcome index 2.
_SERIES_INDEX_BY_DATE = {
    "2026-01-01T00:00:00": 1,  # m1 Bo3 (2, 1)
    "2026-01-02T00:00:00": 1,  # m2 Bo3 (2, 1)
    "2026-01-03T00:00:00": 1,  # m3 Bo3 (2, 1)
    "2026-01-04T00:00:00": 2,  # m4 Bo5 (3, 2)
}


def _league_tables():
    """Build the synthetic matches/maps/labels/splits frames for the stub run.

    Assembles the frames from the module-level row constants, deriving
    the labels table's per-map four-way ``outcome_ordinal`` from
    :data:`_MAP_ORDINAL_BY_DATE` (each map inherits its match date's
    ordinal) so the stub model's one-hot predictions and the scored
    truth can never drift out of sync — the league is consistent by
    construction.

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects.

    Raises:
        Nothing.
    """
    matches_df = pd.DataFrame(
        _MATCH_ROWS,
        columns=["match_id", "date", "team1_id", "team2_id", "best_of", "status"],
    )
    maps_df = pd.DataFrame(
        _MAP_ROWS,
        columns=["match_id", "map_index", "map_name", "team1_score", "team2_score", "winner"],
    )
    date_by_match = {
        row.match_id: row.date for row in matches_df.itertuples(index=False)
    }
    label_rows = [
        {
            "match_id": map_row["match_id"],
            "map_index": map_row["map_index"],
            "outcome_ordinal": _MAP_ORDINAL_BY_DATE[
                date_by_match[map_row["match_id"]]
            ],
        }
        for map_row in _MAP_ROWS
    ]
    labels_df = pd.DataFrame(
        label_rows, columns=["match_id", "map_index", "outcome_ordinal"]
    )
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, labels_df, splits_df


def _one_hot(ordinal: int, k: int) -> tuple[float, ...]:
    """Build a one-hot probability vector of length ``k``.

    Returns a valid-simplex vector with ``1.0`` at position ``ordinal``
    and ``0.0`` elsewhere — the stub models' perfect prediction for an
    observation whose true category is ``ordinal``.

    Args:
        ordinal: The category index to place the unit mass on, in
            ``[0, k)``.
        k: The vector length (the number of categories).

    Returns:
        A ``k``-tuple of floats with ``1.0`` at index ``ordinal``.

    Raises:
        IndexError: If ``ordinal`` is outside ``[0, k)``.
    """
    return tuple(1.0 if j == ordinal else 0.0 for j in range(k))


def _stub_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Stub four-way map model returning a one-hot of the map's true outcome.

    Stands in for the fitted M20 ordinal-logit model in the patched
    end-to-end tests: keys the one-hot prediction off the map's match
    date via :data:`_MAP_ORDINAL_BY_DATE`, so every held-out map is
    predicted perfectly and the map arm's per-category ECEs are exactly
    0 — any wiring error (wrong prediction-column order, misaligned
    true indices, wrong category labels) would show up as a nonzero
    ECE and break the test's ``== 0`` assertions.

    Args:
        team1_id / team2_id / map_name / date / matches_df / maps_df:
            The ``ModelFn``-interface arguments; only ``date`` is read.

    Returns:
        The one-hot 4-vector of ``_MAP_ORDINAL_BY_DATE[date]``.

    Raises:
        KeyError: If ``date`` is not a key of :data:`_MAP_ORDINAL_BY_DATE`
            (a held-out match the test league forgot to cover).
    """
    return _one_hot(_MAP_ORDINAL_BY_DATE[date], 4)


def _stub_map_factory(output_dir, version, call_state=None):
    """Stub replacement for MODEL_REGISTRY['ordinal_logit'] returning the map stub.

    Records the ``(output_dir, version)`` it was invoked with (so the
    test can assert the map arm reached the registry factory exactly
    once with the resolved output location) and returns
    :func:`_stub_map_model_fn`.

    Args:
        output_dir: The directory argument the driver passes to the
            factory (recorded).
        version: The version argument the driver passes (recorded).
        call_state: An optional dict whose ``"map_factory_calls"`` key
            records each invocation's ``(output_dir, version)``.

    Returns:
        :func:`_stub_map_model_fn`.

    Raises:
        Nothing.
    """
    if call_state is not None:
        call_state["map_factory_calls"].append((str(output_dir), version))
    return _stub_map_model_fn


def _stub_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """Stub SeriesModelFn returning a one-hot of the series' true scoreline.

    Stands in for the real M31 veto-marginalized pipeline in the
    patched end-to-end tests: keys the one-hot prediction off the
    match date via :data:`_SERIES_INDEX_BY_DATE`, so every held-out
    series is predicted perfectly and the series arm's per-category
    ECEs are exactly 0 (the same wiring-assertion trick as
    :func:`_stub_map_model_fn`). The vector length is ``best_of + 1``
    (``K`` varies by group, exactly like the real M31 output).

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The ``SeriesModelFn``-interface arguments; only ``best_of``
            (fixing ``K``) and ``date`` (fixing the true index) are
            read.

    Returns:
        The one-hot ``best_of + 1``-vector of
        ``_SERIES_INDEX_BY_DATE[date]``.

    Raises:
        KeyError: If ``date`` is not a key of
            :data:`_SERIES_INDEX_BY_DATE` (a held-out series the test
            league forgot to cover).
    """
    n = int(best_of[2:])
    return _one_hot(_SERIES_INDEX_BY_DATE[date], n + 1)


def _stub_make_series_model_fn(
    map_model_fn,
    predictor_fn_by_action,
    n_samples,
    rng,
    map_pool=None,
    call_state=None,
):
    """Stub replacement for make_series_model_fn returning the series stub.

    Asserts the driver wired the two pluggable callables and the
    requested ``n_samples`` / ``rng`` through correctly (the factory
    contract the real M31 factory also has), records the resolved
    ``n_samples`` and the rng into ``call_state``, and returns
    :func:`_stub_series_model_fn`.

    Args:
        map_model_fn: The Stage-2 four-way map model the driver built
            (asserted callable).
        predictor_fn_by_action: The Stage-1 predictor dict the driver
            built (asserted to carry exactly ``ban`` and ``pick``).
        n_samples: The ``--n-samples`` value (recorded and asserted
            positive).
        rng: The seed-derived ``numpy.random.Generator`` (asserted a
            Generator and recorded by identity).
        map_pool: Unused; accepted for signature parity.
        call_state: An optional dict whose ``"series_factory_calls"``
            key records each invocation's ``(n_samples, id(rng))``.

    Returns:
        :func:`_stub_series_model_fn`.

    Raises:
        AssertionError: If any wiring assertion fails (wrong callable
            types, missing predictor key, non-positive ``n_samples``,
            or a non-Generator ``rng``).
    """
    assert callable(map_model_fn)
    assert set(predictor_fn_by_action) == {"ban", "pick"}
    assert n_samples > 0
    assert isinstance(rng, np.random.Generator)
    if call_state is not None:
        call_state["series_factory_calls"].append((n_samples, id(rng)))
    return _stub_series_model_fn


@pytest.fixture
def stub_everything(monkeypatch):
    """Monkeypatch the driver's table loaders and model factories.

    Routes the four input tables to the synthetic league and replaces
    every fitted-artifact / model-factory call with a deterministic
    one-hot-perfect stub, so :func:`erd.main` runs the real CLI/IO, the
    real harnesses and the real pure reliability module without any
    real fitted artifacts and without any real M31 sampling. Call state
    (the registry-factory invocations and the series-factory
    invocations) is exposed on the returned dict. All patches are
    reverted by monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        A dict of call state: ``map_factory_calls`` (list of
        ``(output_dir, version)`` tuples) and ``series_factory_calls``
        (list of ``(n_samples, id(rng))`` tuples).

    Raises:
        Nothing.
    """
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    call_state = {"map_factory_calls": [], "series_factory_calls": []}
    monkeypatch.setattr(
        erd.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_labels_table",
        lambda output_dir, version: labels_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    monkeypatch.setitem(
        erd.evaluate.MODEL_REGISTRY, "ordinal_logit",
        lambda output_dir, version: _stub_map_factory(
            output_dir, version, call_state=call_state
        ),
    )
    monkeypatch.setattr(
        erd, "_load_fitted_models",
        lambda output_dir, version: (None, None, None),
    )
    monkeypatch.setattr(
        erd.ordinal_logit, "make_model_fn",
        lambda model, player_map_stats_df: _stub_map_model_fn,
    )
    monkeypatch.setattr(
        erd.conditional_logit_ban, "make_veto_step_predictor_fn",
        lambda model: (lambda acting_team_id, action, remaining_maps,
                       date, matches_df, maps_df: [0.5] * len(remaining_maps)),
    )
    monkeypatch.setattr(
        erd.conditional_logit_pick, "make_veto_step_predictor_fn",
        lambda model: (lambda acting_team_id, action, remaining_maps,
                       date, matches_df, maps_df: [0.5] * len(remaining_maps)),
    )
    monkeypatch.setattr(
        erd.veto_marginalized_series, "make_series_model_fn",
        lambda map_model_fn, predictor_fn_by_action, n_samples, rng:
        _stub_make_series_model_fn(
            map_model_fn,
            predictor_fn_by_action,
            n_samples,
            rng,
            call_state=call_state,
        ),
    )
    return call_state


# --------------------------------------------------------------------------
# plan#5: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the six documented defaults (version v1, output_dir
    # data, and the decision-10 bin counts / measured single-pass
    # n_samples / repo seed, referenced through the module constants).
    args = erd.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_bins_map == DEFAULT_N_BINS_MAP
    assert args.n_bins_series == DEFAULT_N_BINS_SERIES
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int flag values are rejected
    # by argparse (SystemExit).
    args = erd.parse_args(
        ["--version", "v2", "--output-dir", "out", "--n-bins-map", "7",
         "--n-bins-series", "4", "--n-samples", "3", "--seed", "42"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_bins_map == 7
    assert args.n_bins_series == 4
    assert args.n_samples == 3
    assert args.seed == 42
    with pytest.raises(SystemExit):
        erd.parse_args(["--n-bins-map", "many"])
    with pytest.raises(SystemExit):
        erd.parse_args(["--n-samples", "many"])


# --------------------------------------------------------------------------
# plan#5: synthetic end-to-end main() — both arms + sparse-group skip
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_both_arms_with_sparse_skip(
    tmp_path, stub_everything, caplog
):
    # A full main() run against the synthetic league with every loader/
    # factory stubbed and one-hot-perfect predictions (the league is
    # consistent by construction): the artifact is written with both
    # arms' blocks and the four provenance knobs; the map arm covers
    # all 14 held-out maps with the four OUTCOME_LABELS categories and
    # every per-category ECE is exactly 0 (the wiring assertion); the
    # series arm's Bo3 group (3 series >= n_bins_series 3) is included
    # with the ["2-0","2-1","1-2","0-2"] scoreline labels and ECEs of
    # exactly 0, while the Bo5 group (1 series < 3) is OMITTED with the
    # decision-9 warning naming the group and its count (asserted via
    # caplog); n_eval_total counts all 4 scored series.
    call_state = stub_everything
    rc = erd.main(
        ["--output-dir", str(tmp_path),
         "--n-bins-map", "3", "--n-bins-series", "3",
         "--n-samples", "4", "--seed", "2026"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "reliability_diagrams_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "map", "series", "n_bins_map", "n_bins_series", "n_samples", "seed"
    }
    assert report["n_bins_map"] == 3
    assert report["n_bins_series"] == 3
    assert report["n_samples"] == 4
    assert report["seed"] == 2026

    # Wiring: the map arm reached the registry factory exactly once
    # with the resolved output location, and the series arm's factory
    # once with the passed n_samples and a Generator.
    assert call_state["map_factory_calls"] == [(str(tmp_path), "v1")]
    assert len(call_state["series_factory_calls"]) == 1
    recorded_n_samples, recorded_rng_id = call_state["series_factory_calls"][0]
    assert recorded_n_samples == 4
    assert isinstance(recorded_rng_id, int)

    # Map arm: 14 held-out maps (3+3+3 Bo3 + 5 Bo5), four OUTCOME_LABELS
    # categories, every ECE exactly 0 (perfect one-hot stub), bin counts
    # summing to n_eval per category.
    map_block = report["map"]
    assert map_block["n_eval"] == 14
    assert map_block["n_bins"] == 3
    assert [c["category"] for c in map_block["categories"]] == list(
        harness.OUTCOME_LABELS
    )
    for category in map_block["categories"]:
        assert category["expected_calibration_error"] == pytest.approx(0.0)
        assert len(category["bins"]) == 3
        assert sum(b["count"] for b in category["bins"]) == 14

    # Series arm: Bo3 included, Bo5 omitted (1 < n_bins_series 3) with
    # the skip-and-warn, n_eval_total counting all 4 scored series.
    series_block = report["series"]
    assert set(series_block) == {"Bo3", "n_eval_total"}
    assert series_block["n_eval_total"] == 4
    bo3 = series_block["Bo3"]
    assert bo3["n_eval"] == 3
    assert bo3["n_bins"] == 3
    assert [c["category"] for c in bo3["categories"]] == [
        "2-0", "2-1", "1-2", "0-2"
    ]
    for category in bo3["categories"]:
        assert category["expected_calibration_error"] == pytest.approx(0.0)
        assert sum(b["count"] for b in category["bins"]) == 3

    # The decision-9 warning names the omitted group and its count.
    warning_text = " ".join(
        record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )
    assert "Bo5" in warning_text
    assert "1 held-out series" in warning_text

    # The full report round-trips through json.dumps.
    json.dumps(report)


def test_main_end_to_end_synthetic_bo5_included(tmp_path, stub_everything):
    # The same league at --n-bins-series 1: the Bo5 group (1 series >=
    # 1) is now INCLUDED, so the series block carries both Bo3 and Bo5
    # with their own K (4- and 6-category) reports — proving the
    # multi-best_of, Bo5-present branch of the series arm end to end on
    # a hand-built table (real v1 data has no Bo5 test-split rows). The
    # Bo5 scoreline labels are the f"{a}-{b}" forms of
    # series_outcome_order(5) and every ECE is exactly 0.
    rc = erd.main(
        ["--output-dir", str(tmp_path),
         "--n-bins-map", "3", "--n-bins-series", "1",
         "--n-samples", "4", "--seed", "2026"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "reliability_diagrams_report.json"
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    series_block = report["series"]
    assert set(series_block) == {"Bo3", "Bo5", "n_eval_total"}
    assert series_block["n_eval_total"] == 4
    bo5 = series_block["Bo5"]
    assert bo5["n_eval"] == 1
    assert bo5["n_bins"] == 1
    assert [c["category"] for c in bo5["categories"]] == [
        "3-0", "3-1", "3-2", "2-3", "1-3", "0-3"
    ]
    for category in bo5["categories"]:
        assert category["expected_calibration_error"] == pytest.approx(0.0)
        assert sum(b["count"] for b in category["bins"]) == 1
    bo3 = series_block["Bo3"]
    assert bo3["n_eval"] == 3
    assert [c["category"] for c in bo3["categories"]] == [
        "2-0", "2-1", "1-2", "0-2"
    ]


def test_main_end_to_end_synthetic_map_arm_nonzero_ece_detects_wiring(tmp_path):
    # A negative control for the ECE==0 wiring assertion: with the map
    # arm's stub predicting a FIXED distribution (0.7 on category 0,
    # 0.1 on the other three — a valid simplex giving every true
    # category positive probability, so log_loss is defined) for every
    # map while the true labels vary across matches, the one-vs-rest
    # columns no longer track the truth and at least one category must
    # carry a strictly positive ECE — proving the end-to-end numbers
    # are real signals of the prediction-vs-truth pairing, not
    # constants the driver fabricates. (Runs with the series arm
    # stubbed out too.)
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )

    def fixed_map_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
        """Return a constant 0.7/0.1/0.1/0.1 distribution for every map.

        Args:
            team1_id / team2_id / map_name / date / matches_df / maps_df:
                The ModelFn-interface arguments; deliberately unused.

        Returns:
            The fixed simplex ``(0.7, 0.1, 0.1, 0.1)`` — positive mass
            on every category (so ``log_loss`` is defined for any true
            ordinal) but systematically overpredicting category 0.

        Raises:
            Nothing.
        """
        return (0.7, 0.1, 0.1, 0.1)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        erd.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_labels_table",
        lambda output_dir, version: labels_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    monkeypatch.setitem(
        erd.evaluate.MODEL_REGISTRY, "ordinal_logit",
        lambda output_dir, version: fixed_map_fn,
    )
    monkeypatch.setattr(
        erd, "_load_fitted_models",
        lambda output_dir, version: (None, None, None),
    )
    monkeypatch.setattr(
        erd.veto_marginalized_series, "make_series_model_fn",
        lambda map_model_fn, predictor_fn_by_action, n_samples, rng:
        _stub_series_model_fn,
    )
    try:
        rc = erd.main(
            ["--output-dir", str(tmp_path),
             "--n-bins-map", "3", "--n-bins-series", "3",
             "--n-samples", "4", "--seed", "2026"]
        )
    finally:
        monkeypatch.undo()
    assert rc == 0

    artifact_path = tmp_path / "v1" / "reliability_diagrams_report.json"
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    eces = [
        category["expected_calibration_error"]
        for category in report["map"]["categories"]
    ]
    # 11 of 14 maps are not category 0, and the observed category mix
    # within every bin (0.6 / 0.0 / 0.0 across the three bins) sits far
    # below the constant 0.7 prediction, so at least one non-zero-
    # observed category must be miscalibrated against the fixed
    # distribution.
    assert any(ece > 0.0 for ece in eces)


# --------------------------------------------------------------------------
# plan#5: missing prerequisite artifact and invalid flag values
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(
    tmp_path, monkeypatch
):
    # The tables exist (stubbed to the league) but no fitted model
    # artifact exists under the empty tmp output dir: the map arm's
    # registry factory must raise FileNotFoundError unchanged when it
    # opens ordinal_logit_model.json (the "run the training driver
    # first" signal), never a silent fallback or a wrapped exception.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    monkeypatch.setattr(
        erd.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_labels_table",
        lambda output_dir, version: labels_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        erd.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    with pytest.raises(FileNotFoundError):
        erd.main(["--output-dir", str(tmp_path)])


def test_main_rejects_invalid_flag_values(tmp_path):
    # Non-positive n_samples / n_bins_map / n_bins_series are hard
    # errors before any work starts (even with everything stubbed away
    # the validation must fire first).
    with pytest.raises(ValueError, match="--n-samples"):
        erd.main(["--output-dir", str(tmp_path), "--n-samples", "0"])
    with pytest.raises(ValueError, match="--n-bins-map"):
        erd.main(["--output-dir", str(tmp_path), "--n-bins-map", "0"])
    with pytest.raises(ValueError, match="--n-bins-series"):
        erd.main(["--output-dir", str(tmp_path), "--n-bins-series", "-1"])


# --------------------------------------------------------------------------
# plan#5: real-v1 integration smoke test (skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and model artifacts exist.

    The skip guard for the real-data smoke test: the materialised v1
    matches/maps/labels/splits/player_map_stats tables plus the fitted
    ordinal-logit and ban/pick conditional-logit model artifacts must
    all be present (i.e. ``materialize.py`` / ``labels.py`` /
    ``splits.py`` and the model training drivers have been run).

    Returns:
        A bool: ``True`` iff every required ``data/v1`` file exists.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}").exists()
        for name in (
            "matches.parquet",
            "maps.parquet",
            "labels.parquet",
            "splits.parquet",
            "player_map_stats.parquet",
            "ordinal_logit_model.json",
            "conditional_logit_ban_model.json",
            "conditional_logit_pick_model.json",
        )
    )


@pytest.mark.skipif(
    not _real_v1_available(), reason="real v1 tables/artifacts not present"
)
def test_real_v1_smoke_finite_nonnegative_eces():
    # A tiny real-v1 run (n_samples=2) against the real fitted models:
    # the artifact must be written with both arms and the provenance
    # knobs; the map arm must cover the 35 held-out maps under the four
    # OUTCOME_LABELS categories and the series arm the 15 held-out Bo3
    # series (with no Bo5 group — the real split is 100% Bo3, so the
    # decision-9 skip fires, guarded, never a bare key index); every
    # per-category ECE must be finite and non-negative and every
    # category's bin counts must sum to its n_eval; the f"{a}-{b}"
    # scoreline labels must match series_outcome_order; and every value
    # must be plain json.dumps-serializable.
    rc = erd.main(["--n-samples", "2", "--seed", "2026"])
    assert rc == 0

    artifact_path = Path("data/v1/reliability_diagrams_report.json")
    assert artifact_path.exists()
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert set(report) == {
        "map", "series", "n_bins_map", "n_bins_series", "n_samples", "seed"
    }
    assert report["n_bins_map"] == DEFAULT_N_BINS_MAP
    assert report["n_bins_series"] == DEFAULT_N_BINS_SERIES
    assert report["n_samples"] == 2
    assert report["seed"] == 2026

    map_block = report["map"]
    assert map_block["n_eval"] == 35
    assert map_block["n_bins"] == DEFAULT_N_BINS_MAP
    assert [c["category"] for c in map_block["categories"]] == list(
        harness.OUTCOME_LABELS
    )
    for category in map_block["categories"]:
        assert category["expected_calibration_error"] >= 0.0
        assert len(category["bins"]) == DEFAULT_N_BINS_MAP
        assert sum(b["count"] for b in category["bins"]) == 35

    series_block = report["series"]
    assert "Bo3" in series_block
    assert "Bo5" not in series_block  # 0 Bo5 test-split series in v1
    assert series_block["n_eval_total"] == 15
    bo3 = series_block["Bo3"]
    assert bo3["n_eval"] == 15
    assert bo3["n_bins"] == DEFAULT_N_BINS_SERIES
    assert [c["category"] for c in bo3["categories"]] == [
        "2-0", "2-1", "1-2", "0-2"
    ]
    for category in bo3["categories"]:
        assert category["expected_calibration_error"] >= 0.0
        assert len(category["bins"]) == DEFAULT_N_BINS_SERIES
        assert sum(b["count"] for b in category["bins"]) == 15

    # The full report round-trips through json.dumps.
    json.dumps(report)

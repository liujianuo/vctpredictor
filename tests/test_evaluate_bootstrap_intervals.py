"""Tests for the M36 bootstrap-intervals CLI driver
(drivers/evaluate_bootstrap_intervals.py).

Covers the CLI/IO glue only — the pure interval math is already tested
in tests/test_bootstrap_intervals.py, the block-bootstrap resampler in
tests/test_training_data.py, and the M31 pipeline in
tests/test_veto_marginalized_series.py. The tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the table/artifact loaders, the model factories and
the resamplers monkeypatched to fast deterministic stubs whose outputs
are hand-computable (so the per-map and per-series intervals are
cross-checked against independently re-derived numpy percentiles, and
``n_games_backing`` against the stubbed game counts), including the
per-replicate call-count wiring; the missing-prerequisite-artifact
``FileNotFoundError`` contract; and a ``skipif``-guarded real-v1
integration smoke test asserting well-ordered finite intervals. No real
fitted artifacts are required by the non-smoke tests.
"""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate_bootstrap_intervals as ebi
from features import map_win_rate

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_BOOTSTRAP_MAP = ebi.DEFAULT_N_BOOTSTRAP_MAP
DEFAULT_N_BOOTSTRAP_SERIES = ebi.DEFAULT_N_BOOTSTRAP_SERIES
DEFAULT_BOOTSTRAP_SEED = ebi.DEFAULT_BOOTSTRAP_SEED
DEFAULT_VETO_SEED = ebi.DEFAULT_VETO_SEED
DEFAULT_VETO_N_SAMPLES = ebi.DEFAULT_VETO_N_SAMPLES
DEFAULT_CI_LEVEL = ebi.DEFAULT_CI_LEVEL

# A tiny hand-built league (the evaluate_series league): m1/m2 Bo3 test
# matches, m3 a Bo5 test match, m4 a Bo3 train match. Team ids/dates are
# arbitrary — every stub model ignores the tables.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo5", "status": "completed"},
    {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
     "team2_id": "F", "best_of": "Bo3", "status": "completed"},
]

_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 11, "winner": "A"},
    {"match_id": "m1", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 0, "map_name": "Bind",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m2", "map_index": 2, "map_name": "Split",
     "team1_score": 13, "team2_score": 10, "winner": "A"},
    {"match_id": "m3", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 5, "winner": "C"},
    {"match_id": "m3", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "C"},
    {"match_id": "m3", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "D"},
    {"match_id": "m3", "map_index": 3, "map_name": "Ascent",
     "team1_score": 9, "team2_score": 13, "winner": "D"},
    {"match_id": "m3", "map_index": 4, "map_name": "Icebox",
     "team1_score": 13, "team2_score": 11, "winner": "C"},
    {"match_id": "m4", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 6, "winner": "E"},
    {"match_id": "m4", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "E"},
    {"match_id": "m4", "map_index": 2, "map_name": "Split",
     "team1_score": 5, "team2_score": 13, "winner": "F"},
]

_SPLIT_ROWS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "test"},
    {"match_id": "m4", "split": "train"},
]

_LABEL_ROWS = [
    {"match_id": "m1", "map_index": 0, "outcome_ordinal": 0},
    {"match_id": "m1", "map_index": 1, "outcome_ordinal": 1},
    {"match_id": "m1", "map_index": 2, "outcome_ordinal": 2},
    {"match_id": "m2", "map_index": 0, "outcome_ordinal": 3},
    {"match_id": "m2", "map_index": 1, "outcome_ordinal": 0},
    {"match_id": "m2", "map_index": 2, "outcome_ordinal": 1},
    {"match_id": "m3", "map_index": 0, "outcome_ordinal": 2},
    {"match_id": "m3", "map_index": 1, "outcome_ordinal": 3},
    {"match_id": "m3", "map_index": 2, "outcome_ordinal": 0},
    {"match_id": "m3", "map_index": 3, "outcome_ordinal": 1},
    {"match_id": "m3", "map_index": 4, "outcome_ordinal": 2},
    {"match_id": "m4", "map_index": 0, "outcome_ordinal": 3},
    {"match_id": "m4", "map_index": 1, "outcome_ordinal": 0},
    {"match_id": "m4", "map_index": 2, "outcome_ordinal": 1},
]


def _league_tables():
    """Build the synthetic matches/maps/labels/splits frames for the stub run.

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects built from the module-level row
        constants.

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
    labels_df = pd.DataFrame(
        _LABEL_ROWS,
        columns=["match_id", "map_index", "outcome_ordinal"],
    )
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, labels_df, splits_df


class _StubOrdinalModel:
    """Minimal stand-in for an OrdinalLogitModel in the patched end-to-end test.

    Carries a call-order ``index`` (assigned by the stub ``fit`` counter,
    so the driver's nominal = index 0, per-map replicates = 1..n_map,
    series replicates = n_map+1..n_map+n_series — exactly the call order
    the driver makes, which the test relies on to hand-compute expected
    intervals) and the ``converged`` diagnostic the driver reads for the
    artifact's convergence counts.

    Attributes:
        index: The fit call order (0 for the nominal model).
        converged: Always ``True`` for the stub (every replicate
            "converges", so the config's convergence counts equal the
            replicate counts).
    """

    def __init__(self, index):
        self.index = index
        self.converged = True


def _stub_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df, model):
    """Deterministic stub Stage-2 map model, hand-computable per model index.

    Returns a 4-vector in OUTCOME_LABELS order that depends only on the
    map's first character (which category gets the tilt) and the model's
    call-order ``index`` (tilt size ``0.02 * index``): the nominal model
    (index 0) returns the uniform vector for every map, and each
    replicate shifts probability mass from category ``(j + 1) % 4`` into
    category ``j = ord(map_name[0]) % 4``. The test re-derives the same
    vectors independently, so the driver's per-map interval wiring is
    cross-checked against hand-computed numpy percentiles.

    Args:
        team1_id / team2_id / map_name / date / matches_df / maps_df:
            The model-interface arguments; only ``map_name`` is read.
        model: The ``_StubOrdinalModel`` whose ``index`` sizes the tilt.

    Returns:
        A tuple of four non-negative floats summing to 1.

    Raises:
        Nothing.
    """
    j = ord(map_name[0]) % 4
    vec = [0.25, 0.25, 0.25, 0.25]
    vec[j] += 0.02 * model.index
    vec[(j + 1) % 4] -= 0.02 * model.index
    return tuple(vec)


def _expected_map_interval(map_name, first_index, n_reps, ci_level):
    """Re-derive one map's expected per-category bands from the stub formula.

    Recomputes the ``n_reps`` replicate vectors the stub map model
    produces for models ``first_index .. first_index + n_reps - 1`` and
    returns the per-column ``[lo, hi]`` numpy percentiles at
    ``ci_level`` — the hand-computed expectation the driver's
    ``interval_low``/``interval_high`` entries are checked against.

    Args:
        map_name: The map whose first character fixes the tilted
            category.
        first_index: The first replicate model's call-order index.
        n_reps: The number of replicate models.
        ci_level: The interval level (drives the percentile points).

    Returns:
        A ``(lo, hi)`` tuple of 4-float numpy arrays, per category.

    Raises:
        Nothing.
    """
    rows = []
    for idx in range(first_index, first_index + n_reps):
        j = ord(map_name[0]) % 4
        vec = [0.25, 0.25, 0.25, 0.25]
        vec[j] += 0.02 * idx
        vec[(j + 1) % 4] -= 0.02 * idx
        rows.append(vec)
    matrix = np.asarray(rows)
    lo = np.percentile(matrix, (1 - ci_level) / 2.0 * 100.0, axis=0)
    hi = np.percentile(matrix, (1 + ci_level) / 2.0 * 100.0, axis=0)
    return lo, hi


def _expected_series_interval(best_of_int, first_index, n_reps, ci_level):
    """Re-derive one series' expected K-way bands from the stub formula.

    The stub series factory probes the map model on map ``"Haven"``
    (whose first character tilts category 0), producing tilt ``d =
    0.004 * model.index`` on the first/last scoreline categories; this
    helper recomputes the ``n_reps`` replicate K-vectors for models
    ``first_index .. first_index + n_reps - 1`` and returns the
    per-column ``[lo, hi]`` numpy percentiles at ``ci_level``.

    Args:
        best_of_int: The series' map count (K = ``best_of_int + 1``).
        first_index: The first replicate model's call-order index.
        n_reps: The number of replicate models.
        ci_level: The interval level.

    Returns:
        A ``(lo, hi)`` tuple of ``K``-float numpy arrays, per category.

    Raises:
        Nothing.
    """
    rows = []
    for idx in range(first_index, first_index + n_reps):
        d = 0.004 * idx
        vec = [1.0 / (best_of_int + 1)] * (best_of_int + 1)
        vec[0] += d
        vec[-1] -= d
        rows.append(vec)
    matrix = np.asarray(rows)
    lo = np.percentile(matrix, (1 - ci_level) / 2.0 * 100.0, axis=0)
    hi = np.percentile(matrix, (1 + ci_level) / 2.0 * 100.0, axis=0)
    return lo, hi


def _stub_team_map_win_rate(team_id, map_name, date, matches_df, maps_df, k):
    """Stub features.map_win_rate.team_map_win_rate with hand-computable games.

    Returns a ``SimpleNamespace`` carrying a deterministic ``games``
    count: 10 for the odd-suffix teams (A/C/E), 7 for the even teams on
    Haven, 4 for the even teams on any other map — so every queried
    map's ``n_games_backing`` is a hand-computable ``min`` over the two
    team counts. Also asserts the driver passed ``map_win_rate.DEFAULT_K``
    (the wiring check that the driver uses the shared default).

    Args:
        team_id: The queried team id.
        map_name: The queried map.
        date / matches_df / maps_df: Unused by the stub.
        k: The shrinkage strength; asserted equal to
            ``map_win_rate.DEFAULT_K``.

    Returns:
        A ``SimpleNamespace(games=int)`` carrying the deterministic
        count.

    Raises:
        AssertionError: If ``k`` is not the shared default.
    """
    assert k == map_win_rate.DEFAULT_K
    if team_id in ("A", "C", "E"):
        games = 10
    elif map_name == "Haven":
        games = 7
    else:
        games = 4
    return SimpleNamespace(games=games)


def _stub_make_series_model_fn(map_model_fn, predictor_fn_by_action, n_samples, rng, map_pool=None, call_counter=None):
    """Stub replacement for make_series_model_fn with model-routed variation.

    Asserts the driver wired the two pluggable callables and a fresh
    ``numpy.random.Generator`` through (the factory contract the real
    M31 factory also has — and the fixed-veto-rng mechanism: the driver
    must pass a freshly reconstructed rng, verified as a Generator
    here), probes the Stage-2 ``map_model_fn`` on map ``"Haven"`` to
    route the replicate model's index into the returned scoreline
    vectors (category-0 tilt ``(probe[0] - 0.25) * 0.2``, balanced off
    the last category so the vector stays a valid simplex), and returns
    a ``SeriesModelFn``-shaped closure returning ``best_of + 1``-length
    vectors.

    Args:
        map_model_fn: The Stage-2 map closure to probe.
        predictor_fn_by_action: The Stage-1 dict (asserted to carry
            exactly ``ban`` and ``pick``).
        n_samples: The ``--veto-n-samples`` value (asserted positive).
        rng: The freshly reconstructed veto rng (asserted a Generator).
        map_pool: Unused; accepted for signature parity.
        call_counter: An optional ``dict`` whose ``"series_factory"``
            key is incremented per invocation, so the test can assert
            the factory ran once per model (nominal + replicates).

    Returns:
        A stub ``SeriesModelFn`` closure.

    Raises:
        AssertionError: If any wiring assertion fails.
    """
    if call_counter is not None:
        call_counter["series_factory"] += 1
    assert callable(map_model_fn)
    assert set(predictor_fn_by_action) == {"ban", "pick"}
    assert isinstance(rng, np.random.Generator)
    assert n_samples > 0
    probe = tuple(map_model_fn("A", "B", "Haven", "2026-01-01T00:00:00", None, None))

    def stub_series_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
        n = int(best_of[2:])
        d = (probe[0] - 0.25) * 0.2
        vec = [1.0 / (n + 1)] * (n + 1)
        vec[0] += d
        vec[-1] -= d
        return tuple(vec)

    return stub_series_fn


@pytest.fixture
def stub_everything(monkeypatch):
    """Monkeypatch the driver's loaders, resamplers and model factories.

    Routes the input tables to the synthetic league, replaces the
    artifact loaders, the two design-matrix assemblers, ``ordinal_logit.fit``
    / ``make_model_fn``, ``team_map_win_rate`` and
    ``make_series_model_fn`` with deterministic stubs, and installs call
    counters (exposed on the returned dict) so the end-to-end test can
    verify the driver drew exactly ``n_bootstrap_map`` +
    ``n_bootstrap_series`` resamples, fit exactly one nominal +
    ``n_bootstrap_map`` + ``n_bootstrap_series`` models, and constructed
    exactly one M31 series factory per series model. The real pure
    harness functions (build_held_out_maps / build_held_out_series /
    score_held_out_series) and the real interval helpers run untouched.
    All patches are reverted by monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        A dict of the call counters: ``design``, ``bootstrap``, ``fit``,
        ``series_factory``.

    Raises:
        Nothing.
    """
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    counters = {"design": 0, "bootstrap": 0, "fit": 0, "series_factory": 0}
    fit_counter = itertools.count()

    X_stub = np.zeros((4, 11))
    y_stub = np.zeros(4, dtype=int)

    def stub_design(matches_df, maps_df, labels_df, splits_df, pms_df, split="train"):
        counters["design"] += 1
        assert split == "train"
        return X_stub, y_stub

    def stub_bootstrap(matches_df, maps_df, labels_df, splits_df, pms_df, rng):
        counters["bootstrap"] += 1
        assert isinstance(rng, np.random.Generator)
        rng.integers(0, 1000)  # consume the rng like the real resampler
        return X_stub, y_stub

    def stub_fit(X, y):
        counters["fit"] += 1
        return _StubOrdinalModel(next(fit_counter))

    def stub_make_model_fn(model, player_map_stats_df):
        def stub_map_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
            return _stub_map_model_fn(
                team1_id, team2_id, map_name, date, matches_df, maps_df, model
            )

        return stub_map_fn

    def stub_series_factory(map_model_fn, predictor_fn_by_action, n_samples, rng, map_pool=None):
        return _stub_make_series_model_fn(
            map_model_fn,
            predictor_fn_by_action,
            n_samples,
            rng,
            map_pool=map_pool,
            call_counter=counters,
        )

    monkeypatch.setattr(ebi.evaluate, "load_matches_table",
                        lambda output_dir, version: matches_df)
    monkeypatch.setattr(ebi.evaluate, "load_maps_table",
                        lambda output_dir, version: maps_df)
    monkeypatch.setattr(ebi.evaluate, "load_labels_table",
                        lambda output_dir, version: labels_df)
    monkeypatch.setattr(ebi.evaluate, "load_splits_table",
                        lambda output_dir, version: splits_df)
    monkeypatch.setattr(ebi.evaluate, "load_player_map_stats_table",
                        lambda output_dir, version: player_map_stats_df)
    monkeypatch.setattr(ebi, "_load_fitted_models",
                        lambda output_dir, version: (None, None, None))
    monkeypatch.setattr(ebi.training_data, "assemble_design_matrix", stub_design)
    monkeypatch.setattr(ebi.training_data, "assemble_bootstrap_design_matrix", stub_bootstrap)
    monkeypatch.setattr(ebi.ordinal_logit, "fit", stub_fit)
    monkeypatch.setattr(ebi.ordinal_logit, "make_model_fn", stub_make_model_fn)
    monkeypatch.setattr(ebi.map_win_rate, "team_map_win_rate", _stub_team_map_win_rate)
    monkeypatch.setattr(ebi.veto_marginalized_series, "make_series_model_fn",
                        stub_series_factory)
    return counters


# --------------------------------------------------------------------------
# plan#7a: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the documented defaults for all nine knobs.
    args = ebi.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_bootstrap_map == DEFAULT_N_BOOTSTRAP_MAP
    assert args.n_bootstrap_series == DEFAULT_N_BOOTSTRAP_SERIES
    assert args.bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
    assert args.veto_seed == DEFAULT_VETO_SEED
    assert args.veto_n_samples == DEFAULT_VETO_N_SAMPLES
    assert args.ci_level == DEFAULT_CI_LEVEL


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int counts/seeds and
    # non-float --ci-level are rejected by argparse (SystemExit).
    args = ebi.parse_args(
        ["--version", "v2", "--output-dir", "out",
         "--n-bootstrap-map", "3", "--n-bootstrap-series", "2",
         "--bootstrap-seed", "7", "--veto-seed", "11",
         "--veto-n-samples", "2", "--ci-level", "0.8"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_bootstrap_map == 3
    assert args.n_bootstrap_series == 2
    assert args.bootstrap_seed == 7
    assert args.veto_seed == 11
    assert args.veto_n_samples == 2
    assert args.ci_level == 0.8
    with pytest.raises(SystemExit):
        ebi.parse_args(["--n-bootstrap-map", "many"])
    with pytest.raises(SystemExit):
        ebi.parse_args(["--ci-level", "wide"])


# --------------------------------------------------------------------------
# plan#7b: synthetic end-to-end main() with hand-computable stubs
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_report(tmp_path, stub_everything):
    # A full main() run against the synthetic league with every loader/
    # factory stubbed: the artifact is written with the interval
    # definition caveat, the config block (replicate counts, convergence
    # counts, seeds, knobs), 11 per-map entries and 3 per-series entries
    # (Bo3 + Bo5); every per-map interval matches the hand-computed
    # numpy-percentile re-derivation of the stub replicate vectors, the
    # n_games_backing values match the stubbed min-games rule, the
    # series intervals match the stub series factory's hand-computed
    # bands, all bands are ordered lo <= hi and within [0, 1], and the
    # per-replicate call counts prove the driver drew exactly the
    # requested number of resamples / fits / M31 factories.
    counters = stub_everything
    rc = ebi.main(
        ["--output-dir", str(tmp_path),
         "--n-bootstrap-map", "3", "--n-bootstrap-series", "2",
         "--bootstrap-seed", "7", "--veto-seed", "11",
         "--veto-n-samples", "2", "--ci-level", "0.9"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "bootstrap_intervals_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {"interval_definition", "config", "per_map", "series"}
    assert "joint simplex" in report["interval_definition"]
    config = report["config"]
    assert config == {
        "n_bootstrap_map": 3,
        "n_bootstrap_series": 2,
        "n_bootstrap_map_converged": 3,
        "n_bootstrap_series_converged": 2,
        "bootstrap_seed": 7,
        "veto_seed": 11,
        "veto_n_samples": 2,
        "ci_level": 0.9,
    }

    # Wiring counts: 1 nominal design matrix, 3 + 2 bootstrap resamples,
    # 1 + 3 + 2 fits, and 1 + 2 M31 series factories (nominal + the two
    # series replicates).
    assert counters["design"] == 1
    assert counters["bootstrap"] == 3 + 2
    assert counters["fit"] == 1 + 3 + 2
    assert counters["series_factory"] == 1 + 2

    # Per-map section: 11 held-out test maps (m1/m2/m3). Nominal is the
    # uniform vector for every map (index-0 stub); the interval per
    # category comes from replicate models 1..3.
    per_map = report["per_map"]
    assert len(per_map) == 11
    for entry in per_map:
        assert set(entry) == {
            "match_id", "map_index", "date", "team1_id", "team2_id",
            "map_name", "outcome_ordinal", "nominal", "interval_low",
            "interval_high", "n_games_backing",
        }
        assert entry["nominal"] == [0.25, 0.25, 0.25, 0.25]
        lo, hi = _expected_map_interval(
            entry["map_name"], first_index=1, n_reps=3, ci_level=0.9
        )
        assert entry["interval_low"] == pytest.approx(lo.tolist())
        assert entry["interval_high"] == pytest.approx(hi.tolist())
        for low, high in zip(entry["interval_low"], entry["interval_high"]):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0

    # Hand-check one specific per-map entry: m1 map 1 is Haven, teams
    # A/B -> min(10, 7) = 7 games of backing.
    haven_entry = next(
        e for e in per_map
        if e["match_id"] == "m1" and e["map_name"] == "Haven"
    )
    assert haven_entry["n_games_backing"] == 7
    bind_entry = next(
        e for e in per_map
        if e["match_id"] == "m3" and e["map_name"] == "Bind"
    )
    # m3 teams C/D on Bind -> min(10, 4) = 4.
    assert bind_entry["n_games_backing"] == 4

    # Per-series section: m1/m2 (Bo3, K=4) and m3 (Bo5, K=6). The
    # intervals come from series replicate models 4..5.
    series = report["series"]
    assert len(series) == 3
    by_match = {entry["match_id"]: entry for entry in series}
    for entry in by_match.values():
        best_of_int = entry["best_of_int"]
        lo, hi = _expected_series_interval(
            best_of_int, first_index=4, n_reps=2, ci_level=0.9
        )
        assert entry["interval_low"] == pytest.approx(lo.tolist())
        assert entry["interval_high"] == pytest.approx(hi.tolist())
        for low, high in zip(entry["interval_low"], entry["interval_high"]):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
        assert len(entry["n_games_backing"]) == len(entry["played_maps"])
    # m1's actual played maps in play order, with A/B backing per map.
    m1_series = by_match["m1"]
    assert m1_series["played_maps"] == ["Bind", "Haven", "Split"]
    assert m1_series["n_games_backing"] == [4, 7, 4]
    assert m1_series["best_of_int"] == 3
    m3_series = by_match["m3"]
    assert m3_series["best_of_int"] == 5
    assert len(m3_series["nominal"]) == 6


# --------------------------------------------------------------------------
# plan#7c: missing prerequisite artifact raises FileNotFoundError
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(tmp_path):
    # No fitted artifacts exist under the empty tmp output dir: the
    # first artifact load must raise FileNotFoundError unchanged (the
    # "run the training driver first" signal), never a silent fallback
    # or a wrapped exception.
    with pytest.raises(FileNotFoundError):
        ebi.main(["--output-dir", str(tmp_path)])


def test_main_rejects_invalid_flag_values(tmp_path):
    # Non-positive replicate counts and an out-of-(0, 1) ci_level are
    # hard errors before any work starts (even with everything stubbed
    # away the validation must fire first).
    for bad in ("--n-bootstrap-map", "--n-bootstrap-series"):
        with pytest.raises(ValueError, match="positive"):
            ebi.main(["--output-dir", str(tmp_path), bad, "0"])
    with pytest.raises(ValueError, match="--ci-level"):
        ebi.main(["--output-dir", str(tmp_path), "--ci-level", "1.5"])


# --------------------------------------------------------------------------
# plan#7: real-v1 integration smoke test (skip-guarded)
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
def test_real_v1_smoke_finite_well_ordered_intervals():
    # A tiny real-v1 run (1 map replicate, 1 series replicate, 1 veto
    # sample) against the real fitted models: the artifact must be
    # written, every per-map and per-series band must be ordered
    # lo <= hi with simplex-adjacent [0, 1] bounds (a finite,
    # well-ordered interval is guaranteed by construction only in that
    # weak sense — the plan explicitly forbids asserting the true
    # outcome lies inside the band), every value must be plain
    # json.dumps-serializable, and the config block must carry the
    # convergence diagnostics.
    rc = ebi.main(
        ["--n-bootstrap-map", "1", "--n-bootstrap-series", "1",
         "--veto-n-samples", "1", "--bootstrap-seed", "7",
         "--veto-seed", "11", "--ci-level", "0.9"]
    )
    assert rc == 0

    artifact_path = Path("data/v1/bootstrap_intervals_report.json")
    assert artifact_path.exists()
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    config = report["config"]
    assert config["n_bootstrap_map"] == 1
    assert config["n_bootstrap_series"] == 1
    assert config["veto_n_samples"] == 1
    assert config["n_bootstrap_map_converged"] in (0, 1)
    assert config["n_bootstrap_series_converged"] in (0, 1)

    for entry in report["per_map"]:
        assert len(entry["interval_low"]) == 4
        assert len(entry["interval_high"]) == 4
        for low, high in zip(entry["interval_low"], entry["interval_high"]):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
        assert entry["n_games_backing"] >= 0

    for entry in report["series"]:
        k = entry["best_of_int"] + 1
        assert len(entry["interval_low"]) == k
        assert len(entry["interval_high"]) == k
        for low, high in zip(entry["interval_low"], entry["interval_high"]):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
        assert len(entry["n_games_backing"]) == len(entry["played_maps"])
        assert all(g >= 0 for g in entry["n_games_backing"])

    # The full report round-trips through json.dumps (every value is a
    # plain str/int/float/list/dict).
    json.dumps(report)

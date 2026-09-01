"""Tests for the M33b series-evaluation CLI driver (drivers/evaluate_series.py).

Covers the CLI/IO glue only — the pure two-arm report logic is already
tested in tests/test_series_evaluation.py (M33a) and the M31 pipeline
in tests/test_veto_marginalized_series.py. The three tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the table/artifact loaders and the M31 factory
monkeypatched to fast deterministic stubs, whose held-out set contains
both Bo3 and Bo5 test-split matches so the Bo5-present branch of the
two-arm report is exercised end to end (plan assumption 6 — real v1
data has no Bo5 test-split rows, so the branch must be proven on a
hand-built table); and the missing-prerequisite-artifact
``FileNotFoundError`` contract (plan assumption 4). No real fitted
artifacts are required by any of these tests.
"""

import json

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate_series

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_SAMPLES = evaluate_series.DEFAULT_N_SAMPLES
DEFAULT_SEED = evaluate_series.DEFAULT_SEED

# A tiny hand-built league: m1/m2 Bo3 test matches, m3 a Bo5 test
# match (so the Bo5-present branch is exercised), m4 a Bo3 train match
# (held out of the evaluation). Team ids/dates are arbitrary — every
# stub model ignores the tables.
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

# Three maps per Bo3 match, five per Bo5 match; every scoreline is
# decisive (never tied, never null), so build_held_out_series can
# derive each observed series scoreline.
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


def _league_tables():
    """Build the synthetic matches/maps/splits frames for the stub run.

    Returns:
        A ``(matches_df, maps_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects built from :data:`_MATCH_ROWS` /
        :data:`_MAP_ROWS` / :data:`_SPLIT_ROWS`.

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
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, splits_df


def _stub_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Stub Stage-2 four-way map model returning the uniform 4-vector.

    Never actually invoked by the patched end-to-end test (the M31
    factory itself is monkeypatched to a deterministic stub), but must
    satisfy the ``MapOutcomeModelFn`` shape for the wiring to be
    exercised.

    Args:
        team1_id / team2_id / map_name / date / matches_df / maps_df:
            The model-interface arguments; deliberately unused.

    Returns:
        The uniform 4-vector ``(0.25, 0.25, 0.25, 0.25)``.

    Raises:
        Nothing.
    """
    return (0.25, 0.25, 0.25, 0.25)


def _stub_veto_predictor_fn(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """Stub Stage-1 veto-step predictor returning a uniform distribution.

    Never actually invoked by the patched end-to-end test (see
    :func:`_stub_map_model_fn`); satisfies the ``VetoStepPredictorFn``
    shape so the wiring is exercised.

    Args:
        acting_team_id / action / remaining_maps / date / matches_df /
            maps_df: The predictor-interface arguments; deliberately
            unused.

    Returns:
        A ``list`` of ``len(remaining_maps)`` equal ``float``
        probabilities.

    Raises:
        Nothing.
    """
    n = len(remaining_maps)
    return [1.0 / n] * n


def _stub_series_model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df):
    """Deterministic stub SeriesModelFn returning the uniform scoreline.

    Stands in for the real M31 pipeline in the patched end-to-end test:
    returns the ``best_of + 1``-length uniform vector regardless of the
    arguments, so every held-out row scores a strictly-positive,
    valid-simplex distribution (log loss is well defined) and the
    report values are hand-checkable in shape even though they are not
    the real fitted-model numbers.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The SeriesModelFn-interface arguments; deliberately unused
            except ``best_of``, which fixes the returned length.

    Returns:
        A tuple of ``best_of + 1`` equal floats summing to 1.

    Raises:
        ValueError: If ``best_of`` is not a ``"Bo<N>"`` string with a
            positive odd numeric suffix.
    """
    n = int(best_of[2:])
    return tuple(1.0 / (n + 1) for _ in range(n + 1))


def _stub_make_series_model_fn(map_model_fn, predictor_fn_by_action, n_samples, rng, map_pool=None):
    """Stub replacement for make_series_model_fn returning the uniform stub.

    Asserts the driver wired the two pluggable callables and the
    requested ``n_samples`` / ``rng`` through correctly (the factory
    contract the real M31 factory also has), then returns
    :func:`_stub_series_model_fn`.

    Args:
        map_model_fn: The Stage-2 four-way map model the driver built
            (asserted callable).
        predictor_fn_by_action: The Stage-1 predictor dict the driver
            built (asserted to carry exactly ``ban`` and ``pick``).
        n_samples: The ``--n-samples`` value (asserted passed through).
        rng: The seed-derived ``numpy.random.Generator`` (asserted a
            Generator).
        map_pool: Unused; accepted for signature parity.

    Returns:
        :func:`_stub_series_model_fn`.

    Raises:
        AssertionError: If any wiring assertion fails (wrong callable
            types, missing predictor key, wrong ``n_samples``, or a
            non-Generator ``rng``).
    """
    assert callable(map_model_fn)
    assert set(predictor_fn_by_action) == {"ban", "pick"}
    assert n_samples == 4
    assert isinstance(rng, np.random.Generator)
    return _stub_series_model_fn


@pytest.fixture
def stub_everything(monkeypatch):
    """Monkeypatch the driver's table loaders and model factories.

    Routes the four input tables to the synthetic league and replaces
    every fitted-artifact / model-factory call with a deterministic
    stub, so :func:`evaluate_series.main` runs the real CLI/IO and the
    real pure evaluation functions without any real fitted artifacts
    and without any real M31 sampling. All patches are reverted by
    monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        Nothing (the patches are installed for the test's duration).

    Raises:
        Nothing.
    """
    matches_df, maps_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    monkeypatch.setattr(
        evaluate_series.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        evaluate_series.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        evaluate_series.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        evaluate_series.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    monkeypatch.setattr(
        evaluate_series, "_load_fitted_models",
        lambda output_dir, version: (None, None, None),
    )
    monkeypatch.setattr(
        evaluate_series.ordinal_logit, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_series.ordinal_logit, "make_model_fn",
        lambda model, player_map_stats_df: _stub_map_model_fn,
    )
    monkeypatch.setattr(
        evaluate_series.conditional_logit_ban, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_series.conditional_logit_ban, "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )
    monkeypatch.setattr(
        evaluate_series.conditional_logit_pick, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_series.conditional_logit_pick, "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )
    monkeypatch.setattr(
        evaluate_series.veto_marginalized_series, "make_series_model_fn",
        _stub_make_series_model_fn,
    )


# --------------------------------------------------------------------------
# plan#4a: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the four documented defaults (version v1, output_dir
    # data, and the measured-wall-clock n_samples / repo-seed values
    # referenced through the module constants).
    args = evaluate_series.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int --n-samples/--seed are
    # rejected by argparse (SystemExit).
    args = evaluate_series.parse_args(
        ["--version", "v2", "--output-dir", "out", "--n-samples", "3",
         "--seed", "42"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_samples == 3
    assert args.seed == 42
    with pytest.raises(SystemExit):
        evaluate_series.parse_args(["--n-samples", "many"])


# --------------------------------------------------------------------------
# plan#4b: synthetic end-to-end main() (Bo3 + Bo5 present)
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_two_arm_report(tmp_path, stub_everything):
    # A full main() run against the synthetic league with every loader/
    # factory stubbed: the artifact is written with the two arm blocks
    # plus the deltas block plus the n_samples/seed provenance keys,
    # both Bo3 and Bo5 groups appear in both arms and in the delta
    # block (exercising the Bo5-present branch of the two-arm report on
    # a hand-built table, per plan assumption 6), every headline value
    # is finite and in range, each delta equals arm-minus-baseline, and
    # the file ends with the repo's trailing-newline convention.
    rc = evaluate_series.main(
        ["--output-dir", str(tmp_path), "--n-samples", "4", "--seed", "7"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "series_evaluation_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "flat_series_baseline",
        "veto_marginalized_series",
        "deltas_vs_flat_series_baseline",
        "n_samples",
        "seed",
    }
    assert report["n_samples"] == 4
    assert report["seed"] == 7

    for arm in ("flat_series_baseline", "veto_marginalized_series"):
        arm_report = report[arm]
        # m1/m2 (Bo3) + m3 (Bo5) are held out; m4 is train.
        assert arm_report["n_eval_total"] == 3
        assert arm_report["Bo3"]["n_eval"] == 2
        assert arm_report["Bo5"]["n_eval"] == 1
        for group in ("Bo3", "Bo5"):
            block = arm_report[group]
            assert 0.0 <= block["mean_rps"] <= 3.0 if group == "Bo3" else (
                0.0 <= block["mean_rps"] <= 5.0
            )
            assert block["mean_log_loss"] >= 0.0
            assert 0.0 <= block["marginal_binary_accuracy"] <= 1.0

    # The Bo5-present branch: both deltas blocks carry Bo5 entries, and
    # every delta is exactly arm-minus-baseline.
    deltas = report["deltas_vs_flat_series_baseline"]["veto_marginalized_series"]
    assert set(deltas) == {"Bo3", "Bo5"}
    for group in ("Bo3", "Bo5"):
        delta = deltas[group]
        assert delta["mean_rps_delta"] == pytest.approx(
            report["veto_marginalized_series"][group]["mean_rps"]
            - report["flat_series_baseline"][group]["mean_rps"]
        )
        assert delta["mean_log_loss_delta"] == pytest.approx(
            report["veto_marginalized_series"][group]["mean_log_loss"]
            - report["flat_series_baseline"][group]["mean_log_loss"]
        )
        assert delta["marginal_binary_accuracy_delta"] == pytest.approx(
            report["veto_marginalized_series"][group]["marginal_binary_accuracy"]
            - report["flat_series_baseline"][group]["marginal_binary_accuracy"]
        )


# --------------------------------------------------------------------------
# plan#4c: missing prerequisite artifact raises FileNotFoundError
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(tmp_path):
    # No fitted artifacts exist under the empty tmp output dir: the
    # first artifact load must raise FileNotFoundError unchanged (the
    # "run the training driver first" signal), never a silent fallback
    # or a wrapped exception.
    with pytest.raises(FileNotFoundError):
        evaluate_series.main(["--output-dir", str(tmp_path)])

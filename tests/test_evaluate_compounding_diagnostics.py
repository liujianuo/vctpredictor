"""Tests for the M35 compounding-diagnostics CLI driver
(drivers/evaluate_compounding_diagnostics.py).

Covers the CLI/IO glue only — the pure report logic is already tested
in tests/test_compounding_diagnostics.py. The three tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the table/artifact loaders and the model factories
monkeypatched to fast deterministic stubs, exercising both diagnostics
against a hand-built 3-match Bo3 test split (uniform stub models make
every headline number hand-checkable: predicted sweep prob 0.5 vs
observed 1/3, and a map1-predicts-map2 permutation p-value of exactly
1.0); and the missing-prerequisite-artifact ``FileNotFoundError``
contract. No real fitted artifacts are required by any of these tests.
"""

import json

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate_compounding_diagnostics

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_SAMPLES = evaluate_compounding_diagnostics.DEFAULT_N_SAMPLES
DEFAULT_SEED = evaluate_compounding_diagnostics.DEFAULT_SEED
DEFAULT_N_PERMUTATIONS = (
    evaluate_compounding_diagnostics.DEFAULT_N_PERMUTATIONS
)
DEFAULT_PERMUTATION_SEED = (
    evaluate_compounding_diagnostics.DEFAULT_PERMUTATION_SEED
)

# A tiny hand-built league: m1/m2/m3 Bo3 test matches, m4 a Bo3 train
# match (held out of the evaluation). Dates/team ids are arbitrary —
# every stub model ignores the tables. The scorelines are chosen so
# the observed series scorelines are m1 (2,1) -> outcome_index 1,
# m2 (2,0) -> outcome_index 0 (a sweep), m3 (1,2) -> outcome_index 2.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
     "team2_id": "F", "best_of": "Bo3", "status": "completed"},
]

# Three maps per Bo3 match; every scoreline is decisive (never tied,
# never null), so build_held_out_series can derive each observed series
# scoreline and the harness can join maps to labels.
_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m1", "map_index": 2, "map_name": "Split",
     "team1_score": 13, "team2_score": 11, "winner": "A"},
    {"match_id": "m2", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 5, "winner": "A"},
    {"match_id": "m2", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m3", "map_index": 0, "map_name": "Bind",
     "team1_score": 5, "team2_score": 13, "winner": "B"},
    {"match_id": "m3", "map_index": 1, "map_name": "Haven",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m3", "map_index": 2, "map_name": "Split",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m4", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 6, "winner": "E"},
    {"match_id": "m4", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "E"},
    {"match_id": "m4", "map_index": 2, "map_name": "Split",
     "team1_score": 5, "team2_score": 13, "winner": "F"},
]

# Labels consistent with the scores above (the harness joins on
# (match_id, map_index) for the true outcome_ordinal). Map-1 outcomes:
# m1 A (ord 0), m2 A (ord 0), m3 B (ord 3) — so the
# map1-predicts-map2 diagnostic gets 3 eligible matches with 2 A-won
# and 1 B-won map-1s (both subgroups non-empty).
_LABEL_ROWS = [
    {"match_id": "m1", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m1", "map_index": 1, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m1", "map_index": 2, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m2", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m2", "map_index": 1, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m3", "map_index": 0, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m3", "map_index": 1, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m3", "map_index": 2, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m4", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m4", "map_index": 1, "outcome_label": "A-OT",
     "outcome_ordinal": 1},
    {"match_id": "m4", "map_index": 2, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
]

_SPLIT_ROWS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "test"},
    {"match_id": "m4", "split": "train"},
]


def _league_tables():
    """Build the synthetic matches/maps/labels/splits frames for the stub run.

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects built from :data:`_MATCH_ROWS` /
        :data:`_MAP_ROWS` / :data:`_LABEL_ROWS` / :data:`_SPLIT_ROWS`.

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
        columns=["match_id", "map_index", "outcome_label", "outcome_ordinal"],
    )
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, labels_df, splits_df


def _stub_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Stub Stage-2 four-way map model returning the uniform 4-vector.

    Returns the same vector for every map, so every map-2 prediction
    has ``p_a_regulation + p_a_ot = 0.5`` and the diagnostic-2 numbers
    are hand-checkable.

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

    Never actually invoked by the patched end-to-end test (the M31
    factory itself is monkeypatched to a deterministic stub); satisfies
    the ``VetoStepPredictorFn`` shape so the wiring is exercised.

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
    arguments, so every held-out series' sweep probability is
    hand-checkable (Bo3: 0.25 + 0.25 = 0.5 per row).

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

    Routes the five input tables to the synthetic league and replaces
    every fitted-artifact / model-factory call with a deterministic
    stub, so :func:`evaluate_compounding_diagnostics.main` runs the
    real CLI/IO and the real pure evaluation functions without any real
    fitted artifacts and without any real M31 sampling. All patches are
    reverted by monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        Nothing (the patches are installed for the test's duration).

    Raises:
        Nothing.
    """
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.evaluate, "load_labels_table",
        lambda output_dir, version: labels_df,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics, "_load_fitted_models",
        lambda output_dir, version: (None, None, None),
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.ordinal_logit, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.ordinal_logit, "make_model_fn",
        lambda model, player_map_stats_df: _stub_map_model_fn,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.conditional_logit_ban, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.conditional_logit_ban,
        "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.conditional_logit_pick, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.conditional_logit_pick,
        "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )
    monkeypatch.setattr(
        evaluate_compounding_diagnostics.veto_marginalized_series,
        "make_series_model_fn",
        _stub_make_series_model_fn,
    )


# --------------------------------------------------------------------------
# plan#4b: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the six documented defaults (version v1, output_dir
    # data, the measured-wall-clock M31 n_samples / repo-seed values,
    # and the sub-second permutation-test n_permutations /
    # permutation-seed values, all referenced through the module
    # constants).
    args = evaluate_compounding_diagnostics.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED
    assert args.n_permutations == DEFAULT_N_PERMUTATIONS
    assert args.permutation_seed == DEFAULT_PERMUTATION_SEED


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int --n-samples/--seed/
    # --n-permutations/--permutation-seed are rejected by argparse
    # (SystemExit).
    args = evaluate_compounding_diagnostics.parse_args(
        ["--version", "v2", "--output-dir", "out", "--n-samples", "3",
         "--seed", "42", "--n-permutations", "500", "--permutation-seed", "9"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_samples == 3
    assert args.seed == 42
    assert args.n_permutations == 500
    assert args.permutation_seed == 9
    with pytest.raises(SystemExit):
        evaluate_compounding_diagnostics.parse_args(
            ["--n-permutations", "many"]
        )


# --------------------------------------------------------------------------
# plan#4c: synthetic end-to-end main() (both diagnostics)
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_report(tmp_path, stub_everything):
    # A full main() run against the synthetic league with every loader/
    # factory stubbed: the artifact is written with the two diagnostics'
    # report blocks plus the four provenance keys, the sweep-rate block
    # has hand-checkable numbers (uniform stub series model ->
    # predicted sweep prob 0.25+0.25 = 0.5 per row, observed sweep rate
    # 1/3 since only m2's outcome_index is 0), and the map1-predicts-
    # map2 block has 3 eligible matches with the uniform stub map model
    # giving observed_diff = 0.5 and a permutation p-value of exactly
    # 1.0 (every relabeling of the 2-A/1-B map-1 labels yields
    # |diff| >= 0.5).
    rc = evaluate_compounding_diagnostics.main(
        ["--output-dir", str(tmp_path), "--n-samples", "4", "--seed", "7",
         "--n-permutations", "5000", "--permutation-seed", "11"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "compounding_diagnostics_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "sweep_rate",
        "map1_predicts_map2",
        "n_samples",
        "seed",
        "n_permutations",
        "permutation_seed",
    }
    assert report["n_samples"] == 4
    assert report["seed"] == 7
    assert report["n_permutations"] == 5000
    assert report["permutation_seed"] == 11

    # Sweep-rate diagnostic: m1/m2/m3 held out (m4 train); uniform
    # stub series model -> per-row Bo3 sweep prob 0.25+0.25 = 0.5;
    # only m2 (outcome_index 0) is an observed sweep.
    sweep = report["sweep_rate"]
    assert sweep["n_eval_total"] == 3
    bo3 = sweep["Bo3"]
    assert bo3["n_eval"] == 3
    assert bo3["predicted_mean_sweep_prob"] == pytest.approx(0.5)
    assert bo3["observed_sweep_rate"] == pytest.approx(1.0 / 3.0)
    assert bo3["sweep_calibration_gap"] == pytest.approx(1.0 / 6.0)
    assert bo3["predicted_mean_a_sweep_prob"] == pytest.approx(0.25)
    assert bo3["observed_a_sweep_rate"] == pytest.approx(1.0 / 3.0)
    assert bo3["a_sweep_calibration_gap"] == pytest.approx(-1.0 / 12.0)
    assert bo3["predicted_mean_b_sweep_prob"] == pytest.approx(0.25)
    assert bo3["observed_b_sweep_rate"] == pytest.approx(0.0)
    assert bo3["b_sweep_calibration_gap"] == pytest.approx(0.25)

    # Map1-predicts-map2 diagnostic: 3 eligible matches (m1/m2/m3),
    # map-1 outcomes A/A/B; uniform stub map model -> every map-2
    # prediction p_a_reg + p_a_ot = 0.5; map-2 actuals B/A/B (m1 map1
    # ord 3, m2 map1 ord 0, m3 map1 ord 3) -> residuals -0.5, +0.5,
    # -0.5; mean given A (m1, m2) = 0.0, given B (m3) = -0.5,
    # observed_diff = +0.5. Every relabeling of the 2-A/1-B labels
    # produces |diff| >= 0.5, so the sampled p-value is exactly 1.0
    # at any seed.
    mp = report["map1_predicts_map2"]
    assert mp["n_eligible_matches"] == 3
    assert mp["n_map1_a_won"] == 2
    assert mp["n_map1_b_won"] == 1
    assert mp["mean_residual_given_map1_a_won"] == pytest.approx(0.0)
    assert mp["mean_residual_given_map1_b_won"] == pytest.approx(-0.5)
    assert mp["observed_diff"] == pytest.approx(0.5)
    assert mp["n_permutations"] == 5000
    assert mp["p_value_empirical"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#4d: missing prerequisite artifact raises FileNotFoundError
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(tmp_path):
    # No fitted artifacts exist under the empty tmp output dir: the
    # first artifact load must raise FileNotFoundError unchanged (the
    # "run the training driver first" signal), never a silent fallback
    # or a wrapped exception.
    with pytest.raises(FileNotFoundError):
        evaluate_compounding_diagnostics.main(
            ["--output-dir", str(tmp_path)]
        )

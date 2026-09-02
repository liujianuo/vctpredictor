"""Tests for the M34 stage-isolation CLI driver
(drivers/evaluate_stage_isolation.py).

Covers the CLI/IO glue only — the pure two-arm report logic is already
tested in tests/test_stage_isolation.py. The three tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the table/artifact loaders and the model factories
monkeypatched to fast deterministic stubs, whose held-out set contains
a swept Bo3 (2 played maps) beside a full Bo3 (3 played maps) so the
truncation-to-``n_played`` path of the Arm-B sampler is exercised end
to end; and the missing-prerequisite-artifact ``FileNotFoundError``
contract. No real fitted artifacts are required by any of these tests.
"""

import json

import pandas as pd
import pytest

from drivers import evaluate_stage_isolation

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_SAMPLES = evaluate_stage_isolation.DEFAULT_N_SAMPLES
DEFAULT_SEED = evaluate_stage_isolation.DEFAULT_SEED

# A tiny hand-built league: m1/m2 Bo3 test matches (both swept — only
# 2 of 3 maps played, exercising the truncation-to-n_played path), m3
# a full Bo3 test match (3 maps), m4 a Bo3 train match (held out of
# the evaluation). Dates fall inside config.json's 2026-08-17+ era so
# the real sampler's map-pool resolution succeeds. Team ids are
# arbitrary — every stub model ignores the tables.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-08-20T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-08-21T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-08-22T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m4", "date": "2026-08-23T00:00:00", "team1_id": "E",
     "team2_id": "F", "best_of": "Bo3", "status": "completed"},
]

# Two maps for the swept Bo3 matches, three for the full one; every
# scoreline is decisive (never tied, never null).
_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 0, "map_name": "Split",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m2", "map_index": 1, "map_name": "Ascent",
     "team1_score": 13, "team2_score": 11, "winner": "A"},
    {"match_id": "m3", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 5, "winner": "C"},
    {"match_id": "m3", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "C"},
    {"match_id": "m3", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "D"},
    {"match_id": "m4", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 6, "winner": "E"},
    {"match_id": "m4", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "E"},
]

_LABEL_ROWS = [
    {"match_id": "m1", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m1", "map_index": 1, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m2", "map_index": 0, "outcome_label": "A-OT",
     "outcome_ordinal": 1},
    {"match_id": "m2", "map_index": 1, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m3", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m3", "map_index": 1, "outcome_label": "A-OT",
     "outcome_ordinal": 1},
    {"match_id": "m3", "map_index": 2, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m4", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m4", "map_index": 1, "outcome_label": "A-OT",
     "outcome_ordinal": 1},
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

    Returns the same vector for every map, so Arm A and Arm B are
    numerically identical and every gap is exactly zero — a clean,
    hand-checkable shape assertion for the driver's report.

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
    """Stub Stage-1 veto-step predictor returning a one-hot distribution.

    Always picks the first sorted remaining map, so every sampled walk
    is identical and the Arm-B sampler is fully deterministic (and
    fast) in the end-to-end test.

    Args:
        acting_team_id / action / remaining_maps / date / matches_df /
            maps_df: The predictor-interface arguments; only
            ``remaining_maps`` is read.

    Returns:
        A ``list`` of ``len(remaining_maps)`` probabilities with
        ``1.0`` on the first entry.

    Raises:
        Nothing.
    """
    probs = [0.0] * len(remaining_maps)
    probs[0] = 1.0
    return probs


@pytest.fixture
def stub_everything(monkeypatch):
    """Monkeypatch the driver's table loaders and model factories.

    Routes the four input tables to the synthetic league and replaces
    every fitted-artifact / model-factory call with a deterministic
    stub, so :func:`evaluate_stage_isolation.main` runs the real
    CLI/IO and the real pure evaluation functions without any real
    fitted artifacts and without any real fitted-model scoring. All
    patches are reverted by monkeypatch at test teardown.

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
        evaluate_stage_isolation.evaluate, "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.evaluate, "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.evaluate, "load_labels_table",
        lambda output_dir, version: labels_df,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.evaluate, "load_splits_table",
        lambda output_dir, version: splits_df,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.evaluate, "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation, "_load_fitted_models",
        lambda output_dir, version: (None, None, None),
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.ordinal_logit, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.ordinal_logit, "make_model_fn",
        lambda model, player_map_stats_df: _stub_map_model_fn,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.conditional_logit_ban, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.conditional_logit_ban, "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.conditional_logit_pick, "from_dict",
        lambda d: None,
    )
    monkeypatch.setattr(
        evaluate_stage_isolation.conditional_logit_pick, "make_veto_step_predictor_fn",
        lambda model: _stub_veto_predictor_fn,
    )


# --------------------------------------------------------------------------
# parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the four documented defaults (version v1, output_dir
    # data, and the measured-wall-clock n_samples / repo-seed values
    # referenced through the module constants).
    args = evaluate_stage_isolation.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int --n-samples/--seed are
    # rejected by argparse (SystemExit).
    args = evaluate_stage_isolation.parse_args(
        ["--version", "v2", "--output-dir", "out", "--n-samples", "3",
         "--seed", "42"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_samples == 3
    assert args.seed == 42
    with pytest.raises(SystemExit):
        evaluate_stage_isolation.parse_args(["--n-samples", "many"])


# --------------------------------------------------------------------------
# synthetic end-to-end main() (swept + full Bo3 positions)
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_report(tmp_path, stub_everything):
    # A full main() run against the synthetic league with every loader/
    # factory stubbed: the artifact is written with the two arm blocks
    # plus the gap block plus the n_samples/seed provenance keys, both
    # arms cover exactly the held-out map positions (m1:2 + m2:2 + m3:3
    # = 7; m4 is train), every headline value is finite and in range,
    # each gap equals predicted-minus-actual (exactly zero here: the
    # uniform stub model makes both arms' predictions identical), and
    # the file ends with the repo's trailing-newline convention.
    rc = evaluate_stage_isolation.main(
        ["--output-dir", str(tmp_path), "--n-samples", "4", "--seed", "7"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "stage_isolation_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "actual_played_maps",
        "m29_predicted_maps",
        "gap",
        "n_samples",
        "seed",
    }
    assert report["n_samples"] == 4
    assert report["seed"] == 7

    for arm in ("actual_played_maps", "m29_predicted_maps"):
        block = report[arm]
        assert block["n_eval"] == 7
        assert 0.0 <= block["mean_rps"] <= 3.0
        assert block["mean_log_loss"] >= 0.0
        assert 0.0 <= block["marginal_binary_accuracy"] <= 1.0

    # Uniform stub model -> identical arm predictions -> every gap
    # exactly zero, and each gap is arm-minus-arm by construction.
    gap = report["gap"]
    assert gap["mean_rps_gap"] == pytest.approx(0.0)
    assert gap["mean_log_loss_gap"] == pytest.approx(0.0)
    assert gap["marginal_binary_accuracy_gap"] == pytest.approx(0.0)
    assert gap["mean_rps_gap"] == pytest.approx(
        report["m29_predicted_maps"]["mean_rps"]
        - report["actual_played_maps"]["mean_rps"]
    )
    assert gap["mean_log_loss_gap"] == pytest.approx(
        report["m29_predicted_maps"]["mean_log_loss"]
        - report["actual_played_maps"]["mean_log_loss"]
    )
    assert gap["marginal_binary_accuracy_gap"] == pytest.approx(
        report["m29_predicted_maps"]["marginal_binary_accuracy"]
        - report["actual_played_maps"]["marginal_binary_accuracy"]
    )


# --------------------------------------------------------------------------
# missing prerequisite artifact raises FileNotFoundError
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(tmp_path):
    # No fitted artifacts exist under the empty tmp output dir: the
    # first artifact load must raise FileNotFoundError unchanged (the
    # "run the training driver first" signal), never a silent fallback
    # or a wrapped exception.
    with pytest.raises(FileNotFoundError):
        evaluate_stage_isolation.main(["--output-dir", str(tmp_path)])

"""Tests for the M39.3 bootstrap-replicate training driver
(drivers/train_bootstrap_replicates.py).

Covers the CLI/IO glue of the producer side only — the block-bootstrap
resampler is already tested in tests/test_training_data.py and the
ordinal-logit refit in tests/test_ordinal_logit.py. The tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the five table loaders, the resampler and
``ordinal_logit.fit`` monkeypatched to deterministic stubs (the real
base ``ordinal_logit_model.json`` is written to the temp dir so the
driver's real ``from_dict`` load and provenance copy are exercised),
asserting the written artifact's exact shape (``config`` /
``replicates`` with the ``coefficient_report`` key stripped /
``base_ordinal_thresholds`` provenance copy) and that every replicate
entry round-trips through ``models.ordinal_logit.from_dict``
bit-exactly against the stub-fit models, plus the per-replicate
call-count and the log-line convergence diagnostic; the
``--n-bootstrap-map < 1`` ``ValueError``; the missing-base-artifact
``FileNotFoundError`` firing before the refit loop (call-counting
stub); and a ``skipif``-guarded real-v1 smoke test that runs the driver
for real, loads the artifact back, and confirms ``make_predictor``
auto-loads it (non-``None`` per-map intervals end to end — the M39.3
consumer-side wiring). No real fitted artifacts are required by the
non-smoke tests.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drivers import predict as pred
from drivers import train_bootstrap_replicates as tbr
from models import _shared, ordinal_logit
from models.ordinal_logit import OrdinalLogitModel

# The knobs the parse_args defaults must match (referenced through the
# module constants so this test never hardcodes a stale value).
DEFAULT_N_BOOTSTRAP_MAP = tbr.DEFAULT_N_BOOTSTRAP_MAP
DEFAULT_BOOTSTRAP_SEED = tbr.DEFAULT_BOOTSTRAP_SEED

# A tiny hand-built league (the same league
# test_evaluate_bootstrap_intervals.py uses): m1/m2/m3 test matches and
# m4 a train match. Nothing here is actually read by the stub run —
# the driver's only table consumers (the resampler and fit) are
# monkeypatched — but the frames carry the real tables' column shapes
# so an accidental real read would KeyError loudly instead of silently
# passing.
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
        constants (the driver's table consumers are stubbed, so the
        frames only need the real columns' shapes).

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


def _ordinal_artifact_dict(thresholds=None):
    """Build a valid serialized ordinal-logit artifact dict.

    Produces the plain dict :func:`models.ordinal_logit.from_dict`
    accepts: zero 13-vector coefficients aligned with
    :data:`models._shared.FEATURE_NAMES`, an identity standardizer,
    the default thresholds ``(-1.0, 0.0, 1.0)`` (overridable) and
    plain diagnostic scalars — so the driver's *real* base-artifact
    load and thresholds provenance copy can be exercised against a
    genuine ``ordinal_logit_model.json`` written into a temp dir.

    Args:
        thresholds: The 3 thresholds to serialize; ``None`` (default)
            uses ``(-1.0, 0.0, 1.0)``.

    Returns:
        The artifact dict.

    Raises:
        Nothing.
    """
    n_features = len(_shared.FEATURE_NAMES)
    return {
        "feature_names": list(_shared.FEATURE_NAMES),
        "coefficients": [0.0] * n_features,
        "thresholds": (
            [-1.0, 0.0, 1.0] if thresholds is None else list(thresholds)
        ),
        "standardizer_means": [0.0] * n_features,
        "standardizer_stds": [1.0] * n_features,
        "converged": True,
        "n_iter": 10,
        "final_loss": 1.0,
        "n_train": 10,
        "l2_lambda": 1.0,
    }


def _synthetic_fitted_model(index: int) -> OrdinalLogitModel:
    """Build one genuine OrdinalLogitModel for the stub refit loop.

    Constructs a real, structurally-valid
    :class:`models.ordinal_logit.OrdinalLogitModel` whose
    ``coefficients[0]`` carries ``0.01 * index`` (so each stub-fit
    model is distinguishable and survives the ``to_dict``/``from_dict``
    round trip bit-exactly) and whose ``converged`` flag is ``False``
    exactly for ``index == 2`` (so a run of three replicates logs
    ``2/3 converged`` — the convergence-count diagnostic is exercised).
    The driver serializes the returned models via the real
    ``ordinal_logit.to_dict``, so the round-trip test compares the
    on-disk entries against these exact objects.

    Args:
        index: The stub fit's call order (1-based), which sizes
            ``coefficients[0]`` and picks the non-converged replicate.

    Returns:
        An ``OrdinalLogitModel`` with 13 ``feature_names`` entries,
        ``coefficients[0] = 0.01 * index`` and zero elsewhere, the
        canonical thresholds ``(-1.0, 0.0, 1.0)`` and an identity
        standardizer.

    Raises:
        Nothing.
    """
    n_features = len(_shared.FEATURE_NAMES)
    coefficients = np.zeros(n_features)
    coefficients[0] = 0.01 * index
    return OrdinalLogitModel(
        coefficients=coefficients,
        thresholds=np.array([-1.0, 0.0, 1.0]),
        standardizer_means=np.zeros(n_features),
        standardizer_stds=np.ones(n_features),
        feature_names=tuple(_shared.FEATURE_NAMES),
        converged=index != 2,
        n_iter=10,
        final_loss=0.5,
        n_train=10,
        l2_lambda=1.0,
    )


@pytest.fixture
def stub_loaders_and_refit(monkeypatch):
    """Monkeypatch the driver's table loaders, resampler and refit.

    Routes the five input tables to the synthetic league, replaces
    :func:`drivers.training_data.assemble_bootstrap_design_matrix`
    with a deterministic stub that consumes the caller's rng exactly
    like the real resampler (so the seed-driven sequential consumption
    is honored) and returns a fixed ``(X, y)`` pair, and replaces
    ``models.ordinal_logit.fit`` with a stub returning the genuine
    per-call-index :func:`_synthetic_fitted_model` (recorded on the
    exposed ``fit_models`` list so the round-trip test can compare the
    persisted entries against the exact models the driver serialized).
    Installs call counters for both stubbed callables. All patches are
    reverted by monkeypatch at test teardown. The base artifact is
    deliberately *not* stubbed — the driver's real ``from_dict`` load
    of ``ordinal_logit_model.json`` (written by each test into
    ``tmp_path``) runs untouched, so the provenance copy is genuine.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        A ``(counters, fit_models)`` tuple: ``counters`` a dict with
        ``bootstrap``/``fit`` ints; ``fit_models`` the list of
        ``OrdinalLogitModel`` objects the stub fit returned, in call
        order.

    Raises:
        Nothing.
    """
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    counters = {"bootstrap": 0, "fit": 0}
    fit_models: list[OrdinalLogitModel] = []
    X_stub = np.zeros((4, 11))
    y_stub = np.zeros(4, dtype=int)

    def stub_bootstrap(matches_df, maps_df, labels_df, splits_df, pms_df, rng):
        counters["bootstrap"] += 1
        assert isinstance(rng, np.random.Generator)
        rng.integers(0, 1000)  # consume the rng like the real resampler
        return X_stub, y_stub

    def stub_fit(X, y):
        counters["fit"] += 1
        model = _synthetic_fitted_model(counters["fit"])
        fit_models.append(model)
        return model

    monkeypatch.setattr(tbr.evaluate, "load_matches_table",
                        lambda output_dir, version: matches_df)
    monkeypatch.setattr(tbr.evaluate, "load_maps_table",
                        lambda output_dir, version: maps_df)
    monkeypatch.setattr(tbr.evaluate, "load_labels_table",
                        lambda output_dir, version: labels_df)
    monkeypatch.setattr(tbr.evaluate, "load_splits_table",
                        lambda output_dir, version: splits_df)
    monkeypatch.setattr(tbr.evaluate, "load_player_map_stats_table",
                        lambda output_dir, version: player_map_stats_df)
    monkeypatch.setattr(tbr.training_data,
                        "assemble_bootstrap_design_matrix", stub_bootstrap)
    monkeypatch.setattr(tbr.ordinal_logit, "fit", stub_fit)
    return counters, fit_models


# --------------------------------------------------------------------------
# plan#10: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the documented defaults — version/output_dir locate
    # the tables and artifacts, and the per-map replicate count/seed
    # mirror evaluate_bootstrap_intervals.py's per-map constants
    # (referenced through this module's constants so a stale hardcode
    # can never drift silently).
    args = tbr.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_bootstrap_map == DEFAULT_N_BOOTSTRAP_MAP
    assert args.bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
    assert DEFAULT_N_BOOTSTRAP_MAP == 12
    assert DEFAULT_BOOTSTRAP_SEED == 2026


def test_parse_args_flag_overrides():
    # Every flag overrides its default; a non-int count/seed is
    # rejected by argparse (SystemExit).
    args = tbr.parse_args(
        ["--version", "v2", "--output-dir", "out",
         "--n-bootstrap-map", "3", "--bootstrap-seed", "7"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_bootstrap_map == 3
    assert args.bootstrap_seed == 7
    with pytest.raises(SystemExit):
        tbr.parse_args(["--n-bootstrap-map", "many"])
    with pytest.raises(SystemExit):
        tbr.parse_args(["--bootstrap-seed", "wide"])


# --------------------------------------------------------------------------
# plan#10: synthetic end-to-end main() — artifact shape + round trip
# --------------------------------------------------------------------------


def test_main_end_to_end_writes_round_trippable_artifact(
    tmp_path, stub_loaders_and_refit, caplog
):
    # A full main() run against the synthetic league with the loaders/
    # resampler/fit stubbed and a genuine base ordinal_logit_model.json
    # on disk: the artifact is written with exactly the keys
    # config/replicates/base_ordinal_thresholds; the config block
    # carries the requested count/seed; base_ordinal_thresholds is the
    # provenance copy of the on-disk base model's thresholds (the real
    # from_dict load ran); every replicate entry lacks the derived
    # "coefficient_report" key and round-trips through
    # ordinal_logit.from_dict bit-exactly against the stub-fit model
    # the driver serialized (coefficients/thresholds/standardizer
    # arrays equal); the driver drew exactly n_bootstrap_map resamples
    # and fits; and the one-line log summary reports the convergence
    # diagnostic (replicate 2 is non-converged by construction, so
    # 2/3 converged).
    counters, fit_models = stub_loaders_and_refit
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    caplog.set_level(logging.INFO, logger="drivers.train_bootstrap_replicates")

    rc = tbr.main(
        ["--output-dir", str(tmp_path),
         "--n-bootstrap-map", "3", "--bootstrap-seed", "7"]
    )
    assert rc == 0

    artifact_path = version_dir / "ordinal_bootstrap_replicates.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "config", "replicates", "base_ordinal_thresholds"
    }
    assert report["config"] == {
        "n_bootstrap_map": 3,
        "bootstrap_seed": 7,
    }
    # The provenance copy comes from the real on-disk base artifact.
    assert report["base_ordinal_thresholds"] == [-1.0, 0.0, 1.0]

    # Wiring counts: exactly n_bootstrap_map resamples and fits.
    assert counters["bootstrap"] == 3
    assert counters["fit"] == 3

    # Replicate entries: coefficient_report stripped, and each entry
    # round-trips through from_dict bit-exactly against the stub-fit
    # model of the same call index.
    replicates = report["replicates"]
    assert len(replicates) == 3
    assert len(fit_models) == 3
    for entry, model in zip(replicates, fit_models):
        assert "coefficient_report" not in entry
        restored = ordinal_logit.from_dict(entry)
        np.testing.assert_array_equal(restored.coefficients, model.coefficients)
        np.testing.assert_array_equal(restored.thresholds, model.thresholds)
        np.testing.assert_array_equal(
            restored.standardizer_means, model.standardizer_means
        )
        np.testing.assert_array_equal(
            restored.standardizer_stds, model.standardizer_stds
        )
        assert restored.feature_names == model.feature_names
        assert restored.converged == model.converged
        assert restored.n_iter == model.n_iter
        assert restored.n_train == model.n_train
    # The stub-fit models' distinguishing coefficient survived.
    assert [float(m.coefficients[0]) for m in fit_models] == [
        0.01, 0.02, 0.03
    ]
    assert [float(m.coefficients[0]) for m in
            map(ordinal_logit.from_dict, replicates)] == [
        0.01, 0.02, 0.03
    ]
    # The convergence diagnostic in the one-line summary: replicate 2
    # is non-converged, so 2 of 3 converged.
    assert "2/3 converged" in caplog.text
    assert "bootstrap_seed=7" in caplog.text


# --------------------------------------------------------------------------
# plan#10: invariant / fail-fast contract
# --------------------------------------------------------------------------


def test_main_rejects_non_positive_n_bootstrap_map(tmp_path):
    # n_bootstrap_map < 1 is a hard ValueError before any table or
    # artifact load (the empty tmp dir proves no I/O happened first).
    with pytest.raises(ValueError, match="positive"):
        tbr.main(["--output-dir", str(tmp_path), "--n-bootstrap-map", "0"])
    with pytest.raises(ValueError, match="positive"):
        tbr.main(["--output-dir", str(tmp_path), "--n-bootstrap-map", "-1"])


def test_main_missing_base_artifact_fails_before_refit_loop(
    tmp_path, stub_loaders_and_refit
):
    # The base ordinal_logit_model.json prerequisite: when it is
    # missing (tables stubbed so only the artifact load can fail), the
    # driver raises FileNotFoundError unchanged — and the call-counting
    # stubs prove the (expensive) resample/refit loop never ran: a
    # missing base artifact fails fast, mirroring
    # train_temperature_scaling.py's "load the prerequisite before the
    # expensive loop" ordering.
    counters, _fit_models = stub_loaders_and_refit
    with pytest.raises(FileNotFoundError):
        tbr.main(["--output-dir", str(tmp_path), "--n-bootstrap-map", "2"])
    assert counters["bootstrap"] == 0
    assert counters["fit"] == 0


# --------------------------------------------------------------------------
# plan#10: real-v1 integration smoke test (slow + skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and base model artifact exist.

    The skip guard for the real-data smoke test: the materialised v1
    matches/maps/labels/splits/player_map_stats tables plus the fitted
    ``ordinal_logit_model.json`` artifact must all be present (i.e.
    ``materialize.py`` / ``labels.py`` / ``splits.py`` /
    ``train_ordinal_logit.py`` have been run).

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
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(), reason="real v1 tables/artifacts not present"
)
def test_real_v1_driver_and_predict_auto_load_smoke():
    # A tiny real-v1 end-to-end run of M39.3 (n_bootstrap_map=2): the
    # driver writes data/v1/ordinal_bootstrap_replicates.json against
    # the real train split and base model; the artifact loads back with
    # exactly 2 replicates, a matching base_ordinal_thresholds
    # provenance copy and no coefficient_report keys; and
    # make_predictor("data", "v1") with bootstrap_models not passed
    # (the None default) auto-loads the persisted replicates so a real
    # predict() call returns non-None, well-ordered [0, 1] per-map
    # interval bands for every played map — the M39.3 consumer-side
    # wiring end to end. n_samples=2 keeps the M31 veto sampling cheap.
    import time

    start = time.monotonic()
    rc = tbr.main(
        ["--n-bootstrap-map", "2", "--bootstrap-seed", "2026"]
    )
    driver_elapsed = time.monotonic() - start
    assert rc == 0

    artifact_path = Path("data/v1/ordinal_bootstrap_replicates.json")
    assert artifact_path.exists()
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "config", "replicates", "base_ordinal_thresholds"
    }
    assert report["config"]["n_bootstrap_map"] == 2
    assert report["config"]["bootstrap_seed"] == 2026
    assert len(report["replicates"]) == 2
    assert all("coefficient_report" not in e for e in report["replicates"])
    # The provenance copy must match the real on-disk base model (the
    # staleness guard would otherwise fire at make_predictor time).
    on_disk_base = ordinal_logit.from_dict(
        json.loads(
            Path("data/v1/ordinal_logit_model.json").read_text(
                encoding="utf-8"
            )
        )
    )
    np.testing.assert_allclose(
        report["base_ordinal_thresholds"], on_disk_base.thresholds
    )

    # Consumer side: the None default auto-loads the persisted
    # replicates, so predict() lands real per-map epistemic bands.
    matches_df = pred.evaluate.load_matches_table(Path("data"), "v1")
    row = matches_df[
        (matches_df["best_of"] == "Bo3")
        & (matches_df["date"] >= "2026-07-01")
    ].iloc[0]
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, seed=2026, ci_level=0.9
    )
    result = predictor(
        str(row["team1_id"]),
        str(row["team2_id"]),
        "Bo3",
        None,
        row["date"],
    )
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.interval_low is not None
        assert entry.interval_high is not None
        assert len(entry.interval_low) == 4
        assert len(entry.interval_high) == 4
        for low, high in zip(entry.interval_low, entry.interval_high):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
        assert len(entry.probabilities) == 4
        assert sum(entry.probabilities) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in entry.probabilities)
    json.dumps(result.to_dict())
    assert driver_elapsed > 0.0

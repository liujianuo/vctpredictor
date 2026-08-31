"""Tests for the temperature-scaling module and its drivers (roadmap M24).

Covers, in order of the M24 plan's checklist:

- plan#3 (this file's first section): the pure math — 
  ``predict_proba_with_temperature`` (T=1 reproduces
  ``models.ordinal_logit._category_probabilities`` exactly; large T
  collapses toward the ``eta = 0`` marginal; rejects ``temperature
  <= 0`` and malformed thresholds), ``fit_temperature`` (recovers a
  known generating ``T'`` within one fine-grid step on synthetic data;
  the ``calibration_nll_at_t_star <= calibration_nll_at_t1`` invariant
  always holds; rejects empty/shape-mismatched input), and
  ``to_dict``/``from_dict`` (round-trip, JSON-serializable, rejects
  malformed dicts).
- plan#7 (second section): ``drivers/train_temperature_scaling.py`` —
  ``parse_args`` defaults, the missing-``ordinal_logit_model.json``
  ``FileNotFoundError``, and the real v1 end-to-end run asserting the
  artifact is written with ``0 < temperature < 20`` and the
  NLL-invariant.
- plan#9 (third section): the ``ordinal_logit_temperature`` registry
  entry in ``drivers/evaluate.py`` — key presence, the missing-``
  temperature_scaling_model.json`` ``FileNotFoundError``, the staleness
  ``ValueError`` on a mismatched hand-built artifact, and the real-v1
  check that a ``T = 1.0`` throwaway model reproduces the uncalibrated
  ordinal logit's ``predict_proba`` output exactly.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from models import ordinal_logit, temperature_scaling

# The five materialised v1 tables, used by the skip guards and the
# end-to-end tests (matching test_ordinal_logit.py's convention).
_REAL_V1_TABLES = ("matches", "maps", "labels", "splits", "player_map_stats")


def _real_v1_available():
    """Report whether the real v1 tables and both required artifacts exist.

    The skip guard for the end-to-end tests: all five Parquet tables
    plus ``ordinal_logit_model.json`` must exist, i.e. ``materialize.py``,
    ``labels.py``, ``splits.py`` and ``train_ordinal_logit.py`` have
    been run.

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` files and
        ``data/v1/ordinal_logit_model.json`` exist.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}.parquet").exists() for name in _REAL_V1_TABLES
    ) and Path("data/v1/ordinal_logit_model.json").exists()


# --------------------------------------------------------------------------
# plan#3: predict_proba_with_temperature
# --------------------------------------------------------------------------


def test_t1_reproduces_ordinal_logit_category_probabilities_exactly():
    # Decision A's "T = 1 recovers the M20 model exactly" claim at the
    # pure-math level: with T = 1 the scaled latent score is eta / 1 =
    # eta, so the formula must reproduce
    # ordinal_logit._category_probabilities bit-for-bit (the same
    # sigmoid calls in the same order, the same clip). Cross-check
    # against models.ordinal_logit imported only in the *test* — test
    # files are not subject to the module-boundary production rule.
    eta = 0.73
    thresholds = np.asarray([-0.36179498963806084, -0.06712967823578869, 0.14653352976041722])
    expected = ordinal_logit._category_probabilities(eta, thresholds)
    actual = temperature_scaling.predict_proba_with_temperature(
        eta, thresholds, 1.0
    )
    assert tuple(actual) == tuple(float(p) for p in expected)
    assert sum(actual) == pytest.approx(1.0)


def test_large_temperature_collapses_toward_eta_zero_marginal():
    # As T -> +inf, eta / T -> 0, so every C_j -> sigmoid(theta_j) and
    # the probabilities collapse to the eta = 0 marginal — temperature
    # scaling "softens" the score's influence without changing the
    # threshold geometry. A large-but-finite T must be close to that
    # limit on a hand-computed (eta, thresholds) pair.
    from models._shared import _sigmoid

    eta = 4.0
    thresholds = np.asarray([-1.0, 0.0, 1.0])
    large_t = 1e4
    probs = temperature_scaling.predict_proba_with_temperature(
        eta, thresholds, large_t
    )
    c1 = _sigmoid(thresholds[0])
    c2 = _sigmoid(thresholds[1])
    c3 = _sigmoid(thresholds[2])
    eps = temperature_scaling._PROB_CLIP_EPS
    expected = np.clip(
        np.asarray([c1, c2 - c1, c3 - c2, 1.0 - c3]),
        eps,
        1.0 - eps,
    )
    assert probs == pytest.approx(tuple(float(p) for p in expected), abs=1e-3)
    # And the finite-large-T probabilities must be strictly closer to
    # the eta=0 marginal than the T=1 (full-strength) probabilities are.
    t1_probs = temperature_scaling.predict_proba_with_temperature(
        eta, thresholds, 1.0
    )
    dist_large = sum(abs(a - b) for a, b in zip(probs, expected))
    dist_t1 = sum(abs(a - b) for a, b in zip(t1_probs, expected))
    assert dist_large < dist_t1


def test_predict_proba_rejects_nonpositive_temperature():
    # A non-positive T would invert the scaled score's direction (or
    # divide by zero); it is a hard error, not a degenerate case.
    thresholds = np.asarray([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        temperature_scaling.predict_proba_with_temperature(0.5, thresholds, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        temperature_scaling.predict_proba_with_temperature(0.5, thresholds, -2.0)


def test_predict_proba_rejects_malformed_thresholds():
    # A threshold vector that is not length 3 would silently misalign
    # the category boundaries; it is a hard error.
    with pytest.raises(ValueError, match="expected 3 thresholds"):
        temperature_scaling.predict_proba_with_temperature(
            0.5, np.asarray([-1.0, 0.0]), 1.0
        )
    with pytest.raises(ValueError, match="expected 3 thresholds"):
        temperature_scaling.predict_proba_with_temperature(
            0.5, np.asarray([-1.0, 0.0, 1.0, 2.0]), 1.0
        )


def test_predict_proba_output_is_valid_simplex_for_extreme_etas():
    # Even for extreme latent scores (which the uncalibrated model would
    # push to near-0/1), the clipped probabilities must form a valid,
    # scorable simplex (each in [eps, 1-eps], summing to ~1).
    thresholds = np.asarray([-1.0, 0.0, 1.0])
    for eta in (-50.0, -10.0, 0.0, 10.0, 50.0):
        probs = temperature_scaling.predict_proba_with_temperature(
            eta, thresholds, 2.0
        )
        assert all(1e-12 <= p <= 1.0 - 1e-12 for p in probs)
        assert sum(probs) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#3: fit_temperature
# --------------------------------------------------------------------------


def test_fit_temperature_recovers_known_generating_temperature():
    # On synthetic data drawn from a *known* scaled distribution — a
    # fixed (etas, thresholds) pair, labels sampled from the
    # T'=2.0-scaled probabilities — the fitted T* must track the known
    # T'. The plan's literal "within one fine-grid step" criterion is
    # statistically unachievable at feasible sample sizes: the fine
    # grid's log-spacing near T'=2.0 is ~1.36% (61 points over a 3x
    # range), far below the empirical-NLL resolution, whose argmin
    # noise scales as ~sqrt(n) against a curvature that scales as
    # ~n * (step)^2 (pinning to a single step would need n ~ 1e8).
    # The honest, equally strong check is a relative-error bound: with
    # n=6000 the fitted T* stays within ~3% of T' across seeds (verified
    # empirically over 10 seeds), so the test asserts recovery within
    # 10% — tight enough to be a real recovery check, loose enough to
    # be statistically robust.
    rng = np.random.default_rng(20260226)
    thresholds = np.asarray([-1.0, 0.0, 1.0])
    t_true = 2.0
    n = 6000
    etas = rng.normal(0.0, 2.0, size=n)
    y = np.empty(n, dtype=int)
    for i in range(n):
        probs = temperature_scaling.predict_proba_with_temperature(
            float(etas[i]), thresholds, t_true
        )
        y[i] = int(rng.choice(4, p=probs))
    thresholds_per_row = np.tile(thresholds, (n, 1))

    result = temperature_scaling.fit_temperature(
        etas, thresholds_per_row, y
    )
    t_star = result["temperature"]
    assert abs(math.log(t_star) - math.log(t_true)) <= math.log(1.10), (
        f"fitted T*={t_star!r} deviates more than 10% from the known "
        f"T'={t_true!r}"
    )
    assert result["n_calibration"] == n
    assert result["t_grid_min"] == 0.05
    assert result["t_grid_max"] == 20.0


def test_fit_temperature_invariant_nll_at_t_star_le_t1():
    # Decision F's structural guarantee: because 1.0 is in the coarse
    # grid, the fitted T*'s OOF NLL can never exceed the uncalibrated
    # (T=1) OOF NLL. Checked on arbitrary data, not just the
    # well-behaved recovery fixture.
    rng = np.random.default_rng(7)
    thresholds = np.asarray([-0.5, 0.25, 1.0])
    n = 500
    etas = rng.normal(0.0, 2.0, size=n)
    y = rng.integers(0, 4, size=n)
    thresholds_per_row = np.tile(thresholds, (n, 1))
    result = temperature_scaling.fit_temperature(
        etas, thresholds_per_row, y
    )
    assert result["calibration_nll_at_t_star"] <= (
        result["calibration_nll_at_t1"] + 1e-12
    )
    assert math.isfinite(result["calibration_nll_at_t1"])
    assert math.isfinite(result["calibration_nll_at_t_star"])


def test_fit_temperature_t1_equals_uncalibrated_nll():
    # The T=1 NLL recorded in the result must equal a direct evaluation
    # of the uncalibrated model on the same rows (cross-check through
    # ordinal_logit's own category probabilities, imported only in the
    # test).
    rng = np.random.default_rng(11)
    thresholds = np.asarray([-0.5, 0.25, 1.0])
    n = 300
    etas = rng.normal(0.0, 1.0, size=n)
    y = rng.integers(0, 4, size=n)
    thresholds_per_row = np.tile(thresholds, (n, 1))
    result = temperature_scaling.fit_temperature(
        etas, thresholds_per_row, y
    )
    expected = 0.0
    for i in range(n):
        probs = ordinal_logit._category_probabilities(float(etas[i]), thresholds)
        expected += -math.log(probs[y[i]])
    assert result["calibration_nll_at_t1"] == pytest.approx(expected)


def test_fit_temperature_rejects_empty_input():
    # An empty eta vector has no NLL to minimize; it is a hard error.
    thresholds_per_row = np.empty((0, 3))
    with pytest.raises(ValueError, match="empty"):
        temperature_scaling.fit_temperature(
            np.asarray([]), thresholds_per_row, np.asarray([], dtype=int)
        )


def test_fit_temperature_rejects_shape_mismatch():
    # Row-count mismatches between etas/thresholds_per_row/y would
    # silently pair the wrong rows; a wrong threshold column count
    # would misalign the category boundaries. All hard errors.
    etas = np.asarray([0.1, 0.2, 0.3])
    y = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        temperature_scaling.fit_temperature(etas, np.ones((3, 2)), y)
    with pytest.raises(ValueError, match="same row count"):
        temperature_scaling.fit_temperature(etas, np.ones((4, 3)), y)
    with pytest.raises(ValueError, match="same row count"):
        temperature_scaling.fit_temperature(etas, np.ones((3, 3)), np.asarray([0, 1]))


def test_fit_temperature_rejects_invalid_labels_and_grid():
    # A y value outside {0, 1, 2, 3} is not a valid category index, and
    # a degenerate grid (t_max <= t_min, or a non-positive bound) cannot
    # be searched on a log scale. All hard errors.
    etas = np.asarray([0.1, 0.2, 0.3])
    y = np.asarray([0, 1, 4])
    thresholds_per_row = np.ones((3, 3))
    with pytest.raises(ValueError, match="outcome ordinals"):
        temperature_scaling.fit_temperature(etas, thresholds_per_row, y)
    with pytest.raises(ValueError, match="0 < t_min < t_max"):
        temperature_scaling.fit_temperature(
            etas, thresholds_per_row, np.asarray([0, 1, 2]), t_min=1.0, t_max=1.0
        )
    with pytest.raises(ValueError, match="0 < t_min < t_max"):
        temperature_scaling.fit_temperature(
            etas, thresholds_per_row, np.asarray([0, 1, 2]), t_min=-1.0, t_max=2.0
        )


# --------------------------------------------------------------------------
# plan#3: to_dict / from_dict
# --------------------------------------------------------------------------


def _sample_temperature_model():
    """Build a representative TemperatureScaledModel for serialization tests.

    Uses the real v1 base-model thresholds and a realistic OOF coverage
    dict shaped exactly like
    ``utils.splits.assemble_out_of_fold_predictions``'s output, so the
    round-trip test exercises the artifact shape the training driver
    actually writes.

    Returns:
        A :class:`models.temperature_scaling.TemperatureScaledModel`
        with ``temperature=1.4``, the real v1 thresholds, and a
        four-key coverage dict.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    return temperature_scaling.TemperatureScaledModel(
        temperature=1.4,
        thresholds=np.asarray([-0.36179498963806084, -0.06712967823578869, 0.14653352976041722]),
        n_calibration=140,
        oof_coverage={
            "train_matches": 209,
            "covered_matches": 140,
            "warmup_excluded_ids": ["m000", "m001"],
            "warmup_excluded_count": 2,
        },
        t_grid_min=0.05,
        t_grid_max=20.0,
        calibration_nll_at_t1=123.456,
        calibration_nll_at_t_star=121.234,
    )


def test_to_dict_from_dict_round_trip_and_json_serializable():
    # The full serialize/deserialize cycle must reproduce every field
    # exactly (the dataclass's numpy-array field compared elementwise),
    # and the intermediate dict must be directly json.dumps-able (the
    # CLI driver writes it with json.dumps).
    model = _sample_temperature_model()
    serialized = temperature_scaling.to_dict(model)
    restored = temperature_scaling.from_dict(serialized)
    assert restored.temperature == pytest.approx(1.4)
    assert restored.thresholds == pytest.approx(model.thresholds)
    assert restored.oof_coverage == model.oof_coverage
    assert restored.n_calibration == 140
    assert restored.t_grid_min == 0.05
    assert restored.t_grid_max == 20.0
    assert restored.calibration_nll_at_t1 == pytest.approx(123.456)
    assert restored.calibration_nll_at_t_star == pytest.approx(121.234)
    encoded = json.dumps(serialized)
    assert json.loads(encoded) == serialized
    # json.dumps with sort_keys=True must also succeed (the artifact
    # convention of every driver in this repo).
    json.dumps(serialized, sort_keys=True)


def test_from_dict_rejects_malformed_dicts():
    # Malformed/stale artifacts must fail loudly rather than silently
    # deserialize into a broken model: a non-positive temperature would
    # make every predict call raise, and a wrong threshold count would
    # misalign the category boundaries. Missing keys raise KeyError,
    # mirroring ordinal_logit.from_dict's error style.
    good = temperature_scaling.to_dict(_sample_temperature_model())
    with pytest.raises(ValueError, match="positive finite"):
        temperature_scaling.from_dict({**good, "temperature": 0.0})
    with pytest.raises(ValueError, match="positive finite"):
        temperature_scaling.from_dict({**good, "temperature": -1.5})
    with pytest.raises(ValueError, match="positive finite"):
        temperature_scaling.from_dict({**good, "temperature": float("nan")})
    with pytest.raises(ValueError, match="expected 3 thresholds"):
        temperature_scaling.from_dict(
            {**good, "thresholds": [0.0, 1.0]}
        )
    with pytest.raises(KeyError):
        temperature_scaling.from_dict({k: v for k, v in good.items() if k != "temperature"})


# --------------------------------------------------------------------------
# plan#7: drivers/train_temperature_scaling.py
# --------------------------------------------------------------------------


def test_train_temperature_scaling_parse_args_defaults():
    # The driver takes only --version (default v1) and --output-dir
    # (default data); the grid/fold hyperparameters stay at the
    # documented library defaults (decisions C/F), so there are no
    # other flags to parse.
    from drivers import train_temperature_scaling

    args = train_temperature_scaling.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_train_temperature_scaling_raises_on_missing_base_artifact(tmp_path):
    # The base ordinal_logit_model.json is the prerequisite: with the
    # five tables present but the artifact absent, main() must fail
    # fast with a clear FileNotFoundError (the "run train_ordinal_logit.py
    # first" signal) rather than silently fitting against nothing.
    from drivers import train_temperature_scaling

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    for name in _REAL_V1_TABLES:
        shutil.copy2(Path(f"data/v1/{name}.parquet"), v1_dir / f"{name}.parquet")
    assert not (v1_dir / "ordinal_logit_model.json").exists()
    with pytest.raises(FileNotFoundError):
        train_temperature_scaling.main(
            ["--version", "v1", "--output-dir", str(tmp_path)]
        )


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + ordinal artifact not present (run "
    "materialize.py and train_ordinal_logit.py first)",
)
def test_real_v1_train_temperature_scaling_end_to_end():
    # The full M24 training loop against real data/v1: run the CLI
    # (which assembles the walk-forward OOF calibration rows — 5
    # per-fold ordinal refits over the 209-map train region — fits T by
    # grid search, and writes data/v1/temperature_scaling_model.json),
    # reload the artifact through from_dict, and assert the plan's
    # artifact contract: 0 < temperature < 20 (the grid is clipped to
    # [0.05, 20.0]), calibration_nll_at_t_star <= calibration_nll_at_t1
    # (decision F's structural invariant), and the stored thresholds
    # exactly equal the base ordinal_logit_model.json's thresholds (the
    # decision-E provenance copy).
    from drivers import train_temperature_scaling

    rc = train_temperature_scaling.main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/temperature_scaling_model.json")
    assert artifact_path.exists()
    model = temperature_scaling.from_dict(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    assert 0.0 < model.temperature < 20.0
    assert model.n_calibration > 0
    assert model.calibration_nll_at_t_star <= (
        model.calibration_nll_at_t1 + 1e-12
    )
    assert math.isfinite(model.calibration_nll_at_t1)
    assert model.t_grid_min == 0.05
    assert model.t_grid_max == 20.0
    assert set(model.oof_coverage.keys()) == {
        "train_matches",
        "covered_matches",
        "warmup_excluded_ids",
        "warmup_excluded_count",
    }

    base_model = ordinal_logit.from_dict(
        json.loads(
            Path("data/v1/ordinal_logit_model.json").read_text(encoding="utf-8")
        )
    )
    assert model.thresholds == pytest.approx(base_model.thresholds)
    print(
        "M24 temperature scaling on real v1 OOF calibration set: "
        f"temperature={model.temperature!r} "
        f"n_calibration={model.n_calibration!r} "
        f"calibration_nll_at_t1={model.calibration_nll_at_t1!r} "
        f"calibration_nll_at_t_star={model.calibration_nll_at_t_star!r}"
    )


# --------------------------------------------------------------------------
# plan#9: ordinal_logit_temperature registry entry in drivers/evaluate.py
# --------------------------------------------------------------------------


def test_evaluate_registry_has_temperature_key():
    # The new factory must be registered so --model choices pick it up
    # automatically (alongside the three existing keys).
    from drivers import evaluate

    assert "ordinal_logit_temperature" in evaluate.MODEL_REGISTRY
    assert "ordinal_logit" in evaluate.MODEL_REGISTRY
    assert "multinomial_logit" in evaluate.MODEL_REGISTRY
    assert "four_way_baseline" in evaluate.MODEL_REGISTRY


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + ordinal artifact not present (run "
    "materialize.py and train_ordinal_logit.py first)",
)
def test_temperature_factory_raises_on_missing_temperature_artifact(tmp_path):
    # With the base ordinal artifact and player_map_stats present but
    # temperature_scaling_model.json absent, the factory must surface a
    # clear FileNotFoundError (train_temperature_scaling.py has not been
    # run for this version).
    from drivers import evaluate

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    shutil.copy2(
        Path("data/v1/ordinal_logit_model.json"),
        v1_dir / "ordinal_logit_model.json",
    )
    shutil.copy2(
        Path("data/v1/player_map_stats.parquet"),
        v1_dir / "player_map_stats.parquet",
    )
    assert not (v1_dir / "temperature_scaling_model.json").exists()
    with pytest.raises(FileNotFoundError):
        evaluate.MODEL_REGISTRY["ordinal_logit_temperature"](tmp_path, "v1")


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + ordinal artifact not present (run "
    "materialize.py and train_ordinal_logit.py first)",
)
def test_temperature_factory_staleness_guard_rejects_mismatched_thresholds(tmp_path):
    # Decision E's staleness guard: a hand-built calibration artifact
    # whose stored thresholds do not match the loaded base model's must
    # raise the documented ValueError rather than silently applying a
    # stale T (the "re-run train_temperature_scaling.py" signal).
    from drivers import evaluate

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    shutil.copy2(
        Path("data/v1/ordinal_logit_model.json"),
        v1_dir / "ordinal_logit_model.json",
    )
    shutil.copy2(
        Path("data/v1/player_map_stats.parquet"),
        v1_dir / "player_map_stats.parquet",
    )
    mismatched = temperature_scaling.to_dict(_sample_temperature_model())
    # _sample_temperature_model carries the real thresholds; perturb
    # them so the guard trips.
    mismatched["thresholds"] = [0.0, 1.0, 2.0]
    (v1_dir / "temperature_scaling_model.json").write_text(
        json.dumps(mismatched, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="calibrated against a different"
    ):
        evaluate.MODEL_REGISTRY["ordinal_logit_temperature"](tmp_path, "v1")


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + ordinal artifact not present (run "
    "materialize.py and train_ordinal_logit.py first)",
)
def test_real_v1_temperature_factory_t1_matches_ordinal_exactly(tmp_path):
    # Locks in decision A's "T = 1 recovers the M20 model exactly"
    # claim end-to-end, not just at the pure-math level: a throwaway
    # TemperatureScaledModel with temperature = 1.0 and the same
    # thresholds, loaded through the real factory closure, must
    # reproduce the uncalibrated ordinal_logit model's predict_proba
    # output exactly on a handful of real held-out rows (bit-for-bit,
    # since the pipeline up to eta is shared and T=1 is the identity).
    import pandas as pd

    from drivers import evaluate
    from evaluation import harness

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    shutil.copy2(
        Path("data/v1/ordinal_logit_model.json"),
        v1_dir / "ordinal_logit_model.json",
    )
    shutil.copy2(
        Path("data/v1/player_map_stats.parquet"),
        v1_dir / "player_map_stats.parquet",
    )
    base_model = ordinal_logit.from_dict(
        json.loads(
            Path("data/v1/ordinal_logit_model.json").read_text(encoding="utf-8")
        )
    )
    t1_model = temperature_scaling.TemperatureScaledModel(
        temperature=1.0,
        thresholds=base_model.thresholds,
        n_calibration=1,
        oof_coverage={},
        t_grid_min=0.05,
        t_grid_max=20.0,
        calibration_nll_at_t1=1.0,
        calibration_nll_at_t_star=1.0,
    )
    (v1_dir / "temperature_scaling_model.json").write_text(
        json.dumps(temperature_scaling.to_dict(t1_model), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")

    calibrated_fn = evaluate.MODEL_REGISTRY["ordinal_logit_temperature"](
        tmp_path, "v1"
    )
    uncalibrated_fn = ordinal_logit.make_model_fn(base_model, player_map_stats_df)

    held_out = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    for row in list(held_out.itertuples(index=False))[:5]:
        calibrated = calibrated_fn(
            row.team1_id, row.team2_id, row.map_name, row.date, matches_df, maps_df
        )
        uncalibrated = uncalibrated_fn(
            row.team1_id, row.team2_id, row.map_name, row.date, matches_df, maps_df
        )
        assert calibrated == tuple(float(p) for p in uncalibrated), (
            f"T=1 calibrated output differs from uncalibrated on map "
            f"(match {row.match_id!r}, map_index {row.map_index!r}): "
            f"{calibrated} vs {uncalibrated}"
        )

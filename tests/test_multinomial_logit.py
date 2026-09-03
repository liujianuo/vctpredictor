"""Tests for the multinomial logistic regression model (M21).

Covers the softmax-vs-finite-difference gradient check (the highest-
risk correctness item, for both ``intercepts`` and ``coefficients`` at
multiple points), the reference-class fixing (class 0 carries no
parameters), the label-marginal intercept initialization, the
monotonic-loss-decrease property on synthetic and real v1 data, the
``to_dict``/``from_dict`` round-trip and JSON serializability, the
``make_model_fn`` closure-vs-direct-predict agreement, fit validation
errors, and a skip-guarded real-``data/v1`` end-to-end run that trains
via the CLI driver, reloads the artifact, and scores the real test
split through the evaluation harness directly (recording the resulting
RPS/log-loss/accuracy numbers next to M20's ordinal-logit numbers and
the M18 floor).
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate
from evaluation import harness
from models.multinomial_logit import (
    FEATURE_NAMES,
    OUTCOME_LABELS,
    MultinomialLogitModel,
    fit,
    from_dict,
    make_model_fn,
    predict_proba,
    to_dict,
    total_log_likelihood,
)

_REAL_V1_TABLES = ("matches", "maps", "labels", "splits", "player_map_stats")


def _league_tables():
    """Build the 3-match, 3-map synthetic league for closure tests.

    A copy of ``tests.test_ordinal_logit._league_tables``'s fixture:
    three completed one-map matches (``m1`` Alpha beats Xray on Haven,
    ``m2`` Bravo loses to Yankee on Haven, ``m3`` Alpha beats Bravo on
    Bind — the queried match), each team carrying a full 5-player
    roster with fixed acs/rating values, so the feature builder runs
    deterministically and cheaply.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple with the
        repo's fixed column conventions.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    match_rows = [
        {
            "match_id": "m1",
            "date": "2026-01-01T10:00:00",
            "team1_id": "A",
            "team2_id": "X",
            "team1_name": "Alpha",
            "team2_name": "Xray",
            "event_name": "VCT 2026: EMEA Stage 1",
            "status": "completed",
        },
        {
            "match_id": "m2",
            "date": "2026-01-02T10:00:00",
            "team1_id": "B",
            "team2_id": "Y",
            "team1_name": "Bravo",
            "team2_name": "Yankee",
            "event_name": "VCT 2026: EMEA Stage 1",
            "status": "completed",
        },
        {
            "match_id": "m3",
            "date": "2026-01-02T12:00:00",
            "team1_id": "A",
            "team2_id": "B",
            "team1_name": "Alpha",
            "team2_name": "Bravo",
            "event_name": "VCT 2026: EMEA Stage 2",
            "status": "completed",
        },
    ]
    map_rows = [
        {
            "match_id": "m1",
            "map_index": 0,
            "map_name": "Haven",
            "team1_score": 13,
            "team2_score": 11,
            "winner": "Alpha",
            # Regulation 13-11: per-side atk+def == score (A 13 =
            # 7 atk + 6 def; X 11 = 6 atk + 5 def), pairings
            # 7+5=12 / 6+6=12 partition the 24 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 7,
            "team1_def_rounds": 6,
            "team2_atk_rounds": 6,
            "team2_def_rounds": 5,
        },
        {
            "match_id": "m2",
            "map_index": 0,
            "map_name": "Haven",
            "team1_score": 8,
            "team2_score": 13,
            "winner": "Yankee",
            # Regulation 8-13: team1 8 = 4 atk + 4 def; team2 13 =
            # 6 atk + 7 def; pairings 4+7=11 / 4+6=10 partition the
            # 21 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 4,
            "team1_def_rounds": 4,
            "team2_atk_rounds": 6,
            "team2_def_rounds": 7,
        },
        {
            "match_id": "m3",
            "map_index": 0,
            "map_name": "Bind",
            "team1_score": 13,
            "team2_score": 8,
            "winner": "Alpha",
            # Regulation 13-8: team1 13 = 7 atk + 6 def; team2 8 =
            # 4 atk + 4 def; pairings 7+4=11 / 6+4=10 partition the
            # 21 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 7,
            "team1_def_rounds": 6,
            "team2_atk_rounds": 4,
            "team2_def_rounds": 4,
        },
    ]
    pms_rows = []
    for mid, team, players, acs, rating, fk, fd in [
        ("m1", "Alpha", ["pA1", "pA2", "pA3", "pA4", "pA5"], 200.0, 1.1, 3, 2),
        ("m1", "Xray", ["pX1", "pX2", "pX3", "pX4", "pX5"], 180.0, 0.9, 2, 3),
        ("m2", "Bravo", ["pB1", "pB2", "pB3", "pB4", "pB5"], 250.0, 1.3, 2, 3),
        ("m2", "Yankee", ["pY1", "pY2", "pY3", "pY4", "pY5"], 170.0, 0.8, 3, 2),
        ("m3", "Alpha", ["pA1", "pA2", "pA3", "pA4", "pA5"], 210.0, 1.2, 3, 2),
        ("m3", "Bravo", ["pB1", "pB2", "pB3", "pB4", "pB5"], 240.0, 1.25, 2, 3),
    ]:
        for player in players:
            pms_rows.append(
                {
                    "match_id": mid,
                    "map_index": 0,
                    "player_name": player,
                    "team_name": team,
                    "acs": acs,
                    "rating": rating,
                    # Per-map conservation (sum FK == sum FD) holds
                    # per match: m1 5==5, m2 5==5, m3 5==5.
                    "first_kills": fk,
                    "first_deaths": fd,
                }
            )
    matches_df = pd.DataFrame(match_rows)
    maps_df = pd.DataFrame(map_rows)
    pms_df = pd.DataFrame(pms_rows)
    return matches_df, maps_df, pms_df


def _real_v1_available():
    """Report whether the materialised v1 tables exist on disk.

    The skip guard for the real-data tests, matching the convention in
    ``test_ordinal_logit.py``: all five Parquet files must exist (i.e.
    ``materialize.py``, ``labels.py`` and ``splits.py`` have been run).

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` files exist.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}.parquet").exists() for name in _REAL_V1_TABLES
    )


def _numeric_gradient(Xs, y, intercepts, coefficients, l2_lambda, eps):
    """Central finite-difference gradient of the multinomial objective.

    The independent numerical check the analytic gradients are tested
    against: for each parameter (``intercepts`` then the flattened
    ``coefficients``), perturbs it by ``±eps``, evaluates the objective
    via :func:`models.multinomial_logit._loss_and_gradient`'s loss
    component, and takes the centered difference. The objective is the
    same clipped-softmax-NLL-plus-L2 function the analytic gradient
    differentiates, so at points where the clip is inactive (all test
    points) the two must agree to the finite-difference accuracy.

    Args:
        Xs: The (already-standardized) design matrix.
        y: The true outcome ordinals.
        intercepts: The intercept vector at which to differentiate.
        coefficients: The coefficient matrix at which to differentiate.
        l2_lambda: The L2 strength.
        eps: The finite-difference step size.

    Returns:
        A ``(3 + 3 * p,)`` numpy array of numerical gradient components
        in ``[intercepts, coefficients.ravel()]`` concatenation order.

    Raises:
        Nothing (the objective is total for finite inputs).
    """
    from models import multinomial_logit

    params = np.concatenate(
        [
            np.asarray(intercepts, dtype=float),
            np.asarray(coefficients, dtype=float).ravel(),
        ]
    )
    grad = np.empty(len(params))
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += eps
        minus = params.copy()
        minus[i] -= eps
        loss_plus = multinomial_logit._loss_and_gradient(
            Xs, y, plus[:3], plus[3:].reshape(3, -1), l2_lambda
        )[0]
        loss_minus = multinomial_logit._loss_and_gradient(
            Xs, y, minus[:3], minus[3:].reshape(3, -1), l2_lambda
        )[0]
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def test_analytic_gradients_match_finite_differences():
    # The single highest-risk correctness item: the analytic gradients
    # (w.r.t. both intercepts and the (3, 11) coefficient matrix) must
    # match a central finite-difference numerical gradient (eps=1e-6)
    # of the same batch objective, at multiple points — the
    # initialization point and two random points — not just at one
    # convenient location.
    from models import multinomial_logit

    rng = np.random.default_rng(7)
    Xs = rng.normal(size=(30, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=30)
    p = len(FEATURE_NAMES)
    points = [
        np.concatenate([np.zeros(3), np.zeros(3 * p)]),
        rng.normal(scale=0.4, size=3 + 3 * p),
        rng.normal(scale=0.7, size=3 + 3 * p),
    ]
    for params in points:
        intercepts = params[:3]
        coefficients = params[3:].reshape(3, p)
        _loss, g_int, g_coef = multinomial_logit._loss_and_gradient(
            Xs, y, intercepts, coefficients, 1.0
        )
        analytic = np.concatenate([g_int, g_coef.ravel()])
        numeric = _numeric_gradient(Xs, y, intercepts, coefficients, 1.0, 1e-6)
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_reference_class_fixed_at_zero():
    # Reference-class fixing: class 0 carries no parameters (its logit
    # is exactly zero by construction), so the four probabilities depend
    # only on the three free intercepts/coefficient rows. A model at the
    # initialization point (coefficients == 0, marginal-derived
    # intercepts) must predict exactly the training marginal
    # distribution.
    from models import multinomial_logit

    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=80)
    counts = np.bincount(y, minlength=4)
    model = fit(X, y, max_iter=50)
    # The free-parameter structure: coefficients (3, 11), intercepts (3,).
    assert model.coefficients.shape == (3, len(FEATURE_NAMES))
    assert model.intercepts.shape == (3,)
    # A hand-built model at the initialization point (coefficients == 0,
    # marginal intercepts, identity standardizer) must reproduce the
    # training marginal exactly: P_j == count[j] / n for every class.
    init_model = MultinomialLogitModel(
        coefficients=np.zeros((3, len(FEATURE_NAMES))),
        intercepts=multinomial_logit._initial_intercepts(counts),
        standardizer_means=np.zeros(len(FEATURE_NAMES)),
        standardizer_stds=np.ones(len(FEATURE_NAMES)),
        feature_names=FEATURE_NAMES,
        converged=True,
        n_iter=1,
        final_loss=0.0,
        n_train=len(y),
        l2_lambda=1.0,
    )
    probs = predict_proba(np.zeros(len(FEATURE_NAMES)), init_model)
    for j in range(4):
        assert probs[j] == pytest.approx(counts[j] / len(y), abs=1e-9)


def test_intercept_initialization_matches_label_marginal():
    # The label-marginal initialization formula: intercepts[m] ==
    # log(max(count[m+1], 1) / max(count[0], 1)); with zero coefficients
    # the softmax reproduces the training marginal exactly.
    from models import multinomial_logit

    counts = np.asarray([86, 15, 11, 97])
    intercepts = multinomial_logit._initial_intercepts(counts)
    for m in range(3):
        expected = math.log(max(counts[m + 1], 1) / max(counts[0], 1))
        assert intercepts[m] == pytest.approx(expected)
    # Zero counts (a missing category) must not produce -inf/nan.
    zero_counts = np.asarray([0, 5, 0, 9])
    intercepts_zero = multinomial_logit._initial_intercepts(zero_counts)
    assert np.all(np.isfinite(intercepts_zero))


def test_loss_trace_non_increasing_on_synthetic_batch():
    # The Armijo line search guarantees every accepted step decreases
    # the objective, so the returned iteration trace must be
    # non-increasing and end at final_loss.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=80)
    model = fit(X, y, max_iter=300)
    assert len(model.loss_trace) == model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(model.loss_trace))
    assert model.final_loss == pytest.approx(model.loss_trace[-1])


def test_predict_proba_returns_valid_simplex():
    # predict_proba must return a length-4 tuple summing to ~1 with
    # entries in [0, 1], for both a 1-D vector and a 1-row matrix.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=40)
    model = fit(X, y, max_iter=100)
    x = np.linspace(-1.0, 1.0, len(FEATURE_NAMES))
    probs = predict_proba(x, model)
    assert len(probs) == 4
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == pytest.approx(predict_proba(x.reshape(1, -1), model))


def test_fit_rejects_wrong_feature_count():
    # The model is defined over exactly len(FEATURE_NAMES) features; a
    # mismatched matrix is a hard error, not a silent misalignment.
    X = np.ones((10, 7))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="feature columns"):
        fit(X, y)


def test_fit_rejects_invalid_labels():
    # A label outside 0..3 cannot be scored and must fail loudly.
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.full(10, 7, dtype=int)
    with pytest.raises(ValueError, match="outcome ordinals"):
        fit(X, y)


def test_fit_rejects_row_count_mismatch():
    # X rows and y entries must line up.
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.zeros(5, dtype=int)
    with pytest.raises(ValueError, match="must match"):
        fit(X, y)


def _small_fitted_model():
    """Fit a small deterministic synthetic model for serialization tests.

    Uses a fixed seed so the fitted parameters are stable across runs;
    the model itself is only exercised for round-tripping, not for its
    predictive quality.

    Returns:
        A :class:`MultinomialLogitModel` fit on 40 synthetic rows.

    Raises:
        Nothing.
    """
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=40)
    return fit(X, y, max_iter=150)


def test_to_dict_from_dict_round_trip_and_json_serializable():
    # from_dict(to_dict(model)) must reproduce every serialized field,
    # and to_dict's output must be directly json.dumps-able (the
    # training driver writes it with json.dumps).
    model = _small_fitted_model()
    d = to_dict(model)
    serialized = json.dumps(d)
    restored = from_dict(json.loads(serialized))
    assert isinstance(restored, MultinomialLogitModel)
    assert restored.feature_names == model.feature_names == FEATURE_NAMES
    assert np.allclose(restored.coefficients, model.coefficients)
    assert np.allclose(restored.intercepts, model.intercepts)
    assert np.allclose(restored.standardizer_means, model.standardizer_means)
    assert np.allclose(restored.standardizer_stds, model.standardizer_stds)
    assert restored.l2_lambda == model.l2_lambda
    assert restored.converged == model.converged
    assert restored.n_iter == model.n_iter
    assert restored.final_loss == model.final_loss
    assert restored.n_train == model.n_train
    # The loss trace is a live-fit diagnostic, deliberately not
    # persisted: a deserialized model carries an empty trace.
    assert model.loss_trace
    assert restored.loss_trace == ()


def test_coefficient_report_flat_sorted_by_magnitude():
    # The multinomial coefficient report must be a flat list of 33
    # (feature, class) pairs sorted by |coefficient| descending, each
    # carrying the class label of the free class it belongs to.
    model = _small_fitted_model()
    report = to_dict(model)["coefficient_report"]
    assert len(report) == 3 * len(FEATURE_NAMES)
    assert [abs(e["coefficient"]) for e in report] == sorted(
        (abs(e["coefficient"]) for e in report), reverse=True
    )
    classes = {entry["class"] for entry in report}
    assert classes == set(OUTCOME_LABELS[1:])
    assert all(entry["feature"] in FEATURE_NAMES for entry in report)
    for entry in report:
        # The class-0 reference must never appear in the report.
        assert entry["class"] != OUTCOME_LABELS[0]


def test_from_dict_rejects_shape_mismatch():
    # A corrupt artifact (coefficients not matching feature count) must
    # fail loudly rather than deserialize into a misaligned model.
    d = to_dict(_small_fitted_model())
    d["coefficients"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="coefficients"):
        from_dict(d)


def test_make_model_fn_closure_matches_direct_predict():
    # The returned closure must (a) produce a length-4 simplex and (b)
    # agree exactly with the underlying pure call
    # predict_proba(build_feature_vector(...), model) for the same
    # inputs — no drift between the interface bridge and the pure path.
    model = _small_fitted_model()
    matches_df, maps_df, pms_df = _league_tables()
    model_fn = make_model_fn(model, pms_df)
    probs = model_fn("A", "B", "Bind", "2026-01-02T12:00:00", matches_df, maps_df)
    assert len(probs) == 4
    assert sum(probs) == pytest.approx(1.0)
    from models._shared import build_feature_vector

    x = build_feature_vector(
        "A", "B", "Bind", "2026-01-02T12:00:00", matches_df, maps_df, pms_df
    )
    direct = predict_proba(x, model)
    assert probs == pytest.approx(direct)


def test_total_log_likelihood_matches_manual_sum():
    # total_log_likelihood must equal the manual sum of log(P_y) over
    # the batch computed through predict_proba — the same formula,
    # checked independently so the public helper cannot drift.
    model = _small_fitted_model()
    rng = np.random.default_rng(21)
    X = rng.normal(size=(25, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=25)
    expected = sum(
        math.log(predict_proba(X[i], model)[y[i]]) for i in range(len(y))
    )
    assert total_log_likelihood(X, y, model) == pytest.approx(expected)


def test_evaluate_registry_has_multinomial_logit(tmp_path):
    # plan#6: MODEL_REGISTRY must now hold the multinomial_logit key
    # (so --model choices pick it up automatically), and the stateful
    # factory must raise FileNotFoundError on a missing artifact.
    # (The M24 ordinal_logit_temperature key was added by task 026 and
    # is asserted here too so the exact-set guard stays current.)
    assert set(evaluate.MODEL_REGISTRY) == {
        "four_way_baseline",
        "ordinal_logit",
        "ordinal_logit_temperature",
        "multinomial_logit",
    }
    model_fn = evaluate.MODEL_REGISTRY["four_way_baseline"](Path("data"), "v1")
    assert model_fn is evaluate.harness.four_way_baseline_model
    with pytest.raises(FileNotFoundError):
        evaluate.MODEL_REGISTRY["multinomial_logit"](tmp_path, "v1")


# --------------------------------------------------------------------------
# plan#12: real v1 monotonic loss (module fixture)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_v1_train_model():
    """Fit the model on the real v1 train split once per module.

    Assembles the ``(209, 15)`` training matrix exactly the way
    ``drivers/train_multinomial_logit.py`` does (held-out maps
    restricted to ``split="train"``, one feature vector per row, labels
    from ``outcome_ordinal``) and fits with the documented defaults.
    Shared by the real-data tests so the expensive feature assembly runs
    once.

    Returns:
        The fitted :class:`MultinomialLogitModel`.

    Raises:
        pytest.skip: If the real v1 tables are absent (decorated
            behaviour; the fixture body itself raises nothing).
    """
    if not _real_v1_available():
        pytest.skip("materialised v1 dataset not present (run materialize.py first)")
    from drivers import training_data

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")
    X, y = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="train",
    )
    return fit(X, y)


def test_real_v1_loss_trace_non_increasing(real_v1_train_model):
    # The real assembled v1 training set (209 maps) must also produce a
    # non-increasing iteration trace, ending at the reported final_loss.
    trace = real_v1_train_model.loss_trace
    assert len(trace) == real_v1_train_model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(trace))
    assert real_v1_train_model.final_loss == pytest.approx(trace[-1])


# --------------------------------------------------------------------------
# plan#12: real v1 end-to-end via the training CLI + harness scoring
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_train_and_score_end_to_end():
    # The full M21 loop against real data/v1: run the training CLI
    # (which assembles the 209-map train matrix, fits with the
    # documented defaults, and writes
    # data/v1/multinomial_logit_model.json), reload the artifact through
    # from_dict, wrap it with make_model_fn over the real
    # player_map_stats table, and score the real 35-map test split
    # through the evaluation harness directly (not by shelling out to
    # the evaluate CLI). Prints and records the resulting mean_rps /
    # mean_log_loss / marginal_binary_accuracy next to M20's ordinal
    # numbers (0.7238 / 0.9815 / 0.5429, from data/v1/
    # eval_report_ordinal_logit.json) and the M18 floor (0.7317 /
    # 0.9896 / 0.5143); beating the floor is not a pass/fail gate, but
    # the numbers must be reported.
    from drivers import train_multinomial_logit

    rc = train_multinomial_logit.main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/multinomial_logit_model.json")
    assert artifact_path.exists()
    model = from_dict(json.loads(artifact_path.read_text(encoding="utf-8")))
    assert model.n_train == 209

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")

    held_out = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    assert len(held_out) == 35
    model_fn = make_model_fn(model, player_map_stats_df)
    scored = harness.score_held_out_maps(
        model_fn, held_out, matches_df, maps_df
    )
    report = harness.build_evaluation_report(scored)

    # The numbers are printed (and recorded in the BUILD status line and
    # this task's commit message); the sanity bounds below mirror the
    # ordinal test's contract, not a floor-beating gate.
    print(
        "M21 multinomial-logit on real v1 test split (n_eval=35): "
        f"mean_rps={report['mean_rps']!r} "
        f"mean_log_loss={report['mean_log_loss']!r} "
        f"marginal_binary_accuracy={report['marginal_binary_accuracy']!r}"
    )
    assert report["n_eval"] == 35
    assert 0.0 <= report["mean_rps"] <= 3.0
    assert math.isfinite(report["mean_log_loss"])
    assert report["mean_log_loss"] > 0.0
    assert 0.0 <= report["marginal_binary_accuracy"] <= 1.0
    assert [entry["category"] for entry in report["calibration"]] == list(
        OUTCOME_LABELS
    )
    pred_matrix = scored[list(harness.PREDICTION_COLUMNS)].to_numpy()
    assert math.isfinite(pred_matrix.sum())
    assert (pred_matrix >= 0.0).all()
    assert all(
        abs(pred_matrix[i].sum() - 1.0) < 1e-9 for i in range(len(scored))
    )

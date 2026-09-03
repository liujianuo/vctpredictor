"""Tests for the plain binary logistic regression model (M22).

Covers the analytic-vs-finite-difference gradient check (the highest-
risk correctness item, for both the intercept and the coefficient
vector at multiple points, both with and without L2), the sign-
convention regression (a synthetic single-informative-feature dataset
must fit a positive coefficient when the feature is high exactly on
A-win rows), the label-marginal intercept initialization formula (both
the exact first-iteration loss on a signal-free matrix and the
converged intercept on it), the monotonic-loss-decrease property on
synthetic and real v1 data, the ``to_dict``/``from_dict`` round-trip
and JSON serializability, the ``make_model_fn`` closure-vs-direct-
predict agreement (and that its 2-tuple sums to 1), fit validation
errors, and a skip-guarded real-``data/v1`` end-to-end run that trains
via the CLI driver, reloads the artifact, and reports the fitted
coefficient report.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.binary_logit import (
    FEATURE_NAMES,
    BinaryLogitModel,
    fit,
    from_dict,
    make_model_fn,
    predict_proba,
    to_dict,
)
from tests._shared import _real_v1_available

_MATCHES_COLS = [
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "team1_name",
    "team2_name",
    "event_name",
    "status",
]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
    "team1_first_half_rounds",
    "team2_first_half_rounds",
    "team1_second_half_rounds",
    "team2_second_half_rounds",
    "team1_atk_rounds",
    "team1_def_rounds",
    "team2_atk_rounds",
    "team2_def_rounds",
]
_PMS_COLS = [
    "match_id",
    "map_index",
    "player_name",
    "team_name",
    "acs",
    "rating",
    "first_kills",
    "first_deaths",
]

QUERY_DATE = "2026-01-02T12:00:00"


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
            "date": QUERY_DATE,
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
    matches_df = pd.DataFrame(match_rows, columns=_MATCHES_COLS)
    maps_df = pd.DataFrame(map_rows, columns=_MAPS_COLS)
    pms_df = pd.DataFrame(pms_rows, columns=_PMS_COLS)
    return matches_df, maps_df, pms_df


def _binary_numeric_gradient(Xs, y, alpha, beta, l2_lambda, eps):
    """Central finite-difference gradient of the binary-logit objective.

    The independent numerical check the analytic gradient is tested
    against: for each parameter (the scalar ``alpha`` then the ``beta``
    vector), perturbs it by ``±eps``, evaluates the objective via
    :func:`models.binary_logit._loss_and_gradient`'s loss component,
    and takes the centered difference. The objective is the same
    clipped-NLL-plus-L2 function the analytic gradient differentiates,
    so at points where the clip is inactive (all test points) the two
    must agree to the finite-difference accuracy.

    Args:
        Xs: The (already-standardized) design matrix.
        y: The binary labels.
        alpha: The intercept at which to differentiate.
        beta: The coefficient vector at which to differentiate.
        l2_lambda: The L2 strength.
        eps: The finite-difference step size.

    Returns:
        A ``(p + 1,)`` numpy array of numerical gradient components in
        ``[alpha, beta]`` concatenation order.

    Raises:
        Nothing (the objective is total for finite inputs).
    """
    from models import binary_logit as bl

    params = np.concatenate([[alpha], np.asarray(beta, dtype=float)])
    grad = np.empty(len(params))
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += eps
        minus = params.copy()
        minus[i] -= eps
        loss_plus = bl._loss_and_gradient(Xs, y, plus[0], plus[1:], l2_lambda)[0]
        loss_minus = bl._loss_and_gradient(Xs, y, minus[0], minus[1:], l2_lambda)[0]
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


@pytest.fixture(scope="module")
def real_v1_train_model(real_v1_train_design_matrix):
    """Fit the binary model on the real v1 train split once per module.

    Derives the binary "A wins" target (``y_ordinal <= 1``) from the
    shared session-scoped design matrix and fits with the documented
    defaults. The expensive feature assembly (the five table reads plus
    the per-row :func:`build_feature_vector` loop) runs once per pytest
    session inside the ``real_v1_train_design_matrix`` fixture; this
    fixture only performs the cheap binarization and fit, cached per
    module for the real-data monotonic-loss test.

    Args:
        real_v1_train_design_matrix: The session-scoped fixture
            providing ``(X, y_ordinal, train_rows, matches_df, maps_df,
            player_map_stats_df)``; only ``X`` and ``y_ordinal`` are
            consumed here, both read-only (never mutate them in place).

    Returns:
        The fitted :class:`BinaryLogitModel`.

    Raises:
        pytest.skip: If the real v1 tables are absent (propagated from
            the session fixture's own skip guard; the fixture body
            itself raises nothing).
    """
    X, y_ordinal, _train_rows, _matches_df, _maps_df, _pms_df = (
        real_v1_train_design_matrix
    )
    y_binary = (y_ordinal <= 1).astype(int)
    return fit(X, y_binary)


# --------------------------------------------------------------------------
# plan#3: analytic vs finite-difference gradients (both with and without L2)
# --------------------------------------------------------------------------


def test_analytic_gradients_match_finite_differences():
    # The single highest-risk correctness item: the analytic gradient
    # (w.r.t. both the intercept and the coefficient vector) must match
    # a central finite-difference numerical gradient (eps=1e-6) of the
    # same batch objective, at multiple points and with L2 both on
    # (l2_lambda=1.0) and off (l2_lambda=0.0) — not just at one
    # convenient location.
    from models import binary_logit as bl

    rng = np.random.default_rng(7)
    Xs = rng.normal(size=(30, len(FEATURE_NAMES)))
    z = rng.integers(0, 2, size=30)
    points = [
        (0.0, np.zeros(len(FEATURE_NAMES))),
        (0.4, rng.normal(scale=0.4, size=len(FEATURE_NAMES))),
        (-0.7, rng.normal(scale=0.6, size=len(FEATURE_NAMES))),
    ]
    for l2_lambda in (1.0, 0.0):
        for alpha, beta in points:
            _loss, g_alpha, g_beta = bl._loss_and_gradient(
                Xs, z, alpha, beta, l2_lambda
            )
            analytic = np.concatenate([[g_alpha], g_beta])
            numeric = _binary_numeric_gradient(Xs, z, alpha, beta, l2_lambda, 1e-6)
            assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------
# plan#3: sign-convention regression
# --------------------------------------------------------------------------


def test_sign_convention_positive_coefficient_favors_team_a():
    # The sign-convention regression: a synthetic dataset where feature
    # 0 is strongly positive exactly on A-win rows (labels drawn from
    # the model family itself with beta_true[0] = 2.0) must fit a
    # POSITIVE coefficient for it, not negative — i.e. increasing the
    # feature increases p_a, "favors A". And it must be the dominant
    # coefficient: no noise feature should pick up a stronger signal.
    rng = np.random.default_rng(42)
    n = 250
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    beta_true = np.zeros(len(FEATURE_NAMES))
    beta_true[0] = 2.0
    alpha_true = -0.3
    logits = alpha_true + X @ beta_true
    p_a = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p_a).astype(int)
    model = fit(X, y, l2_lambda=0.1)
    assert model.coefficients[0] > 0.0
    assert model.coefficients[0] >= np.max(np.abs(model.coefficients[1:]))


# --------------------------------------------------------------------------
# plan#3: label-marginal intercept initialization
# --------------------------------------------------------------------------


def test_intercept_initialization_first_loss_reproduces_marginal_entropy():
    # The init formula alpha_0 = logit(clip(mean(y), eps, 1-eps)) must
    # hold exactly: on a signal-free design matrix (all-zero columns
    # standardize to 0.0 via the zero-variance guard), the very first
    # gradient-descent iteration's loss (max_iter=1, no L2) must equal
    # the Bernoulli entropy at the empirical label marginal p0 —
    # -[p0 log p0 + (1-p0) log(1-p0)] — since at (alpha_0, beta=0)
    # every row's p_a is exactly p0.
    rng = np.random.default_rng(21)
    n = 60
    X = np.zeros((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    p0 = float(np.mean(y))
    expected = -(p0 * math.log(p0) + (1.0 - p0) * math.log(1.0 - p0))
    model = fit(X, y, max_iter=1, l2_lambda=0.0)
    assert len(model.loss_trace) == 1
    assert model.loss_trace[0] == pytest.approx(expected)


def test_intercept_converges_to_marginal_logit_on_signal_free_data():
    # On a signal-free design matrix the optimum is the initialization
    # itself: coefficients must stay at ~0 and the fitted intercept must
    # equal logit(mean(y)) (the label-marginal logit), because no
    # feature can add any information.
    rng = np.random.default_rng(22)
    n = 200
    X = np.zeros((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    model = fit(X, y)
    p0 = float(np.mean(y))
    expected_intercept = math.log(p0 / (1.0 - p0))
    assert model.converged
    assert model.intercept == pytest.approx(expected_intercept, abs=1e-4)
    assert np.allclose(model.coefficients, 0.0, atol=1e-4)


# --------------------------------------------------------------------------
# plan#3: monotonic loss decrease on a synthetic batch
# --------------------------------------------------------------------------


def test_loss_trace_non_increasing_on_synthetic_batch():
    # The Armijo line search guarantees every accepted step decreases
    # the objective, so the returned iteration trace must be
    # non-increasing and end at final_loss.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=80)
    model = fit(X, y, max_iter=300)
    assert len(model.loss_trace) == model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(model.loss_trace))
    assert model.final_loss == pytest.approx(model.loss_trace[-1])


# --------------------------------------------------------------------------
# plan#3: fit validation
# --------------------------------------------------------------------------


def test_fit_rejects_wrong_feature_count():
    # The model is defined over exactly len(FEATURE_NAMES) features; a
    # mismatched matrix is a hard error, not a silent misalignment.
    X = np.ones((10, 7))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="feature columns"):
        fit(X, y)


def test_fit_rejects_invalid_labels():
    # A binary label outside {0, 1} cannot be scored and must fail
    # loudly (the fit contract takes already-binarized labels).
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.full(10, 7, dtype=int)
    with pytest.raises(ValueError, match="binary labels"):
        fit(X, y)


def test_fit_rejects_row_count_mismatch():
    # X rows and y entries must line up.
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.zeros(5, dtype=int)
    with pytest.raises(ValueError, match="must match"):
        fit(X, y)


# --------------------------------------------------------------------------
# plan#3: to_dict / from_dict round-trip and serializability
# --------------------------------------------------------------------------


def _small_fitted_model():
    """Fit a small deterministic synthetic model for serialization tests.

    Uses a fixed seed so the fitted parameters are stable across runs;
    the model itself is only exercised for round-tripping, not for its
    predictive quality.

    Returns:
        A :class:`BinaryLogitModel` fit on 40 synthetic rows.

    Raises:
        Nothing.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=40)
    return fit(X, y, max_iter=150)


def test_to_dict_from_dict_round_trip_and_json_serializable():
    # from_dict(to_dict(model)) must reproduce every serialized field,
    # and to_dict's output must be directly json.dumps-able (the
    # training driver writes it with json.dumps).
    model = _small_fitted_model()
    d = to_dict(model)
    serialized = json.dumps(d)
    restored = from_dict(json.loads(serialized))
    assert isinstance(restored, BinaryLogitModel)
    assert restored.feature_names == model.feature_names == FEATURE_NAMES
    assert restored.intercept == model.intercept
    assert np.allclose(restored.coefficients, model.coefficients)
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


def test_coefficient_report_sorted_by_magnitude_with_directions():
    # The report entries must be sorted by |coefficient| descending and
    # carry the documented sign-derived direction strings (the same
    # "favors A"/"favors B"/"favors neither" language as the ordinal
    # model).
    model = _small_fitted_model()
    report = to_dict(model)["coefficient_report"]
    assert [entry["feature"] for entry in report] == sorted(
        [entry["feature"] for entry in report],
        key=lambda name: abs(
            model.coefficients[list(FEATURE_NAMES).index(name)]
        ),
        reverse=True,
    )
    by_name = {entry["feature"]: entry for entry in report}
    for name, coefficient in zip(FEATURE_NAMES, model.coefficients):
        entry = by_name[name]
        assert entry["coefficient"] == pytest.approx(coefficient)
        if coefficient > 0.0:
            assert entry["direction"] == "favors A"
        elif coefficient < 0.0:
            assert entry["direction"] == "favors B"
        else:
            assert entry["direction"] == "favors neither"


def test_from_dict_rejects_shape_mismatch():
    # A corrupt artifact (coefficients not matching feature_names) must
    # fail loudly rather than deserialize into a misaligned model.
    d = to_dict(_small_fitted_model())
    d["coefficients"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="must match"):
        from_dict(d)


# --------------------------------------------------------------------------
# plan#3: make_model_fn closure vs direct predict
# --------------------------------------------------------------------------


def test_make_model_fn_closure_matches_direct_predict():
    # The returned closure must (a) produce a length-2 pair that sums to
    # 1 and (b) agree exactly with the underlying pure call
    # predict_proba(build_feature_vector(...), model) for the same
    # inputs — no drift between the interface bridge and the pure path.
    model = _small_fitted_model()
    matches_df, maps_df, pms_df = _league_tables()
    model_fn = make_model_fn(model, pms_df)
    probs = model_fn("A", "B", "Bind", QUERY_DATE, matches_df, maps_df)
    assert len(probs) == 2
    assert sum(probs) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in probs)
    from models._shared import build_feature_vector

    x = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    direct = predict_proba(x, model)
    assert probs == pytest.approx(direct)


def test_predict_proba_accepts_both_vector_shapes():
    # predict_proba must work for both a 1-D vector and a 1-row matrix
    # (the closure path always hands it a 1-D vector; the pure path
    # accepts either).
    model = _small_fitted_model()
    x = np.linspace(-1.0, 1.0, len(FEATURE_NAMES))
    assert predict_proba(x, model) == pytest.approx(
        predict_proba(x.reshape(1, -1), model)
    )


def test_predict_proba_returns_valid_simplex_for_extreme_inputs():
    # Even for extreme feature values the returned pair must stay
    # strictly inside (0, 1) (the clip keeps it directly scorable by
    # utils.scoring's log_loss, which raises on a zero probability on
    # the true category) and must sum to ~1.
    model = _small_fitted_model()
    x = np.full(len(FEATURE_NAMES), 1e3)
    p_a, p_b = predict_proba(x, model)
    assert p_a > 0.0 and p_b > 0.0
    assert p_a + p_b == pytest.approx(1.0)
    x = np.full(len(FEATURE_NAMES), -1e3)
    p_a, p_b = predict_proba(x, model)
    assert p_a > 0.0 and p_b > 0.0
    assert p_a + p_b == pytest.approx(1.0)


# --------------------------------------------------------------------------
# plan#3: real v1 monotonic loss (module fixture)
# --------------------------------------------------------------------------


def test_real_v1_loss_trace_non_increasing(real_v1_train_model):
    # The real assembled v1 training set (209 maps, binary "A wins"
    # target) must also produce a non-increasing iteration trace, ending
    # at the reported final_loss.
    trace = real_v1_train_model.loss_trace
    assert len(trace) == real_v1_train_model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(trace))
    assert real_v1_train_model.final_loss == pytest.approx(trace[-1])


# --------------------------------------------------------------------------
# plan#4: real v1 end-to-end via the training CLI
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_train_end_to_end():
    # The full M22 training loop against real data/v1: run the training
    # CLI (which assembles the 209-map train matrix, binarizes the
    # ordinal labels to the "A wins" target, fits with the documented
    # defaults, and writes data/v1/binary_logit_model.json), reload the
    # artifact through from_dict, and record the fitted diagnostics and
    # the top of the coefficient report. n_train must be 209 (the known
    # v1 train-split size); the binary marginal (A wins = ordinals
    # {0, 1}) is 101/209 positive per the M22 plan's verified context.
    from drivers import train_binary_logit

    rc = train_binary_logit.main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/binary_logit_model.json")
    assert artifact_path.exists()
    model = from_dict(json.loads(artifact_path.read_text(encoding="utf-8")))
    assert model.n_train == 209
    assert len(model.coefficients) == len(FEATURE_NAMES)
    report = to_dict(model)["coefficient_report"]
    assert len(report) == len(FEATURE_NAMES)
    print(
        "M22 binary-logit on real v1 train split (n_train=209): "
        f"converged={model.converged} n_iter={model.n_iter} "
        f"final_loss={model.final_loss!r} intercept={model.intercept!r} "
        f"top_coefficient={report[0]!r}"
    )

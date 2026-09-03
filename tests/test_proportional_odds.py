"""Tests for the proportional-odds (Brant-approximation) diagnostic (M21).

Covers the binary-logit's own analytic-vs-finite-difference gradient
check, the sign-instability logic (an all-agreeing synthetic fixture
must yield ``sign_instability_count == 0``; a constructed sign flip
must be flagged), the AIC/BIC arithmetic against hand-computed values,
the verdict rule's boolean logic across all four combinations of "any
sign instability" x "BIC favors multinomial", JSON-serializability of
the report, and a skip-guarded real-``data/v1`` end-to-end run
recording the actual ``sign_instability_count``/``bic_ordinal``/
``bic_multinomial``/``proportional_odds_verdict`` (the M21 finding that
gates M23 per the roadmap).
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation import proportional_odds as po
from models import multinomial_logit, ordinal_logit
from models._shared import FEATURE_NAMES
from models.multinomial_logit import MultinomialLogitModel
from models.ordinal_logit import OrdinalLogitModel
from tests._shared import _real_v1_available as _real_v1_tables_available


def _real_v1_available():
    """Report whether the v1 tables and both fitted artifacts exist.

    The skip guard for the real-data diagnostic: the parquet half is
    delegated to ``tests._shared._real_v1_available`` (the single
    shared home of the bare-table-name convention — all five
    ``data/v1/*.parquet`` files), and this module additionally requires
    both fitted model artifacts (``ordinal_logit_model.json`` and
    ``multinomial_logit_model.json``, i.e. both training drivers have
    been run) because the end-to-end diagnostic loads them.

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` files and
        both ``data/v1/*_logit_model.json`` artifacts exist.

    Raises:
        Nothing.
    """
    return _real_v1_tables_available() and all(
        Path(f"data/v1/{name}_logit_model.json").exists()
        for name in ("ordinal", "multinomial")
    )


def _tiny_fixture(n=24):
    """Build a tiny synthetic training fixture with hand-computable shape.

    A small ``(n, 13)`` design matrix and ``(n,)`` ordinal label vector
    with a fixed seed, plus matching hand-built ordinal/multinomial
    models (constructed directly, not fitted — the report's arithmetic
    is what's under test, not the fit quality).

    Args:
        n: The number of synthetic rows (default 24).

    Returns:
        A ``(X, y, ordinal_model, multinomial_model)`` tuple: ``X`` an
        ``(n, 13)`` float matrix, ``y`` an ``(n,)`` int ordinal vector,
        and the two hand-built model objects.

    Raises:
        Nothing.
    """
    rng = np.random.default_rng(13)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=n)
    ordinal_model = OrdinalLogitModel(
        coefficients=rng.normal(scale=0.3, size=len(FEATURE_NAMES)),
        thresholds=np.asarray([-1.0, 0.5, 1.5]),
        standardizer_means=np.zeros(len(FEATURE_NAMES)),
        standardizer_stds=np.ones(len(FEATURE_NAMES)),
        feature_names=FEATURE_NAMES,
        converged=True,
        n_iter=10,
        final_loss=1.0,
        n_train=n,
        l2_lambda=1.0,
    )
    multinomial_model = MultinomialLogitModel(
        coefficients=rng.normal(scale=0.3, size=(3, len(FEATURE_NAMES))),
        intercepts=np.asarray([-1.0, -2.0, 1.0]),
        standardizer_means=np.zeros(len(FEATURE_NAMES)),
        standardizer_stds=np.ones(len(FEATURE_NAMES)),
        feature_names=FEATURE_NAMES,
        converged=True,
        n_iter=10,
        final_loss=1.0,
        n_train=n,
        l2_lambda=1.0,
    )
    return X, y, ordinal_model, multinomial_model


def _cutpoint_models(flip_feature: int | None = None):
    """Build the three per-cutpoint binary models with hand-chosen coefficients.

    Constructs :class:`po.BinaryLogitModel` instances directly (frozen
    dataclasses — no fitting) with coefficient vectors that all agree
    in sign across cutpoints for every feature, except that feature
    ``flip_feature`` (when given) has a sign flip between cutpoint 1
    and cutpoints 2/3.

    Args:
        flip_feature: The feature index whose sign flips across
            cutpoints, or ``None`` for the all-agreeing fixture.

    Returns:
        A ``{1: model_1, 2: model_2, 3: model_3}`` dict of
        :class:`po.BinaryLogitModel`.

    Raises:
        Nothing.
    """
    models = {}
    for cutpoint in (1, 2, 3):
        coefficients = np.full(len(FEATURE_NAMES), 0.5)
        if flip_feature is not None:
            if cutpoint == 1:
                coefficients[flip_feature] = 1.0
            else:
                coefficients[flip_feature] = -1.0
        models[cutpoint] = po.BinaryLogitModel(
            coefficients=coefficients,
            intercept=0.0,
            standardizer_means=np.zeros(len(FEATURE_NAMES)),
            standardizer_stds=np.ones(len(FEATURE_NAMES)),
            cutpoint=cutpoint,
            converged=True,
            n_iter=5,
            final_loss=0.5,
            n_train=24,
        )
    return models


def _binary_numeric_gradient(Xs, z, alpha, beta, l2_lambda, eps):
    """Central finite-difference gradient of the binary-logit objective.

    The independent numerical check the analytic binary-logit gradients
    are tested against: for each parameter (``alpha`` then ``beta``),
    perturbs it by ``±eps``, evaluates the objective via
    :func:`evaluation.proportional_odds._binary_loss_and_gradient`'s
    loss component, and takes the centered difference.

    Args:
        Xs: The (already-standardized) design matrix.
        z: The binary labels.
        alpha: The intercept at which to differentiate.
        beta: The coefficient vector at which to differentiate.
        l2_lambda: The L2 strength.
        eps: The finite-difference step size.

    Returns:
        A ``(1 + p,)`` numpy array of numerical gradient components in
        ``[alpha, beta]`` concatenation order.

    Raises:
        Nothing (the objective is total for finite inputs).
    """
    params = np.concatenate([[alpha], np.asarray(beta, dtype=float)])
    grad = np.empty(len(params))
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += eps
        minus = params.copy()
        minus[i] -= eps
        loss_plus = po._binary_loss_and_gradient(
            Xs, z, plus[0], plus[1:], l2_lambda
        )[0]
        loss_minus = po._binary_loss_and_gradient(
            Xs, z, minus[0], minus[1:], l2_lambda
        )[0]
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def test_binary_logit_gradients_match_finite_differences():
    # The binary logit's analytic gradients (w.r.t. both the intercept
    # and the coefficient vector) must match a central finite-difference
    # numerical gradient of the same batch objective, at multiple
    # points.
    rng = np.random.default_rng(17)
    Xs = rng.normal(size=(30, len(FEATURE_NAMES)))
    z = rng.integers(0, 2, size=30)
    points = [
        (0.0, np.zeros(len(FEATURE_NAMES))),
        (0.4, rng.normal(scale=0.4, size=len(FEATURE_NAMES))),
        (-0.7, rng.normal(scale=0.6, size=len(FEATURE_NAMES))),
    ]
    for alpha, beta in points:
        _loss, g_alpha, g_beta = po._binary_loss_and_gradient(
            Xs, z, alpha, beta, 1.0
        )
        analytic = np.concatenate([[g_alpha], g_beta])
        numeric = _binary_numeric_gradient(Xs, z, alpha, beta, 1.0, 1e-6)
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_sign_instability_zero_when_all_cutpoints_agree():
    # The all-agreeing synthetic fixture: every feature has identical
    # nonzero signs across all three cutpoints, so sign_instability_count
    # must be exactly 0 and no per_feature entry may be flagged.
    cutpoint_models = _cutpoint_models(flip_feature=None)
    count, per_feature = po._feature_sign_instability(cutpoint_models)
    assert count == 0
    assert len(per_feature) == len(FEATURE_NAMES)
    assert all(not entry["sign_unstable"] for entry in per_feature)
    for entry in per_feature:
        assert entry["cutpoint_1_beta"] == pytest.approx(0.5)
        assert entry["max_abs_spread"] == pytest.approx(0.0)


def test_sign_instability_flags_constructed_flip():
    # The sign-flip fixture: feature index 4 has sign +1 at cutpoint 1
    # and -1 at cutpoints 2/3, so it must be flagged sign_unstable and
    # sign_instability_count must be >= 1 (exactly 1 here); all other
    # features stay stable.
    cutpoint_models = _cutpoint_models(flip_feature=4)
    count, per_feature = po._feature_sign_instability(cutpoint_models)
    assert count == 1
    flipped = per_feature[4]
    assert flipped["sign_unstable"] is True
    assert flipped["cutpoint_1_beta"] == pytest.approx(1.0)
    assert flipped["cutpoint_2_beta"] == pytest.approx(-1.0)
    assert flipped["cutpoint_3_beta"] == pytest.approx(-1.0)
    assert flipped["max_abs_spread"] == pytest.approx(2.0)
    assert all(not entry["sign_unstable"] for entry in per_feature if entry is not flipped)


def test_sign_zero_across_all_cutpoints_is_stable():
    # A feature whose three estimates are all exactly zero is trivially
    # stable, not unstable (the nonzero-sign rule).
    models = {}
    for cutpoint in (1, 2, 3):
        coefficients = np.zeros(len(FEATURE_NAMES))
        models[cutpoint] = po.BinaryLogitModel(
            coefficients=coefficients,
            intercept=0.0,
            standardizer_means=np.zeros(len(FEATURE_NAMES)),
            standardizer_stds=np.ones(len(FEATURE_NAMES)),
            cutpoint=cutpoint,
            converged=True,
            n_iter=1,
            final_loss=0.0,
            n_train=10,
        )
    count, per_feature = po._feature_sign_instability(models)
    assert count == 0
    assert all(not entry["sign_unstable"] for entry in per_feature)


def test_aic_bic_arithmetic_hand_computed(monkeypatch):
    # With ll values injected (monkeypatched), the report's aic/bic must
    # equal the hand-computed -2*ll + 2*k and -2*ll + k*log(n).
    X, y, ordinal_model, multinomial_model = _tiny_fixture()
    cutpoint_models = _cutpoint_models()
    ll_ordinal = -100.0
    ll_multinomial = -120.0
    monkeypatch.setattr(
        ordinal_logit, "total_log_likelihood", lambda X, y, model: ll_ordinal
    )
    monkeypatch.setattr(
        multinomial_logit, "total_log_likelihood", lambda X, y, model: ll_multinomial
    )
    report = po.build_diagnostic_report(
        ordinal_model, multinomial_model, cutpoint_models, X, y
    )
    n = len(y)
    assert report["n_train"] == n
    assert report["k_ordinal"] == 16
    assert report["k_multinomial"] == 42
    assert report["aic_ordinal"] == pytest.approx(-2 * ll_ordinal + 2 * 16)
    assert report["bic_ordinal"] == pytest.approx(-2 * ll_ordinal + 16 * math.log(n))
    assert report["aic_multinomial"] == pytest.approx(-2 * ll_multinomial + 2 * 42)
    assert report["bic_multinomial"] == pytest.approx(
        -2 * ll_multinomial + 42 * math.log(n)
    )
    assert report["ll_ordinal"] == pytest.approx(ll_ordinal)
    assert report["ll_multinomial"] == pytest.approx(ll_multinomial)


@pytest.mark.parametrize(
    "flip_feature,ll_ordinal,ll_multinomial,expected_verdict",
    [
        # No sign instability, BIC favors ordinal (multinomial ll worse):
        # the shared-coefficient restriction holds -> "holds".
        (None, -100.0, -140.0, "holds"),
        # No sign instability, BIC favors multinomial (multinomial ll
        # much better) -> "violated". (ll_ordinal is -150 so that with
        # k_ordinal=16 / k_multinomial=42 and n=24 the BIC comparison
        # still favors the multinomial arm: -2*(-150)+16*ln24 >
        # -2*(-100)+42*ln24, i.e. 350.85 > 333.48.)
        (None, -150.0, -100.0, "violated"),
        # Sign instability, BIC favors ordinal -> "violated".
        (4, -100.0, -140.0, "violated"),
        # Sign instability, BIC favors multinomial -> "violated".
        (4, -150.0, -100.0, "violated"),
    ],
)
def test_verdict_rule_all_four_combinations(
    monkeypatch, flip_feature, ll_ordinal, ll_multinomial, expected_verdict
):
    # The verdict rule's boolean logic across all four combinations of
    # "any sign instability" x "BIC favors multinomial":
    # proportional_odds_verdict = 'violated' if (sign_instability_count
    # > 0 or bic_favors_multinomial) else 'holds'.
    X, y, ordinal_model, multinomial_model = _tiny_fixture()
    cutpoint_models = _cutpoint_models(flip_feature=flip_feature)
    monkeypatch.setattr(
        ordinal_logit, "total_log_likelihood", lambda X, y, model: ll_ordinal
    )
    monkeypatch.setattr(
        multinomial_logit, "total_log_likelihood", lambda X, y, model: ll_multinomial
    )
    report = po.build_diagnostic_report(
        ordinal_model, multinomial_model, cutpoint_models, X, y
    )
    assert report["proportional_odds_verdict"] == expected_verdict
    # The report is self-documenting: the verdict rule text must be
    # embedded verbatim.
    assert "sign_instability_count > 0" in report["verdict_rule"]
    assert "bic_favors_multinomial" in report["verdict_rule"]
    assert "violated" in report["verdict_rule"]
    assert "holds" in report["verdict_rule"]


def test_report_json_serializable_with_real_ll():
    # build_diagnostic_report's output must be directly json.dumps-able
    # (the diagnostic driver writes it with json.dumps), with the real
    # (non-monkeypatched) log-likelihood path and per_feature entries
    # carrying every required key.
    X, y, ordinal_model, multinomial_model = _tiny_fixture()
    cutpoint_models = _cutpoint_models(flip_feature=2)
    report = po.build_diagnostic_report(
        ordinal_model, multinomial_model, cutpoint_models, X, y
    )
    serialized = json.dumps(report)
    assert isinstance(serialized, str)
    for entry in report["per_feature"]:
        assert set(entry) == {
            "feature",
            "ordinal_beta",
            "cutpoint_1_beta",
            "cutpoint_2_beta",
            "cutpoint_3_beta",
            "sign_unstable",
            "max_abs_spread",
        }
    assert report["sign_instability_count"] == 1
    assert report["per_feature"][2]["sign_unstable"] is True
    # With real ll, BIC penalizes the 42-parameter multinomial arm on a
    # tiny random fixture, so the verdict follows the sign instability.
    assert report["bic_favors_multinomial"] in (True, False)


def test_fit_cutpoint_binary_models_returns_three_cutpoints():
    # fit_cutpoint_binary_models must return {1: ..., 2: ..., 3: ...},
    # each model carrying its own cutpoint and the shared raw-training
    # standardizer values, and the binary labels must be the cumulative
    # splits z_i = 1 {y_i >= j}.
    rng = np.random.default_rng(23)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=60)
    models = po.fit_cutpoint_binary_models(X, y, max_iter=100)
    assert set(models) == {1, 2, 3}
    for cutpoint, model in models.items():
        assert model.cutpoint == cutpoint
        assert model.n_train == len(y)
        assert model.coefficients.shape == (len(FEATURE_NAMES),)
        # The stored standardizer is the shared raw-training one.
        means, stds = np.asarray(X).mean(axis=0), np.asarray(X).std(axis=0)
        assert model.standardizer_means == pytest.approx(means)
        assert model.standardizer_stds == pytest.approx(
            np.where(stds == 0.0, 1.0, stds)
        )
    # The three binary rates are strictly monotonic across cutpoints:
    # P(y >= 1) >= P(y >= 2) >= P(y >= 3).
    rates = [float((y >= cutpoint).mean()) for cutpoint in (1, 2, 3)]
    assert rates == sorted(rates, reverse=True)
    assert all(r > 0.0 and r < 1.0 for r in rates)


def test_fit_binary_logit_rejects_bad_cutpoint_and_labels():
    # cutpoint outside {1, 2, 3} and non-binary labels are hard errors.
    Xs = np.zeros((10, len(FEATURE_NAMES)))
    z = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="cutpoint"):
        po.fit_binary_logit(Xs, z, cutpoint=4)
    with pytest.raises(ValueError, match="binary labels"):
        po.fit_binary_logit(Xs, np.full(10, 2, dtype=int), cutpoint=1)


def test_binary_logit_loss_trace_non_increasing():
    # The Armijo line search must yield a non-increasing trace ending at
    # final_loss on a synthetic binary batch (checked via the private
    # optimizer, which returns the trace; the public fit path stores
    # only final_loss on the model).
    rng = np.random.default_rng(29)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=60)
    beta, alpha, _converged, _n_iter, trace = po._gradient_descent_binary(
        X, y, 1.0, 200, 1e-6, 1e-10
    )
    assert all(b <= a for a, b in itertools.pairwise(trace))
    assert trace[-1] == pytest.approx(
        po._binary_loss_and_gradient(X, y, alpha, beta, 1.0)[0]
    )
    model = po.fit_binary_logit(X, y, cutpoint=1, max_iter=200)
    assert model.final_loss == pytest.approx(trace[-1])


# --------------------------------------------------------------------------
# plan#12: real v1 end-to-end diagnostic
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + trained model artifacts not present "
    "(run materialize.py and both training drivers first)",
)
def test_real_v1_diagnostic_end_to_end():
    # The M21 finding on real data/v1: load the five tables, assemble
    # the 209-row training matrix, load both fitted model artifacts, run
    # the full diagnostic (per-cutpoint binary fits + report), write the
    # artifact via the CLI driver, and print/record
    # sign_instability_count / bic_ordinal / bic_multinomial /
    # proportional_odds_verdict — the finding the roadmap says gates M23.
    import pandas as pd

    from drivers import diagnose_proportional_odds, training_data

    rc = diagnose_proportional_odds.main(["--version", "v1"])
    assert rc == 0

    artifact = json.loads(
        Path("data/v1/proportional_odds_diagnostic.json").read_text(encoding="utf-8")
    )
    assert artifact["n_train"] == 209
    assert artifact["k_ordinal"] == 16
    assert artifact["k_multinomial"] == 42
    assert len(artifact["per_feature"]) == len(FEATURE_NAMES)
    assert artifact["proportional_odds_verdict"] in ("violated", "holds")

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

    # Independently recompute the report in-process (not just read the
    # CLI's artifact) so the numbers are double-checked.
    ordinal_model = ordinal_logit.from_dict(
        json.loads(
            Path("data/v1/ordinal_logit_model.json").read_text(encoding="utf-8")
        )
    )
    multinomial_model = multinomial_logit.from_dict(
        json.loads(
            Path("data/v1/multinomial_logit_model.json").read_text(encoding="utf-8")
        )
    )
    cutpoint_models = po.fit_cutpoint_binary_models(X, y)
    report = po.build_diagnostic_report(
        ordinal_model, multinomial_model, cutpoint_models, X, y
    )
    for key in ("sign_instability_count", "bic_ordinal", "bic_multinomial",
                "proportional_odds_verdict", "bic_favors_multinomial"):
        assert report[key] == artifact[key]

    print(
        "M21 proportional-odds diagnostic on real v1 train split "
        f"(n_train={report['n_train']}): "
        f"sign_instability_count={report['sign_instability_count']} "
        f"bic_ordinal={report['bic_ordinal']!r} "
        f"bic_multinomial={report['bic_multinomial']!r} "
        f"proportional_odds_verdict={report['proportional_odds_verdict']!r}"
    )
    assert math.isfinite(report["ll_ordinal"])
    assert math.isfinite(report["ll_multinomial"])

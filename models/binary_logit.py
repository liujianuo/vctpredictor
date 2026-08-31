"""Plain binary logistic regression on the "A wins" target (roadmap M22).

The fourth fitted model in the repository and the M22 granularity-
ablation arm: a plain sigmoid-link binary logistic regression fit
directly on the binary "does team1 (A) win the map" target
(``outcome_ordinal in {0, 1}`` — A-regulation or A-OT), over the
identical M13-M17 11-feature vector and identical M10 train/test split
as M20's ordinal logit (:mod:`models.ordinal_logit`). Its predictions
(a 2-vector ``(p_a, p_b)``) are compared against the ordinal model's
four-way output marginalized down to the same binary target by
:mod:`evaluation.granularity_ablation`, and that comparison is the
roadmap's "viability gate" for the four-way granularity: if the plain
binary model beats the marginalized ordinal model by a substantial,
documented margin on binary log loss and/or binary accuracy, the
four-way spec is judged to be "costing" accuracy.

Design decisions (recorded here, do not re-derive in later milestones):

- **Target and sign convention (must never be inverted).** The binary
  label is ``y_i = 1`` if team1 ("A") wins the map (``outcome_ordinal_i
  in {0, 1}``) else ``0`` (B wins). The link is ``eta_i = alpha +
  xs_i . beta`` — one scalar intercept ``alpha`` plus an 11-vector
  ``beta``, both fit, unlike ``ordinal_logit``'s "no intercept,
  thresholds serve as per-category intercepts" convention (a two-class
  model needs exactly one intercept) — with ``p_a_i = sigmoid(eta_i)``
  and ``p_b_i = 1 - p_a_i``. A *positive* ``beta_k`` increases ``eta``,
  increases ``p_a``, hence "favors A" — the identical direction
  convention ``ordinal_logit._coefficient_report`` already uses, so
  this module's own coefficient report speaks the same "favors A" /
  "favors B" / "favors neither" language.

- **The ``BinaryLogitModel`` name intentionally collides with
  ``evaluation.proportional_odds.BinaryLogitModel``.** The two are
  unrelated types in different packages serving different purposes:
  this one is the M22 primary binary model, persisted as its own
  artifact (``data/<version>/binary_logit_model.json``); that one is
  M21's private per-cutpoint diagnostic auxiliary fit (three
  independent models for the Brant sign-instability count), never
  persisted. The name collision is intentional-but-flagged, not a
  copy-paste mistake: "a fitted binary logistic regression" is
  genuinely the right name for both, and they live in non-colliding
  namespaces (``models.binary_logit.BinaryLogitModel`` vs
  ``evaluation.proportional_odds.BinaryLogitModel``). The M21
  diagnostic fit must NOT be reused here for three structural reasons:
  (a) ``models/`` cannot import from ``evaluation/`` — the DAG runs
  ``utils -> features -> models -> evaluation -> drivers``, evaluation
  depends downward on models, never the reverse, so this is
  structurally impossible, not just discouraged; (b) it is a private,
  diagnostic-scoped fit, not an exported reusable primary model; and
  (c) the established precedent in this codebase (recorded in task
  024's Design Decision A: "each of these fits is an independent
  implementation, not a shared driver, because each has a different
  parameter shape/loss/target") is that a fourth independent
  binary-logit implementation living in ``models/`` — sharing only
  :mod:`models._shared`'s generic feature/standardizer/validator/
  sigmoid infrastructure, exactly as the other three fitted models
  already do — is the pattern to follow, not a smell to avoid.

- **Loss and gradient.** Per-row NLL ``-[y * log(p_a) + (1 - y) *
  log(1 - p_a)]`` with ``p_a`` clipped into ``[_PROB_CLIP_EPS,
  1 - _PROB_CLIP_EPS]`` before any log (the same epsilon convention as
  every other model in this package). Batch objective ``mean(NLL) +
  (l2_lambda / 2) * sum(beta ** 2)`` — L2 on ``beta`` only, never on
  the intercept ``alpha`` (the "don't shrink the intercept" rule
  ``ordinal_logit`` applies to its thresholds and
  ``proportional_odds`` to its per-cutpoint ``alpha``). Gradient:
  ``d_eta_i = p_a_i - y_i`` (the exact derivative of the *unclipped*
  NLL; the clip epsilon is inactive for any realistic finite ``eta``),
  ``d(alpha) = mean(d_eta)``, ``d(beta) = mean(d_eta_i * xs_i) +
  l2_lambda * beta``. A dedicated test verifies the analytic gradient
  against central finite differences at multiple points — the
  non-negotiable correctness bar tasks 023/024 already set.

- **Initialization.** ``beta = zeros(11)``; ``alpha =
  logit(clip(mean(y), eps, 1 - eps))`` where ``eps`` is
  ``_PROB_CLIP_EPS`` — the intercept that reproduces the training label
  marginal at ``beta = 0``, the same role
  ``ordinal_logit._initial_raw_thresholds`` and ``multinomial_logit``'s
  label-marginal intercept init play for their own models.

- **Independent optimizer.** Full-batch gradient descent with Armijo
  backtracking line search, same structure/tolerances/defaults as the
  other three fitted models (``l2_lambda=1.0, max_iter=2000,
  grad_tol=1e-6, loss_tol=1e-10``), using ``_ARMIJO_C`` /
  ``_LINE_SEARCH_MAX_STEPS`` from :mod:`models._shared`. This is a
  fifth independent Armijo-loop implementation (ordinal, multinomial,
  and the three proportional-odds cutpoints already exist); per the
  task-024 precedent ("each of these fits is an independent
  implementation, not a shared driver"), no shared optimizer driver is
  extracted. A non-converged fit returns the best point found with
  ``converged=False`` (never raises), matching every sibling model's
  contract.

- **The caller passes the already-binarized ``y``.** :func:`fit`'s
  contract is a generic binary-label fit (labels in ``{0, 1}``); it
  has no notion of "outcome ordinal". The ordinal-to-binary conversion
  ``y_binary = (y_ordinal <= 1).astype(int)`` ("A wins" — ordinals 0
  and 1) happens in the training driver (:mod:`drivers.train_binary_logit`),
  keeping ``fit``'s contract symmetric with
  ``ordinal_logit.fit``/``multinomial_logit.fit``, which also take an
  already-prepared label vector.

- **Deliberately not a ``harness.ModelFn``.** The natural output of
  this model is a **2-vector** ``(p_a, p_b)``; it has no honest way to
  produce four category probabilities (regulation vs OT is not
  something a binary win/loss model predicts), and forcing one through
  the M19 harness's fixed 4-vector ``ModelFn`` contract would mean
  inventing a fake OT/regulation split with no basis — exactly the
  "fabricated data to satisfy an interface" this project's conventions
  warn against. This module is therefore deliberately *not* registered
  in ``drivers/evaluate.py``'s ``MODEL_REGISTRY`` and is never scored
  through the four-way harness; :func:`make_model_fn`'s closure
  returns a 2-tuple and is only ever called from
  :mod:`drivers.ablate_granularity`.

- **Standardization is its own, separate leakage boundary.** A
  per-feature ``(mean, std)`` z-score standardizer is fit *only on the
  assembled training design matrix* — never on test/held-out rows — via
  :func:`models._shared.fit_standardizer` /
  :func:`models._shared.apply_standardizer`, exactly as every other
  fitted model does (see :mod:`models._shared`'s docstring's
  standardization bullet for the leakage argument). A zero-variance
  training column standardizes to ``0.0`` for every row rather than
  dividing by zero.

- **Missing-value fallback policy.** Inherited unchanged from
  :mod:`models._shared`'s :func:`build_feature_vector` (per-side
  neutral fallbacks before the A-minus-B subtraction); this module adds
  no feature logic of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models._shared import (
    _ARMIJO_C,
    _LINE_SEARCH_MAX_STEPS,
    _PROB_CLIP_EPS,
    FEATURE_NAMES,
    _sigmoid,
    _validate_l2_lambda,
    _validate_positive_float,
    _validate_positive_int,
    apply_standardizer,
    build_feature_vector,
    fit_standardizer,
)

# The module's public API, declared explicitly so the linter treats the
# names re-exported from models._shared (FEATURE_NAMES et al., which
# tests import via ``from models.binary_logit import ...`` and must
# therefore keep resolving here) as intentional re-exports rather than
# unused imports. Mirrors the identical convention in
# ``models/ordinal_logit.py``'s ``__all__``.
__all__ = (
    "FEATURE_NAMES",
    "BinaryLogitModel",
    "apply_standardizer",
    "build_feature_vector",
    "fit",
    "fit_standardizer",
    "from_dict",
    "make_model_fn",
    "predict_proba",
    "to_dict",
)


def _loss_and_gradient(
    Xs: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    l2_lambda: float,
) -> tuple[float, float, np.ndarray]:
    """Return the binary-logit batch objective and its analytic gradient.

    Computes the full-batch objective ``mean(NLL) + (l2_lambda / 2) *
    sum(beta ** 2)`` where the per-row NLL is the Bernoulli
    cross-entropy ``-(y * log(p_a) + (1 - y) * log(1 - p_a))`` with
    ``p_a = sigmoid(alpha + xs . beta)`` clipped into
    ``[eps, 1 - eps]`` before the log (the clip is the same defensive
    floor/sky coverage as every other model in this package; it is
    inactive for any realistic finite ``eta``). The per-row gradient of
    the NLL w.r.t. ``eta`` is ``p_a - y`` (the exact derivative of the
    unclipped NLL), so ``d(alpha) = mean(p_a - y)`` and ``d(beta) =
    mean((p_a - y) * xs) + l2_lambda * beta`` — the L2 term on ``beta``
    only, never on the intercept (see the module docstring's loss
    bullet).

    Args:
        Xs: The (already-standardized) design matrix, ``(n, p)``
            floats.
        y: The binary labels, ``(n,)`` ints in ``{0, 1}``.
        alpha: The current scalar intercept.
        beta: The current coefficient vector, length ``p``.
        l2_lambda: The L2 regularization strength on ``beta`` only
            (non-negative finite float).

    Returns:
        A ``(loss, grad_alpha, grad_beta)`` tuple: the scalar batch
        objective, its gradient w.r.t. ``alpha``, and its gradient
        w.r.t. ``beta`` (including the L2 term).

    Raises:
        ValueError: If ``Xs`` rows and ``len(y)`` differ, or if ``y``
            contains a value outside ``{0, 1}``.
    """
    n = Xs.shape[0]
    if n != len(y):
        raise ValueError(
            f"Xs has {n} rows but y has {len(y)} entries; they must match"
        )
    total_nll = 0.0
    grad_alpha = 0.0
    grad_beta = np.zeros_like(beta, dtype=float)
    for i in range(n):
        eta = alpha + float(np.dot(Xs[i], beta))
        p = _sigmoid(eta)
        p_clipped = min(max(p, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
        total_nll += -(
            y[i] * math.log(p_clipped)
            + (1 - y[i]) * math.log(1.0 - p_clipped)
        )
        d_eta = p - y[i]
        grad_alpha += d_eta
        grad_beta += d_eta * Xs[i]
    nll = total_nll / n
    grad_alpha /= n
    grad_beta /= n
    # L2 on beta only (the intercept is not shrunk); the penalty is NOT
    # averaged over the batch, matching every sibling model.
    l2_penalty = (l2_lambda / 2.0) * float(np.sum(beta**2))
    grad_beta += l2_lambda * beta
    return nll + l2_penalty, grad_alpha, grad_beta


def _gradient_descent(
    Xs: np.ndarray,
    y: np.ndarray,
    l2_lambda: float,
    max_iter: int,
    grad_tol: float,
    loss_tol: float,
) -> tuple[np.ndarray, float, bool, int, tuple[float, ...]]:
    """Run full-batch gradient descent with Armijo backtracking.

    The binary logit's own optimizer (a fifth independent Armijo-loop
    implementation, per the module docstring's no-shared-optimizer
    rule). Starts from ``beta = 0`` and ``alpha =
    logit(clip(mean(y), eps, 1 - eps))`` — the intercept reproducing
    the training label marginal at ``beta = 0`` (see the module
    docstring's initialization bullet) — then iterates: compute the
    loss/gradient at the current point; stop (converged) if the
    combined gradient norm over ``(alpha, beta)`` is below ``grad_tol``
    or the loss improvement between iterations drops below ``loss_tol``;
    otherwise try step size ``1.0``, halving up to
    :data:`_LINE_SEARCH_MAX_STEPS` times until the Armijo
    sufficient-decrease condition
    (``loss(p - step*grad) <= loss(p) - _ARMIJO_C * step * ||grad||^2``)
    holds, then take that step. If the line search cannot find any
    acceptable step, or if ``max_iter`` is hit, the run stops with
    ``converged=False`` and returns the best point found — a
    non-converged fit is a valid (if suboptimal) model, not an error.
    The returned loss trace is non-increasing by construction (every
    accepted step satisfies Armijo).

    Args:
        Xs: The already-standardized design matrix, ``(n, p)`` floats.
        y: The binary labels, ``(n,)`` ints in ``{0, 1}``.
        l2_lambda: The L2 strength (validated non-negative finite).
        max_iter: The iteration cap (validated positive int).
        grad_tol: The gradient-norm convergence tolerance.
        loss_tol: The loss-improvement convergence tolerance.

    Returns:
        A ``(best_beta, best_alpha, converged, n_iter, loss_trace)``
        tuple: the best coefficient vector, the best intercept, whether
        the run converged, the number of iterations actually executed,
        and the per-iteration loss trace (non-increasing, length
        ``n_iter``).

    Raises:
        ValueError: If ``y`` is empty or contains a value outside
            ``{0, 1}``, or if the shapes are inconsistent (propagated
            from :func:`_loss_and_gradient`).
    """
    if len(y) == 0:
        raise ValueError("cannot run gradient descent on an empty label vector")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(
            f"y must contain only binary labels 0/1, got values "
            f"{sorted(set(np.unique(y).tolist()))}"
        )
    beta = np.zeros(Xs.shape[1], dtype=float)
    mean_y = float(np.mean(y))
    p0 = min(max(mean_y, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
    alpha = math.log(p0 / (1.0 - p0))

    best_beta = beta.copy()
    best_alpha = alpha
    best_loss = float("inf")
    loss_trace: list[float] = []
    prev_loss: float | None = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        loss, grad_alpha, grad_beta = _loss_and_gradient(
            Xs, y, alpha, beta, l2_lambda
        )
        n_iter = iteration + 1
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
            best_alpha = alpha
        loss_trace.append(loss)

        grad_norm = math.sqrt(
            float(np.sum(grad_beta**2)) + grad_alpha * grad_alpha
        )
        if grad_norm < grad_tol:
            converged = True
            break
        if prev_loss is not None and (prev_loss - loss) < loss_tol:
            converged = True
            break

        step = 1.0
        accepted = False
        for _ in range(_LINE_SEARCH_MAX_STEPS):
            trial_beta = beta - step * grad_beta
            trial_alpha = alpha - step * grad_alpha
            trial_loss, _, _ = _loss_and_gradient(
                Xs, y, trial_alpha, trial_beta, l2_lambda
            )
            if trial_loss <= loss - _ARMIJO_C * step * grad_norm**2:
                beta = trial_beta
                alpha = trial_alpha
                accepted = True
                break
            step *= 0.5
        if not accepted:
            # No step size down to step == 2^-50 satisfies Armijo: the
            # current point is the best we can reach; report as
            # non-converged rather than raising.
            converged = False
            break
        prev_loss = loss
    else:
        # The for-loop exhausted max_iter without breaking: the cap was
        # hit without convergence; return the best point found.
        converged = False

    return best_beta, best_alpha, converged, n_iter, tuple(loss_trace)


@dataclass(frozen=True)
class BinaryLogitModel:
    """A fitted plain binary logistic regression on the "A wins" target.

    Holds the fitted parameters and the diagnostics needed to (a) make
    predictions via :func:`predict_proba`, (b) serialize/deserialize
    via :func:`to_dict` / :func:`from_dict`, and (c) interpret the fit
    via the coefficient report. ``coefficients`` has one entry per
    feature in ``feature_names`` (the :data:`FEATURE_NAMES` order); the
    sign convention is documented in the module docstring (positive
    favors team A). ``intercept`` is the scalar bias term (the only
    intercept-like parameter — unlike ``ordinal_logit``'s threshold
    triple, a two-class model needs exactly one). ``standardizer_means``
    / ``standardizer_stds`` describe the *training* design matrix (the
    second leakage boundary; see the module docstring). ``loss_trace``
    is a live-fit diagnostic (per-iteration loss, non-increasing) that
    is deliberately *not* persisted by :func:`to_dict` — a deserialized
    model carries an empty trace.

    Note: this type deliberately shares its name with
    ``evaluation.proportional_odds.BinaryLogitModel``; the two are
    unrelated (see the module docstring's name-collision bullet).

    Attributes:
        intercept: The fitted scalar intercept ``alpha``.
        coefficients: The 11-vector of fitted coefficients ``beta``.
        standardizer_means: Per-feature training-column means (length
            11).
        standardizer_stds: Per-feature training-column stds (length 11;
            a zero-variance column's std is ``1.0`` per the guard).
        feature_names: The feature name tuple (:data:`FEATURE_NAMES`).
        converged: Whether gradient descent converged (``True``) or hit
            ``max_iter``/line-search failure (``False`` — still a
            valid, if suboptimal, model).
        n_iter: Number of gradient-descent iterations executed.
        final_loss: The objective value at the returned point.
        n_train: Number of training rows the model was fit on.
        l2_lambda: The L2 strength used for this fit (stored so the
            artifact records its own regularization).
        loss_trace: The per-iteration objective trace (non-increasing,
            length ``n_iter``); ``()`` for a deserialized model.
    """

    intercept: float
    coefficients: np.ndarray
    standardizer_means: np.ndarray
    standardizer_stds: np.ndarray
    feature_names: tuple[str, ...]
    converged: bool
    n_iter: int
    final_loss: float
    n_train: int
    l2_lambda: float
    loss_trace: tuple[float, ...] = ()


def fit(
    X: np.ndarray,
    y: np.ndarray,
    l2_lambda: float = 1.0,
    max_iter: int = 2000,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-10,
) -> BinaryLogitModel:
    """Fit the binary logistic regression by Armijo gradient descent.

    Assembles a complete :class:`BinaryLogitModel` from a raw feature
    matrix and an already-binarized label vector: fits the per-feature
    z-score standardizer on the *training* matrix only (the second
    leakage boundary; see :func:`models._shared.fit_standardizer`) and
    transforms with it, then runs full-batch gradient descent with
    Armijo backtracking (see :func:`_gradient_descent`) initialized at
    ``beta = 0`` and the label-marginal intercept. The returned model
    carries the training standardizer so :func:`predict_proba` can
    standardize later rows with the exact training-population
    statistics.

    Args:
        X: The raw (unstandardized) training design matrix, ``(n, 11)``
            floats in :data:`FEATURE_NAMES` order — the output of
            :func:`build_feature_vector` over the training rows. The
            standardizer is fit on this matrix inside this function.
        y: The *already-binarized* true labels, ``(n,)`` ints in
            ``{0, 1}`` (``1`` = "team1/A wins", ``0`` = "team2/B
            wins"). This function itself has no notion of "outcome
            ordinal"; the ordinal-to-binary conversion
            ``(y_ordinal <= 1).astype(int)`` happens in the training
            driver, keeping this contract symmetric with the sibling
            models' fits.
        l2_lambda: L2 regularization strength on ``beta`` only; must be
            non-negative finite (default ``1.0`` — kept identical to
            the other three fitted models for comparability).
        max_iter: Cap on gradient-descent iterations (default 2000). If
            the cap is hit without convergence, ``fit`` returns the best
            point found with ``converged=False`` rather than raising.
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).

    Returns:
        A frozen :class:`BinaryLogitModel` with the fitted
        ``intercept``/``coefficients``, the training standardizer, the
        diagnostics (``converged``/``n_iter``/``final_loss``/
        ``n_train``/``l2_lambda``/``loss_trace``), and
        ``feature_names = FEATURE_NAMES``.

    Raises:
        ValueError: If ``X`` is not a 2-D array, ``y`` is not 1-D,
            their row counts differ, ``X`` does not have exactly
            ``len(FEATURE_NAMES)`` columns, ``y`` is empty or contains
            a value outside ``{0, 1}``, or any hyperparameter is
            invalid (see the validate helpers).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if X_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2-D design matrix, got {X_arr.ndim} dimension(s)"
        )
    if y_arr.ndim != 1:
        raise ValueError(
            f"y must be a 1-D label vector, got {y_arr.ndim} dimension(s)"
        )
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} "
            "entries; they must match"
        )
    if X_arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have exactly {len(FEATURE_NAMES)} feature columns "
            f"(one per {FEATURE_NAMES} entry), got {X_arr.shape[1]}"
        )
    if y_arr.size == 0:
        raise ValueError("cannot fit a binary model on an empty label vector")
    if set(np.unique(y_arr).tolist()) - {0, 1}:
        raise ValueError(
            f"y must contain only binary labels 0/1, got values "
            f"{sorted(set(np.unique(y_arr).tolist()))}"
        )

    l2 = _validate_l2_lambda(l2_lambda)
    max_it = _validate_positive_int(max_iter, "max_iter")
    g_tol = _validate_positive_float(grad_tol, "grad_tol")
    l_tol = _validate_positive_float(loss_tol, "loss_tol")

    means, stds = fit_standardizer(X_arr)
    Xs = apply_standardizer(X_arr, means, stds)
    beta, alpha, converged, n_iter, trace = _gradient_descent(
        Xs, y_arr, l2, max_it, g_tol, l_tol
    )
    return BinaryLogitModel(
        intercept=alpha,
        coefficients=beta,
        standardizer_means=means,
        standardizer_stds=stds,
        feature_names=FEATURE_NAMES,
        converged=converged,
        n_iter=n_iter,
        final_loss=trace[-1],
        n_train=len(y_arr),
        l2_lambda=l2,
        loss_trace=trace,
    )


def predict_proba(
    x: np.ndarray,
    model: BinaryLogitModel,
) -> tuple[float, float]:
    """Predict the two binary probabilities for one feature vector.

    Applies the model's stored standardizer (training-population means/
    stds — the second leakage boundary, applied identically to every
    later row), computes ``eta = alpha + x_standardized . beta``, and
    returns ``(p_a, p_b) = (clip(sigmoid(eta)), 1 - clip(sigmoid(eta)))``
    — A first, matching every other model's "A-side probabilities
    first" convention in this codebase. The clip into
    ``[eps, 1 - eps]`` keeps the returned pair strictly positive so it
    is directly scorable by ``utils.scoring``'s ``log_loss`` (which
    raises on a zero probability on the true category) even for extreme
    inputs.

    Args:
        x: A raw feature vector, length 11 in :data:`FEATURE_NAMES`
            order (the output of :func:`build_feature_vector`).
        model: The fitted model whose stored standardizer and
            coefficients are applied.

    Returns:
        The 2-tuple ``(p_a, p_b)`` of ``float`` probabilities (A first:
        index 0 is "team1/A wins", index 1 is "team2/B wins"), each in
        ``[eps, 1 - eps]`` and summing to approximately 1.

    Raises:
        ValueError: If ``x`` does not have exactly as many entries as
            the model has coefficients (a feature-vector/model mismatch
            would silently misalign).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.shape[0] != len(model.coefficients):
        raise ValueError(
            f"feature vector has {x_arr.shape[0]} entries but the model "
            f"has {len(model.coefficients)} coefficients; the feature "
            "vector must match the model's feature_names order"
        )
    # Reshape to a 1-row design matrix: apply_standardizer's contract is
    # 2-D matrices; the prediction path hands it a single feature vector.
    xs = apply_standardizer(
        x_arr.reshape(1, -1),
        model.standardizer_means,
        model.standardizer_stds,
    )[0]
    eta = model.intercept + float(np.dot(xs, model.coefficients))
    p_a = min(max(_sigmoid(eta), _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
    return float(p_a), float(1.0 - p_a)


def _coefficient_report(model: BinaryLogitModel) -> list[dict]:
    """Build the human-readable coefficient report for a fitted model.

    One entry per feature: ``{"feature": name, "coefficient": value,
    "direction": label}`` where ``direction`` is derived from the
    coefficient's sign under the module-docstring convention — a
    positive coefficient favors team A (increasing the feature raises
    ``p_a``), a negative one favors team B, and an exactly-zero one
    favors neither. Entries are sorted by ``abs(coefficient)``
    descending so the most influential features read first. Same shape
    and sort as ``ordinal_logit._coefficient_report``; the binary
    model's coefficients speak the identical "favors A"/"favors B"
    language because the two models share the same A-minus-B feature
    convention.

    Args:
        model: The fitted model whose ``coefficients`` and
            ``feature_names`` are reported.

    Returns:
        A list of dicts, one per feature, sorted by descending
        ``abs(coefficient)``.

    Raises:
        Nothing (the model's own shape validation guarantees the two
            arrays line up).
    """
    entries = []
    for name, coefficient in zip(model.feature_names, model.coefficients):
        if coefficient > 0.0:
            direction = "favors A"
        elif coefficient < 0.0:
            direction = "favors B"
        else:
            direction = "favors neither"
        entries.append(
            {
                "feature": name,
                "coefficient": float(coefficient),
                "direction": direction,
            }
        )
    return sorted(entries, key=lambda entry: abs(entry["coefficient"]), reverse=True)


def to_dict(model: BinaryLogitModel) -> dict:
    """Serialize a fitted model to a plain JSON-serializable dict.

    Produces the artifact dict the training driver writes:
    ``feature_names``, ``coefficients``, ``intercept`` (this model's one
    new scalar key relative to ``ordinal_logit.to_dict``),
    ``standardizer_means``, ``standardizer_stds``, ``l2_lambda``,
    ``converged``, ``n_iter``, ``final_loss``, ``n_train``, plus a
    ``coefficient_report`` list (from :func:`_coefficient_report`).
    Every value is a plain str/int/float/list so ``json.dumps`` accepts
    the dict directly. The ``loss_trace`` diagnostic is deliberately
    *not* persisted (it is a live-fit trace, not model parameters). No
    file I/O happens here.

    Args:
        model: The fitted model to serialize.

    Returns:
        A plain dict as described, directly ``json.dumps``-serializable
        (with ``sort_keys=True`` for deterministic artifacts).

    Raises:
        Nothing.
    """
    return {
        "feature_names": list(model.feature_names),
        "coefficients": [float(c) for c in model.coefficients],
        "intercept": float(model.intercept),
        "standardizer_means": [float(m) for m in model.standardizer_means],
        "standardizer_stds": [float(s) for s in model.standardizer_stds],
        "l2_lambda": float(model.l2_lambda),
        "converged": bool(model.converged),
        "n_iter": int(model.n_iter),
        "final_loss": float(model.final_loss),
        "n_train": int(model.n_train),
        "coefficient_report": _coefficient_report(model),
    }


def from_dict(d: dict) -> BinaryLogitModel:
    """Deserialize a fitted model from a to_dict-produced dict.

    Reconstructs a :class:`BinaryLogitModel` from the plain dict
    :func:`to_dict` produces (or from ``json.loads`` of the artifact
    the training driver writes). Arrays are rebuilt as numpy arrays;
    shape consistency is validated (coefficients/means/stds must all
    line up with ``feature_names``). The ``coefficient_report`` key is
    ignored on read (it is derived, not stored) and ``loss_trace`` is
    empty for a deserialized model. No file I/O happens here.

    Args:
        d: The dict to load; must carry the parameter/diagnostic keys
            (``coefficient_report`` optional, ignored).

    Returns:
        A :class:`BinaryLogitModel` whose parameters reproduce the
        serialized ones exactly (``feature_names`` as a tuple, arrays
        as ``float`` numpy arrays, diagnostics as plain scalars).

    Raises:
        KeyError: If a required key is absent (propagated from dict
            indexing).
        ValueError: If the shapes are inconsistent (coefficient count
            != feature count, or means/stds length != coefficient
            count), or if a numeric field cannot be coerced.
    """
    feature_names = tuple(str(name) for name in d["feature_names"])
    coefficients = np.asarray(d["coefficients"], dtype=float)
    means = np.asarray(d["standardizer_means"], dtype=float)
    stds = np.asarray(d["standardizer_stds"], dtype=float)
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"coefficients has {len(coefficients)}; they must match"
        )
    if len(means) != len(coefficients) or len(stds) != len(coefficients):
        raise ValueError(
            f"standardizer means ({len(means)}) / stds ({len(stds)}) must "
            f"each have one entry per coefficient ({len(coefficients)})"
        )
    return BinaryLogitModel(
        intercept=float(d["intercept"]),
        coefficients=coefficients,
        standardizer_means=means,
        standardizer_stds=stds,
        feature_names=feature_names,
        converged=bool(d["converged"]),
        n_iter=int(d["n_iter"]),
        final_loss=float(d["final_loss"]),
        n_train=int(d["n_train"]),
        l2_lambda=float(d["l2_lambda"]),
    )


def make_model_fn(
    model: BinaryLogitModel,
    player_map_stats_df: pd.DataFrame,
) -> Callable[[str, str, str, str, pd.DataFrame, pd.DataFrame], tuple[float, float]]:
    """Wrap a fitted model into the 6-argument interface shape (2-tuple).

    Bridges the fixed generic model-interface calling convention —
    ``(team1_id, team2_id, map_name, date, matches_df, maps_df)`` — to
    a feature builder that needs a *seventh* table
    (``player_map_stats_df``) by closing that table over at load time.
    The returned closure has exactly the 6-argument shape
    (structurally, without importing the harness), calls
    :func:`build_feature_vector` with the closed-over table, and
    returns :func:`predict_proba`'s 2-tuple.

    **Explicitly NOT a ``harness.ModelFn``.** :data:`evaluation.harness.ModelFn`
    is fixed at returning a 4-vector in ``OUTCOME_LABELS`` order; this
    closure returns a 2-vector ``(p_a, p_b)`` because a binary win/loss
    model has no honest four-way output. It is never registered in
    ``drivers/evaluate.py``'s ``MODEL_REGISTRY`` (see the module
    docstring's not-a-ModelFn bullet) and is only ever called from
    :mod:`drivers.ablate_granularity`.

    As with every sibling model, this is only correct because callers
    always invoke the returned closure with the same ``matches_df`` /
    ``maps_df`` that came from the same ``<output_dir>/<version>`` the
    closed-over ``player_map_stats_df`` was loaded from — a mismatched
    pairing would silently misalign.

    Args:
        model: The fitted model to predict with.
        player_map_stats_df: The materialised ``player_map_stats`` table
            for the same dataset version as the ``matches_df``/``maps_df``
            the closure will be invoked with.

    Returns:
        A closure ``(team1_id, team2_id, map_name, date, matches_df,
        maps_df) -> (p_a, p_b)`` (a 2-tuple of floats, A first, summing
        to approximately 1).

    Raises:
        ValueError: If the feature vector length mismatches the model
            (propagated from :func:`predict_proba`), or if any feature
            computation fails (propagated from
            :func:`build_feature_vector`).
        KeyError: If any table lacks a required column (propagated from
            :func:`build_feature_vector`).
    """

    def model_fn(
        team1_id: str,
        team2_id: str,
        map_name: str,
        date: str,
        matches_df: pd.DataFrame,
        maps_df: pd.DataFrame,
    ) -> tuple[float, float]:
        """Predict the two binary probabilities for one held-out map.

        Computes the raw 11-feature vector via
        :func:`build_feature_vector` (using the closed-over
        ``player_map_stats_df`` — the table the generic interface does
        not pass) and returns :func:`predict_proba`'s 2-tuple. See
        :func:`make_model_fn`'s docstring for the closed-table contract
        and the explicit not-a-``harness.ModelFn`` note.

        Args:
            team1_id: The queried team1's stable id ("A").
            team2_id: The queried team2's stable id ("B").
            map_name: The map to predict for.
            date: The as-of cutoff (the map's own match date).
            matches_df: The full materialised ``matches`` table from the
                same dataset version the closed-over
                ``player_map_stats_df`` was loaded from.
            maps_df: The full materialised ``maps`` table from the same
                version.

        Returns:
            The 2-tuple ``(p_a, p_b)`` of probabilities, A first,
            summing to approximately 1.

        Raises:
            ValueError: If the feature vector length mismatches the
                model, or if any feature computation fails (propagated
                from :func:`predict_proba` / :func:`build_feature_vector`).
            KeyError: If any table lacks a required column (propagated
                from :func:`build_feature_vector`).
        """
        x = build_feature_vector(
            team1_id,
            team2_id,
            map_name,
            date,
            matches_df,
            maps_df,
            player_map_stats_df,
        )
        return predict_proba(x, model)

    return model_fn

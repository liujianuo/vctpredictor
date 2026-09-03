"""Baseline-category multinomial logistic regression outcome model (roadmap M21).

The comparison arm to M20's ordinal logistic regression
(:mod:`models.ordinal_logit`): a four-class softmax model fit on the
*identical* 13-feature vector (:func:`models._shared.build_feature_vector`
over the M13-M17 features) and the *identical* M10 train/test split, so
the two arms differ only in their link function and parameter count —
which is the entire point of the roadmap's "compare against M20 on
identical splits" requirement. This module is deliberately
dependency-light (no ``sklearn``/``statsmodels``/``scipy`` in the repo):
the softmax link, the cross-entropy gradients, and the Armijo
backtracking optimizer are all hand-rolled here, exactly as
``models.ordinal_logit`` hand-rolls its cumulative-logit machinery.
Like every ``models/`` module it is 100% I/O-free; all Parquet/JSON
I/O lives in ``drivers/``.

Design decisions (recorded here, do not re-derive in later milestones):

- **Reference-class (baseline-category) convention.** Class 0
  ("A-regulation") is the reference: ``eta_0 = 0``, so it carries no
  parameters and its probabilities are determined implicitly by the
  other three classes' free logits. The three free classes are indexed
  ``m = 0, 1, 2`` internally, corresponding to
  ``OUTCOME_LABELS[1..3]`` (A-OT, B-OT, B-regulation); parameters are
  a 3-vector of intercepts and a ``(3, 13)`` coefficient matrix, row
  ``m`` being the free coefficient vector for class ``m + 1``. This is
  the natural four-class analogue of a binary logit and the standard
  multinomial parameterization; it has ``3 * 13 + 3 = 42`` free
  parameters versus the ordinal arm's ``13 + 3 = 16`` — over twice as
  many, on the same 209 training rows, which is exactly the
  overfitting risk the M21 proportional-odds diagnostic
  (:mod:`evaluation.proportional_odds`) is built to surface (via
  BIC's ``k * log(n)`` penalty).

- **Stable softmax link.** For one standardized row ``xs``,
  ``eta = (0, eta_1, eta_2, eta_3)`` with
  ``eta_{m+1} = intercepts[m] + xs . coefficients[m]``; category
  probabilities are ``P_j = exp(eta_j - max(eta)) / sum_k exp(eta_k -
  max(eta))`` (the max subtraction is the numerically-stable form),
  each clipped into ``[_PROB_CLIP_EPS, 1 - _PROB_CLIP_EPS]`` (from
  ``models._shared``) before use, matching ``models.ordinal_logit``'s
  own clip convention exactly.

- **Softmax cross-entropy gradients.** Per-row NLL is ``-log(P_y)``;
  for ``m = 0, 1, 2``, ``d(NLL)/d(eta_{m+1}) = P_{m+1} -
  1{y == m + 1}`` (no gradient is needed or computed for the fixed
  reference ``eta_0``), then ``d(NLL)/d(intercepts[m])`` equals that
  scalar and ``d(NLL)/d(coefficients[m]) = that * xs``. A dedicated
  test verifies these against central finite differences at multiple
  points — the same non-negotiable correctness bar task 023 set for
  ``models.ordinal_logit``'s gradients.

- **L2 regularization on ``coefficients`` only, not on ``intercepts``.**
  The objective is ``mean(NLL) + (l2_lambda / 2) * sum(coefficients **
  2)`` over all 33 free coefficient entries, with the same deliberately
  conservative default ``l2_lambda = 1.0`` as the ordinal arm — kept
  identical to M20's default for comparability (same regularization
  strength on both arms of the M20-vs-M21 comparison), not
  independently CV-tuned. Intercepts are intercept-like and are *not*
  shrunk (same "regularize slopes, not intercepts" convention as the
  ordinal arm's "not on thresholds" rule).

- **Label-marginal intercept initialization.** ``coefficients = 0``
  and ``intercepts[m] = log(max(count[m + 1], 1) /
  max(count[0], 1))`` where ``count`` is the length-4 per-category
  training label count and the ``max(..., 1)`` guard keeps a zero count
  from producing ``-inf``/``nan`` (counts are non-negative integers, so
  the guard is exact). At ``coefficients = 0`` this reproduces the
  training marginal exactly — the multinomial's natural
  "reproduce-the-training-marginal" starting point, playing the same
  role as ``models.ordinal_logit._initial_raw_thresholds`` but with no
  reparameterization needed (multinomial intercepts carry no ordering
  constraint).

- **Not directly comparable to the ordinal arm's coefficients.**
  A multinomial coefficient is "this category's log-odds shift
  *relative to the A-regulation reference*" for a one-unit feature
  change; the ordinal arm's coefficient is a *single shared
  cumulative-logit slope* across all four categories. The two models'
  coefficient values therefore must not be compared number-for-number
  — see the sign-convention section of :mod:`models.ordinal_logit`'s
  module docstring for the ordinal side of that story. This is why the
  multinomial ``coefficient_report`` (see :func:`_coefficient_report`)
  carries a ``class`` field per entry instead of an ordinal-style
  "favors A / favors B" direction string: a baseline-category
  coefficient does not have a single A-vs-B direction, it has a
  per-category relative meaning.

- **Same Armijo machinery, independently implemented.** The optimizer
  is a full-batch gradient descent with Armijo backtracking using the
  shared constants ``_ARMIJO_C`` / ``_LINE_SEARCH_MAX_STEPS`` from
  ``models._shared`` and the same convergence checks / same
  "non-converged is a valid returned model, not an error" contract as
  ``models.ordinal_logit._gradient_descent`` — but written as this
  module's own self-contained loop for this module's own parameter
  shape (``intercepts (3,)`` + ``coefficients (3, 13)``) and loss
  formula, deliberately *not* extracted into a generic shared
  optimizer (the three Armijo fits in this milestone each have a
  different flat-parameter shape and different loss/gradient formulas;
  forcing them through one generic driver would re-touch
  ``ordinal_logit``'s already-shipped, finite-difference-verified
  gradient code purely for a DRY abstraction).
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
    _N_CATEGORIES,
    _PROB_CLIP_EPS,
    FEATURE_NAMES,
    OUTCOME_LABELS,
    _validate_l2_lambda,
    _validate_positive_float,
    _validate_positive_int,
    apply_standardizer,
    build_feature_vector,
    fit_standardizer,
)


def _category_probabilities(eta: np.ndarray) -> np.ndarray:
    """Return the four category probabilities for one eta vector.

    Implements the stable softmax from the module docstring: subtract
    ``max(eta)`` before exponentiating (the numerically-stable form),
    normalize, then clip each probability into
    ``[eps, 1 - eps]`` (see :data:`_PROB_CLIP_EPS`) so the caller can
    take a log without hitting ``-inf``. The clip is the same epsilon
    convention as ``features/map_win_rate.py``'s ``_PROB_CLIP_EPS`` and
    ``models.ordinal_logit``'s own clip.

    Args:
        eta: The length-4 vector ``(0, eta_1, eta_2, eta_3)`` of linear
            predictors, index 0 being the fixed reference-class value
            (always exactly 0 by construction).

    Returns:
        A 4-vector of ``float`` probabilities in :data:`OUTCOME_LABELS`
        order, each in ``[eps, 1 - eps]``, summing to approximately 1
        (exactly 1 up to the clip's epsilon-scale perturbation).

    Raises:
        ValueError: If ``eta`` does not have exactly 4 entries.
    """
    if len(eta) != _N_CATEGORIES:
        raise ValueError(
            f"expected {_N_CATEGORIES} eta entries, got {len(eta)}"
        )
    shift = float(np.max(eta))
    exp_shifted = np.exp(np.asarray(eta, dtype=float) - shift)
    probs = exp_shifted / np.sum(exp_shifted)
    return np.clip(probs, _PROB_CLIP_EPS, 1.0 - _PROB_CLIP_EPS)


def _row_gradients(
    probs: np.ndarray,
    y: int,
) -> np.ndarray:
    """Return one row's analytic NLL gradients w.r.t. the free etas.

    Implements the gradient rule from the module docstring: for each
    free class index ``m = 0, 1, 2`` (eta position ``m + 1``),
    ``d(NLL)/d(eta_{m+1}) = P_{m+1} - 1{y == m + 1}``. No entry is
    computed for the fixed reference ``eta_0`` (it has no parameters,
    so its gradient is never needed). These are the exact derivatives
    of the *unclipped* negative log-likelihood; the clip epsilon
    (1e-12) is inactive for any realistic finite ``eta``, so in
    practice they are also the derivatives of the clipped objective
    :func:`_loss_and_gradient` actually minimizes. A dedicated
    regression test verifies them against a central finite-difference
    numerical gradient.

    Args:
        probs: The row's four category probabilities (clipped), in
            :data:`OUTCOME_LABELS` order.
        y: The row's true outcome ordinal (0-3).

    Returns:
        A 3-vector ``(d1, d2, d3)`` of the per-row NLL gradients
        w.r.t. ``eta_1, eta_2, eta_3`` respectively.

    Raises:
        ValueError: If ``y`` is not in ``{0, 1, 2, 3}`` or ``probs``
            does not have exactly 4 entries.
    """
    if y not in (0, 1, 2, 3):
        raise ValueError(f"y must be one of 0..3, got {y!r}")
    if len(probs) != _N_CATEGORIES:
        raise ValueError(
            f"expected {_N_CATEGORIES} probabilities, got {len(probs)}"
        )
    grad = np.asarray(probs[1:], dtype=float).copy()
    if y > 0:
        grad[y - 1] -= 1.0
    return grad


def _loss_and_gradient(
    Xs: np.ndarray,
    y: np.ndarray,
    intercepts: np.ndarray,
    coefficients: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the batch objective and its analytic gradient.

    Computes the full-batch objective ``mean(NLL) + (l2_lambda / 2) *
    sum(coefficients ** 2)`` (per-row NLL ``-log(P_y)`` with each
    ``P_j`` clipped into ``[eps, 1 - eps]`` via the stable softmax)
    and its analytic gradient w.r.t. both ``intercepts`` and
    ``coefficients``. Per-row gradients come from :func:`_row_gradients`
    and are averaged over the batch; the L2 term is added to
    ``d/d(coefficients)`` only (intercepts are intercept-like and must
    not be shrunk).

    Args:
        Xs: The (already-standardized) design matrix, ``(n, 13)``
            floats.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        intercepts: The current free-class intercepts, length 3.
        coefficients: The current free-class coefficient matrix,
            ``(3, p)``.
        l2_lambda: The L2 regularization strength on ``coefficients``
            only (non-negative finite float).

    Returns:
        A ``(loss, grad_intercepts, grad_coefficients)`` tuple:
        ``loss`` the scalar batch objective; ``grad_intercepts`` the
        ``(3,)`` gradient w.r.t. ``intercepts``; ``grad_coefficients``
        the ``(3, p)`` gradient w.r.t. ``coefficients`` (including the
        L2 term).

    Raises:
        ValueError: If ``X`` rows and ``len(y)`` differ, or if ``y``
            contains a value outside 0-3 (propagated from
            :func:`_row_gradients`).
    """
    n = Xs.shape[0]
    if n != len(y):
        raise ValueError(
            f"Xs has {n} rows but y has {len(y)} entries; they must match"
        )
    total_nll = 0.0
    grad_intercepts = np.zeros(3, dtype=float)
    grad_coefficients = np.zeros_like(coefficients, dtype=float)
    for i in range(n):
        eta = np.empty(_N_CATEGORIES, dtype=float)
        eta[0] = 0.0
        eta[1:] = intercepts + np.dot(Xs[i], coefficients.T)
        probs = _category_probabilities(eta)
        total_nll += -math.log(probs[y[i]])
        d = _row_gradients(probs, y[i])
        grad_intercepts += d
        grad_coefficients += np.outer(d, Xs[i])
    nll = total_nll / n
    grad_intercepts /= n
    grad_coefficients /= n
    # L2 on coefficients only (intercepts are intercept-like and must
    # not be shrunk); the penalty is NOT averaged over the batch.
    l2_penalty = (l2_lambda / 2.0) * float(np.sum(coefficients**2))
    grad_coefficients += l2_lambda * coefficients
    return nll + l2_penalty, grad_intercepts, grad_coefficients


def _initial_intercepts(counts: np.ndarray) -> np.ndarray:
    """Seed the free-class intercepts from the training label marginal.

    Computes the intercepts that reproduce the empirical marginal
    distribution at ``coefficients = 0`` (the initialization point):
    with all coefficients zero, ``P_{m+1} / P_0 = exp(intercepts[m])``,
    so ``intercepts[m] = log(count[m + 1] / count[0])`` — the standard
    "reproduce-the-marginal" multinomial initialization. Counts are
    non-negative integers, so the exact guard ``max(count[k], 1)``
    keeps a zero count (a missing category) from producing
    ``-inf``/``nan``/``0/0``.

    Args:
        counts: A 4-vector of per-category label counts (index =
            outcome ordinal). Must sum to at least 1.

    Returns:
        A 3-vector of ``float`` intercepts, one per free class.

    Raises:
        ValueError: If ``counts`` does not have exactly 4 elements or
            sums to zero (nothing to initialize from).
    """
    if len(counts) != _N_CATEGORIES:
        raise ValueError(
            f"expected {_N_CATEGORIES} label counts, got {len(counts)}"
        )
    if int(np.sum(counts)) == 0:
        raise ValueError("cannot initialize intercepts from an empty label vector")
    intercepts = np.empty(3, dtype=float)
    for m in range(3):
        intercepts[m] = math.log(
            max(counts[m + 1], 1) / max(counts[0], 1)
        )
    return intercepts


def _gradient_descent(
    Xs: np.ndarray,
    y: np.ndarray,
    l2_lambda: float,
    max_iter: int,
    grad_tol: float,
    loss_tol: float,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[float, ...]]:
    """Run full-batch gradient descent with Armijo backtracking.

    The multinomial's own optimizer (see the module docstring's "same
    Armijo machinery, independently implemented" bullet). Starts from
    ``coefficients = 0`` and the marginal-derived intercepts
    (:func:`_initial_intercepts`), then iterates: compute the
    loss/gradient at the current point; stop (converged) if the
    combined gradient norm over both ``coefficients`` and
    ``intercepts`` is below ``grad_tol`` or if the loss improvement
    between iterations drops below ``loss_tol``; otherwise try step
    size ``1.0``, halving up to :data:`_LINE_SEARCH_MAX_STEPS` times
    until the Armijo sufficient-decrease condition
    (``loss(p - step*grad) <= loss(p) - _ARMIJO_C * step * ||grad||^2``)
    holds, then take that step. If the line search cannot find any
    acceptable step, or if ``max_iter`` is hit, the run stops with
    ``converged=False`` and returns the best point found — a
    non-converged fit is a valid (if suboptimal) model, not an error.
    The returned loss trace is non-increasing by construction (every
    accepted step satisfies Armijo).

    Args:
        Xs: The already-standardized design matrix, ``(n, p)`` floats.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        l2_lambda: The L2 strength (validated non-negative finite).
        max_iter: The iteration cap (validated positive int).
        grad_tol: The gradient-norm convergence tolerance.
        loss_tol: The loss-improvement convergence tolerance.

    Returns:
        A ``(best_coefficients, best_intercepts, converged, n_iter,
        loss_trace)`` tuple: the best coefficient matrix, the best
        intercept vector, whether the run converged, the number of
        iterations actually executed, and the per-iteration loss trace
        (non-increasing, length ``n_iter``).

    Raises:
        ValueError: If the ``y`` values are outside 0-3 or the shapes
            are inconsistent (propagated from
            :func:`_loss_and_gradient`), or if the label vector is
            empty (propagated from :func:`_initial_intercepts`).
    """
    if len(y) == 0:
        raise ValueError("cannot run gradient descent on an empty label vector")
    p = Xs.shape[1]
    coefficients = np.zeros((3, p), dtype=float)
    intercepts = _initial_intercepts(np.bincount(y, minlength=_N_CATEGORIES))

    best_coefficients = coefficients.copy()
    best_intercepts = intercepts.copy()
    best_loss = float("inf")
    loss_trace: list[float] = []
    prev_loss: float | None = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        loss, grad_intercepts, grad_coefficients = _loss_and_gradient(
            Xs, y, intercepts, coefficients, l2_lambda
        )
        n_iter = iteration + 1
        if loss < best_loss:
            best_loss = loss
            best_coefficients = coefficients.copy()
            best_intercepts = intercepts.copy()
        loss_trace.append(loss)

        grad_norm = math.sqrt(
            float(np.sum(grad_coefficients**2))
            + float(np.sum(grad_intercepts**2))
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
            trial_coefficients = coefficients - step * grad_coefficients
            trial_intercepts = intercepts - step * grad_intercepts
            trial_loss, _, _ = _loss_and_gradient(
                Xs, y, trial_intercepts, trial_coefficients, l2_lambda
            )
            if trial_loss <= loss - _ARMIJO_C * step * grad_norm**2:
                coefficients = trial_coefficients
                intercepts = trial_intercepts
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

    return best_coefficients, best_intercepts, converged, n_iter, tuple(loss_trace)


@dataclass(frozen=True)
class MultinomialLogitModel:
    """A fitted baseline-category multinomial logistic regression model.

    Holds the fitted parameters and the diagnostics needed to (a) make
    predictions via :func:`predict_proba`, (b) serialize/deserialize
    via :func:`to_dict` / :func:`from_dict`, and (c) interpret the fit
    via the coefficient report. ``coefficients`` is a ``(3, 13)``
    matrix whose row ``m`` is the free coefficient vector for
    ``OUTCOME_LABELS[m + 1]`` (relative-log-odds against the
    A-regulation reference class; see the module docstring's
    comparability bullet). ``intercepts`` is the matching 3-vector of
    free-class intercepts. ``standardizer_means`` /
    ``standardizer_stds`` describe the *training* design matrix (the
    second leakage boundary; see :mod:`models._shared`'s docstring).
    ``loss_trace`` is a live-fit diagnostic (per-iteration loss,
    non-increasing) that is deliberately *not* persisted by
    :func:`to_dict` — a deserialized model carries an empty trace.

    Attributes:
        coefficients: The ``(3, 13)`` matrix of free-class
            coefficients.
        intercepts: The 3-vector of free-class intercepts.
        standardizer_means: Per-feature training-column means (length
            13).
        standardizer_stds: Per-feature training-column stds (length 13;
            a zero-variance column's std is ``1.0`` per the guard).
        feature_names: The feature name tuple (:data:`FEATURE_NAMES`).
        converged: Whether gradient descent converged (``True``) or hit
            ``max_iter``/line-search failure (``False`` — still a valid,
            if suboptimal, model).
        n_iter: Number of gradient-descent iterations executed.
        final_loss: The objective value at the returned point.
        n_train: Number of training rows the model was fit on.
        l2_lambda: The L2 strength used for this fit (stored so the
            artifact records its own regularization).
        loss_trace: The per-iteration objective trace (non-increasing,
            length ``n_iter``); ``()`` for a deserialized model.
    """

    coefficients: np.ndarray
    intercepts: np.ndarray
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
) -> MultinomialLogitModel:
    """Fit the multinomial logistic regression by Armijo gradient descent.

    Assembles a complete :class:`MultinomialLogitModel` from a raw
    feature matrix: fits the per-feature z-score standardizer on the
    *training* matrix only (the second leakage boundary; see
    :func:`models._shared.fit_standardizer`) and transforms with it,
    then runs full-batch gradient descent with Armijo backtracking (see
    :func:`_gradient_descent`) initialized at ``coefficients = 0`` and
    marginal-derived intercepts (see :func:`_initial_intercepts`). The
    returned model carries the training standardizer so
    :func:`predict_proba` can standardize later rows with the exact
    training-population statistics.

    Args:
        X: The raw (unstandardized) training design matrix, ``(n, 13)``
            floats in :data:`FEATURE_NAMES` order — the output of
            :func:`models._shared.build_feature_vector` over the
            training rows. The standardizer is fit on this matrix inside
            this function.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        l2_lambda: L2 regularization strength on ``coefficients`` only;
            must be non-negative finite (default ``1.0`` — the
            documented conservative default, kept identical to M20's
            ordinal arm for comparability, not CV-tuned in this task).
        max_iter: Cap on gradient-descent iterations (default 2000). If
            the cap is hit without convergence, ``fit`` returns the best
            point found with ``converged=False`` rather than raising.
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).

    Returns:
        A frozen :class:`MultinomialLogitModel` with the fitted
        ``coefficients``/``intercepts``, the training standardizer, the
        diagnostics (``converged``/``n_iter``/``final_loss``/
        ``n_train``/``l2_lambda``/``loss_trace``), and
        ``feature_names = FEATURE_NAMES``.

    Raises:
        ValueError: If ``X`` is not a 2-D array, ``y`` is not 1-D, their
            row counts differ, ``X`` does not have exactly
            ``len(FEATURE_NAMES)`` columns, ``y`` is empty or contains a
            value outside ``{0, 1, 2, 3}``, or any hyperparameter is
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
        raise ValueError("cannot fit a multinomial model on an empty label vector")
    if set(np.unique(y_arr).tolist()) - set(range(_N_CATEGORIES)):
        raise ValueError(
            f"y must contain only outcome ordinals 0..{_N_CATEGORIES - 1}, "
            f"got values {sorted(set(np.unique(y_arr).tolist()))}"
        )

    l2 = _validate_l2_lambda(l2_lambda)
    max_it = _validate_positive_int(max_iter, "max_iter")
    g_tol = _validate_positive_float(grad_tol, "grad_tol")
    l_tol = _validate_positive_float(loss_tol, "loss_tol")

    means, stds = fit_standardizer(X_arr)
    Xs = apply_standardizer(X_arr, means, stds)
    coefficients, intercepts, converged, n_iter, trace = _gradient_descent(
        Xs, y_arr, l2, max_it, g_tol, l_tol
    )
    return MultinomialLogitModel(
        coefficients=coefficients,
        intercepts=intercepts,
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
    model: MultinomialLogitModel,
) -> tuple[float, float, float, float]:
    """Predict the four category probabilities for one feature vector.

    Applies the model's stored standardizer (training-population means/
    stds — the second leakage boundary, applied identically to every
    later row), computes ``eta = (0, intercepts + xs . coefficients)``
    and returns the four :data:`OUTCOME_LABELS`-ordered probabilities
    via :func:`_category_probabilities` (each clipped into
    ``[eps, 1-eps]``, so the tuple is a valid, scorable simplex even
    for extreme inputs).

    Args:
        x: A raw feature vector, length 13 in :data:`FEATURE_NAMES`
            order (the output of
            :func:`models._shared.build_feature_vector`).
        model: The fitted model whose stored standardizer and
            coefficients are applied.

    Returns:
        The 4-tuple ``(p_a_regulation, p_a_ot, p_b_ot, p_b_regulation)``
        of ``float`` probabilities in :data:`OUTCOME_LABELS` order,
        each in ``[eps, 1 - eps]`` and summing to approximately 1.

    Raises:
        ValueError: If ``x`` does not have exactly as many entries as
            the model has features (a feature-vector/model mismatch
            would silently misalign).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.shape[0] != len(model.feature_names):
        raise ValueError(
            f"feature vector has {x_arr.shape[0]} entries but the model "
            f"has {len(model.feature_names)} features; the feature "
            "vector must match the model's feature_names order"
        )
    # Reshape to a 1-row design matrix: apply_standardizer's contract is
    # 2-D matrices; the prediction path hands it a single feature vector.
    xs = apply_standardizer(
        x_arr.reshape(1, -1),
        model.standardizer_means,
        model.standardizer_stds,
    )[0]
    eta = np.empty(_N_CATEGORIES, dtype=float)
    eta[0] = 0.0
    eta[1:] = model.intercepts + np.dot(xs, model.coefficients.T)
    probs = _category_probabilities(eta)
    return (
        float(probs[0]),
        float(probs[1]),
        float(probs[2]),
        float(probs[3]),
    )


def _coefficient_report(model: MultinomialLogitModel) -> list[dict]:
    """Build the human-readable coefficient report for a fitted model.

    One entry per free (feature, class) pair — 33 entries total — each
    ``{"feature": name, "class": label, "coefficient": value}`` where
    ``label`` is ``OUTCOME_LABELS[m + 1]`` for the row ``m`` carrying
    the coefficient. The meaning of each value is "this category's
    log-odds shift relative to the A-regulation reference class" for a
    one-unit feature change, which is why there is deliberately no
    ordinal-style ``direction`` field: a baseline-category coefficient
    does not have a single A-vs-B direction (see the module docstring's
    comparability bullet). Entries are sorted by ``abs(coefficient)``
    descending so the most influential (feature, class) pairs read
    first.

    Args:
        model: The fitted model whose ``coefficients`` and
            ``feature_names`` are reported.

    Returns:
        A list of 33 dicts, one per free (feature, class) pair, sorted
        by descending ``abs(coefficient)``.

    Raises:
        Nothing (the model's own shape validation guarantees the two
            arrays line up).
    """
    entries = []
    for m in range(3):
        label = OUTCOME_LABELS[m + 1]
        for k, name in enumerate(model.feature_names):
            entries.append(
                {
                    "feature": name,
                    "class": label,
                    "coefficient": float(model.coefficients[m, k]),
                }
            )
    return sorted(
        entries, key=lambda entry: abs(entry["coefficient"]), reverse=True
    )


def to_dict(model: MultinomialLogitModel) -> dict:
    """Serialize a fitted model to a plain JSON-serializable dict.

    Produces the artifact dict the training driver writes:
    ``feature_names``, ``coefficients`` (a nested ``3 x 13`` list via
    ``.tolist()``), ``intercepts`` (a 3-list), ``standardizer_means``,
    ``standardizer_stds``, ``l2_lambda``, ``converged``, ``n_iter``,
    ``final_loss``, ``n_train``, plus a ``coefficient_report`` list
    (from :func:`_coefficient_report`). Every value is a plain
    str/int/float/list so ``json.dumps`` accepts the dict directly. The
    ``loss_trace`` diagnostic is deliberately *not* persisted (it is a
    live-fit trace, not model parameters). No file I/O happens here.

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
        "coefficients": model.coefficients.tolist(),
        "intercepts": [float(i) for i in model.intercepts],
        "standardizer_means": [float(m) for m in model.standardizer_means],
        "standardizer_stds": [float(s) for s in model.standardizer_stds],
        "l2_lambda": float(model.l2_lambda),
        "converged": bool(model.converged),
        "n_iter": int(model.n_iter),
        "final_loss": float(model.final_loss),
        "n_train": int(model.n_train),
        "coefficient_report": _coefficient_report(model),
    }


def from_dict(d: dict) -> MultinomialLogitModel:
    """Deserialize a fitted model from a to_dict-produced dict.

    Reconstructs a :class:`MultinomialLogitModel` from the plain dict
    :func:`to_dict` produces (or from ``json.loads`` of the artifact the
    training driver writes). Arrays are rebuilt as numpy arrays;
    shape consistency is validated (the flat coefficient list must hold
    3 * 13 entries, intercepts/means/stds must line up with
    ``feature_names``). The ``coefficient_report`` key is ignored on
    read (it is derived, not stored) and ``loss_trace`` is empty for a
    deserialized model. No file I/O happens here.

    Args:
        d: The dict to load; must carry the ten parameter/diagnostic
            keys (``coefficient_report`` optional, ignored).

    Returns:
        A :class:`MultinomialLogitModel` whose parameters reproduce the
            serialized ones exactly (``feature_names`` as a tuple,
            arrays as ``float`` numpy arrays, diagnostics as plain
            scalars).

    Raises:
        KeyError: If a required key is absent (propagated from dict
            indexing).
        ValueError: If the shapes are inconsistent (the flat
            coefficient list does not hold ``3 * len(feature_names)``
            entries, means/stds length != feature count, or intercepts
            not length 3), or if a numeric field cannot be coerced.
    """
    feature_names = tuple(str(name) for name in d["feature_names"])
    coefficients = np.asarray(d["coefficients"], dtype=float)
    intercepts = np.asarray(d["intercepts"], dtype=float)
    means = np.asarray(d["standardizer_means"], dtype=float)
    stds = np.asarray(d["standardizer_stds"], dtype=float)
    n_features = len(feature_names)
    if coefficients.shape != (3, n_features):
        raise ValueError(
            f"coefficients must be a 3 x {n_features} matrix to match "
            f"{n_features} features, got shape {coefficients.shape}"
        )
    if len(intercepts) != 3:
        raise ValueError(
            f"expected 3 intercepts (one per free class), got {len(intercepts)}"
        )
    if len(means) != n_features or len(stds) != n_features:
        raise ValueError(
            f"standardizer means ({len(means)}) / stds ({len(stds)}) must "
            f"each have one entry per feature ({n_features})"
        )
    return MultinomialLogitModel(
        coefficients=coefficients,
        intercepts=intercepts,
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
    model: MultinomialLogitModel,
    player_map_stats_df: pd.DataFrame,
) -> Callable[[str, str, str, str, pd.DataFrame, pd.DataFrame], tuple[float, float, float, float]]:
    """Wrap a fitted model into the generic 6-argument model-interface shape.

    Bridges the fixed generic model interface — the M19 harness's
    ``ModelFn`` callable, ``(team1_id, team2_id, map_name, date,
    matches_df, maps_df) -> Sequence[float]`` — to a feature builder that
    needs a *seventh* table (``player_map_stats_df``) by closing that
    table over at load time. The returned closure has exactly the
    6-argument shape (structurally, without importing the harness — this
    module must not depend upward on the evaluation layer), calls
    :func:`models._shared.build_feature_vector` with the closed-over
    table, and returns :func:`predict_proba`'s 4-tuple.

    This is only correct because callers (the M19 harness and the
    evaluation driver) always invoke the returned closure with the same
    ``matches_df``/``maps_df`` that came from the same
    ``<output_dir>/<version>`` the closed-over ``player_map_stats_df``
    was loaded from — a mismatched pairing would silently misalign,
    exactly as already implicit in every existing model/table pairing in
    this codebase (and identical to
    :func:`models.ordinal_logit.make_model_fn`'s documented contract).

    Args:
        model: The fitted model to predict with.
        player_map_stats_df: The materialised ``player_map_stats`` table
            for the same dataset version as the ``matches_df``/``maps_df``
            the closure will be invoked with.

    Returns:
        A closure ``(team1_id, team2_id, map_name, date, matches_df,
        maps_df) -> (p_a_regulation, p_a_ot, p_b_ot, p_b_regulation)``
        (a 4-tuple of floats in :data:`OUTCOME_LABELS` order summing to
        approximately 1).

    Raises:
        ValueError: If the feature vector length mismatches the model
            (propagated from :func:`predict_proba`), or if any feature
            computation fails (propagated from
            :func:`models._shared.build_feature_vector`).
        KeyError: If any table lacks a required column (propagated from
            :func:`models._shared.build_feature_vector`).
    """

    def model_fn(
        team1_id: str,
        team2_id: str,
        map_name: str,
        date: str,
        matches_df: pd.DataFrame,
        maps_df: pd.DataFrame,
    ) -> tuple[float, float, float, float]:
        """Predict the four category probabilities for one held-out map.

        Computes the raw 13-feature vector via
        :func:`models._shared.build_feature_vector` (using the
        closed-over ``player_map_stats_df`` — the table the generic
        interface does not pass) and returns :func:`predict_proba`'s
        4-tuple in :data:`OUTCOME_LABELS` order. See
        :func:`make_model_fn`'s docstring for the closed-table contract.

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
            The 4-tuple of probabilities in :data:`OUTCOME_LABELS`
            order, summing to approximately 1.

        Raises:
            ValueError: If the feature vector length mismatches the
                model, or if any feature computation fails (propagated
                from :func:`predict_proba` /
                :func:`models._shared.build_feature_vector`).
            KeyError: If any table lacks a required column (propagated
                from :func:`models._shared.build_feature_vector`).
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


def total_log_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    model: MultinomialLogitModel,
) -> float:
    """Return the model's total log-likelihood on a training batch.

    Computes ``sum(log(P_y))`` over the batch by calling the module's own
    public :func:`predict_proba` once per row (each probability clipped
    into ``[eps, 1 - eps]``, so every log is finite) — deliberately
    implemented through the public prediction path rather than by
    duplicating the internal NLL loop, so there is exactly one place per
    model that computes a probability from a raw feature vector. Consumed
    by :func:`evaluation.proportional_odds.build_diagnostic_report` (M21),
    which needs the training log-likelihoods of both the ordinal and the
    multinomial arm for the AIC/BIC comparison.

    Args:
        X: The raw (unstandardized) training design matrix, ``(n, 13)``
            floats in :data:`FEATURE_NAMES` order. Rows are standardized
            inside :func:`predict_proba` with the model's stored
            training-population statistics.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        model: The fitted model whose predictions are evaluated.

    Returns:
        The total log-likelihood as a ``float`` (``<= 0``; exactly zero
        only for a degenerate batch where every ``P_y`` is 1).

    Raises:
        ValueError: If ``X`` rows and ``y`` entries do not match, if a
            ``y`` value is outside ``{0, 1, 2, 3}``, or if a feature
            vector length mismatches the model (propagated from
            :func:`predict_proba`).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if X_arr.shape[0] != len(y_arr):
        raise ValueError(
            f"X has {X_arr.shape[0]} rows but y has {len(y_arr)} entries; "
            "they must match"
        )
    if set(np.unique(y_arr).tolist()) - set(range(_N_CATEGORIES)):
        raise ValueError(
            f"y must contain only outcome ordinals 0..{_N_CATEGORIES - 1}, "
            f"got values {sorted(set(np.unique(y_arr).tolist()))}"
        )
    total = 0.0
    for i in range(len(y_arr)):
        probs = predict_proba(X_arr[i], model)
        total += math.log(probs[y_arr[i]])
    return total

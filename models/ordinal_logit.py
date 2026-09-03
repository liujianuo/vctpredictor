"""Hand-rolled ordinal logistic regression (proportional odds) outcome model (roadmap M20).

The first *trained* model in the repository: a single coefficient vector
plus three ordered thresholds fit by full-batch gradient descent (with
Armijo backtracking line search) over the M13-M17 feature vector, with a
coefficient report for interpretability, pluggable into the M19
evaluation harness's generic model interface.

The model is deliberately dependency-light: no ``sklearn``/``statsmodels``/
``scipy`` exists in this repo, so the fit (gradient descent), the
cumulative-logit link, and the analytic gradients are all hand-rolled
here. ``numpy`` is imported directly (precedented by ``drivers/labels.py``,
``features/closeness.py``, ``utils/asof.py``, ``utils/splits.py`` and now
pinned in ``requirements.txt``). Like its sibling
``models.four_way_baseline``, this module is 100% I/O-free: it neither
reads nor writes files; all Parquet/JSON file I/O lives in ``drivers/``
(both the new training driver and the existing evaluation driver).

Design decisions (recorded here, do not re-derive in later milestones):

- **Sign convention (must never be inverted).** ``eta = x . beta`` with
  no intercept (the three thresholds serve as per-category intercepts);
  for ``j in {1, 2, 3}``, ``z_j = theta_j + eta`` and ``C_j =
  sigmoid(z_j)``, with probabilities ``P0 = C1`` (A-regulation),
  ``P1 = C2 - C1`` (A-OT), ``P2 = C3 - C2`` (B-OT), ``P3 = 1 - C3``
  (B-regulation). Because every team-specific feature is expressed as an
  A-minus-B difference, a *positive* coefficient must mean "this feature
  favors team A": increasing it shifts probability mass toward the low
  ordinal categories (A wins). ``C_j`` increasing means more mass at
  low categories, so ``C_j`` must increase with ``eta`` — hence
  ``+theta_j + eta``, deliberately *not* the textbook latent-variable
  ``theta_j - eta`` convention. A regression test locks in the direction
  (a synthetic single-informative-feature dataset must fit a positive
  coefficient when the feature is high exactly on A-win rows). Note the
  exact cumulative reading: ``C_1 = P(Y = 0)``, ``C_2 = P(Y <= 1)``,
  ``C_3 = P(Y <= 2)``, so ``theta_j`` at ``eta = 0`` is the logit of the
  empirical cumulative frequency up to (1-based) category ``j`` — i.e.
  up to (0-based) category ``j - 1``.

- **Softplus-reparameterized thresholds.** Strict ordering
  ``theta_1 < theta_2 < theta_3`` is enforced by construction, without
  constrained optimization: ``theta_1 = a1``,
  ``theta_2 = a1 + softplus(a2)``, ``theta_3 = theta_2 + softplus(a3)``
  where ``softplus(a) = log1p(exp(-|a|)) + max(a, 0)`` (the
  numerically-stable form) and ``a1, a2, a3`` are the unconstrained
  parameters gradient descent actually updates. The reverse map (used
  only at initialization) is ``a1 = theta1``, ``a2 = inverse_softplus(
  theta2 - theta1)``, ``a3 = inverse_softplus(theta3 - theta2)``.

- **L2 regularization on ``beta`` only.** The objective per batch is
  ``mean(NLL) + (l2_lambda / 2) * sum(beta_k^2)`` with a deliberately
  conservative default ``l2_lambda = 1.0``: the real v1 train split has
  only 11-15 rows in each OT category, so unregularized coefficients on
  the thin categories would chase noise. Thresholds are intercept-like
  and are *not* shrunk (no L2 on ``a1, a2, a3``). ``l2_lambda`` is a
  documented default, not CV-tuned in this task — tuning it is
  future-milestone scope.

- **Missing-value fallback policy (apply per-side before differencing).**
  The M16/M17 features legitimately return ``None`` for "no signal";
  each is converted to a neutral value *before* the A-minus-B
  subtraction:
  - ``acs_form_diff`` / ``rating_form_diff``: if either side's
    ``FormStat.mean`` is ``None`` (zero qualifying maps), the feature is
    ``0.0`` (no signal), not a partial diff against the other side's
    real value.
  - ``days_since_diff``: if a side's ``days_since_last_match`` is
    ``None`` (unseen team / no strictly-prior match), that side is
    treated as ``0`` before subtracting.
  - ``roster_decay_diff``: if a side's ``decay_multiplier`` is ``None``
    (either ``changed is None`` — fewer than two evaluable maps — or
    ``changed is False``, i.e. no roster change declared), that side is
    treated as ``1.0`` (no penalty) before subtracting.
  - ``map_round_margin_variance``: ``float("nan")`` when ``n <= 1`` for
    that map is replaced with ``0.0`` — documented as "no observed-
    variance signal contributes no information", not a statistically
    principled imputation.

- **Standardization is its own, separate leakage boundary.** A
  per-feature ``(mean, std)`` z-score standardizer is fit *only on the
  assembled training design matrix* — never on test/held-out rows — via
  :func:`fit_standardizer` / :func:`apply_standardizer`. This is a
  second, distinct leakage boundary on top of each feature's own as-of
  ``<`` cutoff (inherited unchanged from each feature module's
  ``utils.asof`` usage), and must be documented as such: the means/stds
  describe the training population and would leak test-distribution
  information if fit on the full data. A zero-variance training column
  standardizes to ``0.0`` for every row (its std is replaced by ``1.0``
  to guard the divide-by-zero) rather than raising, since a degenerate/
  constant column is plausible at this data scale.

- **The M19-interface gap and its resolution.** The generic model
  interface (the M19 harness's ``ModelFn`` shape) is fixed at
  ``(team1_id, team2_id, map_name, date, matches_df, maps_df)`` — it does
  *not* pass ``player_map_stats_df``. But M16 (``features.player_form``)
  and M17's roster change (``features.h2h_context.team_roster_change``)
  both *require* that table. This module resolves the gap by closing
  over ``player_map_stats_df`` at model-load time in
  :func:`make_model_fn` (the returned closure captures the table and
  passes it into :func:`models._shared.build_feature_vector`), rather than by changing
  the shared interface. That is only correct because callers (the M19
  harness and the evaluation driver) always invoke the returned closure
  with the same ``matches_df``/``maps_df`` that came from the same
  ``<output_dir>/<version>`` the closed-over ``player_map_stats_df`` was
  loaded from — a mismatched pairing would silently misalign, exactly as
  already implicit in every existing model/table pairing in this
  codebase.
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
    _sigmoid,
    _validate_l2_lambda,
    _validate_positive_float,
    _validate_positive_int,
    apply_standardizer,
    build_feature_vector,
    fit_standardizer,
)

# The module's public API, declared explicitly so the linter treats the
# names re-exported from models._shared (OUTCOME_LABELS et al., which
# tests import via ``from models.ordinal_logit import ...`` and must
# therefore keep resolving here) as intentional re-exports rather than
# unused imports. Mirrors the identical convention in
# ``drivers/splits.py``'s ``__all__``.
__all__ = (
    "FEATURE_NAMES",
    "OUTCOME_LABELS",
    "OrdinalLogitModel",
    "apply_standardizer",
    "build_feature_vector",
    "fit",
    "fit_standardizer",
    "from_dict",
    "make_model_fn",
    "predict_proba",
    "to_dict",
    "total_log_likelihood",
)


def _softplus(a: float) -> float:
    """Return the numerically-stable softplus of ``a``.

    ``softplus(a) = log1p(exp(-|a|)) + max(a, 0)`` — algebraically
    ``log(1 + exp(a))`` but evaluated without overflowing ``exp(a)`` for
    large positive ``a`` and without underflowing for large negative
    ``a``. Strictly positive for every finite ``a``, which is what makes
    ``theta_{j+1} = theta_j + softplus(a_{j+1})`` a strict-increase
    guarantee by construction.

    Args:
        a: The unconstrained threshold increment parameter.

    Returns:
        The softplus value as a positive ``float``.

    Raises:
        Nothing (the formula is total for any finite input).
    """
    return math.log1p(math.exp(-abs(a))) + max(a, 0.0)


def _inverse_softplus(x: float) -> float:
    """Return the unique ``a`` with ``softplus(a) == x``.

    The reverse of :func:`_softplus`: for ``x > 0``,
    ``a = log(expm1(x))`` (since ``exp(a) - 1 = exp(x) - 1``), with a
    large-``x`` shortcut (``softplus(a) ~= a`` for ``a > 30``, so the
    inverse is approximately ``x`` there) that avoids ``expm1``
    overflowing. Used only at initialization to invert the
    reparameterization for the marginal-derived thresholds.

    Args:
        x: A strictly positive value that ``softplus`` produced (the
            gap between two thresholds).

    Returns:
        The unique ``a`` with ``softplus(a) == x``, as a ``float``.

    Raises:
        ValueError: If ``x <= 0`` (softplus's range is ``(0, inf)``,
            so a non-positive argument is outside it; cannot happen
            from a strictly-increasing threshold triple).
    """
    if x <= 0.0:
        raise ValueError(
            f"inverse softplus requires a strictly positive argument, got {x!r}"
        )
    if x > 30.0:
        return x
    return math.log(math.expm1(x))


def _thresholds_from_raw(raw: np.ndarray) -> np.ndarray:
    """Map the three unconstrained threshold parameters to ordered thresholds.

    Applies the softplus reparameterization from the module docstring:
    ``theta_1 = a1``, ``theta_2 = a1 + softplus(a2)``,
    ``theta_3 = theta_2 + softplus(a3)``. The result is strictly
    increasing by construction (each ``softplus`` term is strictly
    positive), which is the entire point of the reparameterization — the
    ordinal constraint ``theta_1 < theta_2 < theta_3`` never needs an
    explicit constraint during optimization.

    Args:
        raw: A 3-vector ``(a1, a2, a3)`` of unconstrained floats.

    Returns:
        A 3-vector ``(theta_1, theta_2, theta_3)`` as a numpy array of
        ``float``, strictly increasing.

    Raises:
        ValueError: If ``raw`` does not have exactly 3 elements.
    """
    if len(raw) != 3:
        raise ValueError(
            f"expected 3 raw threshold parameters, got {len(raw)}"
        )
    a1 = float(raw[0])
    a2 = float(raw[1])
    a3 = float(raw[2])
    theta1 = a1
    theta2 = a1 + _softplus(a2)
    theta3 = theta2 + _softplus(a3)
    return np.asarray([theta1, theta2, theta3], dtype=float)


def _raw_from_thresholds(thresholds: np.ndarray) -> np.ndarray:
    """Invert the softplus reparameterization (initialization only).

    Computes the unconstrained ``(a1, a2, a3)`` that reproduce the given
    strictly-increasing ``(theta_1, theta_2, theta_3)``:
    ``a1 = theta_1``, ``a2 = inverse_softplus(theta_2 - theta_1)``,
    ``a3 = inverse_softplus(theta_3 - theta_2)``. Used only to seed
    gradient descent from the empirical label-marginal thresholds; the
    forward map is exact so this is an exact inverse.

    Args:
        thresholds: A 3-vector of strictly increasing ``theta`` values.

    Returns:
        The 3-vector ``(a1, a2, a3)`` as a numpy array of ``float``.

    Raises:
        ValueError: If ``thresholds`` does not have exactly 3 elements,
            or if the differences ``theta_2 - theta_1`` /
            ``theta_3 - theta_2`` are not strictly positive (propagated
            from :func:`_inverse_softplus`).
    """
    if len(thresholds) != 3:
        raise ValueError(
            f"expected 3 thresholds, got {len(thresholds)}"
        )
    t1 = float(thresholds[0])
    t2 = float(thresholds[1])
    t3 = float(thresholds[2])
    return np.asarray(
        [t1, _inverse_softplus(t2 - t1), _inverse_softplus(t3 - t2)],
        dtype=float,
    )


def _initial_raw_thresholds(counts: np.ndarray) -> np.ndarray:
    """Seed the raw threshold parameters from the training label marginal.

    Computes the thresholds that reproduce the empirical marginal
    distribution at ``eta = 0`` (i.e. with ``beta = 0``, the
    initialization point): ``theta_j`` is the logit of the cumulative
    empirical frequency of (0-based) categories ``0 .. j-1`` — the exact
    cumulative reading the sign-convention section of the module
    docstring pins down (``C_1 = P(Y = 0)``, ``C_2 = P(Y <= 1)``,
    ``C_3 = P(Y <= 2)``) — then inverts the softplus reparameterization
    to the unconstrained ``(a1, a2, a3)`` form gradient descent updates.
    This gives the no-covariate case a near-correct starting point and is
    the standard-practice initialization for ordinal models. Frequencies
    at the 0/1 boundary are clipped into ``[eps, 1 - eps]`` so a missing
    category (a zero count) yields a finite logit instead of ``+/-inf``.

    Args:
        counts: A 4-vector of per-category label counts (index =
            outcome ordinal). Must sum to at least 1.

    Returns:
        A 3-vector ``(a1, a2, a3)`` of unconstrained raw threshold
        parameters as a numpy array of ``float``.

    Raises:
        ValueError: If ``counts`` does not have exactly 4 elements or
            sums to zero (nothing to initialize from).
    """
    if len(counts) != _N_CATEGORIES:
        raise ValueError(
            f"expected {_N_CATEGORIES} label counts, got {len(counts)}"
        )
    n = int(np.sum(counts))
    if n == 0:
        raise ValueError("cannot initialize thresholds from an empty label vector")
    cum = np.cumsum(counts)
    thetas = np.empty(3, dtype=float)
    for j in range(3):
        p = cum[j] / n
        p = min(max(p, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
        thetas[j] = math.log(p / (1.0 - p))
    return _raw_from_thresholds(thetas)


def _category_probabilities(
    eta: float,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return the four category probabilities for one linear predictor value.

    Implements section C of the design: with ``C_j = sigmoid(theta_j +
    eta)``, ``P0 = C1``, ``P1 = C2 - C1``, ``P2 = C3 - C2``,
    ``P3 = 1 - C3``. Each probability is clipped into
    ``[eps, 1 - eps]`` (see :data:`_PROB_CLIP_EPS`) before being
    returned, so the caller can take a log without hitting ``-inf``; the
    clip is the same epsilon convention as
    ``features/map_win_rate.py``'s ``_PROB_CLIP_EPS``.

    Args:
        eta: The scalar linear predictor ``x . beta``.
        thresholds: The 3-vector of strictly increasing thresholds
            ``(theta_1, theta_2, theta_3)``.

    Returns:
        A 4-vector of ``float`` probabilities in :data:`OUTCOME_LABELS`
        order, each in ``[eps, 1 - eps]``, summing to approximately 1
        (exactly 1 up to the clip's epsilon-scale perturbation).

    Raises:
        ValueError: If ``thresholds`` does not have exactly 3 elements
            (propagated from the array indexing).
    """
    if len(thresholds) != 3:
        raise ValueError(
            f"expected 3 thresholds, got {len(thresholds)}"
        )
    c1 = _sigmoid(float(thresholds[0]) + eta)
    c2 = _sigmoid(float(thresholds[1]) + eta)
    c3 = _sigmoid(float(thresholds[2]) + eta)
    probs = np.asarray([c1, c2 - c1, c3 - c2, 1.0 - c3], dtype=float)
    return np.clip(probs, _PROB_CLIP_EPS, 1.0 - _PROB_CLIP_EPS)


def _row_gradients(
    eta: float,
    thresholds: np.ndarray,
    y: int,
) -> tuple[float, float, float, float]:
    """Return one row's analytic NLL gradients w.r.t. eta and the thetas.

    Implements section D of the design exactly: the per-row derivatives
    of ``-log(P_y)`` with respect to ``eta`` and each of the three
    thresholds ``theta_1, theta_2, theta_3``, expressed in terms of the
    unclipped sigmoids ``C_j = sigmoid(theta_j + eta)``:

    - ``y=0``: ``d(eta) = -(1 - C1)``, ``d(theta1) = -(1 - C1)``,
      ``d(theta2) = 0``, ``d(theta3) = 0``
    - ``y=1``: ``d(eta) = -(C2(1-C2) - C1(1-C1)) / (C2 - C1)``,
      ``d(theta1) = C1(1-C1) / (C2 - C1)``,
      ``d(theta2) = -C2(1-C2) / (C2 - C1)``, ``d(theta3) = 0``
    - ``y=2``: ``d(eta) = -(C3(1-C3) - C2(1-C2)) / (C3 - C2)``,
      ``d(theta2) = C2(1-C2) / (C3 - C2)``,
      ``d(theta3) = -C3(1-C3) / (C3 - C2)``, ``d(theta1) = 0``
    - ``y=3``: ``d(eta) = C3``, ``d(theta3) = C3``, ``d(theta1) = 0``,
      ``d(theta2) = 0``

    These are the exact derivatives of the *unclipped* negative
    log-likelihood; the clip epsilon (1e-12) is inactive for any
    realistic finite ``eta`` and thresholds, so in practice they are also
    the derivatives of the clipped objective :func:`_loss_and_gradient`
    actually minimizes. A dedicated regression test verifies them
    against a central finite-difference numerical gradient.

    Args:
        eta: The row's scalar linear predictor ``x . beta``.
        thresholds: The 3-vector of strictly increasing thresholds.
        y: The row's true outcome ordinal (0-3).

    Returns:
        A ``(d_eta, d_theta1, d_theta2, d_theta3)`` tuple of the four
        per-row gradient components.

    Raises:
        ValueError: If ``y`` is not in ``{0, 1, 2, 3}``.
    """
    if y not in (0, 1, 2, 3):
        raise ValueError(f"y must be one of 0..3, got {y!r}")
    c1 = _sigmoid(float(thresholds[0]) + eta)
    c2 = _sigmoid(float(thresholds[1]) + eta)
    c3 = _sigmoid(float(thresholds[2]) + eta)
    if y == 0:
        return -(1.0 - c1), -(1.0 - c1), 0.0, 0.0
    if y == 1:
        denom = c2 - c1
        d_eta = -(c2 * (1.0 - c2) - c1 * (1.0 - c1)) / denom
        return d_eta, c1 * (1.0 - c1) / denom, -c2 * (1.0 - c2) / denom, 0.0
    if y == 2:
        denom = c3 - c2
        d_eta = -(c3 * (1.0 - c3) - c2 * (1.0 - c2)) / denom
        return d_eta, 0.0, c2 * (1.0 - c2) / denom, -c3 * (1.0 - c3) / denom
    return c3, 0.0, 0.0, c3


def _loss_and_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    raw_thresholds: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the batch objective and its analytic gradient.

    Computes the full-batch objective
    ``mean(NLL) + (l2_lambda / 2) * sum(beta^2)`` (per-row NLL
    ``-log(P_y)`` with each ``P_j`` clipped into ``[eps, 1 - eps]``) and
    its analytic gradient w.r.t. both ``beta`` and the raw threshold
    parameters ``(a1, a2, a3)``. Per-row gradients come from
    :func:`_row_gradients` and are averaged over the batch; the L2 term
    is added to ``d/d(beta)`` only (thresholds are intercept-like and
    must not be shrunk). The threshold gradients are chain-ruled through
    the softplus reparameterization: ``d/d(a1) = sum of the three
    theta-gradients``, ``d/d(a2) = (d/d(theta2) + d/d(theta3)) *
    sigmoid(a2)``, ``d/d(a3) = d/d(theta3) * sigmoid(a3)``.

    Args:
        X: The (already-standardized) design matrix, ``(n, p)`` floats.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        beta: The current coefficient vector, length ``p``.
        raw_thresholds: The current unconstrained threshold parameters,
            length 3.
        l2_lambda: The L2 regularization strength on ``beta``
            (non-negative finite float).

    Returns:
        A ``(loss, grad_beta, grad_raw)`` tuple: ``loss`` the scalar
        batch objective; ``grad_beta`` the ``(p,)`` gradient of the
        objective w.r.t. ``beta`` (including the L2 term);
        ``grad_raw`` the ``(3,)`` gradient w.r.t. ``(a1, a2, a3)``.

    Raises:
        ValueError: If the input shapes are inconsistent (``X`` rows !=
            ``len(y)``), if ``y`` contains a value outside 0-3
            (propagated from :func:`_row_gradients`), or if
            ``raw_thresholds`` does not have exactly 3 elements
            (propagated from :func:`_thresholds_from_raw`).
    """
    n = X.shape[0]
    if n != len(y):
        raise ValueError(
            f"X has {n} rows but y has {len(y)} entries; they must match"
        )
    thresholds = _thresholds_from_raw(raw_thresholds)
    total_nll = 0.0
    grad_beta = np.zeros_like(beta, dtype=float)
    sum_t1 = 0.0
    sum_t2 = 0.0
    sum_t3 = 0.0
    for i in range(n):
        eta = float(np.dot(X[i], beta))
        probs = _category_probabilities(eta, thresholds)
        total_nll += -math.log(probs[y[i]])
        d_eta, d_t1, d_t2, d_t3 = _row_gradients(eta, thresholds, y[i])
        grad_beta += d_eta * X[i]
        sum_t1 += d_t1
        sum_t2 += d_t2
        sum_t3 += d_t3
    nll = total_nll / n
    grad_beta /= n
    sum_t1 /= n
    sum_t2 /= n
    sum_t3 /= n
    # L2 on beta only (thresholds are intercept-like and must not be
    # shrunk); the penalty is NOT averaged over the batch.
    l2_penalty = (l2_lambda / 2.0) * float(np.sum(beta**2))
    grad_beta += l2_lambda * beta
    # Chain rule through the softplus reparameterization
    # (d softplus(a)/da == sigmoid(a)).
    sig_a2 = _sigmoid(float(raw_thresholds[1]))
    sig_a3 = _sigmoid(float(raw_thresholds[2]))
    grad_raw = np.asarray(
        [
            sum_t1 + sum_t2 + sum_t3,
            (sum_t2 + sum_t3) * sig_a2,
            sum_t3 * sig_a3,
        ],
        dtype=float,
    )
    return nll + l2_penalty, grad_beta, grad_raw


def _gradient_descent(
    Xs: np.ndarray,
    y: np.ndarray,
    l2_lambda: float,
    max_iter: int,
    grad_tol: float,
    loss_tol: float,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[float, ...]]:
    """Run full-batch gradient descent with Armijo backtracking.

    Section E of the design. Starts from ``beta = 0`` and the
    marginal-derived raw thresholds (:func:`_initial_raw_thresholds`),
    then iterates: compute the loss/gradient at the current point; stop
    (converged) if ``||gradient|| < grad_tol`` or if the loss
    improvement between iterations drops below ``loss_tol``; otherwise
    try step size ``1.0``, halving up to :data:`_LINE_SEARCH_MAX_STEPS`
    times until the Armijo sufficient-decrease condition
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
        A ``(best_beta, best_raw, converged, n_iter, loss_trace)``
        tuple: the best coefficient vector, the best raw threshold
        parameters, whether the run converged, the number of iterations
        actually executed, and the per-iteration loss trace (non-
        increasing, length ``n_iter``).

    Raises:
        ValueError: If the ``y`` values are outside 0-3 or the shapes
            are inconsistent (propagated from
            :func:`_loss_and_gradient`), or if the label vector is
            empty (propagated from :func:`_initial_raw_thresholds`).
    """
    beta = np.zeros(Xs.shape[1], dtype=float)
    raw = _initial_raw_thresholds(np.bincount(y, minlength=_N_CATEGORIES))

    best_beta = beta.copy()
    best_raw = raw.copy()
    best_loss = float("inf")
    loss_trace: list[float] = []
    prev_loss: float | None = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        loss, grad_beta, grad_raw = _loss_and_gradient(
            Xs, y, beta, raw, l2_lambda
        )
        n_iter = iteration + 1
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
            best_raw = raw.copy()
        loss_trace.append(loss)

        grad_norm = math.sqrt(
            float(np.sum(grad_beta**2)) + float(np.sum(grad_raw**2))
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
            trial_raw = raw - step * grad_raw
            trial_loss, _, _ = _loss_and_gradient(
                Xs, y, trial_beta, trial_raw, l2_lambda
            )
            if trial_loss <= loss - _ARMIJO_C * step * grad_norm**2:
                beta = trial_beta
                raw = trial_raw
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

    return best_beta, best_raw, converged, n_iter, tuple(loss_trace)


@dataclass(frozen=True)
class OrdinalLogitModel:
    """A fitted proportional-odds ordinal logistic regression model.

    Holds the fitted parameters and the diagnostics needed to (a) make
    predictions via :func:`predict_proba`, (b) serialize/deserialize via
    :func:`to_dict` / :func:`from_dict`, and (c) interpret the fit via
    the coefficient report. ``coefficients`` has one entry per feature
    in ``feature_names`` (the :data:`FEATURE_NAMES` order); the sign
    convention is documented in the module docstring (positive favors
    team A). ``thresholds`` is strictly increasing
    (``theta_1 < theta_2 < theta_3``). ``standardizer_means`` /
    ``standardizer_stds`` describe the *training* design matrix (the
    second leakage boundary; see the module docstring). ``loss_trace``
    is a live-fit diagnostic (per-iteration loss, non-increasing) that
    is deliberately *not* persisted by :func:`to_dict` — a deserialized
    model carries an empty trace.

    Attributes:
        coefficients: The 13-vector of fitted coefficients.
        thresholds: The 3-vector of strictly increasing thresholds.
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
    thresholds: np.ndarray
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
) -> OrdinalLogitModel:
    """Fit the ordinal logistic regression by Armijo gradient descent.

    Assembles a complete :class:`OrdinalLogitModel` from a raw feature
    matrix: fits the per-feature z-score standardizer on the *training*
    matrix only (the second leakage boundary; see :func:`fit_standardizer`)
    and transforms with it, then runs full-batch gradient descent with
    Armijo backtracking (see :func:`_gradient_descent`) initialized at
    ``beta = 0`` and thresholds derived from the training label marginal
    (see :func:`_initial_raw_thresholds`). The returned model carries
    the training standardizer so :func:`predict_proba` can standardize
    later rows with the exact training-population statistics.

    Args:
        X: The raw (unstandardized) training design matrix, ``(n, 13)``
            floats in :data:`FEATURE_NAMES` order — the output of
            :func:`build_feature_vector` over the training rows. The
            standardizer is fit on this matrix inside this function.
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        l2_lambda: L2 regularization strength on ``beta`` only; must be
            non-negative finite (default ``1.0`` — the documented
            conservative default given the thin OT categories, not
            CV-tuned in this task).
        max_iter: Cap on gradient-descent iterations (default 2000). If
            the cap is hit without convergence, ``fit`` returns the best
            point found with ``converged=False`` rather than raising.
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).

    Returns:
        A frozen :class:`OrdinalLogitModel` with the fitted
        ``coefficients``/``thresholds``, the training standardizer, the
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
        raise ValueError("cannot fit an ordinal model on an empty label vector")
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
    beta, raw, converged, n_iter, trace = _gradient_descent(
        Xs, y_arr, l2, max_it, g_tol, l_tol
    )
    thresholds = _thresholds_from_raw(raw)
    return OrdinalLogitModel(
        coefficients=beta,
        thresholds=thresholds,
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
    model: OrdinalLogitModel,
) -> tuple[float, float, float, float]:
    """Predict the four category probabilities for one feature vector.

    Applies the model's stored standardizer (training-population means/
    stds — the second leakage boundary, applied identically to every
    later row), computes ``eta = x_standardized . beta``, and returns
    the four :data:`OUTCOME_LABELS`-ordered probabilities via
    :func:`_category_probabilities` (each clipped into ``[eps, 1-eps]``,
    so the tuple is a valid, scorable simplex even for extreme inputs).

    Args:
        x: A raw feature vector, length 13 in :data:`FEATURE_NAMES`
            order (the output of :func:`build_feature_vector`).
        model: The fitted model whose stored standardizer and
            coefficients are applied.

    Returns:
        The 4-tuple ``(p_a_regulation, p_a_ot, p_b_ot, p_b_regulation)``
        of ``float`` probabilities in :data:`OUTCOME_LABELS` order,
        each in ``[eps, 1 - eps]`` and summing to approximately 1.

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
    eta = float(np.dot(xs, model.coefficients))
    probs = _category_probabilities(eta, model.thresholds)
    return (
        float(probs[0]),
        float(probs[1]),
        float(probs[2]),
        float(probs[3]),
    )


def _coefficient_report(model: OrdinalLogitModel) -> list[dict]:
    """Build the human-readable coefficient report for a fitted model.

    One entry per feature: ``{"feature": name, "coefficient": value,
    "direction": label}`` where ``direction`` is derived from the
    coefficient's sign under the module-docstring convention — a positive
    coefficient favors team A (increasing the feature shifts probability
    mass toward A's categories), a negative one favors team B, and an
    exactly-zero one favors neither. Entries are sorted by
    ``abs(coefficient)`` descending so the most influential features read
    first.

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


def to_dict(model: OrdinalLogitModel) -> dict:
    """Serialize a fitted model to a plain JSON-serializable dict.

    Produces the artifact dict the training driver writes: ``feature_names``,
    ``coefficients``, ``thresholds``, ``standardizer_means``,
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
        "coefficients": [float(c) for c in model.coefficients],
        "thresholds": [float(t) for t in model.thresholds],
        "standardizer_means": [float(m) for m in model.standardizer_means],
        "standardizer_stds": [float(s) for s in model.standardizer_stds],
        "l2_lambda": float(model.l2_lambda),
        "converged": bool(model.converged),
        "n_iter": int(model.n_iter),
        "final_loss": float(model.final_loss),
        "n_train": int(model.n_train),
        "coefficient_report": _coefficient_report(model),
    }


def from_dict(d: dict) -> OrdinalLogitModel:
    """Deserialize a fitted model from a to_dict-produced dict.

    Reconstructs an :class:`OrdinalLogitModel` from the plain dict
    :func:`to_dict` produces (or from ``json.loads`` of the artifact the
    training driver writes). Arrays are rebuilt as numpy arrays; shape
    consistency is validated (coefficients/means/stds must all line up
    with ``feature_names``, thresholds must have exactly 3 entries). The
    ``coefficient_report`` key is ignored on read (it is derived, not
    stored) and ``loss_trace`` is empty for a deserialized model. No
    file I/O happens here.

    Args:
        d: The dict to load; must carry the ten parameter/diagnostic
            keys (``coefficient_report`` optional, ignored).

    Returns:
        An :class:`OrdinalLogitModel` whose parameters reproduce the
        serialized ones exactly (``feature_names`` as a tuple, arrays as
        ``float`` numpy arrays, diagnostics as plain scalars).

    Raises:
        KeyError: If a required key is absent (propagated from dict
            indexing).
        ValueError: If the shapes are inconsistent (coefficient count !=
            feature count, means/stds length != coefficient count, or
            thresholds not length 3), or if a numeric field cannot be
            coerced.
    """
    feature_names = tuple(str(name) for name in d["feature_names"])
    coefficients = np.asarray(d["coefficients"], dtype=float)
    thresholds = np.asarray(d["thresholds"], dtype=float)
    means = np.asarray(d["standardizer_means"], dtype=float)
    stds = np.asarray(d["standardizer_stds"], dtype=float)
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"coefficients has {len(coefficients)}; they must match"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"expected 3 thresholds, got {len(thresholds)}"
        )
    if len(means) != len(coefficients) or len(stds) != len(coefficients):
        raise ValueError(
            f"standardizer means ({len(means)}) / stds ({len(stds)}) must "
            f"each have one entry per coefficient ({len(coefficients)})"
        )
    return OrdinalLogitModel(
        coefficients=coefficients,
        thresholds=thresholds,
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
    model: OrdinalLogitModel,
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
    :func:`build_feature_vector` with the closed-over table, and returns
    :func:`predict_proba`'s 4-tuple.

    This is only correct because callers (the M19 harness and the
    evaluation driver) always invoke the returned closure with the same
    ``matches_df``/``maps_df`` that came from the same
    ``<output_dir>/<version>`` the closed-over ``player_map_stats_df``
    was loaded from — a mismatched pairing would silently misalign,
    exactly as already implicit in every existing model/table pairing in
    this codebase.

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
    ) -> tuple[float, float, float, float]:
        """Predict the four category probabilities for one held-out map.

        Computes the raw 13-feature vector via
        :func:`build_feature_vector` (using the closed-over
        ``player_map_stats_df`` — the table the generic interface does
        not pass) and returns :func:`predict_proba`'s 4-tuple in
        :data:`OUTCOME_LABELS` order. See :func:`make_model_fn`'s
        docstring for the closed-table contract.

        Args:
            team1_id: The queried team1's stable id ("A").
            team2_id: The queried team2's stable id ("B").
            map_name: The map to predict for.
            date: The as-of cutoff (the map's own match date).
            matches_df: The full materialised ``matches`` table from the
                same dataset version the closed-over ``player_map_stats_df``
                was loaded from.
            maps_df: The full materialised ``maps`` table from the same
                version.

        Returns:
            The 4-tuple of probabilities in :data:`OUTCOME_LABELS`
            order, summing to approximately 1.

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
def total_log_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    model: OrdinalLogitModel,
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


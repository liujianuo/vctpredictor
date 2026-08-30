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
  passes it into :func:`build_feature_vector`), rather than by changing
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

from features import closeness, elo, h2h_context, map_win_rate, player_form
from utils import asof

# The 11 features in the fixed order the model consumes, one shared
# coefficient each. Team-specific features are expressed as A-minus-B
# differences (A = team1_id, B = team2_id, matching the
# ``drivers.labels``/``models.four_way_baseline`` convention) so one
# coefficient vector applies regardless of which side is "A" in a given
# match row. See the module docstring and :func:`build_feature_vector`
# for the exact per-feature recipe and the missing-value fallbacks.
FEATURE_NAMES = (
    "map_win_rate_diff",
    "elo_differential",
    "close_map_freq_diff",
    "ot_rate_diff",
    "map_round_margin_variance",
    "acs_form_diff",
    "rating_form_diff",
    "h2h_win_rate_centered",
    "event_stage",
    "days_since_diff",
    "roster_decay_diff",
)

# The four outcome categories in ordinal order, mirroring
# ``drivers.labels.OUTCOME_LABELS`` (documented, deliberately *not*
# imported: this module must not depend on ``drivers/``). Index 0 is
# "A-regulation", 1 "A-OT", 2 "B-OT", 3 "B-regulation" — the order
# :func:`predict_proba` returns and ``utils.scoring``'s metrics expect.
OUTCOME_LABELS = ("A-regulation", "A-OT", "B-OT", "B-regulation")

# Clip epsilon for the category probabilities before the log in the
# negative log-likelihood (same epsilon convention as
# ``features.map_win_rate._PROB_CLIP_EPS``). The strict threshold
# ordering guarantees every ``P_j`` is strictly positive for any finite
# ``eta``, so the clip is defensive floor/sky coverage for extreme
# inputs, not a live-data fix.
_PROB_CLIP_EPS = 1e-12

# Armijo line-search constants: the sufficient-decrease parameter and
# the maximum number of step halvings (step starts at 1.0, halving up to
# this many times) before an iteration is declared unable to make
# progress.
_ARMIJO_C = 1e-4
_LINE_SEARCH_MAX_STEPS = 50

# Number of categories (the length of OUTCOME_LABELS / the K of the
# ordinal model).
_N_CATEGORIES = 4


def _sigmoid(x: float) -> float:
    """Return the logistic sigmoid of ``x``, computed stably.

    ``1 / (1 + exp(-x))`` evaluated in the numerically-stable branch
    that avoids ``exp(-x)`` overflowing to ``inf`` for large negative
    ``x`` (and, symmetrically, avoids ``exp(x)`` overflowing for large
    positive ``x``). ``sigmoid`` is strictly increasing, so with
    strictly increasing thresholds the three ``C_j`` stay strictly
    ordered.

    Args:
        x: The scalar argument (typically ``theta_j + eta``).

    Returns:
        The sigmoid value as a ``float`` in ``(0, 1)`` (asymptotically
        approaching but never reaching 0 and 1 for finite ``x``).

    Raises:
        Nothing (the formula is total for any finite input).
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
        coefficients: The 11-vector of fitted coefficients.
        thresholds: The 3-vector of strictly increasing thresholds.
        standardizer_means: Per-feature training-column means (length
            11).
        standardizer_stds: Per-feature training-column stds (length 11;
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


def _validate_l2_lambda(l2_lambda) -> float:
    """Validate the L2 strength and return it as a float.

    The L2 strength must be a non-negative finite real: a negative value
    would anti-regularize (grow coefficients unboundedly) and NaN/inf
    would poison every gradient step.

    Args:
        l2_lambda: The proposed L2 strength.

    Returns:
        ``l2_lambda`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a float, or if the result
            is NaN, infinite, or negative.
    """
    try:
        value = float(l2_lambda)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"l2_lambda must be a non-negative finite real number, got {l2_lambda!r}"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"l2_lambda must be a non-negative finite real number, got {l2_lambda!r}"
        )
    return value


def _validate_positive_int(value, name: str) -> int:
    """Validate a positive integer parameter and return it as an ``int``.

    Used for ``max_iter``: rejects bools (which are int-coercible but
    never intended), non-integral values, and non-positive values.

    Args:
        value: The proposed parameter value.
        name: The parameter name for the error message.

    Returns:
        ``value`` as a positive ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not integer-valued, or
            ``<= 0``.
    """
    if type(value) is bool:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _validate_positive_float(value, name: str) -> float:
    """Validate a positive finite parameter and return it as a ``float``.

    Used for ``grad_tol``/``loss_tol``: both are convergence tolerances
    that must be strictly positive and finite (zero or negative would
    disable or invert the convergence checks; NaN/inf would poison them).

    Args:
        value: The proposed parameter value.
        name: The parameter name for the error message.

    Returns:
        ``value`` as a positive finite ``float``.

    Raises:
        ValueError: If it cannot be coerced to a float, or if the result
            is NaN, infinite, or ``<= 0``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive real number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite real number, got {value!r}"
        )
    return result


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
        X: The raw (unstandardized) training design matrix, ``(n, 11)``
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
        x: A raw feature vector, length 11 in :data:`FEATURE_NAMES`
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


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the per-feature z-score standardizer on a design matrix.

    Computes the per-column ``(mean, std)`` of ``X`` (population std,
    matching numpy's ``std`` convention). A zero-variance column's std
    is replaced with ``1.0`` rather than raising, so
    :func:`apply_standardizer` turns that column into ``0.0`` for every
    row (a degenerate/constant training column contributes no signal);
    see the module docstring's standardization bullet.

    Args:
        X: The design matrix to compute statistics from, ``(n, p)``
            floats.

    Returns:
        A ``(means, stds)`` tuple of ``(p,)`` numpy arrays: the
        per-column means and the per-column stds (zero-variance columns
        carry ``1.0``).

    Raises:
        ValueError: If ``X`` is empty or is not a 2-D array (no
            per-column statistics exist for an empty matrix).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(
            "fit_standardizer requires a non-empty 2-D design matrix, got "
            f"shape {X_arr.shape}"
        )
    means = X_arr.mean(axis=0)
    stds = X_arr.std(axis=0)
    zero_variance = stds == 0.0
    stds = np.where(zero_variance, 1.0, stds)
    return means, stds


def apply_standardizer(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Standardize a design matrix with pre-computed (mean, std) statistics.

    Applies ``(X - means) / stds`` column-wise. With the
    :func:`fit_standardizer` zero-variance guard (``std == 1.0`` for a
    constant column), a zero-variance column standardizes to ``0.0`` for
    every row instead of dividing by zero.

    Args:
        X: The design matrix to standardize, ``(n, p)`` floats.
        means: The per-column means, length ``p``.
        stds: The per-column stds, length ``p`` (all non-zero; see
            :func:`fit_standardizer`).

    Returns:
        The standardized matrix as a ``(n, p)`` numpy array of ``float``.

    Raises:
        ValueError: If ``means``/``stds`` do not have exactly as many
            entries as ``X`` has columns (a shape mismatch would
            silently misalign columns).
    """
    X_arr = np.asarray(X, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(
            f"apply_standardizer requires a 2-D design matrix, got shape {X_arr.shape}"
        )
    if means_arr.shape[0] != X_arr.shape[1] or stds_arr.shape[0] != X_arr.shape[1]:
        raise ValueError(
            f"standardizer has {means_arr.shape[0]} means / {stds_arr.shape[0]} "
            f"stds but X has {X_arr.shape[1]} columns; they must match"
        )
    return (X_arr - means_arr) / stds_arr


def _match_id_for(
    team1_id: str,
    team2_id: str,
    date: str,
    matches_df: pd.DataFrame,
) -> str:
    """Resolve the unique match row for a (team1, team2, date) triple.

    Feature ``event_stage`` needs the *match*'s ``event_name`` (and
    therefore its ``match_id``), but the fixed feature-builder signature
    carries no ``match_id`` argument. This helper recovers it by locating
    the unique match row matching the two team ids and the as-of date.
    At v1 scale the ``(team1_id, team2_id, date)`` triple is unique per
    match (verified against the real data: every match has a distinct
    timestamp), so the lookup is unambiguous; the guard below fails
    loudly if a future dataset makes it ambiguous, rather than silently
    picking an arbitrary row.

    Args:
        team1_id: The queried team1's stable id ("A").
        team2_id: The queried team2's stable id ("B").
        date: The match's own date string (the as-of cutoff).
        matches_df: The materialised ``matches`` table (needs
            ``team1_id``, ``team2_id``, ``date``, ``match_id``).

    Returns:
        The ``match_id`` of the unique matching row, as a ``str``.

    Raises:
        KeyError: If ``matches_df`` lacks a required column (propagated
            from :func:`utils.asof.require_columns`).
        ValueError: If zero or more than one match row matches the
            triple (the event-stage feature would be undefined or
            ambiguous).
    """
    asof.require_columns(
        matches_df,
        (asof.TEAM1_ID_COL, asof.TEAM2_ID_COL, asof.DATE_COL, asof.MATCH_ID_COL),
        "matches_df",
    )
    matches = matches_df[
        (matches_df[asof.TEAM1_ID_COL] == team1_id)
        & (matches_df[asof.TEAM2_ID_COL] == team2_id)
        & (matches_df[asof.DATE_COL] == date)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one match for (team1_id={team1_id!r}, "
            f"team2_id={team2_id!r}, date={date!r}) to resolve the "
            f"event-stage match_id, found {len(matches)}"
        )
    return matches.iloc[0][asof.MATCH_ID_COL]


def build_feature_vector(
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
) -> np.ndarray:
    """Build the 11-feature vector for one map, in FEATURE_NAMES order.

    Computes every feature exactly as section A specifies, each as of
    the map's own match ``date`` (strict ``<`` boundary, inherited
    unchanged from each feature module's own ``utils.asof`` usage — this
    model adds no new date filtering), with team-specific features
    expressed as A-minus-B differences (A = ``team1_id``, B =
    ``team2_id``). The missing-value fallback policy from the module
    docstring is applied per-side *before* differencing. No
    normalization/standardization happens here — the returned values are
    raw feature values; standardization is :func:`fit_standardizer` /
    :func:`apply_standardizer`'s job (training-side only).

    Args:
        team1_id: The queried team1's stable id ("A").
        team2_id: The queried team2's stable id ("B").
        map_name: The map to predict for (normalized inside each feature
            estimator, so case/whitespace never break a match).
        date: The as-of cutoff; rows dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needed by the M16/M17 features; required because the fixed
            model interface does not pass it — see the module docstring
            and :func:`make_model_fn`).

    Returns:
        An 11-vector numpy array of ``float`` in :data:`FEATURE_NAMES`
        order: ``map_win_rate_diff, elo_differential,
        close_map_freq_diff, ot_rate_diff, map_round_margin_variance,
        acs_form_diff, rating_form_diff, h2h_win_rate_centered,
        event_stage, days_since_diff, roster_decay_diff``.

    Raises:
        ValueError: If a feature's as-of history contains a null/tied
            score or a null/unparseable date, if ``k``/window/threshold
            hyperparameters are invalid, if the match row for
            ``(team1_id, team2_id, date)`` is not unique (see
            :func:`_match_id_for`), if a ``player_map_stats`` team name
            matches neither side of its match, or if the two side names
            of a match collide (all propagated from the individual
            feature modules).
        KeyError: If any table lacks a required column (propagated from
            the individual feature modules / :func:`_match_id_for`).
        TypeError: If the query date is list-like (propagated from
            ``utils.asof`` via the feature modules).
        ConfigError: If ``map_name`` or an as-of map's ``map_name`` is
            not a string (propagated from
            :func:`utils.config.normalize_map_name` via the feature
            modules).
    """
    # 1. map_win_rate_diff — shrunk per-map win rate, A minus B.
    map_win_a = map_win_rate.team_map_win_rate(
        team1_id, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    ).mean
    map_win_b = map_win_rate.team_map_win_rate(
        team2_id, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    ).mean
    map_win_rate_diff = map_win_a - map_win_b

    # 2. elo_differential — signed A-minus-B already, from one shared
    #    league replay.
    elo_diff = elo.elo_differential(
        team1_id,
        team2_id,
        date,
        matches_df,
        maps_df,
        k=elo.DEFAULT_K,
        initial_rating=elo.INITIAL_RATING,
    ).differential

    # 3. close_map_freq_diff — unshrunk close-map frequency, A minus B.
    close_a = closeness.team_close_map_frequency(
        team1_id, date, matches_df, maps_df
    ).rate
    close_b = closeness.team_close_map_frequency(
        team2_id, date, matches_df, maps_df
    ).rate
    close_map_freq_diff = close_a - close_b

    # 4. ot_rate_diff — heavily-shrunk team OT rate, A minus B.
    ot_a = closeness.team_ot_rate(
        team1_id,
        date,
        matches_df,
        maps_df,
        k=closeness.DEFAULT_OT_K,
    ).mean
    ot_b = closeness.team_ot_rate(
        team2_id,
        date,
        matches_df,
        maps_df,
        k=closeness.DEFAULT_OT_K,
    ).mean
    ot_rate_diff = ot_a - ot_b

    # 5. map_round_margin_variance — match-level (not a team diff);
    #    NaN (n <= 1) replaced with 0.0 per the documented fallback.
    margin_var = closeness.map_round_margin_variance(
        map_name, date, matches_df, maps_df
    ).variance
    if math.isnan(margin_var):
        map_round_margin_variance = 0.0
    else:
        map_round_margin_variance = float(margin_var)

    # 6/7. acs_form_diff / rating_form_diff — recency-weighted form,
    #    A minus B, with the either-side-None -> 0.0 fallback.
    form_a = player_form.team_player_form(
        team1_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        n=player_form.DEFAULT_FORM_WINDOW,
        decay_rate=player_form.DEFAULT_DECAY_RATE,
    )
    form_b = player_form.team_player_form(
        team2_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        n=player_form.DEFAULT_FORM_WINDOW,
        decay_rate=player_form.DEFAULT_DECAY_RATE,
    )
    if form_a.acs.mean is None or form_b.acs.mean is None:
        acs_form_diff = 0.0
    else:
        acs_form_diff = form_a.acs.mean - form_b.acs.mean
    if form_a.rating.mean is None or form_b.rating.mean is None:
        rating_form_diff = 0.0
    else:
        rating_form_diff = form_a.rating.mean - form_b.rating.mean

    # 8. h2h_win_rate_centered — shrunk H2H mean minus the 0.5 prior
    #    (0 = no/even history, matching the estimator's own
    #    full-shrinkage default).
    h2h_mean = h2h_context.team_pair_h2h(
        team1_id,
        team2_id,
        date,
        matches_df,
        maps_df,
        k=h2h_context.DEFAULT_H2H_K,
    ).mean
    h2h_win_rate_centered = h2h_mean - h2h_context.H2H_PRIOR

    # 9. event_stage — the match's own stage (match-level int), resolved
    #    through the unique (team1, team2, date) match row.
    match_id = _match_id_for(team1_id, team2_id, date, matches_df)
    event_stage = float(
        h2h_context.match_event_stage(match_id, matches_df)
    )

    # 10. days_since_diff — rest gap, A minus B, with None (unseen team
    #     / no strictly-prior match) treated as 0 per the fallback.
    days_a = h2h_context.days_since_last_match(team1_id, date, matches_df)
    days_b = h2h_context.days_since_last_match(team2_id, date, matches_df)
    days_since_diff = (days_a if days_a is not None else 0) - (
        days_b if days_b is not None else 0
    )

    # 11. roster_decay_diff — post-change decay multiplier, A minus B,
    #     with None (changed is None/False) treated as 1.0 per the
    #     fallback.
    roster_a = h2h_context.team_roster_change(
        team1_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        jaccard_threshold=h2h_context.DEFAULT_JACCARD_THRESHOLD,
        half_life_days=h2h_context.DEFAULT_HALF_LIFE_DAYS,
    )
    roster_b = h2h_context.team_roster_change(
        team2_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        jaccard_threshold=h2h_context.DEFAULT_JACCARD_THRESHOLD,
        half_life_days=h2h_context.DEFAULT_HALF_LIFE_DAYS,
    )
    decay_a = roster_a.decay_multiplier if roster_a.decay_multiplier is not None else 1.0
    decay_b = roster_b.decay_multiplier if roster_b.decay_multiplier is not None else 1.0
    roster_decay_diff = decay_a - decay_b

    return np.asarray(
        [
            map_win_rate_diff,
            elo_diff,
            close_map_freq_diff,
            ot_rate_diff,
            map_round_margin_variance,
            acs_form_diff,
            rating_form_diff,
            h2h_win_rate_centered,
            event_stage,
            days_since_diff,
            roster_decay_diff,
        ],
        dtype=float,
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

        Computes the raw 11-feature vector via
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

"""Proportional-odds diagnostic for the M20 ordinal model (roadmap M21).

Implements a **documented, hand-rolled approximation of the Brant
test's "per-category fit comparison"** — three independent binary
logistic fits, one per cutpoint, whose coefficient agreement is then
assessed by sign-instability counting plus an AIC/BIC comparison of the
two *actual* four-class models. This is **not** the classical Brant
chi-square statistic with a p-value: this repo has no ``scipy`` in
``requirements.txt``, and computing a valid Brant p-value would need a
hand-rolled regularized incomplete gamma function (``chi2.sf``/``chi2.cdf``),
which was judged out of scope for this task. This must be stated
verbatim so no later milestone mistakes this for a rigorous
significance test. A violation finding is what triggers M23 (hurdle
model) per the roadmap; this module only *reports* the finding.

The proportional-odds assumption M20 makes is that a *single* shared
coefficient vector ``beta`` works across all three cumulative
splits ``{P(Y >= j)}`` for ``j = 1, 2, 3``. The diagnostic probes that
restriction two complementary ways:

- **Step 1 — three independent binary logistic fits, one per
  cutpoint.** For ``j = 1, 2, 3``, define the binary label
  ``z_i = 1`` if ``outcome_ordinal_i >= j`` else ``0`` and fit a plain
  binary logistic regression (intercept ``alpha`` plus ``beta`` (11,),
  **not** sharing parameters across cutpoints) on the same standardized
  design matrix (fit once, reused for all three cutpoints — the matrix
  is identical across cutpoints, only the binary label differs, so
  there is no reason to refit the standardizer three times). The link,
  gradient, and Armijo optimizer are hand-rolled here (a third
  independent implementation, per the same "no premature shared
  optimizer" rule that kept ``models.multinomial_logit``'s optimizer
  independent); all shared constants come from ``models._shared``.
- **Step 2 — per-feature sign instability.** Proportional odds claims
  one shared coefficient per feature works for every cutpoint, so a
  feature whose three independently-estimated coefficient *signs* are
  not all equal is direct evidence the restriction does not hold for
  that feature. ``sign_instability_count`` counts such features
  (``0..11``); a feature where all three estimates are exactly zero is
  trivially stable, not unstable.
- **Step 3 — quantitative AIC/BIC comparison of the two actual
  four-class models.** ``ll`` for each arm comes from
  :func:`models.ordinal_logit.total_log_likelihood` /
  :func:`models.multinomial_logit.total_log_likelihood` on the same
  training rows; ``k_ordinal = 14`` (11 coefficients + 3 thresholds),
  ``k_multinomial = 36`` (33 coefficients + 3 intercepts); ``aic =
  -2*ll + 2*k``, ``bic = -2*ll + k*log(n_train)``. Lower is better for
  both. **AIC/BIC are valid for comparing non-nested models** (unlike a
  likelihood-ratio chi-square, which requires nesting the ordinal model
  does not satisfy relative to the multinomial's different link
  function — a LRT p-value between these two models is deliberately
  never computed or reported). BIC's ``k * log(n)`` penalty is exactly
  what surfaces the multinomial's 2x-parameter overfitting risk at
  ``n_train = 209``.
- **Step 4 — the verdict.** ``proportional_odds_verdict = "violated"
  if (sign_instability_count > 0 or bic_favors_multinomial) else
  "holds"`` — a simple, fully-documented rule (any single sign flip
  across cutpoints, or the richer model actually justifying its extra
  parameters under BIC's penalty, counts as evidence against the
  shared-coefficient restriction). The exact rule text is embedded in
  the report JSON as ``verdict_rule`` so the artifact is
  self-documenting without a source read.

Place in the dependency DAG: this module sits alongside
``evaluation.harness`` — pure and dependency-light (no I/O, no CLI; the
CLI lives in ``drivers/diagnose_proportional_odds.py``), depending
downward on ``models.*`` / ``features.*`` / ``utils.*`` only, never on
``drivers/`` or a sibling ``evaluation/`` module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from models import multinomial_logit, ordinal_logit
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
    fit_standardizer,
)

# The free-parameter counts of the two four-class arms, hard-coded here
# (recorded in the module docstring): the ordinal arm has 11 shared
# coefficients + 3 thresholds = 14; the multinomial arm has 3 * 11
# coefficients + 3 intercepts = 36.
_K_ORDINAL = 14
_K_MULTINOMIAL = 36

# The exact verdict rule, embedded verbatim in the report JSON as
# ``verdict_rule`` so the artifact is self-documenting.
_VERDICT_RULE = (
    "proportional_odds_verdict = 'violated' if (sign_instability_count > 0 "
    "or bic_favors_multinomial) else 'holds'"
)


@dataclass(frozen=True)
class BinaryLogitModel:
    """A fitted binary logistic regression for one ordinal cutpoint.

    A plain sigmoid-link binary logit (intercept ``intercept`` plus
    coefficient vector ``coefficients`` (11,)) fit on one cumulative
    binary split ``z_i = 1 {outcome_ordinal_i >= cutpoint}``. The
    ``standardizer_means``/``standardizer_stds`` describe the shared
    training standardizer and are stored on every entry for
    self-containedness (they are identical across the three cutpoints —
    cheap to store and avoids a caller needing a second object to
    interpret the coefficients). ``cutpoint`` is 1/2/3 (which cumulative
    split this model governs).

    Attributes:
        coefficients: The 11-vector of fitted coefficients.
        intercept: The fitted scalar intercept.
        standardizer_means: Per-feature training-column means (length
            11).
        standardizer_stds: Per-feature training-column stds (length 11;
            a zero-variance column's std is ``1.0`` per the guard).
        cutpoint: The cutpoint this model was fit for (1, 2 or 3).
        converged: Whether gradient descent converged.
        n_iter: Number of gradient-descent iterations executed.
        final_loss: The objective value at the returned point.
        n_train: Number of rows fit on.
    """

    coefficients: np.ndarray
    intercept: float
    standardizer_means: np.ndarray
    standardizer_stds: np.ndarray
    cutpoint: int
    converged: bool
    n_iter: int
    final_loss: float
    n_train: int


def _binary_loss_and_gradient(
    Xs: np.ndarray,
    z: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    l2_lambda: float,
) -> tuple[float, float, np.ndarray]:
    """Return the binary-logit batch objective and its analytic gradient.

    Computes ``mean(NLL) + (l2_lambda / 2) * sum(beta ** 2)`` where the
    per-row NLL is the Bernoulli cross-entropy ``-(z * log(p) + (1 - z)
    * log(1 - p))`` with ``p = sigmoid(alpha + xs . beta)`` clipped into
    ``[eps, 1 - eps]`` before the log. The per-row gradient of the NLL
    w.r.t. ``eta`` is ``p - z`` (the exact derivative of the unclipped
    NLL; the clip epsilon is inactive for realistic finite inputs), so
    ``d(alpha) = mean(p - z)`` and ``d(beta) = mean((p - z) * xs) +
    l2_lambda * beta`` (the L2 term on ``beta`` only — never on the
    intercept, matching the "regularize slopes, not intercepts"
    convention of both four-class arms).

    Args:
        Xs: The (already-standardized) design matrix, ``(n, 11)``
            floats.
        z: The binary labels, ``(n,)`` ints in ``{0, 1}``.
        alpha: The current scalar intercept.
        beta: The current coefficient vector, length ``p``.
        l2_lambda: The L2 regularization strength on ``beta`` only.

    Returns:
        A ``(loss, grad_alpha, grad_beta)`` tuple: the scalar batch
        objective, its gradient w.r.t. ``alpha``, and its gradient
        w.r.t. ``beta`` (including the L2 term).

    Raises:
        ValueError: If ``Xs`` rows and ``len(z)`` differ.
    """
    n = Xs.shape[0]
    if n != len(z):
        raise ValueError(
            f"Xs has {n} rows but z has {len(z)} entries; they must match"
        )
    total_nll = 0.0
    grad_alpha = 0.0
    grad_beta = np.zeros_like(beta, dtype=float)
    for i in range(n):
        eta = alpha + float(np.dot(Xs[i], beta))
        p = _sigmoid(eta)
        p_clipped = min(max(p, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
        total_nll += -(z[i] * math.log(p_clipped) + (1 - z[i]) * math.log(1.0 - p_clipped))
        d_eta = p - z[i]
        grad_alpha += d_eta
        grad_beta += d_eta * Xs[i]
    nll = total_nll / n
    grad_alpha /= n
    grad_beta /= n
    l2_penalty = (l2_lambda / 2.0) * float(np.sum(beta**2))
    grad_beta += l2_lambda * beta
    return nll + l2_penalty, grad_alpha, grad_beta


def _gradient_descent_binary(
    Xs: np.ndarray,
    z: np.ndarray,
    l2_lambda: float,
    max_iter: int,
    grad_tol: float,
    loss_tol: float,
) -> tuple[np.ndarray, float, bool, int, tuple[float, ...]]:
    """Run full-batch gradient descent with Armijo backtracking.

    The binary logit's own optimizer (a third independent Armijo
    implementation, per the module docstring's no-shared-optimizer
    rule). Starts from ``beta = 0`` and ``alpha = logit(clip(mean(z),
    eps, 1 - eps))`` (the intercept that reproduces the marginal binary
    rate at ``beta = 0``), then iterates: compute the loss/gradient;
    stop (converged) if the combined gradient norm over ``beta`` and
    ``alpha`` is below ``grad_tol`` or the loss improvement drops below
    ``loss_tol``; otherwise try step ``1.0`` halving up to
    :data:`_LINE_SEARCH_MAX_STEPS` times until the Armijo
    sufficient-decrease condition holds, then take that step. A
    line-search failure or ``max_iter`` exhaustion stops the run with
    ``converged=False`` and returns the best point found — a
    non-converged fit is a valid (if suboptimal) model, not an error.

    Args:
        Xs: The already-standardized design matrix, ``(n, p)`` floats.
        z: The binary labels, ``(n,)`` ints in ``{0, 1}``.
        l2_lambda: The L2 strength (validated non-negative finite).
        max_iter: The iteration cap (validated positive int).
        grad_tol: The gradient-norm convergence tolerance.
        loss_tol: The loss-improvement convergence tolerance.

    Returns:
        A ``(best_beta, best_alpha, converged, n_iter, loss_trace)``
        tuple: the best coefficient vector, the best intercept, whether
        the run converged, the number of iterations executed, and the
        non-increasing per-iteration loss trace (length ``n_iter``).

    Raises:
        ValueError: If ``z`` contains a value outside ``{0, 1}`` or the
            label vector is empty.
    """
    if len(z) == 0:
        raise ValueError("cannot run gradient descent on an empty label vector")
    if set(np.unique(z).tolist()) - {0, 1}:
        raise ValueError(
            f"z must contain only binary labels 0/1, got values "
            f"{sorted(set(np.unique(z).tolist()))}"
        )
    p = Xs.shape[1]
    beta = np.zeros(p, dtype=float)
    mean_z = float(np.mean(z))
    p0 = min(max(mean_z, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
    alpha = math.log(p0 / (1.0 - p0))

    best_beta = beta.copy()
    best_alpha = alpha
    best_loss = float("inf")
    loss_trace: list[float] = []
    prev_loss: float | None = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        loss, grad_alpha, grad_beta = _binary_loss_and_gradient(
            Xs, z, alpha, beta, l2_lambda
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
            trial_loss, _, _ = _binary_loss_and_gradient(
                Xs, z, trial_alpha, trial_beta, l2_lambda
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


def fit_binary_logit(
    Xs: np.ndarray,
    z: np.ndarray,
    cutpoint: int,
    l2_lambda: float = 1.0,
    max_iter: int = 2000,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-10,
    standardizer_means: np.ndarray | None = None,
    standardizer_stds: np.ndarray | None = None,
) -> BinaryLogitModel:
    """Fit one per-cutpoint binary logistic regression.

    Takes the *already-standardized* design matrix (per the "fit the
    standardizer once, reuse for all three cutpoints" note in the module
    docstring — :func:`fit_cutpoint_binary_models` handles that) and the
    binary label vector, runs the module's own Armijo gradient descent
    (:func:`_gradient_descent_binary`), and assembles the resulting
    :class:`BinaryLogitModel`. The standardizer statistics are *not*
    recomputed from the raw matrix here (the matrix ``Xs`` is already
    standardized): pass the shared training-population means/stds via
    ``standardizer_means``/``standardizer_stds`` (the caller's single
    :func:`models._shared.fit_standardizer` result over the raw
    training matrix) so the returned model is self-contained — every
    entry stores the same shared values, which a caller needs to
    interpret the coefficients in raw feature units. If they are not
    passed, they are derived from ``Xs`` itself (the ``(mean, std)`` of
    the standardized columns, i.e. approximately 0/1) as a documented
    fallback for callers who only have the standardized matrix.

    Args:
        Xs: The already-standardized design matrix, ``(n, 11)`` floats
            (standardized once by the caller with
            :func:`models._shared.apply_standardizer`).
        z: The binary labels, ``(n,)`` ints in ``{0, 1}`` (``1`` iff
            the row's ordinal is ``>= cutpoint``).
        cutpoint: The cutpoint this split corresponds to; must be one
            of ``{1, 2, 3}`` (recorded on the model for report
            self-documentation).
        l2_lambda: L2 regularization strength on ``beta`` only; must be
            non-negative finite (default ``1.0``, kept identical to both
            four-class arms for comparability).
        max_iter: Cap on gradient-descent iterations (default 2000).
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).
        standardizer_means: The per-feature training-column means of
            the *raw* training matrix (length 11), shared across all
            three cutpoints; if ``None``, derived from ``Xs``'s own
            column means (the documented fallback).
        standardizer_stds: The per-feature training-column stds of the
            *raw* training matrix (length 11), shared across all three
            cutpoints; if ``None``, derived from ``Xs``'s own column
            stds (the documented fallback).

    Returns:
        A frozen :class:`BinaryLogitModel` with the fitted
        ``coefficients``/``intercept``, the standardizer statistics
        (the passed shared values, or ``Xs``-derived), ``cutpoint``,
        and the fit diagnostics.

    Raises:
        ValueError: If ``Xs`` is not a 2-D matrix with exactly
            ``len(FEATURE_NAMES)`` columns, if ``z`` is not 1-D with
            matching row count, if ``z`` is empty or contains a value
            outside ``{0, 1}``, if ``cutpoint`` is not in ``{1, 2, 3}``,
            if a passed standardizer length does not match the column
            count, or if any hyperparameter is invalid (see the
            validate helpers).
    """
    Xs_arr = np.asarray(Xs, dtype=float)
    z_arr = np.asarray(z, dtype=int)
    if Xs_arr.ndim != 2:
        raise ValueError(
            f"Xs must be a 2-D design matrix, got {Xs_arr.ndim} dimension(s)"
        )
    if z_arr.ndim != 1:
        raise ValueError(
            f"z must be a 1-D label vector, got {z_arr.ndim} dimension(s)"
        )
    if Xs_arr.shape[0] != z_arr.shape[0]:
        raise ValueError(
            f"Xs has {Xs_arr.shape[0]} rows but z has {z_arr.shape[0]} "
            "entries; they must match"
        )
    if Xs_arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Xs must have exactly {len(FEATURE_NAMES)} feature columns "
            f"(one per {FEATURE_NAMES} entry), got {Xs_arr.shape[1]}"
        )
    if z_arr.size == 0:
        raise ValueError("cannot fit a binary logit on an empty label vector")
    if cutpoint not in (1, 2, 3):
        raise ValueError(
            f"cutpoint must be one of 1, 2, 3, got {cutpoint!r}"
        )

    l2 = _validate_l2_lambda(l2_lambda)
    max_it = _validate_positive_int(max_iter, "max_iter")
    g_tol = _validate_positive_float(grad_tol, "grad_tol")
    l_tol = _validate_positive_float(loss_tol, "loss_tol")

    beta, alpha, converged, n_iter, trace = _gradient_descent_binary(
        Xs_arr, z_arr, l2, max_it, g_tol, l_tol
    )
    if standardizer_means is None or standardizer_stds is None:
        # Documented fallback: derive from the standardized matrix's own
        # column statistics (approximately 0/1); the primary path passes
        # the raw training statistics.
        means, stds = fit_standardizer(Xs_arr)
    else:
        means = np.asarray(standardizer_means, dtype=float)
        stds = np.asarray(standardizer_stds, dtype=float)
        if len(means) != Xs_arr.shape[1] or len(stds) != Xs_arr.shape[1]:
            raise ValueError(
                f"standardizer means ({len(means)}) / stds ({len(stds)}) "
                f"must each have one entry per column ({Xs_arr.shape[1]})"
            )
    return BinaryLogitModel(
        coefficients=beta,
        intercept=alpha,
        standardizer_means=means,
        standardizer_stds=stds,
        cutpoint=cutpoint,
        converged=converged,
        n_iter=n_iter,
        final_loss=trace[-1],
        n_train=len(z_arr),
    )


def fit_cutpoint_binary_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_lambda: float = 1.0,
    max_iter: int = 2000,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-10,
) -> dict[int, BinaryLogitModel]:
    """Fit the three per-cutpoint binary models on a raw training matrix.

    Standardizes ``X_train`` *once* with
    :func:`models._shared.fit_standardizer` /
    :func:`models._shared.apply_standardizer` (the matrix is identical
    across cutpoints — only the binary label differs, so there is no
    reason to refit the standardizer three times), builds the three
    binary labels ``z_i = 1 {y_i >= j}`` for ``j = 1, 2, 3`` (exactly
    the three cumulative splits the ordinal model's three thresholds
    each separately govern), calls :func:`fit_binary_logit` three times
    on the shared matrix, and returns the models keyed by cutpoint.

    Args:
        X_train: The raw (unstandardized) training design matrix,
            ``(n, 11)`` floats in :data:`FEATURE_NAMES` order.
        y_train: The true outcome ordinals, ``(n,)`` ints in
            ``{0, 1, 2, 3}``.
        l2_lambda: L2 regularization strength on the coefficient vector
            of each binary fit (default ``1.0``).
        max_iter: Cap on gradient-descent iterations (default 2000).
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).

    Returns:
        A dict ``{1: model_1, 2: model_2, 3: model_3}`` mapping each
        cutpoint to its fitted :class:`BinaryLogitModel`.

    Raises:
        ValueError: If ``X_train`` rows and ``y_train`` entries differ,
            if ``y_train`` is empty or contains a value outside
            ``{0, 1, 2, 3}``, or if any hyperparameter is invalid
            (propagated from :func:`fit_binary_logit` /
            :func:`models._shared.fit_standardizer`).
    """
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    if X_arr.shape[0] != len(y_arr):
        raise ValueError(
            f"X_train has {X_arr.shape[0]} rows but y_train has "
            f"{len(y_arr)} entries; they must match"
        )
    if y_arr.size == 0:
        raise ValueError("cannot build cutpoint models from an empty label vector")
    if set(np.unique(y_arr).tolist()) - {0, 1, 2, 3}:
        raise ValueError(
            f"y_train must contain only outcome ordinals 0..3, got values "
            f"{sorted(set(np.unique(y_arr).tolist()))}"
        )
    means, stds = fit_standardizer(X_arr)
    Xs = apply_standardizer(X_arr, means, stds)
    models_by_cutpoint: dict[int, BinaryLogitModel] = {}
    for cutpoint in (1, 2, 3):
        z = (y_arr >= cutpoint).astype(int)
        models_by_cutpoint[cutpoint] = fit_binary_logit(
            Xs,
            z,
            cutpoint,
            l2_lambda=l2_lambda,
            max_iter=max_iter,
            grad_tol=grad_tol,
            loss_tol=loss_tol,
            standardizer_means=means,
            standardizer_stds=stds,
        )
    return models_by_cutpoint


def _sign(value: float) -> int:
    """Return the sign of ``value`` as ``-1``, ``0`` or ``1``.

    The sign function used by the sign-instability check: strictly
    positive -> ``1``, strictly negative -> ``-1``, exactly zero -> ``0``.

    Args:
        value: The numeric value to sign.

    Returns:
        ``1`` if ``value > 0``, ``-1`` if ``value < 0``, else ``0``.

    Raises:
        Nothing (total for any numeric input).
    """
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _feature_sign_instability(
    cutpoint_models: dict[int, BinaryLogitModel],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> tuple[int, list[dict]]:
    """Assess per-feature sign instability across the three cutpoints.

    For each feature ``k``, takes the three per-cutpoint coefficients
    ``b1_k, b2_k, b3_k`` (``BinaryLogitModel.coefficients[k]`` for
    cutpoints 1, 2, 3) and computes their signs via :func:`_sign`.
    Feature ``k`` is **sign-unstable** if the *nonzero* signs are not
    all equal (a feature where all three estimates are exactly zero is
    trivially stable, not unstable). Returns the count of unstable
    features and the per-feature detail dicts used by the report.

    Args:
        cutpoint_models: The ``{1: model_1, 2: model_2, 3: model_3}``
            dict from :func:`fit_cutpoint_binary_models`.
        feature_names: The feature vocabulary, one name per
            coefficient position; defaults to
            :data:`FEATURE_NAMES`.

    Returns:
        A ``(sign_instability_count, per_feature)`` tuple:
        ``sign_instability_count`` an int in ``0..len(feature_names)``;
        ``per_feature`` a list (in ``feature_names`` order) of dicts
        each with ``feature`` / ``cutpoint_1_beta`` /
        ``cutpoint_2_beta`` / ``cutpoint_3_beta`` / ``sign_unstable``
        (bool) / ``max_abs_spread`` (the largest pairwise absolute
        difference among the three cutpoint coefficients).

    Raises:
        KeyError: If ``cutpoint_models`` lacks any of the cutpoints
            1/2/3 (propagated from dict indexing).
        ValueError: If a model's coefficient vector has a different
            length than ``feature_names``.
    """
    per_feature: list[dict] = []
    unstable_count = 0
    if any(
        len(cutpoint_models[cutpoint].coefficients) != len(feature_names)
        for cutpoint in (1, 2, 3)
    ):
        raise ValueError(
            "cutpoint model coefficient vectors must have one entry per "
            "feature name"
        )
    for k, name in enumerate(feature_names):
        betas = [
            float(cutpoint_models[cutpoint].coefficients[k])
            for cutpoint in (1, 2, 3)
        ]
        nonzero_signs = {_sign(b) for b in betas} - {0}
        sign_unstable = len(nonzero_signs) > 1
        if sign_unstable:
            unstable_count += 1
        max_abs_spread = max(
            abs(betas[i] - betas[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        per_feature.append(
            {
                "feature": name,
                "cutpoint_1_beta": betas[0],
                "cutpoint_2_beta": betas[1],
                "cutpoint_3_beta": betas[2],
                "sign_unstable": sign_unstable,
                "max_abs_spread": max_abs_spread,
            }
        )
    return unstable_count, per_feature


def _aic_bic(
    log_likelihood: float,
    n_parameters: int,
    n_train: int,
) -> tuple[float, float]:
    """Compute AIC and BIC from a log-likelihood, parameter count and n.

    ``aic = -2 * ll + 2 * k``; ``bic = -2 * ll + k * log(n)``. Lower is
    better for both. Both are valid for comparing non-nested models
    (unlike a likelihood-ratio chi-square), which is why the M21
    diagnostic uses them to compare the ordinal arm against the
    multinomial arm on the identical training rows.

    Args:
        log_likelihood: The model's total training log-likelihood
            (``<= 0``).
        n_parameters: The model's free-parameter count (14 for the
            ordinal arm, 36 for the multinomial arm).
        n_train: The number of training rows.

    Returns:
        An ``(aic, bic)`` tuple of floats.

    Raises:
        ValueError: If ``n_train`` is not positive (``log(n)`` is
            undefined at ``n <= 0``).
    """
    if n_train <= 0:
        raise ValueError(
            f"n_train must be positive to compute BIC, got {n_train}"
        )
    aic = -2.0 * log_likelihood + 2.0 * n_parameters
    bic = -2.0 * log_likelihood + n_parameters * math.log(n_train)
    return aic, bic


def build_diagnostic_report(
    ordinal_model,
    multinomial_model,
    cutpoint_models: dict[int, BinaryLogitModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict:
    """Assemble the full JSON-serializable proportional-odds report.

    Computes the two four-class training log-likelihoods (via
    :func:`models.ordinal_logit.total_log_likelihood` /
    :func:`models.multinomial_logit.total_log_likelihood` on the raw
    ``X_train`` — each model applies its own stored standardizer
    internally), the AIC/BIC pairs (:data:`_K_ORDINAL` /
    :data:`_K_MULTINOMIAL` parameters), the sign-instability assessment
    (:func:`_feature_sign_instability`), and the verdict
    (:data:`_VERDICT_RULE`). Every value is a plain str/int/float/
    bool/list/dict so the whole dict is directly ``json.dumps``-
    serializable.

    Args:
        ordinal_model: The fitted
            :class:`models.ordinal_logit.OrdinalLogitModel`.
        multinomial_model: The fitted
            :class:`models.multinomial_logit.MultinomialLogitModel`.
        cutpoint_models: The ``{1: model_1, 2: model_2, 3: model_3}``
            dict from :func:`fit_cutpoint_binary_models`.
        X_train: The raw (unstandardized) training design matrix,
            ``(n, 11)`` floats in :data:`FEATURE_NAMES` order — the same
            matrix both models were fit on.
        y_train: The true outcome ordinals, ``(n,)`` ints in
            ``{0, 1, 2, 3}``.
        feature_names: The feature vocabulary; defaults to
            :data:`FEATURE_NAMES`.

    Returns:
        A dict with keys ``n_train``, ``ll_ordinal`` /
        ``aic_ordinal`` / ``bic_ordinal`` / ``k_ordinal``,
        ``ll_multinomial`` / ``aic_multinomial`` / ``bic_multinomial`` /
        ``k_multinomial``, ``bic_favors_multinomial`` (bool),
        ``sign_instability_count`` (int), ``per_feature`` (list of
        dicts as described in :func:`_feature_sign_instability`, each
        with an added ``ordinal_beta`` entry), ``proportional_odds_verdict``
        (``"violated"`` or ``"holds"``), and ``verdict_rule`` (the
        literal rule text).

    Raises:
        ValueError: If ``X_train`` rows and ``y_train`` entries differ,
            if a model's feature count does not match ``feature_names``,
            or if ``n_train`` is not positive (propagated from
            :func:`_aic_bic` / :func:`_feature_sign_instability` /
            the two ``total_log_likelihood`` calls).
        KeyError: If ``cutpoint_models`` lacks any cutpoint (propagated
            from :func:`_feature_sign_instability`).
    """
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    if X_arr.shape[0] != len(y_arr):
        raise ValueError(
            f"X_train has {X_arr.shape[0]} rows but y_train has "
            f"{len(y_arr)} entries; they must match"
        )
    if len(ordinal_model.coefficients) != len(feature_names) or (
        multinomial_model.coefficients.shape[1] != len(feature_names)
    ):
        raise ValueError(
            "ordinal/multinomial coefficient vectors must have one entry per "
            "feature name"
        )
    n_train = len(y_arr)

    ll_ordinal = ordinal_logit.total_log_likelihood(
        X_arr, y_arr, ordinal_model
    )
    aic_ordinal, bic_ordinal = _aic_bic(ll_ordinal, _K_ORDINAL, n_train)
    ll_multinomial = multinomial_logit.total_log_likelihood(
        X_arr, y_arr, multinomial_model
    )
    aic_multinomial, bic_multinomial = _aic_bic(
        ll_multinomial, _K_MULTINOMIAL, n_train
    )

    sign_instability_count, per_feature = _feature_sign_instability(
        cutpoint_models, feature_names
    )
    for k, entry in enumerate(per_feature):
        entry["ordinal_beta"] = float(ordinal_model.coefficients[k])

    bic_favors_multinomial = bic_multinomial < bic_ordinal
    proportional_odds_verdict = (
        "violated"
        if (sign_instability_count > 0 or bic_favors_multinomial)
        else "holds"
    )

    return {
        "n_train": n_train,
        "ll_ordinal": ll_ordinal,
        "aic_ordinal": aic_ordinal,
        "bic_ordinal": bic_ordinal,
        "k_ordinal": _K_ORDINAL,
        "ll_multinomial": ll_multinomial,
        "aic_multinomial": aic_multinomial,
        "bic_multinomial": bic_multinomial,
        "k_multinomial": _K_MULTINOMIAL,
        "bic_favors_multinomial": bic_favors_multinomial,
        "sign_instability_count": sign_instability_count,
        "per_feature": per_feature,
        "proportional_odds_verdict": proportional_odds_verdict,
        "verdict_rule": _VERDICT_RULE,
    }

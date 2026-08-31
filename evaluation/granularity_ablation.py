"""Granularity ablation for the M20 ordinal model (roadmap M22).

The M22 "viability gate": compares the plain binary logistic regression
(:mod:`models.binary_logit`, fit directly on the binary "does team1
(A) win the map" target) against the M20 ordinal logit's four-way
output **marginalized down to that same binary target**, both scored
with identical binary metrics on the identical held-out test split.
The report's verdict — ``granularity_verdict`` — answers the roadmap's
question: does the four-way granularity *cost* accuracy relative to a
plain binary model on the same features and split? This module only
*reports* the finding; it does not change the category set or the
ordinal structure itself (that is future-milestone scope if the
verdict says so).

Scope / conventions (recorded here, do not re-derive later):

- **Pure and dependency-light.** This module does no file I/O, has no
  CLI / ``argparse`` entry point, and never touches ``drivers/``. It
  takes already-built prediction/label arrays in and returns a dict
  out — matching ``evaluation.harness`` / ``evaluation.proportional_odds``'s
  own convention. **It deliberately does NOT import
  ``evaluation.harness``**: unlike the ``models/_shared.py`` carve-out,
  there is no shared-module exception at the ``evaluation/`` rung — the
  module-boundary test forbids ANY sibling ``evaluation/`` import
  unconditionally (confirmed by reading the test body: it asserts the
  absence of both import forms — a dotted sibling-import line and a
  bare package import — in the source of every ``EVALUATION_MODULES``
  entry). All table joining and model-loading happens in the driver
  (:mod:`drivers.ablate_granularity`), which passes this module plain
  ``(n, 2)`` prediction arrays plus the true binary label vector.
- **Metric choice (decided here, do not re-derive): binary log loss
  and binary accuracy, both computed via the existing generic
  ``utils.scoring`` batch functions applied directly to the 2-vectors**
  — :func:`utils.scoring.mean_log_loss` and
  :func:`utils.scoring.mean_marginal_binary_accuracy` with the default
  grouping (``group_a_indices=None``, whose "first ``k // 2``
  categories are side A" convention is exactly ``{index 0}`` for
  ``k = 2``, i.e. side A — no override needed). **RPS is deliberately
  not computed for this comparison**: RPS's entire value-add is being
  ordinal-aware across ``> 2`` ordered categories, and on a 2-category
  simplex it carries no additional information beyond what log loss and
  accuracy already report (for ``K = 2`` there is exactly one cut
  point, and the squared-cumulative-gap term reduces to the same
  information a Brier-style squared error on a 2-way split already
  contains) — computing and reporting it here would be a redundant
  third number, not a materially different one.
- **Marginalization rule (the roadmap's stated rule, verbatim).**
  Sum the two A-side categories (ordinals 0 + 1) for ``P(A wins)`` and
  the two B-side categories (ordinals 2 + 3) for ``P(B wins)`` — see
  :func:`marginalize_ordinal_probs`. No renormalization is needed or
  performed: the four input probabilities already sum to
  (approximately) 1 by construction (``ordinal_logit.predict_proba``'s
  own clip-and-return guarantee), so the two summed halves already sum
  to (approximately) 1.
- **Both gaps are signed so a positive value always means "the binary
  model did better on this metric"**: ``accuracy_gap > 0`` means the
  binary model is more accurate; ``log_loss_gap > 0`` means the binary
  model has lower (better) log loss (``log_loss_gap =
  ordinal_marginalized.mean_log_loss - binary_logit.mean_log_loss``).
- **Verdict threshold (decided here — the task's central open question
  — do not re-derive; embedded verbatim in every report as
  ``verdict_rule``, per the ``proportional_odds.py`` precedent of a
  self-documenting artifact):**
  ``granularity_verdict = "costs_accuracy" if (accuracy_gap >= 0.10 or
  log_loss_gap >= 0.05) else "viable"``. Justification: at
  ``n_eval = 35`` (the v1 test-split scale), an ``accuracy_gap`` of
  ``0.10`` is ~3.5 maps — clearly above the ~1-map (``0.0286``)
  run-to-run noise scale already established between the M18/M20/M21
  models on this exact split. A ``log_loss_gap`` of ``0.05`` is roughly
  6-8x the 0.006-0.008 gaps already observed between models that are
  considered legitimate (non-noise) improvements on this split. Both
  thresholds are therefore sized to require a gap clearly outside the
  already-characterized noise band before declaring the four-way spec
  "costs" accuracy — a deliberately conservative bar, consistent with
  ``proportional_odds.py``'s own "a documented heuristic, not a
  calibrated significance threshold" framing for exactly the same
  small-``n`` reason.
"""

from __future__ import annotations

import numpy as np

from utils import scoring

# The verdict thresholds, as module constants (recorded and justified in
# the module docstring): an accuracy gap of ~3.5 maps and a log-loss
# gap of ~6-8x the observed legitimate model-to-model gap at
# n_eval = 35, both clearly outside the characterized noise band.
_ACCURACY_GAP_THRESHOLD = 0.10
_LOG_LOSS_GAP_THRESHOLD = 0.05

# The exact verdict rule, embedded verbatim in the report JSON as
# ``verdict_rule`` so the artifact is self-documenting without a source
# read (mirroring ``proportional_odds.py``'s ``_VERDICT_RULE``).
_VERDICT_RULE = (
    "granularity_verdict = 'costs_accuracy' if (accuracy_gap >= 0.10 "
    "or log_loss_gap >= 0.05) else 'viable'"
)


def marginalize_ordinal_probs(prob_rows_4way: np.ndarray) -> np.ndarray:
    """Marginalize four-way ordinal predictions down to a binary 2-vector.

    Implements the roadmap's stated marginalization rule verbatim: sum
    the two A-side categories (ordinals 0 = A-regulation and 1 = A-OT)
    for ``P(A wins)`` and the two B-side categories (ordinals 2 = B-OT
    and 3 = B-regulation) for ``P(B wins)``. Literally
    ``[p_a_regulation + p_a_ot, p_b_ot + p_b_regulation]`` per row. No
    renormalization step is needed or performed: the four inputs already
    sum to (approximately) 1 by construction (the ordinal model's own
    clip-and-return guarantee), so the two summed halves already sum to
    (approximately) 1.

    Args:
        prob_rows_4way: An ``(n, 4)`` array of ordinal-model predictions
            in :data:`models._shared.OUTCOME_LABELS` order
            (``p_a_regulation, p_a_ot, p_b_ot, p_b_regulation``).

    Returns:
        An ``(n, 2)`` numpy array of ``float``, column 0 =
        ``P(A wins)`` (ordinals 0 + 1), column 1 = ``P(B wins)``
        (ordinals 2 + 3).

    Raises:
        ValueError: If ``prob_rows_4way`` is not a 2-D array with
            exactly 4 columns (a wrong column count would silently
            misalign the summed categories).
    """
    arr = np.asarray(prob_rows_4way, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            "marginalize_ordinal_probs requires an (n, 4) array of "
            f"four-way predictions, got shape {arr.shape}"
        )
    return np.asarray(
        [
            arr[:, 0] + arr[:, 1],
            arr[:, 2] + arr[:, 3],
        ],
        dtype=float,
    ).T


def build_ablation_report(
    binary_probs: np.ndarray,
    ordinal_marginal_probs: np.ndarray,
    true_binary_labels: np.ndarray,
    n_train_binary_model: int,
    n_train_ordinal_model: int,
) -> dict:
    """Assemble the full JSON-serializable granularity-ablation report.

    Computes, for each of the two ``(n, 2)`` prediction arrays,
    ``mean_log_loss`` and ``accuracy``
    (:func:`utils.scoring.mean_marginal_binary_accuracy`, default
    grouping — side A is index 0, exactly this problem's "A wins"
    convention) against ``true_binary_labels``, via the shared
    ``utils.scoring`` batch functions (all metric math stays in that
    one existing place, per the harness's own no-ad-hoc-mean
    precedent). Both gaps are signed so a positive value always means
    "the binary model did better on this metric" (see the module
    docstring), and the verdict follows :data:`_VERDICT_RULE`.

    Args:
        binary_probs: The binary model's predictions, an ``(n, 2)``
            array with column 0 = ``P(A wins)``, column 1 =
            ``P(B wins)`` (the output of
            :func:`models.binary_logit.predict_proba` per row, stacked).
        ordinal_marginal_probs: The ordinal model's four-way predictions
            marginalized down to the same ``(n, 2)`` shape via
            :func:`marginalize_ordinal_probs`.
        true_binary_labels: The true binary labels, an ``(n,)`` array
            of ints in ``{0, 1}``, expressed in **``utils.scoring``'s
            category-index convention: ``0`` = side A = "A wins", ``1``
            = side B = "B wins"** (for ``k = 2`` the default grouping's
            side A is exactly index 0, so ``0`` must mean A wins for
            ``mean_log_loss`` — which reads ``probs[true_index]`` — and
            for ``mean_marginal_binary_accuracy`` to measure the honest
            binary accuracy of the "A wins" target). Note this is the
            *complement* of the model-target convention ``y_binary = 1
            iff A wins`` used by :mod:`models.binary_logit` / the
            training driver: the caller (the ablation driver) derives
            these indices from the test ordinals as
            ``(y_ordinal >= 2).astype(int)``.
        n_train_binary_model: The number of training rows the binary
            model was fit on (read off the loaded model object by the
            caller, not re-derived).
        n_train_ordinal_model: The number of training rows the ordinal
            model was fit on.

    Returns:
        A dict with keys ``n_eval`` (int), ``binary_logit`` and
        ``ordinal_marginalized`` (each ``{"mean_log_loss": float,
        "accuracy": float, "n_train": int}``), ``accuracy_gap`` (float
        = binary accuracy minus ordinal accuracy, positive means the
        binary model was more accurate), ``log_loss_gap`` (float =
        ordinal mean_log_loss minus binary mean_log_loss, positive
        means the binary model had lower/better log loss),
        ``granularity_verdict`` (``"costs_accuracy"`` or ``"viable"``
        per :data:`_VERDICT_RULE`), and ``verdict_rule`` (the literal
        rule text). Every value is a plain str/int/float so the dict is
        directly ``json.dumps``-serializable.

    Raises:
        ValueError: If the row counts of the three arrays differ, if
            either prediction array is not ``(n, 2)``, if
            ``true_binary_labels`` is not 1-D, or if any row fails the
            ``utils.scoring`` validation (e.g. a probability vector not
            summing to 1, or a true label outside ``{0, 1}`` —
            propagated from the scoring functions).
    """
    binary = np.asarray(binary_probs, dtype=float)
    ordinal = np.asarray(ordinal_marginal_probs, dtype=float)
    truth = np.asarray(true_binary_labels, dtype=int)
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError(
            "binary_probs must be an (n, 2) array of binary predictions, "
            f"got shape {binary.shape}"
        )
    if ordinal.ndim != 2 or ordinal.shape[1] != 2:
        raise ValueError(
            "ordinal_marginal_probs must be an (n, 2) array of "
            f"marginalized predictions, got shape {ordinal.shape}"
        )
    if truth.ndim != 1:
        raise ValueError(
            "true_binary_labels must be a 1-D label vector, got "
            f"{truth.ndim} dimension(s)"
        )
    n = binary.shape[0]
    if ordinal.shape[0] != n or truth.shape[0] != n:
        raise ValueError(
            f"prediction/label row counts differ: binary {n}, "
            f"ordinal {ordinal.shape[0]}, truth {truth.shape[0]}; they "
            "must match"
        )

    binary_log_loss = scoring.mean_log_loss(binary, truth)
    binary_accuracy = scoring.mean_marginal_binary_accuracy(binary, truth)
    ordinal_log_loss = scoring.mean_log_loss(ordinal, truth)
    ordinal_accuracy = scoring.mean_marginal_binary_accuracy(ordinal, truth)

    accuracy_gap = binary_accuracy - ordinal_accuracy
    log_loss_gap = ordinal_log_loss - binary_log_loss

    granularity_verdict = (
        "costs_accuracy"
        if (
            accuracy_gap >= _ACCURACY_GAP_THRESHOLD
            or log_loss_gap >= _LOG_LOSS_GAP_THRESHOLD
        )
        else "viable"
    )

    return {
        "n_eval": int(n),
        "binary_logit": {
            "mean_log_loss": binary_log_loss,
            "accuracy": binary_accuracy,
            "n_train": int(n_train_binary_model),
        },
        "ordinal_marginalized": {
            "mean_log_loss": ordinal_log_loss,
            "accuracy": ordinal_accuracy,
            "n_train": int(n_train_ordinal_model),
        },
        "accuracy_gap": accuracy_gap,
        "log_loss_gap": log_loss_gap,
        "granularity_verdict": granularity_verdict,
        "verdict_rule": _VERDICT_RULE,
    }

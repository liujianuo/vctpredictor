"""Proper scoring rules for ordered and unordered categorical outcomes
(roadmap M11).

Four pure metrics, each operating on one prediction (a probability
vector over ``K`` ordered categories) plus a batch-mean wrapper that
averages it over many predictions:

- :func:`rps` — the Ranked Probability Score, the *ordinal-aware*
  metric: it compares cumulative probabilities, so a confident miss
  one category away from the truth is penalised less than an equally
  confident miss three categories away. This is the headline number
  every downstream model milestone (M19 onward) is ranked by.
- :func:`log_loss` — multi-class (categorical) log loss, a strictly
  proper scoring rule that is *unordered*: it only reads the
  probability placed on the true category, so it cannot tell an
  adjacent miss from a distant one.
- :func:`brier_score` — the multi-class Brier score, the unordered
  analogue of RPS: the same squared-error shape as RPS but computed on
  the raw probabilities/one-hot vector rather than on cumulative sums,
  so (like log loss) it is insensitive to category ordering.
- :func:`marginal_binary_accuracy` — collapses the ``K``-way
  prediction/label to a binary "which side won" question (side A vs
  side B) and reports whether the argmax-collapsed binary prediction
  matched the collapsed true label. Inherently a batch metric: the
  per-observation form is just a 0/1 correctness ``bool``, and the
  :func:`mean_marginal_binary_accuracy` wrapper reports the accuracy
  fraction.

Contract:

- **Pure functions.** None of these read or write files, touch the
  cache, or mutate shared state; each maps inputs to a number (or
  ``bool``) and raises ``ValueError`` on invalid input. They are
  deterministic and import-safe.
- **Generic over ``K``.** The library takes plain sequences of
  probabilities and integer category indices; it does not import
  ``drivers.labels`` or its ``OUTCOME_LABELS`` vocabulary. The
  four-way outcome problem (``OUTCOME_LABELS`` ordinal order 0 =
  A-regulation, 1 = A-OT, 2 = B-OT, 3 = B-regulation) is the ``K = 4``
  special case, and :func:`marginal_binary_accuracy`'s default
  grouping (first ``K // 2`` categories are side A) matches it exactly
  for ``K = 4`` (ordinals 0, 1 -> A; 2, 3 -> B).
- **Lives in ``utils/``, not ``drivers/``.** ``drivers/`` is reserved
  for CLI pipeline stages that read/write ``data/<version>/*.parquet``;
  this module has no CLI, no ``argparse`` entry point, and no file
  I/O, so it belongs alongside ``utils/config.py`` and
  ``utils/table_io.py`` as dependency-light shared code consumed by
  later model/evaluation milestones.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence

# Tolerance for the "probabilities sum to 1" simplex check. Floats that
# legitimately sum to 1 (e.g. 0.1 + 0.2 + 0.3 + 0.4) are off by at most
# ~1e-16, so this is generous enough to avoid false positives while
# still rejecting a genuine violation such as [0.5, 0.5, 0.5] (sum 1.5).
_PROB_SUM_TOL = 1e-9


def _validate_probs(probs: Iterable[float]) -> list[float]:
    """Validate a probability vector against the unit simplex.

    Materializes ``probs`` into a ``list`` of ``float`` and checks the
    three conditions a categorical distribution must satisfy: it has at
    least two categories, every entry is a finite non-negative real
    number, and the entries sum to 1 within :data:`_PROB_SUM_TOL`.
    Non-numeric entries are coerced through ``float``; a value that
    cannot be coerced becomes a ``ValueError`` so the failure mode is
    uniform with the numeric-validation failures.

    Args:
        probs: An iterable of ``K`` non-negative probabilities forming
            a valid categorical distribution. Entries may be any real
            numbers (``int``/``float``/numpy scalars); they are coerced
            to ``float``.

    Returns:
        A new ``list`` of ``float`` holding the probabilities in the
        original order — the materialized copy the metric functions
        index into.

    Raises:
        ValueError: If ``probs`` has fewer than two entries, contains a
            non-numeric (uncoercible) entry, contains a non-finite
            (NaN/inf) entry, contains a negative entry, or does not sum
            to 1 within ``_PROB_SUM_TOL``.
    """
    raw = list(probs)
    if len(raw) < 2:
        raise ValueError(
            f"probs must contain at least two categories, got {len(raw)}"
        )
    values: list[float] = []
    for i, entry in enumerate(raw):
        try:
            value = float(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"probs[{i}] must be a real number, got {entry!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"probs[{i}] must be finite, got {value!r}")
        if value < 0.0:
            raise ValueError(f"probs[{i}] must be non-negative, got {value!r}")
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) > _PROB_SUM_TOL:
        raise ValueError(
            f"probs must sum to 1.0, got {total!r} "
            f"(tolerance {_PROB_SUM_TOL:g})"
        )
    return values


def _validate_true_index(true_index: int, k: int) -> int:
    """Validate a true-category index against a category count.

    Coerces ``true_index`` through ``operator.index`` (so plain ints and
    numpy integer scalars are both accepted, and anything else is
    rejected uniformly) and checks it falls inside ``[0, k)``.

    Args:
        true_index: The observed category index. Must be an integer (or
            integer-like, e.g. a numpy integer scalar).
        k: The number of categories.

    Returns:
        The validated index as a plain ``int``.

    Raises:
        ValueError: If ``true_index`` is not integer-like (a float, a
            string, etc.) or lies outside ``[0, k)``.
    """
    try:
        idx = operator.index(true_index)
    except TypeError as exc:
        raise ValueError(
            f"true_index must be an integer, got {true_index!r}"
        ) from exc
    if idx < 0 or idx >= k:
        raise ValueError(f"true_index must be in [0, {k}), got {idx}")
    return idx


def _resolve_group_a(group_a_indices: Sequence[int] | None, k: int) -> frozenset[int]:
    """Resolve which category indices belong to binary side A.

    Turns the caller's grouping (or the default) into a validated set of
    side-A category indices. With ``group_a_indices=None`` the default
    convention is "the first ``k // 2`` categories are side A and the
    rest are side B", which for the four-way outcome vocabulary
    (ordinal order 0 = A-regulation, 1 = A-OT, 2 = B-OT, 3 =
    B-regulation) maps ordinals 0 and 1 to A and 2 and 3 to B. For any
    non-contiguous or non-half-split grouping the caller passes the
    side-A indices explicitly.

    Args:
        group_a_indices: An iterable of category indices forming side A,
            or ``None`` to use the default first-half convention. Must
            be reusable (a list/tuple/set/frozenset), not a one-shot
            generator, because it is iterated more than once.
        k: The number of categories.

    Returns:
        A ``frozenset`` of the side-A indices — a non-empty proper
        subset of ``range(k)``.

    Raises:
        ValueError: If ``group_a_indices`` is empty (side A needs at
            least one category), equals all of ``range(k)`` (side B
            needs at least one category), contains a non-integer, or
            contains an out-of-range index.
    """
    if group_a_indices is None:
        return frozenset(range(k // 2))
    coerced: set[int] = set()
    for raw in group_a_indices:
        try:
            coerced.add(operator.index(raw))
        except TypeError as exc:
            raise ValueError(
                f"group_a_indices contains a non-integer: {raw!r}"
            ) from exc
    if not coerced:
        raise ValueError(
            "group_a_indices must be non-empty "
            "(side A needs at least one category)"
        )
    if len(coerced) >= k:
        raise ValueError(
            "group_a_indices must be a proper subset of the categories "
            "(side B needs at least one category)"
        )
    for idx in coerced:
        if idx < 0 or idx >= k:
            raise ValueError(
                f"group_a_indices contains an out-of-range index {idx}"
            )
    return frozenset(coerced)


def rps(probs: Iterable[float], true_index: int) -> float:
    """Compute the Ranked Probability Score for one prediction.

    RPS is the ordinal-aware squared error: it compares the predicted
    cumulative distribution ``P(category < k)`` against the true
    category's step-function cumulative for each of the ``K - 1``
    cut points ``k = 1 .. K-1``, then sums the squared gaps. Because it
    works on cumulative probabilities, a confident miss located one
    category away from the truth is penalised less than an equally
    confident miss located farther away — the property plain accuracy,
    log loss, and Brier score all lack, and the reason RPS is the
    headline metric for the ordered four-way outcome.

    Args:
        probs: An iterable of ``K`` non-negative probabilities summing
            to 1, ordered by category (index 0 is the first/strongest-A
            category, index ``K-1`` the last/strongest-B category). The
            order is what makes RPS ordinal-aware.
        true_index: The index of the observed (true) category, an
            integer in ``[0, K)``.

    Returns:
        The Ranked Probability Score as a ``float``:
        ``sum_{k=1}^{K-1} (CDF_pred(k) - CDF_true(k)) ** 2`` where
        ``CDF_pred(k)`` is ``sum(probs[:k])`` and ``CDF_true(k)`` is 0
        for ``k <= true_index`` and 1 for ``k > true_index``. Lower is
        better: 0 for a perfect prediction, at most ``K - 1`` for
        putting all mass on the furthest category from the truth.

    Raises:
        ValueError: If ``probs`` fails the simplex validation (see
            :func:`_validate_probs`) or ``true_index`` is out of range
            (see :func:`_validate_true_index`).
    """
    values = _validate_probs(probs)
    k = len(values)
    idx = _validate_true_index(true_index, k)
    cumulative = 0.0
    score = 0.0
    for cut in range(1, k):
        cumulative += values[cut - 1]
        true_cdf = 1.0 if idx < cut else 0.0
        gap = cumulative - true_cdf
        score += gap * gap
    return score


def log_loss(probs: Iterable[float], true_index: int) -> float:
    """Compute the multi-class (categorical) log loss for one prediction.

    Log loss is the strictly proper, *unordered* scoring rule
    ``-ln(probs[true_index])``: it reads only the probability the
    prediction placed on the observed category, so it rewards confidence
    when right and punishes it heavily when wrong, but — unlike RPS —
    cannot distinguish a miss one category away from a miss three
    categories away.

    Args:
        probs: An iterable of ``K`` non-negative probabilities summing
            to 1. Only ``probs[true_index]`` is read.
        true_index: The index of the observed (true) category, an
            integer in ``[0, K)``.

    Returns:
        ``-math.log(probs[true_index])`` as a ``float``. 0 for a
        perfectly confident correct prediction, ``math.log(K)`` for a
        uniform prediction, and unboundedly large (finite, but big) as
        the probability on the true category approaches zero from above.

    Raises:
        ValueError: If ``probs`` fails the simplex validation (see
            :func:`_validate_probs`), ``true_index`` is out of range
            (see :func:`_validate_true_index`), or ``probs[true_index]``
            is exactly 0. The zero-probability case is a hard error
            rather than a clipped value, because the true loss there is
            ``+inf`` and silently substituting an epsilon would quietly
            understate how wrong the prediction was.
    """
    values = _validate_probs(probs)
    idx = _validate_true_index(true_index, len(values))
    p_true = values[idx]
    if p_true == 0.0:
        raise ValueError(
            f"log_loss is undefined when probs[{idx}] == 0 "
            "(the true category was assigned zero probability); "
            "give the true category a positive probability"
        )
    return -math.log(p_true)


def brier_score(probs: Iterable[float], true_index: int) -> float:
    """Compute the multi-class Brier score for one prediction.

    The multi-class Brier score is the unordered analogue of RPS: the
    same sum-of-squared-errors shape, but computed on the raw
    probability vector against the one-hot true vector rather than on
    cumulative probabilities, so — like log loss — it is insensitive to
    category ordering and can therefore not tell adjacent misses from
    distant ones.

    Args:
        probs: An iterable of ``K`` non-negative probabilities summing
            to 1.
        true_index: The index of the observed (true) category, an
            integer in ``[0, K)``.

    Returns:
        ``sum_k (probs[k] - one_hot_true[k]) ** 2`` as a ``float``.
        0 for a perfect prediction, ``2`` for a perfectly confident
        wrong prediction, and ``(K - 1) / K`` for a uniform prediction
        over ``K`` categories.

    Raises:
        ValueError: If ``probs`` fails the simplex validation (see
            :func:`_validate_probs`) or ``true_index`` is out of range
            (see :func:`_validate_true_index`).
    """
    values = _validate_probs(probs)
    idx = _validate_true_index(true_index, len(values))
    return sum(
        (p - (1.0 if i == idx else 0.0)) ** 2
        for i, p in enumerate(values)
    )


def marginal_binary_accuracy(
    probs: Iterable[float],
    true_index: int,
    group_a_indices: Sequence[int] | None = None,
) -> bool:
    """Report whether the collapsed binary side prediction was correct.

    Collapses a ``K``-way prediction and label to a binary "side A wins
    vs side B wins" question: the predicted A-side probability is the
    sum of the probabilities over the side-A category indices, the
    predicted side is whichever of A/B has the larger collapsed
    probability (ties go to A), and the true side is A exactly when
    ``true_index`` is one of the side-A indices. Returns whether the two
    agree.

    This is inherently a batch metric: the per-observation return value
    is just a 0/1 correctness ``bool`` (distinct from the other three
    metrics' per-observation float scores), and
    :func:`mean_marginal_binary_accuracy` aggregates it into an accuracy
    fraction.

    Args:
        probs: An iterable of ``K`` non-negative probabilities summing
            to 1.
        true_index: The index of the observed (true) category, an
            integer in ``[0, K)``.
        group_a_indices: An iterable of category indices forming side A,
            or ``None`` to use the default first-half convention (side A
            = indices ``0 .. K // 2 - 1``, side B = the rest). For the
            four-way outcome vocabulary the default maps ordinals 0, 1
            (A-regulation, A-OT) to A and 2, 3 (B-OT, B-regulation) to
            B. Pass the side-A indices explicitly for any other
            grouping; for odd ``K`` the default places the middle
            category on side B. Must be reusable (list/tuple/set), not a
            one-shot generator.

    Returns:
        ``True`` if the argmax collapsed side matches the collapsed true
        side, ``False`` otherwise. A tie in collapsed probability is
        resolved toward side A.

    Raises:
        ValueError: If ``probs`` fails the simplex validation (see
            :func:`_validate_probs`), ``true_index`` is out of range
            (see :func:`_validate_true_index`), or ``group_a_indices``
            is invalid (see :func:`_resolve_group_a`).
    """
    values = _validate_probs(probs)
    k = len(values)
    idx = _validate_true_index(true_index, k)
    a_indices = _resolve_group_a(group_a_indices, k)
    p_a = math.fsum(values[i] for i in a_indices)
    p_b = 1.0 - p_a
    predict_a = p_a >= p_b
    return predict_a == (idx in a_indices)


def _validate_batch(
    prob_rows: Iterable[Iterable[float]],
    true_indices: Iterable[int],
) -> tuple[list[list[float]], list[int]]:
    """Materialize and length-check a batch of predictions.

    Converts the two iterables to lists and checks they describe the
    same number of observations. It does not validate individual rows —
    each row is validated later by the per-observation metric function,
    whose ``ValueError`` propagates unchanged.

    Args:
        prob_rows: An iterable of per-observation probability vectors
            (each itself an iterable of floats).
        true_indices: An iterable of per-observation true category
            indices.

    Returns:
        A ``(rows, indices)`` tuple: the materialized list of prediction
        vectors and the materialized list of true indices, in matching
        order.

    Raises:
        ValueError: If the batch is empty (a mean over zero observations
            is undefined) or the two iterables have different lengths
            (a length mismatch would silently pair the wrong labels).
    """
    rows = [list(row) for row in prob_rows]
    indices = list(true_indices)
    if not rows:
        raise ValueError("cannot compute a batch mean over zero predictions")
    if len(rows) != len(indices):
        raise ValueError(
            f"prob_rows and true_indices lengths differ: "
            f"{len(rows)} != {len(indices)}"
        )
    return rows, indices


def mean_rps(
    prob_rows: Iterable[Iterable[float]],
    true_indices: Iterable[int],
) -> float:
    """Average the Ranked Probability Score over a batch of predictions.

    Loops the per-observation :func:`rps` over each (prediction, true
    index) pair and returns their arithmetic mean — the single number
    evaluation code reports over a test/fold split.

    Args:
        prob_rows: An iterable of per-observation probability vectors
            (each an iterable of ``K`` floats summing to 1).
        true_indices: An iterable of per-observation true category
            indices, the same length as ``prob_rows``.

    Returns:
        The mean RPS as a ``float``.

    Raises:
        ValueError: If the batch is empty or the two iterables have
            different lengths (see :func:`_validate_batch`), or if any
            individual row fails :func:`rps`'s validation.
    """
    rows, indices = _validate_batch(prob_rows, true_indices)
    total = math.fsum(rps(row, idx) for row, idx in zip(rows, indices))
    return total / len(rows)


def mean_log_loss(
    prob_rows: Iterable[Iterable[float]],
    true_indices: Iterable[int],
) -> float:
    """Average the multi-class log loss over a batch of predictions.

    Loops the per-observation :func:`log_loss` over each (prediction,
    true index) pair and returns their arithmetic mean.

    Args:
        prob_rows: An iterable of per-observation probability vectors
            (each an iterable of ``K`` floats summing to 1).
        true_indices: An iterable of per-observation true category
            indices, the same length as ``prob_rows``.

    Returns:
        The mean log loss as a ``float``.

    Raises:
        ValueError: If the batch is empty or the two iterables have
            different lengths (see :func:`_validate_batch`), or if any
            individual row fails :func:`log_loss`'s validation —
            including a zero probability on the row's true category.
    """
    rows, indices = _validate_batch(prob_rows, true_indices)
    total = math.fsum(log_loss(row, idx) for row, idx in zip(rows, indices))
    return total / len(rows)


def mean_brier_score(
    prob_rows: Iterable[Iterable[float]],
    true_indices: Iterable[int],
) -> float:
    """Average the multi-class Brier score over a batch of predictions.

    Loops the per-observation :func:`brier_score` over each (prediction,
    true index) pair and returns their arithmetic mean.

    Args:
        prob_rows: An iterable of per-observation probability vectors
            (each an iterable of ``K`` floats summing to 1).
        true_indices: An iterable of per-observation true category
            indices, the same length as ``prob_rows``.

    Returns:
        The mean Brier score as a ``float``.

    Raises:
        ValueError: If the batch is empty or the two iterables have
            different lengths (see :func:`_validate_batch`), or if any
            individual row fails :func:`brier_score`'s validation.
    """
    rows, indices = _validate_batch(prob_rows, true_indices)
    total = math.fsum(
        brier_score(row, idx) for row, idx in zip(rows, indices)
    )
    return total / len(rows)


def mean_marginal_binary_accuracy(
    prob_rows: Iterable[Iterable[float]],
    true_indices: Iterable[int],
    group_a_indices: Sequence[int] | None = None,
) -> float:
    """Report the marginal binary accuracy fraction over a batch.

    Loops the per-observation :func:`marginal_binary_accuracy` over each
    (prediction, true index) pair and returns the fraction of collapsed
    binary predictions that were correct — the accuracy the marginal
    metric actually reports, since a single observation is just a 0/1
    ``bool``.

    Args:
        prob_rows: An iterable of per-observation probability vectors
            (each an iterable of ``K`` floats summing to 1).
        true_indices: An iterable of per-observation true category
            indices, the same length as ``prob_rows``.
        group_a_indices: An iterable of category indices forming side A,
            or ``None`` for the default first-half convention (see
            :func:`marginal_binary_accuracy`). Applied identically to
            every row, so it must be reusable, not a one-shot generator.

    Returns:
        The accuracy fraction in ``[0, 1]`` as a ``float``.

    Raises:
        ValueError: If the batch is empty or the two iterables have
            different lengths (see :func:`_validate_batch`), or if any
            individual row (or the grouping) fails
            :func:`marginal_binary_accuracy`'s validation.
    """
    rows, indices = _validate_batch(prob_rows, true_indices)
    correct = sum(
        marginal_binary_accuracy(row, idx, group_a_indices)
        for row, idx in zip(rows, indices)
    )
    return correct / len(rows)

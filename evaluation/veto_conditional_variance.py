"""Pure veto-conditional spread helpers (roadmap M37).

The model-free statistics half of M37 (veto-conditional variance,
structural). The M37 *orchestration* — loading the fixed fitted
Stage-1/Stage-2 artifacts, building the held-out series table, and
calling the sibling M31 entry point
``evaluation.veto_marginalized_series.predict_series_outcome_via_veto_marginalization``
— cannot live in ``evaluation/``: importing that sibling module (or
``evaluation.series_evaluation``) would be a lateral reach forbidden by
the module-boundary DAG encoded in ``tests/test_module_boundaries.py``.
That orchestration lives in ``drivers/evaluate_veto_conditional_variance.py``;
this module holds only the generic, model-free helpers every caller
needs — today that driver, and later M39's ``predict()`` public API
against the per-sample detail M31 already returns:

- :func:`unweighted_scoreline_spread` — one independent per-category
  ``[low_pct, high_pct]`` percentile band per column of a
  ``(n_samples, n_categories)`` matrix of per-sample scoreline
  probability vectors.
- :func:`band_widths` — ``hi - lo`` per category from the bands
  :func:`unweighted_scoreline_spread` returns.
- :func:`mean_band_width` — the mean of :func:`band_widths` across
  categories, the single per-series scalar headline number.
- :func:`weighted_mean_and_variance` — per-category weighted first and
  second moments using the normalized ``weight`` values M31 already
  computes per ancestral sample.

Phase 6 framing (recorded here, do not re-derive in later milestones;
the same text is restated in the driver's docstring): M36 (epistemic —
parameter uncertainty from a finite training sample), M37 (structural —
spread across sampled veto sequences, resolves the moment the veto
happens) and M38 (calibration/reliability) are **three distinct
components, reported separately, never collapsed into one number**.
This module is the statistics engine of the structural component ONLY.

Scope boundary (the M37 plan's assumption 2): M37 holds the Stage-2
(and Stage-1) fitted models **fixed** — the single nominal, already-
trained artifacts, no refitting, no bootstrap resampling of the
training set — and lets the *veto rng* vary naturally, because the veto
sequence itself is exactly the thing whose resulting spread this task
measures. Do not bootstrap any model here (that would reintroduce M36's
epistemic component into a number meant to be purely structural) and do
not touch reliability/calibration (M38). The driver carries out that
fixed-model discipline; this module is deliberately agnostic about
*what* the row spread represents.

**Explicit contrast with M36's fixed-veto-rng mechanism (the M37
plan's assumption 9, restated for REVIEW):** M36 resets a fresh
``numpy.random.default_rng(veto_seed)`` identically for every bootstrap
replicate specifically to *suppress* structural veto variance so its
reported spread is attributable to Stage-2 parameter uncertainty alone.
M37 wants the **opposite** — natural, un-reset variation across the
``n_samples`` draws within one series call, and ordinary sequential
advancement of one ``numpy.random.default_rng(seed)`` across the
different held-out series — so a fixed ``--seed`` reproduces the whole
run byte-identically while the sampled veto sequences vary freely.
This module neither resets nor advances any rng (it is I/O-free and
RNG-free); the driver is responsible for the one-rng sequential
consumption.

**Primary spread metric: unweighted percentile bands.** Each ancestral
draw is already sampled proportional to its own ``sequence_probability``
by construction (M29's ancestral sampling), so treating the
``n_samples`` scoreline vectors as an *unweighted* empirical sample and
taking ``[5th, 95th]``-style percentile bands per category via
:func:`unweighted_scoreline_spread` is the standard Monte-Carlo way to
summarize their spread — the weighting has already happened at draw
time (assumption 3). :func:`weighted_mean_and_variance` is a second,
explicitly-flagged metric using M31's own normalized ``weight`` field,
so REVIEW can see both conventions side by side rather than one being
silently chosen (assumption 4); a full weighted-quantile function is
out of scope for an S-sized task.

**Interval definition: independent per-category percentile bands, not
a joint simplex region.** For each of the K scoreline categories, the
band is the ``[low_percentile, high_percentile]`` of that category's
probability across the ``n_samples`` ancestral draws, computed
independently per category. The K bands do not jointly form a
calibrated region on the simplex (they are marginal bands, and a random
draw respecting every marginal band individually need not itself sum to
1). Do not misread the per-category bands as a joint credible region;
the driver writes this same caveat into the artifact.

**Placement / duplication note.** This module sits in ``evaluation/``
and may depend downward on ``models.*`` / ``features.*`` / ``utils.*``
only — never on ``drivers.*`` and never on a sibling ``evaluation/``
module. It imports ``numpy`` only (precedented by every other module in
this repo) and is 100% I/O-free: it neither reads nor writes files.
:func:`unweighted_scoreline_spread`'s core per-column
``numpy.percentile`` computation is a **second, independent,
behaviour-identical duplicate** of
``evaluation.bootstrap_intervals.replicate_matrix_intervals``'s core
band logic — importing the sibling helper would be exactly the
forbidden lateral reach, and this repo's established convention (three
independent ``_parse_best_of`` copies already exist) is to duplicate
rather than reach sideways. Flagged here (not silently fixed) for a
future shared-utility promotion, mirroring how
``evaluation.bootstrap_intervals.py`` itself flags its own duplicated
parser copies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_ci_level(ci_level: float) -> float:
    """Validate a confidence-interval level and return it as a float.

    The shared guard for :func:`unweighted_scoreline_spread`: a
    ``ci_level`` must be a single real number strictly between 0 and 1
    (``0.90`` means the 5th/95th-percentile band). Values at the
    boundary (``0.0``, ``1.0``) are rejected: a degenerate band (a
    single point, or the full range of the sample) is not a meaningful
    uncertainty statement.

    Args:
        ci_level: The candidate interval level.

    Returns:
        The validated level as a ``float``.

    Raises:
        ValueError: If ``ci_level`` is not a finite real number in
            ``(0, 1)`` (including ``0.0`` and ``1.0`` themselves, and
            non-numeric inputs that cannot be coerced to ``float``).
    """
    try:
        level = float(ci_level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ci_level must be a real number strictly between 0 and 1, "
            f"got {ci_level!r}"
        ) from exc
    if not (0.0 < level < 1.0):
        raise ValueError(
            f"ci_level must be strictly between 0 and 1, got {level!r}"
        )
    return level


def unweighted_scoreline_spread(
    sample_rows: Sequence[Sequence[float]],
    ci_level: float = 0.90,
) -> tuple[tuple[float, float], ...]:
    """Return one per-category percentile band per column of a sample matrix.

    Applies a per-column ``numpy.percentile`` band independently per
    outcome category to a ``(n_samples, n_categories)`` matrix of
    per-sample scoreline probability vectors: column ``j`` holds
    category ``j``'s probability under each of the ``n_samples``
    ancestral veto draws, and the returned tuple's ``j``-th entry is
    that column's ``[lo, hi]`` band at ``ci_level`` (the ``((1 -
    ci_level)/2)``-th and ``((1 + ci_level)/2)``-th percentiles via
    ``numpy.percentile``'s default linear interpolation, so even and
    odd sample counts produce the standard percentile semantics). This
    is the **second, independent, behaviour-identical duplicate** of
    ``evaluation.bootstrap_intervals.replicate_matrix_intervals``'s
    core per-column band computation (see the module docstring's
    duplication note) — the two modules must stay in sync by convention
    until a future milestone promotes the logic to a shared utility.
    The helper is generic over ``n_categories`` (any ``K >= 1``), so it
    works identically for any scoreline length without any
    ``OUTCOME_LABELS``-specific coupling. The bands are independent per
    category (marginal, not a joint simplex region — see the module
    docstring).

    Args:
        sample_rows: A ``(n_samples, n_categories)`` matrix of floats,
            either a 2-D numpy array or a sequence of equal-length row
            sequences (one sampled veto sequence's scoreline
            probability vector per row, in
            ``utils.series_paths.series_outcome_order`` order). Must
            have at least one sample row, and every row must have the
            same positive number of categories.
        ci_level: The confidence level in ``(0, 1)`` (default ``0.90``
            — the 5th/95th-percentile band, mirroring M36's convention
            and reasoning).

    Returns:
        A tuple of ``(lo, hi)`` float pairs, one per category in column
            order: entry ``j`` is the ``[lo, hi]`` percentile band of
            category ``j`` across the sample rows, with ``lo <= hi`` in
            every entry (equality when the category is constant across
            draws — the "resolves the moment the veto happens" boundary
            case where there is effectively no veto-sequence ambiguity).

    Raises:
        ValueError: If ``sample_rows`` has no rows (a percentile band
            over zero draws is undefined); if a row is not 1-D or has a
            different length than the first row (ragged rows); if every
            row has zero categories (no columns to band); or if
            ``ci_level`` is not in ``(0, 1)`` (from
            :func:`_validate_ci_level`).
    """
    level = _validate_ci_level(ci_level)
    rows = list(sample_rows)
    if len(rows) == 0:
        raise ValueError(
            "unweighted_scoreline_spread expects at least one sample "
            "row; a percentile band over zero veto draws is undefined"
        )
    row_arrays: list[np.ndarray] = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"each sample row must be a 1-D scoreline probability "
                f"vector, got {arr.ndim} dimension(s)"
            )
        row_arrays.append(arr)
    n_categories = len(row_arrays[0])
    if n_categories == 0:
        raise ValueError(
            "each sample row must carry at least one category "
            "probability; a zero-category row has no columns to band"
        )
    for index, arr in enumerate(row_arrays):
        if len(arr) != n_categories:
            raise ValueError(
                f"ragged sample rows: row {index} has {len(arr)} "
                f"categories but the first row has {n_categories}; every "
                "sample must carry the same category vector length"
            )
    matrix = np.stack(row_arrays)
    lo_percent = ((1.0 - level) / 2.0) * 100.0
    hi_percent = ((1.0 + level) / 2.0) * 100.0
    lo = np.percentile(matrix, lo_percent, axis=0)
    hi = np.percentile(matrix, hi_percent, axis=0)
    return tuple(
        (float(lo[j]), float(hi[j])) for j in range(n_categories)
    )


def band_widths(
    bands: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    """Return the per-category ``hi - lo`` widths of a bands tuple.

    Applies ``hi - lo`` to each ``(lo, hi)`` entry of the tuple
    :func:`unweighted_scoreline_spread` returns, giving the per-category
    spread width in the same order — the raw material of
    :func:`mean_band_width` and of any per-category width reporting.

    Args:
        bands: A sequence of ``(lo, hi)`` float pairs (e.g. the return
            value of :func:`unweighted_scoreline_spread`), one per
            outcome category.

    Returns:
        A tuple of ``hi - lo`` floats, one per category in ``bands``
            order. Widths are non-negative when every band is ordered
            (``lo <= hi``); this helper does not itself re-validate the
            ordering.

    Raises:
        ValueError: If ``bands`` is empty (a zero-category series has
            no widths to report); if any entry is not a ``(lo, hi)``
            pair of length 2 or holds non-numeric values (from the
            unpacking / arithmetic).
    """
    widths: list[float] = []
    for band in bands:
        pair = tuple(band)
        if len(pair) != 2:
            raise ValueError(
                f"each band must be a (lo, hi) pair, got {band!r}"
            )
        widths.append(float(pair[1]) - float(pair[0]))
    if not widths:
        raise ValueError(
            "band_widths expects at least one band; a zero-category "
            "series has no widths to report"
        )
    return tuple(widths)


def mean_band_width(bands: Sequence[tuple[float, float]]) -> float:
    """Return the mean per-category band width of a bands tuple.

    Computes :func:`band_widths` and averages across categories — the
    single per-series scalar headline "how much does the veto sequence
    move the series outcome" number: a value of exactly 0.0 means every
    sampled veto sequence produced the identical scoreline distribution
    (the "resolves the moment the veto happens" boundary case where
    there is effectively no veto-sequence ambiguity).

    Args:
        bands: A sequence of ``(lo, hi)`` float pairs (e.g. the return
            value of :func:`unweighted_scoreline_spread`), one per
            outcome category.

    Returns:
        The arithmetic mean of the per-category ``hi - lo`` widths as a
            ``float``.

    Raises:
        ValueError: If ``bands`` is empty (from
            :func:`band_widths`); if any entry is not a length-2
            ``(lo, hi)`` pair or holds non-numeric values (from
            :func:`band_widths`).
    """
    widths = band_widths(bands)
    return float(sum(widths) / len(widths))


def weighted_mean_and_variance(
    sample_rows: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return per-category weighted means and variances over sample rows.

    Computes, independently per category ``j`` over a
    ``(n_samples, n_categories)`` matrix of per-sample scoreline
    probability vectors, the weighted first and second moments about the
    weighted mean using the caller-supplied ``weights`` as probability
    mass:

    - ``mean_j = sum_i w_i x_ij / W``
    - ``variance_j = sum_i w_i (x_ij - mean_j)^2 / W``

    where ``W = sum_i w_i`` (the total weight, so the definition stays
    correct for non-normalized weight vectors; M31's own ``weight``
    field is already normalized to sum to 1, making ``W == 1`` in the
    intended call pattern). This is the explicitly-flagged **secondary**
    metric of M37 (plan assumption 4): the same ``n_samples`` vectors
    M31 samples *proportionally* to ``sequence_probability``, summarized
    with their own normalized weights, shown side by side with the
    primary unweighted percentile bands so both conventions are visible
    to REVIEW rather than one being silently chosen. The variance is
    the weighted *population* variance about the weighted mean (a
    ``sum w_i`` denominator, not ``sum w_i - 1`` — appropriate for an
    exact probability-mass interpretation of ``weights``); a full
    weighted-quantile implementation is out of scope for an S-sized
    task.

    Args:
        sample_rows: A ``(n_samples, n_categories)`` matrix of floats
            (one sampled veto sequence's scoreline probability vector
            per row). Must have at least one row; every row must have
            the same positive number of categories.
        weights: A sequence of ``n_samples`` floats, the probability
            mass of each row (M31's per-sample normalized ``weight``
            values in sample order). Every weight must be non-negative
            and the total must be strictly positive (an all-zero weight
            set leaves both moments undefined).

    Returns:
        A ``(means, variances)`` tuple of two equal-length tuples, one
            per category in column order: ``means[j]`` is the weighted
            mean of category ``j``, ``variances[j]`` its weighted
            variance about that mean. Both are ``float``.

    Raises:
        ValueError: If ``sample_rows`` has no rows; if a row is not 1-D
            or has a different length than the first row (ragged rows);
            if every row has zero categories; if ``weights`` has a
            different length than ``sample_rows`` (one weight per
            sample row is required); if any weight is negative; or if
            the total weight is not strictly positive (a degenerate
            all-zero weight set).
    """
    rows = list(sample_rows)
    if len(rows) == 0:
        raise ValueError(
            "weighted_mean_and_variance expects at least one sample "
            "row; moments over zero veto draws are undefined"
        )
    row_arrays: list[np.ndarray] = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"each sample row must be a 1-D scoreline probability "
                f"vector, got {arr.ndim} dimension(s)"
            )
        row_arrays.append(arr)
    n_categories = len(row_arrays[0])
    if n_categories == 0:
        raise ValueError(
            "each sample row must carry at least one category "
            "probability; a zero-category row has no columns to moment"
        )
    for index, arr in enumerate(row_arrays):
        if len(arr) != n_categories:
            raise ValueError(
                f"ragged sample rows: row {index} has {len(arr)} "
                f"categories but the first row has {n_categories}; every "
                "sample must carry the same category vector length"
            )
    weight_list = list(weights)
    if len(weight_list) != len(rows):
        raise ValueError(
            f"weights has {len(weight_list)} entr(ies) but sample_rows "
            f"has {len(rows)} row(s); exactly one weight per sample row "
            "is required"
        )
    if any(float(w) < 0.0 for w in weight_list):
        raise ValueError(
            "weights must be non-negative; a negative weight is "
            "malformed probability mass"
        )
    total_weight = sum(float(w) for w in weight_list)
    if total_weight <= 0.0:
        raise ValueError(
            f"the total weight is {total_weight!r} but must be strictly "
            "positive; an all-zero weight set leaves the weighted "
            "moments undefined"
        )
    matrix = np.stack(row_arrays)
    weight_array = np.asarray(weight_list, dtype=float)
    means = np.average(matrix, axis=0, weights=weight_array)
    deviations = matrix - means
    variances = np.average(
        deviations * deviations, axis=0, weights=weight_array
    )
    return tuple(float(m) for m in means), tuple(float(v) for v in variances)

"""Pure per-category reliability-diagram helpers (roadmap M38).

The M38 (per-category reliability diagrams) statistics half: builds
standard **binned** predicted-vs-observed reliability curves per
outcome category — the "per-class reliability curve" construction
``sklearn.calibration.calibration_curve`` uses, applied once per
category rather than once globally — as a new, additional diagnostic
that *extends* (never replaces) the existing single-point per-category
calibration tables (``evaluation.harness``'s ``predicted_mean_prob``
vs ``observed_frequency`` one-point-per-category report, M19; and
``evaluation.series_evaluation``'s per-``best_of`` metric report,
M33a). The roadmap line ("Calibration assessed per outcome category
rather than on one binary probability, for both the four-way map
output and the series scorelines") is read as: for category ``c``,
treat that category's predicted probability on every held-out
observation against the binary "is this row truly in category ``c``"
indicator as one binary calibration problem, and build a reliability
diagram for it — the plan's design decision 1. The same code scores a
``K=4`` map row and a ``K=4``/``K=6`` series row identically, with no
knowledge of what a "map" or "series" is: both public functions are
fully generic over ``K`` (design decision 1), exactly like
``utils.scoring``'s batch functions.

Scope / conventions (recorded here, do not re-derive later):

- **Quantile (equal-count) binning, not fixed-width probability
  bins** (design decision 2). Observations are sorted by the
  category's predicted probability ascending (a *stable* sort — ties
  keep their original row order, so a run of equal predicted
  probabilities is split deterministically) and split into ``n_bins``
  groups of as-equal-as-possible size via ``numpy.array_split``.
  Fixed-width bins (e.g. deciles of ``[0, 1]``) would leave most bins
  empty for a sparse category like OT, where nearly every predicted
  probability is small; quantile binning always produces non-empty,
  roughly-equal-count bins regardless of the value distribution,
  which is the only viable choice at v1's small held-out N (35 maps /
  15 series).
- **No plotting/image output is in scope** (plan assumption C).
  "Diagram" here means the binned numeric data a plot would be drawn
  from — per-bin ``mean_predicted_prob`` / ``observed_frequency`` /
  ``gap`` pairs in a plain JSON-serializable dict — matching every
  other evaluation milestone's "JSON report, no rendered chart"
  convention (there is no charting library anywhere else in the repo).
- **Validation is lightweight, not a full simplex re-check** (design
  decision 5). The scored tables this module consumes were already
  simplex-validated by ``utils.scoring.rps`` / ``log_loss`` when
  ``evaluation.harness`` / ``evaluation.series_evaluation`` produced
  them, so this module does not re-implement
  ``utils.scoring._validate_probs``; it does check (raising
  ``ValueError``) the structural shape contract — empty input, ragged
  rows, category-count mismatch, length mismatch, out-of-range true
  indices, ``n_bins < 1`` and ``n_bins > n_eval`` — and deliberately
  does NOT re-validate that each row sums to 1 or that every value is
  non-negative (an intentional, stated gap, not an oversight).
- **Place in the dependency DAG.** This module sits in ``evaluation/``
  (not ``utils/``) to match the established convention that
  calibration/reliability logic lives at the evaluation rung
  (``evaluation.harness``'s calibration table,
  ``evaluation.temperature_calibration``), and may depend downward on
  ``models.*`` / ``features.*`` / ``utils.*`` only — never on
  ``drivers.*`` and never on a sibling ``evaluation/`` module
  (encoded as a regression test in ``tests/test_module_boundaries.py``;
  in practice this module imports ``numpy`` only). It takes raw
  prediction matrices + true-index arrays in (not scored DataFrames
  from ``evaluation.harness`` / ``evaluation.series_evaluation``), so
  it needs no import from either sibling — the same "take plain
  matrices in, return plain dicts out" pattern
  ``evaluation.veto_conditional_variance`` (M37) and
  ``evaluation.bootstrap_intervals`` (M36) follow, and it is 100%
  I/O-free and RNG-free.
- **No duplication debt is created here.** Unlike M36/M37's duplicated
  percentile-band helpers, this module's quantile-binning construction
  does not replicate any existing helper's logic: ``harness.py``'s
  calibration block is a single *point* per category (no binning), and
  ``evaluation.temperature_calibration`` operates on already-scored
  tables. The one-vs-rest + quantile-binned reliability curve is new
  logic introduced by M38 itself.

The two public functions:

- :func:`category_reliability_bins` — the core quantile-binning
  routine for one category: a list of per-bin dicts ordered by
  ascending predicted probability.
- :func:`build_reliability_report` — the per-category report builder:
  loops every category index of a ``(n_eval, K)`` matrix, calls
  :func:`category_reliability_bins` per category, and adds the
  count-weighted Expected Calibration Error
  (``ECE = sum(bin.count * bin.gap) / n_eval``) per category.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np


def _validate_n_bins(n_bins: int, n_eval: int) -> int:
    """Validate a bin count against an observation count and return it as an int.

    The shared guard for :func:`category_reliability_bins` and
    :func:`build_reliability_report`: coerces ``n_bins`` through
    ``operator.index`` (so plain ints and numpy integer scalars are
    both accepted, and a float/string/etc. is rejected uniformly, in
    the same house style as ``utils.scoring._validate_true_index``) and
    checks the quantile-binning feasibility range ``1 <= n_bins <=
    n_eval``: more bins than observations would silently under-fill
    bins (some bins empty) rather than erroring under
    ``numpy.array_split``, and the repo's no-silent-clamping doctrine
    (e.g. ``evaluation.bootstrap_intervals``'s ``ci_level``-boundary
    checks) requires failing loud instead.

    Args:
        n_bins: The candidate bin count.
        n_eval: The number of observations to be binned (the row count
            of the input matrix / the length of the indicator vector).

    Returns:
        The validated bin count as a plain ``int``.

    Raises:
        ValueError: If ``n_bins`` is not integer-like (a float, a
            string, etc.), is smaller than ``1``, or exceeds ``n_eval``
            (more bins than observations would leave bins empty).
    """
    try:
        bins = operator.index(n_bins)
    except TypeError as exc:
        raise ValueError(
            f"n_bins must be an integer, got {n_bins!r}"
        ) from exc
    if bins < 1:
        raise ValueError(
            f"n_bins must be at least 1, got {bins}"
        )
    if bins > n_eval:
        raise ValueError(
            f"n_bins ({bins}) must not exceed the number of "
            f"observations ({n_eval}); more bins than observations "
            "would leave bins empty — lower n_bins or add observations"
        )
    return bins


def category_reliability_bins(
    predicted_probs: Sequence[float],
    is_true_category: Sequence[float],
    n_bins: int,
) -> list[dict]:
    """Return one quantile bin per group of a one-category reliability curve.

    The core quantile-binning routine of M38, operating on exactly one
    category's worth of data (the "one-vs-rest" view of design decision
    1): treats ``predicted_probs`` (that category's predicted
    probability on every held-out observation) against
    ``is_true_category`` (a binary indicator of whether each
    observation truly is in this category) as a single binary
    calibration problem, sorts the observations by ``predicted_probs``
    ascending — a *stable* sort via ``numpy.argsort(kind="stable")``,
    so equal predicted probabilities keep their original row order and
    a run of ties is split deterministically — and splits them into
    ``n_bins`` groups of as-equal-as-possible size via
    ``numpy.array_split`` (quantile/equal-count binning, design
    decision 2: with ``n_eval`` not divisible by ``n_bins``, the first
    ``n_eval % n_bins`` bins receive the extra observation each, per
    ``numpy.array_split``'s own semantics). Returns one dict per bin in
    ascending-predicted-probability order.

    Each bin dict carries ``bin_index`` (the bin's position, ``0``
    through ``n_bins - 1``), ``count`` (how many observations fell in
    the bin), ``mean_predicted_prob`` (the arithmetic mean of the
    sorted ``predicted_probs`` in the bin), ``observed_frequency``
    (the mean of ``is_true_category`` in the bin — the fraction of the
    bin's observations truly in this category), and ``gap`` (their
    absolute difference — the per-bin calibration error the reliability
    curve plots). ``n_bins > n_eval`` raises rather than silently
    under-filling (decision 5's no-silent-clamping doctrine); the
    caller decides how to handle an observation count too small for
    its chosen bin count (the driver skips such groups with a warning,
    decision 9 — this pure function itself never silently skips).

    Args:
        predicted_probs: A 1-D sequence of ``n_eval`` floats, the
            predicted probability of this one category on every
            observation (column ``c`` of a ``(n_eval, K)`` matrix).
            Must be non-empty.
        is_true_category: A 1-D sequence of ``n_eval`` binary
            indicators (one per observation), ``1``/``True`` where the
            observation truly is in this category and ``0``/``False``
            otherwise. Must have exactly ``len(predicted_probs)``
            entries.
        n_bins: The number of quantile bins, an integer in
            ``[1, n_eval]`` (more bins than observations raises).

    Returns:
        A list of ``n_bins`` dicts ordered by ascending
        ``mean_predicted_prob``, each with keys ``bin_index`` (int),
        ``count`` (int), ``mean_predicted_prob`` (float),
        ``observed_frequency`` (float) and ``gap`` (float,
        ``abs(mean_predicted_prob - observed_frequency)``). Every value
        is a plain int/float, so the list is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If ``predicted_probs`` is empty (a reliability
            curve over zero observations is undefined); if it is not a
            1-D sequence (a per-category column must be a flat vector);
            if ``is_true_category`` has a different length than
            ``predicted_probs`` (one indicator per observation is
            required); or if ``n_bins`` is not an integer in
            ``[1, n_eval]`` (from :func:`_validate_n_bins`).
    """
    probs = np.asarray(predicted_probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError(
            "predicted_probs must be a 1-D vector of per-observation "
            f"probabilities, got {probs.ndim} dimension(s)"
        )
    n_eval = int(probs.shape[0])
    if n_eval == 0:
        raise ValueError(
            "category_reliability_bins expects at least one observation; "
            "a reliability curve over zero rows is undefined"
        )
    indicators = np.asarray(list(is_true_category), dtype=float)
    if indicators.ndim != 1:
        raise ValueError(
            "is_true_category must be a 1-D vector of binary indicators, "
            f"got {indicators.ndim} dimension(s)"
        )
    if len(indicators) != n_eval:
        raise ValueError(
            f"is_true_category has {len(indicators)} entr(ies) but "
            f"predicted_probs has {n_eval}; exactly one binary indicator "
            "per observation is required"
        )
    bins = _validate_n_bins(n_bins, n_eval)

    # Stable ascending sort of the observations by predicted
    # probability; ties keep their original row order.
    order = np.argsort(probs, kind="stable")
    sorted_probs = probs[order]
    sorted_true = indicators[order]
    # Quantile split: n_eval observations into bins groups of
    # as-equal-as-possible size (numpy.array_split's own semantics).
    groups = np.array_split(np.arange(n_eval), bins)

    result: list[dict] = []
    for bin_index, group in enumerate(groups):
        bin_probs = sorted_probs[group]
        bin_true = sorted_true[group]
        mean_predicted = float(np.mean(bin_probs))
        observed = float(np.mean(bin_true))
        result.append(
            {
                "bin_index": bin_index,
                "count": len(group),
                "mean_predicted_prob": mean_predicted,
                "observed_frequency": observed,
                "gap": abs(mean_predicted - observed),
            }
        )
    return result


def build_reliability_report(
    prob_rows: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    category_labels: Sequence[str],
    n_bins: int,
) -> dict:
    """Build the JSON-serializable per-category reliability report.

    A pure dict builder (no I/O): takes the raw scored prediction
    matrix and the parallel true-category index vector (exactly the
    two arrays ``evaluation.harness`` / ``evaluation.series_evaluation``
    can hand it from their scored tables — the map arm extracts
    ``scored_df[list(harness.PREDICTION_COLUMNS)].to_numpy()`` and
    ``scored_df["outcome_ordinal"].to_numpy()``, the series arm
    ``np.array(list(subset["probabilities"]))`` and
    ``subset["outcome_index"].to_numpy()``), and returns one binned
    reliability diagram per outcome category plus the count-weighted
    Expected Calibration Error per category. Fully generic over ``K``
    (design decision 1): a ``K=4`` map row and a ``K=4``/``K=6``
    series row are scored identically, with no knowledge of what a
    "map" or "series" is.

    For every category index ``c`` in ``0..K-1`` the builder extracts
    column ``c`` of ``prob_rows`` (that category's predicted
    probability on every observation) and the binary indicator
    ``true_indices == c`` (design decision 1's one-vs-rest view), calls
    :func:`category_reliability_bins` on them, and computes the
    category's Expected Calibration Error as the count-weighted mean of
    the per-bin gaps — ``ECE = sum_over_bins(bin.count * bin.gap) /
    n_eval``, the standard ECE definition (a bin with more observations
    contributes proportionally more). Returns
    ``{"n_eval": int, "n_bins": int, "categories": [{"category": label,
    "expected_calibration_error": float, "bins": [...]}, ...]}`` — one
    entry per ``category_labels`` entry, in order.

    Args:
        prob_rows: A ``(n_eval, K)`` matrix of floats, either a 2-D
            numpy array or a sequence of equal-length row sequences
            (one observation's predicted probability vector per row,
            each row's entries summing to 1 — sum-to-1 is *not*
            re-validated here, decision 5). Must have at least one
            row, and every row must have the same positive number of
            categories.
        true_indices: A sequence of ``n_eval`` true-category indices,
            one per row of ``prob_rows``, each an integer in
            ``[0, K)``.
        category_labels: The category vocabulary, one label per column
            of ``prob_rows`` in order (e.g.
            ``evaluation.harness.OUTCOME_LABELS`` for a ``K=4`` map
            matrix, or the ``f"{a}-{b}"`` scoreline strings for a
            series matrix). ``len(category_labels)`` must equal ``K``.
        n_bins: The number of quantile bins per category, an integer
            in ``[1, n_eval]``.

    Returns:
        A dict with keys ``n_eval`` (int), ``n_bins`` (int) and
        ``categories`` (a list, one dict per ``category_labels`` entry
        in order, each with ``category`` (str), 
        ``expected_calibration_error`` (float, the count-weighted ECE
        of design decision 4) and ``bins`` (the
        :func:`category_reliability_bins` list for that category)).
        Every value is a plain str/int/float/list/dict, so the whole
        dict is directly ``json.dumps``-serializable.

    Raises:
        ValueError: If ``prob_rows`` has no rows; if a row is not 1-D
            or has a different length than the first row (ragged
            rows); if ``len(category_labels)`` differs from the row
            width ``K``; if ``true_indices`` has a different length
            than ``prob_rows`` (one true index per row is required);
            if any ``true_indices`` entry lies outside ``[0, K)``; or
            if ``n_bins`` is not an integer in ``[1, n_eval]`` (from
            :func:`_validate_n_bins` — including ``n_bins > n_eval``,
            which raises rather than silently under-filling, and
            ``n_bins < 1``).
    """
    rows = list(prob_rows)
    if len(rows) == 0:
        raise ValueError(
            "build_reliability_report expects at least one observation "
            "row; a reliability report over zero rows is undefined"
        )
    row_arrays: list[np.ndarray] = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"each observation row must be a 1-D probability "
                f"vector, got {arr.ndim} dimension(s)"
            )
        row_arrays.append(arr)
    n_categories = len(row_arrays[0])
    if n_categories == 0:
        raise ValueError(
            "each observation row must carry at least one category "
            "probability; a zero-category row has no columns to "
            "calibrate"
        )
    for index, arr in enumerate(row_arrays):
        if len(arr) != n_categories:
            raise ValueError(
                f"ragged observation rows: row {index} has {len(arr)} "
                f"categories but the first row has {n_categories}; every "
                "row must carry the same probability vector length"
            )
    if len(category_labels) != n_categories:
        raise ValueError(
            f"category_labels has {len(category_labels)} entries but "
            f"each row has {n_categories} categories; they must match "
            "one-to-one"
        )
    n_eval = len(rows)
    if len(true_indices) != n_eval:
        raise ValueError(
            f"true_indices has {len(true_indices)} entr(ies) but "
            f"prob_rows has {n_eval} row(s); exactly one true index per "
            "row is required"
        )
    true_array = np.asarray(list(true_indices), dtype=int)
    if not np.all((true_array >= 0) & (true_array < n_categories)):
        first_bad = int(true_array[np.argmax(
            ~((true_array >= 0) & (true_array < n_categories))
        )])
        raise ValueError(
            f"true_indices must lie in [0, {n_categories}), got an entry "
            f"{first_bad!r} outside that range; a true index must name "
            "a real category column"
        )
    bins_per_category = _validate_n_bins(n_bins, n_eval)

    matrix = np.stack(row_arrays)
    indicator_matrix = np.stack(
        [true_array == c for c in range(n_categories)],
        axis=1,
    )

    categories: list[dict] = []
    for c, label in enumerate(category_labels):
        per_category_bins = category_reliability_bins(
            matrix[:, c], indicator_matrix[:, c], bins_per_category
        )
        # Count-weighted ECE (design decision 4): sum over bins of
        # bin.count * bin.gap, normalized by n_eval, so a bin with more
        # observations contributes proportionally more.
        ece = (
            sum(bin_record["count"] * bin_record["gap"]
                for bin_record in per_category_bins)
            / n_eval
        )
        categories.append(
            {
                "category": label,
                "expected_calibration_error": float(ece),
                "bins": per_category_bins,
            }
        )
    return {
        "n_eval": n_eval,
        "n_bins": bins_per_category,
        "categories": categories,
    }

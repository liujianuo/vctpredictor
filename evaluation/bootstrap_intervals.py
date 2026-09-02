"""Pure percentile-interval and backing-sample-size helpers (roadmap M36).

The model-free statistics half of M36 (bootstrap prediction intervals,
epistemic). The M36 *orchestration* — resample the training set with
replacement, refit the Stage-2 map model, run the M31 series pipeline
B times — cannot live in ``evaluation/``: it needs
``drivers.training_data`` (upward) and the sibling
``evaluation.veto_marginalized_series`` / ``evaluation.series_evaluation``
modules (lateral), both forbidden by the module-boundary DAG encoded in
``tests/test_module_boundaries.py``. That orchestration lives in
``drivers/evaluate_bootstrap_intervals.py``; this module holds only the
generic, model-free helpers every caller needs — today that driver, and
later M39's ``predict()`` public API against a list of already-fitted
replicate models:

- :func:`percentile_interval` — a plain ``[low_pct, high_pct]``
  percentile band over an arbitrary sequence of floats.
- :func:`replicate_matrix_intervals` — the same band applied
  independently per outcome category to a
  ``(n_replicates, n_categories)`` matrix of bootstrap-replicate
  probability vectors (works identically for the 4-way per-map case and
  the K-way per-series scoreline case — no ``OUTCOME_LABELS``-specific
  coupling, mirroring ``utils.scoring``'s "generic over K" doctrine).
- :func:`n_games_backing` — ``min(games_a, games_b)``, the weaker
  side's per-map sample size backing a prediction.

Phase 6 framing (recorded here, do not re-derive in later milestones;
the same text is restated in the driver's docstring): M36 (epistemic —
parameter uncertainty from a finite training sample), M37 (structural —
spread across sampled veto sequences, resolves the moment the veto
happens) and M38 (calibration/reliability) are three distinct
components, reported separately, never collapsed into one number. This
module is the statistics engine of the epistemic component ONLY. Its
helpers are deliberately agnostic about *what* the replicate spread
represents — the driver is responsible for keeping the veto-sampling
randomness fixed across replicates (same seed reconstructed identically
each time) so the spread this module summarizes is attributable to
Stage-2 parameter uncertainty alone, not to veto-sequence variance
(that structural variance is M37's separate remit and must not leak
into these bands). Only the Stage-2 map model
(``models.ordinal_logit``) is bootstrapped anywhere in M36; the M27/M28
conditional-logit ban/pick predictors are loaded as fixed artifacts and
reused unchanged across every replicate (the single biggest scope
decision of the M36 plan — flag for REVIEW).

**Interval definition: independent per-category percentile bands, not a
joint simplex region.** For each of the K categories, the interval is
the ``[low_percentile, high_percentile]`` of that category's
probability across the B bootstrap replicates, computed independently
per category. This is a deliberately simple v1 choice: the K bands do
not jointly form a calibrated region on the simplex (they are marginal
bands, and a random draw respecting every marginal band individually
need not itself sum to 1). Do not misread the per-category bands as a
joint credible region; the driver writes this same caveat into the
artifact.

Placement in the dependency DAG: this module sits in ``evaluation/``
and may depend downward on ``models.*`` / ``features.*`` / ``utils.*``
only — never on ``drivers.*`` and never on a sibling ``evaluation/``
module. It imports ``numpy`` only (precedented by every other module in
this repo) and is 100% I/O-free: it neither reads nor writes files.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_ci_level(ci_level: float) -> float:
    """Validate a confidence-interval level and return it as a float.

    The shared guard for :func:`percentile_interval` and
    :func:`replicate_matrix_intervals`: a ``ci_level`` must be a single
    real number strictly between 0 and 1 (``0.90`` means the
    5th/95th-percentile band). Values at the boundary (``0.0``, ``1.0``)
    are rejected: a degenerate band (a single point, or the full range
    of the sample) is not a meaningful uncertainty statement.

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


def percentile_interval(
    values: Sequence[float],
    ci_level: float = 0.90,
) -> tuple[float, float]:
    """Return the ``[low, high]`` percentile band over a value sequence.

    Computes the ``((1 - ci_level) / 2)``-th and
    ``((1 + ci_level) / 2)``-th percentiles of ``values`` via
    ``numpy.percentile`` (default linear interpolation, so even and odd
    replicate counts produce the standard percentile semantics), i.e.
    the 5th/95th percentiles for the default ``ci_level = 0.90``. The
    band is a *marginal* statement about the empirical distribution of
    ``values`` — there is no joint-simplex claim, per the module
    docstring's interval-definition note.

    Args:
        values: An arbitrary non-empty sequence of floats (e.g. one
            category's probability across the bootstrap replicates).
        ci_level: The confidence level in ``(0, 1)``; the band spans
            the middle ``ci_level`` fraction of the sample (default
            ``0.90``).

    Returns:
        A ``(lo, hi)`` tuple of ``float`` percentiles: ``lo`` is the
            ``((1 - ci_level)/2)``-th percentile, ``hi`` the
            ``((1 + ci_level)/2)``-th, with ``lo <= hi`` (equality when
            all values are equal, or for a single value).

    Raises:
        ValueError: If ``values`` is empty (a percentile of an empty
            sample is undefined); if ``ci_level`` is not in ``(0, 1)``
            (from :func:`_validate_ci_level`); or if ``values`` cannot
            be coerced to a 1-D float array.
    """
    level = _validate_ci_level(ci_level)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"percentile_interval expects a 1-D sequence of values, got "
            f"{arr.ndim} dimension(s)"
        )
    if arr.size == 0:
        raise ValueError(
            "percentile_interval expects a non-empty value sequence; a "
            "percentile band over zero replicates is undefined"
        )
    lo_percent = ((1.0 - level) / 2.0) * 100.0
    hi_percent = ((1.0 + level) / 2.0) * 100.0
    lo, hi = np.percentile(arr, [lo_percent, hi_percent])
    return float(lo), float(hi)


def replicate_matrix_intervals(
    replicate_rows: Sequence[Sequence[float]],
    ci_level: float = 0.90,
) -> tuple[tuple[float, float], ...]:
    """Return one per-category percentile band per column of a replicate matrix.

    Applies :func:`percentile_interval` independently per outcome
    category to a ``(n_replicates, n_categories)`` matrix of
    bootstrap-replicate probability vectors: column ``j`` holds
    category ``j``'s probability under each of the ``n_replicates``
    bootstrap models, and the returned tuple's ``j``-th entry is that
    column's ``[lo, hi]`` band at ``ci_level``. The helper is generic
    over ``n_categories`` (any ``K >= 1``), so it works identically for
    the 4-way per-map case and the K-way per-series scoreline case
    without any ``OUTCOME_LABELS``-specific coupling — mirroring
    ``utils.scoring``'s "generic over K" doctrine. The bands are
    independent per category (marginal, not a joint simplex region —
    see the module docstring).

    Args:
        replicate_rows: A ``(n_replicates, n_categories)`` matrix of
            floats, either a 2-D numpy array or a sequence of equal-
            length row sequences (one replicate probability vector per
            row). Must have at least one replicate row, and every row
            must have the same positive number of categories.
        ci_level: The confidence level in ``(0, 1)``, passed through to
            :func:`percentile_interval` (default ``0.90``).

    Returns:
        A tuple of ``(lo, hi)`` float pairs, one per category in column
            order: entry ``j`` is the ``[lo, hi]`` percentile band of
            category ``j`` across the replicate rows, with ``lo <= hi``
            in every entry.

    Raises:
        ValueError: If ``replicate_rows`` has no rows; if a row is not
            1-D or has a different length than the first row (ragged
            rows); if every row has zero categories; or if ``ci_level``
            is not in ``(0, 1)`` (from
            :func:`_validate_ci_level`).
    """
    level = _validate_ci_level(ci_level)
    rows = list(replicate_rows)
    if len(rows) == 0:
        raise ValueError(
            "replicate_matrix_intervals expects at least one replicate "
            "row; a percentile band over zero replicates is undefined"
        )
    row_arrays: list[np.ndarray] = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"each replicate row must be a 1-D probability vector, "
                f"got {arr.ndim} dimension(s)"
            )
        row_arrays.append(arr)
    n_categories = len(row_arrays[0])
    if n_categories == 0:
        raise ValueError(
            "each replicate row must carry at least one category "
            "probability; a zero-category row has no columns to band"
        )
    for index, arr in enumerate(row_arrays):
        if len(arr) != n_categories:
            raise ValueError(
                f"ragged replicate rows: row {index} has {len(arr)} "
                f"categories but the first row has {n_categories}; every "
                "replicate must carry the same category vector length"
            )
    matrix = np.stack(row_arrays)
    bands: list[tuple[float, float]] = []
    for column in range(n_categories):
        lo, hi = percentile_interval(matrix[:, column], ci_level=level)
        bands.append((lo, hi))
    return tuple(bands)


def n_games_backing(games_a: int, games_b: int) -> int:
    """Return the weaker side's per-map sample size backing a prediction.

    Computes ``min(games_a, games_b)`` — the number of as-of, map-
    specific games the *less data-rich* side contributes to a
    head-to-head map prediction — documented as "the weaker side's
    per-map sample size backing this prediction". ``min`` is chosen
    over ``sum`` because the sum would overstate confidence when one
    side is data-rich and the other is brand new (a 100-0 pairing would
    claim 100 games of backing); ``min`` is the conservative "how much
    do we actually know about THIS specific pairing" number. The two
    game counts are supplied by the caller from
    ``features.map_win_rate.team_map_win_rate(...).games`` (the
    as-of, map-specific count that Stage 2's own ``map_win_rate_diff``
    feature already builds; the shrinkage ``k`` does not affect
    ``games``).

    Args:
        games_a: Side A's as-of game count on the queried map; a
            non-negative integer (numpy integers accepted).
        games_b: Side B's as-of game count on the queried map; a
            non-negative integer (numpy integers accepted).

    Returns:
        ``min(games_a, games_b)`` as an ``int`` — the weaker side's
            backing sample size. Symmetric: swapping the two arguments
            returns the same value.

    Raises:
        ValueError: If either count is negative (a negative game count
            is malformed data) or cannot be coerced to an integer.
    """
    try:
        ga = int(games_a)
        gb = int(games_b)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"games counts must be integers, got games_a={games_a!r} "
            f"games_b={games_b!r}"
        ) from exc
    if ga < 0 or gb < 0:
        raise ValueError(
            f"games counts must be non-negative, got games_a={games_a!r} "
            f"games_b={games_b!r}"
        )
    return min(ga, gb)

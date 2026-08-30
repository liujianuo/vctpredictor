"""Pure chronological split, walk-forward folds, and out-of-fold assembly (M10).

Computes the single shared notion of "past" vs "future" that every
downstream evaluation/calibration stage (M19, M24, M26, M32) needs:
from ``materialize.py``'s (M8) ``data/<version>/matches.parquet`` it
derives a two-way chronological ``train``/``test`` split (split unit =
match), a walk-forward expanding-window fold generator over the
``train`` region, and a self-checking assembler that stitches per-fold
validation-block predictions into one out-of-fold (OOF) calibration
set.

This module is pure in-memory logic: it operates on ``pandas``
DataFrames, has no CLI, no ``argparse`` entry point, no logging
configuration, and no file I/O of its own. The CLI entry point and the
Parquet-reading/writing glue live in ``drivers/splits.py``, which
re-exports every name defined here (see its module docstring). Keeping
the pure split/fold/assembly logic here — rather than in a driver
module — lets other modules (e.g. ``features/map_win_rate.py``)
reuse it without importing a driver, preserving the established
``drivers -> utils -> features`` layering rule.

Key design rules (mirroring ``labels.py``'s rule list):

- **Split unit is the match, not the map.** ``maps.parquet`` has no
  date column of its own and a match's maps are sequential games in
  one series, not independent chronological draws. The split is
  computed once per ``match_id`` and propagated to finer tables via
  :func:`join_split_to_maps`.
- **Two split values only.** ``splits.parquet`` carries exactly
  ``"train"`` (earliest matches) and ``"test"`` (most recent ~15%,
  held out for final evaluation only). There is deliberately no third
  static calibration slice: calibration is collected out-of-fold via
  :func:`assemble_out_of_fold_predictions` instead.
- **Split basis is match count, not calendar time.** A
  calendar-quantile cutoff would make holdout size unpredictable under
  ``data/v1``'s uneven cadence; ``n_test = max(1, round(n *
  test_frac))`` keeps sizes stable.
- **Chronological tie-break is ``(date, match_id)``.** ``date`` is
  parsed via ``pandas.to_datetime`` purely to establish sort order;
  the original string values are written back unchanged so
  ``splits.parquet`` stays joinable against ``matches.parquet``.
- **Walk-forward = expanding window, first block never validated.**
  The training region is split into ``effective_n_folds + 1``
  contiguous chronological blocks (remainder folded into the first,
  "warm-up" block); fold ``i`` trains on blocks ``0..i-1`` and
  validates on block ``i``. The warm-up block has no strictly-prior
  training data and therefore never receives an OOF prediction.
- **:func:`walk_forward_folds` trusts, does not verify, that its input
  is already the training region.** It does not require or check a
  ``split`` column; the caller is responsible for filtering
  ``splits.parquet`` to ``split == "train"`` rows first.
- **:func:`assemble_out_of_fold_predictions` enforces, does not
  trust, the caller's fold assignment.** It recomputes the
  authoritative ``val_match_ids`` per fold itself and rejects any
  submission whose predicted id *set* does not exactly match, so a
  train-fold or ``test``-slice prediction cannot leak into the OOF
  calibration set. The check is a set-membership comparison, not a row
  count, so one-row-per-map submissions validate unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

# The only two split values. The deliberate absence of a third static
# "calibration" value is a design decision (see module docstring):
# calibration is collected out-of-fold by assemble_out_of_fold_predictions.
SPLIT_VALUES = ("train", "test")

# Fixed column order for the splits table, used both to build non-empty
# DataFrames in a deterministic order and to give an empty run a
# schema-correct zero-row table.
SPLITS_COLUMNS = ("match_id", "date", "split")

# Fixed target dtypes for the splits table, applied to every row count
# (including zero) so an empty run writes exactly the same Parquet
# schema as a non-empty one: all three columns are object (pandas'
# default string container, matching what materialize.py's matches
# table produces for match_id/date). Without this, an empty DataFrame
# constructed from ``rows=[]`` would round-trip through Parquet with a
# null dtype and break downstream consumers that assume one fixed
# schema across dataset versions (the same bug labels.LABELS_DTYPES
# exists to prevent).
SPLITS_DTYPES = {
    "match_id": object,
    "date": object,
    "split": object,
}

# Test-holdout fraction. Only one fraction exists: train gets everything
# test does not (n_train = n - n_test), so it is exactly "100% minus
# test" rather than an independently configured 85%.
DEFAULT_TEST_FRAC = 0.15

# Defensive floor on the resulting train-region size (an
# invariant-break, not an expected-absent-data case).
MIN_TRAIN_MATCHES = 20

# Walk-forward fold defaults, chosen by this plan (not by
# description.txt, which specifies neither number) and used only by the
# library-level functions walk_forward_folds /
# assemble_out_of_fold_predictions — not exposed as CLI flags.
DEFAULT_N_FOLDS = 5
MIN_FOLD_BLOCK_MATCHES = 8


def _chronological_order(dates: pd.Series, match_ids: pd.Series) -> list[int]:
    """Return positional row indices sorted by ``(date, match_id)``.

    The shared chronological order used by :func:`split_matches` and
    :func:`walk_forward_folds` (design decision 7): primary key is the
    parsed ``date``, secondary key is ``match_id`` as a deterministic
    tie-break for equal dates. ``date`` is parsed purely to establish
    order — the callers write the original (unparsed) values back into
    their output untouched.

    Args:
        dates: The date column to sort by (any dtype
            ``pandas.to_datetime`` accepts; original values are not
            rewritten by this helper).
        match_ids: The match id column used as the secondary sort key.

    Returns:
        A list of positional indices (into ``dates``/``match_ids``)
        ordered chronologically, oldest first.

    Raises:
        ValueError: If any date cannot be parsed (propagated from
            ``pandas.to_datetime`` with its default ``errors="raise"``),
            or if any parsed date is null (``None``/``NaN`` parses to
            ``NaT`` rather than raising, and ``NaT`` has no
            chronological position — it compares ``False`` in both
            directions against real timestamps and would otherwise
            sort to an arbitrary, order-breaking position).
        TypeError: If ``match_ids`` contains values that are not
            mutually comparable (only possible with a mixed-type
            ``match_id`` column, which M8 never produces).
    """
    parsed = pd.to_datetime(dates)
    null_mask = parsed.isna()
    if null_mask.any():
        null_positions = list(np.flatnonzero(null_mask.to_numpy()))
        raise ValueError(
            f"dates contains {len(null_positions)} null value(s) at row(s) "
            f"{null_positions}: a null date (None/NaN) parses to NaT and "
            "has no chronological position, so it cannot be sorted for "
            "the split"
        )
    # Vectorized two-key sort: build a frame of positional values
    # (parsed date, match_id), sort it, and read the resulting
    # positional order. This replaces the per-comparison Python
    # ``sorted()`` lambda (two ``Series.iloc`` lookups per comparison)
    # with one pandas/numpy sort over the whole column.
    sort_keys = pd.DataFrame(
        {"date": parsed.to_numpy(), "match_id": match_ids.to_numpy()}
    )
    order = sort_keys.sort_values(
        ["date", "match_id"], kind="stable"
    ).index.to_numpy()
    return [int(i) for i in order]


def split_matches(
    matches_df: pd.DataFrame,
    test_frac: float = DEFAULT_TEST_FRAC,
    date_col: str = "date",
    id_col: str = "match_id",
) -> pd.DataFrame:
    """Compute the two-way chronological ``train``/``test`` split.

    Sorts ``matches_df`` by ``(date_col, id_col)``, holds out the most
    recent ``max(1, round(n * test_frac))`` matches as ``"test"`` and
    labels the earliest ``n_train = n - n_test`` matches ``"train"``.
    ``date_col`` is parsed via ``pandas.to_datetime`` purely to
    establish sort order, but the *original* string date values are
    written back into the output (not reformatted datetimes), so
    ``splits.parquet`` stays joinable against ``matches.parquet``'s own
    ``date`` values without a type mismatch.

    Args:
        matches_df: The materialised ``matches`` table (M8's
            ``matches.parquet``) with at least ``date_col`` and
            ``id_col``. Only those two columns are read; every other
            column is ignored.
        test_frac: The fraction of matches to hold out for final
            evaluation. Must be strictly between 0 and 1.
        date_col: The name of the date column to sort by.
        id_col: The name of the match id column to sort by (secondary
            key) and to carry into the output.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SPLITS_COLUMNS`
        (``match_id, date, split``) in chronological order: the
        earliest ``n_train`` rows have ``split == "train"`` and the
        last ``n_test`` rows have ``split == "test"``. The frame always
        carries the fixed :data:`SPLITS_DTYPES` schema regardless of
        row count.

    Raises:
        ValueError: If ``test_frac`` is not in ``(0, 1)``; if the table
            is empty; if the resulting ``n_train`` falls below
            :data:`MIN_TRAIN_MATCHES`; if a date is unparseable
            (propagated from ``pandas.to_datetime``); or if a date is
            null (``None``/``NaN``), which parses to ``NaT`` and has
            no chronological position (raised by
            :func:`_chronological_order`).
        KeyError: If ``matches_df`` lacks ``date_col`` or ``id_col``.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(
            f"test_frac must be in (0, 1), got {test_frac}"
        )

    try:
        match_ids = matches_df[id_col]
        dates = matches_df[date_col]
    except KeyError as exc:
        # pandas raises KeyError for a missing column; re-raise
        # unchanged so the missing-name signal is preserved.
        raise KeyError(str(exc)) from exc

    n = len(matches_df)
    if n == 0:
        raise ValueError("cannot split an empty matches table (0 rows)")

    order = _chronological_order(dates, match_ids)
    sorted_match_ids = [match_ids.iloc[i] for i in order]
    sorted_dates = [dates.iloc[i] for i in order]

    n_test = max(1, round(n * test_frac))
    n_train = n - n_test
    if n_train < MIN_TRAIN_MATCHES:
        raise ValueError(
            f"test_frac={test_frac} leaves only {n_train} training "
            f"matches (minimum {MIN_TRAIN_MATCHES})"
        )

    split_values = ["train"] * n_train + ["test"] * n_test
    splits_df = pd.DataFrame(
        {
            "match_id": sorted_match_ids,
            "date": sorted_dates,
            "split": split_values,
        },
        columns=SPLITS_COLUMNS,
    )
    return splits_df.astype(SPLITS_DTYPES)


def walk_forward_folds(
    train_matches_df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    min_fold_block: int = MIN_FOLD_BLOCK_MATCHES,
    date_col: str = "date",
    id_col: str = "match_id",
) -> Iterator[tuple[int, list, list]]:
    """Yield walk-forward ``(fold_id, train_ids, val_ids)`` fold splits.

    An expanding-window fold generator over a *training-region-only*
    table (see the module docstring's trust boundary): the input is
    sorted by ``(date_col, id_col)`` and divided into
    ``effective_n_folds + 1`` contiguous chronological blocks where
    ``effective_n_folds = min(n_folds, n // min_fold_block - 1)``; any
    remainder is folded into the first ("warm-up") block. Fold ``i``
    (1-based) trains on the expanding union of blocks ``0..i-1`` and
    validates on block ``i``. The warm-up block itself is never
    validated — there is no strictly-prior data to train a model on
    for the training region's own earliest matches.

    Args:
        train_matches_df: The training-region matches table (the
            caller is responsible for having filtered ``splits.parquet``
            to ``split == "train"`` rows; this function does not check
            for a ``split`` column). Needs at least ``date_col`` and
            ``id_col``.
        n_folds: The maximum number of folds to produce; the effective
            count is capped by the available data (see above).
        min_fold_block: The minimum number of matches a validation
            block must contain; together with ``n`` it bounds the
            effective fold count.
        date_col: The name of the date column to sort by.
        id_col: The name of the match id column to sort by and to emit
            as fold ids.

    Yields:
        ``(fold_id, train_match_ids, val_match_ids)`` tuples, one per
        effective fold in ascending 1-based ``fold_id`` order.
        ``train_match_ids`` and ``val_match_ids`` are plain ``list``
        of the ``id_col`` values (not filtered DataFrames), so callers
        stay agnostic to whichever table they want to slice by these
        ids.

    Raises:
        ValueError: If ``effective_n_folds < 1`` (the input is too
            small to form even one fold — e.g. ``n <= min_fold_block``);
            if a date is unparseable (propagated from
            ``pandas.to_datetime``); or if a date is null
            (``None``/``NaN``), which parses to ``NaT`` and has no
            chronological position (raised by
            :func:`_chronological_order`).
        KeyError: If ``train_matches_df`` lacks ``date_col`` or
            ``id_col``.
    """
    try:
        match_ids = train_matches_df[id_col]
        dates = train_matches_df[date_col]
    except KeyError as exc:
        raise KeyError(str(exc)) from exc

    n = len(train_matches_df)
    effective_n_folds = min(n_folds, n // min_fold_block - 1)
    if effective_n_folds < 1:
        raise ValueError(
            f"training region of {n} matches is too small to form even "
            f"one walk-forward fold (need at least "
            f"{2 * min_fold_block} for min_fold_block={min_fold_block})"
        )

    order = _chronological_order(dates, match_ids)
    sorted_ids = [match_ids.iloc[i] for i in order]

    n_blocks = effective_n_folds + 1
    base, remainder = divmod(n, n_blocks)
    # Remainder is folded into the first (warm-up) block; every other
    # block is ``base`` matches. ``effective_n_folds >= 1`` plus the
    # ``min_fold_block`` cap guarantees ``base >= min_fold_block``, so
    # no block is ever empty.
    block_sizes = [base + remainder] + [base] * (n_blocks - 1)

    blocks: list[list] = []
    start = 0
    for size in block_sizes:
        blocks.append(sorted_ids[start : start + size])
        start += size

    for i in range(1, effective_n_folds + 1):
        train_ids = [mid for block in blocks[:i] for mid in block]
        yield (i, train_ids, blocks[i])


def assemble_out_of_fold_predictions(
    train_matches_df: pd.DataFrame,
    fold_predictions: Sequence[tuple[int, pd.DataFrame]],
    n_folds: int = DEFAULT_N_FOLDS,
    min_fold_block: int = MIN_FOLD_BLOCK_MATCHES,
    date_col: str = "date",
    id_col: str = "match_id",
) -> tuple[pd.DataFrame, dict]:
    """Assemble per-fold validation-block predictions into one OOF set.

    The calibration-set assembler for M24 (and the check M20's own
    walk-forward hyperparameter-selection loop will submit to). It does
    **not** trust the caller's claim about which predictions belong to
    which fold: it re-runs :func:`walk_forward_folds` on the same
    ``(train_matches_df, n_folds, min_fold_block)`` configuration to
    independently recompute the authoritative ``val_match_ids`` per
    fold, then verifies each submitted ``predictions_df`` covers
    exactly that fold's validation matches — no extra ids (a train-fold
    or ``test``-slice prediction leaking into the OOF set) and no
    missing ids (an incomplete submission). The comparison is a
    set-membership comparison on ``id_col``, deliberately not a row
    count, so a one-row-per-map submission (``match_id`` repeating
    across a Bo3/Bo5's maps) validates unchanged.

    Args:
        train_matches_df: The training-region matches table (same
            trust boundary as :func:`walk_forward_folds`: must already
            be filtered to the training region). Needs ``date_col`` and
            ``id_col``.
        fold_predictions: A sequence of ``(fold_id, predictions_df)``
            pairs; ``predictions_df`` must contain ``id_col`` plus
            whatever prediction/label columns the caller produced
            (typically one row per finished map for M20). Any
            ``fold_id`` column the caller included is overwritten by
            this function's authoritative ``fold_id``.
        n_folds: Passed to the internal :func:`walk_forward_folds`
            recomputation; must match the value the caller used to
            produce ``fold_predictions``.
        min_fold_block: Passed to the internal recomputation; must
            match the caller's value.
        date_col: The name of the date column to sort by.
        id_col: The name of the match id column.

    Returns:
        A tuple of ``(assembled_df, coverage)``. ``assembled_df`` is
        the concatenation of every ``predictions_df`` in ascending
        ``fold_id`` order with an authoritative ``fold_id`` column
        prepended as the first column. ``coverage`` is a dict with
        ``train_matches`` (row count of ``train_matches_df``),
        ``covered_matches`` (number of distinct match ids covered
        across all folds), ``warmup_excluded_ids`` (the match ids in
        the first/warm-up block, which never receive an OOF
        prediction), and ``warmup_excluded_count`` (their count).

    Raises:
        ValueError: If a ``fold_id`` in ``fold_predictions`` is not one
            of the recomputed fold ids; if the same ``fold_id`` is
            submitted more than once; if ``fold_predictions`` omits one
            of the required fold ids; or if a fold's predicted id set
            does not exactly equal its recomputed ``val_match_ids``
            (extra = leak, missing = incomplete). The first three of
            these (and the set-mismatch case) are the only symptoms a
            wrong ``n_folds``/``min_fold_block`` produces, so their
            messages explicitly point at that likely cause instead of
            leaving the caller to decode a bare id-set error. Also
            propagates :func:`walk_forward_folds`'s ``ValueError``
            (e.g. training region too small) and ``pandas.to_datetime``
            errors.
        KeyError: If a ``predictions_df`` lacks ``id_col`` (or if
            ``train_matches_df`` lacks ``date_col``/``id_col``,
            propagated from :func:`walk_forward_folds`).
    """
    folds = list(
        walk_forward_folds(
            train_matches_df,
            n_folds=n_folds,
            min_fold_block=min_fold_block,
            date_col=date_col,
            id_col=id_col,
        )
    )
    # folds is non-empty: walk_forward_folds raises ValueError before
    # yielding anything when effective_n_folds < 1.
    required_fold_ids = [fold_id for fold_id, _, _ in folds]

    submitted_fold_ids = [fold_id for fold_id, _ in fold_predictions]
    if len(submitted_fold_ids) != len(set(submitted_fold_ids)):
        raise ValueError(
            "fold_predictions contains a duplicate fold_id; each fold "
            "must be submitted exactly once"
        )

    for fold_id in submitted_fold_ids:
        if fold_id not in required_fold_ids:
            raise ValueError(
                f"fold_id {fold_id} is not one of the recomputed fold "
                f"ids {required_fold_ids} (recomputed with n_folds="
                f"{n_folds}, min_fold_block={min_fold_block}); this "
                f"usually means the caller produced fold_predictions "
                f"with a different fold configuration"
            )

    missing = set(required_fold_ids) - set(submitted_fold_ids)
    if missing:
        raise ValueError(
            f"fold_predictions is missing required fold_ids "
            f"{sorted(missing)} (recomputed with n_folds={n_folds}, "
            f"min_fold_block={min_fold_block}); this usually means the "
            f"caller produced fold_predictions with a different fold "
            f"configuration"
        )

    by_fold = {fold_id: predictions_df for fold_id, predictions_df in fold_predictions}
    val_ids_by_fold = {fold_id: set(val_ids) for fold_id, _, val_ids in folds}

    assembled_parts: list[pd.DataFrame] = []
    for fold_id in required_fold_ids:
        predictions_df = by_fold[fold_id]
        if id_col not in predictions_df.columns:
            raise KeyError(f"predictions_df for fold {fold_id} lacks {id_col!r}")
        predicted_ids = set(predictions_df[id_col])
        expected_ids = val_ids_by_fold[fold_id]
        if predicted_ids != expected_ids:
            extra = sorted(predicted_ids - expected_ids)
            missing_ids = sorted(expected_ids - predicted_ids)
            raise ValueError(
                f"fold {fold_id}: predicted match-id set does not match its "
                f"recomputed validation set "
                f"(extra={extra or None}, missing={missing_ids or None}); "
                f"this is either a real leak/incomplete submission or the "
                f"caller's n_folds/min_fold_block differ from the values "
                f"used to produce the predictions"
            )
        sub = predictions_df.drop(columns=["fold_id"], errors="ignore").copy()
        sub.insert(0, "fold_id", fold_id)
        assembled_parts.append(sub)

    assembled_df = pd.concat(assembled_parts, ignore_index=True)

    warmup_excluded_ids = list(folds[0][1])
    coverage = {
        "train_matches": len(train_matches_df),
        "covered_matches": sum(len(val_ids) for _, _, val_ids in folds),
        "warmup_excluded_ids": warmup_excluded_ids,
        "warmup_excluded_count": len(warmup_excluded_ids),
    }
    return assembled_df, coverage


def join_split_to_maps(
    maps_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    id_col: str = "match_id",
) -> pd.DataFrame:
    """Attach the ``split`` column onto a match-id-keyed map table.

    Left-merges ``splits_df[[id_col, "split"]]`` onto ``maps_df`` so a
    map-level table (most obviously M8's ``maps.parquet``) gains the
    two-valued ``split`` label via its ``match_id``. The merge is a
    left join: every ``maps_df`` row is preserved in its original
    order and only the ``split`` column is added.

    Args:
        maps_df: The map-level table to annotate (needs ``id_col``).
        splits_df: The splits table produced by :func:`split_matches`
            (needs ``id_col`` and ``"split"``).
        id_col: The name of the shared match id column.

    Returns:
        ``maps_df`` with an added ``split`` column, all other columns
        and the row order untouched.

    Raises:
        ValueError: If any ``maps_df[id_col]`` value is absent from
            ``splits_df[id_col]`` (guards against joining a stale or
            mismatched dataset version).
        KeyError: If ``maps_df`` lacks ``id_col`` or ``splits_df``
            lacks ``id_col``/``"split"`` (propagated from pandas).
    """
    splits_ids = set(splits_df[id_col])
    missing = sorted(
        {mid for mid in maps_df[id_col].unique() if mid not in splits_ids}
    )
    if missing:
        raise ValueError(
            f"{len(missing)} map match_id(s) are absent from splits_df, "
            f"e.g. {missing[:5]}"
        )
    return maps_df.merge(
        splits_df[[id_col, "split"]],
        on=id_col,
        how="left",
    )

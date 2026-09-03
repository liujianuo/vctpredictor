"""Shared training-set assembly for the fitted-model drivers (roadmap M20/M21).

The one place that turns the four materialised tables plus
``player_map_stats`` into the raw training design matrix and label
vector every trained model consumes, and (since M24) the one place
that assembles the leakage-safe walk-forward out-of-fold calibration
rows the temperature-scaling fit consumes. The M10 train split is
assembled via :func:`evaluation.harness.build_held_out_maps` with
``split="train"`` (the existing, reused way to build a training set —
no new join logic is written), and each training row's 13-feature
vector is built with :func:`models._shared.build_feature_vector` (the
identical feature vector both M20's ordinal logit and M21's
multinomial logit must consume for the "identical splits" comparison).

This helper is called from four drivers in this task:
``drivers/train_ordinal_logit.py`` (refactored from its previous inline
loop), ``drivers/train_multinomial_logit.py`` (new),
``drivers/diagnose_proportional_odds.py`` (new) and
``drivers/train_temperature_scaling.py`` (new, via
:func:`assemble_out_of_fold_eta_rows`) — enough repetition to justify
lifting the loop out of the first driver.

Since M36 (bootstrap prediction intervals), this module also hosts
:func:`assemble_bootstrap_design_matrix`, the match-level block-
bootstrap resampler consumed by
``drivers/evaluate_bootstrap_intervals.py``: it calls
:func:`evaluation.harness.build_held_out_maps` with ``split="train"``
once to get the base train-row table, resamples whole ``match_id``s
with replacement via a caller-supplied
``numpy.random.Generator``, and rebuilds the ``(X, y)`` pair exactly
like :func:`assemble_design_matrix` does (same
:func:`models._shared.build_feature_vector` call per row). It is
deliberately placed here rather than in a fifth duplicate join loop
inside ``evaluation/``: this module already is the one place that
turns the materialised tables into the raw training design matrix,
and it sits at the top of the dependency DAG (no module-boundary
restriction), free to depend on ``evaluation.harness``. ``drivers/`` sits at the top
of the dependency DAG (no module-boundary restriction), so this module
may freely depend on ``evaluation.harness``, ``models._shared``,
``models.ordinal_logit`` and ``utils.splits``; the same assembly cannot
live inside ``models/``, which must stay downward-only.

This module does no file I/O: the five tables are passed in
already-loaded, exactly as every other pure helper in this codebase
takes already-loaded DataFrames (matching ``evaluation.harness``'s
convention). All Parquet/JSON I/O lives in the calling drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import harness
from models import _shared, ordinal_logit
from utils import splits


def assemble_design_matrix(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the raw design matrix and label vector for one split.

    Wraps :func:`evaluation.harness.build_held_out_maps` with the
    requested ``split`` (default ``"train"`` — the M10 train slice, 209
    maps at v1 scale), then iterates the returned rows in order and, for
    each, calls :func:`models._shared.build_feature_vector` with the
    row's ``team1_id``/``team2_id``/``map_name``/``date`` plus the five
    tables, collecting ``X`` (``n x 13`` floats in
    :data:`models._shared.FEATURE_NAMES` order) and reading
    ``row.outcome_ordinal`` into ``y`` (``n,`` ints in ``{0, 1, 2, 3}``).

    Args:
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        labels_df: The materialised ``labels`` table (needs
            ``outcome_ordinal``).
        splits_df: The materialised ``splits`` table.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (required by the M16/M17 features inside
            :func:`models._shared.build_feature_vector`).
        split: The split value to assemble, ``"train"`` by default
            (``"test"`` works too, but the fitted-model drivers and the
            proportional-odds diagnostic all use the train split).

    Returns:
        A ``(X, y)`` tuple: ``X`` an ``(n, 13)`` numpy array of
        ``float`` in :data:`models._shared.FEATURE_NAMES` order, ``y``
        an ``(n,)`` numpy array of ``int`` outcome ordinals — exactly
        the input shape :func:`models.ordinal_logit.fit` /
        :func:`models.multinomial_logit.fit` expect.

    Raises:
        ValueError: If the split-restricted, label-joined result is
            empty (propagated from
            :func:`evaluation.harness.build_held_out_maps`), or if any
            feature computation fails (propagated from
            :func:`models._shared.build_feature_vector`).
        KeyError: If any input table lacks a required column
            (propagated from the harness / the feature modules).
    """
    rows = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split=split
    )
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for row in rows.itertuples(index=False):
        X_rows.append(
            _shared.build_feature_vector(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
                player_map_stats_df,
            )
        )
        y_rows.append(int(row.outcome_ordinal))
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)


def assemble_bootstrap_design_matrix(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one match-level block-bootstrap resampled design matrix.

    The M36 resample step: draws one full resampled training design
    matrix from the ``"train"`` split by resampling *whole matches*
    with replacement (a block bootstrap, per the M36 plan assumption 2
    and ``utils.splits``'s own module doctrine that the split unit is
    the match, not the map — a match's maps are sequential games in one
    series, not independent chronological draws, so plain per-map-row
    iid resampling would be wrong). The base train-row table comes from
    a single :func:`evaluation.harness.build_held_out_maps` call with
    ``split="train"`` (the existing, reused way to build a training
    set — no new join logic is written); the returned rows carry
    ``match_id``, so they can be grouped per match. ``len(unique
    match_ids)`` match ids are drawn with replacement via the
    caller-supplied ``rng``, and the resampled row table is the
    concatenation of every drawn match's *entire* row block in draw
    order (a match drawn twice contributes its full row block twice,
    contiguously — never a match split across resample slots). The
    ``(X, y)`` pair is then built exactly as
    :func:`assemble_design_matrix` does: one
    :func:`models._shared.build_feature_vector` call per resulting row,
    and ``row.outcome_ordinal`` into ``y``.

    **Why a raw ``(X, y)`` pair rather than a ``splits.parquet``-shaped
    table.** ``splits.parquet`` has one row per ``match_id`` and cannot
    represent "this match's rows appear twice" via a join; the
    resampled row set is therefore materialized directly (repeat a
    resampled match's rows verbatim, feature-vector and all) rather
    than forced through ``utils.splits``/``evaluation.harness``'s join
    machinery (M36 plan assumption 3).

    Args:
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        labels_df: The materialised ``labels`` table (needs
            ``outcome_ordinal``).
        splits_df: The materialised ``splits`` table.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (required by the M16/M17 features inside
            :func:`models._shared.build_feature_vector`).
        rng: The ``numpy.random.Generator`` whose draws this function
            consumes sequentially (one ``rng.choice`` call per
            invocation). A fixed seed therefore reproduces the
            resample, and therefore the refit model, byte-identically;
            the caller must keep this rng separate from any veto-
            sampling rng (the M36 driver uses a dedicated
            ``--bootstrap-seed`` rng for exactly this).

    Returns:
        A ``(X, y)`` tuple: ``X`` an ``(n, 13)`` numpy array of
        ``float`` in :data:`models._shared.FEATURE_NAMES` order, ``y``
        an ``(n,)`` numpy array of ``int`` outcome ordinals — the
        resampled training matrix/label vector, sized exactly like the
        ``"train"`` split it was drawn from (``n`` equals the base
        train row count: each draw has ``len(unique match_ids)`` slots
        and each slot contributes that match's full row block).

    Raises:
        ValueError: If the train split is empty (propagated from
            :func:`evaluation.harness.build_held_out_maps`), or if any
            feature computation fails (propagated from
            :func:`models._shared.build_feature_vector`).
        KeyError: If any input table lacks a required column
            (propagated from the harness / the feature modules).
    """
    rows = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="train"
    )
    groups_by_match = {
        match_id: group
        for match_id, group in rows.groupby("match_id", sort=False)
    }
    match_ids = list(groups_by_match)
    drawn = rng.choice(np.asarray(match_ids), size=len(match_ids), replace=True)
    resampled = pd.concat(
        [groups_by_match[mid] for mid in drawn], ignore_index=True
    )
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for row in resampled.itertuples(index=False):
        X_rows.append(
            _shared.build_feature_vector(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
                player_map_stats_df,
            )
        )
        y_rows.append(int(row.outcome_ordinal))
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)


def assemble_out_of_fold_eta_rows(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    n_folds: int = splits.DEFAULT_N_FOLDS,
    min_fold_block: int = splits.MIN_FOLD_BLOCK_MATCHES,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the leakage-safe walk-forward OOF eta/threshold calibration rows.

    Implements decision C of the M24 plan (the calibration-set assembly
    for the temperature-scaling fit): the final M20 artifact was fit on
    the *entire* train split, so there is no data left that is
    simultaneously "not used to fit the final model" and "not the
    held-out test split". Therefore a *temporary* ordinal-logit model is
    refit per walk-forward fold (``utils.splits.walk_forward_folds``
    over the ``splits_df`` ``"train"`` matches, library defaults
    ``DEFAULT_N_FOLDS``/``MIN_FOLD_BLOCK_MATCHES``, same
    ``models.ordinal_logit.fit`` defaults as M20, no new hyperparameter
    flags), each fold predicts on its own validation block *only with
    that fold's own fitted model*, and every OOF row collects
    ``(eta, theta1, theta2, theta3, outcome_ordinal)``.

    **Decision C's asymmetry (documented, deliberate — do not "fix").**
    Each OOF row carries **its own fold model's thresholds** — not the
    final model's — because folds are trained on different amounts of
    data and have their own legitimately different threshold
    calibration; forcing the final model's thresholds onto early,
    data-poor folds would inject noise unrelated to what ``T`` is
    supposed to correct. ``T`` is then fit to correct the *aggregate,
    cross-fold* over/under-confidence pattern, and is *applied* at
    evaluation time to the fixed final model's own ``(eta, thresholds)``
    (decision E; enforced by the staleness guard in
    :func:`drivers.evaluate._ordinal_logit_temperature_factory`).

    Per fold, a synthetic per-fold ``splits_df`` covers *every*
    ``match_id`` in ``matches_df`` (``"train"`` for the fold's train
    ids, ``"val"`` for its validation ids, ``"_unused"`` for everything
    else) — required because ``evaluation.harness.build_held_out_maps``
    -> ``utils.splits.join_split_to_maps`` raises if any map's match is
    absent from the splits table. The fold's training matrix reuses
    :func:`assemble_design_matrix` (``split="train"`` against the
    synthetic table), the fold's validation rows reuse
    ``evaluation.harness.build_held_out_maps`` (``split="val"``), and
    each validation row's ``eta`` is computed via
    ``models.ordinal_logit.apply_standardizer`` (with the *fold model's
    own* training means/stds) dotted with the *fold model's own*
    coefficients — never the final model's. All folds are submitted to
    ``utils.splits.assemble_out_of_fold_predictions``, which is the
    *required* leak check: it independently recomputes each fold's
    authoritative validation match-id set and rejects any submission
    that doesn't exactly match it, so it is not optional plumbing.

    Args:
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        labels_df: The materialised ``labels`` table (needs
            ``outcome_ordinal``).
        splits_df: The materialised ``splits`` table (needs ``split``
            with ``"train"`` rows — the walk-forward region).
        player_map_stats_df: The materialised ``player_map_stats`` table
            (required by the M16/M17 features inside
            :func:`models._shared.build_feature_vector`).
        n_folds: Passed to :func:`utils.splits.walk_forward_folds`
            (default ``DEFAULT_N_FOLDS`` = 5).
        min_fold_block: Passed to
            :func:`utils.splits.walk_forward_folds` (default
            ``MIN_FOLD_BLOCK_MATCHES`` = 8).

    Returns:
        A ``(assembled_df, coverage)`` tuple: ``assembled_df`` is the
        OOF calibration table — the concatenation of every fold's
        ``predictions_df`` in ascending ``fold_id`` order with an
        authoritative ``fold_id`` column prepended — with columns
        ``fold_id, match_id, map_index, eta, theta1, theta2, theta3,
        outcome_ordinal`` (the exact table
        :func:`models.temperature_scaling.fit_temperature` consumes,
        and the return of
        :func:`utils.splits.assemble_out_of_fold_predictions`
        unchanged). ``coverage`` is that function's coverage dict
        (``train_matches``/``covered_matches``/``warmup_excluded_ids``/
        ``warmup_excluded_count``), recorded in the calibration
        artifact.

    Raises:
        ValueError: If ``splits_df`` has no ``"train"`` rows (nothing
            to walk-forward over); if the training region is too small
            to form even one fold (propagated from
            :func:`utils.splits.walk_forward_folds`); if a fold's
            validation block is empty (propagated from
            :func:`evaluation.harness.build_held_out_maps`); if a
            submitted fold's predicted match-id set does not exactly
            match its recomputed validation set (propagated from
            :func:`utils.splits.assemble_out_of_fold_predictions` — the
            leak guard); if a label is invalid or a feature computation
            fails (propagated from
            :func:`models.ordinal_logit.fit` /
            :func:`models._shared.build_feature_vector`); or if a date
            is unparseable (propagated from
            :func:`utils.splits.walk_forward_folds`).
        KeyError: If any input table lacks a required column
            (propagated from the harness / the feature modules /
            :func:`utils.splits.walk_forward_folds`).
    """
    train_match_ids = set(
        splits_df.loc[splits_df["split"] == "train", "match_id"]
    )
    train_matches_df = matches_df[
        matches_df["match_id"].isin(train_match_ids)
    ].copy()
    if len(train_matches_df) == 0:
        raise ValueError(
            "splits_df contains no 'train' rows; cannot assemble "
            "walk-forward OOF calibration rows"
        )
    date_by_id = dict(zip(matches_df["match_id"], matches_df["date"]))

    fold_predictions: list[tuple[int, pd.DataFrame]] = []
    for fold_id, train_ids, val_ids in splits.walk_forward_folds(
        train_matches_df,
        n_folds=n_folds,
        min_fold_block=min_fold_block,
    ):
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)
        # Synthetic per-fold splits table covering *every* match_id in
        # matches_df ("_unused" for anything outside this fold's
        # train/val sets), so join_split_to_maps's absence guard never
        # fires for maps whose match is outside this fold.
        fold_split_rows = []
        for match_id in matches_df["match_id"].unique():
            if match_id in train_id_set:
                split_value = "train"
            elif match_id in val_id_set:
                split_value = "val"
            else:
                split_value = "_unused"
            fold_split_rows.append(
                {
                    "match_id": match_id,
                    "date": date_by_id[match_id],
                    "split": split_value,
                }
            )
        fold_splits_df = pd.DataFrame(
            fold_split_rows, columns=splits.SPLITS_COLUMNS
        ).astype(splits.SPLITS_DTYPES)

        X_fold, y_fold = assemble_design_matrix(
            matches_df,
            maps_df,
            labels_df,
            fold_splits_df,
            player_map_stats_df,
            split="train",
        )
        fold_model = ordinal_logit.fit(X_fold, y_fold)

        val_rows = harness.build_held_out_maps(
            matches_df,
            maps_df,
            labels_df,
            fold_splits_df,
            split="val",
        )
        pred_rows: list[dict] = []
        for row in val_rows.itertuples(index=False):
            x = _shared.build_feature_vector(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
                player_map_stats_df,
            )
            # Standardize with the *fold model's own* training
            # statistics (decision C) and dot with the fold model's own
            # coefficients — never the final model's.
            xs = ordinal_logit.apply_standardizer(
                x.reshape(1, -1),
                fold_model.standardizer_means,
                fold_model.standardizer_stds,
            )[0]
            eta = float(np.dot(xs, fold_model.coefficients))
            pred_rows.append(
                {
                    "match_id": row.match_id,
                    "map_index": row.map_index,
                    "eta": eta,
                    "theta1": float(fold_model.thresholds[0]),
                    "theta2": float(fold_model.thresholds[1]),
                    "theta3": float(fold_model.thresholds[2]),
                    "outcome_ordinal": int(row.outcome_ordinal),
                }
            )
        fold_predictions.append(
            (fold_id, pd.DataFrame(pred_rows))
        )

    # The required leak check: assemble_out_of_fold_predictions
    # independently recomputes each fold's authoritative validation
    # match-id set and rejects any submission that doesn't exactly
    # match it. Its assembled_df *is* the final OOF calibration table.
    assembled_df, coverage = splits.assemble_out_of_fold_predictions(
        train_matches_df,
        fold_predictions,
        n_folds=n_folds,
        min_fold_block=min_fold_block,
    )
    return assembled_df, coverage

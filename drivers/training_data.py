"""Shared training-set assembly for the fitted-model drivers (roadmap M20/M21).

The one place that turns the four materialised tables plus
``player_map_stats`` into the raw training design matrix and label
vector every trained model consumes. The M10 train split is assembled
via :func:`evaluation.harness.build_held_out_maps` with
``split="train"`` (the existing, reused way to build a training set —
no new join logic is written), and each training row's 11-feature
vector is built with :func:`models._shared.build_feature_vector` (the
identical feature vector both M20's ordinal logit and M21's
multinomial logit must consume for the "identical splits" comparison).

This helper is called from three drivers in this task:
``drivers/train_ordinal_logit.py`` (refactored from its previous inline
loop), ``drivers/train_multinomial_logit.py`` (new) and
``drivers/diagnose_proportional_odds.py`` (new) — enough repetition to
justify lifting the loop out of the first driver. ``drivers/`` sits at
the top of the dependency DAG (no module-boundary restriction), so this
module may freely depend on ``evaluation.harness`` and
``models._shared``; the same assembly cannot live inside ``models/``,
which must stay downward-only.

This module does no file I/O: the five tables are passed in
already-loaded, exactly as every other pure helper in this codebase
takes already-loaded DataFrames (matching ``evaluation.harness``'s
convention). All Parquet/JSON I/O lives in the calling drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import harness
from models import _shared


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
    tables, collecting ``X`` (``n x 11`` floats in
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
        A ``(X, y)`` tuple: ``X`` an ``(n, 11)`` numpy array of
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

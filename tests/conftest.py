"""Shared session-scoped pytest fixtures for the tests package.

Hosts the session-scoped ``real_v1_train_design_matrix`` fixture that
reads the five materialised ``data/v1`` parquet tables once per pytest
session, assembles the M10 held-out-train design matrix once (the
~180s ``build_feature_vector`` loop that previously ran independently
inside ``test_binary_logit.py``/``test_ordinal_logit.py``/
``test_multinomial_logit.py``), and hands the results to those files'
per-module ``real_v1_train_model`` fixtures so each just performs its
own cheap ``fit()``.

Read-only contract (roadmap standard #3): a session-scoped fixture is
shared across the whole run, so no consumer may mutate any element of
the returned tuple in place — a test that needs to alter ``X``/
``train_rows``/any DataFrame must ``.copy()`` first. Silent in-place
mutation would corrupt every other consumer in nondeterministic order.
"""

import pandas as pd
import pytest

from drivers import training_data
from evaluation import harness
from tests._shared import _real_v1_available


@pytest.fixture(scope="session")
def real_v1_train_design_matrix():
    """Assemble the real-v1 train design matrix once per pytest session.

    Reads the five materialised ``data/v1`` tables
    (``matches``/``maps``/``labels``/``splits``/``player_map_stats``),
    computes the M10 held-out-train row table via
    :func:`evaluation.harness.build_held_out_maps` (``split="train"``,
    the 209-map slice at v1 scale), and loops that table through
    :func:`drivers.training_data.assemble_design_matrix` (one
    :func:`models._shared.build_feature_vector` call per row) — the
    expensive feature assembly runs exactly once per session. The
    per-module ``real_v1_train_model`` fixtures in the three model test
    files depend on this fixture and only do their own cheap
    ``fit()``/label-derivation on top.

    Args:
        None.

    Returns:
        A ``(X, y_ordinal, train_rows, matches_df, maps_df,
        player_map_stats_df)`` tuple:
        ``X`` — the ``(n, 15)`` float numpy design matrix in
        :data:`models._shared.FEATURE_NAMES` order (``n`` is 209 at
        v1 scale); ``y_ordinal`` — the ``(n,)`` int numpy array of
        ``outcome_ordinal`` labels in ``{0, 1, 2, 3}``; ``train_rows``
        — the held-out-train ``pandas.DataFrame`` from
        :func:`evaluation.harness.build_held_out_maps`
        (``HELD_OUT_COLUMNS`` plus ``outcome_ordinal``, one row per
        train map); ``matches_df``/``maps_df``/``player_map_stats_df``
        — the three materialised tables as read from disk. The returned
        objects must be treated as read-only (standard #3): never
        mutate them in place, ``.copy()`` first if a consumer needs to
        alter any of them.

    Raises:
        pytest.skip: If the real v1 tables are absent (decorated
            behaviour — ``_real_v1_available()`` false means the
            fixture skips and every dependent fixture/test is skipped
            transitively; the fixture body itself raises nothing
            else).
    """
    if not _real_v1_available():
        pytest.skip("materialised v1 dataset not present (run materialize.py first)")
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")
    train_rows = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="train"
    )
    X, y_ordinal = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="train",
    )
    return (X, y_ordinal, train_rows, matches_df, maps_df, player_map_stats_df)

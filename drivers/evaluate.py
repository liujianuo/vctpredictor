"""Command-line evaluation of four-way outcome models (roadmap M19).

Thin command-line wrapper around :mod:`evaluation.harness`, which owns
the pure scoring/calibration logic. This module adds only the
CLI/IO glue: argument parsing (:func:`parse_args`), the model-name to
factory registry (:data:`MODEL_REGISTRY`), loading the input tables
(:func:`load_matches_table` / :func:`load_maps_table` /
:func:`load_labels_table` / :func:`load_splits_table` /
:func:`load_player_map_stats_table`), writing the
two evaluation artifacts (:func:`write_predictions_table` /
:func:`write_report`), and the :func:`main` entry point.

For the evaluation semantics — held-out split choice, the generic
model interface, metric definitions, and calibration conventions — see
:mod:`evaluation.harness`'s module docstring.

Artifacts written per run (scoped by dataset version and model name so
re-running with a different model does not clobber the previous one):

- ``data/<version>/eval_predictions_<model>.parquet`` — one row per
  held-out map: the identifying columns, the true ``outcome_ordinal``,
  the four predicted probabilities, and the per-map
  ``rps``/``log_loss``/``marginal_correct`` scores.
- ``data/<version>/eval_report_<model>.json`` — the report dict from
  :func:`evaluation.harness.build_evaluation_report`, written with
  ``json.dumps(..., indent=2, sort_keys=True)`` the same way
  ``materialize.py`` writes its ``report.json``.

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring
  ``splits.py``/``labels.py``'s raise-for-invariant-break doctrine: an
  empty held-out set, an invalid ``--model`` value (rejected by
  argparse ``choices=``), or a model whose output cannot be scored all
  propagate as exceptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import harness
from models import _shared, multinomial_logit, ordinal_logit, temperature_scaling
from utils.table_io import DEFAULT_OUTPUT_DIR, write_parquet

logger = logging.getLogger(__name__)

# A model factory: given the dataset location, return the ready-to-call
# model function. Stateless models (like the M18 baseline) ignore the
# arguments and return a constant callable; stateful fitted models (like
# M20's ordinal logit) load their artifact and any extra tables from
# ``<output_dir>/<version>`` and return a closure over them. M20+ models
# add an entry here rather than changing the harness itself.
ModelFactory = Callable[[Path, str], harness.ModelFn]


def _four_way_baseline_factory(output_dir: Path, version: str) -> harness.ModelFn:
    """Return the M18 four-way baseline as a model function.

    The trivial stateless factory: it ignores both the dataset location
    arguments (the baseline needs no fitted artifact and no extra
    tables) and returns :func:`evaluation.harness.four_way_baseline_model`
    unchanged, which is already a valid model function.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (ignored by this factory).
        version: The dataset version subdirectory name (ignored by this
            factory).

    Returns:
        :func:`evaluation.harness.four_way_baseline_model` itself.

    Raises:
        Nothing.
    """
    return harness.four_way_baseline_model


def _ordinal_logit_factory(output_dir: Path, version: str) -> harness.ModelFn:
    """Load the fitted ordinal-logit artifact and return its model function.

    The stateful factory for M20's ordinal logistic regression: it reads
    ``<output_dir>/<version>/ordinal_logit_model.json`` (produced by
    ``drivers/train_ordinal_logit.py``), parses it via
    :func:`models.ordinal_logit.from_dict`, loads the
    ``player_map_stats`` table for the same version (the seventh table
    the generic model interface does not pass; see
    :func:`models.ordinal_logit.make_model_fn`), and returns the closure
    :func:`models.ordinal_logit.make_model_fn` builds — which must be
    invoked with the same ``matches_df``/``maps_df`` from this same
    ``<output_dir>/<version>``.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A model function (the 6-argument generic shape) that predicts
        with the fitted ordinal-logit model and the loaded
        ``player_map_stats`` table.

    Raises:
        FileNotFoundError: If ``ordinal_logit_model.json`` or
            ``player_map_stats.parquet`` does not exist for this
            version (i.e. ``train_ordinal_logit.py``/``materialize.py``
            have not been run) — propagated as-is from
            ``json.load``/``pandas.read_parquet``.
        ValueError / KeyError: If the artifact dict is malformed or
            shape-inconsistent (propagated from
            :func:`models.ordinal_logit.from_dict`).
    """
    artifact_path = Path(output_dir) / version / "ordinal_logit_model.json"
    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    model = ordinal_logit.from_dict(artifact)
    player_map_stats_df = load_player_map_stats_table(output_dir, version)
    return ordinal_logit.make_model_fn(model, player_map_stats_df)


def _multinomial_logit_factory(output_dir: Path, version: str) -> harness.ModelFn:
    """Load the fitted multinomial-logit artifact and return its model function.

    The stateful factory for M21's multinomial logistic regression: it
    reads ``<output_dir>/<version>/multinomial_logit_model.json``
    (produced by ``drivers/train_multinomial_logit.py``), parses it via
    :func:`models.multinomial_logit.from_dict`, loads the
    ``player_map_stats`` table for the same version (the seventh table
    the generic model interface does not pass; see
    :func:`models.multinomial_logit.make_model_fn`), and returns the
    closure :func:`models.multinomial_logit.make_model_fn` builds —
    which must be invoked with the same ``matches_df``/``maps_df`` from
    this same ``<output_dir>/<version>``. Mirrors
    :func:`_ordinal_logit_factory` exactly.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A model function (the 6-argument generic shape) that predicts
        with the fitted multinomial-logit model and the loaded
        ``player_map_stats`` table.

    Raises:
        FileNotFoundError: If ``multinomial_logit_model.json`` or
            ``player_map_stats.parquet`` does not exist for this
            version (i.e. ``train_multinomial_logit.py``/
            ``materialize.py`` have not been run) — propagated as-is
            from ``json.load``/``pandas.read_parquet``.
        ValueError / KeyError: If the artifact dict is malformed or
            shape-inconsistent (propagated from
            :func:`models.multinomial_logit.from_dict`).
    """
    artifact_path = Path(output_dir) / version / "multinomial_logit_model.json"
    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    model = multinomial_logit.from_dict(artifact)
    player_map_stats_df = load_player_map_stats_table(output_dir, version)
    return multinomial_logit.make_model_fn(model, player_map_stats_df)


def _ordinal_logit_temperature_factory(output_dir: Path, version: str) -> harness.ModelFn:
    """Load the calibrated (temperature-scaled) M20 artifact pair and wrap them.

    The stateful factory for M24's temperature-scaled ordinal logit: it
    loads the base M20 artifact
    (``<output_dir>/<version>/ordinal_logit_model.json``, exactly like
    :func:`_ordinal_logit_factory`) and the calibration artifact
    (``<output_dir>/<version>/temperature_scaling_model.json``, produced
    by ``drivers/train_temperature_scaling.py``), enforces the
    decision-E staleness guard, loads the ``player_map_stats`` table,
    and returns a closure that computes the raw 11-feature vector
    (:func:`models._shared.build_feature_vector`), standardizes it with
    the *base* model's stored training means/stds
    (:func:`models.ordinal_logit.apply_standardizer`), computes
    ``eta = dot(xs, base_model.coefficients)``, and returns
    :func:`models.temperature_scaling.predict_proba_with_temperature`
    applied to ``(eta, base_model.thresholds, temp_model.temperature)``
    — i.e. the fixed final model's own ``(eta, thresholds)`` with the
    single fitted ``T`` (decision E: ``eta``/``thresholds`` always come
    from the one fixed, fully-trained M20 artifact, never from any fold
    model).

    **Staleness guard (decision E, enforced here — do not weaken).**
    The calibration artifact's ``thresholds`` field is a *provenance
    copy* of the base model's thresholds at calibration-fit time. If the
    loaded base model's thresholds do not ``np.allclose``-match that
    stored copy, the factory raises ``ValueError`` rather than silently
    applying a stale ``T`` (e.g. after ``train_ordinal_logit.py`` was
    re-run without re-running ``train_temperature_scaling.py``).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A model function (the 6-argument generic shape) that predicts
        the four temperature-scaled category probabilities, closing
        over the base model, the calibration model and the loaded
        ``player_map_stats`` table.

    Raises:
        FileNotFoundError: If ``ordinal_logit_model.json``,
            ``temperature_scaling_model.json`` or
            ``player_map_stats.parquet`` does not exist for this
            version (i.e. the two training drivers or
            ``materialize.py`` have not been run) — propagated as-is
            from ``json.load``/``pandas.read_parquet``.
        ValueError: If the calibration artifact's stored thresholds do
            not match the loaded base model's thresholds (the staleness
            guard), or if either artifact dict is malformed/
            shape-inconsistent (propagated from the two ``from_dict``
            functions).
        KeyError: If either artifact dict lacks a required key
            (propagated from the two ``from_dict`` functions).
    """
    base_artifact_path = Path(output_dir) / version / "ordinal_logit_model.json"
    with open(base_artifact_path, encoding="utf-8") as handle:
        base_model = ordinal_logit.from_dict(json.load(handle))
    temp_artifact_path = (
        Path(output_dir) / version / "temperature_scaling_model.json"
    )
    with open(temp_artifact_path, encoding="utf-8") as handle:
        temp_model = temperature_scaling.from_dict(json.load(handle))
    if not np.allclose(temp_model.thresholds, base_model.thresholds):
        raise ValueError(
            "temperature_scaling_model.json was calibrated against a "
            "different ordinal_logit_model.json; re-run "
            "train_temperature_scaling.py"
        )
    player_map_stats_df = load_player_map_stats_table(output_dir, version)

    def model_fn(
        team1_id: str,
        team2_id: str,
        map_name: str,
        date: str,
        matches_df: pd.DataFrame,
        maps_df: pd.DataFrame,
    ) -> tuple[float, float, float, float]:
        """Predict the four temperature-scaled category probabilities.

        Computes the raw 11-feature vector via
        :func:`models._shared.build_feature_vector` (using the
        closed-over ``player_map_stats_df`` — the table the generic
        interface does not pass), standardizes it with the base model's
        stored training-population statistics, computes
        ``eta = dot(xs, base_model.coefficients)``, and returns
        :func:`models.temperature_scaling.predict_proba_with_temperature`
        applied to ``(eta, base_model.thresholds,
        temp_model.temperature)``. See
        :func:`_ordinal_logit_temperature_factory`'s docstring for the
        decision-E contract (fixed final model's ``(eta, thresholds)``,
        single fitted ``T``).

        Args:
            team1_id: The queried team1's stable id ("A").
            team2_id: The queried team2's stable id ("B").
            map_name: The map to predict for.
            date: The as-of cutoff (the map's own match date).
            matches_df: The full materialised ``matches`` table from the
                same dataset version the closed-over
                ``player_map_stats_df`` was loaded from.
            maps_df: The full materialised ``maps`` table from the same
                version.

        Returns:
            The 4-tuple of probabilities in
            :data:`models._shared.OUTCOME_LABELS` order, summing to
            approximately 1 (each clipped into ``[eps, 1-eps]``).

        Raises:
            ValueError: If the feature vector length mismatches the
                base model, if ``temp_model.temperature`` is not
                positive, or if any feature computation fails
                (propagated from :func:`models._shared.build_feature_vector`
                / :func:`models.ordinal_logit.predict_proba`'s shape
                checks / :func:`models.temperature_scaling.predict_proba_with_temperature`).
            KeyError: If any table lacks a required column (propagated
                from :func:`models._shared.build_feature_vector`).
        """
        x = _shared.build_feature_vector(
            team1_id,
            team2_id,
            map_name,
            date,
            matches_df,
            maps_df,
            player_map_stats_df,
        )
        xs = ordinal_logit.apply_standardizer(
            x.reshape(1, -1),
            base_model.standardizer_means,
            base_model.standardizer_stds,
        )[0]
        eta = float(np.dot(xs, base_model.coefficients))
        return temperature_scaling.predict_proba_with_temperature(
            eta, base_model.thresholds, temp_model.temperature
        )

    return model_fn


# The registry of runnable model names -> factories. Each value is a
# :data:`ModelFactory` callable ``(output_dir, version) -> ModelFn``;
# :func:`main` invokes the selected factory with the dataset location
# to obtain the model function, then scores it. ``--model``'s
# ``choices=sorted(MODEL_REGISTRY)`` picks up new keys automatically.
MODEL_REGISTRY: dict[str, ModelFactory] = {
    "four_way_baseline": _four_way_baseline_factory,
    "ordinal_logit": _ordinal_logit_factory,
    "ordinal_logit_temperature": _ordinal_logit_temperature_factory,
    "multinomial_logit": _multinomial_logit_factory,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with three attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and
        ``model`` (``str``, the registered model name to evaluate,
        default ``"four_way_baseline"``). Together they locate the
        input tables under ``<output_dir>/<version>/*.parquet``
        (``matches``/``maps``/``labels``/``splits`` always, plus
        ``player_map_stats`` for the fitted ``ordinal_logit``/
        ``multinomial_logit`` models)
        and the two output artifacts
        ``eval_predictions_<model>.parquet`` /
        ``eval_report_<model>.json`` under the same directory. There is
        deliberately no ``--k`` flag: the shrinkage strength is a
        per-model concern a caller tunes by registering their own
        partially-applied model function (see
        :func:`evaluation.harness.four_way_baseline_model`), not a CLI
        concern.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or an unknown ``--model`` value, which
            is rejected by the ``choices=`` constraint rather than
            silently falling back).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score a registered four-way outcome model against the "
            "held-out test split and write eval_predictions_<model>.parquet "
            "plus eval_report_<model>.json."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="input/output subdirectory name under --output-dir (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory lives under "
            "(default: data)"
        ),
    )
    parser.add_argument(
        "--model",
        default="four_way_baseline",
        choices=sorted(MODEL_REGISTRY),
        help=(
            "registered model name to evaluate (default: "
            "four_way_baseline; choices: "
            f"{', '.join(sorted(MODEL_REGISTRY))})"
        ),
    )
    return parser.parse_args(argv)


def load_matches_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised matches table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function so tests can exercise the pure harness parts
    against in-memory DataFrames and stub/bypass disk entirely.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/matches.parquet`` as a
        ``pandas.DataFrame`` (M8's ``matches`` table).

    Raises:
        FileNotFoundError: If ``matches.parquet`` does not exist for
            this version (i.e. ``materialize.py`` has not been run for
            it) — propagated as-is from ``pandas.read_parquet`` as a
            clear "run materialize.py first" signal rather than
            wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "matches.parquet")


def load_maps_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised maps table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function (see :func:`load_matches_table`).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/maps.parquet`` as a
        ``pandas.DataFrame`` (M8's ``maps`` table).

    Raises:
        FileNotFoundError: If ``maps.parquet`` does not exist for this
            version — propagated as-is from ``pandas.read_parquet``.
        OSError: On any other file-access failure, also propagated
            as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "maps.parquet")


def load_labels_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised labels table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function (see :func:`load_matches_table`).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/labels.parquet`` as a
        ``pandas.DataFrame`` (M9's ``labels`` table).

    Raises:
        FileNotFoundError: If ``labels.parquet`` does not exist for
            this version — propagated as-is from ``pandas.read_parquet``.
        OSError: On any other file-access failure, also propagated
            as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "labels.parquet")


def load_splits_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised splits table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function (see :func:`load_matches_table`).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/splits.parquet`` as a
        ``pandas.DataFrame`` (M10's ``splits`` table).

    Raises:
        FileNotFoundError: If ``splits.parquet`` does not exist for
            this version — propagated as-is from ``pandas.read_parquet``.
        OSError: On any other file-access failure, also propagated
            as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "splits.parquet")


def load_veto_actions_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised veto_actions table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function (see :func:`load_matches_table`).

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/veto_actions.parquet``
        as a ``pandas.DataFrame`` (M4's ``veto_actions`` table).

    Raises:
        FileNotFoundError: If ``veto_actions.parquet`` does not exist
            for this version (i.e. ``materialize.py`` has not been run
            for it) — propagated as-is from ``pandas.read_parquet``.
        OSError: On any other file-access failure, also propagated
            as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "veto_actions.parquet")


def load_player_map_stats_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised player_map_stats table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function (see :func:`load_matches_table`). This is the
    fifth input table: the fitted ``ordinal_logit``/``multinomial_logit``
    models need it at load time (their feature vectors consume M16/M17
    features that read player rows), while the stateless baseline and
    the harness itself do not — so only the two fitted-model factories
    call this helper.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/player_map_stats.parquet``
        as a ``pandas.DataFrame`` (M8's ``player_map_stats`` table).

    Raises:
        FileNotFoundError: If ``player_map_stats.parquet`` does not
            exist for this version (i.e. ``materialize.py`` has not
            been run for it) — propagated as-is from
            ``pandas.read_parquet``.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "player_map_stats.parquet")


def write_predictions_table(
    scored_df: pd.DataFrame,
    output_dir: Path,
    version: str,
    model_name: str,
) -> None:
    """Write the per-map scored predictions artifact to disk.

    Writes ``<output_dir>/<version>/eval_predictions_<model_name>.parquet``
    via :func:`table_io.write_parquet` (``index=False``), creating the
    version directory (including parents) if it does not already exist.
    Overwrites any previous file of the same model name in place —
    re-evaluating the same version+model replaces the artifact rather
    than erroring, matching the idempotent re-run story tasks 008/009
    established, while a different model name never clobbers an
    existing model's artifact.

    Args:
        scored_df: The scored table from
            :func:`evaluation.harness.score_held_out_maps` to write.
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).
        model_name: The registered model name (the ``--model`` value)
            to scope the artifact filename by.

    Returns:
        None.

    Raises:
        OSError: If the directory cannot be created or the file cannot
            be written (e.g. permissions or disk errors).
        ValueError: If the table contains a value that cannot be
            serialized to Parquet (propagated from
            :func:`table_io.write_parquet` / ``DataFrame.to_parquet``).
    """
    write_parquet(
        scored_df,
        Path(output_dir) / version / f"eval_predictions_{model_name}.parquet",
    )


def write_report(
    report: dict,
    output_dir: Path,
    version: str,
    model_name: str,
) -> None:
    """Write the evaluation report artifact to disk as JSON.

    Writes ``<output_dir>/<version>/eval_report_<model_name>.json`` via
    ``json.dumps(report, indent=2, sort_keys=True)`` (plus a trailing
    newline), creating the version directory (including parents) if it
    does not already exist — the same serialization
    ``materialize.py``'s ``write_materialised_tables`` uses for
    ``report.json``. Overwrites any previous file of the same model
    name in place (see :func:`write_predictions_table`).

    Args:
        report: The JSON-serializable report dict from
            :func:`evaluation.harness.build_evaluation_report`.
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).
        model_name: The registered model name (the ``--model`` value)
            to scope the artifact filename by.

    Returns:
        None.

    Raises:
        TypeError: If ``report`` contains a value that is not
            JSON-serializable (propagated from ``json.dumps``; the
            harness guarantees it is not, but a hand-built dict could
            be).
        OSError: If the directory cannot be created or the file cannot
            be written (e.g. permissions or disk errors).
    """
    path = Path(output_dir) / version / f"eval_report_{model_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The evaluation tables are loaded for the requested version
    (``matches``/``maps``/``labels``/``splits`` via the ``load_*``
    helpers; ``player_map_stats`` is loaded lazily inside the
    ``ordinal_logit``/``multinomial_logit`` factories), the held-out
    map set is assembled
    (:func:`evaluation.harness.build_held_out_maps`), the registered
    model's factory is invoked with the dataset location to obtain the
    model function, which is scored over the held-out set
    (:func:`evaluation.harness.score_held_out_maps`), the report is
    built (:func:`evaluation.harness.build_evaluation_report`), both
    artifacts are written (see :func:`write_predictions_table` /
    :func:`write_report`), and a one-line summary of the headline
    numbers is logged (mirroring ``materialize.py``/``labels.py``'s
    summary-line convention).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures (empty held-out set, unscorable model output) are
            raises that propagate to the caller, matching
            ``splits.py``/``labels.py``'s raise-for-invariant-break
            doctrine rather than a second success/failure channel.

    Raises:
        FileNotFoundError: If any of the evaluation tables (or the
            ``player_map_stats`` table, for the ``ordinal_logit``/
            ``multinomial_logit`` models) does not exist for the requested version
            (propagated from the ``load_*`` helpers).
        SystemExit: If ``--model`` is not a registered name (rejected
            by argparse ``choices=`` in :func:`parse_args`).
        ValueError: If the held-out set is empty, a model output is
            not a valid 4-vector, or a metric cannot be computed
            (propagated from the harness's pure functions).
        OSError / TypeError / ValueError: If an output artifact cannot
            be written (propagated from the ``write_*`` helpers).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    matches_df = load_matches_table(output_dir, args.version)
    maps_df = load_maps_table(output_dir, args.version)
    labels_df = load_labels_table(output_dir, args.version)
    splits_df = load_splits_table(output_dir, args.version)

    model_fn = MODEL_REGISTRY[args.model](output_dir, args.version)
    held_out_df = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df
    )
    scored_df = harness.score_held_out_maps(
        model_fn, held_out_df, matches_df, maps_df
    )
    report = harness.build_evaluation_report(scored_df)

    write_predictions_table(scored_df, output_dir, args.version, args.model)
    write_report(report, output_dir, args.version, args.model)

    logger.info(
        "evaluated model %r on %d held-out maps (%s/%s): "
        "mean_rps=%.6f mean_log_loss=%.6f marginal_binary_accuracy=%.6f "
        "most_miscalibrated_category=%s",
        args.model,
        report["n_eval"],
        output_dir,
        args.version,
        report["mean_rps"],
        report["mean_log_loss"],
        report["marginal_binary_accuracy"],
        report["most_miscalibrated_category"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

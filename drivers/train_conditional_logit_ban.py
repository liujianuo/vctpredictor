"""Command-line training of the conditional-logit ban model (roadmap M27).

Thin command-line wrapper around :mod:`models.conditional_logit_ban`,
which owns the pure model logic (feature vector, standardizer, fit,
serialization). This module adds only the CLI/IO glue: argument parsing
(:func:`parse_args`), loading the four input tables (reusing the
``load_*_table`` helpers from :mod:`drivers.evaluate`), building the
train-split held-out veto-step table
(:func:`evaluation.veto_evaluation.build_held_out_veto_matches` with
``split="train"``), building the per-step ban training examples
(:func:`evaluation.veto_evaluation.build_ban_training_examples`),
featurizing each example's candidate maps via
:func:`models.conditional_logit_ban.build_ban_feature_vector` and
flattening them into the ragged-group design matrix
``models.conditional_logit_ban.fit`` consumes (one group per ban step,
``group_boundaries`` slicing each step's candidate rows out of the
flat matrix, ``y_true_row_index`` holding each step's true-map row),
calling :func:`models.conditional_logit_ban.fit`, and writing the
serialized model artifact
``data/<version>/conditional_logit_ban_model.json``.

The train-split assembly needs ``evaluation/veto_evaluation.py``'s
shared teacher-forced replay (era-pool resolution, the real-map-set
equals-pool guard, ``remaining``-set bookkeeping), which ``drivers/``
modules are allowed to use (drivers sit at the top of the dependency
DAG); the same assembly cannot live inside
``models/conditional_logit_ban.py``, which must stay downward-only
(see that module's docstring).

Artifact written per run (scoped by dataset version so re-running with
a different version does not clobber the previous one):

- ``data/<version>/conditional_logit_ban_model.json`` — the
  :func:`models.conditional_logit_ban.to_dict` dict, written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
  newline (the same serialization convention as every other model
  artifact in this repo).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine: a missing
  input table, an empty train split, a held-out table with no ban
  rows, or a feature computation failure all propagate as exceptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from drivers import evaluate
from evaluation import veto_evaluation
from models import conditional_logit_ban
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the train_conditional_logit_ban.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with two attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``) and ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``). Together
        they locate the four input tables under
        ``<output_dir>/<version>/*.parquet`` and the output artifact
        ``<output_dir>/<version>/conditional_logit_ban_model.json``.
        There are deliberately no hyperparameter flags: ``l2_lambda``/
        ``max_iter``/``grad_tol``/``loss_tol`` stay as
        :func:`models.conditional_logit_ban.fit`'s documented
        defaults, matching ``drivers/train_ordinal_logit.py``'s "no
        hyperparameter flags" precedent (tuning them is future-
        milestone scope, not a CLI concern).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the conditional-logit ban model over the five "
            "map-varying veto features on the M10 train split's "
            "teacher-forced ban steps and write "
            "conditional_logit_ban_model.json."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train the conditional-logit ban model end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The four input tables are loaded for the requested version
    (matches/maps/splits via the ``drivers.evaluate`` helpers, plus
    ``veto_actions`` via
    :func:`drivers.evaluate.load_veto_actions_table`), the train-split
    held-out veto-step table is built
    (:func:`evaluation.veto_evaluation.build_held_out_veto_matches`
    with ``split="train"``), the ban training examples are built from
    it (:func:`evaluation.veto_evaluation.build_ban_training_examples`
    — the shared teacher-forced replay, ban rows only), each example's
    candidate maps are featurized
    (:func:`models.conditional_logit_ban.build_ban_feature_vector`)
    and flattened into the ragged-group design matrix
    (``group_boundaries`` = the per-step row offsets, ending at the
    flat row count; ``y_true_row_index`` = each step's within-group
    true-map row, taken from the example's ``true_map_index``), the
    model is fit (:func:`models.conditional_logit_ban.fit` — which
    fits the per-feature standardizer on this flattened training
    matrix only and then runs Armijo gradient descent, so the returned
    artifact carries the training-population means/stds
    :func:`models.conditional_logit_ban.predict_ban_distribution`
    needs), the artifact is written as
    ``<output_dir>/<version>/conditional_logit_ban_model.json``, and a
    one-line summary of the fit diagnostics is logged (``n_train`` ban
    steps, ``converged``, ``n_iter``, ``final_loss`` — the numbers the
    BUILD status line records).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures (missing input table, empty train split, a
            held-out table with no ban rows, a feature computation
            error) are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the four input tables does not
            exist for the requested version (i.e. ``materialize.py`` /
            ``splits.py`` have not been run for it) — propagated as-is
            from the ``load_*`` helpers.
        ValueError: If the train split's held-out veto set is empty,
            contains no ban rows, has a match whose real normalized
            map-name set mismatches its era pool, has a ban row whose
            acting ``team_id`` is neither of its match's
            ``team1_id``/``team2_id``, or if a feature computation or
            the fit fails its shape/validation guards (propagated from
            :mod:`evaluation.veto_evaluation` /
            :mod:`models.conditional_logit_ban`).
        KeyError: If any input table lacks a required column
            (propagated from the pure functions).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    veto_df = evaluate.load_veto_actions_table(output_dir, args.version)

    held_out = veto_evaluation.build_held_out_veto_matches(
        veto_df, matches_df, splits_df, split="train"
    )
    examples = veto_evaluation.build_ban_training_examples(
        held_out, matches_df, maps_df
    )

    # Flatten every training ban step's candidate-map feature rows into
    # the one ragged-group design matrix (decision 5): rows in replay
    # order, group s covering the example's len(remaining_maps) rows,
    # y_true_row_index[s] = the true banned map's within-group row.
    flat_rows: list[np.ndarray] = []
    group_boundaries = [0]
    y_true_row_index: list[int] = []
    for example in examples:
        for map_name in example.remaining_maps:
            flat_rows.append(
                conditional_logit_ban.build_ban_feature_vector(
                    example.acting_team_id,
                    example.opponent_team_id,
                    map_name,
                    example.date,
                    matches_df,
                    maps_df,
                )
            )
        group_boundaries.append(group_boundaries[-1] + len(example.remaining_maps))
        y_true_row_index.append(example.true_map_index)
    X_flat = np.asarray(flat_rows, dtype=float)

    model = conditional_logit_ban.fit(
        X_flat,
        np.asarray(group_boundaries, dtype=int),
        np.asarray(y_true_row_index, dtype=int),
    )

    artifact_path = output_dir / args.version / "conditional_logit_ban_model.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(conditional_logit_ban.to_dict(model), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "trained conditional logit ban model on %d train-split ban steps "
        "(%s/%s): converged=%s n_iter=%d final_loss=%.6f",
        model.n_train,
        output_dir,
        args.version,
        model.converged,
        model.n_iter,
        model.final_loss,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

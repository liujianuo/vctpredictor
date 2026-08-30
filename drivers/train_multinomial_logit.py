"""Command-line training of the multinomial logistic regression model (roadmap M21).

Thin command-line wrapper around :mod:`models.multinomial_logit`, which
owns the pure model logic (softmax link, fit, serialization). This
module adds only the CLI/IO glue, mirroring
``drivers/train_ordinal_logit.py``'s shape exactly: argument parsing
(:func:`parse_args`), loading the five input tables (reusing the four
``load_*_table`` helpers from :mod:`drivers.evaluate` plus that module's
:func:`drivers.evaluate.load_player_map_stats_table`), assembling the
training design matrix via :func:`drivers.training_data.assemble_design_matrix`
(the same shared helper ``train_ordinal_logit.py`` uses, so the M21 arm
is trained on the *identical* feature vector and the *identical* M10
train split as M20 — the entire point of the comparison), calling
:func:`models.multinomial_logit.fit`, and writing the serialized model
artifact ``data/<version>/multinomial_logit_model.json``.

Artifact written per run (scoped by dataset version so re-running with a
different version does not clobber the previous one):

- ``data/<version>/multinomial_logit_model.json`` — the
  :func:`models.multinomial_logit.to_dict` dict, written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing newline
  (the same serialization convention as ``drivers/evaluate.py``'s
  ``write_report``).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine: a missing
  input table, an empty train split, or a feature computation failure
  all propagate as exceptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from drivers import evaluate, training_data
from models import multinomial_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the train_multinomial_logit.py command line.

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
        they locate the five input tables under
        ``<output_dir>/<version>/*.parquet`` and the output artifact
        ``<output_dir>/<version>/multinomial_logit_model.json``. There
        are deliberately no hyperparameter flags: ``l2_lambda``/
        ``max_iter``/``grad_tol``/``loss_tol`` stay as
        :func:`models.multinomial_logit.fit`'s documented defaults
        (kept identical to the ordinal arm's for comparability),
        matching ``drivers/train_ordinal_logit.py``'s no-flags
        precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the multinomial logistic regression model over the "
            "M13-M17 feature vector on the M10 train split and write "
            "multinomial_logit_model.json."
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
    """Train the multinomial logistic regression model end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table`), the training
    map set is assembled
    (:func:`drivers.training_data.assemble_design_matrix` with
    ``split="train"`` — the M10 train slice, 209 maps at v1 scale, the
    identical matrix the ordinal arm is trained on), the model is fit
    (:func:`models.multinomial_logit.fit` — which fits the per-feature
    standardizer on this training matrix only and then runs gradient
    descent, so the returned artifact carries the training-population
    means/stds :func:`models.multinomial_logit.predict_proba` needs),
    the artifact is written as
    ``<output_dir>/<version>/multinomial_logit_model.json``, and a
    one-line summary of the fit diagnostics is logged (mirroring
    ``drivers/train_ordinal_logit.py``'s summary-line convention).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures (missing input table, empty train split, a feature
            computation error) are raises that propagate to the caller,
            matching the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the five input tables does not
            exist for the requested version (propagated from the
            ``load_*`` helpers).
        ValueError: If the train split is empty, a label is invalid, or
            a feature computation fails (propagated from
            :func:`drivers.training_data.assemble_design_matrix` /
            :func:`models.multinomial_logit.fit`).
        KeyError: If any input table lacks a required column (propagated
            from the feature modules / harness).
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
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    X, y = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="train",
    )

    model = multinomial_logit.fit(X, y)

    artifact_path = (
        output_dir / args.version / "multinomial_logit_model.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(multinomial_logit.to_dict(model), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "trained multinomial logit on %d maps (%s/%s): converged=%s "
        "n_iter=%d final_loss=%.6f",
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

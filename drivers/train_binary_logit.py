"""Command-line training of the plain binary logistic regression model (roadmap M22).

Thin command-line wrapper around :mod:`models.binary_logit`, which owns
the pure model logic (feature vector, standardizer, fit,
serialization). This module adds only the CLI/IO glue: argument parsing
(:func:`parse_args`), loading the five input tables (reusing the four
``load_*_table`` helpers from :mod:`drivers.evaluate` plus that
module's :func:`drivers.evaluate.load_player_map_stats_table`),
assembling the training design matrix via
:func:`drivers.training_data.assemble_design_matrix` (the shared
helper every fitted-model driver in this milestone uses — the identical
11-feature matrix M20's ordinal logit and M21's multinomial logit were
trained on, so the M22 granularity comparison is on identical splits),
converting the ordinal labels to the binary "A wins" target, calling
:func:`models.binary_logit.fit`, and writing the serialized model
artifact ``data/<version>/binary_logit_model.json``.

**The ordinal-to-binary conversion happens here, one line, inline:**
``y_binary = (y_ordinal <= 1).astype(int)`` — "A wins" is ordinals 0
(A-regulation) and 1 (A-OT), matching :mod:`models.binary_logit`'s
documented target convention exactly. It is a single boolean
comparison, deliberately not extracted into a shared helper
(:func:`models.binary_logit.fit`'s contract is a generic binary-label
fit with no notion of "outcome ordinal", so the conversion must live at
this driver layer).

Artifact written per run (scoped by dataset version so re-running with
a different version does not clobber the previous one):

- ``data/<version>/binary_logit_model.json`` — the
  :func:`models.binary_logit.to_dict` dict, written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing newline
  (the same serialization convention as every other artifact writer in
  this repo).

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
from models import binary_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the train_binary_logit.py command line.

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
        ``<output_dir>/<version>/binary_logit_model.json``. There are
        deliberately no hyperparameter flags: ``l2_lambda``/``max_iter``/
        ``grad_tol``/``loss_tol`` stay as
        :func:`models.binary_logit.fit`'s documented defaults, matching
        the other training drivers' no-flags precedent (tuning them is
        future-milestone scope, not a CLI concern).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the plain binary logistic regression over the "
            "M13-M17 feature vector on the M10 train split (target: "
            "'A wins' = outcome_ordinal in {0, 1}) and write "
            "binary_logit_model.json."
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
    """Train the binary logistic regression model end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table`), the training
    map set is assembled
    (:func:`drivers.training_data.assemble_design_matrix` with
    ``split="train"`` — the M10 train slice, 209 maps at v1 scale, the
    identical matrix both M20's ordinal logit and M21's multinomial
    logit were trained on), the ordinal labels are binarized to the
    "A wins" target (``y_binary = (y_ordinal <= 1).astype(int)``), the
    model is fit (:func:`models.binary_logit.fit` — which fits the
    per-feature standardizer on this training matrix only and then runs
    gradient descent, so the returned artifact carries the
    training-population means/stds
    :func:`models.binary_logit.predict_proba` needs), the artifact is
    written as ``<output_dir>/<version>/binary_logit_model.json``, and
    a one-line summary of the fit diagnostics is logged (mirroring the
    other training drivers' summary-line convention).

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
            :func:`models.binary_logit.fit`).
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

    X, y_ordinal = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="train",
    )

    # The one-line ordinal-to-binary conversion: "A wins" = ordinals 0
    # (A-regulation) and 1 (A-OT), matching
    # models.binary_logit's documented target convention exactly. Done
    # here, inline, not as a new shared helper — it is a single boolean
    # comparison.
    y_binary = (y_ordinal <= 1).astype(int)

    model = binary_logit.fit(X, y_binary)

    artifact_path = output_dir / args.version / "binary_logit_model.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(binary_logit.to_dict(model), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "trained binary logit on %d maps (%s/%s): converged=%s "
        "n_iter=%d final_loss=%.6f n_positive=%d",
        model.n_train,
        output_dir,
        args.version,
        model.converged,
        model.n_iter,
        model.final_loss,
        int(y_binary.sum()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line proportional-odds diagnostic for the ordinal model (roadmap M21).

Thin command-line wrapper around :mod:`evaluation.proportional_odds`,
which owns the pure diagnostic logic (per-cutpoint binary fits, sign
instability, AIC/BIC, verdict). This module adds only the CLI/IO glue:
argument parsing (:func:`parse_args`), loading the five input tables
(reusing the four ``load_*_table`` helpers from :mod:`drivers.evaluate`
plus that module's :func:`drivers.evaluate.load_player_map_stats_table`),
assembling the training design matrix via
:func:`drivers.training_data.assemble_design_matrix` (the identical
matrix both four-class arms were trained on), loading the two fitted
model artifacts (``ordinal_logit_model.json`` and
``multinomial_logit_model.json``), running
:func:`evaluation.proportional_odds.fit_cutpoint_binary_models` then
:func:`evaluation.proportional_odds.build_diagnostic_report`, and
writing the report artifact
``data/<version>/proportional_odds_diagnostic.json``.

**Prerequisite:** both training drivers must already have been run for
the requested version. This script does not orchestrate training itself
(matching the existing one-script-one-job convention — ``train_ordinal_logit.py``
vs ``evaluate.py`` are already separate scripts for the same reason): if
either model artifact is missing, the ``FileNotFoundError`` propagates
unchanged as a clear "run the training drivers first" signal.

Artifact written per run (scoped by dataset version):

- ``data/<version>/proportional_odds_diagnostic.json`` — the
  :func:`evaluation.proportional_odds.build_diagnostic_report` dict,
  written with ``json.dumps(..., indent=2, sort_keys=True)`` plus a
  trailing newline (the same serialization convention as every other
  artifact in this repo).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from drivers import evaluate, training_data
from evaluation import proportional_odds
from models import multinomial_logit, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the diagnose_proportional_odds.py command line.

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
        they locate the five input tables, the two model artifacts
        (``ordinal_logit_model.json`` / ``multinomial_logit_model.json``)
        and the output artifact
        ``<output_dir>/<version>/proportional_odds_diagnostic.json``.
        There are deliberately no other flags: the diagnostic uses the
        documented defaults throughout (``l2_lambda=1.0`` etc.),
        matching the training drivers' no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the proportional-odds (Brant-approximation) diagnostic "
            "comparing the fitted ordinal and multinomial models on the "
            "M10 train split, and write proportional_odds_diagnostic.json."
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
    """Run the proportional-odds diagnostic end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table`), the training
    design matrix is assembled
    (:func:`drivers.training_data.assemble_design_matrix` with
    ``split="train"`` — the identical matrix both fitted arms were
    trained on), both fitted model artifacts are loaded via ``json.load``
    + the two ``from_dict`` functions, the three per-cutpoint binary
    models are fit (:func:`evaluation.proportional_odds.fit_cutpoint_binary_models`),
    the full report is built
    (:func:`evaluation.proportional_odds.build_diagnostic_report`), the
    artifact is written as
    ``<output_dir>/<version>/proportional_odds_diagnostic.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline, matching every other artifact in this repo), and a one-line
    summary (``proportional_odds_verdict``, ``sign_instability_count``,
    ``bic_ordinal``, ``bic_multinomial``) is logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the five input tables, or either
            fitted model artifact (``ordinal_logit_model.json`` /
            ``multinomial_logit_model.json``), does not exist for the
            requested version — the model artifacts propagate as-is
            from ``json.load`` as a clear "run the training drivers
            first" signal; the diagnostic never retrains anything.
        ValueError: If the train split is empty, a label is invalid, a
            feature computation fails, or the report validation fails
            (propagated from :func:`drivers.training_data.assemble_design_matrix`
            / :func:`evaluation.proportional_odds.fit_cutpoint_binary_models`
            / :func:`evaluation.proportional_odds.build_diagnostic_report`).
        KeyError: If any input table or model artifact lacks a required
            key/column (propagated from the feature modules / harness /
            the two ``from_dict`` functions).
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

    ordinal_path = output_dir / args.version / "ordinal_logit_model.json"
    with open(ordinal_path, encoding="utf-8") as handle:
        ordinal_model = ordinal_logit.from_dict(json.load(handle))
    multinomial_path = output_dir / args.version / "multinomial_logit_model.json"
    with open(multinomial_path, encoding="utf-8") as handle:
        multinomial_model = multinomial_logit.from_dict(json.load(handle))

    cutpoint_models = proportional_odds.fit_cutpoint_binary_models(X, y)
    report = proportional_odds.build_diagnostic_report(
        ordinal_model, multinomial_model, cutpoint_models, X, y
    )

    artifact_path = (
        output_dir / args.version / "proportional_odds_diagnostic.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "proportional-odds diagnostic on %d training maps (%s/%s): "
        "verdict=%s sign_instability_count=%d bic_ordinal=%.4f "
        "bic_multinomial=%.4f",
        report["n_train"],
        output_dir,
        args.version,
        report["proportional_odds_verdict"],
        report["sign_instability_count"],
        report["bic_ordinal"],
        report["bic_multinomial"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

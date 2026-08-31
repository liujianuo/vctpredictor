"""Command-line temperature-calibration comparison — the M24 report driver.

Thin command-line wrapper around
:mod:`evaluation.temperature_calibration`, which owns the pure
comparison logic (pairing two already-scored tables, computing the
metric deltas and the decision-agreement rates). This module adds only
the CLI/IO glue: argument parsing (:func:`parse_args`), loading the
five input tables (reusing the ``load_*_table`` helpers from
:mod:`drivers.evaluate`), building the held-out ``split="test"`` map
set via :func:`evaluation.harness.build_held_out_maps` (the same
35-map v1 held-out set the M19 harness scores), obtaining both model
functions by reusing :mod:`drivers.evaluate`'s ``MODEL_REGISTRY``
factories directly — ``"ordinal_logit"`` for the uncalibrated M20
model and ``"ordinal_logit_temperature"`` for the M24 scaled variant,
with no duplicated closure-building logic — scoring the identical
held-out rows with both, building the comparison report
(:func:`evaluation.temperature_calibration.build_calibration_comparison_report`),
and writing the artifact
``data/<version>/temperature_calibration_report.json``.

**Prerequisite:** both training drivers must already have been run for
the requested version (``drivers/train_ordinal_logit.py`` for the M20
arm, ``drivers/train_temperature_scaling.py`` for the M24 arm — the
latter also produces the ``temperature_scaling_model.json`` the
``ordinal_logit_temperature`` factory loads, enforcing its
thresholds-staleness guard). This script does not orchestrate training
itself (matching the one-script-one-job convention): if either model
artifact is missing, the ``FileNotFoundError`` propagates unchanged as
a clear "run the training drivers first" signal.

**Why both models go through the registry factories (and not, as in
:mod:`drivers.ablate_granularity`, direct ``predict_proba`` calls on a
pre-assembled design matrix).** Unlike the M22 binary model, the
temperature-scaled model *is* an honest four-way model with a natural
:data:`evaluation.harness.ModelFn` shape — the registry factory is its
canonical construction site (it owns the artifact loading, the
decision-E staleness guard, and the ``player_map_stats`` closure), and
the harness's :func:`evaluation.harness.score_held_out_maps` is the
standard way to score a ``ModelFn``. Reusing both means the report's
two arms are exactly what ``drivers/evaluate.py --model
ordinal_logit_temperature`` would produce, and there is exactly one
place that builds a scaled-eta prediction. This mirrors how
``evaluation.harness.build_evaluation_report`` numbers are the report's
own headline numbers.

Artifact written per run (scoped by dataset version):

- ``data/<version>/temperature_calibration_report.json`` — the
  :func:`evaluation.temperature_calibration.build_calibration_comparison_report`
  dict, written with ``json.dumps(..., indent=2, sort_keys=True)``
  plus a trailing newline (the same serialization convention as every
  other artifact in this repo).

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

from drivers import evaluate
from evaluation import harness, temperature_calibration
from models import temperature_scaling
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_temperature_calibration.py command line.

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
        (``ordinal_logit_model.json`` /
        ``temperature_scaling_model.json``) and the output artifact
        ``<output_dir>/<version>/temperature_calibration_report.json``.
        There are deliberately no other flags: the comparison uses the
        documented defaults throughout, matching the other drivers'
        no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M24 temperature-calibration comparison: score the "
            "fitted temperature-scaled ordinal model and the "
            "uncalibrated ordinal logit on the M10 test split, compare "
            "mean RPS / log loss / marginal binary accuracy and the "
            "decision-agreement rates, and write "
            "temperature_calibration_report.json."
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
    """Run the temperature-calibration comparison end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table` — loaded here
    as a fail-fast existence check even though each registry factory
    loads it again internally), the held-out test map set is built once
    (:func:`evaluation.harness.build_held_out_maps` with
    ``split="test"`` — the identical row table both models are scored
    on, in the identical order, which is exactly the row-alignment
    contract the comparison report validates), both model functions are
    obtained by reusing :mod:`drivers.evaluate`'s ``MODEL_REGISTRY``
    factories directly (``"ordinal_logit"`` and
    ``"ordinal_logit_temperature"`` — the latter owns the
    decision-E thresholds-staleness guard), both are scored via
    :func:`evaluation.harness.score_held_out_maps`, the comparison
    report is built
    (:func:`evaluation.temperature_calibration.build_calibration_comparison_report`),
    the artifact is written as
    ``<output_dir>/<version>/temperature_calibration_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), the fitted temperature is read from the calibration
    artifact for the summary, and a one-line summary (``temperature``,
    the three metric deltas, ``decisions_invariant``) is logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the five input tables, or either
            model artifact (``ordinal_logit_model.json`` /
            ``temperature_scaling_model.json``), does not exist for the
            requested version — the model artifacts propagate as-is
            from ``json.load`` (inside the factories / the temperature
            read below) as a clear "run train_ordinal_logit.py and
            train_temperature_scaling.py first" signal; this driver
            never retrains anything.
        ValueError: If the test split is empty, if the two model
            artifacts are shape-inconsistent, if the
            ``ordinal_logit_temperature`` factory's staleness guard
            trips (the calibration artifact was fit against a
            different ``ordinal_logit_model.json``), if a prediction
            fails the harness/metric validation, or if the two scored
            tables are not row-aligned (all propagated from
            :func:`evaluation.harness.build_held_out_maps` /
            :func:`evaluation.harness.score_held_out_maps` /
            :func:`evaluation.temperature_calibration.build_calibration_comparison_report`
            / the two ``from_dict`` functions).
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
    evaluate.load_player_map_stats_table(output_dir, args.version)

    # The one held-out row table both models are scored on, in the one
    # order — the row-alignment contract of the comparison report.
    held_out = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )

    # Both model functions come straight from the registry factories:
    # no duplicated closure-building logic, and the temperature arm
    # enforces the decision-E staleness guard as part of loading.
    uncalibrated_fn = evaluate.MODEL_REGISTRY["ordinal_logit"](
        output_dir, args.version
    )
    calibrated_fn = evaluate.MODEL_REGISTRY["ordinal_logit_temperature"](
        output_dir, args.version
    )
    scored_uncalibrated = harness.score_held_out_maps(
        uncalibrated_fn, held_out, matches_df, maps_df
    )
    scored_calibrated = harness.score_held_out_maps(
        calibrated_fn, held_out, matches_df, maps_df
    )

    report = temperature_calibration.build_calibration_comparison_report(
        scored_uncalibrated, scored_calibrated
    )

    artifact_path = (
        output_dir / args.version / "temperature_calibration_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The fitted T, for the summary line: read from the calibration
    # artifact the training driver wrote (the same file the factory
    # just loaded and validated).
    temp_path = output_dir / args.version / "temperature_scaling_model.json"
    with open(temp_path, encoding="utf-8") as handle:
        temp_model = temperature_scaling.from_dict(json.load(handle))

    logger.info(
        "temperature calibration on %d held-out maps (%s/%s): "
        "temperature=%.6f mean_rps_delta=%+.4f mean_log_loss_delta=%+.4f "
        "marginal_binary_accuracy_delta=%+.4f "
        "binary_side_agreement_rate=%.4f "
        "argmax_category_agreement_rate=%.4f decisions_invariant=%s",
        report["n_eval"],
        output_dir,
        args.version,
        temp_model.temperature,
        report["mean_rps_delta"],
        report["mean_log_loss_delta"],
        report["marginal_binary_accuracy_delta"],
        report["binary_side_agreement_rate"],
        report["argmax_category_agreement_rate"],
        report["decisions_invariant"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

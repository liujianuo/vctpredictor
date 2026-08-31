"""Command-line veto evaluation — the M26 head-to-head report driver.

Thin command-line wrapper around :mod:`evaluation.veto_evaluation`,
which owns the pure per-step scoring and report logic. This module
adds only the CLI/IO glue: argument parsing (:func:`parse_args`),
loading the four input tables (reusing the ``load_*_table`` helpers
from :mod:`drivers.evaluate`), building the held-out
``split="test"`` veto-step set via
:func:`evaluation.veto_evaluation.build_held_out_veto_matches`,
scoring both arms over that identical row table via
:func:`evaluation.veto_evaluation.score_veto_steps` — the M25 greedy
arm (:func:`evaluation.veto_evaluation.greedy_veto_step_model`) and
the global play-frequency baseline
(:func:`evaluation.veto_evaluation.most_frequent_map_baseline_model`)
— building the comparison report
(:func:`evaluation.veto_evaluation.build_veto_comparison_report`), and
writing the artifact
``data/<version>/veto_evaluation_report.json``.

Both arms are stateless: neither needs a fitted artifact or extra
tables, so unlike ``drivers/evaluate_temperature_calibration.py`` this
driver has no training-driver prerequisite beyond ``materialize.py``
and ``splits.py`` having been run for the requested version.

Artifact written per run (scoped by dataset version):

- ``data/<version>/veto_evaluation_report.json`` — the
  :func:`evaluation.veto_evaluation.build_veto_comparison_report`
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
from evaluation import veto_evaluation
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_veto.py command line.

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
        they locate the four input tables (``matches``/``maps``/
        ``splits``/``veto_actions``) and the output artifact
        ``<output_dir>/<version>/veto_evaluation_report.json``. There
        are deliberately no other flags: both arms use the documented
        defaults throughout (``k = features.map_win_rate.DEFAULT_K``
        for the greedy arm, ``map_pool=None`` era resolution for the
        scorer), matching the other drivers' no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M26 veto evaluation: score the M25 greedy "
            "simulator and the most-frequently-played-map baseline "
            "per-step over the held-out test split's real veto logs, "
            "compare mean cross-entropy / top-1 / top-3 accuracy, and "
            "write veto_evaluation_report.json."
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
    """Run the veto evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The four input tables are loaded for the requested version
    (matches/maps/splits via the ``drivers.evaluate`` helpers, plus
    ``veto_actions`` via
    :func:`drivers.evaluate.load_veto_actions_table`), the held-out
    test veto-step table is built once
    (:func:`evaluation.veto_evaluation.build_held_out_veto_matches`
    with ``split="test"`` — the identical row table both arms are
    scored on, in the identical order, which is exactly the
    row-alignment contract the comparison report validates), both arms
    are scored via
    :func:`evaluation.veto_evaluation.score_veto_steps`
    (``greedy_veto_step_model`` for the M25 arm and
    ``most_frequent_map_baseline_model`` for the frequency baseline,
    both stateless), the comparison report is built
    (:func:`evaluation.veto_evaluation.build_veto_comparison_report`),
    the artifact is written as
    ``<output_dir>/<version>/veto_evaluation_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary (``n_steps``, both arms'
    ``mean_cross_entropy``/``top1_accuracy``/``top3_accuracy``, and
    the three deltas) is logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the four input tables does not
            exist for the requested version (i.e. ``materialize.py`` /
            ``splits.py`` have not been run for it) — propagated as-is
            from the ``load_*`` helpers as a clear "run materialize.py
            first" signal.
        ValueError: If the test split's held-out veto set is empty, an
            abbreviation fails reconciliation, a match's real veto
            map-name set mismatches its era pool, a predictor output is
            not a valid distribution, a metric cannot be computed, or
            the two scored tables are not row-aligned (all propagated
            from :mod:`evaluation.veto_evaluation`'s pure functions).
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

    # The one held-out veto-step row table both arms are scored on, in
    # the one order — the row-alignment contract of the comparison
    # report.
    held_out = veto_evaluation.build_held_out_veto_matches(
        veto_df, matches_df, splits_df, split="test"
    )

    scored_greedy = veto_evaluation.score_veto_steps(
        veto_evaluation.greedy_veto_step_model,
        held_out,
        matches_df,
        maps_df,
    )
    scored_baseline = veto_evaluation.score_veto_steps(
        veto_evaluation.most_frequent_map_baseline_model,
        held_out,
        matches_df,
        maps_df,
    )

    report = veto_evaluation.build_veto_comparison_report(
        scored_greedy, scored_baseline
    )

    artifact_path = output_dir / args.version / "veto_evaluation_report.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "veto evaluation on %d held-out test-split steps (%s/%s): "
        "greedy mean_cross_entropy=%.6f top1=%.4f top3=%.4f | "
        "baseline mean_cross_entropy=%.6f top1=%.4f top3=%.4f | "
        "delta mean_cross_entropy=%+.6f top1=%+.4f top3=%+.4f",
        report["greedy"]["n_steps"],
        output_dir,
        args.version,
        report["greedy"]["mean_cross_entropy"],
        report["greedy"]["top1_accuracy"],
        report["greedy"]["top3_accuracy"],
        report["baseline"]["mean_cross_entropy"],
        report["baseline"]["top1_accuracy"],
        report["baseline"]["top3_accuracy"],
        report["delta"]["mean_cross_entropy"],
        report["delta"]["top1_accuracy"],
        report["delta"]["top3_accuracy"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

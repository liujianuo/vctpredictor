"""Command-line evaluation of the conditional-logit pick model (roadmap M28).

Thin command-line wrapper around :mod:`models.conditional_logit_pick`
(the fitted predictor) and :mod:`evaluation.veto_evaluation` (the
per-step scorer and the report builder). This module adds only the
CLI/IO glue: argument parsing (:func:`parse_args`), loading the fitted
model artifact (``conditional_logit_pick_model.json`` via
:func:`models.conditional_logit_pick.from_dict`) plus the four input
tables, building the held-out ``split="test"`` veto-step set via
:func:`evaluation.veto_evaluation.build_held_out_veto_matches`,
scoring three arms over the identical held-out rows via
:func:`evaluation.veto_evaluation.score_veto_steps` with the new
``actions_to_score={"pick"}`` filter — the fitted conditional-logit
predictor (:func:`models.conditional_logit_pick.make_veto_step_predictor_fn`),
the M25 greedy arm
(:func:`evaluation.veto_evaluation.greedy_veto_step_model`), and the
global play-frequency baseline
(:func:`evaluation.veto_evaluation.most_frequent_map_baseline_model`)
— building the three-arm comparison report
(:func:`evaluation.veto_evaluation.build_veto_multi_arm_report` with
``baseline_arm="baseline"``), and writing the artifact
``data/<version>/conditional_logit_pick_evaluation_report.json``.

**Prerequisite:** ``drivers/train_conditional_logit_pick.py`` must
already have been run for the requested version. This script does not
orchestrate training itself (matching the one-script-one-job
convention); if the model artifact is missing, the
``FileNotFoundError`` propagates unchanged as a clear "run the
training driver first" signal. This is the same shape of prerequisite
distinction the M27 pick-side evaluation driver already draws — which
is why this is a separate driver from ``drivers/evaluate_veto.py``
rather than a flag added to it.

**Why ``actions_to_score={"pick"}.``** The conditional-logit predictor
only supports ``action == "pick"`` (decision 5), but the shared
teacher-forced replay must walk the *full* held-out sequence (decider
and ban steps included) so the ``remaining``-set bookkeeping stays
correct across ban steps too. The ``actions_to_score`` filter scores
only pick steps while still consuming every step's bookkeeping — the
identical replay contract all three arms share, and therefore the
row-alignment contract
:func:`evaluation.veto_evaluation.build_veto_multi_arm_report`
validates.

Artifact written per run (scoped by dataset version):

- ``data/<version>/conditional_logit_pick_evaluation_report.json`` —
  the :func:`evaluation.veto_evaluation.build_veto_multi_arm_report`
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
from models import conditional_logit_pick
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_conditional_logit_pick.py command line.

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
        they locate the four input tables, the fitted model artifact
        ``<output_dir>/<version>/conditional_logit_pick_model.json``,
        and the output artifact
        ``<output_dir>/<version>/conditional_logit_pick_evaluation_report.json``.
        There are deliberately no other flags: all three arms use the
        documented defaults throughout (``k =
        features.map_win_rate.DEFAULT_K`` for the greedy arm,
        ``map_pool=None`` era resolution for the scorer), matching the
        other drivers' no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M28 pick-model evaluation: score the fitted "
            "conditional-logit pick model, the M25 greedy simulator and "
            "the most-frequently-played-map baseline per-step over the "
            "held-out test split's real veto logs (pick actions only), "
            "compare mean cross-entropy / top-1 / top-3 accuracy, and "
            "write conditional_logit_pick_evaluation_report.json."
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
    """Run the conditional-logit pick evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The fitted model artifact is loaded first
    (:func:`models.conditional_logit_pick.from_dict` on
    ``conditional_logit_pick_model.json`` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training driver first"
    signal), the four input tables are loaded for the requested
    version (matches/maps/splits via the ``drivers.evaluate`` helpers,
    plus ``veto_actions`` via
    :func:`drivers.evaluate.load_veto_actions_table`), the held-out
    test veto-step table is built once
    (:func:`evaluation.veto_evaluation.build_held_out_veto_matches`
    with ``split="test"`` — the identical row table all three arms are
    scored on, in the identical order), all three arms are scored via
    :func:`evaluation.veto_evaluation.score_veto_steps` with
    ``actions_to_score={"pick"}`` (the fitted
    :func:`models.conditional_logit_pick.make_veto_step_predictor_fn`
    closure for the trained arm,
    :func:`evaluation.veto_evaluation.greedy_veto_step_model` for the
    M25 arm, and
    :func:`evaluation.veto_evaluation.most_frequent_map_baseline_model`
    for the frequency baseline — the same two stateless arms
    ``drivers/evaluate_veto.py`` scores, restricted to pick steps so
    the comparison is apples-to-apples against the pick-only trained
    model), the three-arm report is built
    (:func:`evaluation.veto_evaluation.build_veto_multi_arm_report`
    with ``baseline_arm="baseline"`` — the frequency baseline, matching
    the M26 report's baseline convention), the artifact is written as
    ``<output_dir>/<version>/conditional_logit_pick_evaluation_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary (``n_steps``, all three arms'
    ``mean_cross_entropy``/``top1_accuracy``/``top3_accuracy``, and
    both arms' deltas vs the baseline) is logged — the headline
    comparison the roadmap milestone exists to produce.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If the fitted model artifact or any of the
            four input tables does not exist for the requested version
            (i.e. the training driver / ``materialize.py`` /
            ``splits.py`` have not been run for it) — propagated
            unchanged as a clear "run the prerequisite first" signal.
        ValueError: If the test split's held-out veto set is empty,
            contains no pick rows, an abbreviation fails
            reconciliation, a match's real veto map-name set mismatches
            its era pool, a predictor output is not a valid
            distribution (including the conditional-logit predictor's
            ``action != "pick"`` guard and its
            unresolvable/ambiguous-opponent guard), a metric cannot be
            computed, or the three scored tables are not row-aligned
            (all propagated from :mod:`evaluation.veto_evaluation` /
            :mod:`models.conditional_logit_pick`).
        KeyError: If any input table or the model artifact lacks a
            required column/key (propagated from the pure functions /
            :func:`models.conditional_logit_pick.from_dict`).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    model_path = (
        output_dir / args.version / "conditional_logit_pick_model.json"
    )
    with open(model_path, encoding="utf-8") as handle:
        model = conditional_logit_pick.from_dict(json.load(handle))

    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    veto_df = evaluate.load_veto_actions_table(output_dir, args.version)

    # The one held-out veto-step row table all three arms are scored
    # on, in the one order — the row-alignment contract of the
    # multi-arm report. actions_to_score={"pick"} scores only pick
    # steps while the shared replay still consumes every step's
    # `remaining` bookkeeping (decision 5's filter).
    held_out = veto_evaluation.build_held_out_veto_matches(
        veto_df, matches_df, splits_df, split="test"
    )

    scored_conditional = veto_evaluation.score_veto_steps(
        conditional_logit_pick.make_veto_step_predictor_fn(model),
        held_out,
        matches_df,
        maps_df,
        actions_to_score={"pick"},
    )
    scored_greedy = veto_evaluation.score_veto_steps(
        veto_evaluation.greedy_veto_step_model,
        held_out,
        matches_df,
        maps_df,
        actions_to_score={"pick"},
    )
    scored_baseline = veto_evaluation.score_veto_steps(
        veto_evaluation.most_frequent_map_baseline_model,
        held_out,
        matches_df,
        maps_df,
        actions_to_score={"pick"},
    )

    report = veto_evaluation.build_veto_multi_arm_report(
        {
            "conditional_logit_pick": scored_conditional,
            "greedy": scored_greedy,
            "baseline": scored_baseline,
        },
        baseline_arm="baseline",
    )

    artifact_path = (
        output_dir
        / args.version
        / "conditional_logit_pick_evaluation_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "conditional-logit pick evaluation on %d held-out test-split pick "
        "steps (%s/%s): conditional_logit_pick mean_cross_entropy=%.6f "
        "top1=%.4f top3=%.4f | greedy mean_cross_entropy=%.6f top1=%.4f "
        "top3=%.4f | baseline mean_cross_entropy=%.6f top1=%.4f top3=%.4f "
        "| delta vs baseline mean_cross_entropy=%+.6f top1=%+.4f "
        "top3=%+.4f (conditional_logit_pick), %+.6f/%+.4f/%+.4f (greedy)",
        report["conditional_logit_pick"]["n_steps"],
        output_dir,
        args.version,
        report["conditional_logit_pick"]["mean_cross_entropy"],
        report["conditional_logit_pick"]["top1_accuracy"],
        report["conditional_logit_pick"]["top3_accuracy"],
        report["greedy"]["mean_cross_entropy"],
        report["greedy"]["top1_accuracy"],
        report["greedy"]["top3_accuracy"],
        report["baseline"]["mean_cross_entropy"],
        report["baseline"]["top1_accuracy"],
        report["baseline"]["top3_accuracy"],
        report["deltas_vs_baseline"]["conditional_logit_pick"][
            "mean_cross_entropy"
        ],
        report["deltas_vs_baseline"]["conditional_logit_pick"]["top1_accuracy"],
        report["deltas_vs_baseline"]["conditional_logit_pick"]["top3_accuracy"],
        report["deltas_vs_baseline"]["greedy"]["mean_cross_entropy"],
        report["deltas_vs_baseline"]["greedy"]["top1_accuracy"],
        report["deltas_vs_baseline"]["greedy"]["top3_accuracy"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

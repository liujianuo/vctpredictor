"""Command-line two-arm series evaluation — the M33b headline report driver.

Thin command-line wrapper around :mod:`evaluation.series_evaluation`
(the series harness core, M33a) and :mod:`evaluation.veto_marginalized_series`
(the two-stage veto-marginalised pipeline, M31), mirroring the
``drivers/evaluate_veto.py`` / ``drivers/evaluate_conditional_logit_ban.py``
shape exactly: ``parse_args`` -> load tables/artifacts -> build the
held-out series set -> score both arms -> build the multi-arm report ->
write the JSON artifact -> log a one-line summary. This module adds
only the CLI/IO glue and the model wiring — all comparison logic is
delegated to
:func:`evaluation.series_evaluation.build_series_multi_arm_report`,
which is the milestone's "emit the report table" engine.

The two arms scored over the *identical* held-out ``split="test"``
series table (built once via
:func:`evaluation.series_evaluation.build_held_out_series` — the
row-alignment contract :func:`evaluation.series_evaluation.build_series_multi_arm_report`
validates):

- **``veto_marginalized_series``** — the M31 two-stage pipeline:
  :func:`evaluation.veto_marginalized_series.make_series_model_fn`
  over the *real fitted* Stage-2 four-way map model
  (``models.ordinal_logit.make_model_fn`` on the fitted
  ``ordinal_logit_model.json`` + ``player_map_stats.parquet``) and the
  *real fitted* Stage-1 veto-step predictors
  (``models.conditional_logit_ban.make_veto_step_predictor_fn`` /
  ``models.conditional_logit_pick.make_veto_step_predictor_fn`` on
  their fitted artifacts). This wiring is copied verbatim from
  ``tests/test_veto_marginalized_series.py``'s
  ``_real_v1_fitted_pipeline()`` / ``test_real_v1_harness_end_to_end``
  (the reference wiring, read in full before this driver was written).
  ``n_samples`` / ``rng`` are CLI-configurable: a single
  ``numpy.random.default_rng(seed)`` is constructed once and passed
  straight into ``make_series_model_fn``, whose returned closure
  consumes it sequentially across every held-out series — so a fixed
  ``--seed`` reproduces the whole run byte-identically.
- **``flat_series_baseline``** — the M32 mapless baseline, adapted via
  :func:`evaluation.series_evaluation.flat_series_baseline_model`.

**Which arm is the baseline.** ``baseline_arm="flat_series_baseline"``
(M32): the roadmap asks "was the two-stage pipeline worth it?", i.e.
the two-stage M31 arm is measured *against* the cheap M32 baseline —
the same "baseline is always the cheaper/simpler arm" convention
``drivers/evaluate_veto.py`` / ``drivers/evaluate_conditional_logit_ban.py``
use.

**Prerequisite artifacts.** This driver depends on
``drivers/train_ordinal_logit.py``, ``drivers/train_conditional_logit_ban.py``
and ``drivers/train_conditional_logit_pick.py`` having already been
run for the requested version (three artifacts, mirroring
``drivers/evaluate_conditional_logit_ban.py``'s single-artifact
prerequisite pattern). A missing artifact raises ``FileNotFoundError``
unchanged — no silent fallback, the same "run the training driver
first" doctrine every existing ``evaluate_*.py`` driver follows.

**The ``n_samples`` default is a measured wall-clock choice, not a
guess.** :func:`make_series_model_fn` requires an explicit ``rng``
(no default) and an ``n_samples`` int; there is no existing project
default. The plan proposed 50 (M29's own real-v1 smoke-test scale) but
flagged that the M31 harness smoke test runs at ``n_samples=2`` over a
5-match slice for speed, so BUILD was required to *time* a real run
and pick a default that finishes in reasonable CLI time. Measured on
real v1 fitted models: ~1.5-1.8 s per sampled sequence per series
(three ordinal-logit map scores + one veto walk each), so the full
15-match Bo3 test split at ``n_samples=50`` would be ~20 minutes.
``DEFAULT_N_SAMPLES = 10`` therefore (within the plan's anticipated
10-20 range): the full real-v1 run lands around 4-5 minutes of wall
clock, a reasonable single CLI invocation. The measured per-sample
cost and the chosen default are recorded in the ``tasks/036`` BUILD
status note.

**Bo5 sparsity is expected, not a bug.** M33a's build notes record that
all 15 held-out v1 test-split matches are Bo3 (v1's ``matches.parquet``
has 96 Bo3 rows and only 2 Bo5 rows total, across train+test), so the
real v1 report legitimately has **no** ``"Bo5"`` key:
:func:`evaluation.series_evaluation.build_series_evaluation_report`
omits an empty ``best_of`` group rather than erroring, and the delta
block omits a group absent from either arm. The driver must not assume
Bo5 is present — the summary-logging code guards with ``in`` / ``.get``
checks, never a bare key index. The Bo5 branch of the two-arm report is
exercised by a synthetic held-out table in
``tests/test_evaluate_series.py`` instead of real v1 data.

Artifact written per run (scoped by dataset version):

- ``data/<version>/series_evaluation_report.json`` — the
  :func:`evaluation.series_evaluation.build_series_multi_arm_report`
  dict (keys ``"veto_marginalized_series"``,
  ``"flat_series_baseline"``, ``"deltas_vs_flat_series_baseline"``)
  plus the provenance keys ``"n_samples"`` and ``"seed"`` (so the
  report is self-describing and reproducible without cross-referencing
  the invocation command), written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
  newline (the repo-wide convention).

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

import numpy as np

from drivers import evaluate
from evaluation import series_evaluation, veto_marginalized_series
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The default number of M29 veto walks sampled per held-out series by
# the M31 arm, and the default RNG seed. Both are documented choices,
# not constants-of-convenience: DEFAULT_N_SAMPLES is the measured
# wall-clock compromise described in the module docstring (~1.5-1.8 s
# per sampled sequence per series on real v1 fitted models, so the
# full 15-match split at 10 samples/series lands around 4-5 minutes),
# and DEFAULT_SEED matches the repo's "current year" seed convention
# already used by tasks/035's real-data smoke test.
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 2026


def _load_fitted_models(
    output_dir: Path, version: str
) -> tuple[object, object, object]:
    """Load the three fitted model artifacts this driver needs.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json``, the fitted M27 conditional-logit ban
    model from ``conditional_logit_ban_model.json``, and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the exact artifact-loading
    pattern ``drivers/evaluate_conditional_logit_ban.py`` already uses
    inline, factored into one in-driver helper so the triplicated
    ``json.load`` + ``from_dict`` boilerplate lives in exactly one
    place (it stays in this driver module per the plan's assumption 2:
    no new evaluation/ or models/ code is needed for artifact loading).
    The three ``from_dict`` calls are deliberately independent of each
    other and of the four input tables, so a missing artifact fails
    fast with the standard "run the training driver first" signal.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ordinal_model, ban_model, pick_model)`` tuple of the three
        deserialized fitted models, in the order the driver wires them
        (the ordinal model feeds ``map_model_fn``; the ban/pick models
        feed the ``predictor_fn_by_action`` dict).

    Raises:
        FileNotFoundError: If any of the three model artifacts does not
            exist for the requested version (i.e. the corresponding
            training driver has not been run for it) — propagated
            unchanged from ``open()`` as a clear "run the training
            driver first" signal, never wrapped or silently skipped.
        KeyError: If an artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        ValueError: If an artifact's shapes are inconsistent (propagated
            from the ``from_dict`` calls).
    """
    ordinal_model = ordinal_logit.from_dict(
        json.loads(
            (
                output_dir / version / "ordinal_logit_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    ban_model = conditional_logit_ban.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_ban_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    pick_model = conditional_logit_pick.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_pick_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    return ordinal_model, ban_model, pick_model


def _summarize_group(report: dict, best_of: str) -> str:
    """Format one ``best_of`` group's headline numbers for the summary line.

    Composes the per-group segment of the CLI's one-line summary: both
    arms' ``mean_rps`` / ``mean_log_loss`` / ``marginal_binary_accuracy``
    plus the veto arm's deltas vs the flat baseline, all read from the
    ``build_series_multi_arm_report`` dict. A ``best_of`` group that is
    absent from the delta block (present in only one arm) is rendered
    without a delta segment rather than raising — the same "never bare
    key index" guard the module docstring requires, since the real v1
    report has no ``Bo5`` group at all and the summary line must not
    assume one.

    Args:
        report: The ``build_series_multi_arm_report`` dict (with the
            ``"veto_marginalized_series"`` / ``"flat_series_baseline"``
            arm blocks and the ``"deltas_vs_flat_series_baseline"``
            block).
        best_of: The ``"Bo<N>"`` group key to summarize (e.g.
            ``"Bo3"``, ``"Bo5"``).

    Returns:
        A single-line string like
        ``"Bo3: veto_marginalized_series mean_rps=1.234567
        mean_log_loss=1.234567 marginal_binary_accuracy=0.8000 |
        flat_series_baseline mean_rps=... | delta mean_rps=+0.012345
        mean_log_loss=... marginal=..."`` — the two arm blocks joined
        by ``" | "`` with the delta segment appended only when the
        group is present in the delta block.

    Raises:
        KeyError: If ``best_of`` is absent from either arm's report
            block (propagated from dict indexing — the caller must only
            pass groups that both arms actually contain).
    """
    veto_block = report["veto_marginalized_series"][best_of]
    flat_block = report["flat_series_baseline"][best_of]
    line = (
        f"{best_of}: veto_marginalized_series "
        f"mean_rps={veto_block['mean_rps']:.6f} "
        f"mean_log_loss={veto_block['mean_log_loss']:.6f} "
        f"marginal_binary_accuracy={veto_block['marginal_binary_accuracy']:.4f}"
        f" | flat_series_baseline "
        f"mean_rps={flat_block['mean_rps']:.6f} "
        f"mean_log_loss={flat_block['mean_log_loss']:.6f} "
        f"marginal_binary_accuracy={flat_block['marginal_binary_accuracy']:.4f}"
    )
    delta_block = report["deltas_vs_flat_series_baseline"][
        "veto_marginalized_series"
    ]
    if best_of in delta_block:
        delta = delta_block[best_of]
        line += (
            f" | delta mean_rps={delta['mean_rps_delta']:+.6f} "
            f"mean_log_loss={delta['mean_log_loss_delta']:+.6f} "
            f"marginal_binary_accuracy="
            f"{delta['marginal_binary_accuracy_delta']:+.4f}"
        )
    return line


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_series.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with four attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and the
        two M31 reproducibility knobs from plan assumption 5:
        ``n_samples`` (``int``, the M29 walks sampled per held-out
        series, default :data:`DEFAULT_N_SAMPLES` — the measured
        wall-clock choice documented in the module docstring) and
        ``seed`` (``int``, the ``numpy.random.default_rng`` seed for
        the whole run, default :data:`DEFAULT_SEED`). Together they
        locate the four input tables, the three fitted model artifacts,
        and the output artifact
        ``<output_dir>/<version>/series_evaluation_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-samples``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M33b two-arm series evaluation: score the M31 "
            "veto-marginalised two-stage pipeline (real fitted ordinal-"
            "logit map model + conditional-logit ban/pick veto-step "
            "predictors) and the M32 flat series baseline over the "
            "identical held-out test-split series, compare mean RPS / "
            "mean log loss / marginal match-win accuracy per Bo3 and "
            "Bo5, and write series_evaluation_report.json."
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
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per held-out series by the M31 "
            f"arm (default: {DEFAULT_N_SAMPLES} — a measured wall-clock "
            "compromise: ~1.5-1.8s per sampled sequence on real v1 "
            "fitted models, so the full 15-series v1 test split takes "
            "~4-5 minutes; raise it for a smoother aggregate, lower it "
            "for a faster run)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, giving the whole run "
            f"byte-identical reproducibility (default: {DEFAULT_SEED})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the two-arm series evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The three fitted model artifacts are loaded first
    (:func:`_load_fitted_models` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training drivers first"
    signal), the four input tables are loaded for the requested version
    (matches/maps/splits/player_map_stats via the ``drivers.evaluate``
    helpers), the M31 arm's pluggable callables are wired per plan
    assumption 4 (``map_model_fn`` =
    :func:`models.ordinal_logit.make_model_fn` over the fitted ordinal
    model + ``player_map_stats.parquet``; ``predictor_fn_by_action`` =
    the ban/pick conditional-logit
    ``make_veto_step_predictor_fn`` closures), a single
    ``numpy.random.default_rng(args.seed)`` is constructed and passed
    into :func:`evaluation.veto_marginalized_series.make_series_model_fn`
    with ``n_samples=args.n_samples`` (the closure consumes that one
    ``rng`` sequentially across every held-out series, so a fixed seed
    reproduces the whole run), the held-out test-split series table is
    built once
    (:func:`evaluation.series_evaluation.build_held_out_series` — the
    identical row table both arms are scored on, in the identical
    order, exactly the row-alignment contract
    :func:`evaluation.series_evaluation.build_series_multi_arm_report`
    validates), both arms are scored via
    :func:`evaluation.series_evaluation.score_held_out_series` (once
    with the M31 ``veto_model_fn``, once with
    :func:`evaluation.series_evaluation.flat_series_baseline_model`),
    the two-arm report is built
    (:func:`evaluation.series_evaluation.build_series_multi_arm_report`
    with ``baseline_arm="flat_series_baseline"``), the ``n_samples`` /
    ``seed`` provenance keys are merged in (plan assumption 5), the
    artifact is written as
    ``<output_dir>/<version>/series_evaluation_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: per-``best_of``
    ``mean_rps`` / ``mean_log_loss`` / ``marginal_binary_accuracy``
    for both arms plus the veto arm's deltas vs the flat baseline —
    guarded so a ``best_of`` group that is absent from the report (the
    real v1 report legitimately has no ``Bo5`` group, plan assumption
    6) simply does not appear in the summary line.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the three fitted model artifacts
            or any of the four input tables does not exist for the
            requested version (i.e. the training drivers /
            ``materialize.py`` / ``splits.py`` have not been run for
            it) — propagated unchanged as a clear "run the prerequisite
            first" signal.
        ValueError: If the test split's held-out series set is empty,
            a held-out match has zero maps / a tied or null score /
            a malformed ``best_of`` (from
            :func:`evaluation.series_evaluation.build_held_out_series`);
            if the M31 arm's map model or the scoreline enumeration
            rejects any input, including a degenerate all-zero-
            probability sample set (propagated from
            :mod:`evaluation.veto_marginalized_series`); if a scored
            vector has the wrong length or fails the metric validation
            (from :func:`evaluation.series_evaluation.score_held_out_series`,
            including ``log_loss``'s hard error on a zero-probability
            true scoreline); or if the two scored tables are not
            row-aligned (from
            :func:`evaluation.series_evaluation.build_series_multi_arm_report`).
        KeyError: If any input table or model artifact lacks a required
            column/key (propagated from the pure functions /
            ``from_dict`` calls).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    ordinal_model, ban_model, pick_model = _load_fitted_models(
        output_dir, args.version
    )

    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The M31 arm's pluggable callables, wired exactly as
    # tests/test_veto_marginalized_series.py's _real_v1_fitted_pipeline
    # does (plan assumption 4): the fitted M20 ordinal-logit closure as
    # the Stage-2 four-way map model, and the fitted M27/M28
    # conditional-logit closures as the Stage-1 ban/pick veto-step
    # predictors.
    map_model_fn = ordinal_logit.make_model_fn(
        ordinal_model, player_map_stats_df
    )
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }
    rng = np.random.default_rng(args.seed)
    veto_model_fn = veto_marginalized_series.make_series_model_fn(
        map_model_fn,
        predictor_fn_by_action,
        n_samples=args.n_samples,
        rng=rng,
    )

    # The one held-out series row table both arms are scored on, in the
    # one order — the row-alignment contract of the multi-arm report.
    held_out = series_evaluation.build_held_out_series(
        matches_df, maps_df, splits_df, split="test"
    )

    scored_veto = series_evaluation.score_held_out_series(
        veto_model_fn, held_out, matches_df, maps_df
    )
    scored_flat = series_evaluation.score_held_out_series(
        series_evaluation.flat_series_baseline_model,
        held_out,
        matches_df,
        maps_df,
    )

    report = series_evaluation.build_series_multi_arm_report(
        {
            "flat_series_baseline": scored_flat,
            "veto_marginalized_series": scored_veto,
        },
        baseline_arm="flat_series_baseline",
    )
    # Provenance (plan assumption 5): record the resolved knobs so the
    # report is self-describing and reproducible without cross-
    # referencing the invocation command.
    report["n_samples"] = args.n_samples
    report["seed"] = args.seed

    artifact_path = (
        output_dir / args.version / "series_evaluation_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # One-line summary. Only best_of groups both arms actually contain
    # are summarized (guarded by an explicit membership check against
    # the flat arm's report blocks), so the real v1 report's legitimate
    # absence of a Bo5 group (plan assumption 6) never raises here.
    groups_present = sorted(
        key
        for key in report["flat_series_baseline"]
        if key != "n_eval_total"
    )
    summary = "; ".join(
        _summarize_group(report, best_of) for best_of in groups_present
    )
    logger.info(
        "series evaluation on %d held-out test-split series (%s/%s, "
        "n_samples=%d seed=%d): %s",
        report["veto_marginalized_series"]["n_eval_total"],
        output_dir,
        args.version,
        args.n_samples,
        args.seed,
        summary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

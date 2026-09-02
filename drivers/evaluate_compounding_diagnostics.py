"""Command-line compounding-diagnostics evaluation — the M35 headline report driver.

Thin command-line wrapper around :mod:`evaluation.compounding_diagnostics`
(the pure M35 analysis module) plus the existing harnesses that produce
its scored input tables, mirroring the ``drivers/evaluate_series.py`` /
``drivers/evaluate_stage_isolation.py`` shape exactly: ``parse_args``
-> load tables/artifacts -> build + score the held-out tables -> build
both diagnostics' reports -> write the JSON artifact -> log a one-line
summary. This module adds only the CLI/IO glue and the model wiring —
all comparison logic is delegated to the two pure report functions in
``evaluation/compounding_diagnostics.py``.

The two diagnostics' input tables are assembled separately, because
they answer different questions:

- **Diagnostic 1 (predicted vs observed sweep rate)** needs the scored
  *series* table — the full M31 two-stage pipeline's scoreline
  distributions. It is built via
  :func:`evaluation.series_evaluation.build_held_out_series` +
  :func:`evaluation.series_evaluation.score_held_out_series` using
  :func:`evaluation.veto_marginalized_series.make_series_model_fn`
  over the three real fitted artifacts (the identical M31 wiring
  ``drivers/evaluate_series.py`` already uses: the fitted M20
  ordinal-logit closure as the Stage-2 four-way map model, and the
  fitted M27/M28 conditional-logit closures as the Stage-1 ban/pick
  veto-step predictors), with this driver's own ``--n-samples`` /
  ``--seed`` pair for the veto sampler. This is the identical
  sampling workload ``drivers/evaluate_series.py`` runs, so its
  measured wall-clock figures apply: ~1.5-1.8 s per sampled sequence
  per series on real v1 fitted models, making :data:`DEFAULT_N_SAMPLES`
  (10) land the full 15-series v1 test split around 4-5 minutes.
- **Diagnostic 2 (does map 1's outcome predict map 2's residual)**
   needs the scored *map* table — Stage 2 alone, no veto sampling,
   since it only ever queries the *actual* played map-1/map-2
   identities. It is built via
   :func:`evaluation.harness.build_held_out_maps` +
   :func:`evaluation.harness.score_held_out_maps` using
   :func:`models.ordinal_logit.make_model_fn` over the fitted
   ordinal-logit artifact alone.

The two diagnostics' reports are pure functions of their scored tables
(:func:`evaluation.compounding_diagnostics.sweep_rate_report` and
:func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`),
so no other comparison logic lives in this driver.

**The permutation default is a cost-free choice, not a guess.** The
``--n-permutations`` / ``--permutation-seed`` pair drives diagnostic
2's hand-rolled permutation test (a pure ``numpy``
shuffle-and-recompute loop, no ``scipy``): at the n=15 v1 scale each
permutation is a handful of array operations, so
:data:`DEFAULT_N_PERMUTATIONS` (10000) is sub-second — no wall-clock
measurement needed for this part (unlike the veto-sampling side), and
the choice is recorded explicitly here. The permutation draw uses a
``numpy.random.default_rng(permutation_seed)`` constructed once and
passed to :func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`,
so a fixed ``--permutation-seed`` reproduces diagnostic 2
byte-identically; the ``--seed`` RNG (a separate
``numpy.random.default_rng``) drives only the M31 veto sampler and is
fully independent of it.

**Prerequisite artifacts.** This driver depends on
``drivers/train_ordinal_logit.py``, ``drivers/train_conditional_logit_ban.py``
and ``drivers/train_conditional_logit_pick.py`` having already been
run for the requested version (three artifacts, mirroring
``drivers/evaluate_series.py``'s prerequisite pattern). A missing
artifact raises ``FileNotFoundError`` unchanged — no silent fallback,
the same "run the training driver first" doctrine every existing
``evaluate_*.py`` driver follows.

Artifact written per run (scoped by dataset version):

- ``data/<version>/compounding_diagnostics_report.json`` — a dict with
  top-level keys ``"sweep_rate"`` (the
  :func:`evaluation.compounding_diagnostics.sweep_rate_report` dict,
  itself grouped by ``best_of``) and ``"map1_predicts_map2"`` (the
  :func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`
  dict), plus the provenance keys ``"n_samples"``, ``"seed"``,
  ``"n_permutations"`` and ``"permutation_seed"`` (so the report is
  self-describing and reproducible without cross-referencing the
  invocation command), written with
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
from evaluation import (
    compounding_diagnostics,
    harness,
    series_evaluation,
    veto_marginalized_series,
)
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The default number of M29 veto walks sampled per held-out series by
# diagnostic 1's M31 arm, and its RNG seed. Both are documented
# choices, not constants-of-convenience: DEFAULT_N_SAMPLES is the
# measured wall-clock compromise described in the module docstring
# (~1.5-1.8 s per sampled sequence per series on real v1 fitted
# models, so the full 15-series split at 10 samples/series lands
# around 4-5 minutes), and DEFAULT_SEED matches the repo's "current
# year" seed convention already used by drivers/evaluate_series.py /
# drivers/evaluate_stage_isolation.py.
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 2026

# The default number of relabelings drawn by diagnostic 2's empirical
# permutation test, and its RNG seed. DEFAULT_N_PERMUTATIONS (10000) is
# a sub-second pure-numpy loop at the n=15 v1 scale — no wall-clock
# constraint, unlike the veto-sampling side — and DEFAULT_PERMUTATION_SEED
# follows the same repo seed convention (the two RNGs are fully
# independent: permutation-seed drives diagnostic 2, seed drives the
# M31 sampler).
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_PERMUTATION_SEED = 2026


def _load_fitted_models(
    output_dir: Path, version: str
) -> tuple[object, object, object]:
    """Load the three fitted model artifacts this driver needs.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json``, the fitted M27 conditional-logit ban
    model from ``conditional_logit_ban_model.json``, and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the exact artifact-loading
    pattern ``drivers/evaluate_series.py::_load_fitted_models`` already
    provides, copied (not imported) into this driver per the repo's
    convention that each ``evaluate_*.py`` driver independently
    duplicates this helper. The three ``from_dict`` calls are
    deliberately independent of each other and of the four input
    tables, so a missing artifact fails fast with the standard "run
    the training driver first" signal.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ordinal_model, ban_model, pick_model)`` tuple of the three
        deserialized fitted models, in the order the driver wires them
        (the ordinal model feeds both ``map_model_fn`` uses; the
        ban/pick models feed the ``predictor_fn_by_action`` dict).

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


def _summarize_sweep_rate(sweep_report: dict) -> str:
    """Format the sweep-rate report's headline numbers for the summary line.

    Composes the per-``best_of`` segment of the CLI's one-line summary:
    each group's ``n_eval``, predicted mean sweep probability, observed
    sweep rate and calibration gap, read from the
    :func:`evaluation.compounding_diagnostics.sweep_rate_report` dict.
    Groups are iterated in sorted order; the ``n_eval_total`` key (a
    plain int, not a group block) is skipped. The real v1 report has
    only a ``"Bo3"`` group (all 15 held-out test-split matches are
    Bo3, per the M33b build notes), so a missing group can never raise
    here — but the loop is written over whatever groups are present
    rather than assuming any particular one.

    Args:
        sweep_report: The ``sweep_rate_report`` dict (one block per
            ``best_of`` group plus ``n_eval_total``).

    Returns:
        A single-line string like ``"Bo3 n=15 predicted_sweep=0.5333
        observed=0.3333 gap=+0.2000"`` — one segment per group, joined
        by ``"; "``, in sorted ``best_of`` order.

    Raises:
        Nothing.
    """
    segments = []
    for best_of in sorted(
        key for key in sweep_report if key != "n_eval_total"
    ):
        block = sweep_report[best_of]
        segments.append(
            f"{best_of} n={block['n_eval']} "
            f"predicted_sweep={block['predicted_mean_sweep_prob']:.4f} "
            f"observed={block['observed_sweep_rate']:.4f} "
            f"gap={block['sweep_calibration_gap']:+.4f}"
        )
    return "; ".join(segments)


def _summarize_map1_predicts_map2(report: dict) -> str:
    """Format the map1-predicts-map2 report's headline numbers for the summary line.

    Composes the diagnostic-2 segment of the CLI's one-line summary:
    the eligible-match count with its A/B subgroup split, the two
    group-mean residuals, the observed group-mean difference, and the
    empirical permutation p-value with its permutation count — all read
    from the
    :func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`
    dict.

    Args:
        report: The ``map1_predicts_map2_report`` dict.

    Returns:
        A single-line string like ``"n=15 (a_won=8, b_won=7)
        mean_resid_given_a=0.0200 mean_resid_given_b=-0.0500
        observed_diff=+0.0700 p_empirical=0.4231 (10000 permutations)"``.

    Raises:
        Nothing.
    """
    return (
        f"n={report['n_eligible_matches']} "
        f"(a_won={report['n_map1_a_won']}, b_won={report['n_map1_b_won']}) "
        f"mean_resid_given_a="
        f"{report['mean_residual_given_map1_a_won']:.4f} "
        f"mean_resid_given_b="
        f"{report['mean_residual_given_map1_b_won']:.4f} "
        f"observed_diff={report['observed_diff']:+.4f} "
        f"p_empirical={report['p_value_empirical']:.4f} "
        f"({report['n_permutations']} permutations)"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_compounding_diagnostics.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with six attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), the two
        M31 reproducibility knobs ``n_samples`` (``int``, the M29
        walks sampled per held-out series, default
        :data:`DEFAULT_N_SAMPLES`) and ``seed`` (``int``, the
        ``numpy.random.default_rng`` seed for the M31 sampler, default
        :data:`DEFAULT_SEED`), and the two diagnostic-2 knobs
        ``n_permutations`` (``int``, the relabelings drawn by the
        permutation test, default :data:`DEFAULT_N_PERMUTATIONS`) and
        ``permutation_seed`` (``int``, the
        ``numpy.random.default_rng`` seed for the permutation draw,
        default :data:`DEFAULT_PERMUTATION_SEED`). Together they
        locate the five input tables, the three fitted model artifacts,
        and the output artifact
        ``<output_dir>/<version>/compounding_diagnostics_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-permutations``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M35 compounding diagnostics: (1) predicted-vs-"
            "observed sweep rate (2-0/3-0) per best_of from the M31 "
            "two-stage pipeline's scored series table, and (2) a "
            "hand-rolled empirical permutation test of whether map 1's "
            "outcome predicts map 2's Stage-2 residual beyond what the "
            "features explain (fitted M20 ordinal-logit scored on the "
            "held-out test-split maps); write "
            "compounding_diagnostics_report.json."
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
            "M29 veto sequences sampled per held-out series by "
            "diagnostic 1's M31 arm (default: "
            f"{DEFAULT_N_SAMPLES} — a measured wall-clock compromise "
            "on real v1 fitted models; raise it for a smoother "
            "aggregate, lower it for a faster run)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, driving diagnostic 1's "
            f"M31 veto sampler (default: {DEFAULT_SEED})"
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=(
            "relabelings drawn by diagnostic 2's empirical permutation "
            f"test (default: {DEFAULT_N_PERMUTATIONS} — sub-second at "
            "the n=15 v1 scale, so the default is a cost-free choice "
            "rather than a wall-clock compromise)"
        ),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=DEFAULT_PERMUTATION_SEED,
        help=(
            "seed for numpy.random.default_rng, driving diagnostic 2's "
            "permutation draw (independent of --seed; default: "
            f"{DEFAULT_PERMUTATION_SEED})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compounding-diagnostics evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The three fitted model artifacts are loaded first
    (:func:`_load_fitted_models` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training drivers first"
    signal), the five input tables are loaded for the requested version
    (matches/maps/labels/splits/player_map_stats via the
    ``drivers.evaluate`` helpers), the pluggable callables are wired
    (``map_model_fn`` = :func:`models.ordinal_logit.make_model_fn`
    over the fitted ordinal model + ``player_map_stats.parquet``;
    ``predictor_fn_by_action`` = the ban/pick conditional-logit
    ``make_veto_step_predictor_fn`` closures — the identical wiring
    ``drivers/evaluate_series.py`` uses), and then:

    - **Diagnostic 2** (map1-predicts-map2) is built first: the held-
      out test-split map table via
      :func:`evaluation.harness.build_held_out_maps` (Stage 2 only —
      no veto sampling needed, since this diagnostic only ever queries
      the *actual* played map-1/map-2 identities), scored via
      :func:`evaluation.harness.score_held_out_maps`, and passed to
      :func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`
      with ``numpy.random.default_rng(args.permutation_seed)`` and
      ``args.n_permutations``.
    - **Diagnostic 1** (sweep rate) is built second: a single
      ``numpy.random.default_rng(args.seed)`` is constructed and passed
      into
      :func:`evaluation.veto_marginalized_series.make_series_model_fn`
      with ``n_samples=args.n_samples`` (the closure consumes that one
      ``rng`` sequentially across every held-out series, so a fixed
      seed reproduces the whole run), the held-out test-split series
      table is built once
      (:func:`evaluation.series_evaluation.build_held_out_series`),
      scored via
      :func:`evaluation.series_evaluation.score_held_out_series`, and
      passed to
      :func:`evaluation.compounding_diagnostics.sweep_rate_report`.

    The combined report (``"sweep_rate"`` + ``"map1_predicts_map2"``
    plus the four provenance keys) is written as
    ``<output_dir>/<version>/compounding_diagnostics_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: the per-``best_of``
    sweep headline numbers (n, predicted, observed, gap) followed by
    diagnostic 2's headline (eligible n with A/B split, group-mean
    residuals, observed diff, empirical p-value).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the three fitted model artifacts
            or any of the five input tables does not exist for the
            requested version (i.e. the training drivers /
            ``materialize.py`` / ``splits.py`` / ``labels.py`` have not
            been run for it) — propagated unchanged as a clear "run the
            prerequisite first" signal.
        ValueError: If the test split's held-out series or map set is
            empty, a held-out match has zero maps / a tied or null
            score / a malformed ``best_of`` (from
            :func:`evaluation.series_evaluation.build_held_out_series`);
            if the M31 arm's map model or the scoreline enumeration
            rejects any input (propagated from
            :mod:`evaluation.veto_marginalized_series`); if a scored
            vector has the wrong length or fails the metric validation
            (from the ``score_held_out_*`` functions); if a
            ``probabilities`` vector length desyncs with its
            ``best_of_int`` or the scored series table is empty (from
            :func:`evaluation.compounding_diagnostics.sweep_rate_report`);
            or if fewer than 2 matches have both map rows, either
            map-1-outcome subgroup is empty, or ``n_permutations`` is
            invalid (from
            :func:`evaluation.compounding_diagnostics.map1_predicts_map2_report`).
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
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The pluggable callables, wired exactly as
    # drivers/evaluate_series.py does: the fitted M20 ordinal-logit
    # closure as the Stage-2 four-way map model (used by *both*
    # diagnostics), and the fitted M27/M28 conditional-logit closures
    # as the Stage-1 ban/pick veto-step predictors (used by diagnostic
    # 1's M31 arm only).
    map_model_fn = ordinal_logit.make_model_fn(
        ordinal_model, player_map_stats_df
    )
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    # Diagnostic 2 first: Stage 2 alone on the actual played maps.
    held_out_maps = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    scored_maps = harness.score_held_out_maps(
        map_model_fn, held_out_maps, matches_df, maps_df
    )
    map1_map2_report = compounding_diagnostics.map1_predicts_map2_report(
        scored_maps,
        np.random.default_rng(args.permutation_seed),
        args.n_permutations,
    )

    # Diagnostic 1: the full M31 two-stage pipeline on the held-out
    # series (the identical sampling workload evaluate_series.py runs).
    rng = np.random.default_rng(args.seed)
    veto_model_fn = veto_marginalized_series.make_series_model_fn(
        map_model_fn,
        predictor_fn_by_action,
        n_samples=args.n_samples,
        rng=rng,
    )
    held_out_series = series_evaluation.build_held_out_series(
        matches_df, maps_df, splits_df, split="test"
    )
    scored_series = series_evaluation.score_held_out_series(
        veto_model_fn, held_out_series, matches_df, maps_df
    )
    sweep_report = compounding_diagnostics.sweep_rate_report(scored_series)

    report = {
        "sweep_rate": sweep_report,
        "map1_predicts_map2": map1_map2_report,
        # Provenance: record the resolved knobs so the report is
        # self-describing and reproducible without cross-referencing
        # the invocation command.
        "n_samples": args.n_samples,
        "seed": args.seed,
        "n_permutations": args.n_permutations,
        "permutation_seed": args.permutation_seed,
    }

    artifact_path = (
        output_dir / args.version / "compounding_diagnostics_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "compounding diagnostics on %d held-out test-split series "
        "(%s/%s, n_samples=%d seed=%d, n_permutations=%d "
        "permutation_seed=%d): sweep_rate[%s] | map1_predicts_map2[%s]",
        sweep_report["n_eval_total"],
        output_dir,
        args.version,
        args.n_samples,
        args.seed,
        args.n_permutations,
        args.permutation_seed,
        _summarize_sweep_rate(sweep_report),
        _summarize_map1_predicts_map2(map1_map2_report),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

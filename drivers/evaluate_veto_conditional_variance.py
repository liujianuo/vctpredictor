"""Veto-conditional variance driver — the M37 structural report (roadmap M37).

The orchestration half of M37 (veto-conditional variance, structural):
reuses M31's own per-sample scorelines directly —
:func:`evaluation.veto_marginalized_series.predict_series_outcome_via_veto_marginalization`
is called once per held-out test-split series with a large ``n_samples``
and the resulting ``prediction.samples[i].scoreline_probabilities``
(already an exact, deterministic enumeration per sampled veto sequence)
plus ``prediction.samples[i].weight`` are fed through the pure spread
helpers in :mod:`evaluation.veto_conditional_variance` (a downward-only
``evaluation/`` module with no orchestration, per the module-boundary
DAG); this driver owns the "load fixed artifacts + call M31 with more
samples + summarize the per-sample detail" loop because only
``drivers/`` may depend laterally on the sibling
``evaluation.veto_marginalized_series`` /
``evaluation.series_evaluation`` modules.

Phase 6 framing (restated here, do not re-derive later): M36
(epistemic — parameter uncertainty from a finite training sample), M37
(structural — spread across sampled veto sequences, resolves the moment
the veto happens) and M38 (calibration/reliability) are **three
distinct components, reported separately, never collapsed into one
number**. This driver builds ONLY the structural component: it holds
the Stage-2 (and Stage-1) fitted models **fixed** — the single nominal,
already-trained artifacts, no refitting, no bootstrap resampling of the
training set — and lets the *veto rng* vary naturally, because the veto
sequence itself is exactly the thing whose resulting spread this task
measures (the M37 plan's assumption 2). It does not bootstrap any model
(that would reintroduce M36's epistemic component into a number meant
to be purely structural) and does not touch reliability/calibration
(M38).

**Explicit contrast with M36's fixed-veto-rng mechanism (the M37
plan's assumption 9 — easy to misread as "should copy M36", it must
not):** M36 resets a fresh ``numpy.random.default_rng(veto_seed)``
identically for every bootstrap replicate specifically to *suppress*
structural veto variance so its reported spread is attributable to
Stage-2 parameter uncertainty alone. M37 is the mirror image: here a
**single** ``numpy.random.default_rng(args.seed)`` is constructed once
and consumed *sequentially* — the ``n_samples`` draws within one series
call vary naturally, and the one rng advances ordinarily across the
different held-out series (mirroring ``drivers/evaluate_series.py``'s
convention) — so a fixed ``--seed`` reproduces the whole run
byte-identically while the sampled veto sequences vary freely. **Do not
"fix" this to reset the rng per series** — that would collapse the
structural variance this task exists to measure.

Scope decisions (recorded here, do not re-derive later; all are the M37
plan's assumptions, restated for REVIEW):

1. **Reuse M31's own per-sample scorelines; no new sampler.** Each
   ancestral draw already carries an exact, deterministic scoreline
   distribution (``SeriesVetoSample.scoreline_probabilities``) and its
   normalized ``weight``; M37 needs no new sampling architecture, just
   a larger ``n_samples`` fed to the existing M31 entry point and the
   per-sample detail read off the returned prediction (assumption 1 —
   the reason this task adds no new models/ or evaluation/-sampling
   code).
2. **Only the single nominal fitted Stage-2 model, and the fixed
   Stage-1 ban/pick artifacts — no bootstrapping of anything.**
   ``ordinal_logit_model.json`` / ``conditional_logit_ban_model.json`` /
   ``conditional_logit_pick_model.json`` are loaded exactly as
   ``drivers/evaluate_series.py::_load_fitted_models`` does (copied
   here per the repo's convention that each ``evaluate_*.py`` driver
   independently duplicates this small helper — the same choice M36's
   driver made and flagged; a shared loading spot is a flagged future
   refactor, out of scope) and used unchanged for every one of the
   ``n_samples`` draws for every held-out series.
3. **Primary spread metric: unweighted percentile bands across the
   ``n_samples`` draws, per outcome category** (assumption 3). Each
   ancestral draw is already sampled proportional to its own
   ``sequence_probability`` by construction, so treating the
   ``n_samples`` scoreline vectors as an *unweighted* empirical sample
   and taking ``[5th, 95th]``-style bands per category via
   :func:`evaluation.veto_conditional_variance.unweighted_scoreline_spread`
   is the standard Monte-Carlo summary — the weighting has already
   happened at draw time. Default ``ci_level = 0.90``, exposed as
   ``--ci-level``.
4. **Secondary, explicitly-flagged metric: weighted mean/variance per
   category using M31's own normalized ``weight`` field** (assumption
   4), via
   :func:`evaluation.veto_conditional_variance.weighted_mean_and_variance`,
   reported alongside the primary unweighted bands so REVIEW sees both
   conventions side by side rather than one being silently chosen. No
   full weighted-quantile function (out of scope for an S task).
5. **Series-level only, per the roadmap's literal wording ("the series
   distribution")** (assumption 5). No per-map structural-variance
   number is computed here (that would overlap M34's stage-isolation
   report in spirit); per-map structural spread is a future milestone.
6. **Determinism / single sequential rng** (assumption 9, see the
   contrast note above).
7. **Bo1/Bo5 sparsity is expected, not a bug** (assumption 10). v1's
   held-out test split is 100% Bo3; the real artifact will legitimately
   have no ``"Bo5"`` group. All per-``best_of`` aggregation and summary
   logging is guarded with ``in`` / ``.get`` / ``setdefault`` checks,
   never a bare key index.
8. **Artifact scope: the test split only** (assumption 11). No general
   ``predict()``-callable "structural spread for one arbitrary future
   match" entry point is built here (that is M39's job); the reusable
   pure statistics helpers in
   :mod:`evaluation.veto_conditional_variance` are written generically
   enough (plain matrices/weight vectors in, bands/moments out) that
   M39 can call them directly later.

**The ``--n-samples`` default is a measured wall-clock choice, not a
guess** (assumption 8). Prior tasks measured roughly 1.1-1.8 s per
sampled veto sequence per series on real v1 fitted models (each sample
scores every played map through ``build_feature_vector`` +
``predict_proba`` plus one veto walk). BUILD timed one real run first
(mirroring the "probe timed, then choose default from measurement"
precedent of tasks 036/037/038/039) and picked the largest default that
keeps the full 15-series v1 test-split run comfortably under roughly 15
minutes; the measured per-draw cost and the chosen default are recorded
in the ``tasks/040`` BUILD status note. Measured on real v1 fitted
models: ~1.5 s per sampled veto sequence per series (a 15-series probe
at ``n_samples=2`` took 45 s of wall clock), so
:data:`DEFAULT_N_SAMPLES` (30) lands the full 15-series run around
11-12 minutes — within the plan's ~15-minute budget (the plan's
starting estimate of 30 draws/series × 15 series × ~1.5s ≈ 11 minutes
was confirmed by the measurement, not revised).

**Prerequisite artifacts.** This driver depends on
``drivers/train_ordinal_logit.py``, ``drivers/train_conditional_logit_ban.py``
and ``drivers/train_conditional_logit_pick.py`` having already been
run for the requested version (three artifacts, mirroring
``drivers/evaluate_series.py``'s prerequisite pattern). A missing
artifact raises ``FileNotFoundError`` unchanged — no silent fallback,
the same "run the training driver first" doctrine every existing
``evaluate_*.py`` driver follows.

Artifact written per run (scoped by dataset version):

- ``data/<version>/veto_conditional_variance_report.json`` — a dict
  with keys ``"interval_definition"`` (a plain-string restatement of
  the "marginal, independent-per-category percentile bands over
  ancestral veto draws, not a joint simplex region, and not to be
  confused with M36's epistemic bands" caveat), ``"config"``
  (``n_samples``, ``seed``, ``ci_level``), ``"per_series"`` (a list,
  one entry per held-out test series: the identifying columns,
  ``outcome_order`` (the ``(a_wins, b_wins)`` scoreline vocabulary in
  canonical order), ``point_estimate`` (the M31 weighted-average
  aggregate — the same number ``evaluate_series.py`` reports, for
  context alongside the spread), the unweighted per-category bands
  (``unweighted_band_low`` / ``unweighted_band_high``), their
  ``band_widths`` and ``mean_band_width`` (the per-series scalar
  headline), and the ``weighted_mean`` / ``weighted_variance``
  secondary metrics) and ``"aggregate_by_best_of"`` (a dict keyed by
  ``"Bo<N>"``: each group's series count and the mean of that group's
  ``mean_band_width`` values — the single headline "how much does the
  veto sequence move the series outcome" number per ``best_of``).
  Written with ``json.dumps(..., indent=2, sort_keys=True)`` plus a
  trailing newline (the repo-wide convention).

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
    series_evaluation,
    veto_conditional_variance,
    veto_marginalized_series,
)
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The default M29 veto walks sampled per held-out series and the two
# spread-knob defaults. DEFAULT_N_SAMPLES is the measured wall-clock
# compromise described in the module docstring (assumption 8): prior
# tasks measured roughly 1.1-1.8 s per sampled veto sequence per series
# on real v1 fitted models; BUILD timed a real probe run first and
# picked the largest default keeping the full 15-series v1 test-split
# run comfortably under ~15 minutes (see the tasks/040 BUILD status
# note for the measured per-draw cost). DEFAULT_SEED matches the repo's
# "current year" seed convention already used by drivers/evaluate_series.py;
# DEFAULT_CI_LEVEL mirrors M36's 0.90 convention (5th/95th percentile
# bands).
DEFAULT_N_SAMPLES = 30
DEFAULT_SEED = 2026
DEFAULT_CI_LEVEL = 0.90


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
    duplicates this helper (the same choice
    ``drivers/evaluate_bootstrap_intervals.py`` /
    ``drivers/evaluate_compounding_diagnostics.py`` made and flagged —
    flagged here too: a shared loading spot would be a refactor
    opportunity, out of scope for this task). All three artifacts are
    used **unchanged** for every one of the ``n_samples`` draws for
    every held-out series (the M37 plan's assumption 2 — fixed models,
    varying veto rng, no bootstrapping anywhere). The three
    ``from_dict`` calls are deliberately independent of each other and
    of the four input tables, so a missing artifact fails fast with the
    standard "run the training driver first" signal.

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


def _series_spread_record(
    prediction: veto_marginalized_series.VetoMarginalizedSeriesPrediction,
    ci_level: float,
) -> dict:
    """Summarize one M31 prediction's per-sample spread into a record dict.

    Builds the ``(n_samples, best_of + 1)`` sample-row matrix from
    ``prediction.samples[i].scoreline_probabilities`` and the parallel
    weight vector from ``prediction.samples[i].weight`` (the exact
    deterministic per-sample scoreline detail M31 already returns —
    assumption 1), then computes the M37 summary via the pure helpers
    in :mod:`evaluation.veto_conditional_variance`: the unweighted
    per-category percentile bands
    (:func:`evaluation.veto_conditional_variance.unweighted_scoreline_spread`),
    their widths
    (:func:`evaluation.veto_conditional_variance.band_widths`), the
    per-series scalar mean width
    (:func:`evaluation.veto_conditional_variance.mean_band_width`),
    and the explicitly-flagged secondary weighted mean/variance
    (:func:`evaluation.veto_conditional_variance.weighted_mean_and_variance`).
    The point estimate (``prediction.probabilities``) is *not* included
    here — the caller attaches it alongside the identifying columns.

    Args:
        prediction: The M31 prediction whose per-sample detail is
            summarized; must carry ``best_of + 1``-length
            ``scoreline_probabilities`` per sample and one ``weight``
            per sample (as :func:`evaluation.veto_marginalized_series
            .predict_series_outcome_via_veto_marginalization` returns).
        ci_level: The band level in ``(0, 1)``, passed through to
            :func:`evaluation.veto_conditional_variance
            .unweighted_scoreline_spread` (the driver's ``--ci-level``).

    Returns:
        A dict with keys ``unweighted_band_low`` (the per-category band
        lows), ``unweighted_band_high`` (the per-category band highs),
        ``band_widths`` (the per-category ``hi - lo`` widths),
        ``mean_band_width`` (their mean — the per-series scalar
        headline; exactly ``0.0`` when every sampled veto sequence
        produced the identical scoreline distribution), ``weighted_mean``
        and ``weighted_variance`` (the per-category weighted first /
        second moments using the samples' own normalized weights).
        Every value is JSON-serializable.

    Raises:
        ValueError: If a sample's scoreline vector length is
            inconsistent with the others, if ``n_samples`` is zero, or
            if the weight vector mismatches the rows / is malformed
            (all propagated from the pure helpers in
            :mod:`evaluation.veto_conditional_variance`).
    """
    rows = [
        list(sample.scoreline_probabilities)
        for sample in prediction.samples
    ]
    weights = [float(sample.weight) for sample in prediction.samples]
    bands = veto_conditional_variance.unweighted_scoreline_spread(
        rows, ci_level=ci_level
    )
    widths = veto_conditional_variance.band_widths(bands)
    weighted_means, weighted_variances = (
        veto_conditional_variance.weighted_mean_and_variance(
            rows, weights
        )
    )
    return {
        "unweighted_band_low": [lo for lo, _hi in bands],
        "unweighted_band_high": [hi for _lo, hi in bands],
        "band_widths": list(widths),
        "mean_band_width": veto_conditional_variance.mean_band_width(bands),
        "weighted_mean": list(weighted_means),
        "weighted_variance": list(weighted_variances),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_veto_conditional_variance.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with five attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and the
        three M37 reproducibility/spread knobs from the plan's
        assumptions 3/8/9: ``n_samples`` (``int``, the M29 walks
        sampled per held-out series — larger than
        ``evaluate_series.py``'s ``DEFAULT_N_SAMPLES = 10`` because a
        stable *spread* estimate needs more draws than a stable *mean*
        estimate; default :data:`DEFAULT_N_SAMPLES`, the measured
        wall-clock choice documented in the module docstring), ``seed``
        (``int``, the single ``numpy.random.default_rng`` seed consumed
        sequentially across every held-out series — deliberately NOT
        the fixed-per-call-reset rng M36 uses; default
        :data:`DEFAULT_SEED`), and ``ci_level`` (``float``, the
        unweighted percentile-band level in ``(0, 1)``, default
        :data:`DEFAULT_CI_LEVEL`). Together they locate the four input
        tables, the three fitted model artifacts, and the output
        artifact
        ``<output_dir>/<version>/veto_conditional_variance_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-samples``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M37 veto-conditional variance report: for each "
            "held-out test-split series, call the M31 veto-marginalized "
            "pipeline directly with a larger n_samples (fixed fitted "
            "Stage-1/Stage-2 models — no bootstrapping, the veto rng "
            "varies naturally) and summarize the per-sample scoreline "
            "spread across the sampled veto sequences: unweighted "
            "per-category percentile bands as the primary metric plus "
            "weighted mean/variance as a flagged secondary metric; "
            "write veto_conditional_variance_report.json."
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
            "M29 veto sequences sampled per held-out series (default: "
            f"{DEFAULT_N_SAMPLES} — a measured wall-clock choice on "
            "real v1 fitted models: ~1.5s per sampled sequence per "
            "series, so the full 15-series v1 test split takes ~11-12 "
            "minutes; sized larger than evaluate_series.py's 10 because "
            "a stable spread estimate needs more draws than a stable "
            "mean estimate; raise it for smoother percentile bands, "
            "lower it for a faster run)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, giving the whole run "
            f"byte-identical reproducibility (default: {DEFAULT_SEED}) — "
            "one rng consumed sequentially across every held-out series; "
            "deliberately NOT the fixed-per-call-reset rng M36 uses, "
            "since M37 measures exactly the veto-sequence variation that "
            "reset would suppress"
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help=(
            "unweighted percentile-band level in (0, 1): each "
            "per-category band spans the middle ci_level fraction of "
            f"the sampled veto draws (default: {DEFAULT_CI_LEVEL} — "
            "5th/95th percentiles)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the veto-conditional variance report end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The three fitted model artifacts are loaded first
    (:func:`_load_fitted_models` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training drivers first"
    signal), the four input tables are loaded for the requested version
    (matches/maps/splits/player_map_stats via the ``drivers.evaluate``
    helpers), the flag values are validated (positive ``n_samples``,
    ``ci_level`` in ``(0, 1)``), the fixed pluggable callables are
    wired per the M37 plan's assumption 2 (``map_model_fn`` =
    :func:`models.ordinal_logit.make_model_fn` over the fitted ordinal
    model + ``player_map_stats.parquet``; ``predictor_fn_by_action`` =
    the ban/pick conditional-logit ``make_veto_step_predictor_fn``
    closures — both loaded once and used unchanged for every draw,
    no bootstrapping), and **one** ``numpy.random.default_rng(args.seed)``
    is constructed. The held-out test-split series table is built once
    (:func:`evaluation.series_evaluation.build_held_out_series`), and
    each row is predicted via
    :func:`evaluation.veto_marginalized_series
    .predict_series_outcome_via_veto_marginalization` **directly** (not
    the closure adapter, which throws the per-sample detail away) with
    ``n_samples=args.n_samples`` and the *same* ``rng`` object passed
    to every call — consumed sequentially across the ``n_samples``
    draws within a series and across the different series (assumption
    9: natural, un-reset veto variation; a fixed ``--seed`` reproduces
    the whole run byte-identically). Each prediction's per-sample
    scoreline detail is summarized via :func:`_series_spread_record`
    and recorded per series alongside the identifying columns, the
    ``outcome_order`` vocabulary, and the point estimate. An aggregate
    is computed per ``best_of`` group (guarded ``setdefault``, never a
    bare key index — the real v1 report legitimately has no Bo5 group,
    assumption 10): the mean of that group's ``mean_band_width`` values
    (the headline "how much does the veto sequence move the series
    outcome" number per ``best_of``). The combined report
    (``interval_definition`` caveat string, the ``config`` block, the
    ``per_series`` list and the ``aggregate_by_best_of`` block) is
    written as
    ``<output_dir>/<version>/veto_conditional_variance_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: the series count, the
    resolved knobs, and per-``best_of`` the series count plus mean of
    ``mean_band_width`` (the headline finding).

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
        ValueError: If ``n_samples`` is not positive or ``ci_level`` is
            not in ``(0, 1)``; if the test split's held-out series set
            is empty or malformed (from
            :func:`evaluation.series_evaluation.build_held_out_series`);
            if the M31 pipeline rejects any input (propagated from
            :mod:`evaluation.veto_marginalized_series`, including a
            degenerate all-zero-probability sample set); or if the
            spread helpers reject malformed sample/weight collections
            (from :mod:`evaluation.veto_conditional_variance`).
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

    if args.n_samples < 1:
        raise ValueError(
            f"--n-samples must be a positive integer, got {args.n_samples}"
        )
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError(
            f"--ci-level must be strictly between 0 and 1, got "
            f"{args.ci_level}"
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

    # The fixed Stage-1/Stage-2 callables (assumption 2): loaded once,
    # used unchanged for every draw of every held-out series — no
    # bootstrapping anywhere.
    map_model_fn = ordinal_logit.make_model_fn(
        ordinal_model, player_map_stats_df
    )
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    # ONE rng, consumed sequentially (assumption 9 — the M36-contrast
    # mechanism): never reconstructed per series, never reset.
    rng = np.random.default_rng(args.seed)

    held_out = series_evaluation.build_held_out_series(
        matches_df, maps_df, splits_df, split="test"
    )

    per_series: list[dict] = []
    for row in held_out.itertuples(index=False):
        prediction = (
            veto_marginalized_series.predict_series_outcome_via_veto_marginalization(
                row.team1_id,
                row.team2_id,
                row.best_of,
                row.date,
                matches_df,
                maps_df,
                map_model_fn,
                predictor_fn_by_action,
                n_samples=args.n_samples,
                rng=rng,
            )
        )
        spread = _series_spread_record(prediction, ci_level=args.ci_level)
        per_series.append(
            {
                "match_id": row.match_id,
                "date": row.date,
                "team1_id": row.team1_id,
                "team2_id": row.team2_id,
                "best_of": row.best_of,
                "best_of_int": prediction.best_of,
                "a_wins": row.a_wins,
                "b_wins": row.b_wins,
                "outcome_index": row.outcome_index,
                "outcome_order": [
                    list(scoreline) for scoreline in prediction.outcome_order
                ],
                "point_estimate": list(prediction.probabilities),
                **spread,
            }
        )

    # Per-best_of aggregate (assumption 10: guarded setdefault, never a
    # bare key index — the real v1 report legitimately has no Bo5
    # group). The headline per group: the mean of that group's
    # mean_band_width values.
    group_widths: dict[str, list[float]] = {}
    for entry in per_series:
        group_widths.setdefault(entry["best_of"], []).append(
            entry["mean_band_width"]
        )
    aggregate_by_best_of = {
        best_of: {
            "n_series": len(widths),
            "mean_mean_band_width": float(np.mean(widths)),
        }
        for best_of, widths in sorted(group_widths.items())
    }

    report = {
        "interval_definition": (
            "marginal, independent-per-category percentile bands over "
            "the sampled ancestral veto draws (NOT a joint simplex "
            "credible region, and NOT to be confused with M36's "
            "epistemic bootstrap bands): each category's band is the "
            "[low_percentile, high_percentile] of that category's "
            "probability across the n_samples veto draws, computed "
            "independently per category; models are fixed (no "
            "bootstrapping), so this spread is purely structural — it "
            "resolves the moment the veto happens"
        ),
        "config": {
            "n_samples": args.n_samples,
            "seed": args.seed,
            "ci_level": args.ci_level,
        },
        "per_series": per_series,
        "aggregate_by_best_of": aggregate_by_best_of,
    }

    artifact_path = (
        output_dir / args.version / "veto_conditional_variance_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # One-line summary: per-best_of series count and mean of the group's
    # mean_band_width (the headline structural-spread number), guarded
    # for the real v1 report's legitimate Bo5 absence.
    summary = "; ".join(
        (
            f"{best_of}: {agg['n_series']} series, "
            f"mean mean_band_width={agg['mean_mean_band_width']:.6f}"
        )
        for best_of, agg in aggregate_by_best_of.items()
    )
    logger.info(
        "veto-conditional variance on %d held-out test-split series "
        "(%s/%s, n_samples=%d seed=%d ci_level=%.2f): %s",
        len(per_series),
        output_dir,
        args.version,
        args.n_samples,
        args.seed,
        args.ci_level,
        summary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compounding diagnostics over the deployed two-stage pipeline (roadmap M35).

Two cheap checks on the assumptions Stage 3 rests on, implemented as a
pure, table-in analysis module: given already-scored DataFrames in, it
returns JSON-serializable report dicts out, with no file I/O and no
model or harness imports.

Architecture decision (recorded verbatim from the task plan): this
module is **schema-contract, not reimplementation**. It does not build
its own held-out tables and does not import any sibling
``evaluation/`` module: the two report functions take DataFrames
already shaped like ``evaluation.series_evaluation.SCORED_SERIES_COLUMNS``
(the sweep-rate diagnostic) and ``evaluation.harness.SCORED_COLUMNS``
(the map1-predicts-map2 diagnostic) by convention, documented via the
column-name constants below. There is no *logic* to duplicate here —
only a column-name contract — so a local reimplementation of
``build_held_out_maps`` / ``build_held_out_series`` (the pattern M34
used for genuinely different logic) would be lateral-import-rule
theater: the joins are identical to the existing precedent with no
variation. The ``drivers/evaluate_compounding_diagnostics.py`` CLI
driver does the actual assembly (it may import ``evaluation/`` /
``models/`` modules freely) and hands both scored tables in. This
module's only imports are ``utils.series_paths`` (the generic sweep-
index vocabulary — a genuine downward dependency) plus
``numpy``/``pandas``; it has no ``models/``/``features/`` dependency
at all.

Diagnostic 1 — predicted vs observed sweep rate. A "sweep" is the
series ending at the maximum possible dominance for either side:
scoreline index ``0`` (``(threshold, 0)``, side A sweeps) or index
``K - 1`` (``(0, threshold)``, side B sweeps) in
``utils.series_paths.series_outcome_order(best_of_int)``, where
``threshold = series_win_threshold(best_of_int)`` and ``K =
2 * threshold = best_of_int + 1``. For Bo3 this is exactly "2-0"
(either direction); for Bo5, "3-0". The two sweep indices are computed
generically per row from that row's own ``best_of_int`` via
:func:`utils.series_paths.series_win_threshold`, never hardcoded to a
format. Bo1 is a degenerate case (``K = 2``, both outcomes are
trivially "sweeps") — :func:`sweep_rate_report` still computes it if a
Bo1 group is ever present (defensive correctness), but its numbers are
a tautology (predicted and observed sweep rate are both trivially the
marginal match-win rate) and are informationally meaningless; v1 has no
Bo1 matches in ``matches.parquet``, so this is moot in practice.
A positive ``sweep_calibration_gap`` (predicted mean sweep probability
minus observed sweep rate) is Stage 2 / the pipeline being
over-confident about sweeps relative to what actually happens — exactly
the roadmap's "2-0 / 3-0 probabilities are products of same-direction
terms, so they expose Stage 2 overconfidence first" framing. A
directional (per-side) breakdown is reported alongside the combined
headline, because overconfidence could plausibly be asymmetric (e.g.
one side's favorite status compounds more than the other's). No
significance test is attached to this diagnostic — it is a calibration
point-estimate, and the tiny ``n_eval`` at v1 scale (15 held-out Bo3
matches, 0 Bo5) makes any confidence interval on a binomial rate wide
enough that a formal test would overstate precision; this power caveat
is stated here, not silently glossed over.

Diagnostic 2 — does map 1's outcome predict map 2's, beyond what
Stage 2's features explain? The central operationalization (a
statistical judgment call, flagged for REVIEW in the task plan):
Stage 2 (``models.ordinal_logit``'s fitted model) produces
``predicted_p_a_map2 = p_a_regulation + p_a_ot`` for map 2 of a
series, computed from features that are all as-of the *match* date —
per the task-037/038 context reads, this prediction is identical
regardless of map 1's actual result, because no feature channel exists
for it to differ (the per-map ``model_fn`` is called with the same
match-level ``date`` for every map in a series, and no feature module
conditions any as-of computation on which map index is being queried;
this is the load-bearing fact: Stage 2's prediction for map 2 cannot
see map 1's actual outcome through any feature, so any residual
correlation found between map 1's actual result and map 2's prediction
error is genuine unexplained dependence, not a leakage artifact).
"Beyond what features explain" is therefore operationalized as: does
the actual map-1 outcome correlate with Stage 2's *residual* on map 2
(``actual_a_won_map2 (0/1) - predicted_p_a_map2``)? If Stage 2's
features fully captured whatever drives map-2 outcomes, the residual
should be uncorrelated with map-1's result (which the features never
saw); a real correlation means there is compounding information across
maps within a series (e.g. momentum, a map-pool/pick-order effect, or
simply "team A is having a good day") that a per-map,
pre-series-features-only model structurally cannot capture. This is
explicitly a test of *this deployed Stage-2 model's* explanatory
completeness, not a claim about whether *some* richer feature set
could explain the dependency.

Statistical stance (mirroring ``evaluation/proportional_odds.py``'s
honesty convention): this repo has no ``scipy``, so no module
fabricates a parametric p-value it cannot actually compute correctly.
:func:`map1_predicts_map2_report` implements a hand-rolled **empirical
permutation p-value** — pure ``numpy`` shuffle-and-recompute over
random relabelings of the map-1-outcome labels holding the map-2
residuals fixed — clearly labeled for exactly what it is, not the
classical two-sample parametric test. The permutation draw uses a
caller-supplied ``numpy.random.Generator`` (no default global seed,
mirroring M29's decision 4 convention).

Power caveat (stated, not hidden): at v1 scale (15 eligible Bo3
matches) both diagnostics have very low power — a non-significant
``p_value_empirical`` must be read as "undetectable at this sample
size," not "no effect exists." A cheap check that is loud about its
own power limits, exactly as the roadmap's "cheap checks" framing
implies.

Place in the dependency DAG: ``utils/ -> features/ -> models/ ->
evaluation/ -> drivers/``. This module sits in ``evaluation/`` and may
depend downward on ``utils.*`` only — concretely ``utils.series_paths``
for the sweep-index vocabulary — never on ``drivers/``, never on a
sibling ``evaluation/`` module, and (per the architecture decision)
not even on ``models/``/``features/`` (the upstream scored tables
already embody them).
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

from utils import series_paths

# The columns :func:`sweep_rate_report` reads from its input table —
# the schema contract with
# ``evaluation.series_evaluation.score_held_out_series``'s
# SCORED_SERIES_COLUMNS output (documented by name, not imported:
# this module has no sibling ``evaluation/`` import). ``best_of`` is
# the "Bo<N>" string, ``best_of_int`` its parsed integer, and
# ``probabilities`` the variable-length ``best_of_int + 1`` scoreline
# vector in ``utils.series_paths.series_outcome_order`` order.
SWEEP_RATE_COLUMNS = (
    "match_id",
    "best_of",
    "best_of_int",
    "probabilities",
    "outcome_index",
)

# The columns :func:`map1_predicts_map2_report` reads from its input
# table — the schema contract with
# ``evaluation.harness.score_held_out_maps``'s SCORED_COLUMNS output
# (documented by name, not imported). ``outcome_ordinal`` is the true
# four-way outcome ordinal (0 = A-regulation, 1 = A-OT, 2 = B-OT,
# 3 = B-regulation) and ``p_a_regulation`` / ``p_a_ot`` are the two
# side-A prediction columns.
MAP1_PREDICTS_MAP2_COLUMNS = (
    "match_id",
    "map_index",
    "outcome_ordinal",
    "p_a_regulation",
    "p_a_ot",
)


def sweep_rate_report(scored_series_df: pd.DataFrame) -> dict:
    """Build the per-``best_of`` predicted-vs-observed sweep-rate report.

    A pure dict builder (no I/O): turns the scored series table from
    ``evaluation.series_evaluation.score_held_out_series`` into the
    sweep-calibration report diagnostic 1 asks for. Grouped per
    distinct ``best_of`` value present in ``scored_series_df``
    (mirroring ``evaluation.series_evaluation.build_series_evaluation_report``'s
    grouping precedent — a ``best_of`` group with zero rows is simply
    absent from the input, not specially handled). Within each group,
    each row's ``probabilities`` vector (length ``best_of_int + 1`` in
    ``utils.series_paths.series_outcome_order`` order) contributes its
    combined sweep probability ``probabilities[0] +
    probabilities[K - 1]`` and its directional sweep probabilities
    ``probabilities[0]`` (side A) / ``probabilities[K - 1]`` (side B),
    where ``K = 2 * series_win_threshold(best_of_int)``; the row's
    ``outcome_index`` is a sweep if it is ``0`` (A sweeps) or ``K - 1``
    (B sweeps).

    Reported per group: ``n_eval`` (row count), the combined
    ``predicted_mean_sweep_prob`` / ``observed_sweep_rate`` /
    ``sweep_calibration_gap`` (gap = predicted minus observed; a
    positive gap is over-confidence about sweeps, the roadmap's
    "expose Stage 2 overconfidence first" signal), and the two
    directional mirrors ``predicted_mean_a_sweep_prob`` /
    ``observed_a_sweep_rate`` / ``a_sweep_calibration_gap`` (side A
    only: ``probabilities[0]`` / ``outcome_index == 0``) and
    ``*_b_sweep_*`` (side B only: ``probabilities[K - 1]`` /
    ``outcome_index == K - 1``). A combined ``n_eval_total`` count
    spans groups (mirroring the M33a precedent), but no
    cross-``K`` blended sweep number: a Bo3 sweep probability and a
    Bo5 sweep probability are per-format statements and must not be
    averaged together.

    Validation: ``ValueError`` if ``scored_series_df`` is empty, or if
    any row's ``probabilities`` length does not match
    ``2 * series_win_threshold(row's best_of_int)`` — an internal
    desync with the M33a contract that should never happen with real
    ``score_held_out_series`` output, but is checked defensively
    rather than trusting the input shape blindly (the repo's
    validate-then-trust convention).

    Args:
        scored_series_df: The scored series table from
            ``evaluation.series_evaluation.score_held_out_series``
            (needs :data:`SWEEP_RATE_COLUMNS`; the per-row
            ``rps``/``log_loss``/``marginal_correct`` columns are not
            read).

    Returns:
        A dict with one key per distinct ``best_of`` value present
        (each holding the combined + directional sweep fields listed
        above, all plain floats/int) plus ``n_eval_total`` (int, the
        full row count). Every value is a plain str/int/float/dict, so
        the whole dict is directly ``json.dumps``-serializable.

    Raises:
        ValueError: If ``scored_series_df`` is empty (a mean over zero
            series is undefined); or if any row's ``probabilities``
            column has a length other than
            ``2 * series_win_threshold(row's best_of_int)``, naming
            the offending series' ``match_id``/``best_of``.
        KeyError: If ``scored_series_df`` lacks a
            :data:`SWEEP_RATE_COLUMNS` column (propagated from
            pandas column indexing).
        TypeError: If a row's ``probabilities`` entry is not iterable
            (e.g. a malformed scalar — ``list(row.probabilities)``
            fails), propagated unchanged.
    """
    if len(scored_series_df) == 0:
        raise ValueError(
            "cannot build a sweep-rate report over zero scored series"
        )
    report: dict = {}
    for best_of in sorted(scored_series_df["best_of"].unique()):
        subset = scored_series_df[scored_series_df["best_of"] == best_of]
        n = len(subset)
        combined_pred_sum = 0.0
        a_pred_sum = 0.0
        b_pred_sum = 0.0
        n_a_sweep = 0
        n_b_sweep = 0
        for row in subset.itertuples(index=False):
            threshold = series_paths.series_win_threshold(row.best_of_int)
            k = 2 * threshold
            probs = list(row.probabilities)
            if len(probs) != k:
                raise ValueError(
                    f"series (match {row.match_id!r}, best_of "
                    f"{row.best_of!r}) has a probabilities vector of "
                    f"length {len(probs)}; expected exactly "
                    f"{k} = 2 * series_win_threshold("
                    f"{row.best_of_int}) entries"
                )
            combined_pred_sum += probs[0] + probs[k - 1]
            a_pred_sum += probs[0]
            b_pred_sum += probs[k - 1]
            if row.outcome_index == 0:
                n_a_sweep += 1
            if row.outcome_index == k - 1:
                n_b_sweep += 1
        combined_predicted = combined_pred_sum / n
        combined_observed = (n_a_sweep + n_b_sweep) / n
        a_predicted = a_pred_sum / n
        a_observed = n_a_sweep / n
        b_predicted = b_pred_sum / n
        b_observed = n_b_sweep / n
        report[str(best_of)] = {
            "n_eval": n,
            "predicted_mean_sweep_prob": combined_predicted,
            "observed_sweep_rate": combined_observed,
            "sweep_calibration_gap": combined_predicted - combined_observed,
            "predicted_mean_a_sweep_prob": a_predicted,
            "observed_a_sweep_rate": a_observed,
            "a_sweep_calibration_gap": a_predicted - a_observed,
            "predicted_mean_b_sweep_prob": b_predicted,
            "observed_b_sweep_rate": b_observed,
            "b_sweep_calibration_gap": b_predicted - b_observed,
        }
    report["n_eval_total"] = len(scored_series_df)
    return report


def map1_predicts_map2_report(
    scored_maps_df: pd.DataFrame,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict:
    """Test whether map 1's outcome predicts map 2's Stage-2 residual.

    Diagnostic 2 of M35, implemented per the module docstring's
    operationalization: restrict ``scored_maps_df`` to matches where
    both a ``map_index == 0`` and a ``map_index == 1`` row are present
    (inner join of the two subsets on ``match_id`` — every Bo3+ match
    qualifies by construction; Bo1 matches have no map 2 at all and are
    excluded structurally by the join), then per qualifying match
    compute ``map1_a_won = outcome_ordinal_map1 in {0, 1}``
    (A-regulation/A-OT), ``predicted_p_a_map2 = p_a_regulation_map2 +
    p_a_ot_map2``, ``actual_a_won_map2 = int(outcome_ordinal_map2 in
    {0, 1})``, and ``residual_map2 = actual_a_won_map2 -
    predicted_p_a_map2``. The observed test statistic is
    ``observed_diff = mean(residual_map2 | map1_a_won) -
    mean(residual_map2 | not map1_a_won)`` — the group-mean residual
    difference between series whose map 1 side A won and those whose
    map 1 side B won. If Stage 2's features fully captured whatever
    drives map-2 outcomes, the residual should be uncorrelated with
    map-1's result (which the features never saw), so a large
    ``|observed_diff|`` is genuine unexplained compounding dependence.

    The significance claim is an **empirical permutation p-value, not
    a parametric test** (stated verbatim, mirroring
    ``evaluation/proportional_odds.py``'s own "this is not the
    classical chi-square" stance): draw ``n_permutations`` random
    relabelings of the ``map1_a_won`` labels across the qualifying
    matches (via the caller-supplied ``rng`` — no default global seed,
    mirroring M29's decision 4 convention), recompute the group-mean
    difference for each relabeling holding ``residual_map2`` fixed
    (permuting labels preserves the two group sizes exactly, so the
    test conditions on the observed A/B subgroup counts), and report
    ``p_value_empirical = mean(|permuted_diff| >= |observed_diff|)``
    (two-sided). This needs no ``scipy``; it is pure ``numpy``
    shuffle-and-recompute, consistent with the repo-wide "no scipy"
    constraint.

    Explicit limitation, stated rather than hidden: at v1 scale (n=15
    eligible Bo3 matches) this test has very low power — a
    non-significant ``p_value_empirical`` should be read as
    "undetectable at this sample size," not "no effect exists."

    Degenerate-input doctrine (mirroring the repo's fail-loud
    convention): ``ValueError`` if fewer than 2 matches qualify in
    total, or if either the map-1-A-won or map-1-B-won subgroup is
    empty (a group mean over zero rows is undefined — this is a real,
    expected possibility at v1's n=15 scale, not defensive theater);
    ``ValueError`` if ``n_permutations`` is not a positive integer.

    Args:
        scored_maps_df: The scored map table from
            ``evaluation.harness.score_held_out_maps`` (needs
            :data:`MAP1_PREDICTS_MAP2_COLUMNS`; the remaining
            ``SCORED_COLUMNS`` columns are not read). One row per
            ``(match_id, map_index)``, per the M19 contract.
        rng: The ``numpy.random.Generator`` driving the permutation
            draw; supplied by the caller (the driver constructs one
            from ``--permutation-seed``), never a global default.
        n_permutations: How many random relabelings to draw for the
            permutation p-value. Any integer-like value (plain int or
            numpy integer scalar) is coerced via ``operator.index``;
            must be at least 1 (a single permutation — the identity
            relabeling — yields ``p_value_empirical`` of either 1.0 or
            0.0, degenerate but well-defined; callers choose a large
            value such as 10000 for a real run).

    Returns:
        A dict with ``n_eligible_matches`` (int), ``n_map1_a_won``
        (int), ``n_map1_b_won`` (int),
        ``mean_residual_given_map1_a_won`` (float),
        ``mean_residual_given_map1_b_won`` (float), ``observed_diff``
        (float, ``mean_residual_given_map1_a_won -
        mean_residual_given_map1_b_won``), ``n_permutations`` (int)
        and ``p_value_empirical`` (float in ``(0.0, 1.0]`` — the
        identity relabeling always reproduces ``observed_diff`` exactly,
        so the empirical p-value is never 0.0). Every value is a plain
        str/int/float, so the whole dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If fewer than 2 matches have both a map-1 and a
            map-2 row (the inner join produced fewer than 2 rows); if
            either the map-1-A-won or map-1-B-won subgroup is empty
            (cannot form a group mean); or if ``n_permutations`` is not
            a positive integer-like value.
        KeyError: If ``scored_maps_df`` lacks a
            :data:`MAP1_PREDICTS_MAP2_COLUMNS` column (propagated
            from pandas column indexing / the merge).
        AttributeError: If ``rng`` does not provide a
            ``numpy.random.Generator``-style ``permutation`` method
            (propagated from the draw loop).
    """
    try:
        n_perms = operator.index(n_permutations)
    except TypeError as exc:
        raise ValueError(
            f"n_permutations must be a positive integer, got "
            f"{n_permutations!r}"
        ) from exc
    if n_perms < 1:
        raise ValueError(
            "n_permutations must be at least 1 (a permutation test "
            f"needs at least one relabeling), got {n_perms}"
        )

    # Inner join of the map-1 subset and the map-2 subset on match_id:
    # a match qualifies for this diagnostic iff both rows exist (every
    # Bo3+ match by construction; Bo1 matches are excluded structurally).
    map1 = scored_maps_df[scored_maps_df["map_index"] == 0]
    map2 = scored_maps_df[scored_maps_df["map_index"] == 1]
    merged = map1[["match_id", "outcome_ordinal"]].merge(
        map2[
            ["match_id", "p_a_regulation", "p_a_ot", "outcome_ordinal"]
        ],
        on="match_id",
        suffixes=("_map1", "_map2"),
    )
    n_eligible = len(merged)
    if n_eligible < 2:
        raise ValueError(
            "map1_predicts_map2 needs at least 2 matches with both a "
            f"map-index-0 and a map-index-1 row, got {n_eligible}"
        )

    map1_a_won = merged["outcome_ordinal_map1"].isin((0, 1)).to_numpy()
    # p_a_regulation/p_a_ot exist only in the right (map-2) frame, so
    # the merge suffixes leave them unsuffixed — only the overlapping
    # outcome_ordinal column gained the _map1/_map2 suffixes.
    predicted_p_a_map2 = merged["p_a_regulation"] + merged["p_a_ot"]
    actual_a_won_map2 = (
        merged["outcome_ordinal_map2"].isin((0, 1)).to_numpy(dtype=float)
    )
    residual_map2 = actual_a_won_map2 - predicted_p_a_map2.to_numpy(
        dtype=float
    )

    n_a = int(map1_a_won.sum())
    n_b = int((~map1_a_won).sum())
    if n_a == 0 or n_b == 0:
        raise ValueError(
            "map1_predicts_map2 needs at least one map-1 outcome in each "
            f"direction to form both group means: got n_map1_a_won={n_a}, "
            f"n_map1_b_won={n_b} across {n_eligible} eligible matches"
        )

    mean_a = float(residual_map2[map1_a_won].mean())
    mean_b = float(residual_map2[~map1_a_won].mean())
    observed_diff = mean_a - mean_b

    # Empirical permutation p-value: relabel map1_a_won (permuting
    # preserves the group sizes exactly), recompute the group-mean
    # difference holding residual_map2 fixed, and count relabelings
    # whose |difference| is at least the observed one (two-sided).
    count = 0
    for _ in range(n_perms):
        perm_labels = map1_a_won[rng.permutation(n_eligible)]
        permuted_diff = float(
            residual_map2[perm_labels].mean()
            - residual_map2[~perm_labels].mean()
        )
        if abs(permuted_diff) >= abs(observed_diff):
            count += 1

    return {
        "n_eligible_matches": n_eligible,
        "n_map1_a_won": n_a,
        "n_map1_b_won": n_b,
        "mean_residual_given_map1_a_won": mean_a,
        "mean_residual_given_map1_b_won": mean_b,
        "observed_diff": observed_diff,
        "n_permutations": n_perms,
        "p_value_empirical": count / n_perms,
    }

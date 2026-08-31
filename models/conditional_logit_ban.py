"""Conditional (McFadden) logit model for map-veto ban decisions (roadmap M27).

The first *trained* veto-stage model: a feature-based scoring function
over the remaining candidate maps with a per-step softmax
normalisation across the shrinking candidate set, trained by
cross-entropy against the observed bans of the M10 train split. It is
the trained successor M25's ``models.greedy_veto_simulator`` points
at ("the hard-argmax limit of roadmap M27"), and M26's
``evaluation.veto_evaluation`` harness is the evaluation bed it drops
into via the same :data:`evaluation.veto_evaluation.VetoStepPredictorFn`
interface shape (structurally matched here — this module must not
import ``evaluation/``).

The model is deliberately dependency-light, like its M20/M21 siblings:
no ``sklearn``/``scipy`` exists in this repo, so the ragged-group
softmax NLL, its analytic gradient, and the Armijo-backtracking
gradient-descent optimizer are all hand-rolled here. Like every
``models/`` module it is 100% I/O-free; all Parquet/JSON I/O lives in
``drivers/``.

Design decisions (recorded here, do not re-derive in later milestones;
the M27 plan's decisions 3 and 6 live in the modules they touch —
``features/closeness.py``'s new :func:`map_ot_rate` and
``evaluation/veto_evaluation.py``'s shared teacher-forced replay):

1. **This is a McFadden / conditional logit, not the M20/M21 four-way
   models.** The candidate set (remaining maps) shrinks every step and
   has no fixed identity across steps or matches, so there is no
   per-alternative intercept space to fit. An intercept — or any
   feature identical across every map in a step's candidate set —
   adds the same constant to every candidate's score and cancels
   exactly under softmax normalisation: it is mathematically inert,
   not merely a design choice to regularize away. This rules out every
   M13-M17 team-pair-level feature used by
   ``models._shared.build_feature_vector`` (``elo_differential``,
   ``h2h_win_rate_centered``, ``event_stage``, ``days_since_diff``,
   ``roster_decay_diff`` — all constant across the candidate maps of
   one step) as a *direct* input. Only features that vary **by map**,
   for a fixed (acting team, opponent, date), are legitimate softmax
   inputs. This is also the formal reading of "scored by features,
   never by map identity": there is no per-map dummy/intercept in the
   design at all, by construction, not just by omission.
2. **Feature set (5 features, all map-varying, no intercept), in the
   fixed :data:`FEATURE_NAMES` order:**
   1. ``acting_map_win_rate`` — ``features.map_win_rate
      .team_map_win_rate(acting_team_id, map_name, date, ...).mean``
      (shrunk with ``map_win_rate.DEFAULT_K``, the same documented
      default the M25 greedy arm's evaluation uses; lower means "this
      is my weak map" — a natural ban candidate).
   2. ``opponent_map_win_rate`` — the same estimator for
      ``opponent_team_id`` (higher means "this is their strong map" —
      the other classic ban motive).
   3. ``acting_map_specialization`` — ``acting_map_win_rate -
      features.map_win_rate.team_overall_win_rate(acting_team_id,
      date, ...).rate``: how much stronger/weaker this specific map is
      for the acting team relative to their own baseline. **No extra
      missing-value fallback is needed for features 1-3:** when a team
      has zero as-of games on a map, ``team_map_win_rate.mean`` equals
      ``team_overall_win_rate.rate`` exactly (full shrinkage), so the
      specialization degrades to ``0.0`` automatically, and when a team
      has zero as-of games at all, ``team_overall_win_rate.rate`` is
      exactly ``0.5`` by that function's own documented empty-history
      default. Both existing functions handle every degenerate case;
      this module adds no new imputation.
   4. ``map_round_margin_variance`` — ``features.closeness
      .map_round_margin_variance(map_name, date, ...).variance``, with
      the same NaN (``n <= 1``) -> ``0.0`` fallback
      ``models._shared.build_feature_vector`` already uses ("no
      observed-variance signal contributes no information").
   5. ``map_ot_rate`` — ``features.closeness.map_ot_rate(map_name,
      date, ...).mean``, the per-map heavily-shrunk OT rate added to
      M15's module for exactly this model (its ``mean`` is always
      finite — ``alpha + beta >= k > 0`` — so no NaN fallback is
      needed).
3. **No intercept, ever — not even a shared constant later dropped by
   softmax.** ``beta`` has exactly 5 entries (one per
   :data:`FEATURE_NAMES` feature); there is no 6th "bias" parameter,
   because decision 1 already proves any such parameter would be a free
   variable with literally zero effect on every training/prediction
   output. Do not add one "for consistency" with
   ``ordinal_logit``/``multinomial_logit`` — the standardizer's
   :func:`models._shared.apply_standardizer` still applies its usual
   ``(x - mean) / std`` column transform per feature (no confusion with
   an intercept: it recenters each *feature column*, not the score).
4. **Score and softmax.** For a step with acting team A, opponent B,
   date ``d`` and (alphabetically sorted, per M26 decision 3) candidate
   list ``remaining_maps``: ``score(map) = build_ban_feature_vector(A,
   B, map, d, ...) . beta`` (after standardizing the raw feature vector
   with the model's stored training-population ``(mean, std)``, exactly
   like ``ordinal_logit``/``multinomial_logit``'s
   :func:`models._shared.apply_standardizer` usage). The predicted
   distribution is ``softmax(scores)`` over ``remaining_maps``, using
   this module's own copy of the numerically-stable max-subtraction
   pattern (:func:`_softmax`), clipped into
   ``[_PROB_CLIP_EPS, 1 - _PROB_CLIP_EPS]`` per entry before
   renormalizing, matching every other model's clip convention.
5. **Optimizer: hand-rolled ragged-group NLL + analytic gradient +
   Armijo GD, reusing ``models._shared``'s constants/helpers exactly
   like ``ordinal_logit``/``multinomial_logit`` do.**
   - **Objective.** Flatten every training ban step's candidate-map
     feature rows into one design matrix ``X`` (shape
     ``(total_candidates, 5)``) plus a parallel ``group_boundaries``
     array (offsets ``[b_0, b_1, ..., b_G]`` with ``b_0 = 0``,
     ``b_G = n_rows``, group ``s`` covering rows ``[b_s, b_{s+1})``)
     and a ``y_true_row_index`` per group (the within-group row of the
     true banned map). The standardizer is fit on the *flattened
     training* ``X`` — one shared ``(mean, std)`` per feature across
     every candidate row of every training step (there is no other
     sensible population to standardize against; this is the same
     "training design matrix, never test rows" leakage boundary
     ``models._shared``'s docstring already documents, just applied to
     a ragged-group matrix instead of a fixed one-row-per-match
     matrix).
   - **Per-step loss.** For step ``s`` with standardized candidate rows
     ``{xs_1, ..., xs_{n_s}}`` and true index ``y_s``:
     ``scores_i = xs_i . beta``; ``probs = softmax(scores)``;
     ``NLL_s = -log(probs[y_s])``. Batch objective:
     ``mean_s(NLL_s) + (l2_lambda / 2) * sum(beta^2)`` — L2 on the
     whole (intercept-free) ``beta``, no "exclude the intercept"
     carve-out needed since there is none.
   - **Gradient.** The standard conditional-logit / softmax gradient:
     ``d(NLL_s)/d(beta) = sum_i (probs_i - 1{i == y_s}) * xs_i``,
     summed over every candidate ``i`` in step ``s``'s group (the same
     functional form as ``multinomial_logit._row_gradients`` chained
     into ``xs_i``, summed over a variable-size group instead of
     picked out at 3 fixed free-class positions). A dedicated test
     verifies it against a central finite-difference numerical
     gradient at several random points, matching the non-negotiable
     correctness bar ``ordinal_logit``/``multinomial_logit`` set.
   - **Initialization.** ``beta = zeros(5)``. Unlike
     ``ordinal_logit``'s threshold reparameterization or
     ``multinomial_logit``'s marginal-derived intercepts, ``beta = 0``
     already has a clean, principled meaning here: every candidate
     scores ``0``, so softmax is exactly uniform over each step's
     remaining maps — the natural "no informative features yet"
     starting point, and no reparameterization is needed since there
     is no ordering constraint to enforce.
   - **Armijo backtracking GD.** Reuses ``models._shared._ARMIJO_C`` /
     ``_LINE_SEARCH_MAX_STEPS``; same convergence checks (``grad_tol``
     on the gradient norm, ``loss_tol`` on loss improvement) and same
     "non-converged returns the best point found, not an error"
     contract as ``ordinal_logit``/``multinomial_logit``'s optimizers.
     Written as this module's own self-contained loop (same rationale
     ``multinomial_logit``'s docstring gives for not sharing one
     generic optimizer across the three different parameter shapes).
6. **Opponent resolution at evaluation (prediction) time.** M26's
   ``VetoStepPredictorFn`` interface is fixed at ``(acting_team_id,
   action, remaining_maps, date, matches_df, maps_df)`` — it does not
   carry ``opponent_team_id``, unlike the training examples (which get
   it for free from the held-out table). This module resolves it
   itself inside the wrapped predictor closure via
   :func:`_resolve_opponent_id`: filters ``matches_df`` to rows where
   ``date`` matches exactly (same exact-equality convention as
   ``models._shared._match_id_for``) **and** ``acting_team_id`` is
   either ``team1_id`` or ``team2_id``, asserts exactly one such row,
   and returns the other id. Raises ``ValueError`` on zero or more
   than one match (an ambiguous or unresolvable opponent), matching
   ``_match_id_for``'s own fail-loud convention. This is
   model-specific plumbing (not shared by any other module today), so
   it lives here, not in ``models/_shared.py`` or ``utils/``.
7. **The wrapped predictor only supports ``action == "ban"``.** M27
   trains a ban-only model (M28 will add the pick arm with independent
   coefficients per the roadmap). :func:`make_veto_step_predictor_fn`
   returns a closure matching ``VetoStepPredictorFn`` structurally, but
   raises ``ValueError`` if called with ``action != "ban"`` —
   documented, not silently wrong. This is exactly why
   ``evaluation.veto_evaluation.score_veto_steps``'s new
   ``actions_to_score={"ban"}`` filter exists: it lets the M27
   evaluation driver call ``score_veto_steps`` with this predictor over
   the *full* held-out sequence (so ``remaining`` bookkeeping stays
   correct across pick steps too) while only ever invoking the
   predictor on ban steps.
8. **Sign convention (documented as intuition, asserted only as an
   empirical outcome).** A *positive* coefficient means "a higher
   feature value makes the map *more* bannable" (increases its softmax
   share at every step). The prior intuition: if the model learns "ban
   my weak maps", the coefficient on ``acting_map_win_rate`` should fit
   *negative* (lower acting win rate -> higher ban probability), and if
   it learns "ban their strong maps" the coefficient on
   ``opponent_map_win_rate`` should fit *positive*. Neither sign is
   enforced in code — the fitted signs are empirical findings, and the
   coefficient report (see :func:`_coefficient_report`) frames each
   feature's direction as "favors more bannable" / "favors less
   bannable" accordingly.

Module placement (preserves the DAG): this module may import
``models._shared`` (the one explicitly-allowed lateral models/ import),
``features.map_win_rate``, ``features.closeness``, ``utils.config`` and
``utils.asof`` only. It must **not** import ``evaluation.*`` (models/
sits below evaluation/ in the DAG), which is why :func:`_resolve_opponent_id`
is written from scratch here rather than reusing anything from
``evaluation.veto_evaluation``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import closeness, map_win_rate
from models._shared import (
    _ARMIJO_C,
    _LINE_SEARCH_MAX_STEPS,
    _PROB_CLIP_EPS,
    _validate_l2_lambda,
    _validate_positive_float,
    _validate_positive_int,
    apply_standardizer,
    fit_standardizer,
)
from utils import asof

# The module's public API, declared explicitly so the linter treats the
# names re-exported from models._shared (apply_standardizer et al.,
# which tests import via ``from models.conditional_logit_ban import
# ...`` and must therefore keep resolving here) as intentional
# re-exports rather than unused imports. Mirrors the identical
# convention in ``models/ordinal_logit.py``'s ``__all__``.
__all__ = (
    "FEATURE_NAMES",
    "ConditionalLogitBanModel",
    "apply_standardizer",
    "build_ban_feature_vector",
    "fit",
    "fit_standardizer",
    "from_dict",
    "make_veto_step_predictor_fn",
    "predict_ban_distribution",
    "to_dict",
)

# The 5 map-varying ban features, in the fixed order every consuming
# step's feature vector is built and every coefficient vector is
# aligned to (decision 2). All five vary across the candidate maps of a
# single (acting team, opponent, date) step; see the module docstring's
# decision 2 for the exact recipe of each.
FEATURE_NAMES = (
    "acting_map_win_rate",
    "opponent_map_win_rate",
    "acting_map_specialization",
    "map_round_margin_variance",
    "map_ot_rate",
)


def _softmax(scores: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax of a sequence of real scores.

    Subtracts the maximum score before exponentiating (so
    ``exp(x - max)`` never overflows), normalizes by the sum, clips
    every entry into ``[eps, 1 - eps]`` (see :data:`_PROB_CLIP_EPS`),
    and renormalizes — the clip-before-renormalize convention the M27
    plan pins down (decision 4), matching the clip spirit of
    ``ordinal_logit``/``multinomial_logit``'s category-probability
    helpers. Because raw softmax entries are strictly positive and
    sum to 1, the clip only perturbs entries below 1e-12 and the
    renormalization is an epsilon-scale correction; the returned vector
    sums to 1 within float rounding and has no exactly-zero entries, so
    ``utils.scoring.log_loss`` never hits its zero-probability hard
    error on this module's outputs.

    Args:
        scores: A sequence of finite real numbers, one per candidate.

    Returns:
        A numpy array of ``float`` probabilities, one per input in the
        same order, non-negative, each in ``[eps, 1 - eps]``, and
        summing to approximately ``1.0``.

    Raises:
        ValueError: If ``scores`` is empty (no distribution to form),
            or if any score is non-finite (NaN/inf would poison the
            exponentials).
    """
    values = np.asarray(scores, dtype=float).ravel()
    if len(values) == 0:
        raise ValueError("cannot compute softmax over an empty score list")
    if not np.isfinite(values).all():
        raise ValueError(f"scores must be finite, got {values!r}")
    shift = float(np.max(values))
    exp_shifted = np.exp(values - shift)
    raw = exp_shifted / np.sum(exp_shifted)
    clipped = np.clip(raw, _PROB_CLIP_EPS, 1.0 - _PROB_CLIP_EPS)
    return clipped / np.sum(clipped)


def _resolve_opponent_id(
    acting_team_id: str | None,
    date: str,
    matches_df: pd.DataFrame,
) -> str:
    """Resolve the unique opponent team id for one acting team and date.

    The evaluation-time opponent lookup (decision 6). The fixed
    ``VetoStepPredictorFn`` interface does not carry the opponent, so
    the wrapped predictor recovers it here: filters ``matches_df`` to
    rows whose ``date`` matches exactly (the same exact-equality
    convention as ``models._shared._match_id_for``) **and** whose
    ``team1_id`` or ``team2_id`` equals ``acting_team_id``, requires
    exactly one such row, and returns the other id of the pair. Zero
    matches (an unseen acting team on that date, or a ``None`` acting
    id) and more than one match (the same team playing twice in one
    timestamp — ambiguous) both raise ``ValueError`` rather than
    silently guessing an opponent.

    Args:
        acting_team_id: The acting team's stable id (``None`` only for
            a decider, which the scorer never routes through a
            predictor).
        date: The exact date string to match (the held-out step's own
            date, byte-for-byte equal to the matches-table value it
            was joined from).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``, ``date``).

    Returns:
        The opponent team's stable id as a ``str`` (the other of the
        matched row's ``{team1_id, team2_id}`` pair).

    Raises:
        KeyError: If ``matches_df`` lacks a required column (propagated
            from :func:`utils.asof.require_columns`).
        ValueError: If zero or more than one match row matches the
            ``(acting_team_id, date)`` pair (the opponent is
            unresolvable or ambiguous).
    """
    asof.require_columns(
        matches_df,
        (asof.MATCH_ID_COL, asof.TEAM1_ID_COL, asof.TEAM2_ID_COL, asof.DATE_COL),
        "matches_df",
    )
    acting_str = str(acting_team_id)
    matches = matches_df[
        (matches_df[asof.DATE_COL] == date)
        & (
            (matches_df[asof.TEAM1_ID_COL] == acting_str)
            | (matches_df[asof.TEAM2_ID_COL] == acting_str)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one match on date {date!r} involving team "
            f"{acting_team_id!r} to resolve the veto opponent, found "
            f"{len(matches)}"
        )
    row = matches.iloc[0]
    if str(row[asof.TEAM1_ID_COL]) == acting_str:
        return str(row[asof.TEAM2_ID_COL])
    return str(row[asof.TEAM1_ID_COL])


def build_ban_feature_vector(
    acting_team_id: str,
    opponent_team_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> np.ndarray:
    """Build the 5-feature ban vector for one map, in FEATURE_NAMES order.

    Computes every feature exactly as decision 2 specifies, each as of
    the step's own ``date`` (strict ``<`` boundary, inherited unchanged
    from each feature module's own ``utils.asof`` usage — this module
    adds no new date filtering). Features 1-3 are team-and-map varying;
    features 4-5 are map-only. The returned values are raw feature
    values — standardization is :func:`fit`'s job, applied to the
    flattened training design matrix only (decision 5).

    Args:
        acting_team_id: The banning team's stable id.
        opponent_team_id: The other team's stable id.
        map_name: The candidate map to score (normalized inside each
            feature estimator, so case/whitespace never break a match).
        date: The as-of cutoff; rows dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.

    Returns:
        A 5-vector numpy array of ``float`` in :data:`FEATURE_NAMES`
        order: ``acting_map_win_rate, opponent_map_win_rate,
        acting_map_specialization, map_round_margin_variance,
        map_ot_rate``.

    Raises:
        ValueError: If a feature's as-of history contains a null/tied
            score or a null/unparseable date, or if ``k`` is invalid
            (all propagated from
            :func:`features.map_win_rate.team_map_win_rate` /
            :func:`features.map_win_rate.team_overall_win_rate` /
            :func:`features.closeness.map_round_margin_variance` /
            :func:`features.closeness.map_ot_rate`).
        KeyError: If either table lacks a required column (propagated
            from the feature modules / ``utils.asof``).
        TypeError: If ``date`` is list-like (propagated from
            ``utils.asof`` via the feature modules).
        ConfigError: If ``map_name`` or an as-of map's ``map_name`` is
            not a string (propagated from
            :func:`utils.config.normalize_map_name` via the feature
            modules).
    """
    # 1/2. acting_map_win_rate / opponent_map_win_rate — shrunk per-map
    # win-rate posterior means at map_win_rate.DEFAULT_K (the same
    # documented default the M25 greedy evaluation arm uses, keeping the
    # two arms' win-rate estimates directly comparable).
    acting_mean = map_win_rate.team_map_win_rate(
        acting_team_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).mean
    opponent_mean = map_win_rate.team_map_win_rate(
        opponent_team_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).mean

    # 3. acting_map_specialization — this map's rate vs the acting
    #    team's own overall baseline; full shrinkage (zero map games)
    #    degrades it to 0.0 exactly, and an unseen team degrades both
    #    terms to 0.5 (decision 2's no-imputation note).
    acting_overall = map_win_rate.team_overall_win_rate(
        acting_team_id, date, matches_df, maps_df
    ).rate
    acting_specialization = acting_mean - acting_overall

    # 4. map_round_margin_variance — match-level (not a team diff);
    #    NaN (n <= 1) replaced with 0.0 per the documented fallback.
    margin_var = closeness.map_round_margin_variance(
        map_name, date, matches_df, maps_df
    ).variance
    if math.isnan(margin_var):
        map_round_margin_variance = 0.0
    else:
        map_round_margin_variance = float(margin_var)

    # 5. map_ot_rate — the per-map heavily-shrunk OT posterior mean
    #    (always finite: alpha + beta >= k > 0, so no NaN fallback).
    map_ot = closeness.map_ot_rate(map_name, date, matches_df, maps_df).mean

    return np.asarray(
        [
            acting_mean,
            opponent_mean,
            acting_specialization,
            map_round_margin_variance,
            map_ot,
        ],
        dtype=float,
    )


def _loss_and_gradient(
    Xs_flat: np.ndarray,
    group_boundaries: np.ndarray,
    y_true_row_index: np.ndarray,
    beta: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray]:
    """Return the ragged-group softmax objective and its analytic gradient.

    Implements decision 5's objective: the batch is a set of variable-
    size groups (one per training ban step), each group's rows being
    that step's standardized candidate-map feature vectors. For each
    group ``s`` with rows ``[b_s, b_{s+1})`` and true row ``y_s``,
    ``scores = Xs[b_s:b_{s+1}] @ beta``, ``probs = softmax(scores)``
    (stable, clipped), ``NLL_s = -log(probs[y_s])``; the batch
    objective is ``mean_s(NLL_s) + (l2_lambda / 2) * sum(beta^2)``.
    The analytic gradient is the standard conditional-logit gradient:
    ``d(NLL_s)/d(beta) = sum_i (probs_i - 1{i == y_s}) * xs_i``,
    averaged over groups, plus the L2 term ``l2_lambda * beta`` (L2 on
    the whole intercept-free ``beta`` — no carve-out needed). These are
    the exact derivatives of the *unclipped* objective; the clip
    epsilon (1e-12) is inactive for any realistic finite scores, so in
    practice they are also the derivatives of the clipped objective this
    function actually minimizes (the same argument
    ``multinomial_logit`` makes for its softmax). A dedicated
    regression test verifies them against central finite differences.

    Args:
        Xs_flat: The already-standardized flattened design matrix,
            ``(total_candidates, p)`` floats.
        group_boundaries: A 1-D int array of length ``n_groups + 1``
            with ``group_boundaries[0] == 0``,
            ``group_boundaries[-1] == total_candidates`` and strictly
            increasing offsets: group ``s`` covers rows
            ``[group_boundaries[s], group_boundaries[s + 1])``.
        y_true_row_index: A 1-D int array of length ``n_groups``; entry
            ``s`` is the within-group row index (0-based) of group
            ``s``'s true banned map.
        beta: The current coefficient vector, length ``p``.
        l2_lambda: The L2 regularization strength on ``beta``
            (non-negative finite float).

    Returns:
        A ``(loss, grad_beta)`` tuple: ``loss`` the scalar batch
        objective; ``grad_beta`` the ``(p,)`` gradient of the objective
        w.r.t. ``beta`` (including the L2 term).

    Raises:
        ValueError: If the shapes are inconsistent (``Xs_flat`` not 2-D,
            ``group_boundaries`` not a strictly-increasing array from 0
            to ``Xs_flat.shape[0]`` with at least two entries,
            ``y_true_row_index`` not one entry per group, a true index
            outside its group's row range, or ``beta`` not one entry per
            feature column).
    """
    Xs = np.asarray(Xs_flat, dtype=float)
    bounds = np.asarray(group_boundaries, dtype=int).ravel()
    ys = np.asarray(y_true_row_index, dtype=int).ravel()
    beta_arr = np.asarray(beta, dtype=float).ravel()

    if Xs.ndim != 2:
        raise ValueError(
            f"Xs_flat must be a 2-D design matrix, got {Xs.ndim} dimension(s)"
        )
    if len(bounds) < 2:
        raise ValueError(
            "group_boundaries must have at least two entries (one group "
            "boundary per step plus the closing offset)"
        )
    if bounds[0] != 0 or bounds[-1] != Xs.shape[0]:
        raise ValueError(
            f"group_boundaries must start at 0 and end at the row count "
            f"{Xs.shape[0]}, got first={bounds[0]} last={bounds[-1]}"
        )
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(
            "group_boundaries must be strictly increasing (every group "
            "must contain at least one candidate row)"
        )
    n_groups = len(bounds) - 1
    if len(ys) != n_groups:
        raise ValueError(
            f"y_true_row_index has {len(ys)} entries but "
            f"group_boundaries defines {n_groups} groups; they must match"
        )
    for s in range(n_groups):
        if not (0 <= ys[s] < bounds[s + 1] - bounds[s]):
            raise ValueError(
                f"y_true_row_index[{s}] = {ys[s]} is outside its group's "
                f"row range [0, {bounds[s + 1] - bounds[s]})"
            )
    if beta_arr.shape[0] != Xs.shape[1]:
        raise ValueError(
            f"beta has {beta_arr.shape[0]} entries but Xs_flat has "
            f"{Xs.shape[1]} feature columns; they must match"
        )

    total_nll = 0.0
    grad = np.zeros_like(beta_arr, dtype=float)
    for s in range(n_groups):
        start = int(bounds[s])
        end = int(bounds[s + 1])
        group_xs = Xs[start:end]
        scores = group_xs @ beta_arr
        probs = _softmax(scores)
        y = int(ys[s])
        total_nll += -math.log(float(probs[y]))
        indicator = np.zeros(len(probs), dtype=float)
        indicator[y] = 1.0
        grad += (probs - indicator) @ group_xs
    nll = total_nll / n_groups
    grad /= n_groups
    # L2 on the whole (intercept-free) beta; the penalty is NOT averaged
    # over the batch.
    l2_penalty = (l2_lambda / 2.0) * float(np.sum(beta_arr**2))
    grad += l2_lambda * beta_arr
    return nll + l2_penalty, grad


def _gradient_descent(
    Xs: np.ndarray,
    group_boundaries: np.ndarray,
    y_true_row_index: np.ndarray,
    l2_lambda: float,
    max_iter: int,
    grad_tol: float,
    loss_tol: float,
) -> tuple[np.ndarray, bool, int, tuple[float, ...]]:
    """Run full-batch gradient descent with Armijo backtracking.

    The conditional-logit optimizer (decision 5). Starts from
    ``beta = zeros(5)`` — the clean "uniform over every step's
    remaining maps" initialization (decision 5's initialization
    bullet) — then iterates: compute the loss/gradient at the current
    point; stop (converged) if ``||gradient|| < grad_tol`` or if the
    loss improvement between iterations drops below ``loss_tol``;
    otherwise try step size ``1.0``, halving up to
    :data:`_LINE_SEARCH_MAX_STEPS` times until the Armijo
    sufficient-decrease condition (``loss(beta - step*grad) <= loss(beta)
    - _ARMIJO_C * step * ||grad||^2``) holds, then take that step. If
    the line search cannot find any acceptable step, or if ``max_iter``
    is hit, the run stops with ``converged=False`` and returns the best
    point found — a non-converged fit is a valid (if suboptimal) model,
    not an error. The returned loss trace is non-increasing by
    construction (every accepted step satisfies Armijo).

    Args:
        Xs: The already-standardized flattened design matrix,
            ``(total_candidates, p)`` floats.
        group_boundaries: The group-offset array (see
            :func:`_loss_and_gradient`).
        y_true_row_index: The per-group true-row indices (see
            :func:`_loss_and_gradient`).
        l2_lambda: The L2 strength (validated non-negative finite).
        max_iter: The iteration cap (validated positive int).
        grad_tol: The gradient-norm convergence tolerance.
        loss_tol: The loss-improvement convergence tolerance.

    Returns:
        A ``(best_beta, converged, n_iter, loss_trace)`` tuple: the
        best coefficient vector, whether the run converged, the number
        of iterations actually executed, and the per-iteration loss
        trace (non-increasing, length ``n_iter``).

    Raises:
        ValueError: If the shapes are inconsistent (propagated from
            :func:`_loss_and_gradient`).
    """
    beta = np.zeros(Xs.shape[1], dtype=float)

    best_beta = beta.copy()
    best_loss = float("inf")
    loss_trace: list[float] = []
    prev_loss: float | None = None
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        loss, grad = _loss_and_gradient(
            Xs, group_boundaries, y_true_row_index, beta, l2_lambda
        )
        n_iter = iteration + 1
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
        loss_trace.append(loss)

        grad_norm = math.sqrt(float(np.sum(grad**2)))
        if grad_norm < grad_tol:
            converged = True
            break
        if prev_loss is not None and (prev_loss - loss) < loss_tol:
            converged = True
            break

        step = 1.0
        accepted = False
        for _ in range(_LINE_SEARCH_MAX_STEPS):
            trial_beta = beta - step * grad
            trial_loss, _ = _loss_and_gradient(
                Xs, group_boundaries, y_true_row_index, trial_beta, l2_lambda
            )
            if trial_loss <= loss - _ARMIJO_C * step * grad_norm**2:
                beta = trial_beta
                accepted = True
                break
            step *= 0.5
        if not accepted:
            # No step size down to step == 2^-50 satisfies Armijo: the
            # current point is the best we can reach; report as
            # non-converged rather than raising.
            converged = False
            break
        prev_loss = loss
    else:
        # The for-loop exhausted max_iter without breaking: the cap was
        # hit without convergence; return the best point found.
        converged = False

    return best_beta, converged, n_iter, tuple(loss_trace)


@dataclass(frozen=True)
class ConditionalLogitBanModel:
    """A fitted conditional-logit ban model.

    Holds the fitted parameters and the diagnostics needed to (a) make
    predictions via :func:`predict_ban_distribution`, (b)
    serialize/deserialize via :func:`to_dict` / :func:`from_dict`, and
    (c) interpret the fit via the coefficient report. ``coefficients``
    has exactly one entry per feature in ``feature_names`` (the
    :data:`FEATURE_NAMES` order); there is deliberately **no
    intercept** (decision 3 — a shared constant cancels exactly under
    the per-step softmax). ``standardizer_means`` / ``standardizer_stds``
    describe the *training* design matrix (the second leakage boundary;
    see :mod:`models._shared`'s docstring). ``loss_trace`` is a
    live-fit diagnostic (per-iteration loss, non-increasing) that is
    deliberately *not* persisted by :func:`to_dict` — a deserialized
    model carries an empty trace.

    Attributes:
        coefficients: The 5-vector of fitted coefficients (one per
            :data:`FEATURE_NAMES` feature, no intercept).
        standardizer_means: Per-feature training-column means (length
            5).
        standardizer_stds: Per-feature training-column stds (length 5;
            a zero-variance column's std is ``1.0`` per the guard).
        feature_names: The feature name tuple (:data:`FEATURE_NAMES`).
        converged: Whether gradient descent converged (``True``) or hit
            ``max_iter``/line-search failure (``False`` — still a valid,
            if suboptimal, model).
        n_iter: Number of gradient-descent iterations executed.
        final_loss: The objective value at the returned point.
        n_train: Number of training *ban steps* the model was fit on
            (the group count, not the candidate-row count).
        l2_lambda: The L2 strength used for this fit (stored so the
            artifact records its own regularization).
        loss_trace: The per-iteration objective trace (non-increasing,
            length ``n_iter``); ``()`` for a deserialized model.
    """

    coefficients: np.ndarray
    standardizer_means: np.ndarray
    standardizer_stds: np.ndarray
    feature_names: tuple[str, ...]
    converged: bool
    n_iter: int
    final_loss: float
    n_train: int
    l2_lambda: float
    loss_trace: tuple[float, ...] = ()


def fit(
    X_flat: np.ndarray,
    group_boundaries: np.ndarray,
    y_true_row_index: np.ndarray,
    l2_lambda: float = 1.0,
    max_iter: int = 2000,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-10,
) -> ConditionalLogitBanModel:
    """Fit the conditional-logit ban model by Armijo gradient descent.

    Assembles a complete :class:`ConditionalLogitBanModel` from the
    ragged training design: fits the per-feature z-score standardizer
    on the *flattened training* matrix only (the second leakage
    boundary, applied to the ragged-group matrix exactly as
    :func:`models._shared.fit_standardizer` applies it to a flat one;
    see the module docstring's decision 5) and transforms with it, then
    runs full-batch gradient descent with Armijo backtracking (see
    :func:`_gradient_descent`) initialized at ``beta = 0`` (the
    "uniform over every step's remaining maps" starting point, decision
    5). The returned model carries the training standardizer so
    :func:`predict_ban_distribution` can standardize later rows with
    the exact training-population statistics.

    Args:
        X_flat: The raw (unstandardized) flattened training design
            matrix, ``(total_candidates, 5)`` floats in
            :data:`FEATURE_NAMES` order — the concatenation of every
            training ban step's per-candidate
            :func:`build_ban_feature_vector` rows. The standardizer is
            fit on this matrix inside this function.
        group_boundaries: A 1-D int array of length ``n_groups + 1``
            with ``group_boundaries[0] == 0``,
            ``group_boundaries[-1] == total_candidates`` and strictly
            increasing offsets: group ``s`` (one training ban step)
            covers rows ``[group_boundaries[s],
            group_boundaries[s + 1])``.
        y_true_row_index: A 1-D int array of length ``n_groups``; entry
            ``s`` is the within-group row index (0-based) of group
            ``s``'s true banned map.
        l2_lambda: L2 regularization strength on the whole (intercept-
            free) ``beta``; must be non-negative finite (default
            ``1.0`` — the same conservative default the M20/M21 arms
            use, not CV-tuned in this task).
        max_iter: Cap on gradient-descent iterations (default 2000). If
            the cap is hit without convergence, ``fit`` returns the best
            point found with ``converged=False`` rather than raising.
        grad_tol: Gradient-norm convergence tolerance (default 1e-6).
        loss_tol: Loss-improvement convergence tolerance (default
            1e-10).

    Returns:
        A frozen :class:`ConditionalLogitBanModel` with the fitted
        ``coefficients``, the training standardizer, the diagnostics
        (``converged``/``n_iter``/``final_loss``/``n_train``/
        ``l2_lambda``/``loss_trace``), and ``feature_names =
        FEATURE_NAMES``.

    Raises:
        ValueError: If ``X_flat`` is not a 2-D array, is empty, or does
            not have exactly ``len(FEATURE_NAMES)`` columns;
            ``group_boundaries`` is not a strictly-increasing 1-D array
            from 0 to ``X_flat.shape[0]``; ``y_true_row_index`` does
            not have one entry per group or contains an index outside
            its group's row range; or any hyperparameter is invalid (see
            the ``models._shared`` validate helpers).
    """
    X_arr = np.asarray(X_flat, dtype=float)
    bounds = np.asarray(group_boundaries, dtype=int).ravel()
    ys = np.asarray(y_true_row_index, dtype=int).ravel()

    if X_arr.ndim != 2:
        raise ValueError(
            f"X_flat must be a 2-D design matrix, got {X_arr.ndim} dimension(s)"
        )
    if X_arr.shape[0] == 0:
        raise ValueError("cannot fit a conditional-logit model on an empty design matrix")
    if X_arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X_flat must have exactly {len(FEATURE_NAMES)} feature columns "
            f"(one per {FEATURE_NAMES} entry), got {X_arr.shape[1]}"
        )
    if len(bounds) < 2:
        raise ValueError(
            "group_boundaries must have at least two entries (one group "
            "boundary per step plus the closing offset)"
        )
    if bounds[0] != 0 or bounds[-1] != X_arr.shape[0]:
        raise ValueError(
            f"group_boundaries must start at 0 and end at the row count "
            f"{X_arr.shape[0]}, got first={bounds[0]} last={bounds[-1]}"
        )
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(
            "group_boundaries must be strictly increasing (every group "
            "must contain at least one candidate row)"
        )
    n_groups = len(bounds) - 1
    if len(ys) != n_groups:
        raise ValueError(
            f"y_true_row_index has {len(ys)} entries but group_boundaries "
            f"defines {n_groups} groups; they must match"
        )
    for s in range(n_groups):
        if not (0 <= ys[s] < bounds[s + 1] - bounds[s]):
            raise ValueError(
                f"y_true_row_index[{s}] = {ys[s]} is outside its group's "
                f"row range [0, {bounds[s + 1] - bounds[s]})"
            )

    l2 = _validate_l2_lambda(l2_lambda)
    max_it = _validate_positive_int(max_iter, "max_iter")
    g_tol = _validate_positive_float(grad_tol, "grad_tol")
    l_tol = _validate_positive_float(loss_tol, "loss_tol")

    means, stds = fit_standardizer(X_arr)
    Xs = apply_standardizer(X_arr, means, stds)
    beta, converged, n_iter, trace = _gradient_descent(
        Xs, bounds, ys, l2, max_it, g_tol, l_tol
    )
    return ConditionalLogitBanModel(
        coefficients=beta,
        standardizer_means=means,
        standardizer_stds=stds,
        feature_names=FEATURE_NAMES,
        converged=converged,
        n_iter=n_iter,
        final_loss=trace[-1],
        n_train=len(ys),
        l2_lambda=l2,
        loss_trace=trace,
    )


def predict_ban_distribution(
    acting_team_id: str,
    opponent_team_id: str,
    remaining_maps: Sequence[str],
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    model: ConditionalLogitBanModel,
) -> list[float]:
    """Predict the ban probability distribution over the remaining maps.

    The core scoring function (decision 4): for every map in
    ``remaining_maps`` builds the raw 5-feature vector via
    :func:`build_ban_feature_vector`, standardizes it with the model's
    stored training-population statistics
    (:func:`models._shared.apply_standardizer` — the second leakage
    boundary, applied identically to every later row), computes
    ``scores = Xs @ coefficients`` (no intercept, decision 3), and
    returns :func:`_softmax` of the scores — a probability distribution
    aligned 1:1 to ``remaining_maps``' order, summing to 1.

    Args:
        acting_team_id: The banning team's stable id.
        opponent_team_id: The other team's stable id.
        remaining_maps: The alphabetically sorted (by normalized name,
            per M26 decision 3) list of maps still in play; must be
            non-empty. The returned probabilities align 1:1 to this
            order.
        date: The single as-of cutoff for every feature lookup (the
            held-out step's own date).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.
        model: The fitted model whose stored standardizer and
            coefficients are applied.

    Returns:
        A ``list`` of ``len(remaining_maps)`` ``float`` probabilities
        summing to approximately ``1.0``, each in
        ``[eps, 1 - eps]``, aligned to ``remaining_maps``' order.

    Raises:
        ValueError: If ``remaining_maps`` is empty (no distribution to
            form); if a feature computation fails (propagated from
            :func:`build_ban_feature_vector`); or if the standardized
            row count mismatches the model's coefficient count
            (propagated from :func:`models._shared.apply_standardizer`
            — cannot happen given a length-5 vector and a 5-coefficient
            model).
        KeyError: If either table lacks a required column (propagated
            from :func:`build_ban_feature_vector`).
        TypeError: If ``date`` is list-like (propagated from
            :func:`build_ban_feature_vector`).
        ConfigError: If a map name is not a string (propagated from
            :func:`build_ban_feature_vector`).
    """
    maps = list(remaining_maps)
    if not maps:
        raise ValueError(
            "predict_ban_distribution needs at least one remaining map "
            "to form a distribution"
        )
    raw_rows = [
        build_ban_feature_vector(
            acting_team_id, opponent_team_id, name, date, matches_df, maps_df
        )
        for name in maps
    ]
    X = np.asarray(raw_rows, dtype=float)
    Xs = apply_standardizer(X, model.standardizer_means, model.standardizer_stds)
    scores = Xs @ model.coefficients
    return [float(p) for p in _softmax(scores)]


def make_veto_step_predictor_fn(
    model: ConditionalLogitBanModel,
) -> Callable[[str | None, str, Sequence[str], str, pd.DataFrame, pd.DataFrame], Sequence[float]]:
    """Wrap a fitted model into the M26 per-step predictor-interface shape.

    Bridges the fixed M26 per-step predictor interface (structurally
    matched without importing ``evaluation.veto_evaluation``, which
    models/ must not depend on) — a callable ``(acting_team_id, action,
    remaining_maps, date, matches_df, maps_df) -> Sequence[float]`` —
    to this model's feature builder, which additionally needs the
    *opponent* team id. The closure resolves the opponent itself via
    :func:`_resolve_opponent_id` (decision 6) and returns
    :func:`predict_ban_distribution`'s probability list.

    The wrapped predictor only supports ``action == "ban"`` (decision
    7): M27 trains a ban-only model, so any other action raises
    ``ValueError`` — documented, not silently wrong. This is exactly
    why the evaluation driver passes ``actions_to_score={"ban"}`` to
    ``evaluation.veto_evaluation.score_veto_steps``: the scorer replays
    the *full* held-out sequence (so ``remaining`` bookkeeping stays
    correct across pick steps too) but only ever invokes the predictor
    on ban steps.

    Args:
        model: The fitted model to predict with.

    Returns:
        A closure ``(acting_team_id, action, remaining_maps, date,
        matches_df, maps_df) -> list[float]`` (a probability
        distribution aligned to ``remaining_maps``' order, summing to
        approximately 1).

    Raises:
        ValueError: If the closure is called with ``action != "ban"``
            (naming the offending action); if the opponent cannot be
            resolved (propagated from :func:`_resolve_opponent_id`); or
            if a feature computation fails (propagated from
            :func:`predict_ban_distribution`).
        KeyError: If either table lacks a required column (propagated
            from :func:`_resolve_opponent_id` /
            :func:`predict_ban_distribution`).
    """

    def predictor_fn(
        acting_team_id: str | None,
        action: str,
        remaining_maps: Sequence[str],
        date: str,
        matches_df: pd.DataFrame,
        maps_df: pd.DataFrame,
    ) -> Sequence[float]:
        """Predict the ban distribution over the remaining maps.

        Resolves the opponent id from the matches table (the interface
        does not carry it), then returns
        :func:`predict_ban_distribution`'s softmax list. See
        :func:`make_veto_step_predictor_fn`'s docstring for the
        ban-only contract.

        Args:
            acting_team_id: The banning team's stable id.
            action: Must be ``"ban"`` (anything else raises — M27 is a
                ban-only model; M28 adds the pick arm).
            remaining_maps: The alphabetically sorted list of maps still
                in play.
            date: The as-of cutoff (the step's own date).
            matches_df: The full materialised ``matches`` table.
            maps_df: The full materialised ``maps`` table.

        Returns:
            The ban probability distribution over ``remaining_maps``.

        Raises:
            ValueError: If ``action != "ban"``; if the opponent is
                unresolvable or ambiguous (propagated from
                :func:`_resolve_opponent_id`); or if a feature
                computation fails (propagated from
                :func:`predict_ban_distribution`).
            KeyError: If either table lacks a required column
                (propagated from :func:`_resolve_opponent_id` /
                :func:`predict_ban_distribution`).
        """
        if action != "ban":
            raise ValueError(
                f"the conditional-logit veto predictor only supports "
                f"action 'ban', got action {action!r} (M28 adds the "
                "pick arm with independent coefficients)"
            )
        opponent_team_id = _resolve_opponent_id(acting_team_id, date, matches_df)
        return predict_ban_distribution(
            acting_team_id,
            opponent_team_id,
            remaining_maps,
            date,
            matches_df,
            maps_df,
            model,
        )

    return predictor_fn


def _coefficient_report(model: ConditionalLogitBanModel) -> list[dict]:
    """Build the human-readable coefficient report for a fitted model.

    One entry per feature: ``{"feature": name, "coefficient": value,
    "direction": label}`` where ``direction`` is derived from the
    coefficient's sign under the module-docstring convention (decision
    8): a positive coefficient means "a higher feature value makes the
    map more bannable" (``"favors more bannable"``), a negative one
    ``"favors less bannable"``, and an exactly-zero one ``"no effect"``.
    Entries are sorted by ``abs(coefficient)`` descending so the most
    influential features read first.

    Args:
        model: The fitted model whose ``coefficients`` and
            ``feature_names`` are reported.

    Returns:
        A list of dicts, one per feature, sorted by descending
        ``abs(coefficient)``.

    Raises:
        Nothing (the model's own shape validation guarantees the two
            arrays line up).
    """
    entries = []
    for name, coefficient in zip(model.feature_names, model.coefficients):
        if coefficient > 0.0:
            direction = "favors more bannable"
        elif coefficient < 0.0:
            direction = "favors less bannable"
        else:
            direction = "no effect"
        entries.append(
            {
                "feature": name,
                "coefficient": float(coefficient),
                "direction": direction,
            }
        )
    return sorted(entries, key=lambda entry: abs(entry["coefficient"]), reverse=True)


def to_dict(model: ConditionalLogitBanModel) -> dict:
    """Serialize a fitted model to a plain JSON-serializable dict.

    Produces the artifact dict the training driver writes:
    ``feature_names``, ``coefficients``, ``standardizer_means``,
    ``standardizer_stds``, ``l2_lambda``, ``converged``, ``n_iter``,
    ``final_loss``, ``n_train``, plus a ``coefficient_report`` list
    (from :func:`_coefficient_report`). Every value is a plain
    str/int/float/list so ``json.dumps`` accepts the dict directly. The
    ``loss_trace`` diagnostic is deliberately *not* persisted (it is a
    live-fit trace, not model parameters). No file I/O happens here.

    Args:
        model: The fitted model to serialize.

    Returns:
        A plain dict as described, directly ``json.dumps``-serializable
        (with ``sort_keys=True`` for deterministic artifacts).

    Raises:
        Nothing.
    """
    return {
        "feature_names": list(model.feature_names),
        "coefficients": [float(c) for c in model.coefficients],
        "standardizer_means": [float(m) for m in model.standardizer_means],
        "standardizer_stds": [float(s) for s in model.standardizer_stds],
        "l2_lambda": float(model.l2_lambda),
        "converged": bool(model.converged),
        "n_iter": int(model.n_iter),
        "final_loss": float(model.final_loss),
        "n_train": int(model.n_train),
        "coefficient_report": _coefficient_report(model),
    }


def from_dict(d: dict) -> ConditionalLogitBanModel:
    """Deserialize a fitted model from a to_dict-produced dict.

    Reconstructs a :class:`ConditionalLogitBanModel` from the plain dict
    :func:`to_dict` produces (or from ``json.loads`` of the artifact the
    training driver writes). Arrays are rebuilt as numpy arrays; shape
    consistency is validated (coefficients/means/stds must all line up
    with ``feature_names``). The ``coefficient_report`` key is ignored
    on read (it is derived, not stored) and ``loss_trace`` is empty for
    a deserialized model. No file I/O happens here.

    Args:
        d: The dict to load; must carry the nine parameter/diagnostic
            keys (``coefficient_report`` optional, ignored).

    Returns:
        A :class:`ConditionalLogitBanModel` whose parameters reproduce
            the serialized ones exactly (``feature_names`` as a tuple,
            arrays as ``float`` numpy arrays, diagnostics as plain
            scalars).

    Raises:
        KeyError: If a required key is absent (propagated from dict
            indexing).
        ValueError: If the shapes are inconsistent (coefficient count !=
            feature count, or means/stds length != coefficient count),
            or if a numeric field cannot be coerced.
    """
    feature_names = tuple(str(name) for name in d["feature_names"])
    coefficients = np.asarray(d["coefficients"], dtype=float)
    means = np.asarray(d["standardizer_means"], dtype=float)
    stds = np.asarray(d["standardizer_stds"], dtype=float)
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"coefficients has {len(coefficients)}; they must match"
        )
    if len(means) != len(coefficients) or len(stds) != len(coefficients):
        raise ValueError(
            f"standardizer means ({len(means)}) / stds ({len(stds)}) must "
            f"each have one entry per coefficient ({len(coefficients)})"
        )
    return ConditionalLogitBanModel(
        coefficients=coefficients,
        standardizer_means=means,
        standardizer_stds=stds,
        feature_names=feature_names,
        converged=bool(d["converged"]),
        n_iter=int(d["n_iter"]),
        final_loss=float(d["final_loss"]),
        n_train=int(d["n_train"]),
        l2_lambda=float(d["l2_lambda"]),
    )

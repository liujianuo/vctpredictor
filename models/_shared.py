"""Shared helper logic for the ``models/`` package (not a genuine utility).

This module is the single home for logic genuinely shared by more than
one ``models/`` module, so no models module ever imports a private
helper from a sibling models module. It deliberately lives under
``models/`` (not ``utils/``): it is model-support code — feature
assembly, standardization, hyperparameter validation, and the shared
numeric constants — not a leaf-level utility, and placing it in
``utils/`` would reopen the exact lateral-dependency problem this
package layout closes (and would violate the "``utils/`` is reserved
for genuine, leaf-level utilities" rule).

It is consumed by :mod:`models.ordinal_logit` (roadmap M20) and
:mod:`models.multinomial_logit` (roadmap M21), both of which need the
identical 15-feature vector (the roadmap's "compare M21 against M20 on
identical splits" requirement means the identical vector, not merely a
similar one), the same per-feature z-score standardizer, the same
hyperparameter validators, and the same probability-clip and Armijo
line-search constants. Like the rest of ``models/`` it has no CLI and
no file I/O of its own; history still flows exclusively through
``utils.asof`` inside the feature calls.

**Missing-value fallback policy (apply per-side before differencing).**
The M16/M17 features legitimately return ``None`` for "no signal";
each is converted to a neutral value *before* the A-minus-B
subtraction in :func:`build_feature_vector`:

- ``acs_form_diff`` / ``rating_form_diff``: if either side's
  ``FormStat.mean`` is ``None`` (zero qualifying maps), the feature is
  ``0.0`` (no signal), not a partial diff against the other side's
  real value.
- ``days_since_diff``: if a side's ``days_since_last_match`` is
  ``None`` (unseen team / no strictly-prior match), that side is
  treated as ``0`` before subtracting.
- ``roster_decay_diff``: if a side's ``decay_multiplier`` is ``None``
  (either ``changed is None`` — fewer than two evaluable maps — or
  ``changed is False``, i.e. no roster change declared), that side is
  treated as ``1.0`` (no penalty) before subtracting.
- ``map_round_margin_variance``: ``float("nan")`` when ``n <= 1`` for
  that map is replaced with ``0.0`` — documented as "no observed-
  variance signal contributes no information", not a statistically
  principled imputation.
- ``attack_side_win_rate_diff`` / ``defense_side_win_rate_diff`` /
  ``signed_margin_diff`` / ``first_blood_diff`` (M38.5 additions): **no
  fallback needed**. All three wrapped estimators
  (``side_win_rate.team_map_side_rate``,
  ``signed_margin.team_map_signed_margin``,
  ``first_blood.team_map_first_blood_rate``) return a numeric posterior
  ``.mean`` that degrades to exactly the (non-``None``) prior at zero
  as-of history, so a zero-history side contributes ``prior`` and the
  A-minus-B difference of two zero-history sides is ``0.0`` with no
  conditional (unlike the M16/M17 ``None``-returning estimators above).

**Standardization is its own, separate leakage boundary.** A
per-feature ``(mean, std)`` z-score standardizer is fit *only on the
assembled training design matrix* — never on test/held-out rows — via
:func:`fit_standardizer` / :func:`apply_standardizer`. This is a
second, distinct leakage boundary on top of each feature's own as-of
``<`` cutoff, and must be documented as such: the means/stds describe
the training population and would leak test-distribution information
if fit on the full data. A zero-variance training column standardizes
to ``0.0`` for every row (its std is replaced by ``1.0`` to guard the
divide-by-zero) rather than raising, since a degenerate/constant
column is plausible at this data scale.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features import (
    closeness,
    elo,
    first_blood,
    h2h_context,
    map_win_rate,
    player_form,
    side_win_rate,
    signed_margin,
)
from utils import asof

# The 15 features in the fixed order every model consumes, one shared
# coefficient each. Team-specific features are expressed as A-minus-B
# differences (A = team1_id, B = team2_id, matching the
# ``drivers.labels``/``models.four_way_baseline`` convention) so one
# coefficient vector applies regardless of which side is "A" in a given
# match row. See this module's docstring and :func:`build_feature_vector`
# for the exact per-feature recipe and the missing-value fallbacks.
FEATURE_NAMES = (
    "map_win_rate_diff",
    "elo_differential",
    "close_map_freq_diff",
    "ot_rate_diff",
    "map_round_margin_variance",
    "acs_form_diff",
    "rating_form_diff",
    "h2h_win_rate_centered",
    "event_stage",
    "days_since_diff",
    "roster_decay_diff",
    "attack_side_win_rate_diff",
    "defense_side_win_rate_diff",
    "signed_margin_diff",
    "first_blood_diff",
)

# The four outcome categories in ordinal order, mirroring
# ``drivers.labels.OUTCOME_LABELS`` (documented, deliberately *not*
# imported: this module must not depend on ``drivers/``). Index 0 is
# "A-regulation", 1 "A-OT", 2 "B-OT", 3 "B-regulation" — the order
# each consuming model's :func:`predict_proba` returns and
# ``utils.scoring``'s metrics expect.
OUTCOME_LABELS = ("A-regulation", "A-OT", "B-OT", "B-regulation")

# Clip epsilon for the category probabilities before the log in the
# negative log-likelihood (same epsilon convention as
# ``features.map_win_rate._PROB_CLIP_EPS``). Every model's link
# (cumulative logit, softmax, binary logit) guarantees strictly
# positive probabilities for any finite linear predictor, so the clip
# is defensive floor/sky coverage for extreme inputs, not a live-data
# fix.
_PROB_CLIP_EPS = 1e-12

# Armijo line-search constants: the sufficient-decrease parameter and
# the maximum number of step halvings (step starts at 1.0, halving up to
# this many times) before an iteration is declared unable to make
# progress.
_ARMIJO_C = 1e-4
_LINE_SEARCH_MAX_STEPS = 50

# Number of categories (the length of OUTCOME_LABELS / the K of each
# consuming model).
_N_CATEGORIES = 4


def _sigmoid(x: float) -> float:
    """Return the logistic sigmoid of ``x``, computed stably.

    ``1 / (1 + exp(-x))`` evaluated in the numerically-stable branch
    that avoids ``exp(-x)`` overflowing to ``inf`` for large negative
    ``x`` (and, symmetrically, avoids ``exp(x)`` overflowing for large
    positive ``x``). ``sigmoid`` is strictly increasing, so with
    strictly increasing thresholds the three ``C_j`` of the ordinal
    link stay strictly ordered.

    Args:
        x: The scalar argument (typically ``theta_j + eta`` for the
            ordinal link or ``alpha + x . beta`` for the binary link).

    Returns:
        The sigmoid value as a ``float`` in ``(0, 1)`` (asymptotically
        approaching but never reaching 0 and 1 for finite ``x``).

    Raises:
        Nothing (the formula is total for any finite input).
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _validate_l2_lambda(l2_lambda) -> float:
    """Validate the L2 strength and return it as a float.

    The L2 strength must be a non-negative finite real: a negative value
    would anti-regularize (grow coefficients unboundedly) and NaN/inf
    would poison every gradient step.

    Args:
        l2_lambda: The proposed L2 strength.

    Returns:
        ``l2_lambda`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a float, or if the result
            is NaN, infinite, or negative.
    """
    try:
        value = float(l2_lambda)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"l2_lambda must be a non-negative finite real number, got {l2_lambda!r}"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"l2_lambda must be a non-negative finite real number, got {l2_lambda!r}"
        )
    return value


def _validate_positive_int(value, name: str) -> int:
    """Validate a positive integer parameter and return it as an ``int``.

    Used for ``max_iter``: rejects bools (which are int-coercible but
    never intended), non-integral values, and non-positive values.

    Args:
        value: The proposed parameter value.
        name: The parameter name for the error message.

    Returns:
        ``value`` as a positive ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not integer-valued, or
            ``<= 0``.
    """
    if type(value) is bool:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _validate_positive_float(value, name: str) -> float:
    """Validate a positive finite parameter and return it as a ``float``.

    Used for ``grad_tol``/``loss_tol``: both are convergence tolerances
    that must be strictly positive and finite (zero or negative would
    disable or invert the convergence checks; NaN/inf would poison them).

    Args:
        value: The proposed parameter value.
        name: The parameter name for the error message.

    Returns:
        ``value`` as a positive finite ``float``.

    Raises:
        ValueError: If it cannot be coerced to a float, or if the result
            is NaN, infinite, or ``<= 0``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive real number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite real number, got {value!r}"
        )
    return result


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the per-feature z-score standardizer on a design matrix.

    Computes the per-column ``(mean, std)`` of ``X`` (population std,
    matching numpy's ``std`` convention). A zero-variance column's std
    is replaced with ``1.0`` rather than raising, so
    :func:`apply_standardizer` turns that column into ``0.0`` for every
    row (a degenerate/constant training column contributes no signal);
    see this module's docstring's standardization bullet.

    Args:
        X: The design matrix to compute statistics from, ``(n, p)``
            floats.

    Returns:
        A ``(means, stds)`` tuple of ``(p,)`` numpy arrays: the
        per-column means and the per-column stds (zero-variance columns
        carry ``1.0``).

    Raises:
        ValueError: If ``X`` is empty or is not a 2-D array (no
            per-column statistics exist for an empty matrix).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(
            "fit_standardizer requires a non-empty 2-D design matrix, got "
            f"shape {X_arr.shape}"
        )
    means = X_arr.mean(axis=0)
    stds = X_arr.std(axis=0)
    zero_variance = stds == 0.0
    stds = np.where(zero_variance, 1.0, stds)
    return means, stds


def apply_standardizer(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Standardize a design matrix with pre-computed (mean, std) statistics.

    Applies ``(X - means) / stds`` column-wise. With the
    :func:`fit_standardizer` zero-variance guard (``std == 1.0`` for a
    constant column), a zero-variance column standardizes to ``0.0`` for
    every row instead of dividing by zero.

    Args:
        X: The design matrix to standardize, ``(n, p)`` floats.
        means: The per-column means, length ``p``.
        stds: The per-column stds, length ``p`` (all non-zero; see
            :func:`fit_standardizer`).

    Returns:
        The standardized matrix as a ``(n, p)`` numpy array of ``float``.

    Raises:
        ValueError: If ``means``/``stds`` do not have exactly as many
            entries as ``X`` has columns (a shape mismatch would
            silently misalign columns).
    """
    X_arr = np.asarray(X, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(
            f"apply_standardizer requires a 2-D design matrix, got shape {X_arr.shape}"
        )
    if means_arr.shape[0] != X_arr.shape[1] or stds_arr.shape[0] != X_arr.shape[1]:
        raise ValueError(
            f"standardizer has {means_arr.shape[0]} means / {stds_arr.shape[0]} "
            f"stds but X has {X_arr.shape[1]} columns; they must match"
        )
    return (X_arr - means_arr) / stds_arr


def _match_id_for(
    team1_id: str,
    team2_id: str,
    date: str,
    matches_df: pd.DataFrame,
) -> str:
    """Resolve the unique match row for a (team1, team2, date) triple.

    Feature ``event_stage`` needs the *match*'s ``event_name`` (and
    therefore its ``match_id``), but the fixed feature-builder signature
    carries no ``match_id`` argument. This helper recovers it by locating
    the unique match row matching the two team ids and the as-of date.
    At v1 scale the ``(team1_id, team2_id, date)`` triple is unique per
    match (verified against the real data: every match has a distinct
    timestamp), so the lookup is unambiguous; the guard below fails
    loudly if a future dataset makes it ambiguous, rather than silently
    picking an arbitrary row.

    Args:
        team1_id: The queried team1's stable id ("A").
        team2_id: The queried team2's stable id ("B").
        date: The match's own date string (the as-of cutoff).
        matches_df: The materialised ``matches`` table (needs
            ``team1_id``, ``team2_id``, ``date``, ``match_id``).

    Returns:
        The ``match_id`` of the unique matching row, as a ``str``.

    Raises:
        KeyError: If ``matches_df`` lacks a required column (propagated
            from :func:`utils.asof.require_columns`).
        ValueError: If zero or more than one match row matches the
            triple (the event-stage feature would be undefined or
            ambiguous).
    """
    asof.require_columns(
        matches_df,
        (asof.TEAM1_ID_COL, asof.TEAM2_ID_COL, asof.DATE_COL, asof.MATCH_ID_COL),
        "matches_df",
    )
    matches = matches_df[
        (matches_df[asof.TEAM1_ID_COL] == team1_id)
        & (matches_df[asof.TEAM2_ID_COL] == team2_id)
        & (matches_df[asof.DATE_COL] == date)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one match for (team1_id={team1_id!r}, "
            f"team2_id={team2_id!r}, date={date!r}) to resolve the "
            f"event-stage match_id, found {len(matches)}"
        )
    return matches.iloc[0][asof.MATCH_ID_COL]


def build_feature_vector(
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
) -> np.ndarray:
    """Build the 15-feature vector for one map, in FEATURE_NAMES order.

    Computes every feature exactly as the consuming models' designs
    specify (section A of :mod:`models.ordinal_logit`'s design; the
    identical vector is used unchanged by :mod:`models.multinomial_logit`
    — that is the entire point of "identical splits" comparability),
    each as of the map's own match ``date`` (strict ``<`` boundary,
    inherited unchanged from each feature module's own ``utils.asof``
    usage — this module adds no new date filtering), with team-specific
    features expressed as A-minus-B differences (A = ``team1_id``, B =
    ``team2_id``). The missing-value fallback policy from this module's
    docstring is applied per-side *before* differencing. No
    normalization/standardization happens here — the returned values are
    raw feature values; standardization is :func:`fit_standardizer` /
    :func:`apply_standardizer`'s job (training-side only).

    The four features added by roadmap M38.5 (steps 12-15 below —
    ``attack_side_win_rate_diff``, ``defense_side_win_rate_diff``,
    ``signed_margin_diff``, ``first_blood_diff``) need **no
    missing-value fallback**: all three wrapped estimators
    (:func:`features.side_win_rate.team_map_side_rate`,
    :func:`features.signed_margin.team_map_signed_margin`,
    :func:`features.first_blood.team_map_first_blood_rate`) return a
    numeric posterior ``.mean`` that degrades to exactly the (non-``None``)
    prior when the team has zero as-of history on the map — never
    ``None``. A zero-history side therefore contributes ``prior`` and the
    A-minus-B difference of two zero-history sides is naturally ``0.0``,
    so no ``if ... is None`` fallback code is needed (unlike the
    ``acs_form_diff``/``days_since_diff``/``roster_decay_diff`` steps
    immediately above, which wrap M16/M17 estimators that do return
    ``None``).

    Args:
        team1_id: The queried team1's stable id ("A").
        team2_id: The queried team2's stable id ("B").
        map_name: The map to predict for (normalized inside each feature
            estimator, so case/whitespace never break a match).
        date: The as-of cutoff; rows dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needed by the M16/M17 features and the M38.4 first-blood
            feature; required because the fixed model interface does not
            pass it — see this module's docstring and the consuming
            models' :func:`models.ordinal_logit.make_model_fn` /
            :func:`models.multinomial_logit.make_model_fn`).

    Returns:
        A 15-vector numpy array of ``float`` in :data:`FEATURE_NAMES`
        order: ``map_win_rate_diff, elo_differential,
        close_map_freq_diff, ot_rate_diff, map_round_margin_variance,
        acs_form_diff, rating_form_diff, h2h_win_rate_centered,
        event_stage, days_since_diff, roster_decay_diff,
        attack_side_win_rate_diff, defense_side_win_rate_diff,
        signed_margin_diff, first_blood_diff``.

    Raises:
        ValueError: If a feature's as-of history contains a null/tied
            score or a null/unparseable date, if ``k``/window/threshold
            hyperparameters are invalid, if the match row for
            ``(team1_id, team2_id, date)`` is not unique (see
            :func:`_match_id_for`), if a ``player_map_stats`` team name
            matches neither side of its match, or if the two side names
            of a match collide (all propagated from the individual
            feature modules).
        KeyError: If any table lacks a required column (propagated from
            the individual feature modules / :func:`_match_id_for`).
        TypeError: If the query date is list-like (propagated from
            ``utils.asof`` via the feature modules).
        ConfigError: If ``map_name`` or an as-of map's ``map_name`` is
            not a string (propagated from
            :func:`utils.config.normalize_map_name` via the feature
            modules).
    """
    # 1. map_win_rate_diff — shrunk per-map win rate, A minus B.
    map_win_a = map_win_rate.team_map_win_rate(
        team1_id, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    ).mean
    map_win_b = map_win_rate.team_map_win_rate(
        team2_id, map_name, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    ).mean
    map_win_rate_diff = map_win_a - map_win_b

    # 2. elo_differential — signed A-minus-B already, from one shared
    #    league replay.
    elo_diff = elo.elo_differential(
        team1_id,
        team2_id,
        date,
        matches_df,
        maps_df,
        k=elo.DEFAULT_K,
        initial_rating=elo.INITIAL_RATING,
    ).differential

    # 3. close_map_freq_diff — unshrunk close-map frequency, A minus B.
    close_a = closeness.team_close_map_frequency(
        team1_id, date, matches_df, maps_df
    ).rate
    close_b = closeness.team_close_map_frequency(
        team2_id, date, matches_df, maps_df
    ).rate
    close_map_freq_diff = close_a - close_b

    # 4. ot_rate_diff — heavily-shrunk team OT rate, A minus B.
    ot_a = closeness.team_ot_rate(
        team1_id,
        date,
        matches_df,
        maps_df,
        k=closeness.DEFAULT_OT_K,
    ).mean
    ot_b = closeness.team_ot_rate(
        team2_id,
        date,
        matches_df,
        maps_df,
        k=closeness.DEFAULT_OT_K,
    ).mean
    ot_rate_diff = ot_a - ot_b

    # 5. map_round_margin_variance — match-level (not a team diff);
    #    NaN (n <= 1) replaced with 0.0 per the documented fallback.
    margin_var = closeness.map_round_margin_variance(
        map_name, date, matches_df, maps_df
    ).variance
    if math.isnan(margin_var):
        map_round_margin_variance = 0.0
    else:
        map_round_margin_variance = float(margin_var)

    # 6/7. acs_form_diff / rating_form_diff — recency-weighted form,
    #    A minus B, with the either-side-None -> 0.0 fallback.
    form_a = player_form.team_player_form(
        team1_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        n=player_form.DEFAULT_FORM_WINDOW,
        decay_rate=player_form.DEFAULT_DECAY_RATE,
    )
    form_b = player_form.team_player_form(
        team2_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        n=player_form.DEFAULT_FORM_WINDOW,
        decay_rate=player_form.DEFAULT_DECAY_RATE,
    )
    if form_a.acs.mean is None or form_b.acs.mean is None:
        acs_form_diff = 0.0
    else:
        acs_form_diff = form_a.acs.mean - form_b.acs.mean
    if form_a.rating.mean is None or form_b.rating.mean is None:
        rating_form_diff = 0.0
    else:
        rating_form_diff = form_a.rating.mean - form_b.rating.mean

    # 8. h2h_win_rate_centered — shrunk H2H mean minus the 0.5 prior
    #    (0 = no/even history, matching the estimator's own
    #    full-shrinkage default).
    h2h_mean = h2h_context.team_pair_h2h(
        team1_id,
        team2_id,
        date,
        matches_df,
        maps_df,
        k=h2h_context.DEFAULT_H2H_K,
    ).mean
    h2h_win_rate_centered = h2h_mean - h2h_context.H2H_PRIOR

    # 9. event_stage — the match's own stage (match-level int), resolved
    #    through the unique (team1, team2, date) match row.
    match_id = _match_id_for(team1_id, team2_id, date, matches_df)
    event_stage = float(
        h2h_context.match_event_stage(match_id, matches_df)
    )

    # 10. days_since_diff — rest gap, A minus B, with None (unseen team
    #     / no strictly-prior match) treated as 0 per the fallback.
    days_a = h2h_context.days_since_last_match(team1_id, date, matches_df)
    days_b = h2h_context.days_since_last_match(team2_id, date, matches_df)
    days_since_diff = (days_a if days_a is not None else 0) - (
        days_b if days_b is not None else 0
    )

    # 11. roster_decay_diff — post-change decay multiplier, A minus B,
    #     with None (changed is None/False) treated as 1.0 per the
    #     fallback.
    roster_a = h2h_context.team_roster_change(
        team1_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        jaccard_threshold=h2h_context.DEFAULT_JACCARD_THRESHOLD,
        half_life_days=h2h_context.DEFAULT_HALF_LIFE_DAYS,
    )
    roster_b = h2h_context.team_roster_change(
        team2_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        jaccard_threshold=h2h_context.DEFAULT_JACCARD_THRESHOLD,
        half_life_days=h2h_context.DEFAULT_HALF_LIFE_DAYS,
    )
    decay_a = roster_a.decay_multiplier if roster_a.decay_multiplier is not None else 1.0
    decay_b = roster_b.decay_multiplier if roster_b.decay_multiplier is not None else 1.0
    roster_decay_diff = decay_a - decay_b

    # 12/13. attack_side_win_rate_diff / defense_side_win_rate_diff —
    #     two-level shrunk per-map-phase round win rates (M38.2), A minus
    #     B, with the CV-chosen outer k (BEST_K_ATTACK / BEST_K_DEFENSE;
    #     NOT DEFAULT_K, which is the no-CV fallback). Zero history on
    #     the map degrades each side to its shrunk overall-phase prior,
    #     never None, so no fallback branch is needed (see the docstring).
    attack_a = side_win_rate.team_map_side_rate(
        team1_id,
        map_name,
        side_win_rate.PHASE_ATTACK,
        date,
        matches_df,
        maps_df,
        side_win_rate.BEST_K_ATTACK,
    ).mean
    attack_b = side_win_rate.team_map_side_rate(
        team2_id,
        map_name,
        side_win_rate.PHASE_ATTACK,
        date,
        matches_df,
        maps_df,
        side_win_rate.BEST_K_ATTACK,
    ).mean
    attack_side_win_rate_diff = attack_a - attack_b
    defense_a = side_win_rate.team_map_side_rate(
        team1_id,
        map_name,
        side_win_rate.PHASE_DEFENSE,
        date,
        matches_df,
        maps_df,
        side_win_rate.BEST_K_DEFENSE,
    ).mean
    defense_b = side_win_rate.team_map_side_rate(
        team2_id,
        map_name,
        side_win_rate.PHASE_DEFENSE,
        date,
        matches_df,
        maps_df,
        side_win_rate.BEST_K_DEFENSE,
    ).mean
    defense_side_win_rate_diff = defense_a - defense_b

    # 14. signed_margin_diff — two-level shrunk mean signed round margin
    #     (M38.3), A minus B, with the fixed outer k DEFAULT_MAP_K (M38.3
    #     ran no CV; DEFAULT_MAP_K is the real, documented constant).
    #     Zero history degrades each side to the 0.0 league prior.
    margin_a = signed_margin.team_map_signed_margin(
        team1_id,
        map_name,
        date,
        matches_df,
        maps_df,
        signed_margin.DEFAULT_MAP_K,
    ).mean
    margin_b = signed_margin.team_map_signed_margin(
        team2_id,
        map_name,
        date,
        matches_df,
        maps_df,
        signed_margin.DEFAULT_MAP_K,
    ).mean
    signed_margin_diff = margin_a - margin_b

    # 15. first_blood_diff — two-level shrunk first-blood rate (M38.4), A
    #     minus B, with the CV-chosen outer k first_blood.BEST_K. The one
    #     M38.5 feature that also reads player_map_stats_df (already a
    #     parameter). Zero history degrades each side to the 0.5
    #     structural prior.
    fb_a = first_blood.team_map_first_blood_rate(
        team1_id,
        map_name,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        first_blood.BEST_K,
    ).mean
    fb_b = first_blood.team_map_first_blood_rate(
        team2_id,
        map_name,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        first_blood.BEST_K,
    ).mean
    first_blood_diff = fb_a - fb_b

    return np.asarray(
        [
            map_win_rate_diff,
            elo_diff,
            close_map_freq_diff,
            ot_rate_diff,
            map_round_margin_variance,
            acs_form_diff,
            rating_form_diff,
            h2h_win_rate_centered,
            event_stage,
            days_since_diff,
            roster_decay_diff,
            attack_side_win_rate_diff,
            defense_side_win_rate_diff,
            signed_margin_diff,
            first_blood_diff,
        ],
        dtype=float,
    )

"""Four-way baseline outcome model (roadmap M18).

The crude benchmark every later model must beat. ``P(A wins map)`` is
derived from the two teams' *independent* Bayesian-shrunk per-map win
rates (:func:`features.map_win_rate.team_map_win_rate`), combined by
plain normalization rather than a head-to-head estimator, then split
into the four ordinal outcome categories — A-regulation, A-OT, B-OT,
B-regulation — by the league-wide global OT base rate
(:func:`features.closeness.global_ot_rate`).

Design decisions (recorded here, do not re-derive in later milestones):

- **Normalization, not log5.** ``features.map_win_rate.team_map_win_rate``
  returns *marginal* rates (each team's win rate on the map against the
  general field), not a head-to-head rate — there is no pairwise map
  win-rate estimator in this codebase. The crude combination chosen is
  ``p_win_a = mean_a / (mean_a + mean_b)`` (simpler than the log5
  formula ``(mean_a - mean_a*mean_b) / (mean_a + mean_b -
  2*mean_a*mean_b)``; log5 or a proper head-to-head estimator is left
  for a later milestone). When both means are exactly ``0.0`` (each
  team has zero wins over its as-of history on the queried map, i.e.
  ``prior == 0.0`` and full shrinkage) the division is undefined, so
  ``p_win_a`` falls back to ``0.5`` — the same least-committal default
  the rate estimators themselves use for "no evidence".
- **One symmetric global OT rate.** Per the roadmap ("the global OT
  base rate", not a team-specific rate), the OT split uses
  ``features.closeness.global_ot_rate`` directly — *not*
  ``team_ot_rate``, which is team-specific and already shrunk toward
  the same global rate (using it here would double up shrinkage). The
  same ``p_ot`` value is applied to both team1 and team2, so the OT
  split is symmetric by construction — a simplifying assumption a later,
  more sophisticated model would drop.
- **Output order matches ``drivers.labels.OUTCOME_LABELS``.**
  :meth:`FourWayPrediction.as_tuple` returns the four probabilities in
  exactly the order ``("A-regulation", "A-OT", "B-OT", "B-regulation")``
  (ordinals 0-3), so the tuple is directly usable by ``utils.scoring``'s
  ``rps``/``log_loss``/etc. and by the M19 evaluation harness without a
  remapping step. The vocabulary is documented here rather than
  imported because this module must not import from ``drivers/`` (module
  boundary standard: ``models/`` may depend on ``features/`` and
  ``utils/`` only). "A" = team1 (``team1_id``), "B" = team2, matching
  ``drivers/labels.py``'s "A and B are column positions, not team
  identities" convention.
- **One shared shrinkage strength.** ``k`` defaults to
  ``features.map_win_rate.DEFAULT_K`` (the module's documented "sane
  default when no CV has been run"); a caller who wants the
  CV-selected value runs ``features.map_win_rate.select_k`` and passes
  the result in. The same ``k`` is used for both team1 and team2 (no
  per-team shrinkage strength).
- **Leakage contract inherited, not reimplemented.** This module does
  no direct Parquet/file I/O; all history access flows through the two
  feature estimators, which themselves flow through ``utils.asof``
  (strict ``<`` as-of boundary, scores-derived wins, fail-loud on
  ties/null scores). A prediction as of a match's own date therefore
  never sees that match or any later one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features import closeness, map_win_rate

# The four outcome categories in ordinal order, mirroring
# ``drivers.labels.OUTCOME_LABELS`` (documented, deliberately *not*
# imported: ``models/`` may not depend on ``drivers/``). Index 0 is
# "A-regulation", 1 "A-OT", 2 "B-OT", 3 "B-regulation" — exactly the
# order :meth:`FourWayPrediction.as_tuple` returns and the order
# ``utils.scoring``'s metrics expect for ``K = 4``.
OUTCOME_LABELS = ("A-regulation", "A-OT", "B-OT", "B-regulation")


@dataclass(frozen=True)
class FourWayPrediction:
    """A predicted four-way outcome distribution for one map.

    Holds the four named category probabilities (``p_a_regulation``,
    ``p_a_ot``, ``p_b_ot``, ``p_b_regulation``) plus the intermediate
    values useful for interpretability/debugging: ``p_win_a`` (the
    post-normalization win probability for team1), ``p_ot`` (the global
    OT base rate the split used), and the two underlying
    ``features.map_win_rate.ShrunkWinRate`` objects (``shrunk_a`` for
    team1, ``shrunk_b`` for team2). The four probabilities always form
    a valid unit simplex (they sum to 1.0 and are non-negative);
    :meth:`as_tuple` exposes them in
    :data:`OUTCOME_LABELS` order ready for ``utils.scoring``.
    """

    p_a_regulation: float
    p_a_ot: float
    p_b_ot: float
    p_b_regulation: float
    p_win_a: float
    p_ot: float
    shrunk_a: map_win_rate.ShrunkWinRate
    shrunk_b: map_win_rate.ShrunkWinRate

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the four probabilities in OUTCOME_LABELS ordinal order.

        The tuple ``(p_a_regulation, p_a_ot, p_b_ot, p_b_regulation)``
        is exactly the ``K = 4`` probability vector ``utils.scoring``'s
        ``rps``/``log_loss``/``brier_score`` consume, with category
        indices matching ``drivers.labels.OUTCOME_LABELS`` (0 =
        A-regulation, 1 = A-OT, 2 = B-OT, 3 = B-regulation).

        Returns:
            A 4-tuple of ``float`` probabilities summing to ``1.0``,
            in :data:`OUTCOME_LABELS` order.

        Raises:
            Nothing.
        """
        return (self.p_a_regulation, self.p_a_ot, self.p_b_ot, self.p_b_regulation)


def predict_map_outcome(
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=map_win_rate.DEFAULT_K,
) -> FourWayPrediction:
    """Predict the four-way outcome distribution for one map.

    Combines the two teams' independent shrunk per-map win rates into
    ``P(team1 wins the map)`` by plain normalization
    (``mean_a / (mean_a + mean_b)``, with a ``0.5`` fallback when both
    means are exactly ``0.0``), then splits that win probability into
    the four ordinal categories by the league-wide global OT base rate
    at the same as-of cutoff:

    - ``p_a_regulation = p_win_a * (1 - p_ot)``
    - ``p_a_ot = p_win_a * p_ot``
    - ``p_b_ot = (1 - p_win_a) * p_ot``
    - ``p_b_regulation = (1 - p_win_a) * (1 - p_ot)``

    Both teams are estimated with the *same* shrinkage strength ``k``,
    and the same ``p_ot`` is applied symmetrically to both sides (see
    the module docstring's design-decision bullets). No file I/O
    happens here: both feature estimators obtain their history through
    ``utils.asof`` (strict ``<`` boundary), so predicting as of a
    match's own date never leaks that match's outcome.

    Args:
        team1_id: The queried team1's stable id ("A" in the outcome
            vocabulary).
        team2_id: The queried team2's stable id ("B" in the outcome
            vocabulary).
        map_name: The map to predict for; normalized via
            :func:`utils.config.normalize_map_name` inside the feature
            estimator, so ``"breeze"``/``" Breeze "`` both match
            ``"Breeze"``.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the columns
            the as-of layer requires).
        k: The shrinkage strength (effective prior sample size) shared
            by both teams' estimates; defaults to
            ``features.map_win_rate.DEFAULT_K`` (the documented sane
            default when no CV has been run). Must be a positive finite
            real number.

    Returns:
        A :class:`FourWayPrediction` whose four category probabilities
        sum to ``1.0`` (each in ``[0, 1]``), together with the
        intermediate ``p_win_a``, ``p_ot`` and the two
        ``ShrunkWinRate`` objects for interpretability.

    Raises:
        ValueError: If ``k`` is not a positive finite real number; if
            an as-of map has a null/NaN score or tied scores; or if the
            query date or a row date is null/unparseable/timezone-aware
            (all propagated from :func:`features.map_win_rate.team_map_win_rate`
            / :func:`features.closeness.global_ot_rate`).
        KeyError: If either table lacks a required column (propagated
            from the same calls; includes ``map_name``,
            ``team1_score``, ``team2_score``).
        TypeError: If the query date is list-like (propagated from the
            same calls).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    shrunk_a = map_win_rate.team_map_win_rate(
        team1_id, map_name, date, matches_df, maps_df, k
    )
    shrunk_b = map_win_rate.team_map_win_rate(
        team2_id, map_name, date, matches_df, maps_df, k
    )

    mean_a = shrunk_a.mean
    mean_b = shrunk_b.mean
    if mean_a + mean_b == 0.0:
        # Both means exactly 0.0: each team has zero wins on the queried
        # map over its as-of history (prior == 0.0 with full shrinkage),
        # so the normalization is undefined. Fall back to the same
        # least-committal 0.5 the rate estimators use for "no evidence".
        p_win_a = 0.5
    else:
        p_win_a = mean_a / (mean_a + mean_b)

    p_ot = closeness.global_ot_rate(date, matches_df, maps_df).rate

    return FourWayPrediction(
        p_a_regulation=p_win_a * (1.0 - p_ot),
        p_a_ot=p_win_a * p_ot,
        p_b_ot=(1.0 - p_win_a) * p_ot,
        p_b_regulation=(1.0 - p_win_a) * (1.0 - p_ot),
        p_win_a=p_win_a,
        p_ot=p_ot,
        shrunk_a=shrunk_a,
        shrunk_b=shrunk_b,
    )

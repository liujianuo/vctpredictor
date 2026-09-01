"""Flat series-scoreline baseline model (roadmap M32).

The crude benchmark the two-stage M31 pipeline must beat: a single
classifier predicting the *scoreline* of a whole series directly,
ignoring maps entirely. No per-map identity, no map pool, no veto/pick
data, and no per-map win-rate estimator (M13) — exactly one
team-vs-team strength number feeds the whole series, and the same
single-map win probability is applied identically to every map of the
series. It is built first in the M32 → M33a → M31 → M33b build order
so M33a's series evaluation harness has a real arm to score from day
one, and its output uses the ordinal scoreline vocabulary that
``utils.series_paths.series_outcome_order`` (M30) already fixed.

Design decisions (recorded here, do not re-derive in later milestones):

- **"Ignoring maps entirely" means one all-map strength number.**
  Concretely the estimator is
  :func:`features.map_win_rate.team_overall_win_rate` — the *all-map*
  as-of win rate (a leaf-level, as-of-safe estimator), *not*
  ``team_map_win_rate`` (which is per-map and belongs to the two-stage
  pipeline). A per-map or per-veto feature of any kind would defeat the
  purpose of this arm: the decisive later comparison (M33b) depends on
  this baseline being genuinely mapless, not a disguised per-map model
  averaged after the fact.
- **Plain normalization, not log5, not a fitted model.** The two
  teams' overall rates are combined exactly the way M18's
  ``models.four_way_baseline`` combines its map-level means:
  ``p_win_a = rate_a / (rate_a + rate_b)``, with a ``0.5`` fallback
  when both rates are exactly ``0.0`` (each team has zero wins over
  its as-of history). This keeps M32 genuinely "crude by design" with
  no training step, no hyperparameter, and no CV. A *fitted* flat
  scoreline classifier (e.g. a multinomial logit over the whole
  scoreline vocabulary) is explicitly out of scope for M32.
- **One probability, every map.** ``map_win_probs = [p_win_a] *
  best_of`` is fed to
  :func:`utils.series_paths.enumerate_series_paths`, which turns it
  into the exact terminal-scoreline distribution. The
  independence-across-maps assumption this implies is the whole
  meaning of "ignoring maps entirely": no map-specific or veto-order
  information can possibly vary ``p_win_a`` map-to-map here.
- **``best_of`` string parsing lives here.** ``matches.parquet``'s
  ``best_of`` column holds a string (``"Bo1"``/``"Bo3"``/``"Bo5"``),
  but ``utils.series_paths`` wants a plain positive odd ``int``, and
  no shared ``"Bo3"`` -> ``3`` parser exists anywhere in the repo
  (``models/greedy_veto_simulator.py`` and
  ``models/ancestral_veto_sampler.py`` key their own tables by the
  string form and never convert it). :func:`_parse_best_of` is a small
  local helper rather than a sideways reach into a ``models/``
  sibling, and is not promoted to a shared utility unless a later
  milestone (M33a/M33b) demonstrably needs the same parser.
- **New model-interface convention: ``SeriesModelFn``.** M19's
  ``evaluation.harness.ModelFn`` is per-map
  (``(team1_id, team2_id, map_name, date, matches_df, maps_df) ->
  4-vector``); a series-scoreline model needs a different, series-level
  signature. This module defines and documents the convention
  ``(team1_id, team2_id, best_of, date, matches_df, maps_df) ->
  Sequence[float]`` where the returned vector has exactly
  ``best_of + 1`` entries in
  ``utils.series_paths.series_outcome_order(int(best_of))`` order and
  ``best_of`` is the ``"Bo<N>"`` string as carried by
  ``matches.parquet`` (M33a will need to agree with this signature —
  it is chosen here now so M33a is built against a real, working arm
  rather than inventing the interface itself). Note that
  :func:`predict_series_outcome` returns the richer
  :class:`FlatSeriesPrediction`; ``SeriesModelFn``-conformant callers
  use its :meth:`FlatSeriesPrediction.as_tuple`.
- **Module boundary placement.** ``models/`` depends downward on
  ``features.map_win_rate`` and ``utils.series_paths`` only (both
  already-permitted edges; no ``models/ -> models/`` or
  ``models/ -> evaluation/`` edge). This mirrors
  ``models/four_way_baseline.py``'s precedent exactly: a stateless
  prediction dataclass + a ``predict_*`` function, no fitted
  parameters, no training driver, no CLI entry point. The
  ``MODEL_REGISTRY`` in ``drivers/evaluate.py`` is scoped to the four
  Stage-2 four-way-outcome factories and is deliberately **not**
  touched: M32 is series-level, not map-level, and M33a (not yet
  built) is what will own registry/wiring for series models.
- **Leakage contract inherited, not reimplemented.** No direct
  Parquet/file I/O happens here; all history access flows through
  :func:`features.map_win_rate.team_overall_win_rate`, which flows
  through ``utils.asof`` (strict ``<`` boundary, scores-derived wins,
  fail-loud on ties/null scores). A prediction as of a match's own
  date therefore never sees that match or any later one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from features import map_win_rate
from utils import series_paths

# The series-level model interface M33a's harness will consume: a
# callable taking the two team ids, the ``"Bo<N>"`` best-of string, the
# as-of date, and the full matches/maps tables, and returning the
# ``best_of + 1`` scoreline probabilities in
# ``utils.series_paths.series_outcome_order`` order (a plain sequence
# of floats). Documented here rather than in ``evaluation/`` because
# ``evaluation/harness.py`` is map-level and M33a's series harness does
# not exist yet.
SeriesModelFn = Callable[
    [str, str, str, str, pd.DataFrame, pd.DataFrame],
    Sequence[float],
]


def _parse_best_of(best_of: str) -> int:
    """Parse a ``"Bo<N>"`` series-length string into a plain odd int.

    Converts the ``"Bo1"``/``"Bo3"``/``"Bo5"`` strings carried by
    ``matches.parquet``'s ``best_of`` column (and any other
    ``"Bo<N>"`` string whose ``N`` is a positive odd integer) into the
    plain integer map count that ``utils.series_paths`` expects. The
    suffix must be exactly one or more decimal digits: anything else —
    a non-``"Bo"`` prefix, a non-numeric suffix, an even or non-positive
    map count, a non-string input — is rejected with ``ValueError``
    (or ``TypeError`` for a non-string input, which violates the
    annotated contract rather than being a malformed string) rather
    than silently coerced. This helper is local to this module
    by design: no shared ``"Bo3"`` -> ``3`` parser exists in the repo,
    and it is not promoted to a utility until a later milestone
    (M33a/M33b) demonstrably needs the same parser elsewhere.

    Args:
        best_of: The series-length string to parse; must be exactly a
            ``"Bo"`` prefix followed by decimal digits spelling a
            positive odd integer.

    Returns:
        The parsed map count as a plain ``int`` (``1`` for ``"Bo1"``,
        ``3`` for ``"Bo3"``, ``5`` for ``"Bo5"``).

    Raises:
        TypeError: If ``best_of`` is not a string at all (violates the
            annotated ``str`` contract).
        ValueError: If ``best_of`` is a string that does not start with
            ``"Bo"``, has a non-digit suffix (e.g. ``"BestOf3"``, an
            empty suffix, a trailing-space suffix), or spells an even
            or non-positive map count (e.g. ``"Bo2"``, ``"Bo0"``).
    """
    if not isinstance(best_of, str):
        raise TypeError(
            f"best_of must be a 'Bo<N>' string, got {best_of!r}"
        )
    if not best_of.startswith("Bo") or len(best_of) <= 2:
        raise ValueError(
            f"best_of must be a 'Bo<N>' string like 'Bo3', got {best_of!r}"
        )
    suffix = best_of[2:]
    if not suffix.isdigit():
        raise ValueError(
            f"best_of must be a 'Bo<N>' string with a numeric suffix, "
            f"got {best_of!r}"
        )
    n = int(suffix)
    if n < 1:
        raise ValueError(
            f"best_of must be a positive odd map count, got {best_of!r}"
        )
    if n % 2 == 0:
        raise ValueError(
            f"best_of must be odd (an even map count cannot produce a "
            f"guaranteed series winner), got {best_of!r}"
        )
    return n


@dataclass(frozen=True)
class FlatSeriesPrediction:
    """A predicted series-scoreline distribution for one match.

    Holds the ``best_of + 1`` terminal-scoreline probabilities in the
    canonical ordinal order fixed by
    :func:`utils.series_paths.series_outcome_order` (for Bo3:
    ``(2,0), (2,1), (1,2), (0,2)``), plus the intermediate values
    useful for interpretability/debugging: ``p_win_a`` (the single
    per-map win probability applied to every map), the parsed
    ``best_of`` map count, the ``outcome_order`` tuple itself, and the
    two underlying :class:`features.map_win_rate.OverallWinRate`
    objects (``overall_a`` for team1, ``overall_b`` for team2) whose
    rates were normalized into ``p_win_a``. The probabilities always
    form a valid unit simplex (they sum to ``1.0`` and are
    non-negative); :meth:`as_tuple` exposes them as a plain sequence
    ready for the :data:`SeriesModelFn` convention and, later, for
    ``utils.scoring``'s metrics.
    """

    probabilities: tuple[float, ...]
    best_of: int
    outcome_order: tuple[tuple[int, int], ...]
    p_win_a: float
    overall_a: map_win_rate.OverallWinRate
    overall_b: map_win_rate.OverallWinRate

    def as_tuple(self) -> tuple[float, ...]:
        """Return the scoreline probabilities in ordinal order.

        The tuple returned is exactly ``self.probabilities`` — the
        ``best_of + 1`` floats in
        :func:`utils.series_paths.series_outcome_order` order — so a
        caller consuming the :data:`SeriesModelFn` convention (or, once
        M33a exists, the series evaluation harness) gets the
        ready-to-score probability vector without a remapping step.
        Kept as a method for parity with
        :meth:`models.four_way_baseline.FourWayPrediction.as_tuple`.

        Returns:
            A tuple of ``best_of + 1`` non-negative floats summing to
            ``1.0``, in the ordinal order of ``self.outcome_order``.

        Raises:
            Nothing.
        """
        return self.probabilities


def predict_series_outcome(
    team1_id: str,
    team2_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> FlatSeriesPrediction:
    """Predict the full series-scoreline distribution for one match.

    The flat, mapless baseline: it computes each team's overall
    (all-map) as-of win rate via
    :func:`features.map_win_rate.team_overall_win_rate`, combines the
    two rates by plain normalization (``rate_a / (rate_a + rate_b)``
    with a ``0.5`` fallback when both rates are exactly ``0.0`` — the
    same convention M18's four-way baseline established), parses the
    ``"Bo<N>"`` ``best_of`` string via :func:`_parse_best_of`, applies
    the resulting single per-map win probability ``p_win_a`` identically
    to every map of the series (``map_win_probs = [p_win_a] *
    best_of``), and feeds that vector to
    :func:`utils.series_paths.series_probabilities_in_order` to obtain
    the exact terminal-scoreline distribution in
    ``series_outcome_order`` order. No file I/O happens here: the
    feature estimator obtains its history through ``utils.asof``
    (strict ``<`` boundary), so predicting as of a match's own date
    never leaks that match's outcome.

    Args:
        team1_id: The queried team1's stable id ("A" in the series
            scoreline vocabulary; the first element of every
            ``(a_wins, b_wins)`` scoreline).
        team2_id: The queried team2's stable id ("B" in the series
            scoreline vocabulary).
        best_of: The series length as the ``"Bo<N>"`` string carried by
            ``matches.parquet`` (e.g. ``"Bo3"``, ``"Bo5"``, ``"Bo1"``);
            parsed to a plain odd int by :func:`_parse_best_of`.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner``, ``team1_score``, ``team2_score`` in addition to
            the columns the as-of layer requires).

    Returns:
        A :class:`FlatSeriesPrediction` whose ``best_of + 1``
        ``probabilities`` sum to ``1.0`` (each in ``[0, 1]``), in
        :func:`utils.series_paths.series_outcome_order` order, together
        with the parsed ``best_of``, the ``outcome_order`` tuple, the
        single per-map ``p_win_a``, and the two underlying
        ``OverallWinRate`` objects for interpretability.

    Raises:
        ValueError: If ``best_of`` is not a valid ``"Bo<N>"`` string
            (see :func:`_parse_best_of`); if an as-of map has a
            null/NaN score or tied scores; or if the query date or a
            row date is null/unparseable/timezone-aware (all propagated
            from :func:`features.map_win_rate.team_overall_win_rate`).
        KeyError: If either table lacks a required column (propagated
            from :func:`features.map_win_rate.team_overall_win_rate`).
        TypeError: If the query date is list-like (propagated from
            :func:`features.map_win_rate.team_overall_win_rate`).
    """
    overall_a = map_win_rate.team_overall_win_rate(
        team1_id, date, matches_df, maps_df
    )
    overall_b = map_win_rate.team_overall_win_rate(
        team2_id, date, matches_df, maps_df
    )

    rate_a = overall_a.rate
    rate_b = overall_b.rate
    if rate_a + rate_b == 0.0:
        # Both rates exactly 0.0: each team has zero wins over its as-of
        # history, so the normalization is undefined. Fall back to the
        # same least-committal 0.5 the rate estimator itself uses for
        # "no evidence" (mirroring M18's four-way baseline).
        p_win_a = 0.5
    else:
        p_win_a = rate_a / (rate_a + rate_b)

    parsed_best_of = _parse_best_of(best_of)
    outcome_order = series_paths.series_outcome_order(parsed_best_of)
    probabilities = series_paths.series_probabilities_in_order(
        [p_win_a] * parsed_best_of, parsed_best_of
    )

    return FlatSeriesPrediction(
        probabilities=tuple(probabilities),
        best_of=parsed_best_of,
        outcome_order=outcome_order,
        p_win_a=p_win_a,
        overall_a=overall_a,
        overall_b=overall_b,
    )

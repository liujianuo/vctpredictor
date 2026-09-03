"""Closeness and overtime features (roadmap M15).

A pure in-memory library producing four features from the materialised
matches/maps tables, each queried as-of a cutoff date:

- ``team_close_map_frequency`` — a team's unshrunk frequency of "close"
  maps (absolute round margin ``<= 2``) over its as-of history;
- ``team_ot_rate`` — a team's *heavily-shrunk* overtime rate, pulled
  toward a global pooled OT base rate (:func:`global_ot_rate`);
- ``map_round_margin_variance`` — the per-map-name historical *sample*
  variance (``ddof=1``) of absolute round margins, computed over all
  teams' as-of maps on that map;
- ``map_ot_rate`` — the per-map-name *heavily-shrunk* overtime rate,
  pulled toward the same global pooled OT base rate
  (:func:`global_ot_rate`), over all teams' as-of maps on that map.

Like the rest of ``features/``, this module has no CLI, no ``argparse``
entry point, and no file I/O of its own — it operates on the
already-materialised ``matches_df``/``maps_df`` DataFrames a caller
passes in.

**Leakage contract (the hard requirement).** Every feature's history
originates from ``utils.asof`` — either ``maps_as_of`` (single-team) for
the per-team features, or this module's own league-wide filter
:func:`_league_maps_as_of` (all teams, mirroring ``features.elo.py``'s
documented pattern) for the global pool — never ``pd.read_parquet`` on
``matches.parquet``/``maps.parquet`` directly. The strict ``<`` boundary
is enforced by ``utils.asof``'s date parsing (reused here, not
reimplemented), so a map dated equal to or after the query date never
enters any estimate, for any team.

**Resolved design decisions (do not re-derive in later milestones):**

- **Recompute margin/OT from raw scores, never join ``labels.parquet``.**
  ``drivers/labels.py`` writes a separate ``labels.parquet`` keyed by
  ``(match_id, map_index)`` with no ``team_id``/``team_is_team1``
  orientation column; joining it into a team-oriented as-of query would
  need the orientation re-derived anyway and would add a second read
  path outside ``utils.asof`` (a leakage gap if the two tables ever
  drift). This module therefore derives ``abs(margin)`` and the OT flag
  directly from ``team1_score``/``team2_score`` using the *same* two
  formulas ``drivers.labels.py::compute_outcome`` establishes as
  canonical — signed margin ``team1_score - team2_score`` (we use its
  absolute value, which orientation does not change; "closeness" is
  symmetric) and OT criterion ``min(team1_score, team2_score) >= 12``.
  This mirrors how ``features.elo.py`` independently re-derives
  ``drivers/labels.py``'s winner-from-scores rule: features derive facts
  from raw scores via ``utils.asof``, never by reading a driver-produced
  table.
- **"Close map" = ``abs(team1_score - team2_score) <= 2``** (roadmap's
  "round margin <= 2" read as an absolute value — a close win by either
  side is close).
- **Global OT base rate pools every team's as-of maps, not just the
  queried team.** Roadmap M15 says the global rate should come from "a
  wider pooled slice than the v1 era". **Known limitation, recorded not
  silently resolved:** only v1 data exists in this repository today, so
  "wider than per-team" is realised as "the whole league as-of the query
  date" — genuinely wider than one team's few dozen maps (a real
  variance reduction), but **not** wider than the v1 era itself. A true
  wider-than-v1-era pool is blocked on ingesting additional historical
  data (a future milestone); the prior returned by
  :func:`global_ot_rate` is named/documented so a future contributor can
  swap in a genuinely wider slice without changing the shrinkage math.
- **Per-team close-map frequency is a point estimate, unshrunk** (the
  roadmap only asks for shrinkage on the OT rate). Empty history yields
  ``rate = 0.0`` ("no evidence of closeness"), deliberately unlike
  ``features.map_win_rate.OverallWinRate``'s ``0.5`` default: a win-rate
  prior is a coin-flip default, but a frequency of a rare event with no
  evidence is most honestly 0.0. The ``close_maps``/``total_maps``
  counts ride alongside the rate so a caller can judge sample size.
- **Shrunk OT rate reuses ``features.map_win_rate``'s shrinkage formula**
  ``mean = (events + k*prior) / (games + k)`` with the same Beta
  posterior field shape (``alpha``/``beta``/``mean``/``variance`` plus
  ``events``/``games``/``prior``/``raw_rate``), renaming the count field
  from ``wins`` to ``events`` because this counts OT occurrences, not
  wins. ``raw_rate = prior`` exactly when ``games == 0`` (full
  shrinkage), matching ``ShrunkWinRate``.
- **``DEFAULT_OT_K`` is a documented fixed "heavy" shrinkage constant.**
  OT is rare (global rate ~0.12 in v1) so a team's own OT count is tiny
  (a team plays ~30 maps -> ~3 OT events), and the roadmap asks for
  *heavy* shrinkage with no CV routine. M13 cross-validates its win-rate
  ``k`` via ``select_k`` and reports ``best_k = 100`` on real v1; an
  order of magnitude above that is **1000**, which is also ~4x the total
  pooled map count (244) — i.e. the global prior carries the weight of
  the whole league's worth of observations, so a team's handful of OT
  events can barely move the estimate. This is a judgment call isolated
  as :data:`DEFAULT_OT_K`, not a magic number, and every caller can
  override ``k``.
- **Per-map OT rate (M27) reuses the same heavy-shrinkage math, with the
  same constant.** :func:`map_ot_rate` counts ``events``/``games`` over
  the league-wide as-of pool restricted to one ``map_name`` (the same
  :func:`_league_maps_as_of` + ``normalize_map_name`` filter pattern as
  ``map_round_margin_variance``) and shrinks toward the same global
  pooled prior (:func:`global_ot_rate`). A single map's historical
  sample is even narrower than one team's (a map appears only ~30-40
  times in v1), so the same ``k = 1000.0`` heavy constant is at least as
  appropriate there — the prior's weight relative to a per-map sample is
  even larger. The map-level default is exposed as a separate named
  constant :data:`DEFAULT_MAP_OT_K` (equal to :data:`DEFAULT_OT_K`) so
  the choice is documented and overridable per caller.
- **Per-map margin variance uses sample variance (``ddof=1``),** matching
  ``ShrunkWinRate``'s Beta variance being a "true" (not biased
  population) variance. Degenerate cases are explicit, not left to
  numpy's NaN-producing defaults: ``n == 0`` -> ``float("nan")`` (no
  observations, not "zero variance" — asserting 0.0 would falsely claim
  confidence) and ``n == 1`` -> ``float("nan")`` (sample variance is
  undefined: division by ``n - 1 == 0``). ``n`` is exposed alongside
  ``variance`` so callers can judge reliability.
- **Null-score defensiveness matches ``features.elo.py``'s fail-loud
  convention (task 017, deliverable B).** Every row-level score
  comparison goes through :func:`_margin_and_ot`, which rejects a
  null/NaN score with ``ValueError`` *before* any arithmetic — NaN
  compares neither equal nor greater to anything, so an unguarded
  comparison would silently yield a wrong ``False``/``0``. Verified 0
  such rows exist in real ``data/v1``, so this is defensive-only.

**Data-shape findings (re-derived against real ``data/v1``, item A1):**

- At a mid-dataset as-of cutoff (``2026-07-18T13:25:00``) the league-wide
  as-of filter survives 125 finished maps, of which 28 are close
  (``abs(margin) <= 2``, rate 0.224) and 15 are OT
  (``min(team1_score, team2_score) >= 12``, rate 0.12) — both
  non-degenerate (not all-zero, not all-maps), so the fixtures in the
  test suite are not vacuous.
- ``maps.parquet["winner"]`` holds display team names (e.g. ``FNATIC``),
  never a stable ``"team1"``/``"team2"`` marker; margin/OT are therefore
  derived from scores, never the ``winner`` string (see the first
  resolved-design-decision bullet).

**Module note on the league-wide filter.** ``utils.asof`` is single-team
by design, so this module writes its own small league-wide filter
(:func:`_league_maps_as_of`) that mirrors ``matches_as_of``'s
completed + strictly-before masks *without* the team-id mask, then
inner-joins to ``maps_df`` with the finished-map (``winner.notna()``)
filter. The date parsing reuses ``utils.asof``'s public parse helpers
(``parse_query_date`` / ``parse_date_column`` / ``require_columns``)
exactly as ``features.elo.py`` already does — a documented, already-
precedented ``features`` -> ``utils`` dependency (feature modules may
depend downward on genuine ``utils/`` utilities), not a fresh
reimplementation, and not a driver import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import asof, config

# Close-map threshold: a map is "close" when the absolute round margin
# is at most this many rounds (roadmap M15's "<= 2").
CLOSE_MARGIN = 2

# OT threshold: a map went to overtime when the losing side reached at
# least this many rounds (the same criterion drivers/labels.py uses).
OT_MIN_SCORE = 12

# Default shrinkage strength for the team OT rate. Heavily shrunk: one
# order of magnitude above M13's cross-validated win-rate ``k`` of 100,
# and ~4x the total pooled v1 map count (244). See the module docstring's
# "DEFAULT_OT_K" bullet for the full justification.
DEFAULT_OT_K = 1000.0

# Default shrinkage strength for the per-map OT rate (M27). Identical
# value to DEFAULT_OT_K: a single map's historical sample is even
# narrower than one team's (a map appears ~30-40 times in v1), so the
# same "heavy" prior weight is at least as appropriate there; kept as
# its own named constant so the choice is explicit and overridable.
DEFAULT_MAP_OT_K = DEFAULT_OT_K

# Column names this module reads from the M8 maps table. ``match_id``,
# ``date`` and ``status`` come from the matches table via ``utils.asof``'s
# constants; ``winner`` is the map-completion signal and ``map_name`` /
# ``team1_score`` / ``team2_score`` are M8's maps-table columns, named
# here once so functions and tests share one spelling.
TEAM1_SCORE_COL = "team1_score"
TEAM2_SCORE_COL = "team2_score"
MAP_NAME_COL = "map_name"

# The columns :func:`_league_maps_as_of` needs on each table. Unlike
# ``features.elo.py``'s equivalent, team-id columns are *not* required:
# the closeness pool is over all teams and never needs to orient a score
# to a particular side.
_MATCHES_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.DATE_COL,
    asof.STATUS_COL,
)
_MAPS_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.WINNER_COL,
    MAP_NAME_COL,
    TEAM1_SCORE_COL,
    TEAM2_SCORE_COL,
)


@dataclass(frozen=True)
class CloseMapFrequency:
    """A team's unshrunk frequency of close maps over its as-of history.

    ``rate`` is ``close_maps / total_maps`` when ``total_maps > 0`` and
    exactly ``0.0`` when ``total_maps == 0`` ("no evidence of
    closeness", unlike a win-rate prior's coin-flip ``0.5`` default — a
    rare-event frequency with no observations is most honestly 0.0).
    The raw ``close_maps``/``total_maps`` counts ride alongside the rate
    so a caller can judge sample size itself.
    """

    close_maps: int
    total_maps: int
    rate: float


@dataclass(frozen=True)
class GlobalOTRate:
    """The league-wide pooled overtime base rate as of a cutoff.

    ``rate`` is ``events / games`` when ``games > 0`` and exactly
    ``0.0`` when ``games == 0`` (the empty-history case, reachable only
    at/near the dawn of the dataset; 0.0 is the least-committal default
    for a rare event known a priori to be asymmetric, not a coin flip).
    ``events`` counts OT maps and ``games`` all finished maps in the
    pool.
    """

    events: int
    games: int
    rate: float


@dataclass(frozen=True)
class ShrunkOTRate:
    """The Beta posterior for a team's overtime rate.

    ``alpha`` / ``beta`` are the full posterior parameters
    ``Beta(events + k*prior, (games - events) + k*(1 - prior))`` —
    exposed, not just the point estimate, so callers can read off the
    uncertainty. ``mean`` is the shrinkage point estimate
    ``alpha / (alpha + beta)`` (roadmap's ``(events + k*prior) /
    (games + k)``); ``variance`` is the Beta variance
    ``alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))``. ``events`` /
    ``games`` are the team's OT-map and total-map counts over its as-of
    history; ``prior`` is the global pooled OT base rate
    (:func:`global_ot_rate`) fed in as the prior mean; ``raw_rate`` is
    the unshrunk team rate ``events / games``, or exactly ``prior`` when
    ``games == 0`` (full shrinkage — no raw sample to compare against).
    """

    events: int
    games: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


@dataclass(frozen=True)
class MapMarginVariance:
    """The historical sample variance of absolute round margins on one map.

    Computed over all teams' as-of maps on the named map (a map's
    difficulty/swinginess is a property of the map, not of one team).
    ``n`` is the number of maps that went into the estimate (exposed so
    a caller can judge reliability); ``variance`` is the sample variance
    (``ddof=1``), or exactly ``float("nan")`` when ``n <= 1`` (no
    observations, or a single observation from which sample variance is
    undefined).
    """

    n: int
    variance: float


def _validate_k(k) -> float:
    """Validate the shrinkage strength ``k`` and return it as a float.

    ``k`` is the effective prior sample size for the OT-rate shrinkage:
    ``alpha0 = k*prior`` and ``beta0 = k*(1 - prior)``, so
    ``alpha0 + beta0 = k``. It must be a positive finite real number.
    ``k <= 0`` is rejected rather than tolerated: ``k == 0`` would drop
    the prior term entirely (degenerating to the raw rate) and negative
    ``k`` would produce negative pseudo-counts.

    Args:
        k: The shrinkage strength. Any real number (``int``/``float``/
            numpy scalar) is coerced to ``float``.

    Returns:
        ``k`` as a ``float``.

    Raises:
        ValueError: If ``k`` cannot be coerced to a ``float``, or if the
            result is NaN, infinite, or ``<= 0``.
    """
    try:
        value = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k must be a positive real number, got {k!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"k must be a positive finite real number, got {k!r}")
    return value


def _margin_and_ot(team1_score, team2_score):
    """Derive abs margin, close flag and OT flag from two score columns.

    The single shared row-level derivation behind every public feature
    in this module: ``abs_margin = abs(team1_score - team2_score)``,
    ``is_close = abs_margin <= CLOSE_MARGIN``, and
    ``is_ot = min(team1_score, team2_score) >= OT_MIN_SCORE``. The
    null-score check runs *before* any arithmetic so a NaN can never
    silently propagate into a ``False`` comparison (IEEE-754 NaN
    compares neither equal nor greater to anything) — the exact bug
    class task 017's deliverable B fixes in ``features.elo.py``. Margin is
    team1-oriented (``team1_score - team2_score``), but only its absolute
    value is exposed because "closeness" is symmetric and orientation
    does not change ``abs()``.

    Args:
        team1_score: Rounds won by team 1. A scalar, or an array-like /
            ``pandas.Series`` of scores (one per map).
        team2_score: Rounds won by team 2, same shape as ``team1_score``.

    Returns:
        A ``(abs_margin, is_close, is_ot)`` tuple of numpy arrays (or
        numpy scalars for scalar input): ``abs_margin`` the absolute
        round margin as float; ``is_close`` the ``abs_margin <= 2``
        boolean; ``is_ot`` the ``min(team1_score, team2_score) >= 12``
        boolean.

    Raises:
        ValueError: If any ``team1_score`` or ``team2_score`` value is
            null/NaN (checked first, before the arithmetic).
    """
    t1 = np.asarray(team1_score, dtype=float)
    t2 = np.asarray(team2_score, dtype=float)
    null_mask = np.isnan(t1) | np.isnan(t2)
    if null_mask.any():
        raise ValueError(
            f"{int(null_mask.sum())} map(s) have a null/NaN score "
            "(team1_score or team2_score is missing); cannot derive "
            "margin/OT from a missing score"
        )
    abs_margin = np.abs(t1 - t2)
    is_close = abs_margin <= CLOSE_MARGIN
    is_ot = np.minimum(t1, t2) >= OT_MIN_SCORE
    return abs_margin, is_close, is_ot


def _league_maps_as_of(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return every finished map of every team strictly before a cutoff.

    The league-wide as-of filter the global pool needs. It mirrors
    ``utils.asof.matches_as_of``'s two boolean masks (completed,
    strictly-before) *without* the per-team mask — the global OT prior
    and per-map variance are league-wide properties, not team-specific —
    then inner-joins to ``maps_df`` with the same finished-map
    (``winner.notna()``) filter ``utils.asof.maps_as_of`` applies.

    Date parsing reuses ``utils.asof``'s public parse helpers
    (``parse_query_date`` / ``parse_date_column`` /
    ``require_columns``) rather than duplicating them, so the strict-``<``
    boundary, null-date rejection and timezone-naive-only rules are
    byte-for-byte identical to every other as-of consumer. The shared
    access is a deliberate, documented choice over a reimplementation
    (the same tradeoff ``features.elo.py`` already makes).

    Args:
        date: The as-of cutoff; rows dated equal to or after this are
            excluded (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``status``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner``, ``map_name``, ``team1_score``, ``team2_score``).

    Returns:
        A ``pandas.DataFrame`` with columns ``match_id``, ``map_name``,
        ``team1_score``, ``team2_score`` and ``date`` — exactly the rows
        whose parent match is completed and strictly before the cutoff
        *and* whose own ``winner`` is non-null (a finished map). The
        output is unsorted and preserves no input index (the merge
        resets it).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from ``utils.asof.require_columns``).
        ValueError: If the filtered ``matches`` frame contains duplicate
            ``match_id`` values (the join would fan out and duplicate
            map rows); or if the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            ``utils.asof.parse_query_date`` /
            ``utils.asof.cached_parsed_date_column``).
        TypeError: If the query date is list-like (propagated from
            ``utils.asof.parse_query_date``).
    """
    asof.require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    asof.require_columns(maps_df, _MAPS_REQUIRED, "maps_df")

    parsed_dates = asof.cached_parsed_date_column(matches_df)
    query = asof.parse_query_date(date)

    is_completed = matches_df[asof.STATUS_COL] == asof.COMPLETED_STATUS
    is_before = parsed_dates < query
    matches = matches_df[is_completed & is_before]

    if not matches[asof.MATCH_ID_COL].is_unique:
        duplicates = (
            matches.loc[
                matches[asof.MATCH_ID_COL].duplicated(keep=False),
                asof.MATCH_ID_COL,
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "matches_df contains duplicate match_id value(s) "
            f"{duplicates} after as-of filtering; the maps join would "
            "fan out and duplicate map rows"
        )

    finished_maps = maps_df[maps_df[asof.WINNER_COL].notna()][
        [asof.MATCH_ID_COL, MAP_NAME_COL, TEAM1_SCORE_COL, TEAM2_SCORE_COL]
    ]
    join_frame = matches[[asof.MATCH_ID_COL, asof.DATE_COL]]
    return finished_maps.merge(join_frame, on=asof.MATCH_ID_COL, how="inner")


def team_close_map_frequency(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> CloseMapFrequency:
    """Return a team's unshrunk frequency of close maps as of a cutoff.

    Fetches the team's completed, strictly-earlier maps through
    :func:`utils.asof.maps_as_of` (never by reading the Parquet tables
    directly) and derives, per row, the absolute round margin via
    :func:`_margin_and_ot`. A map is "close" when
    ``abs(team1_score - team2_score) <= 2``; orientation does not
    matter because the absolute value is symmetric.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``team1_score``
            and ``team2_score`` in addition to the columns ``maps_as_of``
            already requires).

    Returns:
        A :class:`CloseMapFrequency` with ``close_maps``/``total_maps``
        counted over every as-of map, and ``rate = close_maps /
        total_maps`` — or exactly ``0.0`` when ``total_maps == 0``
        (unseen team, or a cutoff before the team's first completed
        match).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of` /
            :func:`utils.asof.require_columns`; includes
            ``team1_score``, ``team2_score``).
        ValueError: If an as-of map has a null/NaN score (see
            :func:`_margin_and_ot`); or if the query date or a row date
            is null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
    """
    asof.require_columns(maps_df, (TEAM1_SCORE_COL, TEAM2_SCORE_COL), "maps_df")
    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)

    _, is_close, _ = _margin_and_ot(maps[TEAM1_SCORE_COL], maps[TEAM2_SCORE_COL])
    total_maps = len(maps)
    close_maps = int(is_close.sum())
    rate = close_maps / total_maps if total_maps else 0.0
    return CloseMapFrequency(close_maps=close_maps, total_maps=total_maps, rate=rate)


def global_ot_rate(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> GlobalOTRate:
    """Return the league-wide pooled overtime base rate as of a cutoff.

    Computes the global OT base rate over *every* team's as-of maps (the
    league-wide helper :func:`_league_maps_as_of`), not just one team's —
    the "wider pooled slice" the roadmap asks for, realised within the
    only data that exists (v1). See the module docstring's
    "Global OT base rate" bullet for the recorded limitation that this
    is wider than per-team but not wider than the v1 era itself.

    Args:
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.

    Returns:
        A :class:`GlobalOTRate` with ``events`` (OT maps) and ``games``
        (all finished maps) counted over the league-wide as-of pool, and
        ``rate = events / games`` — or exactly ``0.0`` when ``games ==
        0``.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_league_maps_as_of`).
        ValueError: If an as-of map has a null/NaN score (see
            :func:`_margin_and_ot`); or if the query date or a row date
            is null/unparseable/timezone-aware (propagated from
            :func:`_league_maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_league_maps_as_of`).
    """
    maps = _league_maps_as_of(date, matches_df, maps_df)

    _, _, is_ot = _margin_and_ot(maps[TEAM1_SCORE_COL], maps[TEAM2_SCORE_COL])
    games = len(maps)
    events = int(is_ot.sum())
    rate = events / games if games else 0.0
    return GlobalOTRate(events=events, games=games, rate=rate)


def team_ot_rate(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=DEFAULT_OT_K,
) -> ShrunkOTRate:
    """Return a team's heavily-shrunk overtime rate as of a cutoff.

    The shrinkage estimator. It fetches the team's own as-of maps via
    :func:`utils.asof.maps_as_of` (its ``events``/``games`` inputs) and
    the global pooled OT base rate via :func:`global_ot_rate` at the
    *same* date (its ``prior``), then applies the Beta posterior
    ``mean = (events + k*prior) / (games + k)``. With ``games == 0`` the
    formula degrades to ``mean = prior`` exactly (full shrinkage — the
    correct behaviour, not a special case). ``k`` defaults to the heavy
    :data:`DEFAULT_OT_K` (one order of magnitude above M13's
    cross-validated win-rate ``k``), so the team's tiny OT sample is
    pulled hard toward the global rate.

    Args:
        team_id: The queried team's stable id.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``team1_score``
            and ``team2_score`` in addition to the columns ``maps_as_of``
            already requires).
        k: The shrinkage strength (effective prior sample size); must be
            a positive finite real number (see :func:`_validate_k`).

    Returns:
        A :class:`ShrunkOTRate` with the team's ``events``/``games``, the
        global ``prior``, the unshrunk ``raw_rate`` (equal to ``prior``
        when ``games == 0``), and the posterior ``alpha``, ``beta``,
        ``mean`` and ``variance``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`_validate_k`); if an as-of map has a null/NaN score
            (see :func:`_margin_and_ot`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.maps_as_of` / :func:`global_ot_rate`).
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of` /
            :func:`global_ot_rate`; includes ``team1_score``,
            ``team2_score``).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of` / :func:`global_ot_rate`).
    """
    k_value = _validate_k(k)
    asof.require_columns(maps_df, (TEAM1_SCORE_COL, TEAM2_SCORE_COL), "maps_df")
    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)

    _, _, is_ot = _margin_and_ot(maps[TEAM1_SCORE_COL], maps[TEAM2_SCORE_COL])
    events = int(is_ot.sum())
    games = len(maps)
    prior = global_ot_rate(date, matches_df, maps_df).rate
    raw_rate = events / games if games else prior

    alpha = events + k_value * prior
    beta = (games - events) + k_value * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))

    return ShrunkOTRate(
        events=events,
        games=games,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def map_round_margin_variance(
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> MapMarginVariance:
    """Return the historical sample variance of absolute round margins on one map.

    Computed over *all teams'* as-of maps on the named map (via
    :func:`_league_maps_as_of`), filtered by ``map_name`` normalized
    through :func:`utils.config.normalize_map_name` (matching
    ``features.map_win_rate``'s established convention for this exact
    concern, so ``"breeze"``/``" Breeze "`` both match ``"Breeze"``).
    The variance is the sample variance (``ddof=1``) of the absolute
    round margins — the same "true variance, not biased population
    variance" convention as ``ShrunkWinRate``'s Beta variance.
    Degenerate cases are explicit: ``n == 0`` or ``n == 1`` yields
    ``variance == float("nan")`` (no observations, or a single
    observation from which sample variance is undefined), rather than a
    numpy runtime warning.

    Args:
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the columns
            the league-wide filter already requires).

    Returns:
        A :class:`MapMarginVariance` with ``n`` (the number of as-of maps
        on the named map that went into the estimate) and ``variance``
        (the ``ddof=1`` sample variance of absolute margins, or
        ``float("nan")`` when ``n <= 1``).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_league_maps_as_of`).
        ValueError: If an as-of map has a null/NaN score (see
            :func:`_margin_and_ot`); or if the query date or a row date
            is null/unparseable/timezone-aware (propagated from
            :func:`_league_maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_league_maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    maps = _league_maps_as_of(date, matches_df, maps_df)

    normalized = config.normalize_map_name(map_name)
    map_rows = maps[maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized]

    abs_margin, _, _ = _margin_and_ot(
        map_rows[TEAM1_SCORE_COL], map_rows[TEAM2_SCORE_COL]
    )
    n = len(map_rows)
    variance = float(np.var(abs_margin, ddof=1)) if n > 1 else float("nan")
    return MapMarginVariance(n=n, variance=variance)


def map_ot_rate(
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=DEFAULT_MAP_OT_K,
) -> ShrunkOTRate:
    """Return the heavily-shrunk overtime rate on one map as of a cutoff.

    The per-map analogue of :func:`team_ot_rate`, added for M27's
    conditional-logit ban model: it counts ``events``/``games`` over the
    *league-wide* as-of pool (:func:`_league_maps_as_of`) restricted to
    the named ``map_name`` (normalized via
    :func:`utils.config.normalize_map_name`, matching
    :func:`map_round_margin_variance`'s filter pattern) and shrinks
    toward the *same* global pooled OT prior (:func:`global_ot_rate` at
    the same date) with the same Beta-posterior formula
    ``mean = (events + k*prior) / (games + k)``. With ``games == 0`` on
    the map the formula degrades to ``mean = prior`` exactly (full
    shrinkage). ``k`` defaults to the heavy :data:`DEFAULT_MAP_OT_K`
    (equal to :data:`DEFAULT_OT_K`; a single map's sample is even
    narrower than one team's, so the prior's weight is if anything
    larger).

    Args:
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching, so
            ``"breeze"``/``" Breeze "`` both match ``"Breeze"``.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the columns
            the league-wide filter already requires).
        k: The shrinkage strength (effective prior sample size); must be
            a positive finite real number (see :func:`_validate_k`).

    Returns:
        A :class:`ShrunkOTRate` with the map's ``events``/``games``
        (counted over all teams' as-of maps on that map), the global
        ``prior``, the unshrunk ``raw_rate`` (equal to ``prior`` when
        ``games == 0``), and the posterior ``alpha``, ``beta``, ``mean``
        and ``variance``.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_league_maps_as_of`).
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`_validate_k`); if an as-of map has a null/NaN score
            (see :func:`_margin_and_ot`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from
            :func:`_league_maps_as_of` / :func:`global_ot_rate`).
        TypeError: If the query date is list-like (propagated from
            :func:`_league_maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)
    maps = _league_maps_as_of(date, matches_df, maps_df)

    normalized = config.normalize_map_name(map_name)
    map_rows = maps[maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized]

    _, _, is_ot = _margin_and_ot(
        map_rows[TEAM1_SCORE_COL], map_rows[TEAM2_SCORE_COL]
    )
    events = int(is_ot.sum())
    games = len(map_rows)
    prior = global_ot_rate(date, matches_df, maps_df).rate
    raw_rate = events / games if games else prior

    alpha = events + k_value * prior
    beta = (games - events) + k_value * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))

    return ShrunkOTRate(
        events=events,
        games=games,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )

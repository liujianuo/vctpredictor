"""Side-specific (attack/defence phase) shrunk round win rates (roadmap M38.2).

A pure in-memory library estimator for a team's regulation round win rate
on one ``phase`` (``"attack"`` or ``"defense"``) of one map, partial-pooled
exactly as M13 partial-pools map *win* rates but at *round* resolution:
``mean = (rounds_won + k*prior) / (rounds_played + k)``, a Beta posterior
mean whose likelihood is the map-side phase's regulation rounds. The full
posterior (``alpha``/``beta``/``mean``/``variance``) is exposed at both
levels of the shrinkage hierarchy, not just the point estimate. Like the
rest of ``features/`` it has no CLI, no ``argparse`` entry point and no
file I/O of its own — it operates on the already-materialised
``matches_df``/``maps_df`` DataFrames a caller passes in (the round-detail
columns the estimator reads are the eight ``team*_{atk,def}_rounds``
columns of M8's maps table, the substrate ``features.round_detail``
derives from).

**Seat vs phase — the naming collision this module resolves.** The roadmap
and M13 call this milestone's quantity "side-specific", and
``features/round_detail.py`` (the substrate this module trains against)
emits records keyed by a ``side`` column — but that column is a **seat**
marker (the literal string ``"team1"``/``"team2"``, i.e. which side of the
match a team sat on), and carries no attack/defence meaning whatsoever.
This milestone's "side" is a **phase** (``"attack"``/``"defense"``, which
half a team was playing). To make it impossible to write ``side`` and mean
two different things in one file, this module never uses the bare word
``side`` as a parameter/variable name for its own attack/defence concept:
its public API is ``phase``-parameterised end to end
(``team_map_side_rate(team_id, map_name, phase, date, ...)`` — the
"side" in the *function* names is the roadmap-vocabulary phase, dictated
by the milestone's own M38.2 title) and the two allowed values are the
named constants :data:`PHASE_ATTACK`/:data:`PHASE_DEFENSE`. When this
module reads ``round_detail``'s output it refers to that ``side`` column
explicitly as the record's **seat** marker in comments and docstrings, so
a reader is never left guessing which concept a bare ``side`` reference
means.

**Two-level shrinkage hierarchy — one CV'd ``k`` plus one fixed constant
(the largest interpretive call in this milestone).** The roadmap's formula
is given for only one level: ``(rounds_won + k*prior)/(rounds_played + k)``
shrinking the team-map-side rate toward the team's *overall same-side*
rate, with singular "choose k by walk-forward CV" (matching M13's own
single-CV'd-hyperparameter phrasing). Read literally, and resolved here
exactly as ``closeness.py`` already resolved its analogous situation
(a heavy, fixed, *not* CV'd ``DEFAULT_OT_K`` shrinking toward
``global_ot_rate``, justified by scale rather than tuned):

- **Outer level** (:func:`team_map_side_rate`, the headline estimator):
  Beta-posterior mean ``(rounds_won + k*prior)/(rounds_played + k)`` whose
  ``prior`` is the team's *shrunk* overall same-phase rate — level 2's
  output, not a raw rate, which is what makes this a genuine two-level
  hierarchy rather than two independent single-level shrinkages — and
  whose ``k`` is chosen by :func:`select_k` via walk-forward CV,
  independently of M13's ``k`` (this module never imports
  ``map_win_rate.select_k``; the roadmap says the round-level ``k`` is fit
  separately, so it needs its own grid and its own CV run).
- **Inner level** (:func:`team_overall_side_rate`): Beta-posterior mean
  ``(rounds_won + prior_k*league_rate)/(rounds_played + prior_k)`` pulling
  the team's overall same-phase rate toward the unshrunk league-wide
  pooled phase rate (:func:`league_side_rate`), where ``prior_k`` is the
  fixed, documented constant :data:`DEFAULT_PRIOR_K` — *not* CV'd, exactly
  as the roadmap gives a formula for only the outer level and
  ``closeness.py``'s fixed-constant precedent covers the inner one.

  Assumption flagged per plan item 4: if the roadmap were instead read as
  "both levels share one CV'd ``k``" (or "the inner level is a second CV
  grid dimension"), the two-level design below would differ; this BUILD
  records the one-CV'd-k-plus-one-fixed-constant reading as its explicit
  interpretation rather than silently picking it.

**Round-level, not map-level, CV scoring (the one place M13's pattern
cannot be copied verbatim).** M13's ``select_k`` scores a binary per-map
outcome (did the team win the map) with binary log loss, because its
``mean`` predicts P(map win). This module's ``team_map_side_rate.mean``
predicts a *round*-level win rate, so the natural validation target is the
held-out map-side's own ``(rounds_won, rounds_played)`` for the phase,
scored with binomial log loss
``-(rounds_won*log(p) + (rounds_played - rounds_won)*log(1 - p))`` with
``p`` clipped into ``[eps, 1-eps]`` exactly as ``map_win_rate`` clips. The
aggregate score is the **rounds-weighted** mean: ``sum(instance log
loss) / sum(rounds_played over all instances)`` — weighting by rounds
rather than by map-side instances, since the whole point of this feature
is resolution at the round level. Assumption flagged per plan item 7: the
roadmap does not specify the weighting; the equal-instance-weighting
alternative is defensible but under-weights exactly the high-round-count
instances that make the CV signal reliable.

**``select_k`` is ``phase``-parameterised and run twice.** VALORANT maps
are attacker/defender asymmetric by the roadmap's own framing, and the two
phases enter ``build_feature_vector`` (eventually, M38.5) as *separate*
opponent differentials — so :func:`select_k` takes a ``phase`` argument
and the real-data smoke test calls it once per phase, reporting two
curves and two ``best_k`` values. Assumption flagged per plan item 5: the
roadmap's singular "report the curve" matches M13's own docstring for a
single ``select_k`` call, so singular phrasing is not strong evidence for
sharing one ``k`` across phases.

**Leakage contract (the hard requirement, reused from M12, not
reimplemented).** Every estimator obtains history exclusively through
``utils.asof`` — :func:`utils.asof.maps_as_of` for the per-team levels and
this module's own league-wide pool :func:`_league_maps_as_of` (the
``closeness.py``/``elo.py`` precedent) for the league rate — never by
reading ``matches.parquet``/``maps.parquet`` directly. The strict ``<``
boundary, null-date rejection and timezone-naive-only rules are enforced
by ``utils.asof``'s public parse helpers (``parse_query_date`` /
``parse_date_column`` / ``require_columns``), so a map dated equal to or
after the query date never enters any estimate, and ``select_k`` scores
every held-out map against a snapshot taken at that map's own match
timestamp.

**Column/phase terminology of the derived substrate.** The round counts
this module sums come from ``features.round_detail``'s derived records:
per ``(match_id, map_index, seat)`` the record carries
``atk_rounds_won``/``atk_rounds_played`` and ``def_rounds_won``/
``def_rounds_played``, which are **regulation-only** (OT rounds are kept
separate by M38.1 and never folded into a side), and the played counts are
the opposing-side pairings (a seat's ``atk_rounds_played`` counts every
round of the half it attacked, won by it attacking or by the opponent
defending). Every rate in this module is therefore a regulation-side rate;
the OT rounds of an OT map carry no per-side attribution and never enter
these denominators.

**Data-shape findings (re-derived against real ``data/v1``, plan item 1 —
not copied from the plan; ``derive_map_round_details`` on the full
244-row ``maps.parquet``, no as-of filter):**

- 484 team-map records survive derivation (242 non-null maps x 2 seats;
  the 2 null-round-column maps of match 712803 are excluded as before).
- Pooled over every record with no as-of filter, attack rounds are
  2583 won of 5075 played (rate ~0.509) and defence rounds 2492 won of
  5075 played (rate ~0.491) — nearly a coin flip at whole-dataset scale,
  confirming the ``0.5`` uninformative default for the league-wide prior
  (the ``OverallWinRate`` convention, not ``GlobalOTRate``'s ``0.0``: a
  side round win is a genuine ~50/50 quantity, not a rare event).
- Per-team-map ``atk_rounds_played``/``def_rounds_played`` range 1-12
  across the 484 records (mean ~10.5, median 12): a single ``(team, map)``
  phase sample is small, confirming the map-level estimate genuinely needs
  shrinkage.
- Per-team overall same-phase totals (records joined to
  ``matches.parquet`` to resolve ``team_id``; 16 distinct teams):
  ``atk_rounds_played`` 121-496 (mean ~317) and ``def_rounds_played``
  122-533 (mean ~317) — two orders of magnitude above a single map's
  ~1-12, the concrete reason the two shrinkage levels carry genuinely
  different amounts of evidence (and the scale basis for
  :data:`DEFAULT_PRIOR_K` below). The 16 unshrunk team attack rates span
  0.408-0.592 and defence rates 0.415-0.585: real between-team dispersion
  exists, so the inner level must not over-shrink it away.

**Module note on the league-wide pool.** ``utils.asof`` is single-team by
design, so this module writes its own small league-wide filter
(:func:`_league_maps_as_of`) mirroring ``matches_as_of``'s completed +
strictly-before masks *without* the team-id mask (exactly the
``closeness.py``/``elo.py`` pattern), then inner-joins to ``maps_df`` with
the finished-map filter, selecting the full
``round_detail.REQUIRED_COLUMNS`` set plus ``date`` so the result feeds
straight into ``features.round_detail.derive_map_round_details``. Date
parsing reuses ``utils.asof``'s public parse helpers — a documented,
already-precedented ``features`` -> ``utils`` dependency (feature modules
may depend downward on genuine ``utils/`` utilities), not a fresh
reimplementation. The two downward imports from ``utils.splits`` in
:func:`select_k` (``split_matches`` + ``walk_forward_folds``) follow the
identical precedent set by ``map_win_rate.select_k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from features import round_detail as rd
from features._shared import _validate_k
from utils import asof, config
from utils.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_FRAC,
    MIN_FOLD_BLOCK_MATCHES,
    split_matches,
    walk_forward_folds,
)

# The two phases this module estimates. The roadmap's "side" vocabulary
# (and the derived-records ``side`` column of features.round_detail) is a
# *seat* marker with no attack/defence meaning; this module's own
# attack/defence concept is spelled ``phase`` everywhere and restricted to
# these two values (see the module docstring's collision paragraph).
PHASE_ATTACK = "attack"
PHASE_DEFENSE = "defense"
_VALID_PHASES = (PHASE_ATTACK, PHASE_DEFENSE)

# The maps-table map-name column this module filters on (the map filter of
# the map-level estimator); ``team_is_team1``/``date`` come from
# utils.asof's maps output, and the round-detail columns from M8's maps
# table via features.round_detail's constants.
MAP_NAME_COL = "map_name"

# Fixed inner-level shrinkage strength: the effective prior sample size
# (in *rounds*) that :func:`team_overall_side_rate` gives the league-wide
# pooled phase rate when shrinking a team's overall same-phase rate toward
# it. It is deliberately NOT cross-validated — the roadmap gives a formula
# for only the outer (map-level) ``k``, and this constant mirrors
# ``closeness.py``'s already-shipped ``DEFAULT_OT_K`` precedent of a fixed,
# documented constant justified by scale rather than tuned. The scale
# argument (module docstring's Data-shape findings): a team's overall
# same-phase sample is 121-533 rounds in v1 (mean ~317) — two orders of
# magnitude above a single map-phase's ~1-12 — while the league prior it
# shrinks toward is itself pooled over ~5000 rounds and essentially
# precisely known; the genuine unknown is the *between-team* dispersion
# (the 16 v1 team attack rates span 0.408-0.592, defence 0.415-0.585),
# which over-shrinking would compress away. ``prior_k = 50`` is ~16% of
# the mean per-team-phase sample: at the v1 minimum sample (121 rounds)
# the prior still holds less than 30% of the posterior weight, at the mean
# ~14% — the team's own much-larger same-phase sample dominates. This is
# an order of magnitude *lighter* than ``closeness.DEFAULT_OT_K = 1000.0``
# (that constant is heavy because OT is a genuinely rare event even
# pooled; a side round win is not). A judgment call, isolated as a named
# constant and overridable per call, not a magic number.
DEFAULT_PRIOR_K = 50.0

# Documented fallback shrinkage strength for ad-hoc callers of the
# outer-level estimator (:func:`team_map_side_rate`). It is NOT what
# cross-validation reports: the chosen value is :func:`select_k`'s
# ``best_k``, and this constant only gives hand-written calls a sane
# default when no CV has been run (mirroring ``map_win_rate.DEFAULT_K``).
DEFAULT_K = 10.0

# The default candidate grid :func:`select_k` searches over when the
# caller does not pass one. Starts from M13's own grid
# ``(1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)`` as the plan-mandated
# baseline, then extended at BUILD time because the real-v1 CV argmin
# landed at the top edge of that grid for both phases (the round-count
# denominators here are ~10-12 per map-phase, smaller than M13's per-map
# denominators, so the effective range needed extending — the same
# "explore before freezing a specific default" caution
# ``map_win_rate.DEFAULT_K_GRID`` already states). On real ``data/v1``
# the attack curve's optimum sits in the heavy-k asymptote (scores flat
# from ``k = 1000`` upward: 0.693237 at 1000 vs 0.693221 at 10000) and
# the defence curve has a genuine interior optimum at ``k = 500``
# (turning back up by ``k = 1000+``); values beyond 1000 carry no
# additional signal (per-round score differences below 2e-5), so the
# grid stops there. A pragmatic geometric grid; no principled default is
# specified by roadmap M38.2, so this is a tunable constant, not a magic
# number buried in the CV loop.
DEFAULT_K_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

# Clip epsilon for the per-round probability handed to binomial log loss
# inside :func:`select_k`, mirroring ``map_win_rate._PROB_CLIP_EPS``: a
# posterior mean can reach exactly 0.0/1.0 (a 100%-win sample with a 1.0
# prior), where log loss is -inf/raises; clipping into ``[eps, 1-eps]``
# keeps the score finite and consistent with ``utils.scoring.log_loss``'s
# zero-probability-raises convention.
_PROB_CLIP_EPS = 1e-12

# The columns :func:`_league_maps_as_of` needs on each table. Unlike
# ``utils.asof``'s single-team functions, team-id columns are *not*
# required: the league pool is over all teams and never needs to orient a
# score to a particular side. The maps side must carry the full
# ``round_detail`` required set (so the pool can be fed straight into
# ``derive_map_round_details``) plus ``winner`` (the finished-map signal
# ``maps_as_of`` applies).
_MATCHES_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.DATE_COL,
    asof.STATUS_COL,
)
_MAPS_REQUIRED = tuple(rd.REQUIRED_COLUMNS) + (asof.WINNER_COL,)


@dataclass(frozen=True)
class LeagueSideRate:
    """The league-wide pooled regulation round win rate for one phase.

    ``rate`` is ``rounds_won / rounds_played`` when ``rounds_played > 0``
    and exactly ``0.5`` (the maximally uninformative value) when
    ``rounds_played == 0`` — the empty case is reachable only at/near the
    dawn of the dataset, and ``0.5`` is the least-committal stand-in for a
    genuine ~50/50 quantity (the ``OverallWinRate`` convention, not a rare
    event's ``0.0``). ``rounds_won``/``rounds_played`` are regulation
    round counts summed over *every* derived seat record of every finished
    map in the as-of pool (a league pool does not care which seat a team
    sat in).
    """

    rounds_won: int
    rounds_played: int
    rate: float


@dataclass(frozen=True)
class ShrunkOverallSideRate:
    """The inner-level Beta posterior for a team's overall same-phase rate.

    The level-2 posterior of the two-level hierarchy: a team's regulation
    round win rate on one phase pooled over *all* its as-of maps, shrunk
    toward the league-wide pooled phase rate (:func:`league_side_rate`)
    with the fixed strength :data:`DEFAULT_PRIOR_K`. ``alpha``/``beta``
    are the full posterior parameters
    ``Beta(rounds_won + prior_k*prior, (rounds_played - rounds_won) +
    prior_k*(1 - prior))`` — exposed, not just the point estimate, so
    callers can read off the uncertainty. ``mean`` is the shrinkage point
    estimate ``alpha / (alpha + beta)``; ``variance`` is the Beta variance.
    ``rounds_won``/``rounds_played`` are the team's same-phase totals over
    its as-of history; ``prior`` is the league-wide pooled phase rate fed
    in as the prior mean; ``raw_rate`` is the unshrunk team rate
    ``rounds_won / rounds_played``, or exactly ``prior`` when
    ``rounds_played == 0`` (full shrinkage — no raw sample to compare
    against).
    """

    rounds_won: int
    rounds_played: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


@dataclass(frozen=True)
class ShrunkSideWinRate:
    """The outer-level (final, headline) Beta posterior for one ``(team, map, phase)``.

    The level-1 posterior of the two-level hierarchy: a team's regulation
    round win rate on one phase of one named map, shrunk toward the team's
    *shrunk overall same-phase rate* (the inner level's output) with the
    CV-chosen strength ``k``. ``alpha``/``beta`` are the full posterior
    parameters ``Beta(rounds_won + k*prior, (rounds_played - rounds_won) +
    k*(1 - prior))`` — exposed, not just the point estimate. ``mean`` is
    the shrinkage point estimate ``alpha / (alpha + beta)`` (the roadmap's
    ``(rounds_won + k*prior) / (rounds_played + k)``); ``variance`` is the
    Beta variance. ``rounds_won``/``rounds_played`` are the team's
    same-phase totals *on this map* (not overall); ``prior`` here is
    ``ShrunkOverallSideRate.mean`` — the *shrunk* inner estimate, not the
    raw team-overall rate — which is exactly what makes the hierarchy
    two-level rather than two independent single-level shrinkages;
    ``raw_rate`` is the unshrunk map-phase rate ``rounds_won /
    rounds_played``, or exactly ``prior`` when ``rounds_played == 0``
    (full shrinkage).
    """

    rounds_won: int
    rounds_played: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


def _validate_phase(phase: str) -> str:
    """Validate a phase argument and return the canonical value.

    The single choke point behind every ``phase``-taking function in this
    module: the only two valid values are :data:`PHASE_ATTACK` and
    :data:`PHASE_DEFENSE`, and anything else is rejected before any
    computation can run.

    Args:
        phase: The phase to validate; must be exactly ``"attack"`` or
            ``"defense"`` (a non-string, ``None`` or any other value is
            rejected).

    Returns:
        The validated ``phase`` unchanged (canonical spelling).

    Raises:
        ValueError: If ``phase`` is not one of :data:`PHASE_ATTACK` /
            :data:`PHASE_DEFENSE`; the message lists the allowed values.
    """
    if phase not in _VALID_PHASES:
        raise ValueError(
            f"invalid phase {phase!r}; phase must be one of "
            f"{list(_VALID_PHASES)}"
        )
    return phase


def _phase_columns(phase: str) -> tuple[str, str]:
    """Map a phase to its ``(won_col, played_col)`` record column names.

    The derived records of ``features.round_detail`` carry four round
    columns per seat (``atk_rounds_won``/``atk_rounds_played`` and
    ``def_rounds_won``/``def_rounds_played``); this helper picks the pair
    belonging to the queried phase so every summing site in the module
    shares one spelling. It validates ``phase`` first (see
    :func:`_validate_phase`), so every caller is covered by the
    invalid-phase guard through one choke point.

    Args:
        phase: The phase to map (validated).

    Returns:
        A ``(won_col, played_col)`` tuple of record column names:
        ``("atk_rounds_won", "atk_rounds_played")`` for
        :data:`PHASE_ATTACK` and ``("def_rounds_won",
        "def_rounds_played")`` for :data:`PHASE_DEFENSE`.

    Raises:
        ValueError: If ``phase`` is invalid (propagated from
            :func:`_validate_phase`).
    """
    _validate_phase(phase)
    if phase == PHASE_ATTACK:
        return "atk_rounds_won", "atk_rounds_played"
    return "def_rounds_won", "def_rounds_played"


def _beta_posterior(rounds_won: int, rounds_played: int, prior: float, k: float):
    """Compute the Beta posterior parameters from counts, prior and strength.

    The single shared arithmetic behind every shrinkage site in the
    hierarchy (inner level, outer level, and :func:`select_k`'s scoring
    sweep): given a ``rounds_won``-of-``rounds_played`` sample, a prior
    mean ``prior`` and a shrinkage strength ``k`` (effective prior sample
    size), return ``alpha = rounds_won + k*prior``, ``beta =
    (rounds_played - rounds_won) + k*(1 - prior)``, the posterior mean
    ``alpha / (alpha + beta)`` and the Beta variance
    ``alpha*beta / ((alpha + beta)^2 * (alpha + beta + 1))``. Because
    ``select_k`` computes each held-out estimate through this helper with
    k-independent inputs, its sweep reproduces ``team_map_side_rate``'s
    mean for every candidate ``k`` without re-running the as-of queries.

    Args:
        rounds_won: The sample's round wins (``int``-coercible).
        rounds_played: The sample's rounds played.
        prior: The prior mean fed in (``ShrunkOverallSideRate.mean`` at
            the outer level, ``LeagueSideRate.rate`` at the inner level).
        k: The shrinkage strength (positive finite real; validated by the
            caller).

    Returns:
        An ``(alpha, beta, mean, variance)`` tuple of floats.

    Raises:
        Nothing: inputs are assumed pre-validated (``k > 0``, finite
            ``prior``, ``0 <= rounds_won <= rounds_played``); passing
            unvalidated values would silently produce a degenerate
            posterior.
    """
    alpha = rounds_won + k * prior
    beta = (rounds_played - rounds_won) + k * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return alpha, beta, mean, variance


def _league_maps_as_of(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return every finished map of every team strictly before a cutoff.

    The league-wide as-of filter the pooled phase rate needs. It mirrors
    ``utils.asof.matches_as_of``'s two boolean masks (completed,
    strictly-before) *without* the per-team mask — the league pool is a
    league-wide property, not team-specific — then inner-joins to
    ``maps_df`` with the same finished-map (``winner.notna()``) filter
    ``utils.asof.maps_as_of`` applies, selecting the full
    ``round_detail.REQUIRED_COLUMNS`` set plus ``date`` so the result can
    be fed straight into ``features.round_detail.derive_map_round_details``
    (the module's own copy of the ``closeness.py``/``elo.py`` pattern;
    those modules' league helpers keep too few columns for round detail,
    and this module must not import from either).

    Date parsing reuses ``utils.asof``'s public parse helpers
    (``parse_query_date`` / ``parse_date_column`` /
    ``require_columns``) rather than duplicating them, so the strict-``<``
    boundary, null-date rejection and timezone-naive-only rules are
    byte-for-byte identical to every other as-of consumer.

    Args:
        date: The as-of cutoff; rows dated equal to or after this are
            excluded (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``status``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner`` and the full ``round_detail`` required column set:
            ``map_index``, the two score columns and the eight round-detail
            columns).

    Returns:
        A ``pandas.DataFrame`` with columns ``date`` followed by
        ``round_detail.REQUIRED_COLUMNS`` (``match_id``, ``map_index``,
        ``team1_score``, ``team2_score`` and the eight round-detail
        columns) — exactly the rows whose parent match is completed and
        strictly before the cutoff *and* whose own ``winner`` is non-null
        (a finished map). The output is unsorted and preserves no input
        index (the merge resets it).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from ``utils.asof.require_columns``).
        ValueError: If the filtered ``matches`` frame contains duplicate
            ``match_id`` values (the join would fan out and duplicate map
            rows); or if the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            ``utils.asof.parse_query_date`` /
            ``utils.asof.parse_date_column``).
        TypeError: If the query date is list-like (propagated from
            ``utils.asof.parse_query_date``).
    """
    asof.require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    asof.require_columns(maps_df, _MAPS_REQUIRED, "maps_df")

    parsed_dates = asof.parse_date_column(matches_df[asof.DATE_COL])
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
        list(rd.REQUIRED_COLUMNS)
    ]
    join_frame = matches[[asof.MATCH_ID_COL, asof.DATE_COL]]
    return finished_maps.merge(join_frame, on=asof.MATCH_ID_COL, how="inner")


def _team_phase_rounds(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    phase: str,
    map_name: str | None = None,
) -> tuple[int, int]:
    """Sum a team's as-of same-phase ``(rounds_won, rounds_played)``.

    The shared resolution helper behind both the team-overall and the
    team-map estimators (plan decision 3, steps 1-6 in one place). It
    fetches the team's completed, strictly-earlier maps through
    :func:`utils.asof.maps_as_of` (never by reading the Parquet tables
    directly), runs :func:`features.round_detail.derive_map_round_details`
    on that as-of frame (which validates and derives both seats' records
    for every surviving row, excluding null-round-column rows exactly as
    it does on the raw table), then resolves each surviving record to the
    queried team: a record's ``(match_id, map_index)`` key is looked up in
    the as-of frame's ``team_is_team1`` orientation column, and only the
    record whose seat (``round_detail``'s ``side`` marker — a *seat*, see
    the module docstring) equals ``"team1"`` when the team was team1, else
    ``"team2"``, is kept. When ``map_name`` is given the as-of frame is
    first filtered to that map (both sides normalized through
    :func:`utils.config.normalize_map_name`, matching
    ``map_win_rate.team_map_win_rate``'s established convention), and the
    phase's ``won``/``played`` record columns are summed over the kept
    rows.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs the full
            ``round_detail`` required column set plus ``map_name`` when
            ``map_name`` is not ``None``, in addition to the columns
            ``maps_as_of`` already requires).
        phase: The phase to sum (validated; see :func:`_validate_phase`).
        map_name: When not ``None``, restrict the sum to the queried team's
            as-of maps on this map (normalized via
            :func:`utils.config.normalize_map_name`).

    Returns:
        A ``(rounds_won, rounds_played)`` tuple of ``int``, summed over
        the queried team's own seat records of the surviving as-of maps
        (optionally map-restricted). An empty as-of history (or a
        map-restriction matching nothing) yields ``(0, 0)`` — a normal,
        non-error outcome.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of` /
            ``features.round_detail``'s required-column check; includes
            ``map_name`` when a map filter is requested).
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if an as-of map fails ``round_detail``'s validation (a null
            score on a surviving row or a case-split/pairing violation —
            propagated from
            :func:`features.round_detail.derive_map_round_details`); or
            if the query date or a row date is null/unparseable/
            timezone-aware (propagated from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    _validate_phase(phase)
    won_col, played_col = _phase_columns(phase)

    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)
    if map_name is not None:
        asof.require_columns(maps_df, (MAP_NAME_COL,), "maps_df")
        normalized = config.normalize_map_name(map_name)
        maps = maps[
            maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized
        ]

    derived = rd.derive_map_round_details(maps)
    records = derived.records
    if records.empty:
        return 0, 0

    # Resolve each derived record to the queried team via the as-of
    # frame's per-row orientation: keep exactly the record whose seat
    # marker matches the seat the team occupied on that map.
    seat_lookup = maps[
        [rd.MATCH_ID_COL, rd.MAP_INDEX_COL, asof.TEAM_ORIENTATION_COL]
    ]
    joined = records.merge(
        seat_lookup, on=[rd.MATCH_ID_COL, rd.MAP_INDEX_COL], how="inner"
    )
    seat_is_team1 = joined["side"] == rd.TEAM1_SIDE
    keep = seat_is_team1 == joined[asof.TEAM_ORIENTATION_COL].astype(bool)
    own = joined[keep]

    rounds_won = int(own[won_col].sum())
    rounds_played = int(own[played_col].sum())
    return rounds_won, rounds_played


def league_side_rate(
    phase: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> LeagueSideRate:
    """Return the league-wide pooled regulation round win rate for a phase.

    The inner level's prior source and the module's pooled base rate. It
    derives round detail over the league-wide as-of pool
    (:func:`_league_maps_as_of` — every finished map of every team
    strictly before the cutoff) and sums the phase's ``won``/``played``
    record columns over *every* surviving record, both seats (a league
    pool does not care which seat a team sat in). Rates are regulation
    round win rates (``round_detail``'s atk/def columns are
    regulation-only).

    Args:
        phase: The phase to pool (``"attack"`` or ``"defense"``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs the full
            ``round_detail`` required column set).

    Returns:
        A :class:`LeagueSideRate` with ``rounds_won``/``rounds_played``
        summed over every derived seat record of the as-of pool, and
        ``rate = rounds_won / rounds_played`` — or exactly ``0.5`` when
        ``rounds_played == 0`` (an empty pool, reachable only at/near the
        dawn of the dataset).

    Raises:
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if a pool map fails ``round_detail``'s validation (see
            :func:`_league_maps_as_of` / 
            :func:`features.round_detail.derive_map_round_details`); or if
            the query date or a row date is null/unparseable/
            timezone-aware (propagated from :func:`_league_maps_as_of`).
        KeyError: If either table lacks a required column (propagated from
            :func:`_league_maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_league_maps_as_of`).
    """
    _validate_phase(phase)
    won_col, played_col = _phase_columns(phase)

    pool = _league_maps_as_of(date, matches_df, maps_df)
    derived = rd.derive_map_round_details(pool)
    rounds_won = int(derived.records[won_col].sum())
    rounds_played = int(derived.records[played_col].sum())
    rate = rounds_won / rounds_played if rounds_played else 0.5
    return LeagueSideRate(rounds_won=rounds_won, rounds_played=rounds_played, rate=rate)


def team_overall_side_rate(
    team_id: str,
    phase: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    prior_k=DEFAULT_PRIOR_K,
) -> ShrunkOverallSideRate:
    """Return a team's shrunk overall same-phase round win rate as of a cutoff.

    The inner level of the shrinkage hierarchy. It sums the team's
    same-phase ``(rounds_won, rounds_played)`` over *all* its as-of maps
    via :func:`_team_phase_rounds` (no map filter) and takes the
    league-wide pooled phase rate (:func:`league_side_rate` at the same
    date) as its prior, then applies the Beta posterior
    ``mean = (rounds_won + prior_k*prior) / (rounds_played + prior_k)``
    with the fixed strength ``prior_k`` (default
    :data:`DEFAULT_PRIOR_K`; see its comment for the scale reasoning).
    With ``rounds_played == 0`` the formula degrades to ``mean = prior``
    exactly (full shrinkage — the correct behaviour, not a special case).

    Args:
        team_id: The queried team's stable id.
        phase: The phase to estimate (``"attack"`` or ``"defense"``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs the full
            ``round_detail`` required column set in addition to the
            columns ``maps_as_of`` already requires).
        prior_k: The fixed inner-level shrinkage strength (effective prior
            sample size in rounds); must be a positive finite real number
            (see :func:`_validate_k`).

    Returns:
        A :class:`ShrunkOverallSideRate` with the team's same-phase
        ``rounds_won``/``rounds_played`` over all its as-of maps, the
        league ``prior``, the unshrunk ``raw_rate`` (equal to ``prior``
        when ``rounds_played == 0``), and the posterior ``alpha``,
        ``beta``, ``mean`` and ``variance``.

    Raises:
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if ``prior_k`` is not a positive finite real number (see
            :func:`_validate_k`); if an as-of map fails
            ``round_detail``'s validation (propagated from
            :func:`_team_phase_rounds`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from the
            as-of helpers).
        KeyError: If either table lacks a required column (propagated from
            the as-of helpers).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
    """
    _validate_phase(phase)
    prior_k_value = _validate_k(prior_k)

    rounds_won, rounds_played = _team_phase_rounds(
        team_id, date, matches_df, maps_df, phase
    )
    prior = league_side_rate(phase, date, matches_df, maps_df).rate
    raw_rate = rounds_won / rounds_played if rounds_played else prior

    alpha, beta, mean, variance = _beta_posterior(
        rounds_won, rounds_played, prior, prior_k_value
    )
    return ShrunkOverallSideRate(
        rounds_won=rounds_won,
        rounds_played=rounds_played,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def team_map_side_rate(
    team_id: str,
    map_name: str,
    phase: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k,
) -> ShrunkSideWinRate:
    """Return the outer-level shrunk round win rate for one ``(team, map, phase)``.

    The headline estimator of the milestone (the roadmap's
    ``(rounds_won + k*prior) / (rounds_played + k)`` at round resolution).
    It sums the team's same-phase ``(rounds_won, rounds_played)`` on the
    queried ``map_name`` via :func:`_team_phase_rounds` (both sides
    normalized through :func:`utils.config.normalize_map_name`, so
    case/whitespace never break a match) and takes the inner level's
    *shrunk* overall same-phase rate
    (:func:`team_overall_side_rate`, whose ``mean`` is the posterior of
    the team-overall-toward-league shrinkage — not the raw team-overall
    rate, which is what makes this a genuine two-level hierarchy) as its
    prior, then applies the Beta posterior with the CV-chosen strength
    ``k``. With ``rounds_played == 0`` on the map the formula degrades to
    ``mean = prior`` exactly (full shrinkage — the correct behaviour, not
    a special case). No map-pool/era filtering happens here: a map name
    outside the caller's active pool is still a legitimate historical map
    to count (pool filtering is a caller concern, e.g. M38.5).

    Args:
        team_id: The queried team's stable id.
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching, so
            ``"breeze"``/``" Breeze "`` both match ``"Breeze"``.
        phase: The phase to estimate (``"attack"`` or ``"defense"``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name`` and
            the full ``round_detail`` required column set in addition to
            the columns ``maps_as_of`` already requires).
        k: The outer-level shrinkage strength (effective prior sample size
            in rounds); must be a positive finite real number (see
            :func:`_validate_k`).

    Returns:
        A :class:`ShrunkSideWinRate` with the map-phase
        ``rounds_won``/``rounds_played``, the *shrunk* overall-phase
        ``prior``, the unshrunk ``raw_rate`` (equal to ``prior`` when
        ``rounds_played == 0``), and the posterior ``alpha``, ``beta``,
        ``mean`` and ``variance``.

    Raises:
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if ``k`` is not a positive finite real number (see
            :func:`_validate_k`); if an as-of map fails
            ``round_detail``'s validation (propagated from
            :func:`_team_phase_rounds`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from the
            as-of helpers).
        KeyError: If either table lacks a required column (propagated from
            the as-of helpers; includes ``map_name`` and the
            ``round_detail`` columns).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    _validate_phase(phase)
    k_value = _validate_k(k)

    map_won, map_played = _team_phase_rounds(
        team_id, date, matches_df, maps_df, phase, map_name=map_name
    )
    overall = team_overall_side_rate(
        team_id, phase, date, matches_df, maps_df
    )
    prior = overall.mean
    raw_rate = map_won / map_played if map_played else prior

    alpha, beta, mean, variance = _beta_posterior(
        map_won, map_played, prior, k_value
    )
    return ShrunkSideWinRate(
        rounds_won=map_won,
        rounds_played=map_played,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def _collect_validation_instances(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    phase: str,
    folds: list[tuple[int, list, list]],
) -> list[tuple[str, str, str, int, int]]:
    """Build the scored ``(team_id, map_name, date, rounds_won, rounds_played)`` instances.

    Turns the walk-forward fold assignment into the flat list of held-out
    map-phase round outcomes that :func:`select_k` scores, mirroring
    ``map_win_rate._collect_validation_instances``'s shape but returning
    round counts instead of a binary ``won``. Every finished validation
    map yields *two* instances — one per side, resolved to ``team_id`` via
    the match row (each side is an independent as-of query and a genuine
    test of the shrinkage estimate, the same "both sides are independent
    instances" convention as M13's assumption 8). Each instance's round
    counts are the side's derived regulation rounds for the queried
    ``phase`` (``features.round_detail``'s atk/def records). A validation
    map that ``round_detail`` would exclude (null round columns) has no
    derived round detail to validate against and contributes no instance
    for either side — it is skipped, not raised. A seat record whose
    ``rounds_played == 0`` for the queried phase (theoretically possible
    per ``round_detail``'s ``[0, 12]`` bound, never observed in v1) also
    contributes no instance for that side (a binomial score over zero
    trials is undefined weight, not a raise).

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``); its
            ``match_id`` values must be unique.
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_name``, ``winner`` and the full ``round_detail``
            required column set). Only finished maps (``winner`` non-null)
            contribute instances.
        phase: The phase whose round columns define each instance's
            ``rounds_won``/``rounds_played`` (validated).
        folds: The ``(fold_id, train_ids, val_ids)`` tuples from
            :func:`utils.splits.walk_forward_folds`.

    Returns:
        A list of ``(team_id, map_name, date, rounds_won, rounds_played)``
        tuples in fold order, up to two per finished validation map whose
        round detail survives derivation.

    Raises:
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if a validation ``match_id`` is absent from ``matches_df``;
            or if a finished validation map fails ``round_detail``'s
            validation (a null score on a surviving row or a
            case-split/pairing violation — propagated from
            :func:`features.round_detail.derive_map_round_details`).
        KeyError: If ``maps_df`` lacks a required column (propagated from
            ``features.round_detail``).
    """
    _validate_phase(phase)
    won_col, played_col = _phase_columns(phase)

    if not matches_df[asof.MATCH_ID_COL].is_unique:
        raise ValueError(
            "matches_df contains duplicate match_id values; the "
            "validation-instance lookup would silently collapse them"
        )
    match_by_id = {
        getattr(row, asof.MATCH_ID_COL): row
        for row in matches_df.itertuples(index=False)
    }
    finished = maps_df[maps_df[asof.WINNER_COL].notna()]

    instances: list[tuple[str, str, str, int, int]] = []
    for _fold_id, _train_ids, val_ids in folds:
        for mid in val_ids:
            match = match_by_id.get(mid)
            if match is None:
                raise ValueError(
                    f"validation match_id {mid!r} is absent from matches_df"
                )
            match_maps = finished[finished[asof.MATCH_ID_COL] == mid]
            if match_maps.empty:
                continue
            result = rd.derive_map_round_details(match_maps)
            excluded_keys = {
                (row.match_id, row.map_index) for row in result.excluded
            }
            records = result.records
            for map_row in match_maps.itertuples(index=False):
                map_key = (
                    getattr(map_row, rd.MATCH_ID_COL),
                    getattr(map_row, rd.MAP_INDEX_COL),
                )
                if map_key in excluded_keys:
                    continue
                key_records = records[
                    (records[rd.MATCH_ID_COL] == map_key[0])
                    & (records[rd.MAP_INDEX_COL] == map_key[1])
                ]
                team1_record = key_records[key_records["side"] == rd.TEAM1_SIDE].iloc[0]
                team2_record = key_records[key_records["side"] == rd.TEAM2_SIDE].iloc[0]
                for seat_record, seat_team_id in (
                    (team1_record, getattr(match, asof.TEAM1_ID_COL)),
                    (team2_record, getattr(match, asof.TEAM2_ID_COL)),
                ):
                    rounds_won = int(seat_record[won_col])
                    rounds_played = int(seat_record[played_col])
                    if rounds_played == 0:
                        continue
                    instances.append(
                        (
                            seat_team_id,
                            getattr(map_row, MAP_NAME_COL),
                            getattr(match, asof.DATE_COL),
                            rounds_won,
                            rounds_played,
                        )
                    )
    return instances


def select_k(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    phase: str,
    k_grid=DEFAULT_K_GRID,
    n_folds: int = DEFAULT_N_FOLDS,
    min_fold_block: int = MIN_FOLD_BLOCK_MATCHES,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple:
    """Choose the outer-level shrinkage strength ``k`` for one phase by walk-forward CV.

    The CV harness for the map-level shrinkage strength of one phase
    (attack and defence are asymmetric distributions and enter
    ``build_feature_vector`` as separate differentials, so ``phase`` is a
    required argument and the CV is run once per phase). For each
    candidate ``k`` in ``k_grid`` it scores, with rounds-weighted mean
    binomial log loss (see the module docstring's round-level scoring
    rationale), the held-out map-phase round outcomes of a walk-forward
    fold scheme over the training region (``split_matches`` carves out the
    final test slice, which is never scored; ``walk_forward_folds`` then
    yields the expanding-window folds over the train region). Each held-out
    instance is estimated *exactly as it would be live*: the as-of cutoff
    is that map's own match timestamp (not the fold boundary), so the
    estimate is built from a strictly independent snapshot — this is what
    proves CV itself is leakage-safe. Both sides of every finished
    validation map count as separate instances (each with its own round
    counts; the two rounds of the per-instance estimate that do not depend
    on ``k`` — the map-phase ``rounds_won``/``rounds_played`` and the
    shrunk overall prior — are precomputed once per instance, and each
    candidate ``k`` is then applied through the shared posterior formula,
    which reproduces ``team_map_side_rate(...).mean`` for every ``k``
    exactly because that estimator's inputs are ``k``-independent). The
    returned ``scores_by_k`` holds the *rounds-weighted* mean binomial log
    loss per candidate, and ``best_k`` is the argmin (lower is better;
    ties break toward the earliest ``k`` in the grid).

    Probabilities are clipped into ``[eps, 1 - eps]`` (see
    :data:`_PROB_CLIP_EPS`) before scoring so a degenerate 0/1 posterior
    mean cannot produce an infinite log loss.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``status``). Only completed matches participate in the
            split/folds (a live match has no scoreable map outcome).
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``winner`` and the full ``round_detail`` required column set).
        phase: The phase to cross-validate (``"attack"`` or
            ``"defense"``).
        k_grid: The candidate strengths to search; any iterable of
            positive finite reals (default :data:`DEFAULT_K_GRID`).
            Duplicate values collapse to one dict entry.
        n_folds: Passed to :func:`utils.splits.walk_forward_folds`.
        min_fold_block: Passed to :func:`utils.splits.walk_forward_folds`.
        test_frac: Passed to :func:`utils.splits.split_matches`.

    Returns:
        A ``(best_k, scores_by_k)`` tuple. ``best_k`` is the grid value
        with the lowest rounds-weighted mean binomial log loss (an element
        of, and key in, ``scores_by_k``). ``scores_by_k`` maps each grid
        value to its rounds-weighted mean binomial log loss over all
        validation instances.

    Raises:
        ValueError: If ``phase`` is invalid (see :func:`_validate_phase`);
            if ``k_grid`` is empty; if a candidate ``k`` is not a positive
            finite real number (see :func:`_validate_k`); if the completed
            matches table is too small for the split/fold machinery
            (propagated from :func:`utils.splits.split_matches` /
            :func:`utils.splits.walk_forward_folds`); if the folds produce
            zero scoreable validation round instances; if a validation
            ``match_id`` is missing from ``matches_df`` or a finished
            validation map fails ``round_detail``'s validation (see
            :func:`_collect_validation_instances`); or if an as-of query
            inside scoring fails (propagated from
            :func:`_team_phase_rounds` / :func:`team_overall_side_rate`).
        KeyError: If a table lacks a required column (propagated from
            pandas / the as-of helpers).
        ConfigError: If a map name is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    _validate_phase(phase)
    grid = list(k_grid)
    if not grid:
        raise ValueError("k_grid must contain at least one candidate k")
    for k in grid:
        _validate_k(k)

    completed = matches_df[
        matches_df[asof.STATUS_COL] == asof.COMPLETED_STATUS
    ].copy()

    splits_df = split_matches(completed, test_frac=test_frac)
    train_ids = set(
        splits_df.loc[splits_df["split"] == "train", asof.MATCH_ID_COL]
    )
    train_matches = completed[completed[asof.MATCH_ID_COL].isin(train_ids)]
    folds = list(
        walk_forward_folds(
            train_matches,
            n_folds=n_folds,
            min_fold_block=min_fold_block,
        )
    )

    instances = _collect_validation_instances(matches_df, maps_df, phase, folds)
    if not instances:
        raise ValueError(
            f"select_k for phase {phase!r} produced zero scoreable "
            "validation instances; cannot choose k from an empty "
            "held-out set"
        )

    # Precompute the per-instance estimator inputs that do not depend on
    # k: the map-phase sample the estimator would see as-of this
    # instance's own date, and the shrunk overall prior. Each candidate k
    # is then applied arithmetically via _beta_posterior, reproducing
    # team_map_side_rate(...).mean exactly (its inputs are k-independent),
    # so the sweep does not re-run the as-of queries once per grid value.
    precomputed: list[tuple[int, int, float, int, int]] = []
    for team_id, map_name, date, _won, _played in instances:
        est_won, est_played = _team_phase_rounds(
            team_id, date, matches_df, maps_df, phase, map_name=map_name
        )
        prior = team_overall_side_rate(
            team_id, phase, date, matches_df, maps_df
        ).mean
        precomputed.append((est_won, est_played, prior, _won, _played))

    total_rounds = sum(played for _w, _p, _pr, _won, played in precomputed)

    scores_by_k: dict = {}
    for k in grid:
        k_value = _validate_k(k)
        total_log_loss = 0.0
        for est_won, est_played, prior, won, played in precomputed:
            _alpha, _beta, mean, _variance = _beta_posterior(
                est_won, est_played, prior, k_value
            )
            p = min(max(mean, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
            total_log_loss += -(
                won * math.log(p) + (played - won) * math.log(1.0 - p)
            )
        scores_by_k[k] = total_log_loss / total_rounds

    best_k = min(grid, key=lambda candidate: scores_by_k[candidate])
    return best_k, scores_by_k

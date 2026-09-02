"""Team-map first-blood differential: shrunk FK/(FK+FD) rate (roadmap M38.4).

A pure in-memory library estimator for a team's first-blood rate on one
named map — the share of first-blood events, ``FK / (FK + FD)``, that the
team won, where ``FK`` is its players' first kills summed over a map's
roster rows and ``FD`` is its players' first deaths summed the same way
(the roadmap's "summed over the map's five players" level-0 shape). The
quantity is partial-pooled up a two-level Beta-binomial shrinkage
hierarchy exactly as M38.2 pools side round win rates: the headline
estimator :func:`team_map_first_blood_rate` returns the Beta posterior
mean ``(FK + k*prior) / (FK + FD + k)`` for one ``(team, map_name)``
pooled over the team's whole as-of history on that map, and the inner
level :func:`team_overall_first_blood_rate` pools the team's *overall*
as-of first bloods the same way toward the league-wide constant
:data:`LEAGUE_FIRST_BLOOD_RATE` (``0.5``). Like the rest of
``features/`` it has no CLI, no ``argparse`` entry point and no file I/O
of its own — it operates on the already-materialised
``matches_df``/``maps_df``/``player_map_stats_df`` DataFrames a caller
passes in.

**Data access: the ``features.player_form`` pattern, reused not
reimplemented (decision 3).** ``player_map_stats`` has neither a ``date``
nor a ``team_id`` column, and ``utils/asof.py``'s own docstring says
wiring that table in is deliberately out of scope "pending an
id-resolution step" — the step :func:`features.player_form.team_player_form`
(roadmap M16) already is. This module reuses its shape almost verbatim:
(1) :func:`utils.asof.maps_as_of` returns the team's finished,
strictly-prior ``(match_id, map_index)`` keys carrying ``team_is_team1``
and (via ``maps_df``'s own carried-through columns) ``map_name``; (2)
:func:`features._shared._build_match_name_lookup` resolves each as-of
map's ``(team1_name, team2_name)`` and the queried team's display name
for that match is ``team1_name`` iff ``team_is_team1``; (3)
``player_map_stats_df`` is grouped by ``(match_id, map_index)`` and each
as-of map's group is looked up and passed through
:func:`features._shared._validated_roster` — the exact fail-loud
"``team_name`` matches neither side" guard the roadmap's ambiguity 4
asks for; (4) a missing group (no ``player_map_stats`` rows at all for a
finished as-of map) is **skipped and counted** in ``maps_skipped``, the
same skip-and-count ``player_form.py`` established (which is what
countably excludes the two no-player-rows maps of match 712803 below).
``features.round_detail`` is deliberately **not** imported: this module
never touches ``maps_df``'s round-count columns, so the round-detail
substrate is irrelevant to it (the module-boundary lateral allowance for
``round_detail.py`` exists for the round-detail-consuming trio
M38.1-M38.3; this milestone was explicitly built in parallel against
``player_map_stats`` and must not import it just because the allowance
exists).

**Two hard invariants, asserted; one soft fact, documented only
(decision 4).** (a) *No nulls* in ``first_kills``/``first_deaths``: each
resolved group's two columns are null-checked before any sum (the
resolved roster is a subset of the group, so this subsumes the
roster-level check the roadmap asks for, and it keeps the conservation
sums well-defined), and a null raises ``ValueError`` — the source data
is clean (0 nulls in v1), so this is defensive fail-loud coverage, not
a live-data fix. (b)
*Per-map conservation* ``Σ FK == Σ FD`` over the **full**
``(match_id, map_index)`` group (both teams' up-to-10 rows): FK on one
side is definitionally an FD on the other (a round's first-kill event
always has exactly one killer and exactly one victim, on opposite
teams), so the two totals are two countings of the same event set and
must be exactly equal; the check runs at the group-lookup site, before
roster filtering, and a violation raises ``ValueError`` naming the
offending ``(match_id, map_index)``. The *rounds-agreement* fact
(``Σ FK`` over both teams ≈ the map's rounds played) is **not** asserted
anywhere — it is recorded in the Data-shape findings below as a soft
observation only (per the roadmap-reading ambiguity #6 in the task
brief).

**The exact-``0.5`` league identity — why the inner prior is a literal
constant (decision 6/Ground truth).** Because ``Σ FK == Σ FD`` holds on
every individual map, it holds on the sum over *any* subset of maps (a
sum of exact equalities is exact) — including every as-of pool at every
cutoff date. The league-wide pooled first-blood rate
``ΣFK / (ΣFK + ΣFD)`` is therefore **exactly ``0.5``, always, by
construction** — the same structural-identity shape M38.3's
``LEAGUE_MEAN_SIGNED_MARGIN = 0.0`` established for signed margin. It is
exposed as the module constant :data:`LEAGUE_FIRST_BLOOD_RATE` rather
than a function that queries ``matches_df``/``maps_df`` (no league-wide
as-of pool is needed), and the identity is proven against real
``data/v1`` in the Data-shape findings below.

**Two-level Beta-binomial shrinkage hierarchy — one CV'd ``k`` plus one
fixed constant (decision 6).** ``FK/(FK+FD)`` is a genuine Beta-binomial
rate, so both levels use the Beta posterior mean shape
``(FK + k*prior) / (FK + FD + k)`` (``map_win_rate``/``side_win_rate``'s
form — not the linear pooled mean ``signed_margin`` uses, which is only
appropriate for the continuous signed quantity M38.3 estimates):

- **Inner level** (:func:`team_overall_first_blood_rate`): the team's
  overall first-blood rate pooled over *all* its as-of maps, shrunk
  toward the *exactly-known* structural constant
  :data:`LEAGUE_FIRST_BLOOD_RATE` (``0.5``) with the fixed strength
  :data:`DEFAULT_OVERALL_K` — **not** CV'd. The prior it shrinks toward
  is exactly known by the conservation identity (no pooling uncertainty
  at all, the same reasoning M38.3's ``DEFAULT_OVERALL_K`` used for its
  exactly-known ``0.0`` prior), and the team-overall sample is already
  large (243-1053 trials, mean ~647 in v1), so a fixed, comparatively
  light constant is justified — see :data:`DEFAULT_OVERALL_K`'s comment
  for the concrete scale reasoning.
- **Outer level** (:func:`team_map_first_blood_rate`, the headline
  estimator M38.5 will consume): the team's first-blood rate on one
  named map pooled over its whole as-of history on that map, shrunk
  toward the team's *shrunk overall* rate — level 2's output, not the
  raw team-overall rate and not ``0.5`` directly, which is what makes
  the hierarchy genuinely two-level — with the strength ``k`` chosen by
  :func:`select_k` via walk-forward CV. Unlike M38.3 (sized S, no CV
  instruction, continuous quantity), this milestone is sized **M** and
  the quantity is a clean Beta-binomial rate for which
  ``side_win_rate.select_k``'s harness is a direct template. Flagged
  assumption: if a reviewer wanted both levels CV'd (or neither), that
  is the interpretive call to revisit; this BUILD records the
  CV'd-outer/fixed-inner reading as its explicit interpretation,
  mirroring M38.2's own resolution.

**``select_k``'s scoring (first-blood-trial-weighted binomial log loss,
not map-weighted).** Each held-out finished validation map yields two
instances — one per side, resolved to ``team_id`` via the match row —
and each instance's ground truth is that map's own per-side
``(FK, FD)`` (the single map instance's roster sums, obtained through
the same shared per-map resolver the team-level aggregators use).
Each instance's estimate is the outer estimator evaluated at that map's
own match timestamp (as-of history strictly before it — the leakage
proof), and the aggregate score is the *first-blood-trial-weighted* mean
binomial log loss ``sum(instance log loss) / sum(FK + FD over all
instances)`` with probabilities clipped via :data:`_PROB_CLIP_EPS` —
weighting by trials rather than by map instances, since the whole point
of the feature is resolution at the first-blood-event level (the
analogue of ``side_win_rate``'s rounds-weighted aggregate).

**Exclusion/countability contract (decision 5).** The two no-player-rows
maps are handled entirely by the skip-and-count step above — no separate
exclusion list or dataclass. Every public result-bearing
function/dataclass exposes ``maps_used``/``maps_skipped`` counts (the
``PlayerFormResult.as_of_maps``/``skipped_maps`` mirror, at the
per-function level), so the exclusion is visible and countable at every
query; the real-data smoke test asserts the whole-history skip count for
a match-712803 team is exactly 2.

**Data-shape findings (re-derived against real ``data/v1``, plan item 1
— not copied from the plan; ``player_map_stats.parquet`` (2420 rows) +
``maps.parquet`` (244 rows) + ``matches.parquet`` (98 rows), no as-of
filter):**

- **No nulls.** ``first_kills``/``first_deaths`` have 0 null values
  across all 2420 ``player_map_stats`` rows.
- **Exactly 5 players per ``(match_id, map_index, team_name)`` group**,
  every time (484 groups, all size 5) — "summed over the map's five
  players" is a literal, universal shape in v1, not just the common
  case.
- **242 of 244 maps have player rows.** The 2 missing are
  ``(match_id=712803, map_index=0)`` and ``(match_id=712803,
  map_index=1)`` — the same two maps M38.1 excludes for null half-split
  round columns; the two exclusions are independent code paths (M38.1
  checks ``maps.parquet`` round columns; this module checks for an
  absent ``player_map_stats`` group) that happen to agree, not a shared
  invariant to assert cross-module.
- **Per-map first-blood conservation holds exactly on all 242 maps with
  player rows:** summed over both teams' 10 players,
  ``first_kills.sum() == first_deaths.sum()`` on every one (0
  violations) — the hard invariant asserted at derivation time.
- **The exact-``0.5`` league identity:** global ``first_kills.sum() ==
  first_deaths.sum()`` (both equal **5178**) over all 2420 rows, by the
  per-map conservation argument above.
- **The "total first bloods ≈ rounds played" fact is soft-only.** Per-map
  ``(fk_sum - rounds_played)`` over both teams: median ``0.0``, mean
  ``-0.037``, std ``0.40``, range ``[-6, 0]`` (never positive; some
  rounds have no recorded first blood). Documented as an observation,
  never asserted.
- **``team_name`` resolves 1:1** onto ``matches_df``'s
  ``team1_name``/``team2_name`` with zero mismatches, and every resolved
  ``team_name`` maps to exactly one ``team_id`` (16 distinct teams, 16
  distinct ids) — re-verified via a full outer join.
- **Sample sizes** (``n = FK + FD``, the Beta-binomial trial count):
  per single map instance, one team (484 rows): range **13-32**, mean
  **~21.4**; per ``(team, map_name)`` pooled over the team's as-of
  history (137 distinct pairs): range **13-263**, mean **~75.6**, median
  **64** — the outer level's target sample scale; per team overall (16
  teams): range **243-1053**, mean **~647** — roughly an order of
  magnitude above the outer level, confirming the two levels carry
  genuinely different amounts of evidence.
- **Real between-team dispersion exists around the exact ``0.5``
  prior:** unshrunk per-team overall first-blood rate spans
  **0.4195-0.5836** (mean ~0.5026) across the 16 teams — shrinking
  straight to ``0.5`` at low sample sizes would discard a real signal,
  the same justification M38.2/M38.3 both make for keeping two levels.

**Leakage contract (the hard requirement, reused from M12, not
reimplemented).** Every estimate obtains history exclusively through
:func:`utils.asof.maps_as_of` — never by reading
``matches.parquet``/``maps.parquet``/``player_map_stats.parquet``
directly. ``player_map_stats`` carries no date of its own; its rows
enter a query *only* through the ``(match_id, map_index)`` keys of the
as-of-filtered maps, so it inherits the same strict-``<`` boundary
automatically. The strict ``<`` boundary, null-date rejection and
timezone-naive-only rules are enforced by ``utils.asof``'s public parse
helpers, and ``select_k`` scores every held-out map against a snapshot
taken at that map's own match timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from features._shared import (
    TEAM1_NAME_COL,
    TEAM2_NAME_COL,
    TEAM_NAME_COL,
    _build_match_name_lookup,
    _validate_k,
    _validated_roster,
)
from utils import asof, config
from utils.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_FRAC,
    MIN_FOLD_BLOCK_MATCHES,
    split_matches,
    walk_forward_folds,
)

# The exact league-wide first-blood rate, *by construction*, not by
# estimation. Proof: on every individual map, summed over both teams' 10
# players, ``sum(first_kills) == sum(first_deaths)`` exactly — FK on one
# side is definitionally an FD on the other (each round's first-kill
# event has exactly one killer and exactly one victim, on opposite
# teams, so the two totals are two countings of the same event set).
# Summing that exact equality over any subset of maps keeps it exact, so
# the pooled league rate ``sum(FK) / (sum(FK) + sum(FD))`` is exactly
# 0.5 at every as-of cutoff by construction (verified against real
# data/v1: both global sums equal 5178 — see the module docstring's
# Data-shape findings). Because the value is a structural constant
# rather than an empirical pool statistic, it is exposed as a literal
# module constant — the inner level's ``prior`` — not derived from
# ``matches_df``/``maps_df``/``player_map_stats_df`` at call time, which
# is also why this module needs no league-wide as-of pool.
LEAGUE_FIRST_BLOOD_RATE = 0.5

# The maps-table map-name column this module filters on (the map filter
# of the map-level estimator); ``team_is_team1``/``date`` come from
# utils.asof's maps output.
MAP_NAME_COL = "map_name"

# The player_map_stats columns this module sums. A map-instance group's
# ``first_kills`` is summed over the team's five roster rows (the level-0
# aggregation); ``first_deaths`` the same way. Named once here so the
# functions, the docstrings and the tests share one spelling.
FIRST_KILLS_COL = "first_kills"
FIRST_DEATHS_COL = "first_deaths"

# Fixed inner-level (level-2) shrinkage strength: the effective prior
# sample size (in *first-blood trials*) that
# :func:`team_overall_first_blood_rate` gives the exactly-known
# :data:`LEAGUE_FIRST_BLOOD_RATE` (``0.5``) prior when shrinking a
# team's overall first-blood rate toward it. It is deliberately NOT
# cross-validated — the roadmap gives a formula for only the outer
# (map-level) ``k`` (the same reading M38.2 resolved), and this constant
# mirrors ``side_win_rate.DEFAULT_PRIOR_K``'s fixed-constant precedent
# (itself the ``closeness.DEFAULT_OT_K`` analogue), justified by scale
# rather than tuned. The scale argument (module docstring's Data-shape
# findings): a team's overall sample is 243-1053 trials in v1 (mean
# ~647) — an order of magnitude above the per-(team, map) level and
# *larger* than the 121-533-round per-team-phase scale where
# ``side_win_rate.DEFAULT_PRIOR_K = 50`` was chosen — while the prior it
# shrinks toward is *exactly* known (a pure structural constant with no
# pooling uncertainty at all), so the genuine unknown is the
# *between-team* dispersion (the 16 v1 teams' unshrunk rates span
# 0.4195-0.5836 around the exact 0.5), which over-shrinking would
# compress away. Reusing ``k = 50`` on this larger sample shrinks
# *less* than it does in ``side_win_rate``, which is the correct
# direction: at the v1 minimum sample (243 trials) the prior holds
# 50/293 ~ 17% of the posterior weight (vs ~29% at ``side_win_rate``'s
# 121-round minimum), at the mean (~647 trials) ~7% (vs ~14% there),
# and at the maximum (1053) ~4.5% — the team's own much-larger sample
# dominates throughout, exactly as it should when the only unknown is a
# real but bounded between-team spread. A judgment call, isolated as a
# named constant and overridable per call, not a magic number.
DEFAULT_OVERALL_K = 50.0

# Documented fallback shrinkage strength for ad-hoc callers of the
# outer-level estimator (:func:`team_map_first_blood_rate`). It is NOT
# what cross-validation reports: the chosen value is :func:`select_k`'s
# ``best_k``, and this constant only gives hand-written calls a sane
# default when no CV has been run (mirroring ``map_win_rate.DEFAULT_K``
# and ``side_win_rate.DEFAULT_K``).
DEFAULT_K = 10.0

# The default candidate grid :func:`select_k` searches over when the
# caller does not pass one. Starts from ``side_win_rate.DEFAULT_K_GRID``'s
# shape as the plan-mandated baseline, then extended/confirmed at BUILD
# time once the real-v1 CV argmin was observed (the same "explore before
# freezing a specific default" caution ``map_win_rate.DEFAULT_K_GRID``
# states). On real ``data/v1`` the curve decreases monotonically across
# the whole grid and its optimum sits in the heavy-k asymptote: the
# trial-weighted mean log loss runs {1.0: 0.705236, 2.0: 0.704359,
# 5.0: 0.702344, 10.0: 0.700182, 20.0: 0.697837, 50.0: 0.695309,
# 100.0: 0.694158, 200.0: 0.693561, 500.0: 0.693245, 1000.0: 0.693158},
# i.e. the argmin lands at the grid's top edge (best_k 1000.0). Probes
# beyond the top confirm an asymptote near ln 2 ~ 0.6931 (2000: 0.693121,
# 5000: 0.693101, 10000: 0.693095 — the honest-prediction entropy floor,
# consistent with ``side_win_rate``'s own ~0.693 scale): per-doubling
# gains beyond k = 1000 are at or below ~4e-5 per trial, so values past
# 1000 carry no material additional signal and the grid stops there. A
# pragmatic geometric grid; no principled default is specified by roadmap
# M38.4, so this is a tunable constant, not a magic number buried in the
# CV loop.
DEFAULT_K_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

# Clip epsilon for the per-trial probability handed to binomial log loss
# inside :func:`select_k`, mirroring ``map_win_rate._PROB_CLIP_EPS`` /
# ``side_win_rate._PROB_CLIP_EPS``: a posterior mean can reach exactly
# 0.0/1.0 (a 100%-first-blood sample with a 1.0 prior), where log loss
# is -inf/raises; clipping into ``[eps, 1-eps]`` keeps the score finite.
_PROB_CLIP_EPS = 1e-12

# Extra columns this module needs on each table, beyond what
# ``utils.asof.maps_as_of`` already requires. The matches extras are the
# two side names used by the per-match name resolution; the maps extra
# is the per-match map-ordering/grouping key (carried through
# ``maps_as_of``'s output unchanged); the player_map_stats set is the
# full set the aggregation reads (``player_name`` is deliberately not
# required: the per-map sums count roster rows by team, not identities).
_MATCHES_REQUIRED = (TEAM1_NAME_COL, TEAM2_NAME_COL)
_MAPS_REQUIRED = (asof.MAP_INDEX_COL,)
_PMS_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.MAP_INDEX_COL,
    TEAM_NAME_COL,
    FIRST_KILLS_COL,
    FIRST_DEATHS_COL,
)


@dataclass(frozen=True)
class ShrunkOverallFirstBloodRate:
    """The inner-level (level-2) Beta posterior for a team's overall first-blood rate.

    The level-2 posterior of the two-level hierarchy: a team's
    first-blood rate pooled over *all* its as-of maps, shrunk toward the
    exactly-known structural constant :data:`LEAGUE_FIRST_BLOOD_RATE`
    (``0.5``) with the fixed strength :data:`DEFAULT_OVERALL_K`.
    ``first_kills``/``first_deaths`` are the team's roster-summed totals
    over its as-of history (``n = first_kills + first_deaths`` is the
    Beta-binomial trial count); ``alpha``/``beta`` are the full posterior
    parameters ``Beta(FK + k*prior, FD + k*(1 - prior))`` — exposed, not
    just the point estimate — and ``mean`` is the shrinkage point
    estimate ``alpha / (alpha + beta)`` (``== (FK + k*prior) / (FK + FD +
    k)``); ``variance`` is the Beta variance. ``prior`` is always
    ``LEAGUE_FIRST_BLOOD_RATE`` (the literal ``0.5`` constant, never
    re-derived from data); ``raw_rate`` is the unshrunk team rate
    ``FK / (FK + FD)``, or exactly ``prior`` when the sample is empty
    (full shrinkage). ``maps_used``/``maps_skipped`` count the as-of maps
    that contributed / were skipped-and-counted for a missing or
    empty-roster ``player_map_stats`` group (the exclusion contract).

    Attributes:
        first_kills: The team's total roster-summed first kills over its
            as-of maps.
        first_deaths: The team's total roster-summed first deaths over
            its as-of maps.
        maps_used: Number of as-of maps whose group contributed a
            ``(first_kills, first_deaths)`` pair.
        maps_skipped: Number of as-of maps skipped-and-counted (missing
            ``player_map_stats`` group, or a group with no rows for the
            queried team).
        prior: The prior mean fed in (always
            :data:`LEAGUE_FIRST_BLOOD_RATE`).
        raw_rate: The unshrunk team rate ``FK / (FK + FD)``, or exactly
            ``prior`` when the sample is empty.
        alpha: The posterior ``alpha`` parameter.
        beta: The posterior ``beta`` parameter.
        mean: The shrinkage point estimate ``alpha / (alpha + beta)``.
        variance: The Beta posterior variance.
    """

    first_kills: int
    first_deaths: int
    maps_used: int
    maps_skipped: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


@dataclass(frozen=True)
class ShrunkFirstBloodRate:
    """The outer-level (final, headline) Beta posterior for one ``(team, map)``.

    The level-1 posterior of the two-level hierarchy: a team's
    first-blood rate on one named map pooled over its as-of history on
    that map, shrunk toward the team's *shrunk overall* first-blood rate
    (the inner level's output) with the CV-chosen strength ``k``.
    ``first_kills``/``first_deaths`` are the team's roster-summed totals
    on that map over its as-of history; ``alpha``/``beta`` are the full
    posterior parameters ``Beta(FK + k*prior, FD + k*(1 - prior))``;
    ``mean`` is the shrinkage point estimate ``alpha / (alpha + beta)``
    (the roadmap's ``(FK + k*prior) / (FK + FD + k)``) — this is the
    value M38.5 will read for its one opponent differential; ``variance``
    is the Beta variance. ``prior`` here is
    ``ShrunkOverallFirstBloodRate.mean`` — the *shrunk* inner estimate,
    not the raw team-overall rate and not ``0.5`` directly — which is
    exactly what makes the hierarchy two-level rather than two
    independent single-level shrinkages; ``raw_rate`` is the unshrunk
    map rate ``FK / (FK + FD)``, or exactly ``prior`` when the sample is
    empty (full shrinkage). ``maps_used``/``maps_skipped`` count the
    as-of maps on this map name that contributed / were skipped (the
    exclusion contract, visible at map level too).

    Attributes:
        first_kills: The team's total roster-summed first kills on this
            map over its as-of history.
        first_deaths: The team's total roster-summed first deaths on
            this map over its as-of history.
        maps_used: Number of as-of maps on this map name whose group
            contributed a ``(first_kills, first_deaths)`` pair.
        maps_skipped: Number of as-of maps on this map name
            skipped-and-counted (missing group or empty queried-team
            roster).
        prior: The prior mean fed in (``ShrunkOverallFirstBloodRate.mean``
            at the same date — the *shrunk* overall rate).
        raw_rate: The unshrunk map rate ``FK / (FK + FD)``, or exactly
            ``prior`` when the sample is empty.
        alpha: The posterior ``alpha`` parameter.
        beta: The posterior ``beta`` parameter.
        mean: The shrinkage point estimate ``alpha / (alpha + beta)``.
        variance: The Beta posterior variance.
    """

    first_kills: int
    first_deaths: int
    maps_used: int
    maps_skipped: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


def _beta_posterior(
    first_kills: int, first_deaths: int, prior: float, k: float
):
    """Compute the Beta posterior parameters from counts, prior and strength.

    The single shared arithmetic behind every shrinkage site in the
    hierarchy (inner level, outer level, and :func:`select_k`'s scoring
    sweep): given a ``first_kills``-of-``(first_kills + first_deaths)``
    sample, a prior mean ``prior`` and a shrinkage strength ``k``
    (effective prior sample size), return ``alpha = FK + k*prior``,
    ``beta = FD + k*(1 - prior)``, the posterior mean ``alpha / (alpha +
    beta)`` and the Beta variance ``alpha*beta / ((alpha + beta)^2 *
    (alpha + beta + 1))``. Because ``select_k`` computes each held-out
    estimate through this helper with k-independent inputs, its sweep
    reproduces ``team_map_first_blood_rate``'s mean for every candidate
    ``k`` without re-running the as-of queries.

    Args:
        first_kills: The sample's first-kill count (``int``-coercible).
        first_deaths: The sample's first-death count.
        prior: The prior mean fed in (``ShrunkOverallFirstBloodRate.mean``
            at the outer level, ``LEAGUE_FIRST_BLOOD_RATE`` at the inner
            level).
        k: The shrinkage strength (positive finite real; validated by the
            caller).

    Returns:
        An ``(alpha, beta, mean, variance)`` tuple of floats.

    Raises:
        Nothing: inputs are assumed pre-validated (``k > 0``, finite
            ``prior``, non-negative counts); passing unvalidated values
            would silently produce a degenerate posterior.
    """
    alpha = first_kills + k * prior
    beta = first_deaths + k * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return alpha, beta, mean, variance


def _side_counts_from_group(
    group: pd.DataFrame,
    resolved_name: str,
    valid_names: set,
    match_id: str,
    map_index: int,
) -> tuple[int, int] | None:
    """Validate one ``player_map_stats`` group and return one side's ``(FK, FD)``.

    The shared per-map resolver behind both the team-level aggregators
    and :func:`select_k`'s validation-instance collector (plan decision
    4, steps in one place). It enforces the two hard invariants on the
    group: (a) the **no-nulls check** runs first over the full group's
    ``first_kills``/``first_deaths`` columns — a null anywhere in the
    group (including the opponent's rows, which the conservation check
    below must sum) raises ``ValueError``; this subsumes the resolved-
    roster check, since the roster is a subset of the group; (b) the
    **conservation check** then runs on the full (now null-free) group
    (both teams' up-to-10 rows — it needs both sides present, so it
    runs here at the group site, before any roster filtering): if
    ``sum(first_kills) != sum(first_deaths)`` over the whole group, a
    ``ValueError`` naming the offending ``(match_id, map_index)`` is
    raised. The group is then passed through
    :func:`features._shared._validated_roster` (the fail-loud
    team-name reconciliation: a row whose ``team_name`` matches neither
    side of the match raises). An empty resolved roster (the group
    holds rows only for the opponent) is *not* an error: it returns
    ``None`` and the caller treats it as skip-and-count, the same
    convention ``player_form``/``_validated_roster`` establish.

    Args:
        group: The ``player_map_stats`` rows for one ``(match_id,
            map_index)`` key (non-empty; the caller short-circuits an
            absent group before calling).
        resolved_name: The queried team's display name for this match
            (``team1_name`` if the team was team1 else ``team2_name``).
        valid_names: The match's two side names
            (``{team1_name, team2_name}``).
        match_id: The match id (used in error messages).
        map_index: The map index (used in error messages).

    Returns:
        A ``(first_kills, first_deaths)`` tuple of ``int`` for the
        queried team's roster rows, or ``None`` when the queried team has
        no rows in the group (skip-and-count case).

    Raises:
        ValueError: If any ``first_kills``/``first_deaths`` value in the
            full group is null (no-nulls hard invariant); if the full
            group's ``sum(first_kills)`` differs from
            ``sum(first_deaths)`` (per-map conservation violation; the
            message names the ``(match_id, map_index)`` and both sums);
            or if any row's ``team_name`` matches neither side of the
            match (propagated from :func:`_validated_roster`).
    """
    if (
        group[FIRST_KILLS_COL].isna().any()
        or group[FIRST_DEATHS_COL].isna().any()
    ):
        raise ValueError(
            f"player_map_stats for match {match_id!r} map_index {map_index} "
            "contains a null first_kills/first_deaths value; the no-nulls "
            "hard invariant requires every value in the full group to be "
            "present (a null would also poison the conservation sums)"
        )
    group_fk = int(group[FIRST_KILLS_COL].sum())
    group_fd = int(group[FIRST_DEATHS_COL].sum())
    if group_fk != group_fd:
        raise ValueError(
            f"player_map_stats for match {match_id!r} map_index {map_index} "
            "violates the per-map first-blood conservation invariant: "
            f"sum(first_kills) == {group_fk} != sum(first_deaths) == "
            f"{group_fd} over the full group (both teams); FK on one side "
            "is definitionally an FD on the other, so the two totals must "
            "be exactly equal"
        )
    roster = _validated_roster(
        group, resolved_name, valid_names, match_id, map_index
    )
    if roster.empty:
        return None
    return int(roster[FIRST_KILLS_COL].sum()), int(
        roster[FIRST_DEATHS_COL].sum()
    )


def _pms_groups(player_map_stats_df: pd.DataFrame) -> dict:
    """Build the ``(match_id, map_index) -> group`` lookup for the stats table.

    Groups the already-required-columns-validated ``player_map_stats``
    frame by ``(match_id, map_index)`` once so both the team-level loops
    and :func:`select_k`'s collector share one lookup. The map-index key
    is cast to ``int`` (matching ``player_form``'s convention) so a
    float-typed fixture column cannot silently miss its group.

    Args:
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index`` and the columns the
            consumers read).

    Returns:
        A ``dict`` mapping each ``(match_id, map_index)`` key to its
        group DataFrame.

    Raises:
        Nothing.
    """
    return {
        (key[0], int(key[1])): group
        for key, group in player_map_stats_df.groupby(
            [asof.MATCH_ID_COL, asof.MAP_INDEX_COL], sort=False
        )
    }


def _team_first_bloods(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    map_name: str | None = None,
) -> tuple[int, int, int, int]:
    """Sum a team's as-of ``(first_kills, first_deaths)`` and count used/skipped maps.

    The shared aggregator behind both public levels (plan decision 3,
    steps 1-6 in one place). It fetches the team's completed,
    strictly-earlier maps through :func:`utils.asof.maps_as_of` (never by
    reading the Parquet tables directly), resolves each as-of map's
    ``(match_id, map_index)`` group in ``player_map_stats_df`` and passes
    it through :func:`_side_counts_from_group` (conservation check on the
    full group, then roster validation/name reconciliation, then the
    no-nulls check) to get the queried team's roster-summed pair. When
    ``map_name`` is given the as-of frame is first filtered to that map
    (both sides normalized through
    :func:`utils.config.normalize_map_name`, matching
    ``map_win_rate.team_map_win_rate``'s established convention), so the
    returned sums are restricted to the queried map. A missing group (no
    ``player_map_stats`` rows at all for that key) or an empty
    queried-team roster is skipped and counted in ``maps_skipped``; only
    maps whose group yields a real pair increment ``maps_used``.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``team1_name``/``team2_name`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires).
        maps_df: The materialised ``maps`` table (needs ``map_index``,
            and ``map_name`` when ``map_name`` is not ``None``, in
            addition to the columns ``maps_as_of`` already requires).
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``first_kills``, ``first_deaths``). Its rows enter the query
            only via the as-of maps' keys, so it inherits the strict-``<``
            boundary.
        map_name: When not ``None``, restrict the sums to the queried
            team's as-of maps on this map (normalized via
            :func:`utils.config.normalize_map_name`).

    Returns:
        A ``(first_kills, first_deaths, maps_used, maps_skipped)`` tuple
        of ``int``, summed over the queried team's own roster rows of the
        surviving as-of maps (optionally map-restricted). An empty as-of
        history (or a map restriction matching nothing) yields
        ``(0, 0, 0, 0)`` — a normal, non-error outcome.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of` /
            :func:`utils.asof.require_columns`; includes
            ``team1_name``/``team2_name``, ``map_index``, ``map_name``
            (when a map filter is requested) and the
            ``player_map_stats`` columns).
        ValueError: If a group violates the conservation or no-nulls hard
            invariants, or carries a ``team_name`` matching neither side
            of its match (see :func:`_side_counts_from_group` /
            :func:`features._shared._validated_roster`); if a match's two
            side names are identical (see
            :func:`features._shared._build_match_name_lookup`); or if the
            query date or a row date is null/unparseable/timezone-aware
            (propagated from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    asof.require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    asof.require_columns(maps_df, _MAPS_REQUIRED, "maps_df")
    asof.require_columns(
        player_map_stats_df, _PMS_REQUIRED, "player_map_stats_df"
    )

    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)
    if map_name is not None:
        asof.require_columns(maps_df, (MAP_NAME_COL,), "maps_df")
        normalized = config.normalize_map_name(map_name)
        maps = maps[
            maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized
        ]

    match_names = _build_match_name_lookup(matches_df)
    groups = _pms_groups(player_map_stats_df)

    first_kills = 0
    first_deaths = 0
    maps_used = 0
    maps_skipped = 0

    for row in maps.itertuples(index=False):
        match_id = getattr(row, asof.MATCH_ID_COL)
        map_index = int(getattr(row, asof.MAP_INDEX_COL))
        team_is_team1 = bool(getattr(row, asof.TEAM_ORIENTATION_COL))

        team1_name, team2_name = match_names[match_id]
        resolved_name = team1_name if team_is_team1 else team2_name

        group = groups.get((match_id, map_index))
        if group is None:
            maps_skipped += 1
            continue
        pair = _side_counts_from_group(
            group, resolved_name, {team1_name, team2_name}, match_id, map_index
        )
        if pair is None:
            maps_skipped += 1
            continue
        first_kills += pair[0]
        first_deaths += pair[1]
        maps_used += 1

    return first_kills, first_deaths, maps_used, maps_skipped


def team_overall_first_blood_rate(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    k=DEFAULT_OVERALL_K,
) -> ShrunkOverallFirstBloodRate:
    """Return a team's shrunk overall first-blood rate as of a cutoff.

    The inner level (level 2) of the shrinkage hierarchy. It sums the
    team's ``(first_kills, first_deaths)`` over *all* its as-of maps via
    :func:`_team_first_bloods` (no map filter) and shrinks the resulting
    raw rate toward the structural constant
    :data:`LEAGUE_FIRST_BLOOD_RATE` (``0.5`` — *not* re-derived from data
    at call time; see the module docstring's exact-``0.5`` identity) with
    the Beta posterior ``mean = (FK + k*prior) / (FK + FD + k)``, using
    the fixed strength ``k`` (default :data:`DEFAULT_OVERALL_K`; see its
    comment for the scale reasoning). With an empty sample the formula
    degrades to ``mean == prior == 0.5`` exactly (full shrinkage — the
    correct behaviour, not a special case). The full posterior
    (``alpha``/``beta``/``mean``/``variance``) is exposed, and the
    map-level estimator consumes ``.mean`` as its own prior.

    Args:
        team_id: The queried team's stable id.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``team1_name``/``team2_name`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires).
        maps_df: The materialised ``maps`` table (needs ``map_index`` in
            addition to the columns ``maps_as_of`` already requires).
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``first_kills``, ``first_deaths``).
        k: The inner-level shrinkage strength (effective prior sample
            size in first-blood trials); must be a positive finite real
            number (see :func:`features._shared._validate_k`).

    Returns:
        A :class:`ShrunkOverallFirstBloodRate` with the team's
        ``first_kills``/``first_deaths`` over all its as-of maps, the
        structural ``prior`` (``LEAGUE_FIRST_BLOOD_RATE``, always
        ``0.5``), the unshrunk ``raw_rate`` (equal to ``prior`` when the
        sample is empty), the map ``maps_used``/``maps_skipped`` counts,
        and the posterior ``alpha``, ``beta``, ``mean`` and ``variance``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`features._shared._validate_k`); if an as-of map's
            group violates the conservation/no-nulls invariants or the
            team-name reconciliation (see :func:`_team_first_bloods`); or
            if the query date or a row date is null/unparseable/
            timezone-aware (propagated from the as-of helpers).
        KeyError: If either table lacks a required column (propagated
            from the as-of helpers).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
    """
    k_value = _validate_k(k)

    first_kills, first_deaths, maps_used, maps_skipped = _team_first_bloods(
        team_id, date, matches_df, maps_df, player_map_stats_df
    )
    prior = LEAGUE_FIRST_BLOOD_RATE
    n = first_kills + first_deaths
    raw_rate = first_kills / n if n else prior

    alpha, beta, mean, variance = _beta_posterior(
        first_kills, first_deaths, prior, k_value
    )
    return ShrunkOverallFirstBloodRate(
        first_kills=first_kills,
        first_deaths=first_deaths,
        maps_used=maps_used,
        maps_skipped=maps_skipped,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def team_map_first_blood_rate(
    team_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    k,
) -> ShrunkFirstBloodRate:
    """Return the outer-level shrunk first-blood rate for one ``(team, map)``.

    The headline estimator of the milestone (the quantity M38.5 reads
    ``.mean`` from for its opponent differential). It sums the team's
    ``(first_kills, first_deaths)`` on the queried ``map_name`` via
    :func:`_team_first_bloods` (both sides normalized through
    :func:`utils.config.normalize_map_name`, so case/whitespace never
    break a match) and takes the inner level's *shrunk* overall
    first-blood rate (:func:`team_overall_first_blood_rate`, whose
    ``mean`` is the posterior of the team-overall-toward-``0.5``
    shrinkage — not the raw team-overall rate and not ``0.5`` directly,
    which is what makes this a genuine two-level hierarchy) as its prior,
    then applies the Beta posterior
    ``mean = (FK + k*prior) / (FK + FD + k)`` with the CV-chosen strength
    ``k``. With an empty sample on the map the formula degrades to
    ``mean == prior`` exactly (full shrinkage — the correct behaviour,
    not a special case). No map-pool/era filtering happens here: a map
    name outside the caller's active pool is still a legitimate
    historical map to count (pool filtering is a caller concern, e.g.
    M38.5).

    Args:
        team_id: The queried team's stable id.
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching, so
            ``"breeze"``/``" Breeze "`` both match ``"Breeze"``.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``team1_name``/``team2_name`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires).
        maps_df: The materialised ``maps`` table (needs ``map_index``,
            and ``map_name``, in addition to the columns
            ``maps_as_of`` already requires).
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``first_kills``, ``first_deaths``).
        k: The outer-level shrinkage strength (effective prior sample
            size in first-blood trials); must be a positive finite real
            number (see :func:`features._shared._validate_k`).

    Returns:
        A :class:`ShrunkFirstBloodRate` with the map-level
        ``first_kills``/``first_deaths``, the *shrunk overall* ``prior``
        (``team_overall_first_blood_rate(...).mean`` at the same date),
        the unshrunk ``raw_rate`` (equal to ``prior`` when the sample is
        empty), the map-level ``maps_used``/``maps_skipped`` counts, and
        the posterior ``alpha``, ``beta``, ``mean`` and ``variance``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`features._shared._validate_k`); if an as-of map's
            group violates the conservation/no-nulls invariants or the
            team-name reconciliation (see :func:`_team_first_bloods`); or
            if the query date or a row date is null/unparseable/
            timezone-aware (propagated from the as-of helpers).
        KeyError: If either table lacks a required column (propagated
            from the as-of helpers; includes ``map_name``).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)

    first_kills, first_deaths, maps_used, maps_skipped = _team_first_bloods(
        team_id,
        date,
        matches_df,
        maps_df,
        player_map_stats_df,
        map_name=map_name,
    )
    overall = team_overall_first_blood_rate(
        team_id, date, matches_df, maps_df, player_map_stats_df
    )
    prior = overall.mean
    n = first_kills + first_deaths
    raw_rate = first_kills / n if n else prior

    alpha, beta, mean, variance = _beta_posterior(
        first_kills, first_deaths, prior, k_value
    )
    return ShrunkFirstBloodRate(
        first_kills=first_kills,
        first_deaths=first_deaths,
        maps_used=maps_used,
        maps_skipped=maps_skipped,
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
    player_map_stats_df: pd.DataFrame,
    folds: list[tuple[int, list, list]],
) -> list[tuple[str, str, str, int, int]]:
    """Build the scored ``(team_id, map_name, date, first_kills, first_deaths)`` instances.

    Turns the walk-forward fold assignment into the flat list of held-out
    map first-blood outcomes that :func:`select_k` scores, mirroring
    ``side_win_rate._collect_validation_instances``'s shape but returning
    per-side ``(first_kills, first_deaths)`` roster sums instead of round
    counts. Every finished validation map yields up to *two* instances —
    one per side, resolved to ``team_id`` via the match row (each side is
    an independent as-of query and a genuine test of the shrinkage
    estimate, the same "both sides are independent instances" convention
    as M13's assumption 8). Each side's pair comes from the map's
    ``player_map_stats`` group through :func:`_side_counts_from_group`,
    so the two hard invariants (per-map conservation on the full group;
    no nulls on the resolved roster) apply to validation maps exactly as
    they do to as-of history. A validation map with no ``player_map_stats``
    group at all (the real 242/244 gap) contributes no instance for
    either side — it is skipped, not raised; a side whose roster is empty
    inside a present group contributes no instance for that side.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``team1_name``, ``team2_name``); its ``match_id`` values must
            be unique.
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_index``, ``map_name``, ``winner``). Only finished maps
            (``winner`` non-null) contribute instances.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``first_kills``, ``first_deaths``).
        folds: The ``(fold_id, train_ids, val_ids)`` tuples from
            :func:`utils.splits.walk_forward_folds`.

    Returns:
        A list of ``(team_id, map_name, date, first_kills, first_deaths)``
        tuples in fold order, up to two per finished validation map whose
        group yields both sides' roster pairs.

    Raises:
        ValueError: If ``matches_df`` contains duplicate ``match_id``
            values; if a validation ``match_id`` is absent from
            ``matches_df``; or if a validation map's group violates the
            conservation/no-nulls invariants or the team-name
            reconciliation (see :func:`_side_counts_from_group` /
            :func:`features._shared._validated_roster`).
        KeyError: If a table lacks a required column (propagated from
            pandas / :func:`features._shared._build_match_name_lookup`).
    """
    if not matches_df[asof.MATCH_ID_COL].is_unique:
        raise ValueError(
            "matches_df contains duplicate match_id values; the "
            "validation-instance lookup would silently collapse them"
        )
    match_by_id = {
        getattr(row, asof.MATCH_ID_COL): row
        for row in matches_df.itertuples(index=False)
    }
    match_names = _build_match_name_lookup(matches_df)
    groups = _pms_groups(player_map_stats_df)
    finished = maps_df[maps_df[asof.WINNER_COL].notna()]

    instances: list[tuple[str, str, str, int, int]] = []
    for _fold_id, _train_ids, val_ids in folds:
        for mid in val_ids:
            match = match_by_id.get(mid)
            if match is None:
                raise ValueError(
                    f"validation match_id {mid!r} is absent from matches_df"
                )
            team1_id = getattr(match, asof.TEAM1_ID_COL)
            team2_id = getattr(match, asof.TEAM2_ID_COL)
            match_date = getattr(match, asof.DATE_COL)
            team1_name, team2_name = match_names[mid]
            match_maps = finished[finished[asof.MATCH_ID_COL] == mid]
            for map_row in match_maps.itertuples(index=False):
                map_index = int(getattr(map_row, asof.MAP_INDEX_COL))
                map_name = getattr(map_row, MAP_NAME_COL)
                group = groups.get((mid, map_index))
                if group is None:
                    continue
                for side_team_id, resolved_name in (
                    (team1_id, team1_name),
                    (team2_id, team2_name),
                ):
                    pair = _side_counts_from_group(
                        group,
                        resolved_name,
                        {team1_name, team2_name},
                        mid,
                        map_index,
                    )
                    if pair is None:
                        continue
                    instances.append(
                        (
                            side_team_id,
                            map_name,
                            match_date,
                            pair[0],
                            pair[1],
                        )
                    )
    return instances


def select_k(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    k_grid=DEFAULT_K_GRID,
    n_folds: int = DEFAULT_N_FOLDS,
    min_fold_block: int = MIN_FOLD_BLOCK_MATCHES,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple:
    """Choose the outer-level shrinkage strength ``k`` by walk-forward CV.

    The CV harness for the map-level shrinkage strength of the two-level
    hierarchy (the outer ``k`` only — the inner level stays fixed at
    :data:`DEFAULT_OVERALL_K`, mirroring M38.2's one-CV'd-outer-plus-
    one-fixed-inner reading; see the module docstring's decision-6
    paragraph). For each candidate ``k`` in ``k_grid`` it scores, with
    first-blood-trial-weighted mean binomial log loss (see the module
    docstring's scoring paragraph), the held-out per-side first-blood
    outcomes of a walk-forward fold scheme over the training region
    (``split_matches`` carves out the final test slice, which is never
    scored; ``walk_forward_folds`` then yields the expanding-window folds
    over the train region). Each held-out instance is estimated *exactly
    as it would be live*: the as-of cutoff is that map's own match
    timestamp (not the fold boundary), so the estimate is built from a
    strictly independent snapshot — this is what proves CV itself is
    leakage-safe. Both sides of every finished validation map count as
    separate instances (each with its own per-side ``(first_kills,
    first_deaths)`` ground truth from that map's ``player_map_stats``
    group); the two per-instance estimator inputs that do not depend on
    ``k`` — the map-restricted as-of ``(first_kills, first_deaths)`` and
    the shrunk overall prior — are precomputed once per instance, and
    each candidate ``k`` is then applied through the shared posterior
    formula, which reproduces ``team_map_first_blood_rate(...).mean`` for
    every ``k`` exactly because that estimator's inputs are
    ``k``-independent. The returned ``scores_by_k`` holds the
    *first-blood-trial-weighted* mean binomial log loss per candidate
    (``sum over instances of -(fk*log(p) + fd*log(1-p)) / sum(fk + fd)``),
    and ``best_k`` is the argmin (lower is better; ties break toward the
    earliest ``k`` in the grid). Probabilities are clipped into
    ``[eps, 1 - eps]`` (see :data:`_PROB_CLIP_EPS`) before scoring so a
    degenerate 0/1 posterior mean cannot produce an infinite log loss.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``status``, ``team1_name``, ``team2_name``). Only completed
            matches participate in the split/folds (a live match has no
            scoreable map outcome).
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``map_index``, ``winner``).
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``first_kills``, ``first_deaths``).
        k_grid: The candidate strengths to search; any iterable of
            positive finite reals (default :data:`DEFAULT_K_GRID`).
            Duplicate values collapse to one dict entry.
        n_folds: Passed to :func:`utils.splits.walk_forward_folds`.
        min_fold_block: Passed to :func:`utils.splits.walk_forward_folds`.
        test_frac: Passed to :func:`utils.splits.split_matches`.

    Returns:
        A ``(best_k, scores_by_k)`` tuple. ``best_k`` is the grid value
        with the lowest first-blood-trial-weighted mean binomial log loss
        (an element of, and key in, ``scores_by_k``). ``scores_by_k``
        maps each grid value to its trial-weighted mean binomial log loss
        over all validation instances.

    Raises:
        ValueError: If ``k_grid`` is empty; if a candidate ``k`` is not a
            positive finite real number (see
            :func:`features._shared._validate_k`); if the completed
            matches table is too small for the split/fold machinery
            (propagated from :func:`utils.splits.split_matches` /
            :func:`utils.splits.walk_forward_folds`); if the folds
            produce zero scoreable validation instances; if a validation
            ``match_id`` is missing from ``matches_df`` or a validation
            map's group violates the conservation/no-nulls/name
            invariants (see :func:`_collect_validation_instances`); or if
            an as-of query inside scoring fails (propagated from
            :func:`_team_first_bloods` /
            :func:`team_overall_first_blood_rate`).
        KeyError: If a table lacks a required column (propagated from
            pandas / the as-of helpers).
    """
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

    instances = _collect_validation_instances(
        matches_df, maps_df, player_map_stats_df, folds
    )
    if not instances:
        raise ValueError(
            "select_k produced zero scoreable validation instances; "
            "cannot choose k from an empty held-out set"
        )

    # Precompute the per-instance estimator inputs that do not depend on
    # k: the map-restricted as-of sample the estimator would see as of
    # this instance's own date, and the shrunk overall prior. Each
    # candidate k is then applied arithmetically via _beta_posterior,
    # reproducing team_map_first_blood_rate(...).mean exactly (its inputs
    # are k-independent), so the sweep does not re-run the as-of queries
    # once per grid value.
    precomputed: list[tuple[int, int, float, int, int]] = []
    for team_id, map_name, date, _fk, _fd in instances:
        est_fk, est_fd, _used, _skipped = _team_first_bloods(
            team_id,
            date,
            matches_df,
            maps_df,
            player_map_stats_df,
            map_name=map_name,
        )
        prior = team_overall_first_blood_rate(
            team_id, date, matches_df, maps_df, player_map_stats_df
        ).mean
        precomputed.append((est_fk, est_fd, prior, _fk, _fd))

    total_trials = sum(fk + fd for _ef, _ed, _pr, fk, fd in precomputed)

    scores_by_k: dict = {}
    for k in grid:
        k_value = _validate_k(k)
        total_log_loss = 0.0
        for est_fk, est_fd, prior, fk, fd in precomputed:
            _alpha, _beta, mean, _variance = _beta_posterior(
                est_fk, est_fd, prior, k_value
            )
            p = min(max(mean, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
            total_log_loss += -(
                fk * math.log(p) + fd * math.log(1.0 - p)
            )
        scores_by_k[k] = total_log_loss / total_trials

    best_k = min(grid, key=lambda candidate: scores_by_k[candidate])
    return best_k, scores_by_k

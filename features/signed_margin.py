"""Signed round-margin strength: shrunk mean signed round margin (roadmap M38.3).

A pure in-memory library estimator for a team's mean signed round margin
on one named map — the "first moment" of the per-map round-margin
distribution whose "second moment" (``features/closeness.py``'s
``map_round_margin_variance``, roadmap M15) is this feature's *conceptual*
precedent only, never an import. The estimated quantity is exactly the
``signed_margin`` field of ``features.round_detail``'s derived records
(``own_score - opp_score`` per side, OT rounds included — see M38.1's
resolved definition), aggregated per ``(team, map)`` over the team's
as-of history and partial-pooled up a two-level shrinkage hierarchy
(see below). Like the rest of ``features/`` it has no CLI, no
``argparse`` entry point and no file I/O of its own — it operates on the
already-materialised ``matches_df``/``maps_df`` DataFrames a caller
passes in, exactly the M13/M38.1/M38.2 library-only precedent. The
roadmap sizes this milestone **S** and gives no CV instruction, so there
is deliberately no ``select_k`` here (decision 3 below); the headline
estimator is :func:`team_map_signed_margin`, the ``.mean`` of whose
result M38.5 will read for its one opponent differential.

**The symmetric-zero identity — why there is no league-wide as-of pool
(decision 2).** ``closeness.py`` and ``side_win_rate.py`` both needed
their own private league-wide as-of filter because their quantities (OT
rate, side round win rate) are empirical rates with no a-priori value:
the pool has to be queried to know what "average" even means as of a
given date. Signed round margin is different. ``features.round_detail``
derives **both** seats' ``signed_margin`` for every surviving map, and
because each seat's value is ``own_score - opp_score`` computed from the
same two scores, team2's value is always the exact negation of team1's on
the same map. Every as-of pool that contains complete map-side pairs —
which every pool does, because ``utils.asof.maps_as_of`` always returns
whole finished maps and this module always keeps the pair together
through ``derive_map_round_details`` before seat-resolving — therefore
sums to exactly ``0``, by exact integer arithmetic (``signed_margin`` is
stored as ``int``; there is no floating-point residual). This means the
league-wide prior for a team's overall signed margin is not something to
estimate from a pool at all: it is the constant
:data:`LEAGUE_MEAN_SIGNED_MARGIN` (``0.0``), always, at every as-of
cutoff. It is exposed as a documented module constant rather than a
function that queries ``matches_df``/``maps_df``, and the identity is
proven against real ``data/v1`` in the Data-shape findings below (global
sum exactly ``0`` over 484 records; every individual ``(match_id,
map_index)`` pair sums to exactly ``0``). Consequence: this module needs
no league-wide as-of plumbing at all — a meaningful scope reduction
relative to M38.2, consistent with this milestone's **S** size — and
nothing needed lifting into a shared module, because the one thing that
looked shared with ``closeness.py`` (computing a league-wide prior) turns
out not to be needed here in the first place.

**Two-level fixed-constant shrinkage hierarchy — and why it is two levels
rather than one (decision 3).** The roadmap gives no formula for M38.3;
it only names the target quantity ("the shrunk mean signed round margin
per (team, map)") and frames it as capturing that "a 13-3 and a 13-11
... differ ... in *strength*" — a team's *general* dominance, not just
its performance on one map. Signed margin is continuous and signed, so
the natural shrinkage form is the linear pooled mean
``mean = (n*raw_mean + k*prior) / (n + k)`` (equivalently
``(sum_margin + k*prior) / (n_maps + k)``), not a Beta posterior (there
is no proportion/rate here — see decision 4). Two design choices follow:

- **Two levels, not one.** If a team's ``(team, map)`` sample were shrunk
  straight toward the constant ``0.0`` (single level), a team with a
  strong track record but a thin sample on one specific map (the
  per-``(team, map)`` sample is 1-12 maps in v1, mean ~3.5, median 3 —
  see the Data-shape findings) would be pulled toward "no signal" and
  lose exactly the general-strength information the roadmap asks this
  feature to carry, even though real between-team dispersion in overall
  mean margin exists (-3.6 .. +2.86 across v1's 16 teams). The hierarchy
  instead shrinks the map-level raw mean toward the team's own *shrunk
  overall* mean margin (level 2's output, not the raw team-overall mean
  — the same "genuinely two-level, not two independent single-level
  shrinkages" property M38.2's hierarchy has), and only the inner level
  shrinks toward the true structural constant ``0.0``. Flagged
  assumption: if the roadmap's silence on hierarchy depth is read as
  calling for a single level, the outer level's ``prior`` argument would
  become the constant ``0.0`` directly and
  :func:`team_overall_signed_margin` would not exist; this BUILD records
  the two-level reading as its explicit interpretation.
- **Both levels use a fixed, documented constant ``k``, not CV.** Unlike
  M38.2, the roadmap never says "choose k by walk-forward CV" in M38.3's
  text, and the milestone is explicitly sized **S**. This mirrors
  ``closeness.py``'s already-shipped ``DEFAULT_OT_K`` precedent (a fixed,
  scale-justified constant, not CV'd) more closely than M38.2's
  mixed one-CV'd-plus-one-fixed reading. The concrete scale reasoning
  for the chosen values is written into the :data:`DEFAULT_OVERALL_K`
  and :data:`DEFAULT_MAP_K` comments against the real sample sizes below.
  Flagged assumption: if a reviewer wanted CV instead,
  :func:`~features.side_win_rate.select_k`'s shape would need a scoring
  target for a *continuous* quantity (squared error against the
  held-out map's own raw mean margin is the natural analogue of M38.2's
  round-level binomial log loss), but the roadmap's silence and the
  **S** sizing argue against building that harness here.

**No Beta posterior, no variance field (decision 4).** Signed margin is
not a rate/proportion, so ``alpha``/``beta``/Beta ``variance`` (the shape
every existing shrinkage dataclass in this repo uses) do not apply;
inventing a normal-normal conjugate variance would require assuming a
variance model the roadmap never asks for, and M38.3 is explicitly framed
as only "the first moment" (M15's ``map_round_margin_variance`` is
already the separate, differently-scoped second-moment companion — a
per-map-name league-wide quantity, not a per-``(team, map)`` one — and
this module must not import it). Both dataclasses therefore expose the
counts and the raw/prior/shrunk means only, with no uncertainty term.

**Seat resolution (decision 5).** ``features/side_win_rate.py``'s
``_team_phase_rounds`` is the closest precedent (same as-of fetch →
``derive_map_round_details`` → seat-orientation filter → optional
map-name filter → sum shape), but this module must not import it (a
lateral feature-to-feature import the module-boundary test forbids), and
there is no genuinely-shared *logic* between this module and either
sibling once the league-pool need disappears — the only thing
superficially "shared" is the seat-resolution *pattern* (M38.2 sums two
``(won, played)`` column pairs selected by phase; this module sums one
``signed_margin`` column with no phase concept at all). The private
helper :func:`_team_signed_margins` reimplements the pattern
independently, exactly as ``side_win_rate.py`` already mirrored
``closeness.py``'s ``_league_maps_as_of`` pattern rather than importing
it.

**Leakage contract (the hard requirement, reused from M12, not
reimplemented).** Every estimator obtains history exclusively through
``utils.asof`` — :func:`utils.asof.maps_as_of`, exactly as M38.2's
per-team levels do — never by reading ``matches.parquet``/
``maps.parquet`` directly. The strict ``<`` boundary, null-date rejection
and timezone-naive-only rules are enforced by ``utils.asof``'s public
helpers, so a map dated equal to or after the query date never enters any
estimate.

**Data-shape findings (re-derived against real ``data/v1``, plan item 1
— not copied from the plan; ``derive_map_round_details`` on the full
244-row ``maps.parquet``, no as-of filter, joined to ``matches.parquet``
to resolve ``team_id``):**

- 484 team-map records survive derivation (242 non-null maps x 2 seats;
  the 2 null-round-column maps of match 712803 are excluded as before,
  the same base M38.2's ground truth used).
- **The symmetric-zero identity holds exactly, not approximately:** the
  global ``signed_margin`` sum over all 484 records is exactly ``0``
  (integer arithmetic — there is no float residual), and for every
  individual ``(match_id, map_index)`` pair the two seats' values cancel
  exactly (max absolute per-pair deviation is ``0``). This is a
  structural consequence of ``round_detail.py``'s own definition
  (``signed_margin = own_score - opp_score``, so team2's value is always
  the exact negation of team1's on the same map) and holds at *any*
  as-of cutoff, because every surviving map in any as-of pool always
  contributes both of its seat rows together.
- Per-team overall map count (16 distinct teams): range **12-50**, mean
  **~30.25**.
- Per-``(team, map_name)`` sample count: range **1-12**, mean **~3.53**,
  median **3** — roughly an order of magnitude smaller than the per-team
  overall count, confirming the two levels of the hierarchy genuinely
  carry different amounts of evidence.
- ``signed_margin`` itself (pooled over all 484 records, both seats):
  range **-13..13**, mean **0.0** (exactly, per the identity above),
  sample std **~5.80** (pools a team's wins and losses together, so it
  is a measure of match-to-match spread, not of any one team's
  dispersion).
- Per-team overall *mean* signed margin (the quantity level 2 estimates,
  unshrunk): range **-3.6 .. +2.86** across the 16 teams — real,
  non-trivial between-team dispersion in general dominance exists in v1,
  confirming that collapsing straight to the ``0.0`` league prior at low
  sample sizes would discard a real signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features import round_detail as rd
from features._shared import _validate_k
from utils import asof, config

# The exact value every team's signed margins average to league-wide,
# *by construction*, not by estimation. Proof: round_detail derives both
# seats' ``signed_margin`` for every surviving map as
# ``own_score - opp_score``, so on any single map team1's value is the
# exact negation of team2's (same two scores, opposite subtraction);
# summing over any as-of pool — which always contains complete map-side
# pairs — therefore cancels to exactly 0, in exact integer arithmetic
# (no floating-point residual). Verified against real data/v1: the global
# sum over all 484 derived records is exactly 0 and every per-
# (match_id, map_index) pair sums to exactly 0 (see the module
# docstring's Data-shape findings). Because the value is a structural
# constant rather than an empirical pool statistic, it is exposed as a
# literal module constant — the inner level's ``prior`` — not derived
# from ``matches_df``/``maps_df`` at call time, which is also why this
# module needs no league-wide as-of pool.
LEAGUE_MEAN_SIGNED_MARGIN = 0.0

# The maps-table map-name column this module filters on (the map filter
# of the map-level estimator); ``team_is_team1``/``date`` come from
# utils.asof's maps output, and the round-detail columns from M8's maps
# table via features.round_detail's constants.
MAP_NAME_COL = "map_name"

# Fixed inner-level (level-2) shrinkage strength: the effective prior
# sample size (in *maps*) that :func:`team_overall_signed_margin` gives
# the exact :data:`LEAGUE_MEAN_SIGNED_MARGIN` (``0.0``) prior when
# shrinking a team's overall mean signed margin toward it. It is
# deliberately NOT cross-validated (the roadmap's M38.3 text never says
# "choose k by CV" and the milestone is sized S — see the module
# docstring's decision-3 paragraph) and mirrors ``closeness.py``'s
# already-shipped fixed-constant ``DEFAULT_OT_K`` precedent, justified by
# scale rather than tuned. The scale argument (module docstring's
# Data-shape findings): a team's overall sample is already moderately
# sized in v1 (12-50 maps, mean ~30.25) and the prior it shrinks toward
# is *exactly* known — a pure structural constant with no pooling
# uncertainty at all, unlike ``closeness.py``'s empirically-pooled OT
# prior — so the genuine unknown is the *between-team* dispersion in
# general dominance (the 16 v1 teams' unshrunk overall mean margins span
# -3.6 .. +2.86), which over-shrinking would compress away. ``k = 10``
# is on the order of the *smaller end* of the observed per-team sample:
# at the v1 minimum sample (12 maps) the prior still holds under half the
# posterior weight (10/22 ~ 45%), at the mean (~30 maps) ~25%, and at
# the max (50 maps) ~17% — a team's own history dominates once it has
# played more than a handful of maps, while a brand-new team (0 maps)
# still degrades cleanly to the exactly-known ``0.0`` prior. A judgment
# call, isolated as a named constant and overridable per call, not a
# magic number.
DEFAULT_OVERALL_K = 10.0

# Fixed outer-level (level-1) shrinkage strength: the effective prior
# sample size (in *maps*) that :func:`team_map_signed_margin` gives the
# team's *shrunk overall* mean margin (level 2's output) when shrinking
# the map-specific raw mean margin toward it. Like
# :data:`DEFAULT_OVERALL_K` it is deliberately NOT cross-validated
# (same roadmap/sizing reasoning), and it must be heavier *relative to
# its own sample scale*: the per-(team, map) sample is much smaller than
# the per-team overall sample (v1 range 1-12, mean ~3.53, median 3 —
# roughly an order of magnitude below the 12-50 overall range), so the
# constant sits at or somewhat above that mean: ``k = 5``. Concretely
# against the v1 numbers: a single-map sample (n = 1) gives the prior
# 5/6 of the posterior weight — a lone map is pulled hard toward the
# team's general strength, which is exactly the "strength" information
# the roadmap wants carried across a thin map sample — while a
# well-sampled map at the v1 maximum (n = 12) gives the raw mean ~70% of
# the weight (prior 5/17 ~ 29%), so the map's own data shows through once
# it is genuinely informative. A judgment call, isolated as a named
# constant and overridable per call, not a magic number.
DEFAULT_MAP_K = 5.0


@dataclass(frozen=True)
class ShrunkOverallSignedMargin:
    """The inner-level (level-2) shrunk mean signed round margin for a team.

    The level-2 posterior of the two-level hierarchy: a team's mean
    signed round margin pooled over *all* its as-of maps, shrunk toward
    the exactly-known structural constant :data:`LEAGUE_MEAN_SIGNED_MARGIN`
    (``0.0``) with the fixed strength :data:`DEFAULT_OVERALL_K`. Unlike
    the rate estimators in this repo there is no ``alpha``/``beta``/
    ``variance``: signed margin is continuous and signed, not a
    proportion, so the posterior is the plain linear pooled mean (see
    the module docstring's decision-4 paragraph), and this dataclass
    exposes the sample counts and the raw/prior/shrunk means only.
    ``raw_mean`` is the unshrunk team-overall mean margin
    ``sum_margin / n_maps``, or exactly ``prior`` when ``n_maps == 0``
    (full shrinkage — no raw sample to compare against); ``mean`` is the
    shrunk estimate ``(sum_margin + k*prior) / (n_maps + k)``, which
    degrades to exactly ``prior`` when ``n_maps == 0`` (the correct
    behaviour, not a special case).
    """

    n_maps: int
    sum_margin: int
    raw_mean: float
    prior: float
    mean: float


@dataclass(frozen=True)
class ShrunkSignedMargin:
    """The outer-level (final, headline) shrunk mean signed margin for one ``(team, map)``.

    The level-1 posterior of the two-level hierarchy: a team's mean
    signed round margin on one named map, shrunk toward the team's
    *shrunk overall mean margin* (the inner level's output) with the
    fixed strength :data:`DEFAULT_MAP_K`. ``prior`` here is
    ``ShrunkOverallSignedMargin.mean`` — the *shrunk* inner estimate,
    not the raw team-overall mean and not the ``0.0`` structural
    constant directly — which is exactly what makes the hierarchy
    two-level rather than two independent single-level shrinkages (the
    same property M38.2's ``ShrunkSideWinRate.prior`` has relative to
    ``ShrunkOverallSideRate.mean``). This is the dataclass M38.5 will
    read ``.mean`` from for its one opponent differential. As with the
    inner level there is no ``alpha``/``beta``/``variance`` field
    (decision 4): ``raw_mean`` is the unshrunk map-mean margin
    ``sum_margin / n_maps``, or exactly ``prior`` when ``n_maps == 0``;
    ``mean`` is the shrunk estimate
    ``(sum_margin + k*prior) / (n_maps + k)``, degrading to exactly
    ``prior`` when ``n_maps == 0``.
    """

    n_maps: int
    sum_margin: int
    raw_mean: float
    prior: float
    mean: float


def _team_signed_margins(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_name: str | None = None,
) -> tuple[int, int]:
    """Sum a team's as-of signed margins and count the maps behind them.

    The shared seat-resolution helper behind both the team-overall and
    the team-map estimators (plan decision 5, steps 1-6 in one place). It
    fetches the team's completed, strictly-earlier maps through
    :func:`utils.asof.maps_as_of` (never by reading the Parquet tables
    directly), runs :func:`features.round_detail.derive_map_round_details`
    on that as-of frame (which validates and derives both seats' records
    for every surviving row, excluding null-round-column rows exactly as
    it does on the raw table), then resolves each surviving record to the
    queried team: a record's ``(match_id, map_index)`` key is looked up in
    the as-of frame's ``team_is_team1`` orientation column, and only the
    record whose seat (``round_detail``'s ``side`` marker) equals
    ``"team1"`` when the team was team1, else ``"team2"``, is kept. When
    ``map_name`` is given the as-of frame is first filtered to that map
    (both sides normalized through
    :func:`utils.config.normalize_map_name`, matching
    ``map_win_rate.team_map_win_rate``'s established convention) — before
    deriving, exactly as ``side_win_rate._team_phase_rounds`` does — and
    ``signed_margin`` is summed over the kept rows.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name`` when
            ``map_name`` is not ``None``, and the full ``round_detail``
            required column set, in addition to the columns
            ``maps_as_of`` already requires).
        map_name: When not ``None``, restrict the sum to the queried
            team's as-of maps on this map (normalized via
            :func:`utils.config.normalize_map_name`).

    Returns:
        A ``(n_maps, sum_margin)`` tuple of ``int``: ``n_maps`` is the
        number of the queried team's own surviving as-of maps (optionally
        map-restricted) and ``sum_margin`` the sum of that team's
        ``signed_margin`` over them. An empty as-of history (or a
        map-restriction matching nothing) yields ``(0, 0)`` — a normal,
        non-error outcome.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of` /
            ``features.round_detail``'s required-column check; includes
            ``map_name`` when a map filter is requested).
        ValueError: If an as-of map fails ``round_detail``'s validation
            (a null score on a surviving row or a case-split/pairing
            violation — propagated from
            :func:`features.round_detail.derive_map_round_details`); or
            if the query date or a row date is null/unparseable/
            timezone-aware (propagated from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
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

    n_maps = len(own)
    sum_margin = int(own["signed_margin"].sum())
    return n_maps, sum_margin


def team_overall_signed_margin(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=DEFAULT_OVERALL_K,
) -> ShrunkOverallSignedMargin:
    """Return a team's shrunk overall mean signed margin as of a cutoff.

    The inner level (level 2) of the shrinkage hierarchy. It sums the
    team's ``signed_margin`` over *all* its as-of maps via
    :func:`_team_signed_margins` (no map filter) and shrinks the
    resulting raw mean toward the structural constant
    :data:`LEAGUE_MEAN_SIGNED_MARGIN` (``0.0`` — *not* re-derived from
    ``matches_df``/``maps_df`` at call time; see the module docstring's
    symmetric-zero identity) with the linear pooled-mean formula
    ``mean = (sum_margin + k*prior) / (n_maps + k)``, using the fixed
    strength ``k`` (default :data:`DEFAULT_OVERALL_K`; see its comment
    for the scale reasoning). With ``n_maps == 0`` the formula degrades
    to ``mean == prior == 0.0`` exactly (full shrinkage — the correct
    behaviour, not a special case).

    Args:
        team_id: The queried team's stable id.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs the full
            ``round_detail`` required column set in addition to the
            columns ``maps_as_of`` already requires).
        k: The inner-level shrinkage strength (effective prior sample
            size in maps); must be a positive finite real number (see
            :func:`features._shared._validate_k`).

    Returns:
        A :class:`ShrunkOverallSignedMargin` with the team's
        ``n_maps``/``sum_margin`` over all its as-of maps, the structural
        ``prior`` (``LEAGUE_MEAN_SIGNED_MARGIN``, always ``0.0``), the
        unshrunk ``raw_mean`` (equal to ``prior`` when ``n_maps == 0``),
        and the shrunk ``mean``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`features._shared._validate_k`); if an as-of map fails
            ``round_detail``'s validation (propagated from
            :func:`_team_signed_margins`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from the
            as-of helpers).
        KeyError: If either table lacks a required column (propagated
            from the as-of helpers).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
    """
    k_value = _validate_k(k)

    n_maps, sum_margin = _team_signed_margins(
        team_id, date, matches_df, maps_df
    )
    prior = LEAGUE_MEAN_SIGNED_MARGIN
    raw_mean = sum_margin / n_maps if n_maps else prior
    mean = (sum_margin + k_value * prior) / (n_maps + k_value)
    return ShrunkOverallSignedMargin(
        n_maps=n_maps,
        sum_margin=sum_margin,
        raw_mean=raw_mean,
        prior=prior,
        mean=mean,
    )


def team_map_signed_margin(
    team_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=DEFAULT_MAP_K,
) -> ShrunkSignedMargin:
    """Return the outer-level shrunk mean signed margin for one ``(team, map)``.

    The headline estimator of the milestone (the quantity M38.5 reads
    ``.mean`` from for its opponent differential). It sums the team's
    ``signed_margin`` on the queried ``map_name`` via
    :func:`_team_signed_margins` (both sides normalized through
    :func:`utils.config.normalize_map_name`, so case/whitespace never
    break a match) and takes the inner level's *shrunk* overall mean
    margin (:func:`team_overall_signed_margin`, whose ``mean`` is the
    posterior of the team-overall-toward-``0.0`` shrinkage — not the raw
    team-overall mean, which is what makes this a genuine two-level
    hierarchy) as its prior, then applies the linear pooled-mean formula
    ``mean = (sum_margin + k*prior) / (n_maps + k)`` with the fixed
    strength ``k`` (default :data:`DEFAULT_MAP_K`; see its comment for
    the scale reasoning). With ``n_maps == 0`` on the map the formula
    degrades to ``mean == prior`` exactly (full shrinkage — the correct
    behaviour, not a special case). No map-pool/era filtering happens
    here: a map name outside the caller's active pool is still a
    legitimate historical map to count (pool filtering is a caller
    concern, e.g. M38.5).

    Args:
        team_id: The queried team's stable id.
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching, so
            ``"lotus"``/``" Lotus "`` both match ``"Lotus"``.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name`` and
            the full ``round_detail`` required column set in addition to
            the columns ``maps_as_of`` already requires).
        k: The outer-level shrinkage strength (effective prior sample
            size in maps); must be a positive finite real number (see
            :func:`features._shared._validate_k`).

    Returns:
        A :class:`ShrunkSignedMargin` with the map-level
        ``n_maps``/``sum_margin``, the *shrunk overall* ``prior``
        (``team_overall_signed_margin(...).mean`` at the same date), the
        unshrunk ``raw_mean`` (equal to ``prior`` when ``n_maps == 0``),
        and the shrunk ``mean``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`features._shared._validate_k`); if an as-of map fails
            ``round_detail``'s validation (propagated from
            :func:`_team_signed_margins`); or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from the
            as-of helpers).
        KeyError: If either table lacks a required column (propagated
            from the as-of helpers; includes ``map_name`` and the
            ``round_detail`` columns).
        TypeError: If the query date is list-like (propagated from the
            as-of helpers).
        ConfigError: If ``map_name`` or any as-of map's ``map_name`` value
            is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)

    n_maps, sum_margin = _team_signed_margins(
        team_id, date, matches_df, maps_df, map_name=map_name
    )
    overall = team_overall_signed_margin(
        team_id, date, matches_df, maps_df
    )
    prior = overall.mean
    raw_mean = sum_margin / n_maps if n_maps else prior
    mean = (sum_margin + k_value * prior) / (n_maps + k_value)
    return ShrunkSignedMargin(
        n_maps=n_maps,
        sum_margin=sum_margin,
        raw_mean=raw_mean,
        prior=prior,
        mean=mean,
    )

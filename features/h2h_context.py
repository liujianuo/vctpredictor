"""Head-to-head and match-context features (roadmap M17).

A pure in-memory library producing four features from the materialised
``matches`` / ``maps`` / ``player_map_stats`` tables:

- :func:`team_pair_h2h` — a Bayesian-shrunk head-to-head win rate for a
  pair of teams, overall or restricted to one map;
- :func:`event_stage` / :func:`match_event_stage` — the integer stage
  number parsed out of a match's ``event_name``;
- :func:`days_since_last_match` — a team's rest gap (days since its most
  recent strictly-prior match);
- :func:`team_roster_change` — a roster-change flag (Jaccard similarity
  between a team's two most-recent as-of rosters) plus a post-change
  exponential decay multiplier.

Like the rest of ``features/``, this module has no CLI, no ``argparse``
entry point, and (except for :func:`load_h2h_context_tables`) no file
I/O of its own — it operates on the already-materialised DataFrames a
caller passes in.

**Leakage contract (the hard requirement).** Every feature that uses
history obtains it exclusively through ``utils.asof`` —
:func:`utils.asof.matches_as_of` / :func:`utils.asof.maps_as_of` — never
``pd.read_parquet`` on ``matches.parquet`` / ``maps.parquet`` /
``player_map_stats.parquet`` directly. The strict ``<`` boundary is
enforced by ``utils.asof``'s date parsing (reused here, not
reimplemented), so a row dated equal to or after the query date never
enters any estimate. ``player_map_stats`` carries no date of its own;
its rows enter a query *only* through the ``(match_id, map_index)`` keys
of the as-of-filtered maps, so they inherit the same strict-``<``
boundary automatically. ``event_stage`` is a pure ``str -> int`` parse
with no history and therefore no leakage surface of its own.

**H2H shrinkage design.** The head-to-head rate is a Beta-posterior
shrinkage estimator reusing ``features.map_win_rate``'s arithmetic and
field shape: ``mean = (wins + k*prior) / (games + k)`` with
``alpha = wins + k*prior`` and ``beta = (games - wins) + k*(1 - prior)``.
Unlike M13 (whose prior is the team's *own* overall win rate, pulled
toward itself), the H2H prior is the **flat constant 0.5** — there is no
single-team anchor for a *pair*, and using team A's overall rate would
bias toward A and break the ``h2h(a, b)`` vs ``h2h(b, a)`` symmetry.
0.5 is the symmetric, maximally-uninformative choice (the same
empty-history value M13's ``OverallWinRate`` uses). The ``k`` chosen is
:data:`DEFAULT_H2H_K = 20.0`, a documented fixed "heavy" constant (not
CV-tuned — no CV routine is in scope for M17, matching M14/M16's
documented-default precedent, not M13's CV precedent). Justification,
re-derived against real ``data/v1`` (never assumed): there are 74
distinct team pairs across 244 maps, with **max 8** maps played between
any single pair, median 3.0 and mean 3.30 (see the data-shape findings
below). ``k = 20`` therefore gives the 0.5 prior 2.5x the weight of the
single most-played pair's entire history and ~6.7x the median pair's —
so even a perfect 8-0 head-to-head record only moves the posterior mean
to ``(8 + 10) / (8 + 20) = 0.643``. That is the "heavily shrunk" regime
M17 asks for.

**Event stage — brittle by design, fail loud on format drift.** The real
``data/v1`` ``event_name`` values are exactly ``"VCT 2026: EMEA Stage
1"`` and ``"VCT 2026: EMEA Stage 2"`` (nothing finer — no "Group"/
"Playoffs"/region tokens exist in this column). :func:`event_stage`
therefore parses with the strict regex ``Stage\\s+(\\d+)`` (a ``search``,
so the token may appear anywhere in the string, but the literal token
must be present) and returns the stage number as an **ordinal int**
(stages are chronologically ordered within a season, so an ordinal is
more defensible than an unordered categorical). A string that does not
contain a parseable ``Stage N`` token raises ``ValueError`` rather than
emitting a sentinel/default stage. **Known limitation, intentionally
brittle:** the day a differently-named event or region appears in
``event_name`` (a playoffs finale, another region's formatting), this
*will* raise. That is deliberate — a wrong stage number would silently
corrupt every downstream feature using it, and the repo convention
(``utils.config``, M13/M16/M17) is to fail at the first sign of
unrecognised input, not to guess. Parsing a region or sub-stage
("Group"/"Playoffs") is explicitly **out of scope**: no such substrings
exist in real ``data/v1``, so inventing that parsing now would be a
second brittle regex with no payoff.

**Days since last match — honest ``None`` on empty history.** For the
queried team, the most recent strictly-prior match's date is found via
:func:`utils.asof.matches_as_of` and the returned gap is
``(query_date - that_date).days``, a non-negative ``int``. Note the plan
text's "``> 0`` always" is not quite exact: the strict ``<`` boundary
guarantees the match is strictly *earlier*, but a match at
``2026-01-01T23:00`` queried at ``2026-01-02T00:30`` yields a gap of
``0`` days (``Timedelta.days`` floors sub-day gaps). The precise
invariant is **non-negative** (``>= 0``), with ``0`` possible for a
same-calendar-day gap. Empty history (unseen team, or a cutoff before
the team's first match) returns ``None`` — an honest sentinel, *not* a
fabricated numeric default, matching ``PlayerFormResult``'s ``mean=None``
precedent rather than ``OverallWinRate``'s 0.5 (a "typical rest gap"
number is not a principled quantity the way a coin-flip 0.5 is).

**Roster-change design (the riskiest sub-feature; every assumption
documented inline).** A team's "current roster" is the set of distinct
``player_name`` values for that team on its **single most recent as-of
map that actually has player rows** — not aggregated over multiple maps,
and *not* literally the most-recent map if that map is one of the real
242/244 player-stat gaps (a map with no rows contributes no roster and is
skipped when selecting the two most-recent evaluable maps; using an empty
roster instead would spuriously report a change against the prior map).
The "prior roster" is the same set from the next-most-recent evaluable
map. **Known limitation:** using single maps (not "the whole previous
match") misses mid-match substitutions, which would only be visible by
comparing multiple most-recent maps — deliberately not attempted in this
v1. With fewer than two evaluable maps, ``changed = None`` (unknown, not
``False``): a fresh team must not be reported as "no roster change" when
there is nothing to compare against. Change detection is the Jaccard
similarity ``|current ∩ prior| / |current ∪ prior|``, with a change
declared when ``similarity < :data:`DEFAULT_JACCARD_THRESHOLD` (0.6)``.
For a 5-player roster with ``d`` differing players Jaccard is
``(5-d)/(5+d)``: ``d=1`` gives ``4/6 ≈ 0.67 >= 0.6`` (no change — a
single stand-in sub is common/noise) and ``d=2`` gives ``3/7 ≈ 0.43``
(change — a genuine shake-up). **Assumption called out:** the 0.6
threshold is a judgment call with no ground truth to validate against in
this repo, and is exposed as an overridable keyword arg, not a hardcoded
literal. The decay multiplier is ``0.5 ** (days_since_change /
half_life)`` with documented default
:data:`DEFAULT_HALF_LIFE_DAYS = 14.0` (two weeks — a judgment call,
explicitly unvalidated/arbitrary; there is no labelled ground truth to
fit it against). ``days_since_change`` is the days between the query
date and the current (new-roster) map's date. **Scope boundary:** this
module returns the flag + decay as its own :class:`RosterChangeResult`
and does **not** multiply ``features.player_form``/``features.map_win_rate``
outputs by it — wiring the decay into other features' consumption is
left to the M20 modelling stage (deliberate scope limit, not an
oversight).

**Data-shape findings (re-derived against real ``data/v1``, not
assumed):**

- ``matches.parquet["event_name"]`` unique values: exactly 2, both
  parse cleanly under the ``Stage\\s+(\\d+)`` regex (``"VCT 2026: EMEA
  Stage 1"``, ``"VCT 2026: EMEA Stage 2"``).
- Head-to-head sparsity: 74 distinct team pairs over 244 maps; max 8
  maps between any pair, median 3.0, mean 3.30 (the numbers behind the
  ``k = 20`` choice above).
- ``player_map_stats`` has exactly 5 rows per side per map (min == max
  == 5) across 242 ``(match_id, map_index)`` groups, each with exactly 2
  distinct ``team_name`` values — the shape the roster-change Jaccard
  math assumes.
- 0 rows with ``team1_name == team2_name`` and 0 rows with
  ``team1_id == team2_id`` (the Deliverable-B same-name bug is currently
  dormant on real data; the fix is defensive fail-loudly coverage, not a
  live-data fix — matching task 017's precedent language).

**Module note on reused helpers.** This module imports
``_validate_k`` / ``_wins_from_oriented_maps`` and
``_build_match_name_lookup`` / ``_chronological_maps`` /
``_validated_roster`` (plus the shared column-name constants) from
``features._shared`` — the single home for helpers genuinely shared by
more than one ``features/`` module — rather than reaching into sibling
feature modules, so the Beta arithmetic, the null/tie score guards, and
the per-match name-resolution/join stay defined in exactly one place
each and no feature module imports a private helper from a sibling
feature module. Feature modules may depend downward on genuine
``utils/`` utilities: the shared as-of parse helpers
(``utils.asof.require_columns`` / ``parse_query_date`` /
``parse_date_column``) are public API and are reused by every feature
module, not a driver import. The win/loss rule is scores-only
(``team1_score`` vs ``team2_score`` via the ``team_is_team1``
orientation flag), never the ``winner`` display string.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from features._shared import (
    TEAM1_NAME_COL,
    TEAM2_NAME_COL,
    TEAM_NAME_COL,
    _build_match_name_lookup,
    _chronological_maps,
    _validated_roster,
    _validate_k,
    _wins_from_oriented_maps,
)
from utils import asof, config
from utils.table_io import DEFAULT_OUTPUT_DIR

# Shrinkage strength for the H2H rate (effective prior sample size). A
# documented fixed "heavy" constant: 2.5x the max real per-pair map
# count (8) and ~6.7x the median (3). See the module docstring's H2H
# shrinkage section for the measured-data justification.
DEFAULT_H2H_K = 20.0

# The flat, symmetric H2H prior mean. Not the queried team's own overall
# rate (that would bias the pair toward one side); see module docstring.
H2H_PRIOR = 0.5

# Roster-change detection threshold (Jaccard similarity) and post-change
# decay half-life (days). Both documented judgment calls, overridable
# per call; see module docstring.
DEFAULT_JACCARD_THRESHOLD = 0.6
DEFAULT_HALF_LIFE_DAYS = 14.0

# The strict event-stage regex. A ``search`` (token anywhere in the
# string) that requires the literal "Stage" token followed by whitespace
# and one or more digits, so "Stage" without a number never matches.
_STAGE_PATTERN = re.compile(r"Stage\s+(\d+)")

# Column names this module reads. ``match_id``, ``date``, ``team_is_team1``
# come from utils.asof's constants / maps_as_of output; ``map_index`` is
# an original maps-table column carried through maps_as_of (shared
# constant in utils.asof); the name columns come from
# features._shared (imported above); the rest are M8's
# matches/maps/player_map_stats columns, named here once so functions
# and tests share one spelling.
MAP_NAME_COL = "map_name"
EVENT_NAME_COL = "event_name"
PLAYER_NAME_COL = "player_name"

# Extra columns this module needs on each table, beyond what
# ``utils.asof.maps_as_of`` already requires.
_MATCHES_REQUIRED = (asof.TEAM1_ID_COL, asof.TEAM2_ID_COL, asof.DATE_COL, asof.STATUS_COL)
_MAPS_REQUIRED = (asof.MATCH_ID_COL, asof.WINNER_COL)
_PMS_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.MAP_INDEX_COL,
    TEAM_NAME_COL,
    PLAYER_NAME_COL,
)


@dataclass(frozen=True)
class ShrunkH2H:
    """The Beta posterior for one team's head-to-head win rate vs another.

    Mirrors ``features.map_win_rate.ShrunkWinRate``'s field shape exactly.
    ``alpha`` / ``beta`` are the full posterior parameters
    ``Beta(wins + k*prior, (games - wins) + k*(1 - prior))`` — exposed,
    not just the point estimate, so callers can read off the uncertainty.
    ``mean`` is the shrinkage point estimate
    ``alpha / (alpha + beta)`` (the roadmap's
    ``(wins + k*prior) / (games + k)``); ``variance`` is the Beta
    variance ``alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))``. ``wins``
    / ``games`` are the *queried team's* record against the other team
    (overall or on the named map); ``prior`` is the flat H2H prior
    (always :data:`H2H_PRIOR`); ``raw_rate`` is the unshrunk rate
    ``wins / games``, or exactly ``prior`` when ``games == 0`` (full
    shrinkage — no raw sample to compare against).
    """

    wins: int
    games: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


@dataclass(frozen=True)
class RosterChangeResult:
    """A team's roster-change flag, similarity and post-change decay.

    ``changed`` is ``None`` when fewer than two evaluable rosters exist
    (unknown, not ``False``); otherwise ``True`` when the Jaccard
    similarity between the two most-recent evaluable rosters is below the
    configured threshold, else ``False``. ``similarity`` is the Jaccard
    ``|current ∩ prior| / |current ∪ prior|`` (``None`` when ``changed``
    is ``None``). ``decay_multiplier`` is
    ``0.5 ** (days_since_change / half_life_days)`` and is populated
    **only** when ``changed`` is ``True`` (``None`` otherwise — no change,
    no decay). ``changed_as_of_date`` is the date of the map that first
    showed the new roster (the "current" map's date), also populated only
    when ``changed`` is ``True``.

    Attributes:
        team_id: The queried team id (echoed unchanged).
        date: The as-of cutoff, exactly as passed in (original string).
        changed: ``True`` / ``False`` / ``None`` as described above.
        similarity: The Jaccard similarity as a ``float``, or ``None``.
        decay_multiplier: The post-change decay multiplier as a
            ``float``, or ``None``.
        changed_as_of_date: The new-roster map's date string, or
            ``None``.
    """

    team_id: str
    date: str
    changed: bool | None
    similarity: float | None
    decay_multiplier: float | None
    changed_as_of_date: str | None


def _validate_jaccard_threshold(threshold) -> float:
    """Validate the roster-change Jaccard threshold and return it as a float.

    The threshold is the similarity *below* which a roster change is
    declared. It must be a finite real in ``[0, 1]``: outside that range
    the "more than N of 5 players differ" semantics break down (a
    threshold of 0 would declare every pair with any difference a
    change; one above 1 would declare nothing ever a change).

    Args:
        threshold: The Jaccard threshold. Any real number
            (``int``/``float``/numpy scalar) is coerced to ``float``.

    Returns:
        ``threshold`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a ``float``, or if the
            result is NaN, infinite, ``< 0`` or ``> 1``.
    """
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"jaccard_threshold must be a real number in [0, 1], got {threshold!r}"
        ) from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"jaccard_threshold must be a finite real number in [0, 1], got {threshold!r}"
        )
    return value


def _validate_half_life_days(half_life_days) -> float:
    """Validate the post-change decay half-life and return it as a float.

    The half-life is the number of days over which the decay multiplier
    halves (``0.5 ** (days / half_life)``). It must be a positive finite
    real: ``<= 0`` would make the exponent divide by zero or go negative
    (a multiplier that *grows* over time — nonsensical for "trust decays
    after a change"), and NaN/inf would poison the multiplier.

    Args:
        half_life_days: The half-life in days. Any real number
            (``int``/``float``/numpy scalar) is coerced to ``float``.

    Returns:
        ``half_life_days`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a ``float``, or if the
            result is NaN, infinite, or ``<= 0``.
    """
    try:
        value = float(half_life_days)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"half_life_days must be a positive real number, got {half_life_days!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"half_life_days must be a positive finite real number, got {half_life_days!r}"
        )
    return value


def _pair_as_of_maps(
    team_a_id: str,
    team_b_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return team A's as-of maps against team B, oriented to team A.

    The H2H history source. It fetches team A's completed, strictly-prior
    maps via :func:`utils.asof.maps_as_of` (the leakage boundary), then
    keeps only those whose *opponent* is ``team_b_id``. The opponent id
    is derived per row from the same match row that already carries
    team A's orientation: when ``team_is_team1`` is truthy team A is
    ``team1`` and the opponent is ``team2_id``, otherwise the opponent is
    ``team1_id``. The output keeps the ``team_is_team1`` orientation and
    score columns untouched so :func:`features._shared._wins_from_oriented_maps`
    can count team A's wins directly.

    Args:
        team_a_id: The queried team's stable id.
        team_b_id: The opponent team's stable id.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``team1_id``, ``team2_id`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires).
        maps_df: The materialised ``maps`` table.

    Returns:
        A ``pandas.DataFrame`` — the subset of team A's as-of maps whose
        opponent is ``team_b_id``, with all of :func:`utils.asof.maps_as_of`'s
        output columns (maps columns + ``date`` + ``team_is_team1``)
        intact and the original index preserved.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of`).
        ValueError: If the as-of-filtered matches frame contains
            duplicate ``match_id`` values, or if the query date or a row
            date is null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
    """
    maps = asof.maps_as_of(team_a_id, date, matches_df, maps_df)

    team1_by_match = {
        getattr(row, asof.MATCH_ID_COL): getattr(row, asof.TEAM1_ID_COL)
        for row in matches_df.itertuples(index=False)
    }
    team2_by_match = {
        getattr(row, asof.MATCH_ID_COL): getattr(row, asof.TEAM2_ID_COL)
        for row in matches_df.itertuples(index=False)
    }
    match_ids = maps[asof.MATCH_ID_COL]
    team1_series = match_ids.map(team1_by_match)
    team2_series = match_ids.map(team2_by_match)

    is_team1 = maps[asof.TEAM_ORIENTATION_COL].astype(bool)
    opponent = team2_series.where(is_team1, team1_series)
    return maps[opponent == team_b_id]


def team_pair_h2h(
    team_a_id: str,
    team_b_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_name: str | None = None,
    k=DEFAULT_H2H_K,
) -> ShrunkH2H:
    """Return the shrunk head-to-head win rate for team A vs team B.

    The shrinkage estimator. It obtains team A's as-of maps *against team
    B* via :func:`_pair_as_of_maps` (the leakage boundary), optionally
    restricts them to one normalised ``map_name`` (via
    :func:`utils.config.normalize_map_name`, matching M13's convention),
    counts team A's wins via the scores (``team1_score``/``team2_score``
    read through the ``team_is_team1`` orientation — never the ``winner``
    display string), and applies the flat-prior Beta posterior
    ``mean = (wins + k*0.5) / (games + k)``. With ``games == 0`` (the
    pair has never played, or never on this map) the formula degrades to
    ``mean == 0.5`` exactly (full shrinkage) and ``raw_rate == prior`` —
    the overwhelmingly common case at v1's scale, and a first-class path,
    not an edge case.

    This computes one direction only (team A's perspective). By
    construction — the underlying as-of match/map rows are identical and
    win/loss is derived from scores, which are complementary — the
    reverse call is symmetric: ``h2h(a, b).games == h2h(b, a).games``,
    ``h2h(a, b).wins + h2h(b, a).wins == games``, and (with the shared
    flat 0.5 prior) ``h2h(a, b).mean + h2h(b, a).mean == 1.0`` exactly
    in floating point. This is documented here rather than computing both
    directions to avoid double work/drift.

    Args:
        team_a_id: The queried team's stable id (its wins are counted).
        team_b_id: The opponent team's stable id.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``team1_score``
            and ``team2_score`` — read by the win counter — and, when
            ``map_name`` is given, ``map_name``).
        map_name: Optional map to restrict the head-to-head history to;
            normalized via :func:`utils.config.normalize_map_name` before
            matching. ``None`` (the default) means overall head-to-head
            across every map.
        k: The shrinkage strength (effective prior sample size); must be
            a positive finite real number (see
            :func:`features._shared._validate_k`).

    Returns:
        A :class:`ShrunkH2H` with team A's ``wins``/``games`` against
        team B (overall or on the named map), the flat ``prior`` (always
        :data:`H2H_PRIOR`), the unshrunk ``raw_rate`` (equal to ``prior``
        when ``games == 0``), and the posterior ``alpha``, ``beta``,
        ``mean`` and ``variance``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`features._shared._validate_k`); if an as-of map has
            a null/NaN score or tied scores (see
            :func:`features._shared._wins_from_oriented_maps`); or if
            the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.maps_as_of`).
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of`; includes ``team1_score``/
            ``team2_score``, and ``map_name`` when ``map_name`` is
            given).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)
    pair_maps = _pair_as_of_maps(team_a_id, team_b_id, date, matches_df, maps_df)

    if map_name is not None:
        normalized_map = config.normalize_map_name(map_name)
        pair_maps = pair_maps[
            pair_maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized_map
        ]

    wins = _wins_from_oriented_maps(pair_maps)
    games = len(pair_maps)
    prior = H2H_PRIOR
    raw_rate = wins / games if games else prior

    alpha = wins + k_value * prior
    beta = (games - wins) + k_value * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))

    return ShrunkH2H(
        wins=wins,
        games=games,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def event_stage(event_name: str) -> int:
    """Parse the integer stage number out of an ``event_name`` string.

    The strict, fail-loud stage parser. It requires the literal token
    ``Stage`` followed by whitespace and one or more digits somewhere in
    the string (``Stage\\s+(\\d+)``, searched, not full-matched) and
    returns that number as an ``int``. A string without such a token —
    including a bare ``"Stage"`` with no number, a different casing, or a
    differently-formatted event/region name — raises ``ValueError``
    rather than emitting a sentinel/default. This brittleness is
    intentional and documented in the module docstring.

    Args:
        event_name: The match's ``event_name`` string (e.g.
            ``"VCT 2026: EMEA Stage 1"``).

    Returns:
        The integer stage number (e.g. ``1``).

    Raises:
        TypeError: If ``event_name`` is not a ``str``.
        ValueError: If ``event_name`` does not contain a parseable
            ``Stage N`` token.
    """
    if not isinstance(event_name, str):
        raise TypeError(
            f"event_name must be a string, got {type(event_name).__name__}: {event_name!r}"
        )
    match = _STAGE_PATTERN.search(event_name)
    if match is None:
        raise ValueError(
            f"event_name {event_name!r} does not contain a parseable "
            "'Stage N' token (expected e.g. 'VCT 2026: EMEA Stage 1'); "
            "refusing to guess a stage number"
        )
    return int(match.group(1))


def match_event_stage(match_id: str, matches_df: pd.DataFrame) -> int:
    """Return a match's stage number by parsing its ``event_name``.

    A thin composition convenience: it looks up the given ``match_id`` in
    ``matches_df``, reads that match's ``event_name``, and delegates the
    parse to :func:`event_stage`. This is what a per-match feature vector
    builder composes most naturally, while :func:`event_stage` remains
    the pure, unit-testable core.

    Args:
        match_id: The match's stable id (a string matching the dtype of
            ``matches_df["match_id"]``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id`` and ``event_name``).

    Returns:
        The integer stage number (see :func:`event_stage`).

    Raises:
        KeyError: If ``matches_df`` lacks ``match_id`` or ``event_name``
            (propagated from :func:`utils.asof.require_columns`).
        ValueError: If ``match_id`` is not present in ``matches_df``, or
            if its ``event_name`` does not parse (propagated from
            :func:`event_stage`).
        TypeError: If the match's ``event_name`` cell is not a string
            (propagated from :func:`event_stage`).
    """
    asof.require_columns(matches_df, (asof.MATCH_ID_COL, EVENT_NAME_COL), "matches_df")
    rows = matches_df[matches_df[asof.MATCH_ID_COL] == match_id]
    if rows.empty:
        raise ValueError(f"match_id {match_id!r} is not present in matches_df")
    # A match's event_name is constant across its rows (one row per match
    # in M8's schema); read the first.
    return event_stage(rows[EVENT_NAME_COL].iloc[0])


def days_since_last_match(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
) -> int | None:
    """Return the team's rest gap in days as of a cutoff date.

    Finds the team's most recent completed match strictly before ``date``
    via :func:`utils.asof.matches_as_of` (the leakage boundary) and
    returns ``(query_date - that_date).days``. The result is non-negative
    but can be ``0`` when the gap is under one calendar day
    (``Timedelta.days`` floors sub-day gaps) — the strict ``<`` boundary
    guarantees strictly-earlier, not at-least-one-day-earlier. Empty
    history (unseen team, or a cutoff before the team's first match)
    returns ``None``: there is no principled "typical gap" number the way
    0.5 is a principled coin-flip prior, so an honest sentinel is used
    rather than a fabricated default.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; matches dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``team1_id``, ``team2_id``, ``date``, ``status``).

    Returns:
        The whole-day gap as a non-negative ``int``, or ``None`` when the
        team has no strictly-prior completed match.

    Raises:
        KeyError: If ``matches_df`` lacks a required column (propagated
            from :func:`utils.asof.matches_as_of`).
        ValueError: If the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.matches_as_of` / the parse helpers).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.matches_as_of`).
    """
    matches = asof.matches_as_of(team_id, date, matches_df)
    if matches.empty:
        return None

    last = pd.to_datetime(matches[asof.DATE_COL]).max()
    query = asof.parse_query_date(date)
    return int((query - last).days)


def _decay_multiplier(days_since_change, half_life_days: float) -> float:
    """Return the post-roster-change trust decay multiplier.

    The exponential decay ``0.5 ** (days_since_change / half_life_days)``.
    At ``days_since_change == 0`` the multiplier is ``1.0`` (the new
    roster is fully relevant immediately); at
    ``days_since_change == half_life_days`` it is ``0.5``. ``days_since_change``
    is expected non-negative (the change map is strictly before the query
    date, so its floor-of-days gap is ``>= 0``).

    Args:
        days_since_change: Whole days since the roster changed (any real
            number; negative values would yield a multiplier ``> 1`` and
            are not produced by this module's call sites).
        half_life_days: The half-life in days (already validated positive
            finite real; see :func:`_validate_half_life_days`).

    Returns:
        The decay multiplier as a ``float``.

    Raises:
        Nothing (``half_life_days`` is expected pre-validated, and the
            formula is total for any finite ``days_since_change``).
    """
    return 0.5 ** (days_since_change / half_life_days)


def _roster_sets_chronological(
    maps_sorted: pd.DataFrame,
    match_names: dict,
    pms_groups: dict,
) -> list[tuple[str, set[str]]]:
    """Collect each as-of map's team roster set, in chronological order.

    Walks the chronologically-sorted as-of maps (oldest first) and, for
    each map that actually has player rows for the queried team, extracts
    the set of distinct ``player_name`` values into a
    ``(date, roster_set)`` pair. Maps with no ``player_map_stats`` group
    at all, or with a group that holds no rows for the queried team, are
    skipped: they contribute no roster and must not fabricate an empty
    set (which would spuriously report a change against the next map).

    Args:
        maps_sorted: The output of
            :func:`features._shared._chronological_maps` over the
            queried team's as-of maps (needs ``match_id``, ``map_index``,
            ``date`` and ``team_is_team1``).
        match_names: The ``match_id -> (team1_name, team2_name)`` lookup
            from :func:`features._shared._build_match_name_lookup`.
        pms_groups: A ``{(match_id, map_index): group}`` dict of
            ``player_map_stats`` rows, as built in
            :func:`features.player_form.team_player_form`.

    Returns:
        A list of ``(date_string, roster_set)`` tuples in chronological
        (oldest-first) order, one per evaluable map.

    Raises:
        ValueError: If a ``player_map_stats`` ``team_name`` matches
            neither side of its match (propagated from
            :func:`features._shared._validated_roster`).
    """
    rosters: list[tuple[str, set[str]]] = []
    for row in maps_sorted.itertuples(index=False):
        match_id = getattr(row, asof.MATCH_ID_COL)
        map_index = int(getattr(row, asof.MAP_INDEX_COL))
        team_is_team1 = bool(getattr(row, asof.TEAM_ORIENTATION_COL))
        date_str = getattr(row, asof.DATE_COL)

        team1_name, team2_name = match_names[match_id]
        resolved_name = team1_name if team_is_team1 else team2_name

        group = pms_groups.get((match_id, map_index))
        if group is None:
            continue
        roster = _validated_roster(
            group, resolved_name, {team1_name, team2_name}, match_id, map_index
        )
        players = set(roster[PLAYER_NAME_COL].dropna())
        if not players:
            continue
        rosters.append((date_str, players))
    return rosters


def team_roster_change(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> RosterChangeResult:
    """Return a team's roster-change flag, similarity and post-change decay.

    The single public roster-change entry point. It (1) fetches the
    team's finished, strictly-prior maps via :func:`utils.asof.maps_as_of`
    (the leakage boundary), (2) resolves the team's display name per map
    via the same M16 linkage this module reuses
    (:func:`features._shared._build_match_name_lookup` /
    :func:`features._shared._validated_roster`), (3) extracts the
    player-name roster set of the two most recent maps that actually have
    player rows, and (4) declares a change when the Jaccard similarity of
    those two rosters is strictly below ``jaccard_threshold``. When a
    change is declared, the decay multiplier
    ``0.5 ** (days_since_change / half_life_days)`` is computed from the
    new-roster map's date; otherwise it is ``None``. With fewer than two
    evaluable maps, ``changed``/``similarity``/``decay_multiplier``/
    ``changed_as_of_date`` are all ``None`` (unknown, not ``False``).

    See the module docstring for the roster definition, the Jaccard
    threshold semantics, the single-most-recent-map limitation, and the
    deliberately-simple, overridable defaults.

    Args:
        team_id: The queried team's stable id (a string matching the
            dtype of ``team1_id``/``team2_id``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table. Needs
            ``team1_name``/``team2_name`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires.
        maps_df: The materialised ``maps`` table. Needs ``map_index`` in
            addition to the columns :func:`utils.asof.maps_as_of`
            already requires.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``,
            ``player_name``). Its rows enter the query only via the
            as-of maps' keys, so it inherits the strict-``<`` boundary.
        jaccard_threshold: The similarity below which a change is
            declared (see :func:`_validate_jaccard_threshold`).
        half_life_days: The post-change decay half-life in days (see
            :func:`_validate_half_life_days`).

    Returns:
        A :class:`RosterChangeResult` as described above.

    Raises:
        KeyError: If any table lacks a required column (propagated from
            :func:`utils.asof.maps_as_of` /
            :func:`utils.asof.require_columns`; includes
            ``team1_name``/``team2_name``, ``map_index``, and the
            ``player_map_stats`` columns).
        ValueError: If ``jaccard_threshold`` or ``half_life_days`` is
            invalid (see the validate helpers); if a match's two side
            names are identical (propagated from
            :func:`features._shared._build_match_name_lookup` — the
            Deliverable-B guard); if a ``player_map_stats`` ``team_name``
            matches neither side of its match (propagated from
            :func:`features._shared._validated_roster`); or if the query
            date or a row date is null/unparseable/timezone-aware
            (propagated from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
    """
    threshold = _validate_jaccard_threshold(jaccard_threshold)
    half_life = _validate_half_life_days(half_life_days)

    asof.require_columns(matches_df, (TEAM1_NAME_COL, TEAM2_NAME_COL), "matches_df")
    asof.require_columns(maps_df, (asof.MAP_INDEX_COL,), "maps_df")
    asof.require_columns(player_map_stats_df, _PMS_REQUIRED, "player_map_stats_df")

    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)
    match_names = _build_match_name_lookup(matches_df)
    maps_sorted = _chronological_maps(maps)

    pms_groups = {
        (key[0], int(key[1])): group
        for key, group in player_map_stats_df.groupby(
            [asof.MATCH_ID_COL, asof.MAP_INDEX_COL], sort=False
        )
    }

    rosters = _roster_sets_chronological(maps_sorted, match_names, pms_groups)

    if len(rosters) < 2:
        return RosterChangeResult(
            team_id=team_id,
            date=date,
            changed=None,
            similarity=None,
            decay_multiplier=None,
            changed_as_of_date=None,
        )

    current_date, current = rosters[-1]
    _prior_date, prior = rosters[-2]
    union = current | prior
    intersection = current & prior
    similarity = len(intersection) / len(union) if union else 0.0
    changed = similarity < threshold

    if changed:
        query = asof.parse_query_date(date)
        days_since = int((query - pd.to_datetime(current_date)).days)
        decay = _decay_multiplier(days_since, half_life)
        changed_as_of = current_date
    else:
        decay = None
        changed_as_of = None

    return RosterChangeResult(
        team_id=team_id,
        date=date,
        changed=changed,
        similarity=similarity,
        decay_multiplier=decay,
        changed_as_of_date=changed_as_of,
    )


def load_h2h_context_tables(
    version: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the materialised matches, maps and player_map_stats tables.

    A thin disk-I/O convenience wrapper mirroring
    :func:`features.player_form.load_player_form_tables`. It reads
    ``<output_dir>/<version>/{matches,maps,player_map_stats}.parquet``
    via ``pandas.read_parquet`` and hands all three to the pure feature
    functions — it re-implements no filtering or feature logic.

    Args:
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        output_dir: The parent directory the version subdirectory lives
            under (default :data:`utils.table_io.DEFAULT_OUTPUT_DIR`,
            i.e. ``Path("data")``).

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple of the
        three loaded DataFrames, in that order.

    Raises:
        FileNotFoundError: If any Parquet file does not exist for this
            version (i.e. ``materialize.py`` has not been run for it) —
            propagated as-is from ``pandas.read_parquet`` as a clear
            "run materialize.py first" signal rather than wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    version_dir = Path(output_dir) / version
    matches_df = pd.read_parquet(version_dir / "matches.parquet")
    maps_df = pd.read_parquet(version_dir / "maps.parquet")
    player_map_stats_df = pd.read_parquet(version_dir / "player_map_stats.parquet")
    return matches_df, maps_df, player_map_stats_df

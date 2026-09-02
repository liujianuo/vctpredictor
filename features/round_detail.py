"""Per-map round-detail derivation + validity assertions (roadmap M38.1).

An explicitly-shared feature-support module (the exact analogue of
``features/_shared.py``): it is the definition that both M38.2
(side-specific shrunk win rates) and M38.3 (signed round-margin
strength) train against, so it deliberately lives under ``features/``
and is excluded from ``FEATURE_MODULES`` in
``tests/test_module_boundaries.py`` just like ``_shared.py``. Neither
downstream milestone imports it sideways from a sibling feature module;
both import it directly, because it is shared feature-support code, not
a trainable feature of its own. It has no CLI and no file I/O of its
own — it operates purely on a caller-passed, maps-shaped DataFrame (a
raw ``maps.parquet`` frame or an as-of-filtered slice of one; it does
not care which).

**Dependency line (M6, M8, M9 only — deliberately NOT M12).** This
module must not import ``utils.asof`` and takes no ``team_id``/as-of
``date`` parameters: M38.2 and M38.3 (which *do* depend on M12) perform
their own as-of filtering via ``utils.asof.maps_as_of`` and then apply
this module's pure derivation to whatever as-of-filtered maps rows they
already hold. Because of that, the derived records carry no ``team_id``
and no orientation column: the canonical key is ``(match_id, map_index,
side)`` where ``side`` is the literal string ``"team1"``/``"team2"``.
Resolving ``side`` to a stable ``team_id`` is the caller's job (it holds
``matches_df`` + the ``team_is_team1`` orientation column from
``utils.asof``); a future reader should not expect a ``team_id`` column
here, because one cannot exist in a module with no as-of dependency.
The one import this module makes from ``features._shared``
(``TEAM1_SCORE_COL``/``TEAM2_SCORE_COL``) is a shared-to-shared
constant import — ``_shared.py`` has no downstream dependents of its
own to protect, so it is not a lateral feature-to-feature import in the
problematic sense (called out explicitly in the module-boundary test).

**Resolved design decisions (do not re-derive in later milestones):**

- **Output shape is a long ``pandas.DataFrame``, two rows per input
  map** (one per ``side``), with ``match_id``/``map_index``/``side`` as
  explicit columns. M38.2 aggregates ``(rounds_won + k*prior) /
  (rounds_played + k)`` per ``(team, map)`` and M38.3 aggregates the
  shrunk mean margin per ``(team, map)`` — both over an as-of-filtered
  history that this module's caller must join/group by ``(match_id,
  map_index, side)`` after resolving ``side`` to ``team_id``. A long
  frame with those key columns supports ``groupby``/``merge`` directly,
  which a bare dataclass list would not; the frozen
  :class:`MapRoundDetail` dataclass remains the canonical per-record
  shape the row-level helper emits, and the top-level result's
  ``records`` frame is materialised from those instances so the two
  representations never drift. The result object also returns the
  excluded rows explicitly (see below), never a silently-shorter frame.
- **Null-row exclusion is countable and visible, never imputed.** The 2
  v1 rows with null round-detail columns are excluded and reported as
  :class:`ExcludedMapRow` entries (their ``match_id``/``map_index`` and
  a reason string) riding alongside the derived records, mirroring
  ``drivers/labels.py::build_labels_table``'s "skipped and counted, not
  dropped" pattern. No assertion hardcodes "must be exactly 2" inside
  the module (that would break the day v1 grows); the real-data smoke
  test in ``tests/test_round_detail.py`` pins today's count instead.
- **Validity checks fail loud (``ValueError``) and always run.**
  Validation runs automatically on every derivation call (the cost is
  trivial at v1's 244-row scale and no plausible caller wants
  unvalidated output) and is also exposed as the separately
  importable/testable :func:`validate_map_round_details`. The
  case-split reconciliation and opposing-side-pairing checks raise on
  violation, listing the offending ``(match_id, map_index)`` pairs
  capped at 5 (the ``features/_shared.py::_wins_from_oriented_maps``
  message convention). Null round columns are the one *expected*
  condition and are excluded, not raised.
- **The ``atk``/``def`` columns are regulation-only.** On OT maps each
  side's ``atk + def`` sums to exactly 12 (regulation went 12-12) and
  the OT rounds are attributed to no side; ``ot_rounds_won`` is
  therefore ``score - 12`` on OT maps and ``ot_rounds_played`` is the
  shared quantity ``team1_score + team2_score - 24`` — identical for
  both sides of the same map (no per-side OT "rounds played" asymmetry
  is derivable from the columns, and the roadmap says OT rounds are
  "kept separate, never folded into a side"; the lack of a per-side OT
  split is a documented gap, not something to invent). Confirmed
  sensible against the real data: ``team1_score + team2_score - 24``
  takes only the values ``{2, 4, 6, 8}`` across v1's 29 OT maps.
- **``signed_margin`` is the full-map margin, OT rounds included**:
  ``team{N}_score - team{O}_score``, reusing
  ``drivers/labels.py::compute_outcome``'s established
  ``team1_score - team2_score`` convention verbatim (positive means
  team1). Roadmap M38.3 does not say whether OT rounds count in the
  margin; this module records the resolved choice as yes, because
  ``features/closeness.py``'s existing ``abs(team1_score -
  team2_score)`` precedent (M15) already treats the full score as the
  margin basis, and M38.3 is explicitly the first moment where M15's
  variance is the second — the two features must use the same
  underlying quantity or they are not a mean/variance pair of the same
  distribution.
- **Derivation uses the ``atk``/``def`` columns directly and never
  cross-checks the first/second-half columns as if half-1 == attack.**
  Which physical half a team attacks in is not fixed (see the
  Data-shape findings below: each half-pairing matches the atk-pairing
  on only ~9-10 of 213 regulation rows), so ``atk_rounds_played`` is
  the opposing-side pairing ``team{N}_atk_rounds + team{O}_def_rounds``
  (every round of the half where N attacks is won either by N
  attacking or by O defending), not any first/second-half sum. The
  first/second-half columns are read only for null-row identification
  (they are null together with the atk/def columns on the same rows).
- **The reconciliation check is case-split, not naive.** The naive
  single-side check ``atk + def == score`` over *all* non-null rows
  passes on only 218/242 (checking team1 alone) / 219/242 (team2
  alone) — the roadmap's exact "218/242" is the team1-only figure and
  looks like it "mostly works" while actually hiding the OT-vs-
  regulation structural issue. The case-split assertion is
  ``atk + def == score`` for regulation maps and ``atk + def == 12``
  per side for OT maps (the roadmap's ``atk == def == 12`` shorthand),
  which passes at 213/213 + 29/29. The opposing-side pairing
  (``team1_atk_rounds + team2_def_rounds`` and ``team1_def_rounds +
  team2_atk_rounds``) is asserted as well: on regulation maps each
  pairing lies in ``[0, 12]`` and together they partition the map's
  total rounds (algebraically implied by the case split, kept as an
  explicit invariant so a scrape that mislabels sides fails loudly);
  on OT maps each pairing is exactly 12 on 29/29 (both teams play a
  full 12-round regulation attack half).

**Data-shape findings (re-derived against real ``data/v1``, plan item
1 — not copied from the plan):**

- ``data/v1/maps.parquet`` has 244 rows. ``team1_score``/``team2_score``
  are ``int64`` and never null; the eight round-detail columns
  (``team{1,2}_{first_half,second_half,atk,def}_rounds``) are
  ``float64`` and null together on exactly 2 rows — both maps of one
  match: ``match_id=712803``, ``map_index=0`` (scoreline 6-13) and
  ``map_index=1`` (scoreline 9-13). 242 of 244 rows have all eight
  columns populated.
- Of the 242 non-null rows, ``min(team1_score, team2_score) >= 12``
  (the OT criterion, used verbatim from ``drivers/labels.py``) holds on
  exactly 29 (OT maps); the other 213 are regulation maps.
- On the 29 OT rows, ``team1_atk_rounds + team1_def_rounds == 12`` and
  ``team2_atk_rounds + team2_def_rounds == 12`` hold on 29/29 for both
  sides — the atk/def columns are regulation-only.
- On the 213 regulation rows, ``team{N}_atk_rounds + team{N}_def_rounds
  == team{N}_score`` holds on 213/213 for both N=1 and N=2.
- The naive one-sided reconciliation over all 242 non-null rows passes
  on 218/242 checking ``team1`` alone, 219/242 checking ``team2``
  alone, and only 213/242 requiring *both* sides — reproducing the
  roadmap's "218/242" exactly (it is the team1-only figure) and
  confirming why the check must be case-split instead.
- Opposing-side pairing on regulation rows: ``(team1_atk_rounds +
  team2_def_rounds) + (team1_def_rounds + team2_atk_rounds) ==
  team1_score + team2_score`` holds on 213/213, with each individual
  pairing sum in ``[1, 12]`` across the real data (never 0, never
  >12; this module asserts the theoretical ``[0, 12]`` bound since 0 is
  reachable even if unobserved). On the 29 OT rows both pairing sums
  are exactly 12 on 29/29.
- Neither ``team1_first_half_rounds + team2_second_half_rounds`` nor
  ``team1_second_half_rounds + team2_first_half_rounds`` matches
  ``team1_atk_rounds + team2_def_rounds`` universally (checked both
  directions; each matches on only 9-10 of the 213 regulation rows),
  i.e. which physical half a team attacks in is not fixed — the
  derivation must read the atk/def columns directly and must not
  cross-check the half columns as if half-1 == attack.
- Signed full-map margins ``team1_score - team2_score`` span -13..13;
  OT rounds ``team1_score + team2_score - 24`` take only the values
  ``{2, 4, 6, 8}`` across the 29 OT maps.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from features._shared import TEAM1_SCORE_COL, TEAM2_SCORE_COL

# Column-name constants for the maps-shaped input this module reads.
# ``team1_score``/``team2_score`` are reused from ``features._shared``
# (one shared spelling across features); ``match_id``/``map_index`` and
# the eight round-detail columns are named here (this module has no
# ``utils.asof`` dependency, so it does not import ``asof``'s constants).
MATCH_ID_COL = "match_id"
MAP_INDEX_COL = "map_index"

TEAM1_FIRST_HALF_COL = "team1_first_half_rounds"
TEAM1_SECOND_HALF_COL = "team1_second_half_rounds"
TEAM1_ATK_COL = "team1_atk_rounds"
TEAM1_DEF_COL = "team1_def_rounds"
TEAM2_FIRST_HALF_COL = "team2_first_half_rounds"
TEAM2_SECOND_HALF_COL = "team2_second_half_rounds"
TEAM2_ATK_COL = "team2_atk_rounds"
TEAM2_DEF_COL = "team2_def_rounds"

# The four half-split and four atk/def round columns, as the module's
# spelling of "the eight round-detail columns".
FIRST_HALF_COLS = (TEAM1_FIRST_HALF_COL, TEAM2_FIRST_HALF_COL)
SECOND_HALF_COLS = (TEAM1_SECOND_HALF_COL, TEAM2_SECOND_HALF_COL)
ATK_DEF_COLS = (
    TEAM1_ATK_COL,
    TEAM1_DEF_COL,
    TEAM2_ATK_COL,
    TEAM2_DEF_COL,
)
ROUND_COLS = FIRST_HALF_COLS + SECOND_HALF_COLS + ATK_DEF_COLS

# The columns a maps-shaped input must carry. ``match_id``/``map_index``
# key the derived records; the scores feed the case split, the margin,
# and the OT criterion; the eight round columns feed the null-row
# identification and the per-side derivation.
REQUIRED_COLUMNS = (
    MATCH_ID_COL,
    MAP_INDEX_COL,
    TEAM1_SCORE_COL,
    TEAM2_SCORE_COL,
) + ROUND_COLS

# OT criterion: a map went to overtime when the losing side reached at
# least this many rounds — the same criterion ``drivers/labels.py`` and
# ``features/closeness.py`` use (reused verbatim, not reinvented). The
# same value doubles as (a) the per-side regulation cap on an OT map
# (each side's ``atk + def`` sums to exactly 12 once regulation reached
# 12-12) and (b) the upper bound of a single regulation attack half
# (a side attacks at most 12 rounds in regulation; the 13th round would
# end the map or push it into OT).
OT_MIN_SCORE = 12

# The two canonical side markers of the derived records. ``side`` is the
# literal ``"team1"``/``"team2"`` string, never a resolved ``team_id``
# (see the module docstring's dependency-line note).
TEAM1_SIDE = "team1"
TEAM2_SIDE = "team2"

# The cap on how many offending rows each ValueError message lists
# (the ``features/_shared.py::_wins_from_oriented_maps`` convention).
_MAX_OFFENDERS_LISTED = 5

# The reason string attached to every excluded (null round-column) row,
# so the exclusion is self-describing rather than a bare key.
NULL_ROUNDS_REASON = (
    "null round-detail columns (team*_{first_half,second_half,atk,def}"
    "_rounds are null); the scraper did not record half/atk/def splits"
)


@dataclass(frozen=True)
class ExcludedMapRow:
    """One input map row excluded from derivation because it is unusable.

    A row whose eight round-detail columns are not all present (the only
    exclusion this module makes — an expected, countable data condition,
    never a failure). The row is reported here with its
    ``match_id``/``map_index`` identity and a ``reason`` string so the
    caller can see exactly what was excluded and why, rather than being
    silently dropped or imputed.
    """

    match_id: int | str
    map_index: int
    reason: str


@dataclass(frozen=True)
class MapRoundDetail:
    """One side's derived round detail for one finished map.

    The canonical record :func:`derive_map_round_details` produces, two
    per input map (one per ``side``). ``atk_rounds_won`` /
    ``def_rounds_won`` are the regulation attack/defence round wins the
    scraper recorded for this side; ``atk_rounds_played`` /
    ``def_rounds_played`` are the regulation rounds this side actually
    attacked/defended (the opposing-side pairing — see the module
    docstring), so ``atk_rounds_won + def_rounds_won`` is the side's
    regulation total and ``atk_rounds_played + def_rounds_played`` the
    map's regulation round count. Overtime is kept strictly separate:
    ``ot_rounds_won`` is this side's OT wins (``0`` on a regulation
    map) and ``ot_rounds_played`` is the shared OT round count,
    identical on both sides of the same OT map. ``signed_margin`` is
    the full-map margin ``this side's score - opponent's score`` (OT
    rounds included), positive when this side won.
    """

    match_id: int | str
    map_index: int
    side: str
    atk_rounds_won: int
    atk_rounds_played: int
    def_rounds_won: int
    def_rounds_played: int
    ot_rounds_won: int
    ot_rounds_played: int
    signed_margin: int


@dataclass(frozen=True)
class MapRoundDetailResult:
    """The top-level derivation result: derived records plus exclusions.

    ``records`` is the long-format derived frame (two rows per surviving
    input map, columns ``match_id``, ``map_index``, ``side``,
    ``atk_rounds_won``, ``atk_rounds_played``, ``def_rounds_won``,
    ``def_rounds_played``, ``ot_rounds_won``, ``ot_rounds_played``,
    ``signed_margin`` — the :class:`MapRoundDetail` field order).
    ``excluded`` holds one :class:`ExcludedMapRow` per input row whose
    round-detail columns were null, so the exclusion is visible and
    countable: ``len(records) == 2 * (len(input) - len(excluded))`` for
    an input with unique ``(match_id, map_index)`` keys.
    """

    records: pd.DataFrame
    excluded: tuple[ExcludedMapRow, ...]


def _require_input_columns(maps_df: pd.DataFrame) -> None:
    """Raise ``KeyError`` if a required maps-shaped column is missing.

    This module has no ``utils.asof`` dependency (M38.1's dependency
    line is M6/M8/M9, deliberately not M12), so it performs its own
    required-column check instead of calling ``asof.require_columns``.
    The missing-column signal (``KeyError`` naming the first absent
    column) matches what pandas would otherwise raise on first access,
    kept explicit so the error fires before any partial computation.

    Args:
        maps_df: The candidate maps-shaped DataFrame.

    Returns:
        Nothing.

    Raises:
        KeyError: If any of :data:`REQUIRED_COLUMNS` is absent from
            ``maps_df``.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in maps_df.columns]
    if missing:
        raise KeyError(
            f"maps_df is missing required round-detail column(s) {missing}; "
            "expected a maps-shaped frame with columns "
            f"{list(REQUIRED_COLUMNS)}"
        )


def _null_round_rows(maps_df: pd.DataFrame) -> pd.DataFrame:
    """Return the input rows with any null round-detail column.

    A row is unusable for round-detail derivation when any of the eight
    round columns (halves and atk/def alike) is null — v1's real data
    has them null together on the same rows, and this module never
    imputes a missing split. The caller decides what to do with the
    returned rows (exclude them as countable :class:`ExcludedMapRow`
    entries); this helper only identifies them.

    Args:
        maps_df: A maps-shaped DataFrame with all of
            :data:`REQUIRED_COLUMNS`.

    Returns:
        A ``pandas.DataFrame`` view of the rows of ``maps_df`` where at
        least one of :data:`ROUND_COLS` is null, preserving input row
        order and index. Empty when no row has a null round column.

    Raises:
        KeyError: If a required column is missing (propagated from
            :func:`_require_input_columns`).
    """
    _require_input_columns(maps_df)
    null_mask = maps_df[list(ROUND_COLS)].isna().any(axis=1)
    return maps_df[null_mask]


def validate_map_round_details(maps_df: pd.DataFrame) -> tuple[ExcludedMapRow, ...]:
    """Validate a maps-shaped frame's round-detail invariants.

    The separately importable/testable validation unit that
    :func:`derive_map_round_details` calls automatically on every
    invocation. It (a) identifies rows with null round-detail columns —
    an expected, countable condition returned as
    :class:`ExcludedMapRow` entries, never raised; (b) asserts the
    case-split reconciliation on the surviving rows: per side,
    ``atk_rounds + def_rounds == score`` on a regulation map and
    ``atk_rounds + def_rounds == 12`` on an OT map (the roadmap's
    ``atk == def == 12`` shorthand), where OT is
    ``min(team1_score, team2_score) >= 12``; and (c) asserts the
    opposing-side pairing on the surviving rows: each pairing
    (``team1_atk_rounds + team2_def_rounds`` and ``team1_def_rounds +
    team2_atk_rounds``) lies in ``[0, 12]`` and the two sum to
    ``team1_score + team2_score`` on a regulation map, and each equals
    exactly ``12`` on an OT map. Violations raise ``ValueError``
    (these are structural impossibilities for a finished, non-null
    map), listing the offending ``(match_id, map_index)`` pairs capped
    at :data:`_MAX_OFFENDERS_LISTED`. A null score on a surviving row
    (round detail present but a score missing) also raises
    ``ValueError`` defensively, before any score arithmetic.

    Args:
        maps_df: A maps-shaped DataFrame (raw ``maps.parquet`` or an
            as-of-filtered slice of one) with all of
            :data:`REQUIRED_COLUMNS`. Rows are expected unique per
            ``(match_id, map_index)``.

    Returns:
        A tuple of :class:`ExcludedMapRow` entries, one per input row
        with a null round-detail column, in input order. Empty when no
        row is excluded. Rows returned here are *not* subject to the
        case-split/pairing assertions (a null row has no atk/def
        reconciliation to check).

    Raises:
        KeyError: If ``maps_df`` lacks a required column (see
            :func:`_require_input_columns`).
        ValueError: If a surviving row has a null ``team1_score`` or
            ``team2_score`` (round detail present but a score missing is
            impossible for a finished map); if a surviving row fails the
            case-split reconciliation; or if a surviving row fails the
            opposing-side pairing assertion.
    """
    _require_input_columns(maps_df)
    null_rows = _null_round_rows(maps_df)
    excluded = tuple(
        ExcludedMapRow(
            match_id=row[MATCH_ID_COL],
            map_index=row[MAP_INDEX_COL],
            reason=NULL_ROUNDS_REASON,
        )
        for _, row in null_rows.iterrows()
    )

    surviving = maps_df[~maps_df.index.isin(null_rows.index)]
    if surviving.empty:
        return excluded

    t1_score = surviving[TEAM1_SCORE_COL].to_numpy(dtype=float)
    t2_score = surviving[TEAM2_SCORE_COL].to_numpy(dtype=float)
    t1_atk = surviving[TEAM1_ATK_COL].to_numpy(dtype=float)
    t1_def = surviving[TEAM1_DEF_COL].to_numpy(dtype=float)
    t2_atk = surviving[TEAM2_ATK_COL].to_numpy(dtype=float)
    t2_def = surviving[TEAM2_DEF_COL].to_numpy(dtype=float)

    # Defensive null-score guard (fail loud, before any comparison): a
    # finished map with round detail present but a score missing is
    # structurally impossible.
    null_score = np.isnan(t1_score) | np.isnan(t2_score)
    if null_score.any():
        offending = _offender_keys(surviving, null_score)
        raise ValueError(
            f"{len(offending)} map(s) with non-null round-detail columns "
            "have a null/NaN team1_score or team2_score, which is "
            "impossible for a finished map; offending (match_id, "
            f"map_index): {offending[: _MAX_OFFENDERS_LISTED]}"
        )

    is_ot = np.minimum(t1_score, t2_score) >= OT_MIN_SCORE

    # Case-split reconciliation: atk + def must equal the side's score
    # on a regulation map, and exactly OT_MIN_SCORE (12) on an OT map.
    per_side_sum1 = t1_atk + t1_def
    per_side_sum2 = t2_atk + t2_def
    expected1 = np.where(is_ot, OT_MIN_SCORE, t1_score)
    expected2 = np.where(is_ot, OT_MIN_SCORE, t2_score)
    case_violation = (per_side_sum1 != expected1) | (per_side_sum2 != expected2)
    if case_violation.any():
        offending = _offender_keys(surviving, case_violation)
        raise ValueError(
            f"{len(offending)} map(s) fail the case-split round "
            "reconciliation (per side, atk_rounds + def_rounds must "
            "equal that side's score on a regulation map, and exactly "
            "12 on an OT map; the naive single-side check hides the "
            "OT-vs-regulation split); offending (match_id, map_index): "
            f"{offending[: _MAX_OFFENDERS_LISTED]}"
        )

    # Opposing-side pairing: on an OT map each pairing is exactly 12
    # (both teams played a full 12-round regulation attack half); on a
    # regulation map each pairing lies in [0, 12] and together they
    # partition the map's total rounds.
    pairing1 = t1_atk + t2_def
    pairing2 = t1_def + t2_atk
    ot_pairing_violation = is_ot & ((pairing1 != OT_MIN_SCORE) | (pairing2 != OT_MIN_SCORE))
    if ot_pairing_violation.any():
        offending = _offender_keys(surviving, ot_pairing_violation)
        raise ValueError(
            f"{len(offending)} OT map(s) fail the opposing-side pairing "
            "assertion (team1_atk_rounds + team2_def_rounds and "
            "team1_def_rounds + team2_atk_rounds must each equal 12 on "
            "an OT map); offending (match_id, map_index): "
            f"{offending[: _MAX_OFFENDERS_LISTED]}"
        )
    regulation = ~is_ot
    reg_total = t1_score + t2_score
    reg_pairing_total = pairing1 + pairing2
    reg_partition_violation = regulation & (reg_pairing_total != reg_total)
    reg_bound_violation = regulation & (
        (pairing1 < 0)
        | (pairing1 > OT_MIN_SCORE)
        | (pairing2 < 0)
        | (pairing2 > OT_MIN_SCORE)
    )
    reg_pairing_violation = reg_partition_violation | reg_bound_violation
    if reg_pairing_violation.any():
        offending = _offender_keys(surviving, reg_pairing_violation)
        raise ValueError(
            f"{len(offending)} regulation map(s) fail the opposing-side "
            "pairing assertion (team1_atk_rounds + team2_def_rounds and "
            "team1_def_rounds + team2_atk_rounds must each lie in "
            "[0, 12] and together equal team1_score + team2_score, so a "
            "side-mislabeling scrape fails here rather than producing a "
            "plausible-but-wrong feature); offending (match_id, "
            f"map_index): {offending[: _MAX_OFFENDERS_LISTED]}"
        )

    return excluded


def _offender_keys(maps_df: pd.DataFrame, mask) -> list[tuple]:
    """Return the offending ``(match_id, map_index)`` pairs of ``mask``.

    A small formatting helper shared by every validation error path so
    the raised messages follow one convention (the
    ``features/_shared.py`` cap-at-5 pattern): it converts a boolean
    mask over ``maps_df`` into a list of ``(match_id, map_index)``
    tuples in input row order, ready to be sliced by the caller.

    Args:
        maps_df: The validated maps-shaped DataFrame.
        mask: A boolean array (or Series) over ``maps_df``'s rows
            marking the offenders; truthy entries are collected.

    Returns:
        A list of ``(match_id, map_index)`` tuples, one per truthy
        ``mask`` entry, in input row order. Empty when ``mask`` is all
        false.

    Raises:
        Nothing.
    """
    flagged = maps_df[mask]
    return list(zip(flagged[MATCH_ID_COL].tolist(), flagged[MAP_INDEX_COL].tolist()))


def _side_record(
    side: str,
    match_id,
    map_index,
    is_ot: bool,
    ot_rounds_played,
    own_score,
    opp_score,
    own_atk,
    own_def,
    opp_atk,
    opp_def,
) -> MapRoundDetail:
    """Build the canonical record for one side of one validated map row.

    The per-side half of the derivation: given the shared row facts
    (``match_id``, ``map_index``, ``is_ot`` and the shared
    ``ot_rounds_played``) plus this side's own score/atk/def counts and
    the opponent's atk/def counts, it applies the module docstring's
    derivation formulas and returns one :class:`MapRoundDetail`. The
    round counts are cast to ``int`` (the source columns are float64
    but hold whole round counts).

    Args:
        side: The side marker (:data:`TEAM1_SIDE` or :data:`TEAM2_SIDE`).
        match_id: The map's match id, carried into the record.
        map_index: The map's 0-based index, carried into the record.
        is_ot: Whether the map went to overtime (``min(scores) >= 12``),
            decided once per row by the caller.
        ot_rounds_played: The shared OT round count
            (``team1_score + team2_score - 24`` on an OT map, else 0),
            decided once per row by the caller and identical on both
            sides.
        own_score: This side's rounds won on the finished map.
        opp_score: The opponent's rounds won.
        own_atk: This side's regulation attack-round wins.
        own_def: This side's regulation defence-round wins.
        opp_atk: The opponent's regulation attack-round wins.
        opp_def: The opponent's regulation defence-round wins.

    Returns:
        A :class:`MapRoundDetail` for this side with ``atk_rounds_played``
        = ``own_atk + opp_def``, ``def_rounds_played`` =
        ``own_def + opp_atk`` (the opposing-side pairings),
        ``ot_rounds_won`` = ``own_score - own_atk - own_def`` on an OT
        map else ``0``, ``ot_rounds_played`` as passed in, and
        ``signed_margin`` = ``own_score - opp_score``.

    Raises:
        Nothing: inputs are assumed pre-validated.
    """
    ot_won = own_score - own_atk - own_def if is_ot else 0
    return MapRoundDetail(
        match_id=match_id,
        map_index=map_index,
        side=side,
        atk_rounds_won=int(own_atk),
        atk_rounds_played=int(own_atk + opp_def),
        def_rounds_won=int(own_def),
        def_rounds_played=int(own_def + opp_atk),
        ot_rounds_won=int(ot_won),
        ot_rounds_played=int(ot_rounds_played),
        signed_margin=int(own_score - opp_score),
    )


def _derive_row_round_details(
    match_id,
    map_index,
    team1_score,
    team2_score,
    team1_atk_rounds,
    team1_def_rounds,
    team2_atk_rounds,
    team2_def_rounds,
) -> tuple[MapRoundDetail, MapRoundDetail]:
    """Derive both sides' round detail for one non-null map row.

    The row-level derivation helper behind :func:`derive_map_round_details`
    (analogous to ``closeness.py::_margin_and_ot``). Inputs are the
    scalar values of one *validated, non-null* row — the caller runs
    :func:`validate_map_round_details` first, so no null checks happen
    here. Per side N (opponent O) it computes, per the module docstring's
    derivation formulas: ``atk_rounds_won = team{N}_atk_rounds``;
    ``atk_rounds_played = team{N}_atk_rounds + team{O}_def_rounds``
    (the opposing-side pairing: every round of the half N attacks is won
    by N attacking or O defending); ``def_rounds_won =
    team{N}_def_rounds``; ``def_rounds_played = team{N}_def_rounds +
    team{O}_atk_rounds``; ``ot_rounds_won = team{N}_score -
    team{N}_atk_rounds - team{N}_def_rounds`` (equals
    ``team{N}_score - 12``) on an OT map else ``0``; ``ot_rounds_played
    = team1_score + team2_score - 24`` on an OT map else ``0``,
    identical for both sides (OT rounds are shared, not side-split);
    ``signed_margin = team{N}_score - team{O}_score`` (the full-map
    margin, OT included). ``is_ot`` is the labels.py criterion
    ``min(team1_score, team2_score) >= 12``.

    Args:
        match_id: The map's match id, carried into both records.
        map_index: The map's 0-based index within the match, carried
            into both records.
        team1_score: Rounds won by team1 on the finished map.
        team2_score: Rounds won by team2 on the finished map.
        team1_atk_rounds: Team1's regulation attack-round wins.
        team1_def_rounds: Team1's regulation defence-round wins.
        team2_atk_rounds: Team2's regulation attack-round wins.
        team2_def_rounds: Team2's regulation defence-round wins.

    Returns:
        A ``(MapRoundDetail, MapRoundDetail)`` tuple — team1's record
        then team2's — each with integer round fields and the side
        marker :data:`TEAM1_SIDE` / :data:`TEAM2_SIDE`.

    Raises:
        Nothing: the inputs are assumed pre-validated (non-null,
        case-split and pairing reconciled). Passing unvalidated values
        would silently produce records that violate the module's
        documented invariants; use :func:`derive_map_round_details`
        instead.
    """
    is_ot = min(team1_score, team2_score) >= OT_MIN_SCORE
    ot_rounds_played = (
        team1_score + team2_score - 2 * OT_MIN_SCORE if is_ot else 0
    )
    team1 = _side_record(
        TEAM1_SIDE,
        match_id,
        map_index,
        is_ot,
        ot_rounds_played,
        team1_score,
        team2_score,
        team1_atk_rounds,
        team1_def_rounds,
        team2_atk_rounds,
        team2_def_rounds,
    )
    team2 = _side_record(
        TEAM2_SIDE,
        match_id,
        map_index,
        is_ot,
        ot_rounds_played,
        team2_score,
        team1_score,
        team2_atk_rounds,
        team2_def_rounds,
        team1_atk_rounds,
        team1_def_rounds,
    )
    return team1, team2


def derive_map_round_details(maps_df: pd.DataFrame) -> MapRoundDetailResult:
    """Derive both sides' round detail for every usable row of a maps frame.

    The top-level public derivation function. It first validates the
    whole input via :func:`validate_map_round_details` (always, per the
    module's design decision — no caller wants unvalidated output, and
    the cost is trivial at v1's 244-row scale): rows with null
    round-detail columns are excluded and returned as countable
    :class:`ExcludedMapRow` entries in the result (never imputed,
    never silently dropped), and a surviving row that fails the
    case-split or opposing-side-pairing assertions raises
    ``ValueError``. Every surviving row then yields two
    :class:`MapRoundDetail` records (one per side) via
    :func:`_derive_row_round_details`, materialised into the result's
    long-format ``records`` frame with the :class:`MapRoundDetail`
    field order. ``match_id``/``map_index`` are expected to key the
    input uniquely (M8's ``maps.parquet`` and ``utils.asof.maps_as_of``
    output both satisfy this), which is what makes ``len(records) ==
    2 * (len(maps_df) - len(excluded))`` an exact contract.

    Args:
        maps_df: A maps-shaped DataFrame (raw ``maps.parquet`` or an
            as-of-filtered slice of one) with all of
            :data:`REQUIRED_COLUMNS`, rows unique per
            ``(match_id, map_index)``.

    Returns:
        A :class:`MapRoundDetailResult` whose ``records`` frame holds
        two rows per surviving input map (see :class:`MapRoundDetail`
        for the columns) and whose ``excluded`` tuple holds one
        :class:`ExcludedMapRow` per input row with null round-detail
        columns (empty when none are excluded). An empty input yields an
        empty ``records`` frame with the full schema and an empty
        ``excluded`` tuple.

    Raises:
        KeyError: If ``maps_df`` lacks a required column (see
            :func:`_require_input_columns`).
        ValueError: If a surviving row has a null score, fails the
            case-split reconciliation, or fails the opposing-side
            pairing assertion (propagated from
            :func:`validate_map_round_details`).
    """
    excluded = validate_map_round_details(maps_df)
    excluded_keys = {(row.match_id, row.map_index) for row in excluded}

    records: list[MapRoundDetail] = []
    for row in maps_df.itertuples(index=False):
        key = (row.match_id, row.map_index)
        if key in excluded_keys:
            continue
        team1, team2 = _derive_row_round_details(
            row.match_id,
            row.map_index,
            row.team1_score,
            row.team2_score,
            row.team1_atk_rounds,
            row.team1_def_rounds,
            row.team2_atk_rounds,
            row.team2_def_rounds,
        )
        records.extend((team1, team2))

    if records:
        frame = pd.DataFrame([_as_dict(record) for record in records])
    else:
        frame = pd.DataFrame(
            columns=[field.name for field in fields(MapRoundDetail)]
        )
    return MapRoundDetailResult(records=frame, excluded=excluded)


def _as_dict(record: MapRoundDetail) -> dict:
    """Convert one :class:`MapRoundDetail` to a column-ordered dict.

    The frame-materialisation helper: ``dict(record)``-style conversion
    with the dataclass field order preserved (Python 3.7+ dict order is
    insertion order, so ``vars``/``dataclasses.asdict`` would both work;
    this thin wrapper keeps the import surface minimal and documents
    the ordering contract the ``records`` frame's columns follow).

    Args:
        record: The :class:`MapRoundDetail` instance to convert.

    Returns:
        A ``dict`` mapping each field name to its value, in
        :class:`MapRoundDetail` declaration order (``match_id``,
        ``map_index``, ``side``, then the round fields, then
        ``signed_margin``).

    Raises:
        Nothing.
    """
    return {
        "match_id": record.match_id,
        "map_index": record.map_index,
        "side": record.side,
        "atk_rounds_won": record.atk_rounds_won,
        "atk_rounds_played": record.atk_rounds_played,
        "def_rounds_won": record.def_rounds_won,
        "def_rounds_played": record.def_rounds_played,
        "ot_rounds_won": record.ot_rounds_won,
        "ot_rounds_played": record.ot_rounds_played,
        "signed_margin": record.signed_margin,
    }

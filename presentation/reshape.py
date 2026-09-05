"""P3: artifact reshaping + team-name resolution as a pure-function library.

Turns the object-shaped dataclasses the ``predict`` closure returns
(:class:`drivers.predict.PredictionResult` and its sub-records) into
the post-reshaping, name-resolved wire dict that validates against
P1's :mod:`presentation.contract` TypedDicts and JSON Schema. This is
the export-side half of the presentation layer: :mod:`presentation.derived`
(P2) computes the §4.4 scalars, and this module (P3) performs the §4.5
reshaping plus the §4.2 ``team_id → display name`` resolution, which
must reach both the fixture level and every veto step inside every
ranked entry.

**Purity contract (recorded, do not silently change).** Every function
here is a pure function of its arguments: no file or network I/O, no
:class:`drivers.predict.Predictor` construction, no artifact reading or
writing, no CLI argument parsing, no logging. The one exception to the
"pure dataclass → dict" shape is :func:`build_team_name_map`, which
consumes an already-materialised ``pandas.DataFrame`` passed *in* by
the caller (the P5 export driver loads the table itself); it reads
nothing off disk. Like P2, a caller may import this module and call
any function with an empty filesystem present.

**Design decisions D1-D11 (recorded here, do not silently change).**

- **D1.** One file, ``presentation/reshape.py``, holding both halves of
  P3 (the §4.5 reshaping and the §4.2 name resolution) because they
  share one error type (:class:`UnknownTeamError`); splitting them
  would either duplicate that class or create an intra-package edge for
  no gain. Not re-exported from ``presentation/__init__.py`` — callers
  write ``from presentation.reshape import ...``.
- **D2.** I/O boundary: the module reads nothing off disk and
  constructs no :class:`drivers.predict.Predictor`. The matches table
  arrives already loaded in :func:`build_team_name_map`'s
  ``matches_df`` argument; every reshaping function takes a plain
  ``Mapping[str, str]`` of ``team_id → display name``.
- **D3.** Unknown team: raise, never log, never fabricate. A
  ``team_id`` absent from the map raises :class:`UnknownTeamError` (a
  :class:`ValueError` subclass, so generic handlers still catch it
  while the P5 export driver catches it specifically to log and exclude
  the fixture). The message names the unresolved id.
- **D4.** Contract amendment (the one change outside this module):
  :class:`presentation.contract.VetoAction`'s ``team``/``team_name``
  are widened to accept ``null`` (keys still required) because the
  decider step has no acting team. This module emits ``team: None`` /
  ``team_name: None`` for a decider rather than fabricating a name or
  an empty string.
- **D5.** ``rank`` is derived here, by position, over the listing *as
  given* (``enumerate(..., start=1)``) — never sorted, filtered or
  re-ordered. The caller already ranks descending by
  ``veto_probability``.
- **D6.** Hoisting is guarded, not assumed: ``outcome_order`` is
  hoisted once from ``result.series.outcome_order`` and every ranked
  entry's copy is *verified* equal, raising :class:`ValueError` naming
  the first mismatching rank.
- **D7.** A null top-level ``veto_sensitivity`` is an error, not a
  null field: :func:`reshape_overall_result` raises :class:`ValueError`
  (the schema requires a real object there; null is only ever legal
  inside ranked entries, where the field is omitted entirely).
- **D8.** Everything emitted is JSON-native: vectors are ``list``s,
  ``outcome_order``'s inner pairs are 2-``int`` lists, probabilities
  are ``float``s, counts are ``int``s, flags are ``bool``s — no
  ``tuple`` survives anywhere.
- **D9.** The veto-simulator action type is not imported here: it is
  only reachable through the predict entry point and is not in the
  permitted presentation surface. Veto actions are handled
  structurally by reusing :class:`presentation.derived.VetoActionLike`
  (P2's protocol), so ``presentation.reshape → presentation.derived``
  is the intended intra-package edge (P3 depends on P2) and the DAG
  stays acyclic.
- **D10.** This module's prose never spells an import of the lower
  packages; they are referred to descriptively (the boundary test
  raw-substring-scans every ``presentation/`` module).
- **D11.** Field ownership: P3 fills exactly seven ``Fixture`` keys —
  ``team_a``, ``team_b``, ``outcome_order``, ``scoreline_labels``,
  ``overall``, ``top_vetos``, ``coverage_mass`` — returned as a
  partial ``dict[str, object]``. ``map_leverage`` is P4's;
  ``match_id``/``event``/``scheduled_at``/``best_of``/
  ``best_of_int``/``bo5_unvalidated``/``narrative`` are P5's, as is
  every ``Artifact``-level key including ``intervals_present``.

**Module position (DAG).** This module imports only the permitted
``drivers.predict`` surface (:class:`drivers.predict.PerMapPrediction`,
:class:`drivers.predict.PredictionResult`,
:class:`drivers.predict.RankedVetoPrediction`,
:class:`drivers.predict.VetoSensitivity`), P1's
:mod:`presentation.contract`, P2's :mod:`presentation.derived`, and
``pandas`` (for the one function that touches a DataFrame). It never
reaches around a :class:`drivers.predict.PredictionResult` into the
feature or model layers to recompute a value the result already
carries, and nothing below it may import ``presentation/``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import pandas as pd

from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    VetoSensitivity,
)
from presentation import contract, derived


class UnknownTeamError(ValueError):
    """Raised when a ``team_id`` has no row in the materialised matches table.

    The one error type shared by :func:`build_team_name_map`'s
    consumers and every veto-step resolution: a fixture (or a ranked
    entry's veto step) whose ``team_id`` does not appear in the
    materialised matches table cannot be name-resolved, so the whole
    fixture must be excluded rather than shipped with a bare id or a
    fabricated name.

    Raised by:
        :func:`resolve_team` (both the fixture-level lookup and, via
        :func:`_reshape_veto_action`, every veto-step lookup) when the
        id is absent from the supplied name map.

    Caught by:
        The P5 export driver, which logs the fixture and the offending
        id and drops the fixture (§4.2's "excluded with a logged
        reason" policy) rather than crashing the export.

    Being a :class:`ValueError` subclass means generic error handlers
    still catch it, while P5 can catch it specifically.
    """


def build_team_name_map(matches_df: pd.DataFrame) -> dict[str, str]:
    """Scan both id/name column pairs into one ``team_id → name`` mapping.

    Walks the passed DataFrame row by row and reads the
    ``team1_id``/``team1_name`` and ``team2_id``/``team2_name`` column
    pairs, so a team that appears on either side of any historical
    match contributes its name. Rows whose id or name is missing (NaN /
    null) are skipped, so no id is ever mapped to the string ``"nan"``;
    ids and names are coerced to ``str``. When one id appears with
    several names over time (a rebranded org), the name from the most
    recent ``date`` wins, where ``date`` is the table's sortable
    ISO-8601 string column; ties resolve to the later row. When the
    ``date`` column is absent, or every value is missing, the rule
    degrades to last-row-wins (the later row in DataFrame order wins).

    Args:
        matches_df: The materialised matches table. It must carry the
            four columns ``team1_id``, ``team1_name``, ``team2_id``,
            ``team2_name``; ``date`` is read when present and used for
            the most-recent-name tie-break.

    Returns:
        A ``team_id → display name`` mapping (``str`` → ``str``),
        possibly empty for an empty frame.

    Raises:
        KeyError: If one of the four required id/name columns is
            absent from a non-empty frame — propagated unchanged from
            the per-row column access (an empty frame yields an empty
            map before any column is read).
    """
    name_by_id: dict[str, tuple[str, int, str]] = {}
    column_pairs = (("team1_id", "team1_name"), ("team2_id", "team2_name"))
    for position, (_, row) in enumerate(matches_df.iterrows()):
        raw_date = row.get("date")
        date_key = "" if pd.isna(raw_date) else str(raw_date)
        for id_column, name_column in column_pairs:
            raw_id = row[id_column]
            raw_name = row[name_column]
            if pd.isna(raw_id) or pd.isna(raw_name):
                continue
            team_id = str(raw_id)
            candidate = (date_key, position)
            existing = name_by_id.get(team_id)
            if existing is None or candidate > existing[:2]:
                name_by_id[team_id] = (date_key, position, str(raw_name))
    return {team_id: entry[2] for team_id, entry in name_by_id.items()}


def resolve_team(team_id: str, team_names: Mapping[str, str]) -> contract.Team:
    """Resolve one stable ``team_id`` to the wire ``Team`` dict.

    The single place the ``team_id → display name`` lookup happens, so
    the fixture-level and veto-level resolution cannot diverge.

    Args:
        team_id: The stable ``team_id`` string to resolve.
        team_names: The ``team_id → display name`` mapping (see
            :func:`build_team_name_map`).

    Returns:
        A ``Team`` dict ``{"id": team_id, "name": <display name>}``.

    Raises:
        UnknownTeamError: If ``team_id`` is absent from ``team_names``
            — raised (chained from the underlying ``KeyError``) with a
            message naming the unresolved id.
    """
    try:
        name = team_names[team_id]
    except KeyError as exc:
        raise UnknownTeamError(
            f"team_id {team_id!r} has no display name in the matches "
            "table"
        ) from exc
    return {"id": team_id, "name": name}


def _reshape_veto_action(
    action: derived.VetoActionLike,
    team_names: Mapping[str, str],
) -> contract.VetoAction:
    """Reshape one veto step to the wire ``VetoAction`` shape.

    Reads the four structural fields off the action (via P2's
    :class:`presentation.derived.VetoActionLike` protocol) and emits the
    five-field wire dict P1's schema types. A decider step — signalled
    by ``action.team is None`` — emits ``team: None`` and
    ``team_name: None`` (D4: no team acts, and the value must not be
    fabricated); every other step resolves its acting team through
    :func:`resolve_team`, so an unresolvable id raises the same
    :class:`UnknownTeamError` as the fixture-level lookup.

    Args:
        action: One veto step conforming to
            :class:`presentation.derived.VetoActionLike` (any object
            with ``step_index`` / ``team`` / ``action`` / ``map_name``
            attributes; ``team`` is ``None`` for a decider).
        team_names: The ``team_id → display name`` mapping (see
            :func:`build_team_name_map`).

    Returns:
        A ``VetoAction`` dict with keys ``step_index``, ``team``,
        ``team_name``, ``action``, ``map_name``; ``team``/``team_name``
        are ``None`` for a decider and the resolved id/name otherwise.

    Raises:
        UnknownTeamError: If a non-decider step's ``team`` id is
            absent from ``team_names`` — propagated unchanged from
            :func:`resolve_team`.
        AttributeError: If ``action`` lacks any of the four attributes
            the protocol declares — propagated unchanged from the
            attribute access (callers pass protocol-conforming
            objects, so this only surfaces a genuinely malformed
            input).
    """
    team = action.team
    if team is None:
        team_wire: str | None = None
        team_name_wire: str | None = None
    else:
        resolved = resolve_team(team, team_names)
        team_wire = resolved["id"]
        team_name_wire = resolved["name"]
    return {
        "step_index": action.step_index,
        "team": team_wire,
        "team_name": team_name_wire,
        "action": cast(Literal["ban", "pick", "decider"], action.action),
        "map_name": action.map_name,
    }


def _reshape_veto_actions(
    actions: Sequence[derived.VetoActionLike],
    team_names: Mapping[str, str],
) -> list[contract.VetoAction]:
    """Reshape a whole veto sequence to a list of wire ``VetoAction`` dicts.

    Shared by the greedy sequence (``result.predicted_veto``) and every
    ranked entry's copied sequence, so the two paths cannot diverge in
    how a step is reshaped.

    Args:
        actions: The veto steps in step order (any sequence of
            :class:`presentation.derived.VetoActionLike`-conforming
            steps).
        team_names: The ``team_id → display name`` mapping.

    Returns:
        A ``list`` of :func:`_reshape_veto_action` dicts, one per
        input step, in the same order.

    Raises:
        UnknownTeamError: Propagated unchanged from
            :func:`_reshape_veto_action` (which raises it out of
            :func:`resolve_team`) for any unresolvable non-decider
            step.
        AttributeError: Propagated unchanged from
            :func:`_reshape_veto_action` for a step missing one of the
            four protocol attributes.
    """
    return [_reshape_veto_action(action, team_names) for action in actions]


def _reshape_per_map(entry: PerMapPrediction) -> contract.PerMap:
    """Reshape one played map's prediction to the wire ``PerMap`` shape.

    Converts the four-vector tuples to length-4 ``list``s (D8),
    leaves null intervals as ``None`` (null means "not computed", never
    a fabricated zero vector), coerces the backing game count to
    ``int``, and takes the two derived collapses (``p_a_wins_map`` /
    ``p_overtime``) from P2 so the derivation has a single
    implementation rather than an inline re-derivation.

    Args:
        entry: The played map's
            :class:`drivers.predict.PerMapPrediction` record.

    Returns:
        A ``PerMap`` dict with keys ``map_name``, ``probabilities``,
        ``interval_low``, ``interval_high``, ``n_games_backing``,
        ``p_a_wins_map``, ``p_overtime``.

    Raises:
        ValueError: If ``entry.probabilities`` is not length 4 —
            propagated unchanged from
            :func:`presentation.derived.p_a_wins_map` (the first of
            the two collapse calls, each of which guards identically).
    """
    return {
        "map_name": entry.map_name,
        "probabilities": list(entry.probabilities),
        "interval_low": (
            None if entry.interval_low is None else list(entry.interval_low)
        ),
        "interval_high": (
            None if entry.interval_high is None else list(entry.interval_high)
        ),
        "n_games_backing": int(entry.n_games_backing),
        "p_a_wins_map": derived.p_a_wins_map(entry),
        "p_overtime": derived.p_overtime(entry),
    }


def _reshape_veto_sensitivity(
    sensitivity: VetoSensitivity,
) -> contract.VetoSensitivity:
    """Reshape the structural spread summary to the wire ``VetoSensitivity``.

    Converts each per-category vector to a ``list`` and the scalar
    headline to a ``float`` (D8); no computation is redone here — the
    bands and weights are computed upstream, and this module only
    re-packages the values the result already carries.

    Args:
        sensitivity: The
            :class:`drivers.predict.VetoSensitivity` record (never
            ``None`` at this call site — :func:`reshape_overall_result`
            guards first).

    Returns:
        A ``VetoSensitivity`` dict with the six §6 keys:
        ``unweighted_band_low``, ``unweighted_band_high``,
        ``band_widths``, ``mean_band_width``, ``weighted_mean``,
        ``weighted_variance``.

    Raises:
        Nothing — this is a pure field-by-field copy.
    """
    return {
        "unweighted_band_low": list(sensitivity.unweighted_band_low),
        "unweighted_band_high": list(sensitivity.unweighted_band_high),
        "band_widths": list(sensitivity.band_widths),
        "mean_band_width": float(sensitivity.mean_band_width),
        "weighted_mean": list(sensitivity.weighted_mean),
        "weighted_variance": list(sensitivity.weighted_variance),
    }


def _reshape_ranked_veto(
    entry: RankedVetoPrediction,
    rank: int,
    overall_p_a_wins_series: float,
    team_names: Mapping[str, str],
) -> contract.RankedVeto:
    """Reshape one ranked veto entry to the wire ``RankedVeto`` shape.

    Emits exactly the seven closed keys P1's schema declares for a
    ranked entry — ``rank`` (assigned by the caller, D5), the exact
    ``veto_probability``, the reshaped ``actions``, the played maps in
    play order, the exact-M30 ``series_probabilities``, the derived
    ``p_a_wins_series``, and the derived ``favorite_flips`` against the
    overall value. It emits **no** ``veto_sensitivity`` and **no**
    nested ``top_vetos`` key, and does not re-emit ``outcome_order`` —
    those are §4.5's structurally-constant fields, and the schema's
    ``additionalProperties: false`` turns any leak into a validation
    failure.

    Args:
        entry: The ranked entry
            (:class:`drivers.predict.RankedVetoPrediction`) whose
            ``result`` carries this specific veto's prediction.
        rank: The 1-based position in the top-N listing (the caller
            assigns it by position, never by re-sorting).
        overall_p_a_wins_series: The already-computed overall
            :func:`presentation.derived.p_a_wins_series` value, used as
            the flip baseline so it is evaluated once per fixture
            rather than once per entry.
        team_names: The ``team_id → display name`` mapping.

    Returns:
        A ``RankedVeto`` dict with exactly the seven keys ``rank``,
        ``veto_probability``, ``actions``, ``per_map``,
        ``series_probabilities``, ``p_a_wins_series``,
        ``favorite_flips``.

    Raises:
        ValueError: If ``entry.result.series`` has non-parallel
            ``probabilities``/``outcome_order`` vectors — propagated
            unchanged from :func:`presentation.derived.p_a_wins_series`.
        UnknownTeamError: Propagated unchanged from
            :func:`_reshape_veto_actions` for an unresolvable team id
            inside this entry's veto sequence.
        AttributeError: Propagated unchanged from
            :func:`_reshape_veto_actions` for a step missing one of the
            four protocol attributes.
    """
    inner = entry.result
    entry_p_a_wins_series = derived.p_a_wins_series(inner.series)
    return {
        "rank": rank,
        "veto_probability": float(entry.veto_probability),
        "actions": _reshape_veto_actions(inner.predicted_veto, team_names),
        "per_map": [_reshape_per_map(per_map) for per_map in inner.per_map],
        "series_probabilities": list(inner.series.probabilities),
        "p_a_wins_series": entry_p_a_wins_series,
        "favorite_flips": derived.favorite_flips(
            entry_p_a_wins_series, overall_p_a_wins_series
        ),
    }


def reshape_overall_result(
    result: PredictionResult,
    team_names: Mapping[str, str],
) -> contract.OverallResult:
    """Reshape the top-level result to the wire ``OverallResult`` shape.

    Fills the six §6 fields: the M31-sampled ``series_probabilities``,
    the derived ``p_a_wins_series``, the structural spread summary
    (guarded per D7 — a null top-level ``veto_sensitivity`` raises
    rather than shipping an invalid null), the greedy veto's played
    maps, the reshaped greedy veto steps, and the greedy veto's 1-based
    rank within the top-N listing (``None`` when it is not present,
    never a fabricated ``0``).

    Args:
        result: The top-level
            :class:`drivers.predict.PredictionResult` for the fixture.
        team_names: The ``team_id → display name`` mapping.

    Returns:
        An ``OverallResult`` dict with keys ``series_probabilities``,
        ``p_a_wins_series``, ``veto_sensitivity``, ``per_map``,
        ``greedy_veto``, ``greedy_rank``.

    Raises:
        ValueError: If ``result.veto_sensitivity`` is ``None`` (D7), or
            if ``result.series`` has non-parallel vectors — the latter
            propagated unchanged from
            :func:`presentation.derived.p_a_wins_series`.
        UnknownTeamError: Propagated unchanged from
            :func:`_reshape_veto_actions` for an unresolvable team id
            in the greedy sequence.
        AttributeError: Propagated unchanged from
            :func:`_reshape_veto_actions` or from
            :func:`presentation.derived.greedy_rank` (which projects
            each step's four protocol attributes) for a malformed
            action record.
    """
    if result.veto_sensitivity is None:
        raise ValueError(
            "top-level result has veto_sensitivity None; the schema "
            "requires a real spread summary there (null is only legal "
            "inside ranked entries, where the field is omitted)"
        )
    return {
        "series_probabilities": list(result.series.probabilities),
        "p_a_wins_series": derived.p_a_wins_series(result.series),
        "veto_sensitivity": _reshape_veto_sensitivity(result.veto_sensitivity),
        "per_map": [_reshape_per_map(per_map) for per_map in result.per_map],
        "greedy_veto": _reshape_veto_actions(
            result.predicted_veto, team_names
        ),
        "greedy_rank": derived.greedy_rank(
            result.predicted_veto, result.top_vetos
        ),
    }


def reshape_top_vetos(
    result: PredictionResult,
    overall_p_a_wins_series: float,
    team_names: Mapping[str, str],
) -> list[contract.RankedVeto]:
    """Reshape the top-N ranked veto listing to wire ``RankedVeto`` dicts.

    Walks ``result.top_vetos`` in the order given, assigning ranks by
    position (``enumerate(..., start=1)``, D5 — never sorted, filtered
    or re-ordered), and enforces D6: every entry's ``outcome_order``
    must equal the fixture's hoisted copy (``result.series.outcome_order``),
    otherwise the first mismatch raises naming its rank. Takes the
    already-computed overall ``p_a_wins_series`` so the flip baseline
    is evaluated once per fixture rather than once per entry.

    Args:
        result: The top-level
            :class:`drivers.predict.PredictionResult` carrying the
            ``top_vetos`` listing to reshape.
        overall_p_a_wins_series: The already-computed overall
            :func:`presentation.derived.p_a_wins_series` value (the
            flip baseline passed through to every entry).
        team_names: The ``team_id → display name`` mapping.

    Returns:
        A ``list`` of ``RankedVeto`` dicts in the same order as
        ``result.top_vetos``, with ``rank`` assigned 1..N by position.

    Raises:
        ValueError: If any ranked entry's ``series.outcome_order``
            differs from ``result.series.outcome_order`` (D6 — naming
            the first mismatching rank), or if an entry's series has
            non-parallel vectors (propagated unchanged from
            :func:`presentation.derived.p_a_wins_series` via
            :func:`_reshape_ranked_veto`).
        UnknownTeamError: Propagated unchanged from
            :func:`_reshape_ranked_veto` for an unresolvable team id
            inside an entry's veto sequence.
        AttributeError: Propagated unchanged from
            :func:`_reshape_ranked_veto` for a malformed action
            record.
    """
    hoisted = result.series.outcome_order
    ranked: list[contract.RankedVeto] = []
    for rank, entry in enumerate(result.top_vetos, start=1):
        if entry.result.series.outcome_order != hoisted:
            raise ValueError(
                f"ranked veto at rank {rank} has outcome_order "
                f"{entry.result.series.outcome_order!r} that differs "
                f"from the fixture's hoisted {hoisted!r}"
            )
        ranked.append(
            _reshape_ranked_veto(
                entry, rank, overall_p_a_wins_series, team_names
            )
        )
    return ranked


def reshape_fixture_core(
    result: PredictionResult,
    team_a_id: str,
    team_b_id: str,
    team_names: Mapping[str, str],
) -> dict[str, object]:
    """Reshape one fixture's prediction into the seven P3-owned wire keys.

    The one public entry point P5 calls per fixture. It computes the
    overall ``p_a_wins_series`` once (for the flip baseline), hoists
    ``outcome_order`` from ``result.series.outcome_order`` as a list of
    2-int lists (D8), derives ``scoreline_labels`` from that *same*
    hoisted copy via :func:`presentation.derived.scoreline_labels` (so
    labels and order cannot drift — the contract's parity check is the
    backstop, not the mechanism), resolves both fixture teams through
    :func:`resolve_team`, and assembles the seven D11 keys.

    The returned dict is **partial**: it is a ``dict[str, object]``,
    not a ``Fixture``. The deferred keys and their owning milestones
    are ``map_leverage`` (P4) and ``match_id`` / ``event`` /
    ``scheduled_at`` / ``best_of`` / ``best_of_int`` /
    ``bo5_unvalidated`` / ``narrative`` (P5), as is every
    ``Artifact``-level key including ``intervals_present`` (P5, which
    calls :func:`presentation.derived.intervals_present` itself). P5
    must supply a ``best_of``/``best_of_int`` that agrees with
    ``len(outcome_order) == best_of_int + 1``; the schema's
    Fixture-level ``allOf`` enforces that agreement at validation time.

    Args:
        result: The top-level
            :class:`drivers.predict.PredictionResult` for the fixture.
        team_a_id: Team A's stable ``team_id`` string.
        team_b_id: Team B's stable ``team_id`` string.
        team_names: The ``team_id → display name`` mapping (see
            :func:`build_team_name_map`).

    Returns:
        A partial ``dict[str, object]`` with exactly these seven keys:
        ``team_a``, ``team_b``, ``outcome_order``, ``scoreline_labels``,
        ``overall``, ``top_vetos``, ``coverage_mass`` (see D11 for
        what is deliberately absent).

    Raises:
        UnknownTeamError: If ``team_a_id`` or ``team_b_id`` (or any
            team id inside the reshaped veto sequences) is absent from
            ``team_names`` — propagated unchanged from
            :func:`resolve_team`.
        ValueError: If ``result.veto_sensitivity`` is ``None`` (D7),
            if a ranked entry's ``outcome_order`` differs from the
            hoisted copy (D6), or if ``result.series`` has non-parallel
            vectors — each propagated unchanged from the reshaping
            helper that raises it.
        AttributeError: Propagated unchanged from the reshaping helpers
            for a malformed action record.
    """
    overall_p_a_wins_series = derived.p_a_wins_series(result.series)
    outcome_order = [
        list(scoreline) for scoreline in result.series.outcome_order
    ]
    return {
        "team_a": resolve_team(team_a_id, team_names),
        "team_b": resolve_team(team_b_id, team_names),
        "outcome_order": outcome_order,
        "scoreline_labels": list(
            derived.scoreline_labels(result.series.outcome_order)
        ),
        "overall": reshape_overall_result(result, team_names),
        "top_vetos": reshape_top_vetos(
            result, overall_p_a_wins_series, team_names
        ),
        "coverage_mass": derived.coverage_mass(result.top_vetos),
    }

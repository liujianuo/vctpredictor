"""P4: the §5.4 per-map veto-leverage attribution as a pure function.

Computes, over the top-N ranked veto listing the caller passes in, the
§5.4 per-map attribution table: for every map in the veto pool,
``p_played`` (the probability-weighted share of the top-N mass in which
the map is played), ``p_a_given_played`` and ``p_a_given_not`` (the
probability-weighted conditional series-win probabilities on the
played / not-played branches), their difference ``swing``, whether that
swing ``flips_favorite``, and ``n_vetos_backing`` (how many top-N
entries played the map). The output is a ranked ``list`` of
:class:`presentation.contract.MapLeverage` wire dicts, ready for P5 to
assemble under the eighth ``Fixture`` key.

**Purity contract (recorded, do not silently change).** This module is a
pure function of its input listing: no file or network I/O, no
:class:`drivers.predict.Predictor` construction, no artifact reading or
writing, no CLI argument parsing, no logging. A caller may import it and
call :func:`compute_map_leverage` with an empty filesystem present.

**Design decisions D1-D12 (recorded here, do not silently change).**

- **D1.** One file, ``presentation/leverage.py``, one public function,
  :func:`compute_map_leverage`. Like :mod:`presentation.contract`,
  :mod:`presentation.derived` and :mod:`presentation.reshape`, it is
  **not** re-exported from ``presentation/__init__.py`` — callers write
  ``from presentation import leverage`` or
  ``from presentation.leverage import compute_map_leverage``.
- **D2.** Standalone, not folded into the reshaping module: P4 depends
  on P2 only, while P5 depends on both and does the assembly. Folding
  leverage into the reshaping module would reverse that stated edge and
  contradict the reshaping module's recorded "exactly seven keys"
  contract, so this module must not import the reshaping module and that
  module must not be edited.
- **D3.** The signature takes the narrowest natural input — the top-N
  listing itself, not the whole prediction result — so a caller cannot
  feed a mismatched overall/listing pair and the function is trivially
  testable:
  ``compute_map_leverage(top_vetos: Sequence[RankedVetoPrediction]) ->
  list[contract.MapLeverage]``.
- **D4.** Imports, and only these:
  :class:`drivers.predict.RankedVetoPrediction` (already in the
  permitted presentation ``drivers.predict`` surface),
  :mod:`presentation.contract` and :mod:`presentation.derived`. Veto
  steps are read **structurally** through
  :class:`presentation.derived.VetoActionLike` (P2's protocol, reused
  here — never redefined locally), exactly as the reshaping module does;
  the veto-simulator action type is deliberately not imported.
- **D5.** "Played" is the action-based rule read from the actions, not
  from ``per_map``: a map is played in an entry iff that entry's
  ``predicted_veto`` contains an action whose ``action`` is ``"pick"``
  or ``"decider"`` and whose ``map_name`` is the map in question.
- **D6.** The map universe is the union of *all* action ``map_name``s
  across all entries, bans included, so a map banned in every entry is
  enumerable and can be explicitly skipped as degenerate rather than
  silently missing.
- **D7.** Per-entry values are computed once, in a single pre-pass
  (:func:`_entry_summaries`): each entry's ``veto_probability``, its
  ``p_a_wins_series`` via P2, and its played-map ``frozenset``.
- **D8.** Degenerate-skip, evaluated before any division: a map is
  emitted only when it is played in neither none nor all of the listing
  and both branch masses are strictly positive. A listing whose total
  mass is ``<= 0.0`` (including an empty listing) returns ``[]``.
- **D9.** Ranking: descending by ``abs(swing) * p_played``, ties broken
  by ``map_name`` ascending, via a single ``sorted`` over the emitted
  records. The product is a sort key only, never a wire field.
- **D10.** ``n_vetos_backing`` is a plain ``int`` count and P4 never
  filters on it: a below-minimum-backing map is still emitted here with
  its true (low) count; the narrative threshold is a later milestone's
  config concern.
- **D11.** JSON-native output: a ``list`` of dicts whose values are
  ``str``, four ``float``s, a ``bool`` and an ``int`` — no tuples, no
  numpy scalars, no formatted strings; the numeric and boolean fields
  are coerced at construction so ``0.0`` never serialises as ``0``.
- **D12.** Prose discipline: this module's docstrings and comments never
  spell an import of the lower packages, never name the veto-simulator
  action class, and refer to those layers descriptively (the boundary
  test raw-substring-scans this file).
- **D13.** ``flips_favorite`` is the *within-map* flip: whether
  ``p_a_given_played`` and ``p_a_given_not`` sit on strictly opposite
  sides of ``0.5``, computed by reusing P2's
  :func:`presentation.derived.favorite_flips` predicate (A2, so the
  exactly-``0.5``-never-flips rule lives in one place). It does **not**
  compare either branch against the overall ``p_a_wins_series`` — that
  against-overall reading belongs to P2's per-entry ``favorite_flips``
  on the ranked listing — because D3's signature deliberately excludes
  the overall result and the two numbers this row owns are the only
  ones in scope.

**The four §5.4 correctness constraints and how each is honoured.**

- **(a) Top-N coverage only.** Every number here is conditional on the
  top-N listing it was given and is **not** a decomposition of the full
  veto-sequence space; callers must display these numbers alongside
  P2's coverage mass (already emitted as the adjacent ``Fixture`` field
  by P3, so it is deliberately not recomputed here).
- **(b) Scenario-shaped, not causal.** Docstrings and any narrative
  wording describe branches ("among the top-N entries in which this map
  is played"), never causation ("this map favours a side"). The reason
  is stated once here: the pool is exhaustive, so conditioning on a map
  conditions on a branch, and part of the attributed swing belongs to
  whatever the map displaced. Naming follows: the field is
  ``p_a_given_played``, not an "effect" field.
- **(c) Degenerate maps skipped.** Encoded as D8, before any division,
  with tests for both directions (played in all, played in none).
- **(d) Structural swing never merged with epistemic width.** This
  module never reads the per-map bootstrap interval endpoints or the
  per-map backing game count — veto swing is *structural* (the
  difference between two conditional probabilities) while the bootstrap
  interval is *epistemic* (how uncertain the estimate is); a prediction
  can carry a wide epistemic interval and zero structural swing, or the
  reverse, and the two are never summed or combined into one score.

**Module position (DAG).** This module imports only the permitted
``drivers.predict`` surface (:class:`drivers.predict.RankedVetoPrediction`)
plus P1's :mod:`presentation.contract` and P2's
:mod:`presentation.derived`; it imports nothing from the feature layer
or the model layer, and it does not import the reshaping module (D2).
The intra-package edges are ``leverage → contract`` and
``leverage → derived`` (P4 depends on P2).
"""

from __future__ import annotations

from collections.abc import Sequence

from drivers.predict import RankedVetoPrediction
from presentation import contract, derived


def _played_map_names(
    actions: Sequence[derived.VetoActionLike],
) -> frozenset[str]:
    """Collect the played-map names from one entry's veto action sequence.

    Applies D5's played rule structurally: a map is played iff some
    action in the sequence has ``action`` equal to ``"pick"`` or
    ``"decider"`` and names that map. Ban actions are excluded. The
    result is deduplicated to a :class:`frozenset` because the
    attribution only needs membership; order is irrelevant.

    Args:
        actions: The entry's full veto action sequence (every step,
            including bans and the forced decider), each step
            conforming to :class:`presentation.derived.VetoActionLike`.

    Returns:
        A :class:`frozenset` of the ``map_name`` values whose action is
        ``"pick"`` or ``"decider"``; possibly empty for a ban-only
        sequence.

    Raises:
        AttributeError: If an action record lacks the ``action`` or
            ``map_name`` attribute the protocol declares — propagated
            unchanged from the attribute access (callers pass
            protocol-conforming objects, so this only surfaces a
            genuinely malformed input).
    """
    return frozenset(
        action.map_name
        for action in actions
        if action.action in ("pick", "decider")
    )


def _all_map_names(
    top_vetos: Sequence[RankedVetoPrediction],
) -> list[str]:
    """Enumerate the map universe across every entry's full action sequence.

    Walks every entry's ``predicted_veto`` and collects each distinct
    ``map_name`` in first-appearance order, bans included. Bans are
    included deliberately (D6): a map banned in every entry never
    appears as a played map, but it must still be *enumerable* so the
    caller can reach it and then explicitly skip it as degenerate
    (played in none) rather than silently omitting it for an untested
    reason. The returned order only feeds the deterministic tie-break;
    the final output order is set by D9's sort.

    Args:
        top_vetos: The top-N ranked veto listing whose entries each
            carry a full action sequence in ``result.predicted_veto``.

    Returns:
        A ``list`` of every distinct ``map_name`` across all entries'
        actions, in first-appearance order (empty for an empty listing).

    Raises:
        AttributeError: If an entry lacks ``predicted_veto`` or an
            action record lacks ``map_name`` — propagated unchanged
            from the attribute access.
    """
    seen: set[str] = set()
    names: list[str] = []
    for entry in top_vetos:
        for action in entry.result.predicted_veto:
            map_name = action.map_name
            if map_name not in seen:
                seen.add(map_name)
                names.append(map_name)
    return names


def _entry_summaries(
    top_vetos: Sequence[RankedVetoPrediction],
) -> list[tuple[float, float, frozenset[str]]]:
    """Pre-compute each entry's three attribution inputs exactly once (D7).

    Walks the listing once and, per entry, records a
    ``(veto_probability, p_a_wins_series, played_maps)`` triple: the
    entry's exact joint probability coerced to ``float``, its
    series-win probability via P2's single implementation
    (:func:`presentation.derived.p_a_wins_series`, called exactly once
    per entry), and its played-map names via
    :func:`_played_map_names` (also once per entry). This pre-pass
    exists so the per-map loop never re-derives a value that is
    constant across maps for the same entry (which would be
    ``n_maps x n_entries`` redundant calls).

    Args:
        top_vetos: The top-N ranked veto listing.

    Returns:
        A ``list`` of ``(veto_probability, p_a_wins_series,
        played_maps)`` triples, one per entry, in listing order.

    Raises:
        ValueError: If an entry's series has non-parallel
            ``probabilities``/``outcome_order`` vectors — propagated
            unchanged from
            :func:`presentation.derived.p_a_wins_series`.
        AttributeError: If an entry's action record is malformed (lacks
            ``action``/``map_name``) or the entry lacks
            ``result.series``/``result.predicted_veto`` — propagated
            unchanged from :func:`_played_map_names` and the series
            read.
    """
    summaries: list[tuple[float, float, frozenset[str]]] = []
    for entry in top_vetos:
        inner = entry.result
        summaries.append(
            (
                float(entry.veto_probability),
                derived.p_a_wins_series(inner.series),
                _played_map_names(inner.predicted_veto),
            )
        )
    return summaries


def compute_map_leverage(
    top_vetos: Sequence[RankedVetoPrediction],
) -> list[contract.MapLeverage]:
    """Compute the §5.4 per-map veto-leverage attribution over a listing.

    The single public entry point. It builds the per-entry pre-pass
    summaries (:func:`_entry_summaries`) and the total top-N mass, then
    for each map in the universe (:func:`_all_map_names`, bans
    included) accumulates the probability-weighted played / not-played
    masses and conditional series-win sums in one pass over the
    summaries. Maps that are degenerate — played in none or all of the
    listing, or whose played/not-played branch mass is not strictly
    positive — are skipped *before* any division, so no zero-division
    or NaN can reach the output. Each surviving map becomes one
    :class:`presentation.contract.MapLeverage` wire dict; the returned
    list is ranked by D9.

    Args:
        top_vetos: The top-N ranked veto listing, each entry a
            :class:`drivers.predict.RankedVetoPrediction` carrying its
            exact ``veto_probability``, its full action sequence and
            its conditional series distribution.

    Returns:
        A ``list`` of :class:`presentation.contract.MapLeverage` dicts
        (possibly empty), one per non-degenerate map, sorted descending
        by ``abs(swing) * p_played`` with ``map_name`` ascending as the
        tie-break. For each row, ``flips_favorite`` is the *within-map*
        flip: it compares ``p_a_given_played`` against
        ``p_a_given_not`` on opposite sides of ``0.5`` (the
        against-overall comparison is P2's per-entry ``favorite_flips``,
        not this field).

    Raises:
        Nothing of its own: an empty listing, or a listing whose total
            mass is ``<= 0.0``, returns ``[]``; degenerate maps are
            skipped before any division, so no ``ZeroDivisionError`` is
            reachable.
        ValueError: If an entry's series has non-parallel vectors —
            propagated unchanged from :func:`_entry_summaries` (via
            :func:`presentation.derived.p_a_wins_series`).
        AttributeError: If an entry's action record is malformed —
            propagated unchanged from :func:`_entry_summaries` (via
            :func:`_played_map_names`) or :func:`_all_map_names`.
    """
    if not top_vetos:
        return []
    summaries = _entry_summaries(top_vetos)
    total_mass = sum(veto_probability for veto_probability, _, _ in summaries)
    if total_mass <= 0.0:
        return []
    n_entries = len(summaries)
    records: list[contract.MapLeverage] = []
    for map_name in _all_map_names(top_vetos):
        played_mass = 0.0
        not_played_mass = 0.0
        played_weighted_p_a = 0.0
        not_played_weighted_p_a = 0.0
        n_vetos_backing = 0
        for veto_probability, p_a_wins, played_maps in summaries:
            if map_name in played_maps:
                played_mass += veto_probability
                played_weighted_p_a += veto_probability * p_a_wins
                n_vetos_backing += 1
            else:
                not_played_mass += veto_probability
                not_played_weighted_p_a += veto_probability * p_a_wins
        if n_vetos_backing == 0 or n_vetos_backing == n_entries:
            continue
        if played_mass <= 0.0 or not_played_mass <= 0.0:
            continue
        p_played = played_mass / total_mass
        p_a_given_played = played_weighted_p_a / played_mass
        p_a_given_not = not_played_weighted_p_a / not_played_mass
        swing = p_a_given_played - p_a_given_not
        # A2: reuse P2's favorite_flips predicate so the exactly-0.5
        # rule lives in exactly one place. P2's parameter names
        # (entry_… / overall_…) describe its first call site, but the
        # predicate itself is the symmetric "strictly opposite sides of
        # 0.5"; here the two arguments are the played and not-played
        # branch means (A1: flips_favorite is within-map).
        flips_favorite = derived.favorite_flips(
            p_a_given_played, p_a_given_not
        )
        records.append(
            {
                "map_name": map_name,
                "p_played": float(p_played),
                "p_a_given_played": float(p_a_given_played),
                "p_a_given_not": float(p_a_given_not),
                "swing": float(swing),
                "flips_favorite": bool(flips_favorite),
                "n_vetos_backing": int(n_vetos_backing),
            }
        )
    records.sort(
        key=lambda record: (
            -abs(record["swing"]) * record["p_played"],
            record["map_name"],
        )
    )
    return records

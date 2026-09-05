"""P2: the §4.4 derived display quantities as a pure-function library.

Implements the eight §4.4 derivation-table rows that P2 owns —
:func:`p_a_wins_series`, :func:`scoreline_labels`, :func:`p_a_wins_map`,
:func:`p_overtime`, :func:`coverage_mass`, :func:`favorite_flips`,
:func:`greedy_rank` and :func:`intervals_present` — over the object-shaped
dataclasses the ``predict`` closure in ``drivers/predict.py`` already
returns, producing the derived values §6's wire types carry
(``OverallResult.p_a_wins_series`` / ``greedy_rank``,
``PerMap.p_a_wins_map`` / ``p_overtime``, ``RankedVeto.favorite_flips``,
``Fixture.scoreline_labels`` / ``coverage_mass`` and
``Artifact.intervals_present``).

**Purity contract (recorded, do not silently change).** Every function
here is a pure function of its arguments: no file or network I/O, no
:class:`drivers.predict.Predictor` construction, no artifact reading or
writing, no ``argparse``, no logging. The module reads nothing off
disk — a caller may import it and call any function with an empty
filesystem present. The four-way per-map collapse
(:func:`p_a_wins_map`, :func:`p_overtime`) and the series-win sum
(:func:`p_a_wins_series`) are each written in exactly one place in this
module, and no later milestone (and no frontend code) recomputes
either.

**Design decisions D1-D6 (recorded here, do not silently change).**

- **D1.** The module is ``presentation/derived.py`` (not
  ``derived_quantities.py``): the package name supplies the
  "presentation" half, so ``presentation.derived.p_a_wins_series``
  reads better. Like :mod:`presentation.contract` it is **not**
  re-exported from ``presentation/__init__.py`` — callers import
  ``from presentation.derived import ...``.
- **D2.** Inputs are the dataclasses :func:`drivers.predict.predict`
  returns: :class:`drivers.predict.SeriesPrediction`,
  :class:`drivers.predict.PerMapPrediction` and
  :class:`drivers.predict.RankedVetoPrediction` (all frozen, hence
  value-equal). No ``VetoSensitivity`` field feeds any §4.4 row, so
  it is not imported.
- **D3.** Dataclasses in, primitives out: each function takes the
  narrowest natural input that already carries everything the
  derivation needs (see the per-function docs), so no caller can pass
  a mismatched pair of vectors.
- **D4.** :func:`greedy_rank` compares veto sequences structurally via
  the local :class:`VetoActionLike` protocol and the private
  :func:`_veto_action_key` projection ``(step_index, team, action,
  map_name)`` — it does **not** import the action type, because the
  greedy path builds its actions in the greedy veto simulator
  (``models/greedy_veto_simulator.py``) while the enumerated path
  copies the same four fields via ``drivers/predict.py``'s action-copy
  helper, and both emit ``team=None`` for the decider step. A
  four-field projection is therefore equal across paths whenever the
  sequences are the same, with zero dependency-graph cost. The
  protocol declares its four members as read-only properties, so the
  frozen action records both paths build conform to it.
- **D5.** :func:`intervals_present` is an artifact-level boolean fed
  an *iterable* of per-map records: it returns ``True`` iff any entry
  carries a non-null ``interval_low`` **or** ``interval_high``. The
  signature does not assume it is fed only the fixture's overall
  ``per_map``; the caller may chain ranked entries' per-map records in
  too.
- **D6.** Semantics fixed here: :func:`p_a_wins_series` uses a plain
  ``sum()`` over a generator (not ``math.fsum``, not numpy) for exact
  parity with the CLI log line in ``drivers/predict.py``; scoreline
  labels are ``"<a_wins>-<b_wins>"`` (A's count first, matching
  ``outcome_order``'s own orientation); :func:`favorite_flips` is
  ``True`` iff the two values sit strictly on opposite sides of 0.5
  (exactly 0.5 on either side is not a flip); :func:`coverage_mass`
  over an empty listing is ``0.0``; :func:`greedy_rank` is 1-based and
  returns ``None`` (never ``0``) for "not in the listing".

**Module position (DAG).** This module imports only the permitted
``drivers.predict`` surface (:class:`drivers.predict.PerMapPrediction`,
:class:`drivers.predict.RankedVetoPrediction`,
:class:`drivers.predict.SeriesPrediction`) and nothing from ``features/``
or ``models/``. It never reaches around a
:class:`drivers.predict.PredictionResult` into ``features/`` or
``models/`` to recompute a value the result already carries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from drivers.predict import (
    PerMapPrediction,
    RankedVetoPrediction,
    SeriesPrediction,
)


class VetoActionLike(Protocol):
    """The structural shape :func:`greedy_rank` needs from one veto step.

    A local :class:`typing.Protocol` standing in for the action type
    the greedy simulator produces, so :func:`greedy_rank` can type its
    ``predicted_veto`` parameter and be statically checked without
    importing that type across a layer boundary. The four members
    mirror the field shape of every action record in the veto result
    (``step_index``, ``team``, ``action``, ``map_name``), and any
    object carrying exactly these four members — including the frozen
    dataclass instances both the greedy and enumerated paths build —
    satisfies the protocol structurally.

    The members are declared as read-only properties (not settable
    variable annotations), so a frozen action record — whose fields
    are read-only — conforms to the protocol; a settable declaration
    would reject every ``@dataclass(frozen=True)`` action type.
    """

    @property
    def step_index(self) -> int:
        """The 0-based position of this action in the veto sequence."""
        ...

    @property
    def team(self) -> str | None:
        """The acting team's stable ``team_id``.

        ``None`` for a decider action (the last remaining map is
        forced, not chosen, so no team is credited).
        """
        ...

    @property
    def action(self) -> str:
        """One of ``"ban"``, ``"pick"`` or ``"decider"``."""
        ...

    @property
    def map_name(self) -> str:
        """The chosen map's normalized name."""
        ...


def _veto_action_key(
    action: VetoActionLike,
) -> tuple[int, str | None, str, str]:
    """Project one veto action to the four fields :func:`greedy_rank`
    compares.

    Builds the value-only comparison key for a single veto step —
    ``(step_index, team, action, map_name)`` — so two sequences built
    as *distinct objects with equal field values* (the greedy sequence
    and an enumerated ranked entry's copied sequence) compare equal
    without relying on object identity. This deliberately drops any
    extra fields a richer per-step record might carry (for example the
    sampler-only ``probability`` field), mirroring the field-copy the
    enumerated path already performs.

    Args:
        action: One veto step satisfying :class:`VetoActionLike` (any
            object with the four ``step_index``/``team``/``action``/
            ``map_name`` attributes).

    Returns:
        The 4-tuple ``(step_index, team, action, map_name)`` read from
        ``action``.

    Raises:
        AttributeError: If ``action`` lacks any of the four attributes
            the protocol declares — propagated unchanged from the
            attribute access (callers pass protocol-conforming objects,
            so this only surfaces a genuinely malformed input).
    """
    return (action.step_index, action.team, action.action, action.map_name)


def p_a_wins_series(series: SeriesPrediction) -> float:
    """Sum the series-scoreline probabilities where side A wins the series.

    Adds every entry of ``series.probabilities`` whose paired
    ``series.outcome_order`` scoreline has ``a_wins > b_wins``. This is
    written to match the existing CLI log line in ``drivers/predict.py``
    exactly (a plain ``sum()`` over a generator, not ``math.fsum`` or
    numpy), so the exported number and the logged value cannot diverge
    in the last bit. It is the single place the series-win sum lives;
    both the overall result and every ranked entry expose a
    :class:`drivers.predict.SeriesPrediction`, so one function serves
    both §4.4 call sites.

    Args:
        series: The veto-marginalised series scoreline prediction whose
            ``probabilities`` and ``outcome_order`` are parallel
            vectors (length ``best_of + 1``).

    Returns:
        The summed probability of every scoreline where A wins more
        maps than B, a ``float`` in ``[0, 1]``.

    Raises:
        ValueError: If ``len(series.probabilities)`` differs from
            ``len(series.outcome_order)`` — the two vectors must be
            parallel, and ``zip``'s silent truncation would otherwise
            drop or mis-pair entries.
    """
    if len(series.probabilities) != len(series.outcome_order):
        raise ValueError(
            f"SeriesPrediction has {len(series.probabilities)} "
            f"probabilities but {len(series.outcome_order)} "
            "outcome_order entries; they must be parallel"
        )
    return float(
        sum(
            probability
            for probability, (a_wins, b_wins) in zip(
                series.probabilities, series.outcome_order
            )
            if a_wins > b_wins
        )
    )


def scoreline_labels(
    outcome_order: Sequence[tuple[int, int]],
) -> tuple[str, ...]:
    """Render each terminal scoreline as a positional ``"a-b"`` label.

    Produces a tuple parallel to ``outcome_order`` whose entry ``i`` is
    ``f"{a_wins}-{b_wins}"`` for the pair ``outcome_order[i]`` — always
    A's count first, matching ``outcome_order``'s own ``(a_wins,
    b_wins)`` orientation. No team identity is embedded in the label:
    §6 types ``scoreline_labels`` as a bare ``string[]`` parallel to
    the hoisted ``outcome_order`` (the frontend already knows which
    side is A), so the "which side is larger" attribution is a
    statement about how the label *reads*, not a request to embed a
    team name.

    Args:
        outcome_order: The ``best_of + 1`` terminal ``(a_wins,
            b_wins)`` scorelines in canonical order.

    Returns:
        A tuple of ``"a-b"`` strings, one per input pair and in the
        same order (length equal to ``len(outcome_order)``).

    Raises:
        ValueError: If an ``outcome_order`` entry is not exactly a
            2-element pair (e.g. ``(1, 0, 2)``) — raised by the tuple
            unpacking in the generator expression.
        TypeError: If an ``outcome_order`` entry is not iterable
            (e.g. a bare ``int``) — raised by the tuple unpacking.
    """
    return tuple(f"{a_wins}-{b_wins}" for a_wins, b_wins in outcome_order)


def p_a_wins_map(per_map: PerMapPrediction) -> float:
    """Collapse one played map's four-way vector to P(A wins the map).

    Sums ``probabilities[0] + probabilities[1]`` — the A-regulation and
    A-OT entries, in ``OUTCOME_LABELS`` order (A-regulation, A-OT,
    B-OT, B-regulation). This is the same collapse the ranked-entry
    builder performs internally when it reduces each played map to a
    per-map A-win probability (``drivers/predict.py``'s
    ``_build_ranked_veto_entries``) — a stated equivalence, not an
    import; the collapse is written here and only here.

    Args:
        per_map: The played map's prediction record whose
            ``probabilities`` is a length-4 tuple in ``OUTCOME_LABELS``
            order.

    Returns:
        ``per_map.probabilities[0] + per_map.probabilities[1]`` — the
        probability A wins this map in regulation or overtime, a
        ``float`` in ``[0, 1]``.

    Raises:
        ValueError: If ``len(per_map.probabilities) != 4`` — the
            dataclass annotation is not enforced at runtime, and a
            shorter or longer vector would silently read the wrong
            entries (or fail with an opaque ``IndexError``).
    """
    if len(per_map.probabilities) != 4:
        raise ValueError(
            f"PerMapPrediction.probabilities must be length 4, got "
            f"{len(per_map.probabilities)}"
        )
    return per_map.probabilities[0] + per_map.probabilities[1]


def p_overtime(per_map: PerMapPrediction) -> float:
    """Collapse one played map's four-way vector to P(the map goes to OT).

    Sums ``probabilities[1] + probabilities[2]`` — the A-OT and B-OT
    entries, in ``OUTCOME_LABELS`` order (A-regulation, A-OT, B-OT,
    B-regulation) — i.e. the probability the map is decided in
    overtime regardless of which side wins it. Like
    :func:`p_a_wins_map` this collapse is written once, here, and no
    caller recomputes it.

    Args:
        per_map: The played map's prediction record whose
            ``probabilities`` is a length-4 tuple in ``OUTCOME_LABELS``
            order.

    Returns:
        ``per_map.probabilities[1] + per_map.probabilities[2]`` — the
        probability the map reaches overtime, a ``float`` in
        ``[0, 1]``.

    Raises:
        ValueError: If ``len(per_map.probabilities) != 4`` — the
            dataclass annotation is not enforced at runtime, and a
            shorter or longer vector would silently read the wrong
            entries (or fail with an opaque ``IndexError``).
    """
    if len(per_map.probabilities) != 4:
        raise ValueError(
            f"PerMapPrediction.probabilities must be length 4, got "
            f"{len(per_map.probabilities)}"
        )
    return per_map.probabilities[1] + per_map.probabilities[2]


def coverage_mass(top_vetos: Sequence[RankedVetoPrediction]) -> float:
    """Sum the exact joint probabilities over the top-N veto listing.

    Adds each entry's ``veto_probability`` — the total probability
    mass the displayed top-N listing covers, which tells the frontend
    how representative the listing is of all possible veto outcomes.
    §5.4/§8 require the map-leverage numbers to always be reported
    alongside this mass (leverage is P4's milestone and stays out of
    scope here; the mass is the P2-owned quantity the leverage table
    is read against).

    Args:
        top_vetos: The ranked veto listing (each entry carries its
            exact joint ``veto_probability``).

    Returns:
        The summed ``veto_probability`` as a ``float``; ``0.0`` for an
        empty listing (a listing covering no outcomes has no mass, not
        an error).

    Raises:
        Nothing — an empty listing sums to ``0.0`` (the ``float``
            coercion turns ``sum()``'s int ``0`` into ``0.0``).
    """
    return float(sum(entry.veto_probability for entry in top_vetos))


def favorite_flips(
    entry_p_a_wins_series: float,
    overall_p_a_wins_series: float,
) -> bool:
    """Report whether one ranked entry flips the overall favourite.

    Returns ``True`` iff the two values sit strictly on opposite sides
    of 0.5 — one is ``< 0.5`` and the other is ``> 0.5`` — i.e. this
    specific veto sequence makes side A the favourite when the overall
    prediction makes side B the favourite (or vice versa). Exactly
    ``0.5`` on either side is **not** a flip: at exactly 0.5 there is
    no favourite to flip, so the result is ``False`` regardless of the
    other value. Both inputs are already-computed
    :func:`p_a_wins_series` outputs; taking floats rather than
    dataclasses keeps the overall sum evaluated in exactly one place.

    Args:
        entry_p_a_wins_series: The ranked entry's
            :func:`p_a_wins_series` value (one specific veto sequence's
            P(A wins series)).
        overall_p_a_wins_series: The overall result's
            :func:`p_a_wins_series` value (the veto-marginalised
            aggregate).

    Returns:
        ``True`` if the two values are strictly on opposite sides of
        0.5, ``False`` otherwise (same side, or either side exactly
        0.5).

    Raises:
        Nothing.
    """
    return (entry_p_a_wins_series < 0.5 and overall_p_a_wins_series > 0.5) or (
        entry_p_a_wins_series > 0.5 and overall_p_a_wins_series < 0.5
    )


def greedy_rank(
    predicted_veto: Sequence[VetoActionLike],
    top_vetos: Sequence[RankedVetoPrediction],
) -> int | None:
    """Find the greedy veto's 1-based position in the top-N listing.

    Computes the containment check §2's *alternative* display needs:
    the greedy ``predicted_veto`` and each ranked entry's own
    ``predicted_veto`` are computed independently (the greedy sequence
    is the M25 simulator's walk; the listing is the exact-enumeration
    ranking), so this function searches the listing for the entry whose
    sequence equals the greedy sequence — comparing the four-field
    :func:`_veto_action_key` projections (see D4) rather than object
    identity. The first match wins; the enumeration is over distinct
    permutations, so duplicates cannot occur.

    This function only answers "where does the greedy veto rank". It
    exists because §6's ``OverallResult.greedy_rank`` field is in the
    shipped contract, and §2's *preferred* resolution is to drop the
    greedy veto from the interface entirely — in which case this value
    is never displayed. Whether to populate the field and whether the
    UI shows it are P5/P13 decisions, not this function's.

    Args:
        predicted_veto: The greedy veto's full action sequence (any
            iterable of :class:`VetoActionLike`-conforming steps).
        top_vetos: The ranked veto listing to search, in descending
            ``veto_probability`` order (position 1 = highest).

    Returns:
        The 1-based position of the first listing entry whose
        ``predicted_veto`` equals ``predicted_veto`` under the
        four-field projection, or ``None`` when no entry matches
        (including an empty ``top_vetos`` listing) — ``None`` means
        "not in the listing", never a fabricated ``0``.

    Raises:
        AttributeError: If an entry's ``predicted_veto`` steps lack any
            of the four protocol attributes — propagated unchanged from
            :func:`_veto_action_key`.
    """
    greedy_key = tuple(_veto_action_key(action) for action in predicted_veto)
    for index, entry in enumerate(top_vetos, start=1):
        entry_key = tuple(
            _veto_action_key(action) for action in entry.result.predicted_veto
        )
        if entry_key == greedy_key:
            return index
    return None


def intervals_present(per_map_entries: Iterable[PerMapPrediction]) -> bool:
    """Report whether any per-map record carries an epistemic interval.

    Derives §6's artifact-level ``intervals_present`` boolean: whether
    the auto-load (D10) found bootstrap replicates, signalled by any
    non-null ``interval_low`` **or** ``interval_high`` on any entry.
    The frontend uses it to switch presentation wholesale rather than
    per-field. The signature accepts any iterable of per-map records;
    in practice passing only the fixture's overall ``per_map`` is
    sufficient because the ranked entries inherit the same closed-over
    bootstrap models (G5), so the two agree — but the signature does
    not assume that, and a caller may chain ranked entries' per-map
    records in too.

    Args:
        per_map_entries: Any iterable of
            :class:`drivers.predict.PerMapPrediction` records (e.g. the
            fixture's overall ``per_map``).

    Returns:
        ``True`` if at least one entry has a non-null ``interval_low``
        or a non-null ``interval_high``; ``False`` if every entry has
        both null, or the iterable is empty.

    Raises:
        AttributeError: If an entry lacks ``interval_low`` or
            ``interval_high`` — propagated unchanged from the
            attribute access (callers pass
            :class:`drivers.predict.PerMapPrediction` records, so this
            only surfaces a genuinely malformed input).
    """
    return any(
        entry.interval_low is not None or entry.interval_high is not None
        for entry in per_map_entries
    )

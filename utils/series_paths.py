"""Exact best-of-N series scoreline enumeration over per-map binary win
probabilities (roadmap M30).

Pure combinatorics over an abstract ``(a_wins, b_wins, map_index)``
state space: it takes plain per-map probabilities in and returns plain
terminal scoreline probabilities out, with zero project-specific
coupling (no team ids, no map names, no veto formats, no feature/model
imports). Per the roadmap line's own "Depends on: —" it is a genuine,
leaf-level utility standing entirely on its own; there is no
simulation at this level — enumeration is exact and cheap, and
simulating would only add variance.

Design decisions (recorded for later milestones; do not re-derive):

- **Module placement.** This lives in ``utils/`` because it is pure,
  dependency-free combinatorics. It is not a ``features/`` module (no
  trainable feature a model consumes), not a ``models/`` module (it
  predicts nothing from historical data), and not an ``evaluation/``
  module (it scores nothing against a label). It imports nothing from
  the rest of the repo and may only be consumed downward by
  ``features/``/``models/``/``evaluation/`` modules.

- **``best_of`` is a plain positive odd int** (the map count), *not*
  the ``"Bo3"``/``"Bo5"`` string convention used by ``scraper/``,
  ``models/greedy_veto_simulator.py`` and
  ``models/ancestral_veto_sampler.py`` (whose ``ACTION_SEQUENCES``
  dicts are keyed by those strings). This decoupling is deliberate:
  the roadmap asks for one shared implementation covering Bo3, Bo5
  *and* Bo7 even though no ``ACTION_SEQUENCES`` table in this repo has
  a ``"Bo7"`` entry today. Translating a veto module's ``"Bo3"``
  string into the integer ``3`` is left to whichever later milestone
  wires this module in, not solved here.

- **``map_win_probs`` must have exactly ``best_of`` entries.**
  ``map_win_probs[i]`` is P(side A wins the map played at
  series-map-index ``i``), conditional on the series reaching that
  map. Not every path plays all ``best_of`` maps (the series ends as
  soon as one side reaches ``series_win_threshold(best_of)`` wins),
  but the recursion needs a probability available for every map index
  a path could reach, so the caller supplies the full-length vector
  regardless; a length mismatch is a ``ValueError``, not silently
  truncated/padded. Each entry is an *independent* per-map binary
  probability, so — unlike ``utils.scoring``'s simplex check — there
  is no "sums to 1" validation on the input; the only "sum to 1"
  partners are ``p`` and ``1 - p`` per map, and that is enforced
  structurally by construction.

- **Recursion structure.** The core is a memoized recursion over
  ``(a_wins, b_wins, map_index)`` with the invariant
  ``map_index == a_wins + b_wins``: every map played advances the map
  index by one and increments exactly one of the win counters, so the
  invariant holds at every node and the third coordinate is always
  derivable from the other two. Base case: either side has reached
  ``series_win_threshold(best_of)`` — the series is over; return the
  leaf ``{(a_wins, b_wins): 1.0}``. Recursive case: ``p =
  map_win_probs[map_index]``; the result is the probability-weighted
  union of the ``(a_wins + 1, b_wins, map_index + 1)`` continuation
  weighted by ``p`` and the ``(a_wins, b_wins + 1, map_index + 1)``
  continuation weighted by ``1 - p``, merged by *summing* shared leaf
  keys (a terminal scoreline is reachable by more than one path in
  general — e.g. ``(2, 1)`` in a Bo3 via A-B-A and B-A-A — so
  overwriting would silently drop probability mass). Memoization is
  keyed on ``(a_wins, b_wins)`` (equivalently all three coordinates,
  per the invariant); it is a clarity choice matching the roadmap's
  state-machine framing, not a performance requirement (the state
  space is at most ``(threshold + 1) ** 2 <= 16`` pairs even for
  Bo7).

- **Local validation, no ``utils.scoring`` import.** Per the module-
  boundary rule (``tests/test_module_boundaries.py`` forbids lateral
  ``utils/ -> utils/`` imports other than the named ``asof.py`` ->
  ``table_io.py`` constant exception), this module implements its own
  small validators instead of importing ``utils.scoring``'s
  ``_validate_probs``. :func:`_validate_best_of` mirrors
  ``utils.scoring``'s ``operator.index`` coercion house style, and
  :func:`_validate_map_win_probs` performs the per-entry
  finite-``[0, 1]`` check described above.

- **Core vs convenience reshaping.** :func:`enumerate_series_paths`
  is the core enumeration; :func:`series_win_threshold`,
  :func:`series_outcome_order` and
  :func:`series_probabilities_in_order` are small pure derivations
  that reshape the same result without adding simulation (M31's job)
  or scoring-against-ground-truth (M33's job), so later milestones do
  not each reinvent an ordering convention. Nothing here aggregates
  across samples or evaluates against labels; when in doubt this
  module errs toward leaving aggregation to M31 and evaluation to M33.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence


def _validate_best_of(best_of: int) -> int:
    """Validate a best-of map count and coerce it to a plain int.

    Coerces ``best_of`` through ``operator.index`` (so plain ``int``
    and numpy integer scalars are both accepted, and non-integer types
    such as floats or strings are rejected uniformly — mirroring
    ``utils/scoring.py``'s ``_validate_true_index`` house style) and
    then requires it to be a positive odd number: a series must
    consist of an odd number of maps for a guaranteed strict-majority
    winner to exist (an even map count can end with the maps split
    evenly, so no series winner is guaranteed), and a zero or negative
    count is nonsensical.

    Args:
        best_of: The number of maps in the series. Any integer-like
            value (plain ``int`` or numpy integer scalar).

    Returns:
        The validated map count as a plain ``int``.

    Raises:
        ValueError: If ``best_of`` is not integer-like (e.g. a float
            like ``3.5`` or a string like ``"3"``), is ``0`` or
            negative, or is even.
    """
    try:
        value = operator.index(best_of)
    except TypeError as exc:
        raise ValueError(
            f"best_of must be a positive odd integer, got {best_of!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            "best_of must be at least 1 (a series needs at least one "
            f"map), got {value}"
        )
    if value % 2 == 0:
        raise ValueError(
            "best_of must be odd (an even map count cannot produce a "
            f"guaranteed series winner), got {value}"
        )
    return value


def series_win_threshold(best_of: int) -> int:
    """Return the number of map wins needed to win a best-of series.

    A best-of-``N`` series ends as soon as one side wins a strict
    majority of the ``N`` maps: ``(N + 1) // 2``. This count is the
    base-case predicate of :func:`enumerate_series_paths` and the
    number of winning scorelines each side can hold, which later
    milestones (M33) need to know how many scoreline categories a
    given ``best_of`` has.

    Args:
        best_of: The number of maps in the series (a positive odd
            integer or integer-like value; validated by
            :func:`_validate_best_of`).

    Returns:
        The majority win count ``(best_of + 1) // 2`` as an ``int``:
        ``1`` for Bo1, ``2`` for Bo3, ``3`` for Bo5, ``4`` for Bo7.

    Raises:
        ValueError: If ``best_of`` is invalid (see
            :func:`_validate_best_of`).
    """
    n = _validate_best_of(best_of)
    return (n + 1) // 2


def _validate_map_win_probs(
    map_win_probs: Sequence[float], expected_len: int
) -> list[float]:
    """Validate a per-map A-side win-probability vector.

    Materializes ``map_win_probs`` into a ``list`` of ``float``,
    checks it has exactly ``expected_len`` entries (the series'
    ``best_of`` map count — a shorter vector would leave a reachable
    map index without a probability, and a longer one would carry
    probability mass for maps that can never be played), and coerces
    every entry to a finite ``float`` in the closed interval
    ``[0.0, 1.0]``. Unlike ``utils.scoring``'s simplex validator there
    is deliberately no "sums to 1" check: each entry is an independent
    per-map binary probability, not a categorical distribution — the
    only "sum to 1" partners are ``p`` and ``1 - p`` per map, enforced
    structurally by construction in :func:`enumerate_series_paths`.
    This validator is local to this module: per the module-boundary
    rule, no lateral ``utils/ -> utils/`` import may be used to reuse
    ``utils.scoring``'s checker instead.

    Args:
        map_win_probs: An iterable of ``expected_len`` per-map
            probabilities, where entry ``i`` is P(side A wins the map
            played at series-map-index ``i``), conditional on the
            series reaching that map. Entries may be any real numbers
            (``int``/``float``/numpy scalars); they are coerced to
            ``float``.
        expected_len: The exact number of maps the series can span —
            the validated ``best_of`` map count.

    Returns:
        A new ``list`` of ``float`` holding the validated probabilities
        in the original order.

    Raises:
        ValueError: If ``map_win_probs`` has a length other than
            ``expected_len``, contains a non-numeric (uncoercible)
            entry, contains a non-finite (NaN/inf) entry, or contains
            an entry outside the closed interval ``[0.0, 1.0]``.
    """
    raw = list(map_win_probs)
    if len(raw) != expected_len:
        raise ValueError(
            f"map_win_probs must have exactly {expected_len} entries "
            f"(one per map in a best-of-{expected_len} series), got "
            f"{len(raw)}"
        )
    values: list[float] = []
    for i, entry in enumerate(raw):
        try:
            value = float(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"map_win_probs[{i}] must be a real number, got {entry!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"map_win_probs[{i}] must be finite, got {value!r}"
            )
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"map_win_probs[{i}] must be in [0.0, 1.0], got {value!r}"
            )
        values.append(value)
    return values


def enumerate_series_paths(
    map_win_probs: Sequence[float], best_of: int
) -> dict[tuple[int, int], float]:
    """Enumerate every terminal series scoreline with its exact probability.

    Walks the full series outcome tree via a memoized recursion over
    ``(a_wins, b_wins, map_index)`` with the invariant
    ``map_index == a_wins + b_wins`` (every map played advances the
    map index by one and increments exactly one win counter), starting
    at ``(0, 0, 0)``. At a node where neither side has reached
    ``series_win_threshold(best_of)``, the map at ``map_index`` is won
    by side A with probability ``map_win_probs[map_index]`` and by
    side B with ``1 - map_win_probs[map_index]``; the recursion
    returns the probability-weighted union of the two continuations,
    merging shared terminal scorelines by *summing* their
    contributions (a terminal ``(a_wins, b_wins)`` pair is reachable
    by more than one path in general, so overwriting would drop
    probability mass). The memo is a manual dict local to each call,
    keyed on ``(a_wins, b_wins)`` alone because ``map_index`` is
    pinned by the invariant; a manual dict (rather than
    ``functools.lru_cache``) lets the recursion close over the
    caller-supplied ``map_win_probs`` without hashing it. The state
    space is at most ``(threshold + 1) ** 2 <= 16`` pairs even for
    Bo7, so memoization matches the roadmap's state-machine framing
    rather than serving a performance need.

    Args:
        map_win_probs: An iterable of exactly ``best_of`` per-map
            probabilities, entry ``i`` being P(side A wins the map
            played at series-map-index ``i``), conditional on the
            series reaching that map. Validated by
            :func:`_validate_map_win_probs`.
        best_of: The number of maps in the series (a positive odd
            integer or integer-like value). Validated by
            :func:`series_win_threshold`.

    Returns:
        A ``dict`` mapping every terminal scoreline ``(a_wins,
        b_wins)`` to its exact probability as a ``float``. The values
        sum to 1 within floating-point error, and the dict holds
        ``best_of + 1`` leaves (``threshold`` A-win scorelines plus
        ``threshold`` B-win scorelines).

    Raises:
        ValueError: If ``best_of`` is invalid (see
            :func:`series_win_threshold`) or ``map_win_probs`` fails
            its validation (see :func:`_validate_map_win_probs`).
    """
    threshold = series_win_threshold(best_of)
    probs = _validate_map_win_probs(map_win_probs, best_of)

    # Manual dict memo local to this call, keyed on (a_wins, b_wins):
    # map_index is pinned to a_wins + b_wins by the invariant, so the
    # third coordinate never needs to participate in the key. A manual
    # dict rather than functools.lru_cache keeps map_win_probs, a
    # caller-supplied sequence, closable without hashing it.
    memo: dict[tuple[int, int], dict[tuple[int, int], float]] = {}

    def _recurse(
        a_wins: int, b_wins: int, map_index: int
    ) -> dict[tuple[int, int], float]:
        """Compute the scoreline distribution from one series state.

        Args:
            a_wins: Side A's current map-wins count.
            b_wins: Side B's current map-wins count.
            map_index: The series-map index of the next map to be
                played; the invariant ``map_index == a_wins + b_wins``
                holds at every call.

        Returns:
            The terminal-scoreline probability dict for the remainder
            of the series from this state (the memoized value for this
            ``(a_wins, b_wins)`` state).

        Raises:
            Nothing directly. The recursion stays within validated
            bounds: ``map_index`` never reaches ``len(probs)`` because
            a node with ``a_wins == b_wins == threshold`` would already
            have hit the base case, so every non-base node satisfies
            ``map_index <= 2 * threshold - 2 == best_of - 1``.
        """
        key = (a_wins, b_wins)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if a_wins == threshold or b_wins == threshold:
            result: dict[tuple[int, int], float] = {(a_wins, b_wins): 1.0}
        else:
            p = probs[map_index]
            result = {}
            for scoreline, weight in _recurse(
                a_wins + 1, b_wins, map_index + 1
            ).items():
                result[scoreline] = result.get(scoreline, 0.0) + weight * p
            for scoreline, weight in _recurse(
                a_wins, b_wins + 1, map_index + 1
            ).items():
                result[scoreline] = (
                    result.get(scoreline, 0.0) + weight * (1.0 - p)
                )
        memo[key] = result
        return result

    return _recurse(0, 0, 0)


def series_outcome_order(best_of: int) -> tuple[tuple[int, int], ...]:
    """Return every terminal scoreline in canonical ordinal order.

    A pure function of ``series_win_threshold(best_of)`` alone — no
    probabilities involved. The canonical order runs from side A's
    most dominant win to side B's most dominant win: first all of A's
    winning scorelines ``(threshold, 0) .. (threshold, threshold - 1)``
    in order of increasing B-side wins (most dominant first), then all
    of B's winning scorelines ``(threshold - 1, threshold) .. (0,
    threshold)`` in order of decreasing A-side wins (most dominant
    first). This mirrors the existing ``OUTCOME_LABELS`` ordinal
    vocabulary precedent (M9) at the series level, so M33's scoring
    later has one shared, agreed-upon category order instead of each
    caller re-deriving its own.

    Args:
        best_of: The number of maps in the series (a positive odd
            integer or integer-like value). Validated by
            :func:`series_win_threshold`.

    Returns:
        A tuple of ``best_of + 1`` terminal ``(a_wins, b_wins)``
        scorelines in canonical ordinal order: for Bo3
        ``((2, 0), (2, 1), (1, 2), (0, 2))``, for Bo5
        ``((3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3))``.

    Raises:
        ValueError: If ``best_of`` is invalid (see
            :func:`series_win_threshold`).
    """
    threshold = series_win_threshold(best_of)
    a_wins_order = tuple((threshold, b) for b in range(threshold))
    b_wins_order = tuple(
        (a, threshold) for a in range(threshold - 1, -1, -1)
    )
    return a_wins_order + b_wins_order


def series_probabilities_in_order(
    map_win_probs: Sequence[float], best_of: int
) -> list[float]:
    """Enumerate the series scoreline probabilities in canonical order.

    Convenience composition of :func:`enumerate_series_paths` and
    :func:`series_outcome_order`: enumerates the exact terminal
    scoreline distribution, then reindexes it by the canonical ordinal
    order, returning a plain probability vector (summing to 1 within
    float error) instead of a dict. It reshapes only this module's own
    output — it performs no simulation (M31's job) and no scoring
    against ground truth (M33's job) and, per the module-boundary
    rule, deliberately does not import ``utils.scoring`` even though a
    future caller may hand this vector straight to it.

    Args:
        map_win_probs: An iterable of exactly ``best_of`` per-map
            probabilities (validated as in
            :func:`enumerate_series_paths`).
        best_of: The number of maps in the series (a positive odd
            integer or integer-like value).

    Returns:
        A ``list`` of ``best_of + 1`` floats, entry ``j`` being the
        probability of the ``j``-th scoreline in
        :func:`series_outcome_order`'s canonical order. Sums to 1
        within floating-point error.

    Raises:
        ValueError: If ``best_of`` or ``map_win_probs`` fails
            validation (propagated from :func:`enumerate_series_paths`
            and :func:`series_outcome_order`).
    """
    paths = enumerate_series_paths(map_win_probs, best_of)
    return [paths[scoreline] for scoreline in series_outcome_order(best_of)]

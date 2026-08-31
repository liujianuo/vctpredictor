"""Rule-based greedy veto simulator (roadmap M25).

Deterministic, no-training simulation of a map veto: on each step the
acting team bans the remaining map where *its own* shrunk win rate is
lowest (its weakest map) or picks the remaining map where *its own*
shrunk win rate is highest (its strongest map). Win rates come from
:func:`features.map_win_rate.team_map_win_rate` (roadmap M13), the
Bayesian-shrunk per-map estimator on top of ``utils.asof``'s strict
``<`` history boundary — so the simulator is leakage-safe by
construction and performs no file I/O of its own.

This is the hard-argmax limit of roadmap M27 (a conditional-logit ban
model): a greedy rule the later learned model is compared against
rather than a trainable estimator. It is a ``models/``-layer module in
the same category as ``models/four_way_baseline.py`` (M18) — it
produces a deterministic prediction over existing features, so it may
depend only on ``features/`` and ``utils/`` (module boundary standard).

Design decisions (recorded here, do not re-derive in later milestones):

1. **Whose win rate scores a ban/pick: the acting team's own.** There
   is no head-to-head map win-rate estimator in this codebase (see
   ``models/four_way_baseline.py``'s "Normalization, not log5" note —
   ``features.map_win_rate.team_map_win_rate`` returns each team's
   *marginal* rate against the general field, not a pairwise rate).
   The greedy rule is therefore self-referential: on a team's turn it
   bans the remaining map where *its own* shrunk win rate is lowest
   (its weakest map) and picks the remaining map where *its own*
   shrunk win rate is highest (its strongest map). Each team's scores
   are computed independently from its own as-of history.
2. **Turn order alternates strictly by step index**, regardless of
   action type: step 0 = team_a, step 1 = team_b, step 2 = team_a, ...
   This matches every real Bo3/Bo5 veto note in the fixture suite
   (``tests/test_veto_fixtures.py``) and in
   ``data/v1/veto_actions.parquet`` except one Bo5 outlier (match
   660386, where one team bans twice in a row) — that outlier is a
   real-world deviation the simulator is not expected to reproduce; it
   is exactly the kind of gap M26 measures. ``team_a`` is whichever
   team the caller passes first — the simulator does not know or guess
   who really won a real coin toss.
3. **Per-format action sequences (7-map pool only)**, taken from the
   real fixture evidence (``tests/test_veto_fixtures.py`` module
   docstring, both the Bo3 fixtures and the two real Bo5 notes) plus
   the standard Bo1 convention:
   - Bo1: ``ban, ban, ban, ban, ban, ban, decider``
   - Bo3: ``ban, ban, pick, pick, ban, ban, decider``
   - Bo5: ``ban, ban, pick, pick, pick, pick, decider``
   All three are length 7, matching the current active era's 7-map
   pool (``config.json``'s ``2026-abyss`` era: Abyss, Ascent, Haven,
   Lotus, Split, Summit, Sunset). If the resolved map pool for the
   query date is not exactly 7 maps, :func:`simulate_veto` raises
   ``ValueError`` rather than silently guessing a generalised sequence
   for other pool sizes — out of scope for this S-sized milestone.
   Note for later reference: real vlr.gg pages render **no** veto note
   at all for Bo1 matches (confirmed by ``tests/test_veto_fixtures.py``
   / ``match_page_bo1.html``), so ``data/v1/veto_actions.parquet``
   will never contain a Bo1 sequence to diff against in M26. Bo1
   support is built anyway because the roadmap asks for it, but M26
   has no real Bo1 data to validate it against.
4. **Tie-breaking is deterministic via map name.** Ties are broken by
   ascending normalized map name (:func:`utils.config.normalize_map_name`)
   for both bans (lowest score, then alphabetically first) and picks
   (highest score, then alphabetically first) — one consistent
   secondary key, not two different ones.
5. **Leakage-safety is inherited, not reimplemented.** All win-rate
   lookups go through :func:`features.map_win_rate.team_map_win_rate`,
   which itself goes through ``utils.asof.maps_as_of`` (strict ``<``
   boundary). The simulator computes each team's score for every pool
   map exactly once, using the single as-of ``date`` passed in — not
   the map's real playback order, since a simulated veto happens
   before any of these maps are played and the as-of snapshot cannot
   change mid-simulation.
6. **Output is a new local type, not ``scraper.models.VetoAction``.**
   ``models/`` may depend on ``features/`` and ``utils/`` only (see
   the module-boundary standard and ``models/four_way_baseline.py``'s
   documented layering), so importing ``scraper.models.VetoAction``
   would invert that DAG. A new frozen dataclass
   :class:`SimulatedVetoAction` is defined locally with the same field
   shape (``step_index``, ``team``, ``action``, ``map_name``) but
   ``team`` holds a stable ``team_id`` (matching what
   ``features.map_win_rate`` and ``utils.asof`` consume), not a vlr.gg
   abbreviation like the real ``VetoAction.team``. Reconciling
   ``team_id`` against vlr abbreviations for the real-log diff is
   M26's problem, not M25's.
7. **Module placement: ``models/greedy_veto_simulator.py``.** It is a
   deterministic, no-training rule over existing features, the same
   category as ``models/four_way_baseline.py`` (M18) — not a
   ``features/`` estimator (it doesn't produce a reusable
   point-in-time feature) and not an ``evaluation/`` module (it
   produces predictions, not scores against a metric). Depends only on
   ``features.map_win_rate`` and ``utils.config``/``utils.asof``,
   preserving the DAG.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from features import map_win_rate
from utils import asof, config

# The three supported veto formats mapped to their 7-element action
# sequences. Data, not code — the evidence for these shapes lives in
# tests/test_veto_fixtures.py's module docstring (the Bo3 fixtures and
# the two real Bo5 notes all render 7 segments) plus the standard Bo1
# convention (vlr.gg renders no Bo1 veto note at all — see the module
# docstring's decision 3 — so the Bo1 shape is the documented standard,
# not transcribed from a fixture).
ACTION_SEQUENCES = {
    "Bo1": ("ban", "ban", "ban", "ban", "ban", "ban", "decider"),
    "Bo3": ("ban", "ban", "pick", "pick", "ban", "ban", "decider"),
    "Bo5": ("ban", "ban", "pick", "pick", "pick", "pick", "decider"),
}


@dataclass(frozen=True)
class SimulatedVetoAction:
    """One step of a simulated veto sequence.

    The simulator's output type, deliberately *not*
    ``scraper.models.VetoAction`` (importing it would invert the
    module DAG — see the module docstring's decision 6). It carries the
    same field shape as the real type (``step_index``, ``team``,
    ``action``, ``map_name``) with one semantic difference: ``team``
    holds a stable ``team_id`` (the id string ``features.map_win_rate``
    and ``utils.asof`` consume, e.g. ``"397"``), *not* a vlr.gg
    abbreviation like the real ``VetoAction.team`` (e.g. ``"FNC"``).
    Reconciling the two vocabularies for a real-log diff is M26's job.

    Attributes:
        step_index: 0-based position of this action in the veto
            sequence (0 for the first ban, 1 for the second, ...),
            matching the real ``VetoAction.step_index``.
        team: The acting team's stable ``team_id`` (``team_a_id`` on
            even steps, ``team_b_id`` on odd steps), or ``None`` for a
            decider action — the last remaining map is forced, not
            chosen, so no team is credited with the decision.
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        map_name: The chosen map's normalized name (e.g. ``"Haven"``);
            for a decider, the one map left after all bans/picks.
    """

    step_index: int
    team: str | None
    action: str
    map_name: str

    def to_dict(self) -> dict[str, object]:
        """Serialize this simulated veto action to a JSON-compatible dict.

        Returns:
            A dict with keys ``"step_index"``, ``"team"``, ``"action"``
            and ``"map_name"``, suitable for ``json.dumps`` and for
            round-tripping (the dict shape mirrors
            ``scraper.models.VetoAction.to_dict`` so M26's diff step
            sees the same key names on both sides).
        """
        return {
            "step_index": self.step_index,
            "team": self.team,
            "action": self.action,
            "map_name": self.map_name,
        }


def team_map_scores(
    team_id: str,
    pool: Sequence[str],
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k,
) -> dict[str, float]:
    """Score every map in a pool for one team at one as-of cutoff.

    The shared per-team per-map score computation the greedy rule is
    built on, extracted out of :func:`simulate_veto` so the M26 veto
    evaluation harness can build the same softmax-over-score
    distribution without duplicating the dict comprehension. For every
    map in ``pool`` it returns the mean of
    :func:`features.map_win_rate.team_map_win_rate` for ``team_id`` at
    the single as-of ``date`` (decision 5 of the module docstring: the
    as-of snapshot cannot change mid-simulation, so a veto evaluates
    every pool map at one cutoff, not at the map's hypothetical play
    order). Each pool entry is normalized via
    :func:`utils.config.normalize_map_name` before the lookup and the
    dict key uses the normalized name, so case/whitespace never break
    a match and the dict is keyed the same way the ``remaining`` set
    of :func:`simulate_veto` is.

    Args:
        team_id: The stable id of the team to score for (the same id
            ``features.map_win_rate.team_map_win_rate`` and
            ``utils.asof`` consume).
        pool: An iterable of map names to score; every entry is
            normalized via :func:`utils.config.normalize_map_name`
            (the caller's already-normalized pool passes through
            unchanged, since normalization is idempotent).
        date: The single as-of cutoff used for every win-rate lookup
            (e.g. the simulated match's own ISO-8601 timestamp). Maps
            dated ``>=`` this are excluded (strict ``<``), so no score
            ever sees the match being vetoed or anything later.
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the
            columns ``utils.asof.maps_as_of`` already requires).
        k: The shrinkage strength (effective prior sample size) passed
            to :func:`features.map_win_rate.team_map_win_rate`; must
            be a positive finite real number.

    Returns:
        A ``{normalized_map_name: mean}`` dict with one entry per pool
        map, in ``pool``'s iteration order, where ``mean`` is the
        shrunk win-rate posterior mean of
        :func:`features.map_win_rate.team_map_win_rate` for that map.
        Two distinct pool entries that normalize to the same name
        collapse into one dict key (the caller's already-validated
        pools never do this; duplicate detection lives in
        :func:`simulate_veto`).

    Raises:
        ValueError: If an as-of map has a null/NaN score or tied
            scores, or ``k`` is not a positive finite real, or
            ``date`` is null/unparseable/timezone-aware (all
            propagated from
            :func:`features.map_win_rate.team_map_win_rate`).
        KeyError: If either table lacks a required column (propagated
            from :func:`features.map_win_rate.team_map_win_rate` /
            :func:`utils.asof.maps_as_of`).
        TypeError: If ``date`` is list-like rather than a single
            scalar timestamp (propagated from
            :func:`utils.asof.parse_query_date`).
        ConfigError: If a pool entry or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    return {
        config.normalize_map_name(name): map_win_rate.team_map_win_rate(
            team_id, name, date, matches_df, maps_df, k
        ).mean
        for name in pool
    }


def simulate_veto(
    team_a_id: str,
    team_b_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k,
    map_pool=None,
) -> list[SimulatedVetoAction]:
    """Simulate a full greedy veto for one match.

    Runs one of the fixed :data:`ACTION_SEQUENCES` action lists over
    the resolved 7-map pool. Each team's score for every pool map is
    computed exactly once via :func:`team_map_scores` — the mean of
    :func:`features.map_win_rate.team_map_win_rate` at the single
    as-of ``date`` passed in (decision 5: the as-of snapshot cannot
    change mid-simulation) — giving two
    ``{normalized_map_name: mean}`` dicts. The walk then alternates turns strictly by step-index parity
    (even -> ``team_a_id``, odd -> ``team_b_id``; decision 2), removes
    each chosen map from the ``remaining`` set, and picks:

    - **ban**: the remaining map with the lowest acting-team mean,
      ties broken by ascending map name (decision 4);
    - **pick**: the remaining map with the highest acting-team mean,
      ties broken by ascending map name (decision 4);
    - **decider**: the single remaining map, attributed to no team
      (``team=None``).

    The greedy rule is self-referential (decision 1): each team scores
    by its *own* shrunk win rate, computed independently from its own
    as-of history, because there is no head-to-head estimator in this
    codebase. The map pool is either the caller's (each entry
    normalized via :func:`utils.config.normalize_map_name`, so
    case/whitespace never break a match) or, when ``map_pool=None``,
    the active era's pool resolved from ``config.json`` for the query
    date's calendar date (:meth:`utils.config.Config.era_as_of`). All
    three supported sequences are length 7, so any other pool size is
    rejected with ``ValueError`` rather than silently guessed
    (decision 3's fail-loud clause).

    Args:
        team_a_id: The stable id of the team acting on even steps
            ("team A"); whichever team the caller passes first — the
            simulator does not know or guess who really won a coin
            toss.
        team_b_id: The stable id of the team acting on odd steps
            ("team B").
        best_of: The veto format; must be a key of
            :data:`ACTION_SEQUENCES` (``"Bo1"``, ``"Bo3"`` or
            ``"Bo5"``).
        date: The single as-of cutoff used for every win-rate lookup
            (e.g. the simulated match's own ISO-8601 timestamp).
            Maps dated ``>=`` this are excluded (strict ``<``), so the
            simulation never sees the match being vetoed or anything
            later. When ``map_pool`` is ``None``, the cutoff's UTC
            calendar date also selects the era whose pool is used.
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the columns
            ``utils.asof.maps_as_of`` already requires).
        k: The shrinkage strength (effective prior sample size) passed
            to :func:`features.map_win_rate.team_map_win_rate` for
            both teams' estimates; must be a positive finite real
            number.
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves the pool from
            ``config.json`` via
            :meth:`utils.config.Config.era_as_of` on ``date``'s
            calendar date. Every entry (caller-supplied or from the
            config) is normalized via
            :func:`utils.config.normalize_map_name`; duplicates after
            normalization raise ``ValueError`` (they would collapse in
            the ``remaining`` set and mis-sync the walk).

    Returns:
        A list of exactly ``len(ACTION_SEQUENCES[best_of])``
        :class:`SimulatedVetoAction` objects in step order (one per
        action), each with the acting team's id (or ``None`` for the
        final decider) and the chosen map's normalized name. The list
        is deterministic for identical inputs.

    Raises:
        ValueError: If ``best_of`` is not a key of
            :data:`ACTION_SEQUENCES` (the message names the invalid
            value); if ``len(map_pool) != len(ACTION_SEQUENCES[best_of])``
            after normalization; if the normalized pool contains
            duplicates; if more or fewer than one map remains at the
            decider step (an internal pool/sequence desync, raised as
            ``ValueError`` not ``AssertionError`` per the plan); if
            ``date`` is null/unparseable/timezone-aware (from
            :func:`utils.asof.parse_query_date`, propagated through
            both era resolution and every win-rate lookup); or if an
            as-of map has a null/NaN score or tied scores, or ``k`` is
            not a positive finite real (all propagated from
            :func:`features.map_win_rate.team_map_win_rate`).
        ConfigError: If ``map_pool`` is ``None`` and ``config.json``
            cannot be loaded/validated, if no configured era covers
            ``date``'s calendar date, or if a map name (caller-
            supplied pool entry or any as-of map row) is not a string
            (from :func:`utils.config.load_config` /
            :meth:`utils.config.Config.era_as_of` /
            :func:`utils.config.normalize_map_name`, propagated).
        KeyError: If either table lacks a required column (propagated
            from :func:`features.map_win_rate.team_map_win_rate` /
            :func:`utils.asof.maps_as_of`).
        TypeError: If ``date`` is list-like rather than a single
            scalar timestamp (propagated from
            :func:`utils.asof.parse_query_date`).
    """
    sequence = ACTION_SEQUENCES.get(best_of)
    if sequence is None:
        raise ValueError(
            f"best_of {best_of!r} is not a supported veto format; "
            f"expected one of {sorted(ACTION_SEQUENCES)}"
        )

    if map_pool is None:
        # Resolve the active era's pool for the query date's calendar
        # date. asof.parse_query_date validates the cutoff exactly as
        # the win-rate lookups below will, so a bad date fails loudly
        # here with the same error the as-of layer would raise.
        query_ts = asof.parse_query_date(date)
        pool = list(config.load_config().era_as_of(query_ts.date()).map_pool)
    else:
        pool = [config.normalize_map_name(name) for name in map_pool]

    if len(pool) != len(sequence):
        raise ValueError(
            f"map_pool has {len(pool)} map(s) but a {best_of} veto "
            f"needs {len(sequence)}; the simulator only supports the "
            "7-map-pool sequences in ACTION_SEQUENCES"
        )
    if len(set(pool)) != len(pool):
        duplicates = sorted(
            name for name in set(pool) if pool.count(name) > 1
        )
        raise ValueError(
            f"map_pool contains duplicate map(s) after normalization: "
            f"{duplicates}; duplicates would collapse in the remaining "
            "set and desync the veto walk"
        )

    # Each team's score for every pool map is computed exactly once via
    # the shared team_map_scores helper (extracted in M26 so the veto
    # evaluation harness reuses the same computation rather than
    # duplicating it).
    scores_a = team_map_scores(team_a_id, pool, date, matches_df, maps_df, k)
    scores_b = team_map_scores(team_b_id, pool, date, matches_df, maps_df, k)

    remaining = set(pool)
    actions: list[SimulatedVetoAction] = []
    for step_index, action in enumerate(sequence):
        if action == "decider":
            if len(remaining) != 1:
                raise ValueError(
                    f"internal error: {len(remaining)} map(s) remained "
                    "at the decider step; expected exactly 1 (the "
                    "action sequence and map_pool are out of sync)"
                )
            actions.append(
                SimulatedVetoAction(
                    step_index=step_index,
                    team=None,
                    action="decider",
                    map_name=next(iter(remaining)),
                )
            )
            continue
        if action not in ("ban", "pick"):
            # ACTION_SEQUENCES is module-owned data, so any other value
            # is an internal bug, not bad input — fail loudly.
            raise ValueError(
                f"internal error: unknown veto action {action!r} in "
                f"ACTION_SEQUENCES[{best_of!r}]"
            )

        if step_index % 2 == 0:
            acting_id = team_a_id
            scores = scores_a
        else:
            acting_id = team_b_id
            scores = scores_b

        if action == "ban":
            # Lowest mean; ties broken by ascending name. min over
            # (mean, name) tuples does exactly that.
            chosen = min(remaining, key=lambda name: (scores[name], name))
        else:
            # Highest mean; ties still broken by ascending name
            # (decision 4). min over (-mean, name) picks the most
            # negative mean first (= the highest), then the smallest
            # name among equals.
            chosen = min(remaining, key=lambda name: (-scores[name], name))

        actions.append(
            SimulatedVetoAction(
                step_index=step_index,
                team=acting_id,
                action=action,
                map_name=chosen,
            )
        )
        remaining.remove(chosen)

    return actions

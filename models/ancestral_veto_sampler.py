"""Ancestral veto sampler (roadmap M29).

Samples ``n_samples`` full veto sequences forward through per-step
probability distributions, each sampled sequence carrying its own
``sequence_probability`` (the product of every non-decider step's
drawn probability), ready for Stage 3 (M31) marginalisation — "average
across M29's samples weighted by sequence probability". It is the
stochastic sibling of ``models/greedy_veto_simulator.py``'s
:func:`simulate_veto`: structurally the same walk over the same fixed
action sequences and the same 7-map pool, but at each ``ban``/``pick``
step it draws from a caller-supplied per-step probability distribution
instead of taking a deterministic argmin/argmax.

Design decisions (recorded here, do not re-derive in later
milestones):

1. **"Per-step distribution" means pluggable, not pinned to one
   model.** The sampler consumes *any* per-step predictor matching the
   shape ``evaluation/veto_evaluation.py``'s M26 harness already
   established — a callable ``(acting_team_id, action, remaining_maps,
   date, matches_df, maps_df) -> Sequence[float]`` returning a
   probability distribution aligned 1:1 to the alphabetically-sorted
   ``remaining_maps`` — rather than being hard-wired to the M25 greedy
   rule, the frequency baseline, or the M27/M28 trained models. At v1
   scale the natural Stage-3 choice is the trained M27 ban model plus
   the trained M28 pick model, but the sampler itself depends on none
   of them (decision 6's module-boundary constraint); the caller wires
   in whichever concrete predictor(s) it wants — the M25 greedy arm,
   the frequency baseline, or the M27/M28 closures, interchangeably,
   including in tests. Because the M27 and M28 predictors are each
   *action-restricted* (the ban model's wrapped predictor raises on
   ``action != "ban"``, and symmetrically for pick), one callable
   cannot serve both actions for the trained-model case; the sampler's
   parameter is therefore ``predictor_fn_by_action: dict[str,
   VetoStepPredictorFn]``, keyed by ``"ban"`` / ``"pick"`` (never
   ``"decider"`` — decision 2), not a single callable. A caller using
   an action-agnostic arm (the M25 greedy step model, the frequency
   baseline) simply supplies the same callable under both keys; this
   is the explicit wiring convention — no auto-detection of whether a
   predictor supports both actions.
2. **Sampling mechanics: a stochastic sibling of
   ``models/greedy_veto_simulator.py``'s ``simulate_veto``, not a
   teacher-forced replay.** This is a *generative forward* simulator
   for a match that has not been played (there is no real historical
   sequence to replay), structurally identical to ``simulate_veto``'s
   walk: resolve the action sequence for ``best_of`` from
   :data:`ACTION_SEQUENCES`, resolve the 7-map pool (caller-supplied
   or ``config.Config.era_as_of``), alternate turns strictly by
   step-index parity (step 0 acts as ``team_a_id``, step 1 as
   ``team_b_id``, ... — identical to ``simulate_veto``'s decision 2,
   inherited unchanged, not re-litigated here), and walk the sequence
   removing each step's chosen map from ``remaining``. The one
   structural difference: at each ``ban``/``pick`` step, instead of a
   deterministic argmin/argmax over an own-computed score dict, the
   sampler calls ``predictor_fn_by_action[action](acting_team_id,
   action, sorted(remaining), date, matches_df, maps_df)`` to get a
   probability distribution over the sorted candidate list, then draws
   one index from it via the caller-supplied ``numpy.random.Generator``
   (decision 4). The **decider step is forced**, exactly as in
   ``simulate_veto``: at the decider step exactly one map remains, it
   is emitted deterministically (``team=None``, ``action="decider"``,
   ``probability=1.0``), and **no predictor is consulted and no random
   draw happens** — ``predictor_fn_by_action`` needs no ``"decider"``
   key. The forced step's ``probability=1.0`` is a documentation
   convention for a fixed-type-``float`` field, not a modeled belief;
   because it is excluded from ``sequence_probability``'s product
   (decision 3) it would be numerically inert even if included, but
   the exclusion is stated as an explicit convention (mirroring the
   M27/M28 "decider excluded from the likelihood" precedent) rather
   than left to be inferred from the multiply-by-one coincidence.
3. **Output shape: raw per-sequence samples, each self-contained.**
   Two new frozen dataclasses (module-local, not reused from
   ``models/greedy_veto_simulator.py``'s ``SimulatedVetoAction``,
   which has no probability field and is not extended in place):
   :class:`SampledVetoAction` (``step_index``, ``team``, ``action``,
   ``map_name``, ``probability`` — the drawn map's probability under
   the step's distribution; ``1.0`` for a decider per decision 2, with
   a ``to_dict()`` mirroring ``SimulatedVetoAction.to_dict()``'s shape
   plus the ``probability`` key) and :class:`SampledVetoSequence`
   (``team_a_id``, ``team_b_id``, ``best_of``, ``date``, ``actions``,
   ``sequence_probability`` — the product of every **non-decider**
   step's ``probability``, per decision 2's exclusion convention).
   :func:`sample_veto_sequences` returns ``list[SampledVetoSequence]``
   of length ``n_samples`` — **raw samples, not a pre-aggregated
   distribution**. Grouping/averaging samples by ``sequence_probability``
   (or by the resulting map sequence) is explicitly Stage 3's job
   (M31's "average across M29's samples weighted by sequence
   probability"), not built here; this task ships the ingredient, not
   the aggregation.
4. **Determinism / seeding: an explicit, caller-supplied
   ``numpy.random.Generator``, never a hidden global RNG or a bare int
   seed.** The function signature takes ``rng: numpy.random.Generator``
   (the caller constructs it via ``numpy.random.default_rng(seed)``);
   no default is provided (an omitted ``rng`` is a ``TypeError`` from
   Python's own argument binding, not a silently-reused global state) —
   this matches the codebase's "no hidden randomness" leakage-safety
   ethos and makes tests trivially reproducible (fixed seed ->
   identical output) without resetting any global numpy random state
   between tests. The ``n_samples`` independent walks consume the
   *same* ``rng`` object sequentially (draw order: sample 0's steps in
   order, then sample 1's, ...), so two calls with the same seed and
   same inputs are byte-identical, and two different seeds are
   expected (not merely permitted) to diverge whenever any step's
   distribution has more than one map with nonzero probability mass.
5. **Predictor-output validation and defensive renormalization.** A
   predictor is caller-supplied and must not be trusted blindly:
   before drawing, the sampler validates the returned vector has
   exactly ``len(remaining)`` entries, every entry is finite and
   non-negative, and the entries sum to ``1.0`` within a documented
   absolute tolerance (:data:`_PROB_SUM_TOLERANCE = 1e-6`, a new
   module constant) — violations raise ``ValueError`` naming the
   offending step/arm. The validated vector is then renormalized
   (``probs / probs.sum()``) before being passed to
   ``numpy.random.Generator.choice``, which enforces its own (stricter)
   internal sum-to-one tolerance; this mirrors the clip-then-
   renormalize pattern already used by the M27/M28 conditional-logit
   softmax and the M26 frequency baseline, applied here to an
   *externally supplied* vector rather than one this module computes
   itself. ``n_samples`` is validated as a positive int via
   ``models._shared._validate_positive_int`` (the sanctioned lateral
   ``models._shared`` import — see decision 6).
6. **Module placement: ``models/ancestral_veto_sampler.py``, not
   ``evaluation/``.** This is a *prediction-producing* module (it
   produces sampled sequences, not a score against a metric) — the
   exact same category ``models/greedy_veto_simulator.py``'s own
   module docstring places itself in, and the sampler is structurally
   its stochastic sibling (decision 2). Consequently it may depend
   only on ``utils.config`` / ``utils.asof`` (pool/date resolution,
   identical to ``simulate_veto``'s own dependencies) plus the one
   sanctioned lateral ``models._shared`` import
   (``_validate_positive_int``) — it needs no ``features.*`` import at
   all, since (unlike M25) it never computes a score itself, only
   consumes opaque caller-supplied predictor callables. It must **not**
   depend on ``evaluation.veto_evaluation`` (models/ must not depend
   upward on evaluation/) and must **not** ``import`` the sibling
   ``models/greedy_veto_simulator.py`` module (the models-module
   boundary test forbids any ``from models.`` statement other than
   ``models._shared``) — so :data:`ACTION_SEQUENCES` (the 3-entry
   ``{"Bo1": ..., "Bo3": ..., "Bo5": ...}`` action-sequence table) is
   **duplicated** as a local module constant, following the exact
   precedent ``models/conditional_logit_pick.py`` set by duplicating
   ``FEATURE_NAMES`` rather than importing it from its M27 sibling.
   The :data:`VetoStepPredictorFn` type alias is likewise **duplicated**
   (structurally identical ``Callable`` type, not imported from
   ``evaluation/veto_evaluation.py``) for the same module-boundary
   reason. A parity test cross-checks the duplicated
   :data:`ACTION_SEQUENCES` against the sibling module's constant so
   the two cannot silently drift apart.
7. **No CLI driver in this task.** Roadmap sizes this **S**, and the
   sampler's only consumer today is M31 (Stage 3 series
   marginalisation, not yet built) — there is no standalone reporting
   artifact this milestone needs to produce, matching the precedent
   that the M25 ``models/greedy_veto_simulator.py`` shipped with **no**
   driver of its own (only M26, a full milestone later, added
   ``drivers/evaluate_veto.py``, and only because it needed to write a
   comparison-report artifact). This is a pure library function,
   exercised only by its own unit tests (plus one skip-guarded real-
   ``data/v1`` smoke test) in this task; a CLI surface (if ever needed
   for manual inspection) is deferred to whichever later milestone
   first needs one.
8. **Non-determinism in turn order / decider forcing is an inherited
   assumption, not new.** The strict step-index-parity turn order and
   the "last remaining map is forced at the decider" rule are exactly
   ``models/greedy_veto_simulator.py``'s ``simulate_veto`` decisions 2
   and (the decider handling folded into its main walk) — this task
   does not revisit whether that's the right model of a real veto's
   turn order, it inherits the same assumption for the same reason
   (every real Bo3/Bo5 veto note observed to date matches strict
   alternation except one documented Bo5 outlier).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models._shared import _validate_positive_int
from utils import asof, config

# The three supported veto formats mapped to their 7-element action
# sequences — an intentional independent duplication of the identical
# constant in ``models/greedy_veto_simulator.py`` per decision 6 (the
# models-module boundary test forbids any ``from models.`` statement
# other than ``models._shared``, so the table cannot be imported). The
# evidence for these shapes lives in ``tests/test_veto_fixtures.py``'s
# module docstring (the Bo3 fixtures and the two real Bo5 notes all
# render 7 segments) plus the standard Bo1 convention. A parity test
# in ``tests/test_ancestral_veto_sampler.py`` cross-checks this
# constant against the sibling module's so the duplication cannot
# silently drift apart.
ACTION_SEQUENCES = {
    "Bo1": ("ban", "ban", "ban", "ban", "ban", "ban", "decider"),
    "Bo3": ("ban", "ban", "pick", "pick", "ban", "ban", "decider"),
    "Bo5": ("ban", "ban", "pick", "pick", "pick", "pick", "decider"),
}

# The generic per-step predictor interface, duplicated per decision 6:
# a callable taking the acting team's stable id (``None`` only for a
# decider, which the sampler never asks about), the action (``"ban"``
# or ``"pick"``), the alphabetically sorted list of maps still in play,
# the as-of date, and the full matches/maps tables, and returning a
# probability distribution aligned 1:1 to the sorted remaining-maps
# order, summing to 1. Structurally identical to the type
# ``evaluation/veto_evaluation.py`` names ``VetoStepPredictorFn`` but
# deliberately not imported from there (models/ must not depend upward
# on evaluation/).
VetoStepPredictorFn = Callable[
    [str | None, str, Sequence[str], str, pd.DataFrame, pd.DataFrame],
    Sequence[float],
]

# Absolute tolerance for a predictor's returned vector summing to 1.0
# (decision 5): the vector must be within this of a true distribution
# before it is renormalized and handed to
# ``numpy.random.Generator.choice``. Mirrors the clip-then-renormalize
# convention the M27/M28 softmax and the M26 frequency baseline use.
_PROB_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SampledVetoAction:
    """One step of a sampled veto sequence.

    The sampler's per-step output type, deliberately *not* the sibling
    ``models/greedy_veto_simulator.SimulatedVetoAction`` (per decision
    6 the sampler may not ``import`` that module, and the sibling type
    has no ``probability`` field and is not extended in place — decision
    3). Carries the same four field names as the sibling type plus
    ``probability``; like the sibling, ``team`` holds a stable
    ``team_id`` (the id string ``utils.asof`` consumes, e.g. ``"397"``),
    *not* a vlr.gg abbreviation.

    Attributes:
        step_index: 0-based position of this action in the veto
            sequence (0 for the first ban, 1 for the second, ...).
        team: The acting team's stable ``team_id`` (``team_a_id`` on
            even steps, ``team_b_id`` on odd steps — decision 8's
            strict alternation), or ``None`` for a decider action — the
            last remaining map is forced, not chosen, so no team is
            credited with the decision (decision 2).
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        map_name: The drawn map's normalized name (e.g. ``"Haven"``);
            for a decider, the one map left after all bans/picks.
        probability: The drawn map's probability under the step's
            distribution (the value the predictor returned for that map
            before the draw, i.e. the map's entry of the validated,
            renormalized vector). ``1.0`` for a decider step — a
            documentation convention for the forced last map (decision
            2), not a modeled belief, and excluded from the sequence
            probability's product (decision 3).
    """

    step_index: int
    team: str | None
    action: str
    map_name: str
    probability: float

    def to_dict(self) -> dict[str, object]:
        """Serialize this sampled veto action to a JSON-compatible dict.

        Returns:
            A dict with keys ``"step_index"``, ``"team"``, ``"action"``,
            ``"map_name"`` and ``"probability"``, suitable for
            ``json.dumps``. The first four keys mirror
            ``SimulatedVetoAction.to_dict``'s shape (so a future M31
            aggregation or driver sees the same key names on both
            sides); ``probability`` is the added sampler-only field.

        Raises:
            Nothing.
        """
        return {
            "step_index": self.step_index,
            "team": self.team,
            "action": self.action,
            "map_name": self.map_name,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class SampledVetoSequence:
    """One full sampled veto walk, self-contained with its probability.

    The sampler's per-sample output (decision 3): the full walk of one
    generative forward simulation — ``actions`` covers every step of
    ``ACTION_SEQUENCES[best_of]`` *including* the forced decider —
    plus ``sequence_probability``, the product of every **non-decider**
    step's drawn ``probability`` (the decider's ``1.0`` is excluded by
    convention, decision 2/3). All four identifying fields are
    duplicated onto the sequence (rather than derivable only from the
    call) so a raw sample is self-contained: M31's "average across
    M29's samples weighted by sequence probability" can group samples
    by any subset of these fields without needing the original call
    arguments.

    Attributes:
        team_a_id: The stable id of the team that acted on even steps
            ("team A").
        team_b_id: The stable id of the team that acted on odd steps
            ("team B").
        best_of: The veto format (one of :data:`ACTION_SEQUENCES`'s
            keys).
        date: The as-of date the walk used.
        actions: The ``tuple`` of :class:`SampledVetoAction` objects,
            one per step of ``ACTION_SEQUENCES[best_of]`` in step
            order (the forced decider included).
        sequence_probability: The product of every non-decider step's
            ``probability``, a ``float`` in ``(0, 1]`` (each factor is
            in ``(0, 1]`` after validation/renormalization; a product
            of strictly positive factors is strictly positive).
    """

    team_a_id: str
    team_b_id: str
    best_of: str
    date: str
    actions: tuple[SampledVetoAction, ...]
    sequence_probability: float

    def to_dict(self) -> dict[str, object]:
        """Serialize this sampled sequence to a JSON-compatible dict.

        Returns:
            A dict with keys ``"team_a_id"``, ``"team_b_id"``,
            ``"best_of"``, ``"date"``, ``"actions"`` (a list of each
            action's :meth:`SampledVetoAction.to_dict` dict) and
            ``"sequence_probability"``, suitable for ``json.dumps``.

        Raises:
            Nothing.
        """
        return {
            "team_a_id": self.team_a_id,
            "team_b_id": self.team_b_id,
            "best_of": self.best_of,
            "date": self.date,
            "actions": [action.to_dict() for action in self.actions],
            "sequence_probability": self.sequence_probability,
        }


def _validate_step_distribution(
    probs,
    n_expected: int,
    context: str,
) -> np.ndarray:
    """Validate a predictor's returned vector and renormalize it.

    Implements decision 5's defensive validation of an externally
    supplied per-step distribution, run before any random draw: the
    vector must have exactly ``n_expected`` entries, every entry must
    be finite and non-negative, and the entries must sum to ``1.0``
    within :data:`_PROB_SUM_TOLERANCE`. Any violation raises
    ``ValueError`` naming ``context`` (the match/step/action string the
    caller supplies, so the failure identifies which arm misbehaved).
    The validated vector is then renormalized (``values / sum``) and
    returned — the clip-then-renormalize safeguard so
    ``numpy.random.Generator.choice``'s own (stricter) internal
    sum-to-one check never sees a vector that only approximately sums
    to 1. A vector that sums to ``0`` (all-zero entries, or entries
    that cancel) fails the sum-tolerance check before renormalization
    can divide by zero.

    Args:
        probs: The predictor's raw returned vector (any iterable of
            numbers — a ``list``, tuple or numpy array).
        n_expected: The number of entries required (``len(remaining)``
            at the calling step).
        context: A human-readable description of the step being
            validated (e.g. ``"Bo3 veto step 2 (pick) by team '397'
            at date '...' with 5 remaining map(s)"``); the error
            messages embed it verbatim.

    Returns:
        The validated, renormalized vector as a 1-D ``float`` numpy
        array of length ``n_expected``, summing to ``1.0`` within float
        rounding (well inside any tolerance ``Generator.choice``
        enforces).

    Raises:
        ValueError: If the vector has other than ``n_expected``
            entries (naming both counts); if any entry is NaN/infinite;
            if any entry is negative; or if the entries do not sum to
            ``1.0`` within :data:`_PROB_SUM_TOLERANCE` (naming the
            observed sum).
    """
    values = np.asarray(probs, dtype=float).ravel()
    if values.shape[0] != n_expected:
        raise ValueError(
            f"{context}: predictor returned {values.shape[0]} "
            f"probabilit(ies) but {n_expected} map(s) remain; the "
            "distribution must align 1:1 to the sorted remaining-maps "
            "list"
        )
    if not np.isfinite(values).all():
        bad = [
            (i, float(values[i]))
            for i in range(len(values))
            if not np.isfinite(values[i])
        ]
        raise ValueError(
            f"{context}: predictor returned non-finite probabilit(ies) "
            f"at index/indices {[i for i, _ in bad]}; every entry must "
            "be finite"
        )
    if np.any(values < 0.0):
        bad = [(i, float(values[i])) for i in range(len(values)) if values[i] < 0.0]
        raise ValueError(
            f"{context}: predictor returned negative probabilit(ies) at "
            f"index/indices {[i for i, _ in bad]} with value(s) "
            f"{[v for _, v in bad]}; every entry must be non-negative"
        )
    total = float(values.sum())
    if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
        raise ValueError(
            f"{context}: predictor probabilities sum to {total} but must "
            f"sum to 1.0 within {_PROB_SUM_TOLERANCE}; refusing to draw "
            "from a non-distribution"
        )
    return values / total


def sample_veto_sequences(
    team_a_id: str,
    team_b_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    predictor_fn_by_action: dict[str, VetoStepPredictorFn],
    n_samples,
    rng: np.random.Generator,
    map_pool=None,
) -> list[SampledVetoSequence]:
    """Sample ``n_samples`` full veto sequences forward (M29).

    Runs ``n_samples`` independent generative forward walks of the
    fixed action sequence for ``best_of`` over the resolved 7-map pool
    (decision 2). Each walk alternates turns strictly by step-index
    parity (even -> ``team_a_id``, odd -> ``team_b_id``; decision 8),
    removes each chosen map from the ``remaining`` set, and at every
    ``ban``/``pick`` step:

    - calls ``predictor_fn_by_action[action](acting_team_id, action,
      sorted(remaining), date, matches_df, maps_df)`` — the candidate
      list is the remaining maps sorted by normalized name, so the
      returned distribution aligns 1:1 to it (decision 1);
    - validates and renormalizes the returned vector via
      :func:`_validate_step_distribution` (decision 5);
    - draws one map index via ``rng.choice`` (decision 4) and records
      the drawn map's probability as the step's ``probability``.

    The decider step is forced (decision 2): at that step exactly one
    map remains, it is emitted deterministically with
    ``probability=1.0`` and no predictor is consulted / no draw
    happens. Each walk's ``sequence_probability`` is the product of
    every non-decider step's drawn probability (the decider's ``1.0``
    excluded per decision 3). The ``n_samples`` walks consume the
    single caller-supplied ``rng`` sequentially, so identical
    ``(seed, inputs)`` reproduce byte-identical output.

    Args:
        team_a_id: The stable id of the team acting on even steps
            ("team A"); whichever team the caller passes first.
        team_b_id: The stable id of the team acting on odd steps
            ("team B").
        best_of: The veto format; must be a key of
            :data:`ACTION_SEQUENCES` (``"Bo1"``, ``"Bo3"`` or
            ``"Bo5"``).
        date: The single as-of cutoff passed through to the predictors
            (e.g. the simulated match's own ISO-8601 timestamp; every
            predictor is expected to honor the strict ``<`` boundary
            against it). When ``map_pool`` is ``None``, the cutoff's
            UTC calendar date also selects the era whose pool is used.
        matches_df: The materialised ``matches`` table, passed through
            to every predictor call.
        maps_df: The materialised ``maps`` table, passed through to
            every predictor call.
        predictor_fn_by_action: A dict mapping each non-decider action
            actually present in ``ACTION_SEQUENCES[best_of]`` to a
            :data:`VetoStepPredictorFn`-shaped callable. A ``Bo3``/
            ``Bo5`` sequence needs both ``"ban"`` and ``"pick"`` keys;
            a ``Bo1`` sequence needs only ``"ban"``. An action-agnostic
            arm (the M25 greedy step model, the frequency baseline) is
            supplied under both keys — decision 1's explicit wiring
            convention. Missing keys raise ``ValueError`` before any
            sampling.
        n_samples: How many independent walks to run; must be a
            positive integer (validated via
            ``models._shared._validate_positive_int``).
        rng: The caller-constructed ``numpy.random.Generator``
            (e.g. ``numpy.random.default_rng(seed)``) that all draws
            consume sequentially (decision 4). No default — an omitted
            ``rng`` is a ``TypeError`` from Python's own argument
            binding, not a hidden global state.
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves the pool from
            ``config.json`` via ``utils.config.Config.era_as_of`` on
            ``date``'s calendar date. Every entry (caller-supplied or
            from the config) is normalized via
            ``utils.config.normalize_map_name``; duplicates after
            normalization raise ``ValueError`` (they would collapse in
            the ``remaining`` set and mis-sync the walk), as would a
            pool whose size mismatches ``len(ACTION_SEQUENCES[best_of])``.

    Returns:
        A list of exactly ``n_samples`` :class:`SampledVetoSequence`
        objects, each a complete self-contained walk (decider
        included) with its own ``sequence_probability``. Raw samples,
        not a pre-aggregated distribution — grouping/averaging is M31's
        job (decision 3). The list is reproducible for identical
        ``(seed, inputs)``.

    Raises:
        ValueError: If ``best_of`` is not a key of
            :data:`ACTION_SEQUENCES` (the message names the invalid
            value); if ``n_samples`` is not a positive integer; if
            ``predictor_fn_by_action`` lacks a key for a non-decider
            action in ``ACTION_SEQUENCES[best_of]`` (naming the missing
            action); if ``len(map_pool) != len(ACTION_SEQUENCES[best_of])``
            after normalization; if the normalized pool contains
            duplicates; if more or fewer than one map remains at the
            decider step (an internal pool/sequence desync); if any
            predictor's returned vector fails validation (propagated
            from :func:`_validate_step_distribution`, naming the
            step/arm); or if ``date`` is null/unparseable/timezone-
            aware (from ``utils.asof.parse_query_date``, propagated
            through the pool resolution).
        ConfigError: If ``map_pool`` is ``None`` and ``config.json``
            cannot be loaded/validated, if no configured era covers
            ``date``'s calendar date, or if a map name (caller-supplied
            pool entry) is not a string (from
            ``utils.config.load_config`` /
            ``utils.config.Config.era_as_of`` /
            ``utils.config.normalize_map_name``, propagated).
        TypeError: If ``rng`` is omitted (Python's own argument
            binding) or is not a ``numpy.random.Generator`` (the
            ``rng.choice`` call raises); if ``date`` is list-like
            (propagated from ``utils.asof.parse_query_date``); or if a
            predictor callable itself raises ``TypeError`` (propagated
            verbatim — the sampler does not catch misbehaving
            predictors, it only validates their return values).
        KeyError: If ``predictor_fn_by_action`` is a ``dict``-like
            whose ``[]`` access raises (propagated — after the
            presence check every required key is present, so this only
            surfaces for a non-``dict`` mapping with odd lookup
            semantics).
    """
    sequence = ACTION_SEQUENCES.get(best_of)
    if sequence is None:
        raise ValueError(
            f"best_of {best_of!r} is not a supported veto format; "
            f"expected one of {sorted(ACTION_SEQUENCES)}"
        )

    n = _validate_positive_int(n_samples, "n_samples")

    if map_pool is None:
        # Resolve the active era's pool for the query date's calendar
        # date, mirroring simulate_veto's own resolution exactly
        # (duplicated logic per decision 6, not imported). The date is
        # validated here with the same parse the predictors will use.
        query_ts = asof.parse_query_date(date)
        pool = list(config.load_config().era_as_of(query_ts.date()).map_pool)
    else:
        pool = [config.normalize_map_name(name) for name in map_pool]

    if len(pool) != len(sequence):
        raise ValueError(
            f"map_pool has {len(pool)} map(s) but a {best_of} veto "
            f"needs {len(sequence)}; the sampler only supports the "
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

    required_actions = {action for action in sequence if action != "decider"}
    missing = sorted(required_actions - set(predictor_fn_by_action))
    if missing:
        raise ValueError(
            f"predictor_fn_by_action is missing a predictor for the "
            f"required veto action(s) {missing} of a {best_of} veto; "
            "every non-decider action in the sequence must have a key "
            "(the decider step is forced and needs no predictor)"
        )

    samples: list[SampledVetoSequence] = []
    for _ in range(n):
        remaining = set(pool)
        actions: list[SampledVetoAction] = []
        sequence_probability = 1.0
        for step_index, action in enumerate(sequence):
            if action == "decider":
                if len(remaining) != 1:
                    raise ValueError(
                        f"internal error: {len(remaining)} map(s) remained "
                        "at the decider step; expected exactly 1 (the "
                        "action sequence and map_pool are out of sync)"
                    )
                actions.append(
                    SampledVetoAction(
                        step_index=step_index,
                        team=None,
                        action="decider",
                        map_name=next(iter(remaining)),
                        probability=1.0,
                    )
                )
                continue
            if action not in ("ban", "pick"):
                # ACTION_SEQUENCES is module-owned data, so any other
                # value is an internal bug, not bad input — fail loudly.
                raise ValueError(
                    f"internal error: unknown veto action {action!r} in "
                    f"ACTION_SEQUENCES[{best_of!r}]"
                )

            if step_index % 2 == 0:
                acting_id = team_a_id
            else:
                acting_id = team_b_id

            sorted_maps = sorted(remaining)
            predictor_fn = predictor_fn_by_action[action]
            probs = predictor_fn(
                acting_id,
                action,
                sorted_maps,
                date,
                matches_df,
                maps_df,
            )
            context = (
                f"{best_of} veto step {step_index} ({action}) by team "
                f"{acting_id!r} at date {date!r} with "
                f"{len(sorted_maps)} remaining map(s)"
            )
            validated = _validate_step_distribution(probs, len(sorted_maps), context)
            chosen_index = int(
                rng.choice(len(validated), p=validated)
            )
            chosen = sorted_maps[chosen_index]
            probability = float(validated[chosen_index])
            actions.append(
                SampledVetoAction(
                    step_index=step_index,
                    team=acting_id,
                    action=action,
                    map_name=chosen,
                    probability=probability,
                )
            )
            sequence_probability *= probability
            remaining.remove(chosen)

        samples.append(
            SampledVetoSequence(
                team_a_id=team_a_id,
                team_b_id=team_b_id,
                best_of=best_of,
                date=date,
                actions=tuple(actions),
                sequence_probability=sequence_probability,
            )
        )

    return samples

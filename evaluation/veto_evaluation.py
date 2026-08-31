"""Per-step veto evaluation harness (roadmap M26).

Scores a per-step veto predictor against the real veto logs of the
held-out M10 ``test`` split: at each real historical step of a Bo3
sequence it asks the predictor for a probability distribution over the
maps still in play, then computes per-step cross-entropy and top-1 /
top-3 accuracy (over the alphabetically tie-broken ranking of decision
3), and compares two arms head to head — the M25 greedy simulator
(``models.greedy_veto_simulator``, wrapped as a softmax distribution by
:func:`greedy_veto_step_model`) and the global "most frequently played
map" frequency baseline (:func:`most_frequent_map_baseline_model`).

Scope / conventions (the ten plan decisions, recorded here verbatim so
BUILD and later milestones do not re-derive them):

1. **Team-identity reconciliation is a static, hand-verified 16-entry
   abbreviation → ``team_id`` table** (:data:`TEAM_ABBREVIATION_TO_ID`),
   self-checked at load time by :func:`verify_team_abbreviation_map` —
   not a general mechanical derivation. ``scraper/models.py``'s
   ``VetoAction`` docstring already states short forms are "not
   mechanically derivable from the full names", and pure constraint
   propagation over the veto data does not converge without at least
   one seeded anchor per connected component (verified empirically
   during planning: zero of 16 abbreviations resolve from propagation
   alone). The table is hand-derived from each team's known vlr.gg tag
   and *verified* against every one of the 96 matches in
   ``data/v1/veto_actions.parquet`` (for each match, the two
   abbreviations' mapped ids must equal exactly ``{team1_id,
   team2_id}``) — confirmed to hold with **zero mismatches**. This
   table is a maintenance point, not an automatable derivation: a
   future scrape introducing a 17th abbreviation must make
   :func:`verify_team_abbreviation_map` raise loudly rather than
   silently mis-mapping or dropping the match.
2. **Held-out set: the M10 ``test`` split, restricted to matches that
   have veto data.** Mirrors ``evaluation/harness.py``'s own precedent
   (``"test"`` is the only split reserved for final evaluation).
   Verified during planning: all 15 ``test``-split matches have veto
   rows, all 15 are ``Bo3``, and every one's real veto map-name set
   exactly equals the map pool ``utils.config.Config.era_as_of``
   resolves for that match's date — no pool-mismatch edge case is live
   in the current held-out set, though :func:`score_veto_steps` must
   still fail loudly (not silently) if a future dataset version breaks
   this.
3. **Per-step predictor interface (so M27's conditional-logit model
   drops in unchanged):** :data:`VetoStepPredictorFn`, called as
   ``predictor_fn(acting_team_id, action, remaining_maps, date,
   matches_df, maps_df)`` where ``remaining_maps`` is the
   **alphabetically sorted** (by normalized name) list of maps still
   in play at that step, and the returned sequence is a probability
   distribution aligned 1:1 to that same order, summing to 1. Sorting
   ``remaining_maps`` before every call keeps the tie-break convention
   identical to ``models.greedy_veto_simulator``'s decision 4 and
   makes the interface deterministic regardless of set-iteration
   order. ``action`` is ``"ban"`` or ``"pick"`` (never ``"decider"``
   — see decision 5).
4. **State bookkeeping is teacher-forced on the real observed
   sequence, not the model's own hypothetical rollout.**
   ``models.greedy_veto_simulator.simulate_veto`` only ever produces
   its *own* trajectory; M26 instead needs, at each real historical
   step, "what would this model predict given the true history so
   far." So :func:`score_veto_steps` replays each held-out match's
   *real* sequence, maintaining ``remaining = era_pool - {real
   map_names already consumed by steps 0..i-1}``, and asks the
   predictor for its distribution over that same ``remaining`` set at
   step ``i`` — the standard reason a step-level predictor primitive
   is needed instead of reusing ``simulate_veto`` end-to-end.
5. **Decider steps are excluded from scoring.** At the decider step
   exactly one map remains and it is forced, not chosen (mirrored by
   ``SimulatedVetoAction.team = None`` in M25) — there is no
   distribution to score and no ranking to evaluate, so decider rows
   are skipped entirely by :func:`score_veto_steps` (they are still
   consumed for ``remaining``-set bookkeeping purposes, trivially,
   since nothing is left to remove afterward).
6. **M25's distribution: softmax over :func:`models.greedy_veto_simulator.team_map_scores`,
   temperature fixed at 1, sign flipped for bans.**
   ``simulate_veto`` is deterministic argmin/argmax; M25's own module
   docstring calls it "the hard-argmax limit of roadmap M27 (a
   conditional-logit ban model)". M26 therefore scores M25 as the
   plainest distributional generalization of that same rule: for a
   pick, ``softmax(score)`` over ``remaining_maps`` (higher own
   win-rate mean → higher probability); for a ban, ``softmax(-score)``
   (lower own win-rate mean → higher probability of being banned).
   ``T = 1`` is fixed, not fit — it is a lens for evaluating the
   existing deterministic rule, not a new trained parameter, and is
   documented as such rather than presented as a genuine probabilistic
   claim about M25. The score dict comes from the single shared
   ``team_map_scores`` function (extracted in M26 so the computation
   lives in exactly one place), preserving the DAG
   (``evaluation/`` may depend on ``models/``).
7. **The "most frequently played map" baseline is global and
   action-agnostic — not team-specific, not ban/pick-directional.**
   There is no natural "this baseline predicts bans differently from
   picks" signal in raw play-frequency, so
   :func:`most_frequent_map_baseline_model` returns the *same*
   distribution for ``action="ban"`` and ``action="pick"``: each
   remaining map's probability is its share of historical play counts
   (a map's appearances in ``maps_df``, restricted to completed maps
   with a non-null ``winner``, strictly before the query date —
   leakage-safe by the same strict-``<`` convention as ``utils.asof``)
   among the remaining set, renormalized. If every remaining map has
   zero historical count (all-new pool), it falls back to a uniform
   distribution over ``remaining_maps``. One deviation from the pure
   frequency share, documented here: a map with a zero prior count
   among a nonzero grand total would otherwise receive exactly
   probability 0 — which makes ``utils.scoring.log_loss`` undefined on
   the step where that map is the real choice (a live edge case: one
   real held-out step picks a map with zero strictly-prior plays).
   The distribution therefore floors every share at
   :data:`_BASELINE_PROB_FLOOR` and renormalizes, mirroring the
   clip-into-``[eps, 1-eps]`` convention of
   ``features.map_win_rate.select_k``; the floor is far below any
   ranking-relevant probability, so top-1/top-3 ranks are unchanged
   and only the degenerate-zero case becomes a finite (if large)
   cross-entropy instead of ``+inf``. It ignores ``acting_team_id``
   entirely (documented, not silently unused). The as-of-frequency
   filtering is a small private helper (:func:`_as_of_map_play_counts`)
   local to this module (reusing ``utils.asof.parse_query_date`` /
   ``parse_date_column`` for the date parsing), not a new
   ``utils.asof`` function — it has exactly one caller today, which
   does not clear the "shared by multiple features" bar for extraction
   into ``utils/``.
8. **Metrics, per scored (match, step) pair:**
   - Cross-entropy: ``utils.scoring.log_loss(probs, true_index)``,
     where ``true_index`` is the position of the real chosen map
     within the alphabetically sorted ``remaining_maps`` —
     ``log_loss`` is already generic over any candidate-set size
     ``K``, so no new metric function is needed.
   - Top-1 / top-3: rank the candidates by predicted probability
     descending, tie-broken by ascending map name (same convention as
     decision 3 / M25's decision 4, for one consistent secondary key
     across the whole codebase); top-1 correct iff the true map's rank
     is 1, top-3 correct iff its rank is ``<= min(3, n_remaining)``.
     **Known, documented artifact:** top-3 accuracy is trivially 1.0
     whenever ``n_remaining <= 3`` (every step from roughly the
     midpoint of the sequence onward) — reported as-is rather than
     silently excluding those steps, since there is no clearly-correct
     alternative and dropping steps would complicate aggregation
     without a clear win.
9. **Report shape and artifact.** One JSON artifact,
   ``data/<version>/veto_evaluation_report.json``, holding both arms
   (M25 greedy vs. the frequency baseline) side by side plus their
   deltas — mirroring ``temperature_calibration_report.json``'s "one
   artifact, two arms, a delta block" shape. Each arm's block has an
   aggregate (``n_steps``, ``mean_cross_entropy``, ``top1_accuracy``,
   ``top3_accuracy``) and a per-``step_index`` breakdown list (mirroring
   ``evaluation/harness.py``'s per-category ``calibration`` breakdown
   precedent), since "per-step" in the roadmap line most naturally
   means "broken out by step," not just "averaged over all steps."
10. **Module placement (preserves the DAG):** the abbreviation table +
    verification, the held-out builder, the :data:`VetoStepPredictorFn`
    type, both concrete predictors, the per-step scorer, and the report
    builder all live here; ``models/greedy_veto_simulator.py`` hosts
    the extracted ``team_map_scores``; ``drivers/evaluate_veto.py`` is
    the CLI driver. This module must not import
    ``evaluation.harness`` (the module-boundary test forbids
    evaluation-to-evaluation imports) or anything from ``drivers/``.
11. **The teacher-forced replay is a single shared walk (M27).** The
    per-match pool resolution, the ``real_maps == pool`` guard, and the
    ``remaining``-tracking walk over ``ordered.itertuples()`` live in
    one private generator, :func:`_iter_teacher_forced_steps`, which
    yields one record per *every* step (deciders included, tagged) with
    ``match_id``, ``step_index``, ``action``, ``acting_team_id``,
    ``opponent_team_id`` (the other id of the match's ``{team1_id,
    team2_id}`` pair, resolved from the held-out row's own columns),
    ``sorted_remaining_maps``, ``date`` and ``true_map``.
    :func:`score_veto_steps` is a thin consumer of that generator, and
    it gained an optional ``actions_to_score`` filter (default
    ``None`` = score every non-decider step, exactly M26's behavior):
    when given (e.g. ``{"ban"}``) steps whose action is not in the set
    are skipped from scoring like deciders but still consumed for
    ``remaining`` bookkeeping — this is what lets M27 score ban-only on
    the same teacher-forced replay without a second bespoke loop.
12. **Ban training examples come from the same replay (M27).**
    :func:`build_ban_training_examples` walks
    :func:`_iter_teacher_forced_steps` and keeps only ``action ==
    "ban"`` rows, returning one frozen :class:`BanTrainingExample` per
    ban step: acting/opponent team ids, the sorted remaining candidate
    list, the date, and ``true_map_index`` (the position of the real
    banned map within its own sorted candidate list — the label a
    per-step softmax is scored against). No featurization happens here:
    the function returns raw ids/dates/candidate-lists only, because
    feature building is ``models/``'s job per the module-boundary
    standard.
13. **N-arm comparison report (M27).**
    :func:`build_veto_multi_arm_report` generalises the 2-arm
    :func:`build_veto_comparison_report` to any number of arms: the
    same row-alignment guard (all arms must describe the identical
    held-out rows at identical positions) plus one per-arm
    :func:`build_veto_evaluation_report` block and one delta block per
    non-baseline arm (``deltas_vs_<baseline_arm>``). The 2-arm builder
    is unchanged and still used by ``drivers/evaluate_veto.py``.

Dependency rung: ``utils/ -> features/ -> models/ -> evaluation/ ->
drivers/``. This module may depend downward on ``models.*``,
``features.*`` and ``utils.*`` only (encoded as a regression test in
``tests/test_module_boundaries.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import map_win_rate
from models.greedy_veto_simulator import team_map_scores
from utils import asof, config, scoring, splits

# ---------------------------------------------------------------------------
# Decision 1: the static, hand-verified abbreviation -> team_id table.
# ---------------------------------------------------------------------------

# The closed 16-entry vlr.gg-abbreviation -> stable-team_id table.
# Source: hand-derived from each team's known vlr.gg tag (the parenthesized
# full names are the display names those tags abbreviate) and *verified*
# against every one of the 96 matches in data/v1/veto_actions.parquet by
# verify_team_abbreviation_map: for each match the two abbreviations'
# mapped ids equal exactly {team1_id, team2_id} — confirmed to hold with
# zero mismatches during planning. Values are strings matching the
# object dtype of matches.team1_id/team2_id. This is a maintenance point,
# not an automatable derivation (see the module docstring's decision 1):
# a 17th abbreviation must trip verify_team_abbreviation_map's ValueError.
TEAM_ABBREVIATION_TO_ID: dict[str, str] = {
    "FNC": "2593",  # FNATIC
    "VIT": "2059",  # Team Vitality
    "PCF": "3478",  # PCIFIC Esports
    "EF": "6392",  # Eternal Fire
    "NAVI": "4915",  # Natus Vincere
    "FF": "20085",  # Fire Flux Esports
    "SGE": "14478",  # Eintracht Frankfurt
    "TH": "1001",  # Team Heretics
    "TL": "474",  # Team Liquid
    "BBL": "397",  # BBL Esports
    "FUT": "1184",  # FUT Esports
    "KC": "8877",  # Karmine Corp
    "GX": "14419",  # GIANTX
    "JL": "11232",  # Joblife
    "EP": "876",  # Enterprise Esports
    "M8": "12694",  # Gentle Mates
}

# ---------------------------------------------------------------------------
# Decision 3: the generic per-step predictor interface.
# ---------------------------------------------------------------------------

# The generic model interface every evaluated veto-step predictor must
# satisfy: a callable taking the acting team's stable id (None only for
# a decider, which the scorer never asks about), the action ("ban" or
# "pick"), the alphabetically sorted list of maps still in play, the
# as-of date, and the full matches/maps tables, and returning a
# probability distribution aligned 1:1 to the sorted remaining-maps
# order, summing to 1 (the scorer validates its length; simplex
# validation is delegated to utils.scoring's metric functions).
VetoStepPredictorFn = Callable[
    [str | None, str, Sequence[str], str, pd.DataFrame, pd.DataFrame],
    Sequence[float],
]

# Fixed column order for the held-out veto-step table produced by
# build_held_out_veto_matches.
HELD_OUT_VETO_COLUMNS = (
    "match_id",
    "step_index",
    "team",
    "action",
    "map_name",
    "team_id",
    "date",
    "team1_id",
    "team2_id",
    "best_of",
    "split",
)

# Fixed column order for the scored table produced by
# score_veto_steps: one row per scored (non-decider) step.
SCORED_STEP_COLUMNS = (
    "match_id",
    "step_index",
    "action",
    "n_remaining",
    "cross_entropy",
    "top1_correct",
    "top3_correct",
)

# The identifying pair used for the comparison report's row-alignment
# validation.
_STEP_ID_COLUMNS = ("match_id", "step_index")


def verify_team_abbreviation_map(
    veto_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> None:
    """Verify the abbreviation table reconciles every veto match.

    The decision-1 self-check, run at load time before any downstream
    step trusts :data:`TEAM_ABBREVIATION_TO_ID`. For every ``match_id``
    present in ``veto_df`` it collects the match's distinct non-null
    ``team`` abbreviations, maps each through
    :data:`TEAM_ABBREVIATION_TO_ID`, and asserts the mapped id set
    equals exactly ``{team1_id, team2_id}`` from ``matches_df`` for
    that match. Any mismatch raises ``ValueError`` naming the offending
    match id and abbreviation — a future scrape that introduces a 17th
    abbreviation, a data entry that flips two teams, or a stale
    matches/veto dataset must fail loudly here rather than silently
    mis-map or drop a match downstream.

    Args:
        veto_df: The materialised ``veto_actions`` table (needs
            ``match_id`` and ``team``, where ``team`` is the raw vlr.gg
            abbreviation or ``None`` on decider rows).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``).

    Returns:
        None (raises instead of returning on any violation).

    Raises:
        ValueError: If a veto ``match_id`` is absent from ``matches_df``
            (its ``{team1_id, team2_id}`` pair is unknowable); if a
            match has other than exactly 2 distinct non-null ``team``
            abbreviations; if an abbreviation is absent from
            :data:`TEAM_ABBREVIATION_TO_ID` (naming the abbreviation
            and match); or if the mapped id set does not equal
            ``{team1_id, team2_id}`` for some match (naming the match
            and the offending ids).
        KeyError: If ``veto_df`` lacks ``match_id``/``team`` or
            ``matches_df`` lacks ``match_id``/``team1_id``/``team2_id``
            (propagated from pandas column indexing).
    """
    match_by_id = {
        str(row.match_id): row
        for row in matches_df[["match_id", "team1_id", "team2_id"]].itertuples(
            index=False
        )
    }
    for match_id, group in veto_df.groupby("match_id", sort=True):
        match = match_by_id.get(str(match_id))
        if match is None:
            raise ValueError(
                f"veto match_id {match_id!r} is absent from matches_df; "
                "cannot verify its team abbreviations against "
                "{team1_id, team2_id}"
            )
        abbreviations = {
            str(value) for value in group["team"].dropna().unique()
        }
        if len(abbreviations) != 2:
            raise ValueError(
                f"veto match_id {match_id!r} has {len(abbreviations)} "
                f"distinct non-null team abbreviation(s) "
                f"{sorted(abbreviations)}; expected exactly 2 (the two "
                "acting teams), so the abbreviation reconciliation "
                "cannot proceed"
            )
        mapped: set[str] = set()
        for abbreviation in abbreviations:
            if abbreviation not in TEAM_ABBREVIATION_TO_ID:
                raise ValueError(
                    f"veto match_id {match_id!r} carries unknown team "
                    f"abbreviation {abbreviation!r}, which is absent from "
                    "TEAM_ABBREVIATION_TO_ID; add it manually (the table "
                    "is a hand-maintained, closed set) or the mapping "
                    "cannot be trusted"
                )
            mapped.add(TEAM_ABBREVIATION_TO_ID[abbreviation])
        expected = {str(match.team1_id), str(match.team2_id)}
        if mapped != expected:
            raise ValueError(
                f"veto match_id {match_id!r}: the mapped abbreviation "
                f"ids {sorted(mapped)} do not equal the match's "
                f"{{team1_id, team2_id}} pair {sorted(expected)}; the "
                "abbreviation table or the matches/veto data is stale"
            )


def resolve_veto_team_ids(
    veto_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a resolved ``team_id`` column to the veto-actions table.

    Runs :func:`verify_team_abbreviation_map` first (the decision-1
    self-check must pass before any abbreviation is trusted), then
    returns ``veto_df`` with an added ``team_id`` column holding the
    mapped stable id for each non-null ``team`` abbreviation, and
    ``None`` where ``team`` is ``None`` (decider rows — the last
    remaining map is forced, not chosen, so no team is credited).

    Args:
        veto_df: The materialised ``veto_actions`` table (needs
            ``match_id`` and ``team``).
        matches_df: The materialised ``matches`` table, passed through
            to :func:`verify_team_abbreviation_map`.

    Returns:
        A new ``pandas.DataFrame`` with all of ``veto_df``'s columns in
        their original order plus a trailing ``team_id`` column (object
        dtype, ``None`` on decider rows). Row order is unchanged.

    Raises:
        ValueError / KeyError: Everything propagated from
            :func:`verify_team_abbreviation_map` unchanged (an
            unresolvable abbreviation, a non-two-abbreviation match, a
            stale pair, a missing column, or a veto match absent from
            ``matches_df``).
    """
    verify_team_abbreviation_map(veto_df, matches_df)
    resolved = veto_df.copy()
    resolved["team_id"] = resolved["team"].map(
        lambda abbreviation: (
            TEAM_ABBREVIATION_TO_ID[abbreviation]
            if abbreviation is not None
            else None
        )
    )
    return resolved


def build_held_out_veto_matches(
    veto_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Assemble the held-out veto-step table to evaluate against.

    Builds the decision-2 held-out set: one row per real veto step of
    every match in the requested ``split`` that has veto data. First
    resolves abbreviations to stable ids
    (:func:`resolve_veto_team_ids`), then attaches ``date`` /
    ``team1_id`` / ``team2_id`` / ``best_of`` from ``matches_df``
    (inner join on ``match_id`` — a veto whose match is not materialised
    cannot be scored and is silently dropped), attaches the split label
    via :func:`utils.splits.join_split_to_maps` (reused as-is — it is
    already generic over any ``match_id``-keyed table despite its
    "map-level" docstring framing), and filters to the requested
    ``split`` value.

    All steps are kept, including deciders — :func:`score_veto_steps`
    is what excludes decider rows from scoring, not this builder. The
    result is sorted by ``(match_id, step_index)`` so the scorer can
    replay each match's real sequence in order.

    Args:
        veto_df: The materialised ``veto_actions`` table (needs
            ``match_id``, ``step_index``, ``team``, ``action``,
            ``map_name``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``best_of``).
        splits_df: The ``splits`` table produced by
            :func:`utils.splits.split_matches` (needs ``match_id`` and
            ``split``).
        split: The split value to hold out, ``"test"`` by default (the
            only split ``utils.splits`` defines for final evaluation).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`HELD_OUT_VETO_COLUMNS`
        (``match_id, step_index, team, action, map_name, team_id, date,
        team1_id, team2_id, best_of, split``), one row per veto step in
        ``(match_id, step_index)`` order. Never empty: an empty
        split-restricted result raises instead.

    Raises:
        ValueError: If the split-restricted result is empty (e.g. a
            splits table with no rows of that value); if any veto
            ``match_id`` is absent from ``splits_df`` (propagated from
            :func:`utils.splits.join_split_to_maps`); or everything
            propagated from :func:`resolve_veto_team_ids`
            (unresolvable abbreviation, non-two-abbreviation match,
            stale pair, veto match absent from ``matches_df``).
        KeyError: If ``veto_df`` lacks a veto column, ``matches_df``
            lacks a match column, or ``splits_df`` lacks
            ``match_id``/``split`` (propagated from pandas /
            :func:`utils.splits.join_split_to_maps` /
            :func:`resolve_veto_team_ids`).
    """
    resolved = resolve_veto_team_ids(veto_df, matches_df)
    joined = resolved.merge(
        matches_df[["match_id", "date", "team1_id", "team2_id", "best_of"]],
        on="match_id",
        how="inner",
    )
    with_split = splits.join_split_to_maps(joined, splits_df)
    held_out = with_split[with_split["split"] == split]
    held_out = held_out.sort_values(["match_id", "step_index"])
    held_out = held_out[list(HELD_OUT_VETO_COLUMNS)]
    if len(held_out) == 0:
        raise ValueError(
            f"no held-out veto steps for split {split!r}: joining veto "
            "actions to matches/splits and restricting to that split "
            "yields an empty table"
        )
    return held_out


def _softmax(scores: Sequence[float]) -> list[float]:
    """Numerically stable softmax of a sequence of real scores.

    Subtracts the maximum score before exponentiating (so
    ``exp(x - max)`` never overflows), then normalizes by the sum. Used
    with ``T = 1`` fixed (decision 6): softmax is monotone, so the
    ranking/argmax of the returned distribution matches the argmax of
    the raw scores — exactly what M25's deterministic argmin/argmax
    rule picks, with ties broken by the caller's secondary key rather
    than by float noise.

    Args:
        scores: A sequence of finite real numbers, one per candidate.

    Returns:
        A ``list`` of ``float`` probabilities, one per input in the
        same order, non-negative and summing to ``1.0`` (within float
        rounding). Every entry is strictly positive (softmax never
        assigns exactly 0), so ``utils.scoring.log_loss`` never hits
        its zero-probability hard error on a softmax output.

    Raises:
        ValueError: If ``scores`` is empty (no distribution to form),
            or if any score is non-finite (NaN/inf would poison the
            exponentials).
    """
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("cannot compute softmax over an empty score list")
    for i, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"scores[{i}] must be finite, got {value!r}")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = math.fsum(exponentials)
    return [entry / total for entry in exponentials]


def greedy_veto_step_model(
    acting_team_id: str | None,
    action: str,
    remaining_maps: Sequence[str],
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=map_win_rate.DEFAULT_K,
) -> list[float]:
    """The M25 greedy rule as a per-step probability distribution.

    Decision 6's arm: scores every map in ``remaining_maps`` for the
    acting team via
    :func:`models.greedy_veto_simulator.team_map_scores` (the single
    shared score computation, at the as-of ``date``) and returns the
    softmax distribution over those scores — ``softmax(score)`` for a
    pick (higher own win-rate mean → higher probability) and
    ``softmax(-score)`` for a ban (lower own win-rate mean → higher
    probability of being banned). ``T = 1`` is fixed, not fit: this is
    a lens for evaluating the existing deterministic rule, not a
    genuine probabilistic claim (a raw untempered softmax over win-rate
    means can be over- or under-confident — exactly the gap M27's
    trained conditional logit is meant to close).

    Softmax is monotone, so the distribution's argmax equals the map
    M25's deterministic rule would choose at that step (ties broken by
    the alphabetical ordering of ``remaining_maps``, matching
    ``simulate_veto``'s decision 4).

    Args:
        acting_team_id: The acting team's stable id. Never ``None`` in
            practice (the scorer only calls predictors for non-decider
            steps), but typed per :data:`VetoStepPredictorFn`.
        action: ``"ban"`` or ``"pick"`` (anything else raises — the
            sign flip is defined only for the two choosing actions).
        remaining_maps: The alphabetically sorted (by normalized name)
            list of maps still in play; must be non-empty. The returned
            probabilities align 1:1 to this order.
        date: The single as-of cutoff for every win-rate lookup (the
            held-out match's own date).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.
        k: The shrinkage strength passed to
            :func:`models.greedy_veto_simulator.team_map_scores`;
            defaults to ``features.map_win_rate.DEFAULT_K``.

    Returns:
        A ``list`` of ``len(remaining_maps)`` ``float`` probabilities
        summing to ``1.0``, aligned to ``remaining_maps``' order.

    Raises:
        ValueError: If ``action`` is neither ``"ban"`` nor ``"pick"``;
            if ``remaining_maps`` is empty (propagated from
            :func:`_softmax`); or if any score lookup fails
            (null/NaN score, tied scores, bad ``k``, bad ``date`` —
            propagated from
            :func:`models.greedy_veto_simulator.team_map_scores`).
        KeyError: If either table lacks a required column (propagated
            from :func:`models.greedy_veto_simulator.team_map_scores`).
        TypeError: If ``date`` is list-like (propagated from
            :func:`utils.asof.parse_query_date`).
        ConfigError: If a map name is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    if action not in ("ban", "pick"):
        raise ValueError(
            f"greedy_veto_step_model only defines a distribution for "
            f"bans and picks, got action {action!r}"
        )
    scores = team_map_scores(
        acting_team_id, remaining_maps, date, matches_df, maps_df, k
    )
    ordered = [scores[name] for name in remaining_maps]
    if action == "ban":
        ordered = [-score for score in ordered]
    return _softmax(ordered)


# The floor every frequency share of the baseline is clipped to
# before renormalization. Mirrors the purpose of
# features.map_win_rate._PROB_CLIP_EPS: a pure frequency share can be
# exactly 0 when the true map has no strictly-prior plays (a live v1
# edge case — one held-out step picks a map with zero prior counts),
# and utils.scoring.log_loss raises on an exactly-zero true-category
# probability. 1e-12 keeps the deviation from the pure share below any
# ranking-relevant magnitude (top-1/top-3 ranks unchanged) while
# keeping every cross-entropy finite.
_BASELINE_PROB_FLOOR = 1e-12


def _as_of_map_play_counts(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> dict[str, int]:
    """Count each map's completed play appearances strictly before a date.

    The decision-7 frequency source: filters ``matches_df`` to matches
    dated strictly before ``date`` (any team — the baseline is global,
    not team-specific), inner-joins ``maps_df`` on ``match_id``, keeps
    only rows with a non-null ``winner`` (a finished map), and counts
    ``map_name`` occurrences. Date parsing goes through
    ``utils.asof.parse_query_date`` / ``parse_date_column`` — the same
    strict-``<`` convention as ``utils.asof`` — so the counts are
    leakage-safe by construction. There is deliberately no
    ``status == "completed"`` filter: the plan specifies only the
    strict-before date filter plus the non-null-``winner`` map-level
    signal, and this helper follows that literal specification (a
    winner-bearing map row is a played map regardless of its parent
    match's status column).

    Args:
        date: The as-of cutoff; matches dated equal to or after this
            are excluded (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id`` and ``date``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner``, ``map_name``).

    Returns:
        A ``{normalized_map_name: count}`` dict (map names normalized
        via :func:`utils.config.normalize_map_name`) of each finished
        map's appearances among the strictly-prior matches. Maps with
        zero prior appearances are absent from the dict (callers treat
        a missing key as count 0).

    Raises:
        ValueError: If ``date`` is null/unparseable/timezone-aware or
            any match date is null/unparseable (propagated from
            :func:`utils.asof.parse_query_date` /
            :func:`utils.asof.parse_date_column`).
        TypeError: If ``date`` is list-like (propagated from
            :func:`utils.asof.parse_query_date`).
        KeyError: If ``matches_df`` lacks ``match_id``/``date`` or
            ``maps_df`` lacks ``match_id``/``winner``/``map_name``
            (propagated from pandas column indexing).
        ConfigError: If a map name is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    query = asof.parse_query_date(date)
    parsed_dates = asof.parse_date_column(matches_df[asof.DATE_COL])
    prior_matches = matches_df[parsed_dates < query]
    joined = prior_matches.merge(
        maps_df[["match_id", "winner", "map_name"]], on="match_id", how="inner"
    )
    finished = joined[joined["winner"].notna()]
    counts: dict[str, int] = {}
    for raw_name in finished["map_name"]:
        normalized = config.normalize_map_name(raw_name)
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def most_frequent_map_baseline_model(
    acting_team_id: str | None,
    action: str,
    remaining_maps: Sequence[str],
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> list[float]:
    """The global most-frequently-played-map baseline distribution.

    Decision 7's arm: for each remaining map, its probability is its
    share of historical play counts among the remaining set —
    ``count(map) / sum(count(m) for m in remaining_maps)``, counts from
    :func:`_asof_map_play_counts` (finished maps strictly before
    ``date``, leakage-safe). If every remaining map has zero historical
    count (all-new pool), the distribution falls back to uniform over
    ``remaining_maps``.

    The baseline is global and action-agnostic: it **ignores
    ``acting_team_id`` and ``action`` entirely** (documented, not
    silently unused) — there is no natural "this baseline predicts bans
    differently from picks" signal in raw play-frequency, so bans and
    picks receive the identical distribution for the same remaining
    set. The returned shares are floored at
    :data:`_BASELINE_PROB_FLOOR` and renormalized (see the module
    docstring's decision 7): a remaining map with zero strictly-prior
    plays keeps a tiny positive probability so
    ``utils.scoring.log_loss`` stays defined when that map is the real
    choice, while ranks are unchanged.

    Args:
        acting_team_id: The acting team's stable id; deliberately
            ignored (see above).
        action: ``"ban"`` or ``"pick"``; deliberately ignored (see
            above).
        remaining_maps: The alphabetically sorted (by normalized name)
            list of maps still in play; must be non-empty. The returned
            probabilities align 1:1 to this order.
        date: The as-of cutoff; maps of matches dated ``>=`` this are
            excluded from the counts (strict ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        A ``list`` of ``len(remaining_maps)`` ``float`` probabilities
        summing to ``1.0``, aligned to ``remaining_maps``' order: each
        entry is that map's share of the historical play counts among
        the remaining set (floored at :data:`_BASELINE_PROB_FLOOR` and
        renormalized), or ``1 / len(remaining_maps)`` for every
        entry when no remaining map has any prior count.

    Raises:
        ValueError: If ``remaining_maps`` is empty (no distribution to
            form); if ``date`` is null/unparseable/timezone-aware or a
            match date is null/unparseable (propagated from
            :func:`_asof_map_play_counts`).
        TypeError: If ``date`` is list-like (propagated from
            :func:`_asof_map_play_counts`).
        KeyError: If either table lacks a required column (propagated
            from :func:`_asof_map_play_counts`).
        ConfigError: If a map name is not a string (propagated from
            :func:`_asof_map_play_counts`).
    """
    # acting_team_id and action are deliberately unused (decision 7).
    del acting_team_id, action
    maps = list(remaining_maps)
    if not maps:
        raise ValueError(
            "most_frequent_map_baseline_model needs at least one "
            "remaining map to form a distribution"
        )
    counts = _as_of_map_play_counts(date, matches_df, maps_df)
    totals = [counts.get(name, 0) for name in maps]
    grand_total = math.fsum(totals)
    if grand_total == 0.0:
        # All-new pool: no map has any prior count, so the frequency
        # signal is empty and the least-committal uniform distribution
        # is the documented fallback (decision 7).
        return [1.0 / len(maps)] * len(maps)
    # Floor the pure frequency shares so a zero-prior-count map keeps a
    # tiny positive probability (a live v1 edge case — one held-out
    # step picks a map with zero strictly-prior plays — would otherwise
    # make log_loss undefined on that step), then renormalize so the
    # vector still sums to 1.
    floor = _BASELINE_PROB_FLOOR
    floored = [max(total / grand_total, floor) for total in totals]
    renorm = math.fsum(floored)
    return [value / renorm for value in floored]


def _iter_teacher_forced_steps(
    held_out_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_pool=None,
):
    """Yield one record per real veto step of every held-out match.

    The single shared teacher-forced replay (decision 11), extracted
    from :func:`score_veto_steps`'s original loop body so the
    bookkeeping is never duplicated: grouped by ``match_id`` (rows
    processed in ``step_index`` order) it seeds ``remaining`` from
    ``map_pool`` or, if ``None``, from
    ``utils.config.Config.era_as_of`` on the match's date — mirroring
    ``models.greedy_veto_simulator.simulate_veto``'s resolution exactly
    — verifies the match's real normalized map-name set exactly equals
    the resolved pool (decision 2's fail-loud clause), and walks the
    real sequence maintaining ``remaining``. Every step is yielded
    *including* deciders (tagged by their ``action``), and after each
    yield the *real* chosen map is removed from ``remaining``
    regardless of any predictor's favorite (decision 4's teacher
    forcing), so the yielded ``sorted_remaining_maps`` for step ``i``
    is exactly the candidate set the real history so far leaves in
    play.

    Args:
        held_out_df: The held-out veto-step table from
            :func:`build_held_out_veto_matches` (needs
            :data:`HELD_OUT_VETO_COLUMNS` — at minimum ``match_id``,
            ``step_index``, ``team_id``, ``action``, ``map_name``,
            ``date``; ``team1_id``/``team2_id`` optional, used only to
            resolve the opponent id when present).
        matches_df: The full materialised ``matches`` table (only used
            for the optional ``map_pool=None`` config-era resolution
            path; otherwise passed through unused).
        maps_df: The full materialised ``maps`` table, passed through
            unused (kept in the signature so consumers share the same
            call shape).
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves it per match from
            ``config.json`` via :meth:`utils.config.Config.era_as_of`
            on the match date's calendar date, exactly like
            ``simulate_veto``.

    Yields:
        One dict per veto step (deciders included) with keys:
        ``match_id`` (the group's match id), ``step_index``,
        ``action``, ``acting_team_id`` (the row's ``team_id``, or
        ``None`` on decider rows), ``opponent_team_id`` (the other id
        of the match's ``{team1_id, team2_id}`` pair when the row
        carries those columns and the acting team is one of the two,
        else ``None`` — on decider rows no acting side exists so no
        opponent is resolved), ``sorted_remaining_maps`` (the sorted
        list of maps still in play at this step, before this step's
        choice is removed), ``date`` (the row's date), and ``true_map``
        (the row's normalized map name).

    Raises:
        ValueError: If a match's real normalized map-name set does not
            exactly equal the resolved pool; if a non-decider row's
            acting ``team_id`` is neither of the match's ``team1_id``/
            ``team2_id`` (an unresolvable opponent — fail loudly
            rather than emit a wrong training label); if
            ``map_pool=None`` and no era covers the match's date or the
            config is invalid (propagated from
            :meth:`utils.config.Config.era_as_of` /
            :func:`utils.config.load_config`); or if ``date`` is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.parse_query_date`).
        KeyError: If ``held_out_df`` lacks a required column
            (propagated from pandas column indexing).
        TypeError: If a match date is list-like (propagated from
            :func:`utils.asof.parse_query_date`).
    """
    for match_id, group in held_out_df.groupby("match_id", sort=True):
        ordered = group.sort_values("step_index")
        match_date = ordered["date"].iloc[0]
        if map_pool is None:
            query_ts = asof.parse_query_date(match_date)
            pool = set(config.load_config().era_as_of(query_ts.date()).map_pool)
        else:
            pool = {config.normalize_map_name(name) for name in map_pool}

        real_maps = {config.normalize_map_name(name) for name in ordered["map_name"]}
        if real_maps != pool:
            raise ValueError(
                f"veto match {match_id!r}: the real veto map-name set "
                f"{sorted(real_maps)} does not exactly equal the "
                f"resolved pool {sorted(pool)}; teacher-forced "
                "bookkeeping cannot proceed on a pool/sequence mismatch"
            )

        remaining = set(pool)
        for row in ordered.itertuples(index=False):
            action = row.action
            true_map = config.normalize_map_name(row.map_name)
            acting_team_id = row.team_id
            # Opponent resolution: the other id of the match's
            # {team1_id, team2_id} pair. The held-out builder's rows
            # carry those columns (HELD_OUT_VETO_COLUMNS); hand-built
            # minimal frames may not, in which case the opponent is
            # left None (scoring never needs it; only the ban-example
            # builder consumes it, and that path always feeds full
            # held-out rows).
            opponent_team_id: str | None = None
            if acting_team_id is not None:
                try:
                    team1_id = str(row.team1_id)
                    team2_id = str(row.team2_id)
                except AttributeError:
                    team1_id = None
                    team2_id = None
                if team1_id is not None:
                    acting_str = str(acting_team_id)
                    if acting_str == team1_id:
                        opponent_team_id = team2_id
                    elif acting_str == team2_id:
                        opponent_team_id = team1_id
                    else:
                        raise ValueError(
                            f"veto match {match_id!r} step "
                            f"{row.step_index}: acting team "
                            f"{acting_team_id!r} is neither of the "
                            f"match's {{team1_id, team2_id}} pair "
                            f"{sorted({team1_id, team2_id})}; the "
                            "opponent cannot be resolved"
                        )
            yield {
                "match_id": match_id,
                "step_index": row.step_index,
                "action": action,
                "acting_team_id": acting_team_id,
                "opponent_team_id": opponent_team_id,
                "sorted_remaining_maps": sorted(remaining),
                "date": row.date,
                "true_map": true_map,
            }
            remaining.remove(true_map)


def score_veto_steps(
    predictor_fn: VetoStepPredictorFn,
    held_out_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_pool=None,
    actions_to_score: set[str] | None = None,
) -> pd.DataFrame:
    """Teacher-forced per-step scoring of a predictor over held-out vetoes.

    A thin consumer of the shared teacher-forced walk
    :func:`_iter_teacher_forced_steps` (decision 11): for every yielded
    non-decider step (and, when ``actions_to_score`` is given, every
    yielded step whose ``action`` is in that set) it calls
    ``predictor_fn(acting_team_id, action, sorted_remaining_maps,
    date, matches_df, maps_df)`` (``sorted`` = alphabetical by
    normalized name, decision 3), validates the returned vector's
    length equals ``len(remaining)``, finds the true map's index
    within the sorted list, computes the cross-entropy via
    ``utils.scoring.log_loss``, and computes the map's rank
    (probability descending, ties ascending by name — decision 8).
    Decider steps are skipped entirely (decision 5), and steps whose
    action is not in ``actions_to_score`` are skipped from scoring too
    (but still consumed for ``remaining`` bookkeeping by the
    generator), which is what lets M27 score ban-only on the same full
    replay.

    Args:
        predictor_fn: Any callable satisfying
            :data:`VetoStepPredictorFn` (returns a probability
            distribution over the passed sorted ``remaining_maps``).
        held_out_df: The held-out veto-step table from
            :func:`build_held_out_veto_matches` (needs
            :data:`HELD_OUT_VETO_COLUMNS` — at minimum ``match_id``,
            ``step_index``, ``team_id``, ``action``, ``map_name``,
            ``date``).
        matches_df: The full materialised ``matches`` table, passed
            through to ``predictor_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``predictor_fn`` unchanged.
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves it per match from
            ``config.json`` via :meth:`utils.config.Config.era_as_of`
            on the match date's calendar date, exactly like
            ``simulate_veto``.
        actions_to_score: An optional set of actions to score (e.g.
            ``{"ban"}``); ``None`` (the default) scores every
            non-decider step, exactly M26's behavior — fully
            backward-compatible for the existing call sites. Steps
            whose action is not in the set are skipped from scoring
            like deciders, but are still consumed for ``remaining``
            bookkeeping by the shared walk.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SCORED_STEP_COLUMNS`
        (``match_id, step_index, action, n_remaining, cross_entropy,
        top1_correct, top3_correct``), one row per scored step in
        ``(match_id, step_index)`` order. ``cross_entropy`` is the
        per-step ``log_loss``, ``top1_correct``/``top3_correct`` are
        the per-step ranking booleans (decision 8), and
        ``n_remaining`` is the candidate-set size at that step.

    Raises:
        ValueError: If ``predictor_fn`` returns a vector whose length
            differs from ``len(remaining)`` (with the offending
            match/step named); if ``scored_df`` would be empty (no
            scoreable steps at all); if a match's real normalized
            map-name set does not exactly equal the resolved pool; if
            a non-decider acting team is neither of the match's two
            team ids; if the true map's probability is 0 or the
            returned vector fails the simplex validation (propagated
            from :func:`utils.scoring.log_loss`); if ``map_pool=None``
            and no era covers the match's date or the config is
            invalid (propagated from
            :meth:`utils.config.Config.era_as_of` /
            :func:`utils.config.load_config`); or if ``date`` is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.parse_query_date`).
        KeyError: If ``held_out_df`` lacks a required column or a
            required table column is missing (propagated from pandas
            and the predictor's own lookups).
        TypeError: If a match date is list-like (propagated from
            :func:`utils.asof.parse_query_date`).
    """
    rows: list[dict] = []
    for step in _iter_teacher_forced_steps(
        held_out_df, matches_df, maps_df, map_pool
    ):
        action = step["action"]
        if action == "decider":
            # Decision 5: forced, not chosen — nothing to score.
            continue
        if actions_to_score is not None and action not in actions_to_score:
            # Decision 11: score only the requested actions (e.g. bans
            # only); the generator still consumed this step's real map
            # for remaining bookkeeping.
            continue
        sorted_maps = step["sorted_remaining_maps"]
        probs = list(
            predictor_fn(
                step["acting_team_id"],
                action,
                sorted_maps,
                step["date"],
                matches_df,
                maps_df,
            )
        )
        if len(probs) != len(sorted_maps):
            raise ValueError(
                f"predictor_fn returned {len(probs)} probabilities "
                f"for match {step['match_id']!r} step "
                f"{step['step_index']} (action {action!r}, "
                f"{len(sorted_maps)} maps remaining); expected exactly "
                f"{len(sorted_maps)} aligned to the sorted "
                "remaining-maps order"
            )
        true_index = sorted_maps.index(step["true_map"])
        cross_entropy = scoring.log_loss(probs, true_index)
        # Rank: probability descending, ties broken ascending by
        # map name (one consistent secondary key — decision 8).
        order = sorted(
            range(len(sorted_maps)),
            key=lambda i: (-probs[i], sorted_maps[i]),
        )
        rank = order.index(true_index) + 1
        rows.append(
            {
                "match_id": step["match_id"],
                "step_index": step["step_index"],
                "action": action,
                "n_remaining": len(sorted_maps),
                "cross_entropy": cross_entropy,
                "top1_correct": rank == 1,
                "top3_correct": rank <= min(3, len(sorted_maps)),
            }
        )

    if not rows:
        raise ValueError(
            "score_veto_steps produced zero scored steps; the held-out "
            "table contains no non-decider veto actions"
        )
    return pd.DataFrame(rows, columns=SCORED_STEP_COLUMNS)


@dataclass(frozen=True)
class BanTrainingExample:
    """One teacher-forced training example for a per-step ban model (M27).

    The raw (unfeaturized) record of one real ban step, produced by
    :func:`build_ban_training_examples` from the shared teacher-forced
    replay: the acting team, its opponent, the candidate set the ban
    was made over, the as-of date, and the position of the real banned
    map within that set. ``remaining_maps`` is the alphabetically
    sorted (by normalized name) list of maps still in play at the step
    (decision 3's ordering convention), so ``true_map_index`` is the
    label a per-step softmax over ``remaining_maps`` is scored against
    (the index whose probability the cross-entropy objective raises).
    No featurization happens at this layer — that is the consuming
    model's job, per the module-boundary standard (decision 12).

    Attributes:
        acting_team_id: The banning team's stable id (a string from
            the resolved veto table's ``team_id`` column).
        opponent_team_id: The other team's stable id (the other of the
            match's ``{team1_id, team2_id}`` pair).
        remaining_maps: The sorted tuple of candidate map names still
            in play at this step (normalized, alphabetically ordered).
        date: The step's as-of date string (the match's own date).
        true_map_index: The index of the real banned map within
            ``remaining_maps``.
    """

    acting_team_id: str
    opponent_team_id: str
    remaining_maps: tuple[str, ...]
    date: str
    true_map_index: int


def build_ban_training_examples(
    held_out_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_pool=None,
) -> list[BanTrainingExample]:
    """Build the per-step ban training examples from a held-out veto table.

    Walks the shared teacher-forced replay
    (:func:`_iter_teacher_forced_steps`) and keeps only the rows whose
    ``action == "ban"``, returning one frozen
    :class:`BanTrainingExample` per ban step: the acting team, the
    opponent (the other of the match's ``{team1_id, team2_id}`` pair),
    the sorted remaining candidate list, the date, and
    ``true_map_index = remaining_maps.index(true_map)`` — the position
    of the real banned map within its own sorted candidate list, the
    label a per-step softmax is scored against (decision 12). The
    function returns raw ids/dates/candidate-lists only; featurization
    of each candidate map is the consuming model's job (decision 12's
    module-boundary note).

    Args:
        held_out_df: The held-out veto-step table from
            :func:`build_held_out_veto_matches` (needs
            :data:`HELD_OUT_VETO_COLUMNS` — at minimum ``match_id``,
            ``step_index``, ``team_id``, ``action``, ``map_name``,
            ``date``, plus ``team1_id``/``team2_id`` for opponent
            resolution).
        matches_df: The full materialised ``matches`` table, passed
            through to the shared walk (used only by the optional
            ``map_pool=None`` config-era resolution path).
        maps_df: The full materialised ``maps`` table, passed through
            to the shared walk unused.
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves it per match from
            ``config.json`` via :meth:`utils.config.Config.era_as_of`
            on the match date's calendar date, exactly like
            ``simulate_veto``.

    Returns:
        A list of :class:`BanTrainingExample` objects in
        ``(match_id, step_index)`` replay order, one per ban step of
        the held-out table. Never empty: a held-out table with no ban
        rows raises instead.

    Raises:
        ValueError: If the held-out table contains no ban rows at all;
            if a match's real normalized map-name set does not exactly
            equal the resolved pool; if a ban row's acting ``team_id``
            is neither of the match's ``team1_id``/``team2_id`` (an
            unresolvable opponent); if ``map_pool=None`` and no era
            covers the match's date or the config is invalid
            (propagated from :meth:`utils.config.Config.era_as_of` /
            :func:`utils.config.load_config`); or if ``date`` is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.parse_query_date`).
        KeyError: If ``held_out_df`` lacks a required column
            (propagated from pandas column indexing).
        TypeError: If a match date is list-like (propagated from
            :func:`utils.asof.parse_query_date`).
    """
    examples: list[BanTrainingExample] = []
    for step in _iter_teacher_forced_steps(
        held_out_df, matches_df, maps_df, map_pool
    ):
        if step["action"] != "ban":
            continue
        remaining_maps = step["sorted_remaining_maps"]
        examples.append(
            BanTrainingExample(
                acting_team_id=step["acting_team_id"],
                opponent_team_id=step["opponent_team_id"],
                remaining_maps=tuple(remaining_maps),
                date=step["date"],
                true_map_index=remaining_maps.index(step["true_map"]),
            )
        )
    if not examples:
        raise ValueError(
            "build_ban_training_examples produced zero examples; the "
            "held-out table contains no ban veto actions"
        )
    return examples


def build_veto_evaluation_report(scored_df: pd.DataFrame) -> dict:
    """Build the JSON-serializable per-arm veto evaluation report.

    A pure dict builder (no I/O): turns the scored table from
    :func:`score_veto_steps` into one arm of the head-to-head report.
    The aggregate block holds ``n_steps`` (row count), and the means of
    ``cross_entropy`` / ``top1_correct`` / ``top3_correct`` (top-1 and
    top-3 accuracy). ``by_step_index`` is a list of per-``step_index``
    dicts, one per distinct step index in ascending order, each with
    the same four fields computed on that index's rows — the
    "per-step" reading of the roadmap line (decision 9), and the
    breakdown that lets later milestones (M27+) read top-3 numbers
    without the ``n_remaining <= 3`` ceiling effect washing out early-
    step differences.

    Args:
        scored_df: The scored table from :func:`score_veto_steps`
            (needs :data:`SCORED_STEP_COLUMNS`).

    Returns:
        A dict with keys ``n_steps`` (int),
        ``mean_cross_entropy`` / ``top1_accuracy`` / ``top3_accuracy``
        (floats in ``[0, 1]`` for the accuracies, ``>= 0`` for the
        cross-entropy) and ``by_step_index`` (a list of per-index
        dicts, each ``{"step_index": int, "n_steps": int,
        "mean_cross_entropy": float, "top1_accuracy": float,
        "top3_accuracy": float}``). Every value is a plain
        str/int/float/list/dict, so the whole dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If ``scored_df`` is empty (a mean over zero steps
            is undefined) or contains a step without the required
            columns (propagated from pandas column indexing /
            ``itertuples``).
        KeyError: If ``scored_df`` lacks a :data:`SCORED_STEP_COLUMNS`
            column (propagated from pandas).
    """
    if len(scored_df) == 0:
        raise ValueError(
            "cannot build a veto evaluation report over zero scored steps"
        )

    def _block(subset: pd.DataFrame) -> dict:
        """Compute the four aggregate fields for a scored subset.

        Args:
            subset: A (possibly strict) subset of a
                :func:`score_veto_steps` table.

        Returns:
            A dict with ``n_steps`` / ``mean_cross_entropy`` /
            ``top1_accuracy`` / ``top3_accuracy``.

        Raises:
            ValueError: If ``subset`` is empty (propagated as a bare
                ``ValueError`` by the caller's own empty check for the
                top-level call; per-index subsets of a non-empty table
                are non-empty by construction since each index groups
                at least one row).
        """
        return {
            "n_steps": len(subset),
            "mean_cross_entropy": float(subset["cross_entropy"].mean()),
            "top1_accuracy": float(subset["top1_correct"].mean()),
            "top3_accuracy": float(subset["top3_correct"].mean()),
        }

    report = _block(scored_df)
    by_step_index: list[dict] = []
    for step_index in sorted(scored_df["step_index"].unique()):
        subset = scored_df[scored_df["step_index"] == step_index]
        entry = _block(subset)
        entry["step_index"] = int(step_index)
        by_step_index.append(entry)
    report["by_step_index"] = by_step_index
    return report


def build_veto_comparison_report(
    scored_greedy_df: pd.DataFrame,
    scored_baseline_df: pd.DataFrame,
) -> dict:
    """Build the head-to-head greedy-vs-baseline comparison report.

    Decision 9's "one artifact, two arms, a delta block" shape: takes
    the two scored tables from :func:`score_veto_steps` (produced on
    the *identical* held-out rows, in the identical order), validates
    they are row-aligned (same ``(match_id, step_index)`` pairs at the
    same positions — a misaligned comparison would silently pair two
    different steps' scores and corrupt every delta), and returns
    ``{"greedy": <report>, "baseline": <report>, "delta":
    {"mean_cross_entropy": ..., "top1_accuracy": ...,
    "top3_accuracy": ...}}`` where each delta is
    greedy-minus-baseline (a negative cross-entropy delta means M25 is
    better-calibrated; a positive accuracy delta means M25 ranks
    better). The report asserts no direction for the deltas — the
    actually-measured values are an empirical finding, not an assumed
    one.

    Args:
        scored_greedy_df: The :func:`score_veto_steps` table for the
            M25 greedy arm (needs :data:`SCORED_STEP_COLUMNS`).
        scored_baseline_df: The :func:`score_veto_steps` table for the
            frequency-baseline arm, same column requirements, scored on
            the identical held-out rows.

    Returns:
        A dict with keys ``greedy`` and ``baseline`` (each the
        :func:`build_veto_evaluation_report` dict for that arm) and
        ``delta`` (``{"mean_cross_entropy": float, "top1_accuracy":
        float, "top3_accuracy": float}``, each greedy-minus-baseline).
        Every value is a plain str/int/float/list/dict, so the whole
        dict is directly ``json.dumps``-serializable.

    Raises:
        ValueError: If the two tables have different row counts or
            differ in any ``(match_id, step_index)`` pair at the same
            position (the row-alignment contract, mirroring
            ``evaluation.temperature_calibration``'s guard); or if
            either table is empty (propagated from
            :func:`build_veto_evaluation_report`).
        KeyError: If either table lacks a :data:`SCORED_STEP_COLUMNS`
            column (propagated from pandas column indexing).
    """
    if len(scored_greedy_df) != len(scored_baseline_df):
        raise ValueError(
            f"scored tables have different row counts: greedy "
            f"{len(scored_greedy_df)} vs baseline "
            f"{len(scored_baseline_df)}; they must describe the same "
            "held-out veto steps"
        )
    greedy_ids = scored_greedy_df[list(_STEP_ID_COLUMNS)].to_numpy()
    baseline_ids = scored_baseline_df[list(_STEP_ID_COLUMNS)].to_numpy()
    mismatch_mask = greedy_ids != baseline_ids
    if mismatch_mask.any():
        idx = int(np.argmax(mismatch_mask.any(axis=1)))
        raise ValueError(
            "scored tables are not row-aligned: the held-out veto steps "
            f"differ at position {idx} "
            f"(greedy {tuple(greedy_ids[idx])!r} vs baseline "
            f"{tuple(baseline_ids[idx])!r}); score both arms on the "
            "identical build_held_out_veto_matches table"
        )

    greedy_report = build_veto_evaluation_report(scored_greedy_df)
    baseline_report = build_veto_evaluation_report(scored_baseline_df)
    return {
        "greedy": greedy_report,
        "baseline": baseline_report,
        "delta": {
            "mean_cross_entropy": (
                greedy_report["mean_cross_entropy"]
                - baseline_report["mean_cross_entropy"]
            ),
            "top1_accuracy": (
                greedy_report["top1_accuracy"]
                - baseline_report["top1_accuracy"]
            ),
            "top3_accuracy": (
                greedy_report["top3_accuracy"]
                - baseline_report["top3_accuracy"]
            ),
        },
    }


def build_veto_multi_arm_report(
    scored_by_arm: dict[str, pd.DataFrame],
    baseline_arm: str,
) -> dict:
    """Build the N-arm comparison report over identically-scored tables.

    Decision 13's generalisation of :func:`build_veto_comparison_report`
    to any number of arms (M27 compares three: the conditional-logit
    model, the M25 greedy arm, and the frequency baseline): takes one
    scored table per arm from :func:`score_veto_steps` (all produced on
    the *identical* held-out rows, in the identical order), validates
    they are all row-aligned (same ``(match_id, step_index)`` pairs at
    the same positions across every arm — a misaligned comparison
    would silently pair two different steps' scores and corrupt every
    delta), and returns ``{arm_name: <report>, "deltas_vs_<baseline>":
    {arm_name: {...}}}`` where each arm's block is the
    :func:`build_veto_evaluation_report` dict for that arm and the
    delta block holds one ``{mean_cross_entropy, top1_accuracy,
    top3_accuracy}`` dict per *non-baseline* arm, each arm-minus-
    baseline (a negative cross-entropy delta means the arm is
    better-calibrated; a positive accuracy delta means the arm ranks
    better). The report asserts no direction for the deltas — the
    actually-measured values are an empirical finding, not an assumed
    one.

    Args:
        scored_by_arm: A dict mapping each arm's name to its
            :func:`score_veto_steps` table (needs
            :data:`SCORED_STEP_COLUMNS`). At least two arms are
            required (a one-arm "comparison" is meaningless).
        baseline_arm: The arm every delta is measured against; must be
            a key of ``scored_by_arm``. The baseline arm's own report
            block appears like any other arm's, but it has no delta
            entry in the delta block.

    Returns:
        A dict with one key per arm name (each the
        :func:`build_veto_evaluation_report` dict for that arm) plus
        the key ``"deltas_vs_<baseline_arm>"`` mapping each non-
        baseline arm name to its ``{"mean_cross_entropy": float,
        "top1_accuracy": float, "top3_accuracy": float}`` dict of
        arm-minus-baseline deltas. Every value is a plain
        str/int/float/list/dict, so the whole dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If ``scored_by_arm`` has fewer than two arms; if
            ``baseline_arm`` is not a key of ``scored_by_arm``; if any
            two arms' scored tables have different row counts or
            differ in any ``(match_id, step_index)`` pair at the same
            position (the row-alignment contract, mirroring
            :func:`build_veto_comparison_report`'s guard); or if any
            arm's table is empty (propagated from
            :func:`build_veto_evaluation_report`).
        KeyError: If any arm's table lacks a
            :data:`SCORED_STEP_COLUMNS` column (propagated from pandas
            column indexing).
    """
    if len(scored_by_arm) < 2:
        raise ValueError(
            f"build_veto_multi_arm_report needs at least two arms to "
            f"compare, got {len(scored_by_arm)}"
        )
    if baseline_arm not in scored_by_arm:
        raise ValueError(
            f"baseline_arm {baseline_arm!r} is not a scored arm; got "
            f"arms {sorted(scored_by_arm)}"
        )

    arm_names = list(scored_by_arm)
    reference_ids = scored_by_arm[arm_names[0]][list(_STEP_ID_COLUMNS)].to_numpy()
    for name in arm_names[1:]:
        scored = scored_by_arm[name]
        if len(scored) != len(reference_ids):
            raise ValueError(
                f"scored tables have different row counts: "
                f"{arm_names[0]} {len(reference_ids)} vs {name} "
                f"{len(scored)}; they must describe the same held-out "
                "veto steps"
            )
        arm_ids = scored[list(_STEP_ID_COLUMNS)].to_numpy()
        mismatch_mask = arm_ids != reference_ids
        if mismatch_mask.any():
            idx = int(np.argmax(mismatch_mask.any(axis=1)))
            raise ValueError(
                "scored tables are not row-aligned: the held-out veto "
                f"steps differ at position {idx} ("
                f"{arm_names[0]} {tuple(reference_ids[idx])!r} vs "
                f"{name} {tuple(arm_ids[idx])!r}); score all arms on "
                "the identical build_held_out_veto_matches table"
            )

    arm_reports = {
        name: build_veto_evaluation_report(scored)
        for name, scored in scored_by_arm.items()
    }
    report = dict(arm_reports)
    baseline_report = arm_reports[baseline_arm]
    deltas_key = f"deltas_vs_{baseline_arm}"
    report[deltas_key] = {}
    for name, arm_report in arm_reports.items():
        if name == baseline_arm:
            continue
        report[deltas_key][name] = {
            "mean_cross_entropy": (
                arm_report["mean_cross_entropy"]
                - baseline_report["mean_cross_entropy"]
            ),
            "top1_accuracy": (
                arm_report["top1_accuracy"]
                - baseline_report["top1_accuracy"]
            ),
            "top3_accuracy": (
                arm_report["top3_accuracy"]
                - baseline_report["top3_accuracy"]
            ),
        }
    return report

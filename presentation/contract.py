"""The wire data contract for the presentation artifact (P1).

Defines the *post-reshaping, JSON-native* shape of the artifact the
export driver will eventually write (roadmap P5), as a set of
:class:`typing.TypedDict` wire types mirroring §6 of
``presentation_layer.md`` field-for-field, plus the
:func:`validate_artifact` entry point that validates a candidate
artifact dict against the checked-in JSON Schema
(:data:`presentation/contract.schema.json`).

The contract rules from §6 are encoded mechanically in the schema and
kept in lockstep with the Python side by a drift-guard test
(``tests/test_contract.py``):

- probability-vector order is fixed and never re-sorted (the schema
  fixes per-map vectors to exactly length 4 in ``OUTCOME_LABELS``
  order, but — like §6 itself — deliberately does not *name* that
  order, since a JSON Schema cannot express "this exact vocabulary
  order" and the frontend indexes positionally);
- floats are shipped as JSON numbers, never formatted strings (every
  probability/score field is ``type: "number"``);
- null means "not computed", never zero (``interval_low`` /
  ``interval_high`` / ``greedy_rank`` accept ``null``, and the
  validator never substitutes zeros for it);
- structurally-constant fields are omitted, not shipped null
  (:class:`RankedVeto` declares neither ``veto_sensitivity`` nor a
  nested ``top_vetos``, and closes ``additionalProperties`` so a
  reshaping bug that accidentally re-includes either fails validation).

Module placement (per the Conventions ruling in
``presentation_roadmap.md``): ``presentation/`` is a new top-level DAG
node. This module imports nothing from ``drivers/`` (P1 defines the
wire shape, which is deliberately decoupled from
``drivers/predict.py``'s object-shaped dataclasses — see
``presentation_roadmap.md`` P1's "TypedDict vs dataclass" note), and
nothing from ``features/`` or ``models/``; it depends only on the
standard library and ``jsonschema``. Later milestones (P2/P3) are the
ones that import the permitted ``drivers.predict`` surface
(``PredictionResult``, ``PerMapPrediction``, ``SeriesPrediction``,
``VetoSensitivity``, ``RankedVetoPrediction``, ``Predictor``,
``make_predictor``), and
``tests/test_module_boundaries.py`` asserts that edge from this first
commit onward.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import jsonschema

# The checked-in schema, colocated with this module and resolved
# relative to __file__ so loading never depends on the process CWD.
_SCHEMA_PATH = Path(__file__).with_name("contract.schema.json")


class Knobs(TypedDict):
    """The prediction knobs that produced an artifact (self-describing rule).

    Attributes:
        n_samples: Number of ancestral veto draws used by the M31
            overall-series sampling pass.
        seed: The RNG seed used for that sampling pass.
        ci_level: The confidence level used for the bootstrap
            intervals (e.g. ``0.9``).
        top_n: The number of top-ranked veto sequences listed under
            each fixture's ``top_vetos``.
    """

    n_samples: int
    seed: int
    ci_level: float
    top_n: int


class MetricsSummary(TypedDict):
    """Headline evaluation numbers for the methodology page.

    Intentionally empty at P1 (decision D): P5 emits a stub object and
    P21a populates the real fields later without a schema change, so
    the JSON Schema types this as a plain object with no declared
    properties and ``additionalProperties`` permitted. Any stub object
    P5 emits validates.
    """


class Team(TypedDict):
    """One side of a fixture: the stable team id plus its display name.

    Attributes:
        id: The stable ``team_id`` string (the vocabulary
            ``features``/``utils.asof`` consume, e.g. ``"397"``).
        name: The derived display name (resolved by P3).
    """

    id: str
    name: str


class VetoAction(TypedDict):
    """One step of a veto sequence (§6).

    Attributes:
        step_index: The 0-based position of this action in the veto
            sequence.
        team: The acting team's stable ``team_id``.
        team_name: The acting team's derived display name (resolved by
            P3, but typed here so P5 never has to change the schema).
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        map_name: The chosen map's normalized name.
    """

    step_index: int
    team: str
    team_name: str
    action: Literal["ban", "pick", "decider"]
    map_name: str


class PerMap(TypedDict):
    """One played map's prediction (§6).

    Attributes:
        map_name: The played map's normalized name.
        probabilities: The four map-outcome probabilities in fixed
            ``OUTCOME_LABELS`` order (A-regulation, A-OT, B-OT,
            B-regulation) — always exactly length 4.
        interval_low: The four lower band endpoints in the same order,
            or ``None`` meaning "not computed" (no bootstrap
            replicates) — never a fabricated ``[0, 0, 0, 0]``.
        interval_high: The four upper band endpoints, or ``None``
            alongside ``interval_low``.
        n_games_backing: ``min(games_a, games_b)`` — the weaker side's
            as-of, map-specific game count.
        p_a_wins_map: Derived — ``probabilities[0] + probabilities[1]``.
        p_overtime: Derived — ``probabilities[1] + probabilities[2]``.
    """

    map_name: str
    probabilities: list[float]
    interval_low: list[float] | None
    interval_high: list[float] | None
    n_games_backing: int
    p_a_wins_map: float
    p_overtime: float


class MapLeverage(TypedDict):
    """One map's veto-leverage attribution (§5.4).

    Attributes:
        map_name: The attributed map's normalized name.
        p_played: Probability-weighted share of the top-N mass in
            which this map is played.
        p_a_given_played: Probability-weighted ``p_a_wins_series``
            among entries that play this map.
        p_a_given_not: Probability-weighted ``p_a_wins_series`` among
            entries that do not play this map.
        swing: ``p_a_given_played - p_a_given_not``.
        flips_favorite: Whether the attributed map flips which side is
            favoured.
        n_vetos_backing: How many top-N entries played this map.
    """

    map_name: str
    p_played: float
    p_a_given_played: float
    p_a_given_not: float
    swing: float
    flips_favorite: bool
    n_vetos_backing: int


class VetoSensitivity(TypedDict):
    """The structural (M37) spread summary across sampled veto sequences.

    Attributes:
        unweighted_band_low: The per-category lower band endpoints
            (length ``best_of + 1``).
        unweighted_band_high: The per-category upper band endpoints.
        band_widths: The per-category ``hi - lo`` widths.
        mean_band_width: The mean of ``band_widths`` — the scalar
            headline "how much does the veto move the series outcome".
        weighted_mean: The per-category weighted means over the M31
            sample rows.
        weighted_variance: The per-category weighted population
            variances about the weighted mean.
    """

    unweighted_band_low: list[float]
    unweighted_band_high: list[float]
    band_widths: list[float]
    mean_band_width: float
    weighted_mean: list[float]
    weighted_variance: list[float]


class OverallResult(TypedDict):
    """The top level of ``predict()``'s result, after §4.5 reshaping.

    Attributes:
        series_probabilities: The ``best_of + 1`` scoreline
            probabilities in ``outcome_order`` order (M31 sampled).
        p_a_wins_series: Derived — the summed probability of every
            scoreline where A wins more maps than B.
        veto_sensitivity: The structural spread summary (a real object
            on the top-level result, never null there).
        per_map: The greedy veto's played maps, in play order.
        greedy_veto: The 7-step greedy veto (not displayed by default).
        greedy_rank: The greedy veto's 1-based rank within the top-N
            listing, or ``None`` when it is not in the displayed
            top-N (or the §2 alternative display is not chosen) —
            ``None`` means "not computed", never a fabricated 0.
    """

    series_probabilities: list[float]
    p_a_wins_series: float
    veto_sensitivity: VetoSensitivity
    per_map: list[PerMap]
    greedy_veto: list[VetoAction]
    greedy_rank: float | None


class RankedVeto(TypedDict):
    """One entry of the top-N exact veto listing (§6).

    Structurally-constant fields are omitted, not shipped null: this
    type deliberately has **no** ``veto_sensitivity`` (always null
    here, G6) and **no** nested ``top_vetos`` (always empty here, G1),
    and the JSON Schema closes ``additionalProperties`` on this object
    so a reshaping bug that re-includes either field fails validation.

    Attributes:
        rank: The 1-based position in the top-N listing (integer >= 1).
        veto_probability: The exact joint probability of this
            enumerated sequence.
        actions: The 7 veto steps in step order (always exactly 7).
        per_map: This veto's played maps, in play order.
        series_probabilities: The exact M30 recursion scoreline
            distribution.
        p_a_wins_series: Derived — the summed probability of every
            scoreline where A wins more maps than B.
        favorite_flips: Derived — whether this entry's
            ``p_a_wins_series`` lands on the opposite side of 0.5 from
            the overall ``p_a_wins_series``.
    """

    rank: int
    veto_probability: float
    actions: list[VetoAction]
    per_map: list[PerMap]
    series_probabilities: list[float]
    p_a_wins_series: float
    favorite_flips: bool


class Fixture(TypedDict):
    """One upcoming fixture's self-contained prediction record (§6).

    Attributes:
        match_id: The fixture's match id.
        event: The event display string.
        scheduled_at: The fixture's scheduled start time.
        best_of: One of ``"Bo1"``, ``"Bo3"`` or ``"Bo5"``.
        best_of_int: The parsed map count (``1``/``3``/``5``).
        bo5_unvalidated: Derived (§4.6) — whether this is a Bo5
            prediction backed by too few observations to trust.
        team_a: Team A (id + display name).
        team_b: Team B (id + display name).
        outcome_order: The ``best_of + 1`` terminal ``(a_wins,
            b_wins)`` scorelines, hoisted once per fixture (§4.5).
        scoreline_labels: Derived display labels parallel to
            ``outcome_order``.
        overall: The top-level result.
        top_vetos: The top-N ranked veto listing.
        coverage_mass: Derived — the summed ``veto_probability`` over
            the top-N listing.
        map_leverage: The map-leverage attribution table (§5.4).
        narrative: The deterministic template narrative (§10); P5
            emits ``""`` until P20/P21a fill it (decision D).
    """

    match_id: str
    event: str
    scheduled_at: str
    best_of: Literal["Bo1", "Bo3", "Bo5"]
    best_of_int: Literal[1, 3, 5]
    bo5_unvalidated: bool
    team_a: Team
    team_b: Team
    outcome_order: list[list[int]]
    scoreline_labels: list[str]
    overall: OverallResult
    top_vetos: list[RankedVeto]
    coverage_mass: float
    map_leverage: list[MapLeverage]
    narrative: str


class Artifact(TypedDict):
    """The top-level presentation artifact (§6).

    Attributes:
        generated_at: ISO-8601 export run time (equal to the as-of
            date used for every prediction in the artifact).
        model_version: The git SHA of the training run.
        dataset_version: The dataset version (e.g. ``"v1"``).
        knobs: The prediction knobs that produced the artifact.
        intervals_present: Derived — whether D10's auto-load found
            bootstrap replicates (so the frontend can switch
            presentation wholesale rather than per-field).
        fixtures: The list of per-fixture prediction records.
        metrics: The headline evaluation numbers (loosely-typed stub
            at P1, populated by P21a).
    """

    generated_at: str
    model_version: str
    dataset_version: str
    knobs: Knobs
    intervals_present: bool
    fixtures: list[Fixture]
    metrics: MetricsSummary


# Maps each wire TypedDict to its JSON-Schema definition name so the
# drift-guard test can cross-check the two field sets mechanically
# rather than by manual review. Kept here (not in the test) so adding a
# wire type updates the check in the same edit that adds the type.
WIRE_TYPE_DEFINITIONS: dict[type, str] = {
    Artifact: "Artifact",
    Fixture: "Fixture",
    Knobs: "Knobs",
    Team: "Team",
    MetricsSummary: "MetricsSummary",
    OverallResult: "OverallResult",
    VetoSensitivity: "VetoSensitivity",
    RankedVeto: "RankedVeto",
    VetoAction: "VetoAction",
    PerMap: "PerMap",
    MapLeverage: "MapLeverage",
}


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Load and parse the checked-in JSON Schema once.

    Reads :data:`_SCHEMA_PATH` (``presentation/contract.schema.json``,
    resolved relative to this module's own file so the load never
    depends on the process working directory), parses it as JSON, and
    caches the result so every subsequent call in the process returns
    the same parsed dict without re-reading the file.

    Returns:
        The parsed JSON Schema as a nested ``dict`` (a draft-07 schema
        whose root is a ``$ref`` into its own ``definitions``).

    Raises:
        FileNotFoundError: If ``contract.schema.json`` is missing next
            to this module — propagated unchanged from the file read.
        json.JSONDecodeError: If ``contract.schema.json`` is not valid
            JSON — propagated unchanged from :func:`json.loads`.
        OSError: If the file exists but cannot be read (e.g. a
            permission error) — propagated unchanged from the read.
    """
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_artifact(artifact: dict) -> None:
    """Validate a presentation artifact dict against the wire schema.

    Loads the schema once (via :func:`load_schema`, whose result is
    cached), then validates ``artifact`` against it with
    :func:`jsonschema.validate`. Validation is **fail-fast**: the first
    schema violation raises immediately, and no attempt is made to
    aggregate multiple violations into a report (the simplest behavior,
    and sufficient until a concrete need for multi-error reports
    exists).

    Args:
        artifact: The candidate artifact dict to validate (the
            export driver's assembled ``Artifact`` wire dict).

    Returns:
        None when the artifact is valid.

    Raises:
        jsonschema.exceptions.ValidationError: On the first schema
            violation (e.g. a missing required field, a wrong enum
            value, a probability shipped as a string, a
            length-!=4 probability vector, or an unexpected
            ``veto_sensitivity``/``top_vetos`` key on a
            ``RankedVeto``) — propagated unchanged from
            :func:`jsonschema.validate`, so callers get the library's
            descriptive message with the failing path.
        FileNotFoundError: If the schema file cannot be found on the
            first load — propagated from :func:`load_schema`.
        json.JSONDecodeError: If the schema file is not valid JSON —
            propagated from :func:`load_schema`.
        OSError: If the schema file cannot be read — propagated from
            :func:`load_schema`.
    """
    jsonschema.validate(artifact, load_schema())

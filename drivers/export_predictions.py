"""P5 export driver: build the presentation artifact from ``fixtures.json``.

Reads the hand-maintained fixture list (decision A:
``fixtures.json``, by default at the repo root — see
:data:`DEFAULT_FIXTURES_PATH`), builds **one**
:class:`drivers.predict.Predictor`, calls
``predict(...)`` once per fixture, runs the P2–P4 derivation and
reshaping over each result, assembles each
:class:`presentation.contract.Fixture` and the top-level
:class:`presentation.contract.Artifact`, validates the whole thing
against P1's JSON Schema, and writes it atomically to
``<output-dir>/<version>/predictions.json`` (D16). The exported file is
the §6 wire shape: one top-level object with the seven required
``Artifact`` keys, and one self-contained per-fixture record under
``fixtures``.

**The one-library-call-per-fixture shape (§5.1).** One ``Predictor`` is
constructed once (so the materialised tables and fitted artifacts load
once), then each fixture is predicted, derived, reshaped and assembled
independently: :func:`presentation.reshape.reshape_fixture_core` (P3) →
:func:`presentation.leverage.compute_map_leverage` (P4) → the P5-owned
identity keys → :func:`build_artifact` → :func:`write_artifact`. There
is no per-fixture ``Predictor`` rebuild and no retained result list
beyond the assembled fixtures themselves (D12's ``intervals_present``
is a running OR inside the loop).

**Two library-vs-CLI traps this driver must not fall into (§5.1/§14).**

- It must **not** call :func:`drivers.predict.make_top_vetos_fn` (the
  standalone M39.2 factory): that path performs no D10 auto-load, so
  its per-veto ``per_map`` intervals are always null — silently
  dropping every interval from the ranked listing while the overall
  result still carries them.
- It must **not** use :func:`drivers.predict.Predictor`'s CLI
  ``--stream`` JSONL mode: that would force a pointless
  dataclass→JSON→dict round-trip (``PredictionResult.to_dict`` then
  re-parse) just to feed the reshapers, which consume the dataclasses
  directly.

**``map_pool`` and ``bootstrap_models`` (D10).** Every call is
``predictor.predict(team_a, team_b, best_of, None, as_of_iso,
top_n=args.top_n)`` — ``map_pool=None`` resolves the era pool from
``config.json`` for the as-of date, and ``bootstrap_models`` is never
passed, so :class:`drivers.predict.Predictor`'s ``None`` default
triggers the D10 auto-load of ``ordinal_bootstrap_replicates.json``.
``n_samples``/``seed``/``ci_level`` are construction knobs;
``top_n`` is a per-call knob (D10).

**Failure taxonomy (D13/D14, the inverse of ``predict.py``'s policy).**

- **Fatal, before any fixture** (propagate unchanged): anything raised
  while reading ``fixtures.json`` at file level (a missing file's
  ``FileNotFoundError``, or a malformed top level's
  :class:`FixtureFileError`), loading ``matches.parquet``, or
  constructing the ``Predictor`` — the D10 staleness guard
  ``ValueError``, the temperature/base guard ``ValueError``, a missing
  fitted artifact's ``FileNotFoundError``.
- **Per-fixture** (logged at ERROR with the match id, excluded, loop
  continues): an invalid fixture record (:class:`InvalidFixtureError`),
  a team id absent from the name map (D7, the §4.2 case —
  :class:`presentation.reshape.UnknownTeamError`, a ``ValueError``
  subclass), and ``ValueError`` / ``KeyError`` /
  :class:`utils.config.ConfigError` raised out of ``predict()`` /
  ``reshape_*`` / ``compute_map_leverage`` for that fixture.
- **Never caught**: ``TypeError``, ``AttributeError`` and everything
  else — programming errors, not data failures; swallowing them
  per-fixture would ship a silently thin artifact.

  The caught tuple is :data:`PER_FIXTURE_ERRORS`.

- **Threshold abort**: when ``failed / attempted > --max-failure-fraction``
  (strictly greater; default 0.5 tolerates exactly half) the run raises
  :class:`ExportAbortedError` **before** validation/write, so nothing
  is written (§5.3). ``attempted == 0`` (empty or fully-past list) is
  **not** an abort — it writes a valid artifact with ``"fixtures":
  []``, decision H's off-season state.

**Prerequisite tables/artifacts.** ``matches.parquet``,
``maps.parquet``, ``player_map_stats.parquet`` and the four fitted
``*_model.json`` artifacts for the requested version (i.e.
``materialize.py`` and the four training drivers have been run); the
optional ``ordinal_bootstrap_replicates.json`` is auto-loaded when
present (D10). ``model_provenance.json`` is read softly (D17) via
:func:`drivers.model_provenance.read_model_version`.

**Decision I's two version fields.** ``dataset_version`` is the
``--version`` flag verbatim; ``model_version`` is the sidecar's
``git_sha`` (or ``"unstamped"``), never the export's own ``HEAD``.

**Future-fixture limitation (read before running).** ``predict()``
currently resolves ``event_stage`` by an exact ``(team_a, team_b,
as_of)`` match lookup against ``matches.parquet`` (via
``models/_shared._match_id_for``): it requires **exactly one**
``matches`` row whose ``(team1_id, team2_id, date)`` equals the
queried pair and the run's as-of date. D8 shares one ``as_of`` across
every fixture in a run, and every match in the corpus has a distinct
timestamp, so **at most one distinct team pair can be exported per
run** — only fixtures already present in ``matches.parquet`` at
exactly the run's as-of date can be predicted. The shipped
``fixtures.json`` is therefore a **shape template**, not a runnable
example: invoking the driver on it with default flags fails every
record and aborts (D14). The underlying blocker is a modelling
decision in ``models/`` (see the presentation roadmap's "Notes and
deferrals") and is out of scope for this driver.

**Exit codes.** ``0`` — always. The hard failures are raises instead,
mirroring the rest of ``drivers/``'s raise-for-invariant-break
doctrine.

**Observability (P6).** Three diagnostics, all emitted as log lines and
never written into the artifact (§8):

- **Timing.** ``main()`` times the run end to end and the single
  ``Predictor`` construction; :func:`export_fixtures` times each
  fixture's ``predict()`` call and its ``build_fixture`` assembly and
  logs one INFO line per successful fixture plus an aggregate line.
  These are the driver's own honest boundaries: the M31 sampling, the
  ranked-entry construction and the ``n_games_backing`` feature
  lookups all happen *inside one* ``predict()`` call, so the driver
  cannot time them separately. That inner split is measured
  out-of-band with ``cProfile`` against the unmodified
  ``drivers/predict.py`` (decision E forbids editing it) — no
  profiling code or flag ships here.
- **Null-interval counting** (:func:`count_null_interval_maps`, D19):
  counts a fixture's overall ``per_map`` entries whose
  ``interval_low`` and ``interval_high`` are both ``None``. Reported
  at WARNING (naming ``train_bootstrap_replicates.py`` as the fix)
  when the count is nonzero, else INFO (D19).
- **Coverage diagnostic** (:func:`coverage_diagnostic`, D20): the §8
  reconciliation gap between the coverage-weighted average of the
  ranked entries' ``p_a_wins_series`` and the overall value, logged at
  INFO per fixture and never displayed or exported. No threshold
  constant exists (D21) — a human reads the line.

**Design decisions D1–D22 (recorded here, do not silently change).**

- **D1.** Two modules, not one: this module owns the export;
  ``drivers/model_provenance.py`` owns the sidecar's filename, keys,
  reader and writer (P22 needs an invocable entry point; burying the
  stamping CLI inside this ``main()`` would couple them).
- **D2.** ``fixtures.json`` is a top-level object with one required
  ``"fixtures"`` key holding a list of records, so P17/P18 can add
  file-level metadata without a shape change. Each record requires
  exactly six non-empty-string keys (``match_id``, ``event``,
  ``scheduled_at``, ``best_of``, ``team_a_id``, ``team_b_id``);
  unknown extra keys are ignored (P17 may carry its own fields
  through), while a *missing* required key is an error.
- **D3.** ``best_of`` is the ``"Bo<N>"`` string; ``best_of_int`` is
  derived via the public :data:`BEST_OF_MAP_COUNT` (an unknown value is
  a per-record validation error, not a ``KeyError``).
- **D4.** Fixture identity is P5-owned and assembled at exactly one
  dict-literal site (:func:`build_fixture`): seven keys from P3 + one
  from P4 + seven from P5 = the fifteen the schema requires.
- **D5.** ``bo5_unvalidated = (best_of_int == 5)`` — every Bo5 carries
  the flag, no Bo1/Bo3 does; it is deliberately not wired to a corpus
  count (see assumption A3).
- **D6.** Team names come from ``matches.parquet`` loaded once via
  ``evaluate.load_matches_table`` → ``reshape.build_team_name_map``,
  beside the single ``Predictor`` construction (one extra read of a
  small table, not a reach into the predictor's closed-over frame).
- **D7.** Unknown teams are pre-flighted before ``predict()``:
  membership of both ids in the name map is checked first, so the
  §4.2 exclusion does not pay the ~45s predict call; the
  ``UnknownTeamError`` handler stays as the backstop for ids appearing
  only inside a veto sequence.
- **D8.** One timestamp for everything: a timezone-naive UTC
  ``datetime`` computed once in ``main()``, whose
  ``isoformat(timespec="seconds")`` is the ``as_of_date`` of every
  ``predict()`` call, the artifact's ``generated_at`` and decision H's
  drop reference (naive UTC because the corpus ``date`` column is naive
  UTC and ``utils.asof`` rejects tz-aware values).
- **D9.** Decision H: a fixture with ``scheduled_at < as_of`` (strictly)
  is dropped in the fixture **reader**, not the loop, so P18's scraped
  list inherits it; a fixture scheduled exactly at ``as_of`` is kept;
  the comparison is on parsed ``datetime``s, never strings; dropped
  fixtures are logged as a count and do **not** count toward the
  failure fraction.
- **D10.** ``map_pool=None`` and ``top_n`` are per-call; the rest are
  construction knobs. ``bootstrap_models`` is never passed (the D10
  auto-load must run).
- **D11.** ``knobs`` mirrors the four flags read back off the parsed
  args (self-describing-artifact rule).
- **D12.** ``intervals_present`` is OR-ed across fixtures over each
  result's overall ``per_map`` (ranked entries agree, G5); zero
  exported fixtures ⇒ ``False``.
- **D13.** The failure taxonomy above (see :data:`PER_FIXTURE_ERRORS`).
- **D14.** The threshold abort above (:class:`ExportAbortedError`).
- **D15.** Atomic write: validate before any bytes touch disk; mkstemp
  in the target directory; ``os.replace``; ``finally`` cleanup of the
  temp file if the replace never happened; on any raise the
  pre-existing artifact is untouched.
- **D16.** Output path ``<output-dir>/<version>/predictions.json``; no
  ``--out-path`` flag.
- **D17.** ``model_version`` from the decision-I sidecar via
  :func:`drivers.model_provenance.read_model_version` (soft
  ``"unstamped"`` + WARNING); the export's own ``HEAD`` is never used.
- **D18.** The sidecar's stamping CLI lives in
  ``drivers/model_provenance.py`` with honest-but-overridable defaults
  (``--git-sha`` → current ``HEAD``, ``--trained-at`` → naive-UTC now,
  ``--drivers`` → the five training drivers); P22 passes all three
  explicitly.
- **D19.** Null-interval counting runs over a fixture's *overall*
  ``per_map`` only (never the ranked entries): a record is counted
  when both ``interval_low`` and ``interval_high`` are ``None``,
  reported at WARNING (naming ``train_bootstrap_replicates.py``)
  when the count is nonzero, else INFO.
- **D20.** The §8 coverage diagnostic (:func:`coverage_diagnostic`)
  computes ``mass``, ``weighted``, ``overall`` and ``gap`` over the
  ranked entries; zero ``mass`` returns ``None`` (never a
  ``ZeroDivisionError``, never a fabricated ``0.0`` gap).
- **D21.** No gap-threshold constant exists: the diagnostic is logged
  for a human to read, not compared against a programmatic limit.
- **D22.** The summary line is extended append-only: the five
  pre-existing fields keep their text and order, with
  ``null_interval_maps``, ``predict_seconds`` and ``elapsed_seconds``
  appended after ``dataset_version``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from drivers import evaluate, model_provenance, predict
from presentation import contract, derived, leverage, reshape
from utils.config import ConfigError
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# D3: the fixed best_of -> played-map-count lookup. Public because it
# is the fifth independent "Bo<N>"-parsing duplicate the repo already
# flags as a future housekeeping item; predict.py's own
# _BEST_OF_MAP_COUNT is private and must not be imported.
BEST_OF_MAP_COUNT = {"Bo1": 1, "Bo3": 3, "Bo5": 5}

# D16: the exported artifact's filename under <output-dir>/<version>/.
ARTIFACT_FILENAME = "predictions.json"

# A1: the hand-maintained fixture list lives at the repo root (not
# under data/), resolved relative to the process working directory like
# every other driver path; --fixtures lets P18 point elsewhere.
DEFAULT_FIXTURES_PATH = "fixtures.json"

# D14: abort only on a *majority* failure by default (a couple of
# unknown teams in a hand-maintained list still produce an artifact).
DEFAULT_MAX_FAILURE_FRACTION = 0.5

# D13: the one reviewable line holding the per-fixture failure policy —
# ValueError (incl. UnknownTeamError) / KeyError / ConfigError are data
# failures; TypeError / AttributeError and everything else are
# programming errors and are never swallowed.
PER_FIXTURE_ERRORS = (ValueError, KeyError, ConfigError)


class InvalidFixtureError(ValueError):
    """Raised when a single fixture record fails D2/D3 validation.

    One bad *record* (a missing required key, a blank value, an unknown
    ``best_of``, an unparseable or timezone-aware ``scheduled_at``), as
    opposed to a bad *file* (:class:`FixtureFileError`).

    Raised by:
        :func:`parse_fixture_record`.

    Caught by:
        :func:`read_fixtures`, which logs the record at ERROR, counts
        it toward the invalid-record stats, and continues with the next
        record (D13's per-fixture policy — a bad record never crashes
        the whole export).
    """


class FixtureFileError(ValueError):
    """Raised when the fixtures *file itself* cannot be parsed or shaped.

    A file-level problem — unparseable JSON, a non-object top level, a
    missing ``"fixtures"`` key, or a non-list ``"fixtures"`` value — as
    opposed to one bad record (:class:`InvalidFixtureError`).

    Raised by:
        :func:`read_fixtures`.

    Caught by:
        Nobody — it propagates unchanged out of ``main()`` (D13's
        "fatal, before any fixture" policy). A missing file raises
        ``FileNotFoundError`` instead (the repo's "create the file
        first" signal), which also propagates unchanged.
    """


class ExportAbortedError(RuntimeError):
    """Raised when too many attempted fixtures failed to predict (D14).

    ``failed / attempted > --max-failure-fraction`` after the fixture
    loop and before validation/write, so **nothing is written** — a
    systematic failure stays loud instead of producing a thin artifact
    (§5.3). ``attempted == 0`` (empty or fully-past list) is *not* an
    abort: it writes a valid ``"fixtures": []`` artifact.

    Raised by:
        :func:`main` after :func:`export_fixtures` returns.

    Caught by:
        Nobody — it propagates unchanged. The caller should fix the
        failing fixtures (or lower ``--max-failure-fraction`` only if
        the failures are expected) and re-run.
    """


@dataclass(frozen=True)
class FixtureSpec:
    """One parsed, validated fixture record (six D2 keys + derived map count).

    ``scheduled_at`` keeps the *original ISO string* (stripped), emitted
    verbatim on the wire rather than re-serialized from a parsed
    ``datetime``, while ``best_of_int`` is the derived played-map count
    (D3). ``team_a_id``/``team_b_id`` are the stable ``team_id``
    strings ``predict()`` consumes.

    Attributes:
        match_id: The fixture's match id (non-empty string).
        event: The event display string (non-empty string).
        scheduled_at: The fixture's scheduled start time as the
            original (stripped) ISO-8601 string.
        best_of: The series length as a ``"Bo<N>"`` string (one of
            ``"Bo1"``/``"Bo3"``/``"Bo5"``).
        best_of_int: The parsed map count (``1``/``3``/``5``).
        team_a_id: Team A's stable ``team_id`` string.
        team_b_id: Team B's stable ``team_id`` string.
    """

    match_id: str
    event: str
    scheduled_at: str
    best_of: str
    best_of_int: int
    team_a_id: str
    team_b_id: str


@dataclass(frozen=True)
class FixtureReadStats:
    """The per-run tallies produced by :func:`read_fixtures`.

    Carried back to ``main()`` so the summary line and D14's abort
    arithmetic can be computed without a second pass over the records.

    Attributes:
        n_records: The total number of records in the ``"fixtures"``
            list (before any drop/validation filtering).
        n_dropped_past: How many valid records decision H dropped
            because ``scheduled_at < as_of`` (these do **not** count
            toward the failure fraction, D13/D14).
        n_invalid: How many records failed D2/D3 validation (these
            *do* count toward the failure fraction, D14).
        invalid_ids: The match ids of the invalid records (or the
            placeholder ``"<unreadable>"`` where the id itself was
            missing/blank), for the log.
    """

    n_records: int
    n_dropped_past: int
    n_invalid: int
    invalid_ids: tuple[str, ...]


def parse_fixture_record(record) -> FixtureSpec:
    """Parse and validate one fixture record into a :class:`FixtureSpec`.

    Enforces D2/D3 on a single record: it must be a JSON object with
    all six required keys present, each a non-empty string after
    ``strip()``; ``best_of`` must map through :data:`BEST_OF_MAP_COUNT`
    (an unknown value raises naming the three legal values); and
    ``scheduled_at`` must parse via ``datetime.fromisoformat`` and be
    timezone-naive (``utils.asof`` rejects tz-aware values downstream,
    so catching it here gives the better message). Unknown extra keys
    are ignored (D2 — P17's scraper may carry its own fields through).
    Every violation raises :class:`InvalidFixtureError` with the
    offending ``match_id`` in the message when it is readable.

    Args:
        record: One element of the ``"fixtures"`` list (a JSON object).

    Returns:
        A :class:`FixtureSpec` with the six stripped string fields and
        the derived ``best_of_int``.

    Raises:
        InvalidFixtureError: For a non-object record, a missing
            required key, a blank/non-string value, an unknown
            ``best_of``, an unparseable ``scheduled_at``, or a
            timezone-aware ``scheduled_at``.
    """
    if not isinstance(record, dict):
        raise InvalidFixtureError(
            f"fixture record must be a JSON object, got "
            f"{type(record).__name__}"
        )
    raw_match_id = record.get("match_id")
    match_id = (
        raw_match_id.strip()
        if isinstance(raw_match_id, str) and raw_match_id.strip()
        else None
    )

    def _fail(message: str) -> InvalidFixtureError:
        """Build a prefixed :class:`InvalidFixtureError` for one record.

        Returns (rather than raises) the exception object, so the
        caller can ``raise _fail(...)`` and the message carries the
        record's ``match_id`` whenever it was readable. The prefix is
        ``fixture <match_id>`` in that case and ``fixture record``
        otherwise; ``match_id`` is the enclosing function's local, so
        the prefix reflects whatever was parsed from this record.

        Args:
            message: The record-specific failure description (already
                human-readable, e.g. "missing required key(s)").

        Returns:
            An :class:`InvalidFixtureError` with the prefixed message,
            ready to be raised by the caller.

        Raises:
            Nothing.
        """
        if match_id is not None:
            return InvalidFixtureError(f"fixture {match_id!r}: {message}")
        return InvalidFixtureError(f"fixture record: {message}")

    required = (
        "match_id",
        "event",
        "scheduled_at",
        "best_of",
        "team_a_id",
        "team_b_id",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise _fail(f"missing required key(s): {sorted(missing)}")

    values: dict[str, str] = {}
    for key in required:
        value = record[key]
        if not isinstance(value, str) or not value.strip():
            raise _fail(
                f"key {key!r} must be a non-empty string, got {value!r}"
            )
        values[key] = value.strip()

    best_of_int = BEST_OF_MAP_COUNT.get(values["best_of"])
    if best_of_int is None:
        raise _fail(
            f"best_of {values['best_of']!r} is not one of "
            f"{sorted(BEST_OF_MAP_COUNT)}"
        )

    scheduled = values["scheduled_at"]
    try:
        parsed = datetime.fromisoformat(scheduled)
    except ValueError as exc:
        raise _fail(
            f"scheduled_at {scheduled!r} is not an ISO-8601 datetime: "
            f"{exc}"
        ) from exc
    if parsed.tzinfo is not None:
        raise _fail(
            f"scheduled_at {scheduled!r} is timezone-aware; "
            "scheduled_at must be a naive datetime"
        )

    return FixtureSpec(
        match_id=values["match_id"],
        event=values["event"],
        scheduled_at=scheduled,
        best_of=values["best_of"],
        best_of_int=best_of_int,
        team_a_id=values["team_a_id"],
        team_b_id=values["team_b_id"],
    )


def _match_id_for_log(record) -> str:
    """Extract a readable match id from a (possibly malformed) record.

    Used by :func:`read_fixtures` to populate
    :attr:`FixtureReadStats.invalid_ids` without re-raising or
    depending on the record being valid. A missing, non-string or blank
    ``match_id`` yields the placeholder ``"<unreadable>"`` so the log
    still names *something*.

    Args:
        record: The raw (possibly malformed) fixture record.

    Returns:
        The stripped ``match_id`` string when present and non-blank,
        otherwise ``"<unreadable>"``.

    Raises:
        Nothing.
    """
    if isinstance(record, dict):
        value = record.get("match_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "<unreadable>"


def read_fixtures(
    path, as_of: datetime
) -> tuple[list[FixtureSpec], FixtureReadStats]:
    """Read, validate and past-drop the fixture list for one run.

    Reads and parses ``fixtures.json``; file-level problems raise
    :class:`FixtureFileError` (except ``FileNotFoundError``, which
    propagates unchanged as the repo's "create the file first"
    signal). It then iterates the ``"fixtures"`` records, collecting
    each :class:`InvalidFixtureError` into the returned stats rather
    than raising (each invalid record is logged at ERROR), and applies
    decision H's past-drop (D9): a record whose parsed ``scheduled_at``
    is strictly before ``as_of`` is dropped and counted (logged at INFO
    as a count), while a record scheduled exactly at ``as_of`` is kept.
    The surviving specs are returned in file order.

    Args:
        path: The fixtures JSON file path (e.g. ``"fixtures.json"``).
        as_of: The run's as-of cutoff (D8) as a naive ``datetime``;
            a fixture is dropped iff its parsed ``scheduled_at`` is
            strictly less than this.

    Returns:
        A ``(specs, stats)`` tuple: ``specs`` is the kept
        :class:`FixtureSpec` list in file order (past-dropped and
        invalid records removed); ``stats`` is the
        :class:`FixtureReadStats` tally.

    Raises:
        FileNotFoundError: If the fixtures file does not exist —
            propagated unchanged (the repo's "create the file first"
            signal), never wrapped.
        FixtureFileError: If the file cannot be read (``OSError``) or
            decoded (``json.JSONDecodeError``), or if the top level is
            not an object, the ``"fixtures"`` key is missing, or the
            ``"fixtures"`` value is not a list.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureFileError(
            f"could not read fixtures file {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise FixtureFileError(
            f"fixtures file root must be a JSON object, got "
            f"{type(raw).__name__}"
        )
    if "fixtures" not in raw:
        raise FixtureFileError(
            'fixtures file is missing the required "fixtures" key'
        )
    records = raw["fixtures"]
    if not isinstance(records, list):
        raise FixtureFileError(
            f'fixtures file "fixtures" must be a list, got '
            f"{type(records).__name__}"
        )

    specs: list[FixtureSpec] = []
    invalid_ids: list[str] = []
    dropped = 0
    for record in records:
        try:
            spec = parse_fixture_record(record)
        except InvalidFixtureError as exc:
            invalid_ids.append(_match_id_for_log(record))
            logger.error("excluding invalid fixture record: %s", exc)
            continue
        scheduled = datetime.fromisoformat(spec.scheduled_at)
        if scheduled < as_of:
            dropped += 1
            continue
        specs.append(spec)

    if dropped:
        logger.info(
            "dropped %d past fixture(s) (scheduled_at < as_of %s)",
            dropped,
            as_of.isoformat(timespec="seconds"),
        )

    return specs, FixtureReadStats(
        n_records=len(records),
        n_dropped_past=dropped,
        n_invalid=len(invalid_ids),
        invalid_ids=tuple(invalid_ids),
    )


def build_fixture(
    result,
    spec: FixtureSpec,
    team_names: Mapping[str, str],
) -> contract.Fixture:
    """Assemble one fixture's full fifteen-key wire record (D4).

    The single site where a :class:`presentation.contract.Fixture` is
    built: the seven P3-owned keys from
    :func:`presentation.reshape.reshape_fixture_core` (``team_a``,
    ``team_b``, ``outcome_order``, ``scoreline_labels``, ``overall``,
    ``top_vetos``, ``coverage_mass``), the one P4-owned key
    ``map_leverage`` from
    :func:`presentation.leverage.compute_map_leverage`, and the seven
    P5-owned identity keys (``match_id``, ``event``, ``scheduled_at``,
    ``best_of``, ``best_of_int``, ``bo5_unvalidated`` per D5, and
    ``narrative`` per decision D — ``""`` until P20/P21a fill it).

    Args:
        result: The top-level
            :class:`drivers.predict.PredictionResult` for the fixture.
        spec: The parsed :class:`FixtureSpec` supplying the identity
            keys and the derived ``best_of_int``.
        team_names: The ``team_id → display name`` mapping (see
            :func:`presentation.reshape.build_team_name_map`).

    Returns:
        The assembled :class:`presentation.contract.Fixture` dict with
        exactly the fifteen schema-required keys.

    Raises:
        UnknownTeamError: If a team id (fixture-level or inside a veto
            sequence) is absent from ``team_names`` — propagated
            unchanged from :func:`presentation.reshape.reshape_fixture_core`
            (a ``ValueError`` subclass, hence covered by
            :data:`PER_FIXTURE_ERRORS`).
        ValueError: If the top-level result's ``veto_sensitivity`` is
            ``None`` or a ranked entry's ``outcome_order`` mismatches
            the hoisted copy — propagated unchanged from
            :func:`presentation.reshape.reshape_fixture_core`.
        AttributeError: Propagated unchanged from the reshaping helpers
            for a malformed action record (never caught — D13).
    """
    return {
        **reshape.reshape_fixture_core(
            result, spec.team_a_id, spec.team_b_id, team_names
        ),
        "match_id": spec.match_id,
        "event": spec.event,
        "scheduled_at": spec.scheduled_at,
        "best_of": spec.best_of,
        "best_of_int": spec.best_of_int,
        "bo5_unvalidated": spec.best_of_int == 5,
        "map_leverage": leverage.compute_map_leverage(result.top_vetos),
        "narrative": "",
    }


def count_null_interval_maps(per_map: Sequence[contract.PerMap]) -> int:
    """Count the ``per_map`` entries whose intervals are both ``None`` (D19).

    §4.3's soft-missing case — the replicate artifact was never trained
    — shows up as per-map records carrying null intervals. This counts
    exactly those records over the fixture's *overall* ``per_map``
    only (D19): the ranked entries inherit the same closed-over
    bootstrap models (G5), so counting them too would multiply the
    same fact by ``top_n`` without adding information. A record is
    counted only when **both** ``interval_low`` and ``interval_high``
    are ``None``; a record with one band populated and the other
    ``None`` is malformed data and is *not* counted here (the D19 rule
    is strictly both-null).

    Args:
        per_map: The reshaped :class:`presentation.contract.PerMap`
            entries of one fixture's overall result
            (``fixture["overall"]["per_map"]``).

    Returns:
        The ``int`` count of entries where ``interval_low`` and
        ``interval_high`` are both ``None`` (``0`` for an empty
        sequence).

    Raises:
        KeyError: If an entry lacks the ``interval_low`` or
            ``interval_high`` key — propagated unchanged from the
            subscript, signalling a malformed fixture.
        TypeError: If an entry is not a subscriptable mapping (e.g. a
            non-dict object) — propagated unchanged from the
            subscript.
    """
    return sum(
        1
        for entry in per_map
        if entry["interval_low"] is None and entry["interval_high"] is None
    )


@dataclass(frozen=True)
class CoverageDiagnostic:
    """The §8 reconciliation numbers for one fixture (D20/D21).

    Computed by :func:`coverage_diagnostic` over the reshaped
    :class:`presentation.contract.Fixture` dict: the coverage-weighted
    average of the ranked entries' ``p_a_wins_series`` compared
    against the fixture's overall value. Diagnostic only — it never
    enters the artifact and is never displayed (§8), and there is no
    gap-threshold constant (D21).

    Attributes:
        mass: The summed ``veto_probability`` over the fixture's
            ``top_vetos`` listing — recomputed inside
            :func:`coverage_diagnostic` from the same entries it
            weights, so the diagnostic is self-contained. Equals
            ``fixture["coverage_mass"]`` by construction (D20).
        weighted: The coverage-weighted average of the ranked
            entries' ``p_a_wins_series`` —
            ``Σ (veto_probability × p_a_wins_series) / mass``.
        overall: The fixture's overall ``p_a_wins_series``
            (``fixture["overall"]["p_a_wins_series"]``), the baseline
            the weighted average is compared against.
        gap: ``weighted - overall`` — a large gap at high ``mass``
            indicates a real inconsistency between the sampled M31
            and exact M30 paths (§8).
    """

    mass: float
    weighted: float
    overall: float
    gap: float


def coverage_diagnostic(
    fixture: contract.Fixture,
) -> CoverageDiagnostic | None:
    """Compute the §8 coverage diagnostic for one fixture (D20).

    Implements D20's formula exactly, over the already-reshaped
    :class:`presentation.contract.Fixture` dict:

    .. code-block:: text

        mass     = Σ_i entry_i.veto_probability
        weighted = ( Σ_i entry_i.veto_probability × entry_i.p_a_wins_series
                   ) / mass
        gap      = weighted − overall.p_a_wins_series

    ``mass`` is recomputed from the same ranked entries it weights (so
    the helper is self-contained); by construction it equals
    ``fixture["coverage_mass"]``, and tests assert that equality rather
    than reading the key. When ``mass == 0.0`` (an empty listing, or
    an all-zero-probability listing) the helper returns ``None`` and
    the caller logs a single INFO noting the fixture had no coverage —
    never a ``ZeroDivisionError`` and never a fabricated ``0.0`` gap.
    The result is diagnostic only: it is never displayed and never
    exported (§8).

    Args:
        fixture: The reshaped :class:`presentation.contract.Fixture`
            dict (one entry of the artifact's ``fixtures`` list),
            carrying ``top_vetos`` (each with ``veto_probability`` and
            ``p_a_wins_series``) and ``overall.p_a_wins_series``.

    Returns:
        A :class:`CoverageDiagnostic` with the computed ``mass``,
        ``weighted``, ``overall`` and ``gap``, or ``None`` when the
        summed mass is ``0.0``.

    Raises:
        KeyError: If a ranked entry lacks ``veto_probability`` or
            ``p_a_wins_series``, or the fixture lacks
            ``overall.p_a_wins_series`` / ``top_vetos`` — propagated
            unchanged from the subscript, signalling a malformed
            fixture.
        TypeError: If a ranked entry is not a mapping or the summed
            probability is not numeric — propagated unchanged from the
            subscript / ``float`` conversion.
    """
    entries = fixture["top_vetos"]
    mass = sum(float(entry["veto_probability"]) for entry in entries)
    if mass == 0.0:
        return None
    weighted = sum(
        float(entry["veto_probability"]) * float(entry["p_a_wins_series"])
        for entry in entries
    ) / mass
    overall = float(fixture["overall"]["p_a_wins_series"])
    return CoverageDiagnostic(
        mass=mass,
        weighted=weighted,
        overall=overall,
        gap=weighted - overall,
    )


def export_fixtures(
    predictor,
    specs: Sequence[FixtureSpec],
    team_names: Mapping[str, str],
    *,
    as_of_iso: str,
    top_n: int,
    timings: dict[str, float] | None = None,
) -> tuple[list[contract.Fixture], int, bool]:
    """Run the per-fixture predict → derive → reshape → assemble loop.

    For each kept spec, in file order: pre-flights team membership (D7
    — both ids must be in ``team_names`` before the ~45s ``predict``
    call, else the fixture is excluded with a logged reason), calls
    ``predictor.predict(spec.team_a_id, spec.team_b_id, spec.best_of,
    None, as_of_iso, top_n=top_n)`` exactly once (D10 — ``map_pool=None``
    and no ``bootstrap_models``), assembles the fixture via
    :func:`build_fixture`, and ORs the fixture's overall ``per_map``
    intervals into the running ``intervals_present`` bool (D12). One
    ``try/except PER_FIXTURE_ERRORS`` wraps each iteration, logging at
    ERROR with the match id and the exception, incrementing the failure
    count and continuing (D13); ``TypeError``/``AttributeError`` and
    everything else propagate.

    **Timing (P6, log-only).** Inside the ``try``, the
    ``predictor.predict(...)`` call and the :func:`build_fixture` call
    are each timed separately with :func:`time.perf_counter`; after a
    successful append one INFO line is logged per fixture
    (``"fixture %s predicted in %.2fs, assembled in %.2fs"``) and both
    totals accumulate into local floats. At the end one aggregate INFO
    line reports the totals over *successful* fixtures
    (``"predicted %d fixture(s) in %.1fs total (mean %.1fs/fixture),
    assembled in %.1fs total"``), guarding the mean against a zero
    count. A failed fixture logs nothing extra — its existing ERROR
    line already names it (a partial duration for a failed prediction
    is noise). The return tuple and signature are otherwise unchanged;
    the totals are exposed only through the optional ``timings``
    out-dict (populated when not ``None``) so ``main()`` can put
    ``predict_seconds`` on the summary line.

    Args:
        predictor: The single :class:`drivers.predict.Predictor`
            instance built by ``main()``.
        specs: The kept :class:`FixtureSpec` list (already past-dropped
            and invalid-free, from :func:`read_fixtures`).
        team_names: The ``team_id → display name`` mapping.
        as_of_iso: The shared as-of ISO string for every ``predict``
            call (D8; keyword-only).
        top_n: The ``--top-n`` knob forwarded to every ``predict`` call
            (keyword-only).
        timings: An optional mutable out-dict (keyword-only) that, when
            not ``None``, is populated with ``predict_seconds`` and
            ``assemble_seconds`` — the accumulated wall seconds over
            successful fixtures — so ``main()`` can read them without a
            return-type change. ``None`` (the default) means the totals
            are only logged, not returned.

    Returns:
        A ``(fixtures, failures, intervals_present)`` tuple: the
        successfully assembled :class:`presentation.contract.Fixture`
        list (in spec order), the count of per-fixture failures caught
        and excluded, and the D12 OR-ed ``intervals_present`` bool
        (``False`` when nothing was exported).

    Raises:
        TypeError / AttributeError / anything outside
            :data:`PER_FIXTURE_ERRORS`: Propagated unchanged from
            ``predictor.predict`` or :func:`build_fixture` (D13's
            "never caught" policy).
    """
    fixtures: list[contract.Fixture] = []
    failures = 0
    intervals_present = False
    predict_seconds = 0.0
    assemble_seconds = 0.0
    for spec in specs:
        try:
            if (
                spec.team_a_id not in team_names
                or spec.team_b_id not in team_names
            ):
                missing = [
                    team_id
                    for team_id in (spec.team_a_id, spec.team_b_id)
                    if team_id not in team_names
                ]
                raise reshape.UnknownTeamError(
                    f"team_id(s) {missing} have no display name in the "
                    "matches table"
                )
            predict_started = time.perf_counter()
            result = predictor.predict(
                spec.team_a_id,
                spec.team_b_id,
                spec.best_of,
                None,
                as_of_iso,
                top_n=top_n,
            )
            predict_elapsed = time.perf_counter() - predict_started
            assemble_started = time.perf_counter()
            fixture = build_fixture(result, spec, team_names)
            assemble_elapsed = time.perf_counter() - assemble_started
            predict_seconds += predict_elapsed
            assemble_seconds += assemble_elapsed
            if derived.intervals_present(result.per_map):
                intervals_present = True
            fixtures.append(fixture)
            logger.info(
                "fixture %s predicted in %.2fs, assembled in %.2fs",
                spec.match_id,
                predict_elapsed,
                assemble_elapsed,
            )
        except PER_FIXTURE_ERRORS as exc:
            failures += 1
            logger.error("excluding fixture %s: %s", spec.match_id, exc)
    if fixtures:
        logger.info(
            "predicted %d fixture(s) in %.1fs total (mean %.1fs/fixture), "
            "assembled in %.1fs total",
            len(fixtures),
            predict_seconds,
            predict_seconds / len(fixtures),
            assemble_seconds,
        )
    else:
        logger.info(
            "predicted 0 fixture(s) in %.1fs total, "
            "assembled in %.1fs total",
            predict_seconds,
            assemble_seconds,
        )
    if timings is not None:
        timings["predict_seconds"] = predict_seconds
        timings["assemble_seconds"] = assemble_seconds
    return fixtures, failures, intervals_present


def build_artifact(
    fixtures: Sequence[contract.Fixture],
    *,
    generated_at: str,
    model_version: str,
    dataset_version: str,
    knobs: dict,
    intervals_present: bool,
) -> contract.Artifact:
    """Assemble the top-level seven-key artifact dict.

    Fills exactly the seven required :class:`presentation.contract.Artifact`
    keys (``additionalProperties: false`` on the schema means nothing
    else may be present): the assembled ``fixtures``, the shared
    ``generated_at``/``model_version``/``dataset_version``, the
    ``knobs`` block, the D12 ``intervals_present`` bool, and
    ``metrics={}`` per decision D — an empty stub object (P21a's to
    fill; the schema types ``MetricsSummary`` as an open object, so
    ``{}`` validates today).

    Args:
        fixtures: The assembled per-fixture wire records.
        generated_at: The shared ISO-8601 export/as-of timestamp
            (keyword-only; equals the ``as_of_date`` of every predict
            call, D8/§6).
        model_version: The decision-I model version string (keyword-only).
        dataset_version: The ``--version`` dataset version string
            (keyword-only).
        knobs: The four-key ``Knobs`` dict (keyword-only; D11).
        intervals_present: The D12 artifact-level boolean (keyword-only).

    Returns:
        The assembled :class:`presentation.contract.Artifact` dict with
        exactly the seven required keys.

    Raises:
        Nothing.
    """
    return {
        "generated_at": generated_at,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "knobs": knobs,
        "intervals_present": intervals_present,
        "fixtures": list(fixtures),
        "metrics": {},
    }


def write_artifact(artifact: dict, path) -> None:
    """Validate and atomically write the artifact (D15).

    Implements §5.3/§14's atomic-write invariant exactly, in order:
    validate the artifact against P1's schema **before** any byte
    touches disk (an invalid artifact never reaches disk at all); create
    the target directory; write the repo's
    ``json.dumps(artifact, indent=2, sort_keys=True) + "\n"``
    serialization to a temp file created **in the same directory**
    (``tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
    suffix=".tmp")``, so the rename cannot cross a filesystem), flushed
    and fsynced; then ``os.replace`` the temp file onto the target. The
    temp file is removed in a ``finally`` if anything between its
    creation and the replace raises, so a failed run leaves neither a
    half-written artifact nor a stray temp file — and, because the
    rename is atomic and the temp file never overwrites the target
    in place, **on any raise the pre-existing artifact is untouched**.

    Args:
        artifact: The assembled, JSON-native artifact dict to write.
        path: The destination file path (e.g.
            ``data/v1/predictions.json``); its parent directory is
            created if missing.

    Returns:
        None.

    Raises:
        jsonschema.exceptions.ValidationError: If ``artifact`` fails
            :func:`presentation.contract.validate_artifact` (raised
            before any temp file is created).
        OSError: If the directory cannot be created, the temp file
            cannot be created/written, or the ``os.replace`` fails
            (e.g. permissions/disk errors).
        TypeError: If ``artifact`` is not JSON-serializable (propagated
            from ``json.dumps``).
    """
    contract.validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the export_predictions.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with nine attributes, in declaration
        order: ``version`` (``str``, default ``"v1"``), ``output_dir``
        (``str``, default ``"data"``), ``fixtures`` (``str``, default
        :data:`DEFAULT_FIXTURES_PATH`), ``as_of_date`` (``str`` or
        ``None``; ``None`` means D8's run time — assumption A5),
        ``n_samples`` (``int``, default
        :data:`drivers.predict.DEFAULT_N_SAMPLES`), ``seed`` (``int``,
        default :data:`drivers.predict.DEFAULT_SEED`), ``ci_level``
        (``float``, default :data:`drivers.predict.DEFAULT_CI_LEVEL`),
        ``top_n`` (``int``, default
        :data:`drivers.predict.DEFAULT_TOP_N`) and
        ``max_failure_fraction`` (``float``, default
        :data:`DEFAULT_MAX_FAILURE_FRACTION`). The four knob flags'
        values land in the artifact's ``knobs`` block (D11), and their
        defaults are read from ``drivers.predict``'s constants — never
        re-typed as literals — so they cannot drift.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-float ``--max-failure-fraction``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the presentation artifact from fixtures.json: read "
            "the hand-maintained fixture list, build one Predictor, "
            "predict/derive/reshape/assemble one fixture at a time, "
            "validate the result against the wire contract, and write "
            "predictions.json atomically to <output-dir>/<version>/."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="input/output subdirectory name under --output-dir "
        "(default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory lives under "
            "(default: data)"
        ),
    )
    parser.add_argument(
        "--fixtures",
        default=DEFAULT_FIXTURES_PATH,
        help=(
            "path to the hand-maintained fixtures.json fixture list "
            f"(default: {DEFAULT_FIXTURES_PATH}); note: only fixtures "
            "already present in matches.parquet at exactly the run's "
            "as-of date can be predicted — the shipped fixtures.json is "
            "a shape template, not a runnable example"
        ),
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help=(
            "as-of cutoff for every predict call, the artifact's "
            "generated_at and decision H's past-drop reference "
            "(ISO-8601, naive UTC; default: the run time)"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=predict.DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per predict call by the M31 "
            "pipeline; lands in the artifact's knobs.n_samples "
            f"(default: {predict.DEFAULT_N_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=predict.DEFAULT_SEED,
        help=(
            "seed for the per-call numpy.default_rng; lands in the "
            f"artifact's knobs.seed (default: {predict.DEFAULT_SEED})"
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=predict.DEFAULT_CI_LEVEL,
        help=(
            "interval/spread level in (0, 1); lands in the artifact's "
            f"knobs.ci_level (default: {predict.DEFAULT_CI_LEVEL})"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=predict.DEFAULT_TOP_N,
        help=(
            "how many highest-probability enumerated veto sequences to "
            "rank into each fixture's top_vetos listing; lands in the "
            f"artifact's knobs.top_n (default: {predict.DEFAULT_TOP_N})"
        ),
    )
    parser.add_argument(
        "--max-failure-fraction",
        type=float,
        default=DEFAULT_MAX_FAILURE_FRACTION,
        help=(
            "abort (writing nothing) when failed/attempted exceeds this "
            f"fraction, strictly (default: {DEFAULT_MAX_FAILURE_FRACTION} "
            "— tolerates exactly half); must be in [0, 1]"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export end to end.

    Logging is configured first. ``--max-failure-fraction`` is
    validated to lie in ``[0, 1]`` and ``--as-of-date`` is resolved and
    validated (naive; unparseable or timezone-aware → ``ValueError``)
    **before any I/O**; the resolved ``as_of``'s
    ``isoformat(timespec="seconds")`` becomes the shared as-of string
    (D8/A5). The fixtures are then read and past-dropped
    (:func:`read_fixtures` — the drop happens before the predictor is
    built, so a fully-past list costs no per-fixture work), the
    ``matches.parquet`` table is loaded and the name map built (D6),
    the single :class:`drivers.predict.Predictor` is constructed
    **unconditionally** (D13's fatal path — no ``try`` around it, so
    the D10/temperature guards fire on every run, even an empty one),
    the model version is read softly (D17), the fixture loop runs
    (:func:`export_fixtures`, which times each fixture's ``predict``
    and ``build_fixture``), the P6 diagnostic pass runs over the
    returned fixtures (:func:`count_null_interval_maps` for the D19
    null-interval tally and :func:`coverage_diagnostic` for the §8
    gap, one INFO line per fixture), the D19 null-interval verdict is
    logged (WARNING naming ``train_bootstrap_replicates.py`` when any
    null intervals are present, else INFO), D14's threshold abort is
    applied, the artifact is assembled (:func:`build_artifact`) and
    written (:func:`write_artifact`), and one INFO summary line is
    logged — extended, per D22, with ``null_interval_maps``,
    ``predict_seconds`` and ``elapsed_seconds`` appended after the
    existing fields.

    **Timing (P6).** A run-level :func:`time.perf_counter` starts
    immediately after ``logging.basicConfig`` (so the ``elapsed``
    figure covers everything the process does, including flag
    validation), and the single ``Predictor`` construction is timed and
    logged (``"predictor constructed in %.2fs (version=%s)"``). The
    per-fixture ``predict``/``assemble`` seconds come back from
    :func:`export_fixtures` via its ``timings`` out-dict; nothing else
    is timed here.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller.

    Raises:
        ValueError: If ``--max-failure-fraction`` is outside ``[0, 1]``
            or ``--as-of-date`` is unparseable/timezone-aware (both
            before any I/O); or if the ``Predictor`` constructor
            rejects ``n_samples``/``ci_level`` or trips a staleness
            guard (D13's fatal path).
        FileNotFoundError: If the fixtures file, ``matches.parquet`` or
            a required fitted artifact does not exist — propagated
            unchanged.
        FixtureFileError: If the fixtures file is unreadable or
            malformed at file level (propagated from
            :func:`read_fixtures`).
        ExportAbortedError: If the D14 failure-fraction threshold is
            exceeded (nothing is written).
        TypeError / AttributeError: Propagated unchanged from the
            fixture loop for programming errors (never caught, D13).
        jsonschema.exceptions.ValidationError: If the assembled artifact
            fails validation (propagated from :func:`write_artifact`).
        OSError / TypeError: If the artifact cannot be written
            (propagated from :func:`write_artifact`).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_started = time.perf_counter()

    if not (0.0 <= args.max_failure_fraction <= 1.0):
        raise ValueError(
            f"--max-failure-fraction must be in [0, 1], got "
            f"{args.max_failure_fraction}"
        )

    if args.as_of_date is None:
        as_of = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    else:
        try:
            as_of = datetime.fromisoformat(args.as_of_date)
        except ValueError as exc:
            raise ValueError(
                f"--as-of-date {args.as_of_date!r} is not a valid "
                f"ISO-8601 datetime: {exc}"
            ) from exc
        if as_of.tzinfo is not None:
            raise ValueError(
                f"--as-of-date {args.as_of_date!r} is timezone-aware; "
                "pass a naive UTC datetime"
            )
        as_of = as_of.replace(microsecond=0)
    as_of_iso = as_of.isoformat(timespec="seconds")

    output_dir = Path(args.output_dir)
    specs, stats = read_fixtures(Path(args.fixtures), as_of)

    matches_df = evaluate.load_matches_table(output_dir, args.version)
    team_names = reshape.build_team_name_map(matches_df)

    predictor_started = time.perf_counter()
    predictor = predict.Predictor(
        output_dir,
        args.version,
        n_samples=args.n_samples,
        seed=args.seed,
        ci_level=args.ci_level,
    )
    logger.info(
        "predictor constructed in %.2fs (version=%s)",
        time.perf_counter() - predictor_started,
        args.version,
    )

    model_version = model_provenance.read_model_version(
        output_dir, args.version
    )

    timings: dict[str, float] = {}
    fixtures, failures, intervals_present = export_fixtures(
        predictor,
        specs,
        team_names,
        as_of_iso=as_of_iso,
        top_n=args.top_n,
        timings=timings,
    )

    null_maps = 0
    total_maps = 0
    for fixture in fixtures:
        per_map = fixture["overall"]["per_map"]
        total_maps += len(per_map)
        null_maps += count_null_interval_maps(per_map)
        diagnostic = coverage_diagnostic(fixture)
        if diagnostic is None:
            logger.info(
                "coverage diagnostic %s: no coverage mass, skipped",
                fixture["match_id"],
            )
        else:
            logger.info(
                "coverage diagnostic %s: mass=%.4f weighted=%.4f "
                "overall=%.4f gap=%+.4f",
                fixture["match_id"],
                diagnostic.mass,
                diagnostic.weighted,
                diagnostic.overall,
                diagnostic.gap,
            )

    if null_maps > 0:
        logger.warning(
            "%d of %d exported per_map entries carry null intervals; run "
            "drivers/train_bootstrap_replicates.py to populate them",
            null_maps,
            total_maps,
        )
    elif total_maps > 0:
        logger.info(
            "all %d exported per_map entries carry intervals", total_maps
        )

    attempted = stats.n_records - stats.n_dropped_past
    failed = stats.n_invalid + failures
    if attempted > 0 and failed / attempted > args.max_failure_fraction:
        raise ExportAbortedError(
            f"{failed} of {attempted} attempted fixture(s) failed "
            f"(fraction {failed / attempted:.3f} > "
            f"--max-failure-fraction {args.max_failure_fraction}); "
            "writing nothing"
        )

    knobs = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "ci_level": args.ci_level,
        "top_n": args.top_n,
    }
    artifact = build_artifact(
        fixtures,
        generated_at=as_of_iso,
        model_version=model_version,
        dataset_version=args.version,
        knobs=knobs,
        intervals_present=intervals_present,
    )
    artifact_path = output_dir / args.version / ARTIFACT_FILENAME
    write_artifact(artifact, artifact_path)

    logger.info(
        "exported %d fixture(s) to %s (%s/%s): dropped_past=%d "
        "failed=%d intervals_present=%s model_version=%s "
        "dataset_version=%s null_interval_maps=%d/%d "
        "predict_seconds=%.1f elapsed_seconds=%.1f",
        len(fixtures),
        artifact_path,
        output_dir,
        args.version,
        stats.n_dropped_past,
        failed,
        intervals_present,
        model_version,
        args.version,
        null_maps,
        total_maps,
        timings.get("predict_seconds", 0.0),
        time.perf_counter() - run_started,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line four-way outcome labelling (roadmap M9).

Reads the ``maps`` table that ``materialize.py`` (roadmap M8) already
flattened and validated into ``data/<version>/maps.parquet``, derives
the canonical per-map outcome label from each finished map's two
scores, and writes a new ``data/<version>/labels.parquet`` table
joined to ``maps`` by ``(match_id, map_index)``.

This module sits at the repo root next to ``materialize.py`` per the
same boundary rule tasks 008/009 established: it needs neither
``config`` nor the cache — M8 already finished the job of getting the
scores out of the cache and into a flat column, so this module reads
that column via ``pandas`` and never touches ``scraper.cache`` or
``scraper.models``.

Design rules:

- **A and B are column positions, not team identities.** In M8's
  ``maps`` table, ``team1_score`` is always "A"'s score and
  ``team2_score`` always "B"'s, independent of which real team that
  is in a given match. The label therefore derives purely from the
  two scores — no join against ``matches.parquet`` for team names is
  needed, and the ``winner`` column is not read at all. Any
  "which team should be called A" decision for a specific prediction
  is a presentation/feature-layer remapping done on top of this
  table's fixed team1/team2 convention, not a change to this table.
- **OT criterion derived from the loser side.** A map is overtime
  exactly when the losing side also reached 12+ rounds
  (``min(team1_score, team2_score) >= 12``), the direct reading of
  ``description.txt``'s "any score where both teams reach 12+ is an
  overtime map". This is an independent implementation, sharing no
  code with ``materialize.py``'s ``winner_score > 13`` report-only
  heuristic — the two agree on every validated row (task 002/007's
  invariant makes a regulation win always end at exactly 13), but M9
  does not depend on M8's function for its correctness.
- **Additive output.** ``labels.parquet`` is a new table alongside
  (not merged into) ``maps.parquet``, so M8's artifact stays schema-
  stable and independently regenerable; a consumer wanting the joined
  view does ``maps_df.merge(labels_df, on=["match_id", "map_index"])``.

Exit codes:

- ``0`` — always. An empty ``maps.parquet`` (M8 never materialised
  anything, or the cache was empty) is already flagged by
  ``materialize.py``'s own nonzero exit code and ``report.json``; M9
  does not duplicate that signal and just writes an empty,
  schema-correct ``labels.parquet``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default output parent: <project root>/data, the same convention
# materialize.py uses. Duplicated (not imported) to keep the two
# modules decoupled; both default to "data" and "v1", so running one
# right after the other with no flags addresses the same directory.
DEFAULT_OUTPUT_DIR = Path("data")

# The canonical four-way outcome vocabulary, in ordinal order: index
# == outcome_ordinal. "decisive A -> narrow A -> narrow B -> decisive
# B" is the ordered axis description.txt's Stage 2 section describes,
# and is what the ordinal logistic model consumes.
OUTCOME_LABELS = ("A-regulation", "A-OT", "B-OT", "B-regulation")

# Fixed column order for the labels table, used both to build
# non-empty DataFrames in a deterministic order and to give an empty
# run a schema-correct zero-row table.
LABELS_COLUMNS = (
    "match_id",
    "map_index",
    "outcome_label",
    "outcome_ordinal",
    "round_margin",
)

# Fixed target dtypes for the labels table, applied to every row count
# (including zero) so an empty run writes exactly the same Parquet
# schema as a non-empty one: numeric columns are int64, the two text
# columns stay object (pandas' default string container, matching what
# materialize.py's maps table produces). Without this, an empty
# DataFrame constructed from ``rows=[]`` would come out all-object/
# null dtype and round-trip through Parquet with ``map_index: null``
# instead of ``map_index: int64``, breaking any downstream consumer
# that assumes one fixed schema across dataset versions.
LABELS_DTYPES = {
    "match_id": object,
    "map_index": "int64",
    "outcome_label": object,
    "outcome_ordinal": "int64",
    "round_margin": "int64",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the labels.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with two attributes: ``version``
        (``str``, the output/input subdirectory name, default
        ``"v1"``) and ``output_dir`` (``str``, the parent directory
        the version subdirectory lives under, default ``"data"``).
        Together they locate ``<output_dir>/<version>/maps.parquet``
        to read and ``<output_dir>/<version>/labels.parquet`` to
        write. There is deliberately no ``--db-path`` flag: this
        module never touches the cache.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Derive the four-way outcome label (A-regulation/A-OT/"
            "B-OT/B-regulation) plus a signed round margin for every "
            "finished map in materialize.py's maps.parquet, and write "
            "labels.parquet."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="input/output subdirectory name under --output-dir (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory lives under "
            "(default: data)"
        ),
    )
    return parser.parse_args(argv)


def compute_outcome(team1_score: int, team2_score: int) -> tuple[str, int, int]:
    """Derive the four-way outcome label for one finished map.

    The pure, from-scratch labelling function (see the module
    docstring's design rules): ``team1_score`` is "A"'s score and
    ``team2_score`` "B"'s, so the winner is whichever side has the
    higher score and the label is the winner side plus whether the map
    went to overtime (the losing side reached 12+ rounds). The signed
    round margin is ``team1_score - team2_score``: positive means A
    won, negative means B won, and it decreases monotonically along
    the same axis as ``outcome_ordinal`` (largest positive for
    ``A-regulation`` through largest-magnitude negative for
    ``B-regulation``), the property the v2 continuous-margin
    formulation needs.

    This function does not re-validate a full scoreline — it assumes
    the row already passed task 002/007's score-validity invariant
    (every row in ``maps.parquet`` has) — but it does refuse one
    input that is impossible for a *finished* map: a tie.

    Args:
        team1_score: Rounds "A" (team1) won on the finished map.
        team2_score: Rounds "B" (team2) won on the finished map.

    Returns:
        A tuple of ``(outcome_label, outcome_ordinal, round_margin)``:
        ``outcome_label`` is one of :data:`OUTCOME_LABELS`;
        ``outcome_ordinal`` its 0-based index in that tuple
        (``0`` A-regulation, ``1`` A-OT, ``2`` B-OT, ``3``
        B-regulation); ``round_margin`` the signed margin
        ``team1_score - team2_score``.

    Raises:
        ValueError: If ``team1_score == team2_score`` — a tied
            finished map has no winner, so there is no correct label,
            and silently guessing one would corrupt the training
            target. (Only reachable by direct callers; a real
            ``maps.parquet`` row can never be a tie, since a tie has
            no ``winner`` and is excluded by M8's finished-map
            filter.)
    """
    if team1_score == team2_score:
        raise ValueError(
            f"a tied scoreline ({team1_score}-{team2_score}) has no winner "
            "and cannot be labelled"
        )
    round_margin = team1_score - team2_score
    overtime = min(team1_score, team2_score) >= 12
    if team1_score > team2_score:
        if overtime:
            return "A-OT", 1, round_margin
        return "A-regulation", 0, round_margin
    if overtime:
        return "B-OT", 2, round_margin
    return "B-regulation", 3, round_margin


def build_labels_table(maps_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build the flat ``labels`` table from the materialised maps table.

    One row per map in ``maps_df`` whose ``team1_score`` and
    ``team2_score`` are both non-null, in input row order, labelled
    by a vectorized transcription of :func:`compute_outcome`'s logic
    (``np.select`` over the two score columns — same OT criterion,
    winner side, and signed margin, no row-by-row Python loop). Only
    the ``match_id``, ``map_index``,
    ``team1_score`` and ``team2_score`` columns are read — the
    ``winner`` column (and everything else M8 wrote) is deliberately
    ignored, since the label follows purely from the two scores. A
    malformed row with a null score is skipped and counted in the
    returned integer (logged), not raised and not coerced: the
    null-score case is an expected-absent-data condition that can in
    principle slip into ``maps.parquet`` (``MapResult.__post_init__``'s
    finished-map gate skips all validation if any of
    ``team1_score``/``team2_score``/``winner`` is ``None``, so M8's
    ``winner is not None`` finished test does not itself guarantee
    both scores are present). A *tie* is a different, invariant-breaking
    case and is deliberately *not* caught here — it propagates as a
    ``ValueError`` (the same contract as :func:`compute_outcome`)
    rather than being swallowed (the plan's Assumptions section draws
    exactly this distinction).

    Args:
        maps_df: The materialised ``maps`` table (M8's
            ``maps.parquet``) with at least the ``match_id``,
            ``map_index``, ``team1_score`` and ``team2_score``
            columns.

    Returns:
        A tuple of ``(dataframe, skipped)`` where the dataframe has
        one row per labelled map with columns ``match_id, map_index,
        outcome_label, outcome_ordinal, round_margin``
        (:data:`LABELS_COLUMNS` order) and ``skipped`` is the number
        of input rows excluded because ``team1_score`` or
        ``team2_score`` was null. The dataframe always has the fixed
        :data:`LABELS_DTYPES` schema regardless of row count — an
        empty (all-null or zero-row) input yields a zero-row frame
        with ``map_index``/``outcome_ordinal``/``round_margin`` still
        ``int64`` and the text columns still object, so empty and
        non-empty runs write byte-identical Parquet schemas.

    Raises:
        ValueError: If a non-null row has ``team1_score ==
            team2_score`` (the vectorized transcription of
            :func:`compute_outcome`'s tie check; a tie in a
            "finished" row is a real invariant break and must not be
            silently skipped).
        KeyError: If ``maps_df`` lacks one of the four required
            columns (``match_id``, ``map_index``, ``team1_score``,
            ``team2_score``).
    """
    try:
        match_ids = maps_df["match_id"]
        map_indices = maps_df["map_index"]
        team1_score = maps_df["team1_score"]
        team2_score = maps_df["team2_score"]
    except KeyError as exc:
        # pandas raises KeyError for a missing column; re-raise
        # unchanged so the missing-name signal is preserved.
        raise KeyError(str(exc)) from exc

    # Null scores are the expected-absent case: skip and count them.
    null_scores = team1_score.isna() | team2_score.isna()
    skipped = int(null_scores.sum())
    valid = ~null_scores
    valid_team1 = team1_score[valid]
    valid_team2 = team2_score[valid]

    for label_index in maps_df.index[null_scores.to_numpy()]:
        logger.warning(
            "map (match %s, map_index %s) has a null score (%s-%s); "
            "skipping it from the labels table",
            maps_df.at[label_index, "match_id"],
            maps_df.at[label_index, "map_index"],
            maps_df.at[label_index, "team1_score"],
            maps_df.at[label_index, "team2_score"],
        )

    # A tie among the valid rows is an invariant break: same refusal
    # contract as compute_outcome, raised on the first tied row.
    tie = valid_team1 == valid_team2
    if tie.any():
        first_tied = tie.to_numpy().argmax()
        raise ValueError(
            f"a tied scoreline ({valid_team1.iloc[first_tied]}-"
            f"{valid_team2.iloc[first_tied]}) has no winner and cannot "
            "be labelled"
        )

    # Vectorized transcription of compute_outcome: the OT criterion is
    # min(scores) >= 12 (the loser reached 12+), the winner is the
    # higher score, and the margin is team1 minus team2.
    margin = valid_team1 - valid_team2
    overtime = (valid_team1 >= 12) & (valid_team2 >= 12)
    a_wins = valid_team1 > valid_team2
    label = np.select(
        [a_wins & overtime, a_wins & ~overtime, ~a_wins & overtime],
        ["A-OT", "A-regulation", "B-OT"],
        default="B-regulation",
    )
    ordinal = np.select(
        [a_wins & overtime, a_wins & ~overtime, ~a_wins & overtime],
        [1, 0, 2],
        default=3,
    )

    labels_df = pd.DataFrame(
        {
            "match_id": match_ids.to_numpy()[valid.to_numpy()],
            "map_index": map_indices.to_numpy()[valid.to_numpy()],
            "outcome_label": label,
            "outcome_ordinal": ordinal,
            "round_margin": margin,
        }
    )
    return labels_df.astype(LABELS_DTYPES), skipped


def load_maps_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised maps table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function so tests can exercise
    :func:`build_labels_table` directly against an in-memory
    ``DataFrame`` and stub/bypass disk entirely for the pure parts.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/maps.parquet`` as a
        ``pandas.DataFrame`` (M8's ``maps`` table; see
        :func:`materialize.build_maps_table` for its columns).

    Raises:
        FileNotFoundError: If ``maps.parquet`` does not exist for this
            version (i.e. ``materialize.py`` has not been run for it)
            — propagated as-is from ``pandas.read_parquet`` as a clear
            "run materialize.py first" signal rather than wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "maps.parquet")


def write_labels_table(
    labels_df: pd.DataFrame,
    output_dir: Path,
    version: str,
) -> None:
    """Write the labels table for a dataset version to disk.

    Writes ``<output_dir>/<version>/labels.parquet`` via
    ``pandas.DataFrame.to_parquet`` (``index=False``), creating the
    version directory (including parents) if it does not already
    exist. Overwrites any previous ``labels.parquet`` in place —
    re-labelling the same version replaces the file rather than
    erroring, matching the idempotent re-run story tasks 008/009
    established.

    Args:
        labels_df: The labels table to write (see
            :func:`build_labels_table`).
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        None.

    Raises:
        OSError: If the directory cannot be created or the file cannot
            be written (e.g. permissions or disk errors).
        ValueError: If the table contains a value that cannot be
            serialized to Parquet (propagated from
            ``DataFrame.to_parquet``).
    """
    output_dir = Path(output_dir) / version
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(output_dir / "labels.parquet", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the outcome labelling end to end.

    Logging is configured first so :func:`build_labels_table`'s
    per-map skip warnings are visible from the CLI. The materialised
    maps table is then loaded for the requested version (see
    :func:`load_maps_table`), the labels table is built (see
    :func:`build_labels_table`), a one-line summary of the label
    distribution and skip count is logged (mirroring
    ``materialize.py``'s summary-line convention), and the result is
    written under ``<output-dir>/<version>/labels.parquet`` (see
    :func:`write_labels_table`).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. Unlike ``materialize.py`` there is no nonzero
            exit-code path: an empty ``maps.parquet`` is M8's problem
            to have flagged (its own exit code and ``report.json``),
            so M9 just writes an empty, schema-correct
            ``labels.parquet`` and reports success.

    Raises:
        FileNotFoundError: If ``maps.parquet`` does not exist for the
            requested version (propagated from
            :func:`load_maps_table`).
        ValueError: If a non-null row in the maps table has a tied
            scoreline (propagated from :func:`build_labels_table` /
            :func:`compute_outcome`).
        OSError / ValueError: If the output cannot be written
            (propagated from :func:`write_labels_table`).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    maps_df = load_maps_table(Path(args.output_dir), args.version)
    labels_df, skipped = build_labels_table(maps_df)
    distribution = {
        label: int((labels_df["outcome_label"] == label).sum())
        for label in OUTCOME_LABELS
    }
    logger.info(
        "labelled %d maps (%d skipped for null scores) under "
        "%s/%s: %s",
        len(labels_df),
        skipped,
        args.output_dir,
        args.version,
        distribution,
    )
    write_labels_table(labels_df, Path(args.output_dir), args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())

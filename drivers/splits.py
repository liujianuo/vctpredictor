"""CLI entry point for the chronological split (M10).

Thin command-line wrapper around :mod:`utils.splits`, which owns the
pure split/fold/assembly logic. This module adds only the CLI/IO glue:
argument parsing (:func:`parse_args`), loading the materialised matches
table (:func:`load_matches_table`), writing the result
(:func:`write_splits_table`), and the :func:`main` entry point. It also
re-exports every pure name from :mod:`utils.splits` so existing callers
that import ``drivers.splits`` (notably ``tests/test_splits.py``) keep
working unchanged.

For the split semantics — split unit, the two split values, the
chronological tie-break, the walk-forward folding rules, and the OOF
assembly invariants — see :mod:`utils.splits`'s module docstring.

Exit codes:

- ``0`` — always. An empty ``matches.parquet`` is already flagged by
  ``materialize.py``'s own nonzero exit code and ``report.json``; M10
  does not duplicate that signal and instead raises (an empty table
  cannot be split meaningfully), matching ``labels.py``'s
  raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from utils.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_FRAC,
    MIN_FOLD_BLOCK_MATCHES,
    MIN_TRAIN_MATCHES,
    SPLIT_VALUES,
    SPLITS_COLUMNS,
    SPLITS_DTYPES,
    _chronological_order,
    assemble_out_of_fold_predictions,
    join_split_to_maps,
    split_matches,
    walk_forward_folds,
)
from utils.table_io import DEFAULT_OUTPUT_DIR, write_parquet

# Backward-compatible re-export surface: every pure name from
# utils.splits, so ``from drivers import splits; splits.<name>`` (used
# by tests/test_splits.py) keeps working unchanged. Listed explicitly
# (not ``import *``) so the names stay grep-able, and placed in
# ``__all__`` so the linter treats them as intentional re-exports
# rather than unused imports.
__all__ = (
    "DEFAULT_N_FOLDS",
    "DEFAULT_TEST_FRAC",
    "MIN_FOLD_BLOCK_MATCHES",
    "MIN_TRAIN_MATCHES",
    "SPLITS_COLUMNS",
    "SPLITS_DTYPES",
    "SPLIT_VALUES",
    "_chronological_order",
    "assemble_out_of_fold_predictions",
    "join_split_to_maps",
    "split_matches",
    "walk_forward_folds",
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the splits.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with three attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and
        ``test_frac`` (``float``, the fraction of matches held out for
        final evaluation, default :data:`DEFAULT_TEST_FRAC`). There is
        deliberately no ``--calibration-frac`` flag (no static
        calibration slice exists) and no ``--n-folds`` /
        ``--min-fold-block`` flags (those stay library-only parameters
        chosen by whichever downstream stage calls
        :func:`walk_forward_folds`).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-float ``--test-frac``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the chronological train/test split over "
            "materialize.py's matches.parquet and write splits.parquet."
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
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help=(
            "fraction of matches held out for final evaluation, in "
            f"(0, 1) (default: {DEFAULT_TEST_FRAC})"
        ),
    )
    return parser.parse_args(argv)


def load_matches_table(output_dir: Path, version: str) -> pd.DataFrame:
    """Load the materialised matches table for a dataset version.

    Thin wrapper around ``pandas.read_parquet`` isolating the file I/O
    into one function so tests can exercise :func:`split_matches` and
    friends directly against an in-memory ``DataFrame`` and
    stub/bypass disk entirely for the pure parts.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g.
            ``"v1"``).

    Returns:
        The contents of ``<output_dir>/<version>/matches.parquet`` as a
        ``pandas.DataFrame`` (M8's ``matches`` table; see
        ``materialize.MATCHES_COLUMNS`` for its columns). Only its
        ``match_id`` and ``date`` columns are used by this module.

    Raises:
        FileNotFoundError: If ``matches.parquet`` does not exist for
            this version (i.e. ``materialize.py`` has not been run for
            it) — propagated as-is from ``pandas.read_parquet`` as a
            clear "run materialize.py first" signal rather than
            wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    return pd.read_parquet(Path(output_dir) / version / "matches.parquet")


def write_splits_table(
    splits_df: pd.DataFrame,
    output_dir: Path,
    version: str,
) -> None:
    """Write the splits table for a dataset version to disk.

    Writes ``<output_dir>/<version>/splits.parquet`` via
    :func:`table_io.write_parquet` (``index=False``), creating the
    version directory (including parents) if it does not already
    exist. Overwrites any previous ``splits.parquet`` in place —
    re-splitting the same version replaces the file rather than
    erroring, matching the idempotent re-run story tasks 008/009
    established.

    Args:
        splits_df: The splits table to write (see
            :func:`split_matches`).
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
            :func:`table_io.write_parquet` / ``DataFrame.to_parquet``).
    """
    write_parquet(splits_df, Path(output_dir) / version / "splits.parquet")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chronological split end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The materialised matches table is then loaded for the
    requested version (see :func:`load_matches_table`), the split is
    computed (see :func:`split_matches`), a one-line INFO summary of
    the per-split counts and the two boundary date ranges is logged
    (mirroring ``materialize.py``'s summary-line convention), and the
    result is written under ``<output-dir>/<version>/splits.parquet``
    (see :func:`write_splits_table`).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. Unlike ``materialize.py`` there is no nonzero
            exit-code path: an empty ``matches.parquet`` is M8's
            problem to have flagged (its own exit code and
            ``report.json``), so M10 just lets the resulting
            :func:`split_matches` ``ValueError`` propagate rather than
            inventing a second signal.

    Raises:
        FileNotFoundError: If ``matches.parquet`` does not exist for
            the requested version (propagated from
            :func:`load_matches_table`).
        ValueError: If the split cannot be computed (propagated from
            :func:`split_matches`, e.g. empty table, ``test_frac`` out
            of range, or training region below the floor).
        OSError / ValueError: If the output cannot be written
            (propagated from :func:`write_splits_table`).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    matches_df = load_matches_table(Path(args.output_dir), args.version)
    splits_df = split_matches(matches_df, test_frac=args.test_frac)

    train_df = splits_df[splits_df["split"] == "train"]
    test_df = splits_df[splits_df["split"] == "test"]
    logger.info(
        "wrote %s: %d train (%s..%s), %d test (%s..%s)",
        Path(args.output_dir) / args.version,
        len(train_df),
        train_df["date"].iloc[0],
        train_df["date"].iloc[-1],
        len(test_df),
        test_df["date"].iloc[0],
        test_df["date"].iloc[-1],
    )
    write_splits_table(splits_df, Path(args.output_dir), args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())

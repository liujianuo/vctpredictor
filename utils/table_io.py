"""Shared table I/O and the default output-directory path for the root
pipeline modules.

A tiny, dependency-free module (imports only ``pathlib`` and
``pandas``) so ``materialize.py``, ``labels.py`` and ``splits.py`` can
share two things they previously each copy-pasted:

- :data:`DEFAULT_OUTPUT_DIR` — the ``<project root>/data`` output
  convention, previously a duplicated module-level constant in all
  three (and therefore editable in three places in lockstep);
- :func:`write_parquet` — the "``mkdir(parents=True, exist_ok=True)``
  then ``DataFrame.to_parquet(path, index=False)``" write, previously
  three near-identical inline implementations.

Keeping both here — rather than in ``config.py``, which the pipeline
modules deliberately never import — preserves the existing boundary
rule: this module imports neither ``config`` nor ``scraper``, so
``labels.py``/``splits.py`` still never touch the scraper stack.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a Parquet file, creating parent directories.

    The single shared implementation of the root pipeline's
    write-a-table step. Creates ``path``'s parent directory (including
    any missing intermediate directories) if it does not exist, then
    writes ``df`` via ``pandas.DataFrame.to_parquet`` with
    ``index=False``. Overwrites any existing file at ``path`` in place
    — the root pipeline's idempotent re-run story (a re-run replaces
    the artifact rather than erroring). Centralizing this here means a
    future change (e.g. an atomic write-then-rename) is made once
    instead of three times.

    Args:
        df: The DataFrame to write.
        path: The destination file path (e.g.
            ``data/v1/splits.parquet``); its parent directory is
            created if missing.

    Returns:
        None.

    Raises:
        OSError: If the parent directory cannot be created or the file
            cannot be written (e.g. permissions or disk errors).
        ValueError: If the table contains a value that cannot be
            serialized to Parquet (propagated from
            ``DataFrame.to_parquet``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

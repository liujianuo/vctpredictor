"""Tests for table_io: the shared Parquet write helper and default
output-directory constant re-used by materialize/labels/splits.
"""

from pathlib import Path

import pandas as pd

from utils import table_io


def test_default_output_dir_is_data():
    # The single shared constant the three pipeline modules re-export.
    assert table_io.DEFAULT_OUTPUT_DIR == Path("data")


def test_write_parquet_creates_parents_and_roundtrips(tmp_path):
    # A deep destination path with missing parents is created, then the
    # frame round-trips with no extra index column.
    df = pd.DataFrame([{"a": 1, "b": "x"}], columns=["a", "b"])
    path = tmp_path / "deeply" / "nested" / "t.parquet"
    table_io.write_parquet(df, path)
    assert path.exists()
    written = pd.read_parquet(path)
    assert list(written.columns) == ["a", "b"]
    assert len(written) == 1
    assert written.iloc[0]["a"] == 1
    assert written.iloc[0]["b"] == "x"


def test_write_parquet_overwrites_in_place(tmp_path):
    # Writing to an existing path replaces the file (the idempotent
    # re-run story) rather than erroring or appending rows.
    path = tmp_path / "t.parquet"
    table_io.write_parquet(pd.DataFrame([{"a": 1}], columns=["a"]), path)
    table_io.write_parquet(pd.DataFrame([{"a": 2}], columns=["a"]), path)
    written = pd.read_parquet(path)
    assert len(written) == 1
    assert written.iloc[0]["a"] == 2

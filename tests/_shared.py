"""Shared real-``data/v1`` availability skip-guard for the tests package.

The single home for the "bare table name + ``.parquet`` implied"
real-data guard convention (finding F4 in
``tasks/048-shared-real-data-fixture/plan.md``): test modules import
``_real_v1_available`` (and ``_REAL_V1_TABLES`` where they need the
bare 5-tuple for their own second purpose, e.g. the ``shutil.copy2``
loop in ``test_temperature_scaling.py``) instead of redefining the
guard locally. New test files that need real-``data/v1`` tests must
import these names from here, never redefine them.

Two families of files intentionally do **not** fully migrate to this
module and are called out so nobody re-derives the distinction:

- The five Convention-B files (``test_evaluate_bootstrap_intervals.py``,
  ``test_evaluate_veto_conditional_variance.py``,
  ``test_evaluate_reliability_diagrams.py``,
  ``test_stage_isolation.py``, ``test_veto_marginalized_series.py``)
  check full filenames *including non-parquet model artifacts* (e.g.
  ``ordinal_logit_model.json``) — a different contract from this
  module's bare-table-name one that needs a signature extension
  (a follow-up housekeeping item, out of scope for task 048).
- The four hybrid files (``test_proportional_odds.py``,
  ``test_granularity_ablation.py``,
  ``test_temperature_calibration.py``,
  ``test_temperature_scaling.py``) check the five parquet tables **and
  their own extra fitted-model artifact(s)**. They delegate the parquet
  half of their local ``_real_v1_available()`` to this module's helper
  (so the shared table check is defined in exactly one place) while
  keeping their local artifact checks, so their skip semantics are
  byte-for-byte unchanged.
"""

from collections.abc import Sequence
from pathlib import Path

# The five materialised parquet table names every real-v1 test
# ultimately depends on, as bare names (no ``.parquet`` extension) so
# callers can append the extension themselves — matching the historical
# per-file convention that ``test_temperature_scaling.py`` relies on
# when it copies each table file by name.
_REAL_V1_TABLES = ("matches", "maps", "labels", "splits", "player_map_stats")


def _real_v1_available(tables: Sequence[str] = _REAL_V1_TABLES) -> bool:
    """Report whether the materialised v1 tables exist on disk.

    The shared skip guard for the real-data tests: every named table
    must exist as ``data/v1/{name}.parquet`` (i.e. ``materialize.py``,
    ``labels.py`` and ``splits.py`` have been run against ``data/v1``).
    Files whose real-data tests additionally need fitted model
    artifacts compose this helper with their own extra file checks
    (see the module docstring).

    Args:
        tables: The bare table names to check, defaulting to the full
            five-table :data:`_REAL_V1_TABLES`. Callers needing only a
            subset (e.g. ``("matches", "maps")``) pass it explicitly.

    Returns:
        A bool: ``True`` iff every ``data/v1/{name}.parquet`` file for
        each name in ``tables`` exists.

    Raises:
        Nothing.
    """
    return all(Path(f"data/v1/{name}.parquet").exists() for name in tables)

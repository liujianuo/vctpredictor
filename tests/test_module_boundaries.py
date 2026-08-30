"""Architecture-boundary regression tests for the utils/ <-> features/ split.

Encodes the module-boundary standard as executable assertions, read from
module *source* via ``Path(<module>.py).read_text()`` (matching the
existing layering-test convention in ``test_player_form.py`` /
``test_h2h_context.py``) rather than by importing and introspecting live
modules, so the rule holds even for a module whose import would be
broken by the very violation under test.

The rules enforced here:

- No ``utils/`` module may import another ``utils/`` submodule's names,
  with one explicit, named exception: ``asof.py`` imports
  ``DEFAULT_OUTPUT_DIR`` (a ``Path`` constant) from ``table_io.py``.
- No ``utils/`` module may import from ``features/`` (no
  ``utils/ -> features/`` edges; the dependency graph is rooted at
  ``utils/``).
- No ``features/`` module may import a sibling ``features/`` module
  other than the explicitly shared ``_shared.py`` (no lateral
  feature-to-feature private-helper imports).
"""

from pathlib import Path

# The modules that live under utils/. Update this constant list whenever
# a module is added to or removed from utils/ so the test's coverage
# stays legible and does not silently grow or shrink with the
# filesystem.
UTILS_MODULES = (
    "asof.py",
    "config.py",
    "scoring.py",
    "splits.py",
    "table_io.py",
)

# The modules that live under features/ (excluding _shared.py, which is
# the explicitly-shared feature-support module every other feature may
# import from). Update this constant list whenever a module is added to
# or removed from features/.
FEATURE_MODULES = (
    "closeness.py",
    "elo.py",
    "h2h_context.py",
    "map_win_rate.py",
    "player_form.py",
)

# The one allowed cross-utils-module import: a Path *constant*, not a
# function/method/private-helper import. Kept as an explicit, named
# exception so the rule below has no blanket carve-outs.
ALLOWED_UTILS_CROSS_IMPORT = "from utils.table_io import DEFAULT_OUTPUT_DIR"


def test_no_utils_module_imports_another_utils_module():
    # Every utils/ module must stand alone (no lateral util-to-util
    # imports) except the single named constant import in asof.py.
    for module in UTILS_MODULES:
        source = Path("utils", module).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("from utils.") or stripped.startswith(
                "from utils import"
            ):
                assert stripped == ALLOWED_UTILS_CROSS_IMPORT, (
                    f"utils/{module} has a lateral utils-to-utils import "
                    f"other than the allowed {ALLOWED_UTILS_CROSS_IMPORT!r}: "
                    f"{stripped!r}"
                )


def test_no_utils_module_imports_features():
    # utils/ must never depend upward on features/ (the dependency graph
    # is a DAG rooted at utils/, with no utils/ -> features/ edges).
    for module in UTILS_MODULES:
        source = Path("utils", module).read_text(encoding="utf-8")
        assert "from features" not in source, (
            f"utils/{module} imports from features/; utils/ must not "
            "depend on features/"
        )
        assert "import features" not in source, (
            f"utils/{module} imports features/; utils/ must not depend "
            "on features/"
        )


def test_no_feature_module_imports_sibling_feature_module():
    # features/ modules may only depend on _shared.py among themselves;
    # a private helper must never be reached into from a sibling feature
    # module.
    for module in FEATURE_MODULES:
        source = Path("features", module).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from features."):
                continue
            imported = stripped[len("from features.") :].split()[0]
            assert imported == "_shared", (
                f"features/{module} imports {imported!r} from a sibling "
                "feature module; only features._shared may be imported "
                "laterally"
            )

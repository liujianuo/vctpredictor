"""Architecture-boundary regression tests for the utils/ <-> features/ split,
the models/ layer on top of it, and the evaluation/ layer above that.

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
- The dependency graph is a DAG with ``models/`` on top: no ``utils/``
  or ``features/`` module may import from ``models/`` (no upward
  edges), and a ``models/`` module may only depend downward on
  ``features.*`` / ``utils.*`` — never on ``drivers.*``. The one
  lateral exception is the explicitly-shared
  ``models/_shared.py`` (the exact analogue of the
  ``features/_shared.py`` carve-out above): ``ordinal_logit.py`` and
  ``multinomial_logit.py`` both import from it, and it is deliberately
  excluded from ``MODELS_MODULES`` just as ``features/_shared.py`` is
  excluded from ``FEATURE_MODULES`` (the pre-existing gap that
  ``features/_shared.py`` itself is not scanned is inherited
  unchanged here, not newly introduced).
- One more rung above ``models/``: ``evaluation/`` may depend downward
  on ``models.*`` / ``features.*`` / ``utils.*`` but never on
  ``drivers.*`` or on a sibling ``evaluation/`` module, and nothing in
  ``utils/``, ``features/`` or ``models/`` may depend upward on
  ``evaluation/`` (no ``utils/ -> evaluation``, ``features/ ->
  evaluation`` or ``models/ -> evaluation`` edges; the DAG stays
  rooted at ``utils/``).
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

# The modules that live under models/ (excluding _shared.py, which is
# the explicitly-shared model-support module every other models module
# may import from — the exact analogue of the features/_shared.py
# carve-out one rung down, deliberately not scanned here just as
# features/_shared.py is not scanned either). Update this constant list
# whenever a module is added to or removed from models/ so the test's
# coverage stays legible and does not silently grow or shrink with the
# filesystem. Note this is the top-level models/ package (roadmap M18),
# unrelated to scraper.models (the scraper's pure cache dataclasses).
MODELS_MODULES = (
    "binary_logit.py",
    "four_way_baseline.py",
    "greedy_veto_simulator.py",
    "multinomial_logit.py",
    "ordinal_logit.py",
    "temperature_scaling.py",
)

# The modules that live under evaluation/ (the generic map-outcome
# evaluation harness, roadmap M19). Update this constant list whenever
# a module is added to or removed from evaluation/ so the test's
# coverage stays legible and does not silently grow or shrink with the
# filesystem.
EVALUATION_MODULES = (
    "granularity_ablation.py",
    "harness.py",
    "proportional_odds.py",
    "temperature_calibration.py",
)


def test_no_utils_module_imports_another_utils_module():
    # Every utils/ module must stand alone (no lateral util-to-util
    # imports) except the single named constant import in asof.py.
    for module in UTILS_MODULES:
        source = Path("utils", module).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from utils.", "from utils import")):
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


def test_no_utils_or_features_module_imports_models():
    # models/ is the top of the dependency DAG: nothing in utils/ or
    # features/ may depend upward on it (a models import from below
    # would invert the layering and invite circular imports).
    for directory, modules in (("utils", UTILS_MODULES), ("features", FEATURE_MODULES)):
        for module in modules:
            source = Path(directory, module).read_text(encoding="utf-8")
            assert "from models" not in source, (
                f"{directory}/{module} imports from models/; {directory}/ "
                "must not depend on models/"
            )
            assert "import models" not in source, (
                f"{directory}/{module} imports models/; {directory}/ must "
                "not depend on models/"
            )


def test_models_module_imports_only_features_and_utils():
    # A models/ module may only depend downward on features.* / utils.*;
    # importing from drivers/ (the CLI pipeline layer) or from a sibling
    # models/ module would break the DAG. The one lateral exception is
    # the explicitly-shared models/_shared.py (mirroring the
    # features/_shared.py carve-out in
    # test_no_feature_module_imports_sibling_feature_module): scan each
    # ``from models.`` line and assert the imported submodule name is
    # exactly ``_shared``. Only explicit ``import`` / ``from``
    # statements are scanned (stdlib and third-party imports such as
    # pandas/dataclasses are fine and are not flagged).
    for module in MODELS_MODULES:
        source = Path("models", module).read_text(encoding="utf-8")
        assert "from drivers" not in source, (
            f"models/{module} imports from drivers/; models/ must not "
            "depend on drivers/"
        )
        assert "import drivers" not in source, (
            f"models/{module} imports drivers/; models/ must not depend "
            "on drivers/"
        )
        assert "import models" not in source, (
            f"models/{module} imports a sibling models/ module; models/ "
            "modules must stand alone laterally"
        )
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from models."):
                continue
            imported = stripped[len("from models.") :].split()[0]
            assert imported == "_shared", (
                f"models/{module} imports {imported!r} from a sibling "
                "models/ module; only models._shared may be imported "
                "laterally"
            )


def test_no_utils_features_or_models_module_imports_evaluation():
    # evaluation/ is the top of the dependency DAG above models/: nothing
    # in utils/, features/ or models/ may depend upward on it (an
    # evaluation import from below would invert the layering and invite
    # circular imports, exactly like the models/ rung's own rule).
    for directory, modules in (
        ("utils", UTILS_MODULES),
        ("features", FEATURE_MODULES),
        ("models", MODELS_MODULES),
    ):
        for module in modules:
            source = Path(directory, module).read_text(encoding="utf-8")
            assert "from evaluation" not in source, (
                f"{directory}/{module} imports from evaluation/; "
                f"{directory}/ must not depend on evaluation/"
            )
            assert "import evaluation" not in source, (
                f"{directory}/{module} imports evaluation/; {directory}/ "
                "must not depend on evaluation/"
            )


def test_evaluation_module_imports_only_features_models_and_utils():
    # An evaluation/ module may only depend downward on models.* /
    # features.* / utils.*; importing from drivers/ (the CLI pipeline
    # layer, which is the layer above evaluation/ in the DAG) or from a
    # sibling evaluation/ module would break the DAG. Only explicit
    # ``import`` / ``from`` statements are scanned (stdlib and
    # third-party imports such as pandas/collections are fine and are
    # not flagged).
    for module in EVALUATION_MODULES:
        source = Path("evaluation", module).read_text(encoding="utf-8")
        assert "from drivers" not in source, (
            f"evaluation/{module} imports from drivers/; evaluation/ must "
            "not depend on drivers/"
        )
        assert "import drivers" not in source, (
            f"evaluation/{module} imports drivers/; evaluation/ must not "
            "depend on drivers/"
        )
        assert "from evaluation" not in source, (
            f"evaluation/{module} imports from a sibling evaluation/ "
            "module; evaluation/ modules must stand alone laterally"
        )
        assert "import evaluation" not in source, (
            f"evaluation/{module} imports a sibling evaluation/ module; "
            "evaluation/ modules must stand alone laterally"
        )

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
  other than the explicitly shared ``_shared.py`` and ``round_detail.py``
  (no lateral feature-to-feature private-helper imports).
  ``round_detail.py`` is a second explicitly-shared feature-support
  module (roadmap M38.1 — the shared substrate M38.2 and M38.3 both
  train against): it is excluded from ``FEATURE_MODULES`` below exactly
  like ``_shared.py``, and its one import from ``_shared.py`` (the
  ``TEAM1_SCORE_COL``/``TEAM2_SCORE_COL`` score-column constants) is a
  shared-to-shared dependency — ``_shared.py`` has no downstream
  dependents of its own to protect, so it is not a lateral
  feature-to-feature import in the problematic sense.
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
- One more rung above ``evaluation/``: the new ``presentation/``
  package (roadmap P1). ``presentation/`` may depend downward on
  ``utils/`` and may import exactly one ``drivers/`` edge — a
  ``from drivers.predict import <allowed surface>`` statement whose
  imported names are a subset of
  :data:`ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES` — but may not
  import ``features/`` or ``models/`` at all. Nothing in ``utils/``,
  ``features/``, ``models/`` or ``evaluation/`` may import
  ``presentation/`` (only ``drivers/`` may, and ``drivers/`` is
  deliberately left unscanned here — it is a directory of entry
  points, not a DAG node). The DAG stays rooted at ``utils/``.
"""

import re
from pathlib import Path

# The modules that live under utils/. Update this constant list whenever
# a module is added to or removed from utils/ so the test's coverage
# stays legible and does not silently grow or shrink with the
# filesystem.
UTILS_MODULES = (
    "asof.py",
    "config.py",
    "scoring.py",
    "series_paths.py",
    "splits.py",
    "table_io.py",
)

# The modules that live under features/ (excluding _shared.py and
# round_detail.py, the two explicitly-shared feature-support modules
# every other feature may import from — _shared.py the score-
# orientation/roster helpers, round_detail.py the per-map round-detail
# substrate of roadmap M38.1). Update this constant list whenever a
# module is added to or removed from features/.
FEATURE_MODULES = (
    "closeness.py",
    "elo.py",
    "first_blood.py",
    "h2h_context.py",
    "map_win_rate.py",
    "player_form.py",
    "side_win_rate.py",
    "signed_margin.py",
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
    "ancestral_veto_sampler.py",
    "binary_logit.py",
    "conditional_logit_ban.py",
    "conditional_logit_pick.py",
    "flat_series_baseline.py",
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
    "bootstrap_intervals.py",
    "compounding_diagnostics.py",
    "granularity_ablation.py",
    "harness.py",
    "proportional_odds.py",
    "reliability_diagrams.py",
    "series_evaluation.py",
    "stage_isolation.py",
    "temperature_calibration.py",
    "veto_conditional_variance.py",
    "veto_marginalized_series.py",
    "veto_evaluation.py",
)

# The modules that live under presentation/ (the new top-level DAG node
# of roadmap P1 — the wire data contract plus, in later milestones, the
# derived-quantity/reshaping/leverage libraries). __init__.py is listed
# deliberately: it is docstring-only today (so it passes trivially), but
# scanning it means a future re-export surface in it is caught by the
# drivers-edge test below instead of silently skipped. Update this
# constant list whenever a module is added to or removed from
# presentation/ so the test's coverage stays legible and does not
# silently grow or shrink with the filesystem.
PRESENTATION_MODULES = (
    "__init__.py",
    "contract.py",
    "derived.py",
    "reshape.py",
)

# The exact permitted surface presentation/ may import from
# drivers.predict — the standing contract's own class/def names, spelled
# exactly as drivers/predict.py defines them. The names matter (the
# Conventions warning): a misspelling here is a rule that silently
# matches nothing — there is no ``MapPrediction``, and the ranked entry
# is ``RankedVetoPrediction``. ``make_top_vetos_fn`` is deliberately
# excluded (it is a library-only factory, not part of the export's
# standing surface).
ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES = frozenset({
    "PerMapPrediction",
    "PredictionResult",
    "Predictor",
    "RankedVetoPrediction",
    "SeriesPrediction",
    "VetoSensitivity",
    "make_predictor",
})


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
    # features/ modules may only depend on the two explicitly-shared
    # modules (_shared.py and round_detail.py) among themselves; a
    # private helper must never be reached into from a sibling feature
    # module. Both names are in the allowed set because round_detail.py
    # is a second explicitly-shared feature-support module (roadmap
    # M38.1), mirroring _shared.py's exclusion from FEATURE_MODULES.
    for module in FEATURE_MODULES:
        source = Path("features", module).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from features."):
                continue
            imported = stripped[len("from features.") :].split()[0]
            assert imported in {"_shared", "round_detail"}, (
                f"features/{module} imports {imported!r} from a sibling "
                "feature module; only features._shared and "
                "features.round_detail may be imported laterally"
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


def _folded_from_imports(source):
    """Yield each complete ``from ... import ...`` statement, folded onto
    one line.

    Walks a module's source lines and, when a ``from ... import``
    statement opens parentheses that do not close on the same physical
    line (or ends with an explicit backslash continuation), keeps
    consuming continuation lines until the statement is complete, then
    yields it as a single whitespace-joined string with any
    continuation backslashes removed. Statements that fit on one
    physical line are yielded unchanged. This prevents a parenthesized
    multi-line import from being mis-parsed as an empty name list by
    :func:`_imported_names`.

    Args:
        source: The raw module source text.

    Yields:
        The complete text of each ``from ... import`` statement, one
        per statement, with parentheses balanced, continuation
        backslashes removed, and continuation lines joined by single
        spaces.

    Raises:
        Nothing — lines that neither begin nor continue a ``from ...
        import`` statement are skipped, and an unterminated statement
        at end-of-file is simply not yielded.
    """
    lines = source.splitlines()
    buffer = None
    depth = 0
    for line in lines:
        stripped = line.strip()
        if buffer is None:
            if not (stripped.startswith("from ") and " import " in stripped):
                continue
            buffer = [stripped.rstrip("\\")]
        else:
            buffer.append(stripped.rstrip("\\"))
        depth += stripped.count("(") - stripped.count(")")
        if depth > 0 or stripped.endswith("\\"):
            continue
        yield " ".join(buffer)
        buffer = None
        depth = 0


def _imported_names(from_import_line):
    """Extract the bare imported names from a folded ``from X import``
    statement.

    Parses the text after the ``import`` keyword, strips any wrapping
    parentheses, splits on commas, and drops any ``name as alias``
    renaming (keeping the name imported from the module, which is what
    the allowed-surface check compares) so callers can test each name's
    membership. The statement is expected to already be folded onto one
    logical line by :func:`_folded_from_imports`, so a parenthesized
    multi-line import is passed whole.

    Args:
        from_import_line: A stripped source line beginning with
            ``from <module> import`` (already folded across any
            parenthesized continuation lines).

    Returns:
        A ``frozenset`` of the imported name strings, in no particular
        order.

    Raises:
        Nothing — any line beginning with ``from ... import`` yields a
            (possibly empty) set after splitting, so malformed text is
            surfaced by the caller's membership assertion rather than
            here.
    """
    body = from_import_line.split("import", 1)[1]
    names = []
    for part in body.strip().strip("()").split(","):
        stripped = part.strip()
        if stripped:
            names.append(stripped.split(" as ")[0].strip())
    return frozenset(names)


def test_presentation_module_imports_only_allowed_drivers_predict_names():
    # presentation/ may reach into drivers/ only via the one permitted
    # edge: ``from drivers.predict import <allowed surface>``. Any other
    # ``from drivers.*`` import, any whole-module ``import drivers`` /
    # ``import drivers.predict`` (then reaching into privates), or any
    # name outside the allowed surface is a boundary violation. Imports
    # are folded first (via _folded_from_imports) so a parenthesized
    # multi-line from-import is checked name-by-name rather than skipped.
    for module in PRESENTATION_MODULES:
        source = Path("presentation", module).read_text(encoding="utf-8")
        assert "import drivers" not in source, (
            f"presentation/{module} imports a drivers/ module wholesale; "
            "the only permitted drivers edge is a from-import of the "
            "allowed predict.py surface"
        )
        for statement in _folded_from_imports(source):
            if not statement.startswith("from drivers"):
                continue
            assert statement.startswith("from drivers.predict import "), (
                f"presentation/{module} has a drivers import other than "
                f"from drivers.predict: {statement!r}"
            )
            for name in _imported_names(statement):
                assert name in ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES, (
                    f"presentation/{module} imports {name!r} from "
                    "drivers.predict, which is not in the allowed "
                    "presentation surface"
                )


def test_parenthesized_drivers_import_is_folded_and_fully_checked():
    # Regression: the old per-line scan parsed ``from drivers.predict
    # import (`` as an empty name set and skipped the continuation lines,
    # so a private name imported via a parenthesized multi-line
    # from-import passed with no assertion ever running. Folding must
    # reunite the statement so every imported name is surfaced to the
    # membership check below.
    source = (
        "from drivers.predict import (\n"
        "    PredictionResult,\n"
        "    _private_helper,\n"
        ")\n"
    )
    folded = list(_folded_from_imports(source))
    assert folded == [
        "from drivers.predict import ( PredictionResult, _private_helper, )"
    ]
    imported = set()
    for statement in folded:
        assert statement.startswith("from drivers.predict import ")
        imported |= _imported_names(statement)
    assert imported == {"PredictionResult", "_private_helper"}
    assert not imported <= ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES, (
        "the private helper must be surfaced so the boundary check "
        "rejects it rather than silently skipping it"
    )


def test_no_presentation_module_imports_features_or_models():
    # presentation/ may depend downward on utils/ (and the one permitted
    # drivers.predict edge) only; anything it needs from features/ or
    # models/ it must read off a PredictionResult, never import.
    for module in PRESENTATION_MODULES:
        source = Path("presentation", module).read_text(encoding="utf-8")
        assert "from features" not in source and "import features" not in source, (
            f"presentation/{module} imports features/; presentation/ "
            "must not depend on features/"
        )
        assert "from models" not in source and "import models" not in source, (
            f"presentation/{module} imports models/; presentation/ "
            "must not depend on models/"
        )


def test_no_lower_layer_imports_presentation():
    # Nothing may import presentation/ except drivers/ (and drivers/ is
    # deliberately unscanned by this file — it is a directory of entry
    # points, not a DAG node). utils/, features/, models/ and
    # evaluation/ must never reach upward into presentation/.
    for directory, modules in (
        ("utils", UTILS_MODULES),
        ("features", FEATURE_MODULES),
        ("models", MODELS_MODULES),
        ("evaluation", EVALUATION_MODULES),
    ):
        for module in modules:
            source = Path(directory, module).read_text(encoding="utf-8")
            assert "from presentation" not in source, (
                f"{directory}/{module} imports from presentation/; "
                f"{directory}/ must not depend on presentation/"
            )
            assert "import presentation" not in source, (
                f"{directory}/{module} imports presentation/; "
                f"{directory}/ must not depend on presentation/"
            )


def test_presentation_allowed_driver_names_exist_in_drivers_predict():
    # The allowed-surface names must be spelled exactly as
    # drivers/predict.py defines them — a misspelling in the constant
    # above is a rule that silently matches nothing (the Conventions
    # warning). Pin each name to a real class/def in the source.
    source = Path("drivers", "predict.py").read_text(encoding="utf-8")
    for name in sorted(ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES):
        assert re.search(rf"^(class|def) {name}\b", source, re.MULTILINE), (
            f"{name!r} in ALLOWED_PRESENTATION_DRIVERS_IMPORT_NAMES is "
            "not a class or def in drivers/predict.py; check the spelling"
        )

"""Train-and-persist driver for the per-map bootstrap replicates (roadmap M39.3).

A **producer + persistence** driver (a ``train_*`` driver, not an
``evaluate_*`` one, because it emits a consumed artifact): resamples the
M10 ``"train"`` split at the *match* level with replacement (the M36
block bootstrap, exactly as
:func:`drivers.training_data.assemble_bootstrap_design_matrix`
implements it), refits the Stage-2 map model
(``models.ordinal_logit``) once per replicate — the same per-map refit
loop ``drivers/evaluate_bootstrap_intervals.py`` runs — and persists
the resulting fitted replicate models to
``data/<version>/ordinal_bootstrap_replicates.json`` so that
``drivers/predict.py``'s ``make_predictor`` can auto-load them by
default (M39.3: ``bootstrap_models=None`` now means "load the persisted
replicates"; an explicit ``()`` remains the no-interval escape hatch).
**No modeling code lives here**: no fit math, band math, or result-shape
changes — only the per-map replicate set is produced and persisted.
M36's separate *series*-replicate set stays eval-only and is explicitly
out of scope (it is not produced or persisted by this driver).

**Scope decisions (recorded here, do not re-derive later):**

1. **The per-map refit loop body is duplicated from
   ``evaluate_bootstrap_intervals.py``, not imported or extracted
   (assumption A1 of the M39.3 plan, resolved).** Each replicate
   performs one
   :func:`drivers.training_data.assemble_bootstrap_design_matrix` call
   (consuming the single ``--bootstrap-seed`` rng) followed by one
   ``models.ordinal_logit.fit`` — the exact loop body of that driver's
   per-map section, copied verbatim. The repo's per-driver-duplication
   convention is the stated default (``_load_fitted_models``,
   ``_load_veto_models`` and the other private loader helpers are each
   independently duplicated per driver), and unlike those loaders this
   loop is only *two* calls over already-shared library functions
   (:func:`drivers.training_data.assemble_bootstrap_design_matrix` and
   ``models.ordinal_logit.fit``), so the shared surface it would
   duplicate is trivially thin — an extracted helper would wrap nothing
   but a two-line ``for`` body and would force a re-test of the
   reviewed-clean ``evaluate_bootstrap_intervals.py`` driver for no
   requirement in scope (this milestone's scope reminder forbids
   silently widening). The duplication is flagged here for REVIEW as the
   plan asks, with the alternative (a shared ``drivers/``-layer helper)
   consciously declined.
2. **The base M20 artifact is loaded *before* the refit loop** (the
   "load the prerequisite before the expensive loop" ordering
   ``drivers/train_temperature_scaling.py`` established): a missing
   ``ordinal_logit_model.json`` raises ``FileNotFoundError`` as the
   "run train_ordinal_logit.py first" signal before any resample/refit
   work happens. The base model's ``thresholds`` (3 values) become the
   artifact's ``base_ordinal_thresholds`` provenance copy — the
   staleness-guard input ``make_predictor`` compares against its own
   loaded base model (the M39.3 mirror of decision E's
   temperature/base-model guard).
3. **Non-converged replicate fits are kept, not dropped** (the same
   "transparency diagnostic, not a filter" doctrine M36 decision 5
   records): ``n_bootstrap_map_converged`` (the count of
   ``model.converged``) is reported as a log-line diagnostic only —
   the persisted artifact carries every replicate regardless.
4. **Two local constant duplicates (assumption A5):**
   :data:`DEFAULT_N_BOOTSTRAP_MAP` (12) and
   :data:`DEFAULT_BOOTSTRAP_SEED` (2026) mirror
   ``evaluate_bootstrap_intervals.py``'s per-map constants exactly
   (same names/values), per the repo's "constants duplicated per
   driver" convention — no cross-module constant import.

Artifact written per run (scoped by dataset version):

- ``data/<version>/ordinal_bootstrap_replicates.json`` — a dict with
  keys ``"config"`` (``{"n_bootstrap_map": int, "bootstrap_seed":
  int}``), ``"replicates"`` (a list of
  :func:`models.ordinal_logit.to_dict` dicts with the derived
  ``"coefficient_report"`` key stripped from each — ``from_dict``
  ignores that key on read regardless, so stripping it only shrinks
  the file, and ``to_dict``'s signature/behavior is unchanged) and
  ``"base_ordinal_thresholds"`` (the loaded base model's 3 thresholds,
  a provenance copy). Written with ``json.dumps(..., indent=2,
  sort_keys=True)`` plus a trailing newline (the repo-wide artifact
  convention).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from drivers import evaluate, training_data
from models import ordinal_logit
from models.ordinal_logit import OrdinalLogitModel
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The per-map replicate count and bootstrap seed (assumption A5):
# deliberately duplicated from drivers/evaluate_bootstrap_intervals.py's
# per-map constants — same names, same values — so the two drivers'
# per-map sections stay recognizably identical, and each driver keeps
# its own copy per the repo's "constants duplicated per driver"
# convention. DEFAULT_N_BOOTSTRAP_MAP is a measured wall-clock choice
# (~30 s per per-map replicate on real v1 fitted models in the M36
# driver's measurements — this driver's loop is a subset of that cost:
# resample + refit only, no test-set scoring, so 12 replicates land
# well under the M36 per-map section's ~6 minutes); both knobs are CLI
# flags, not constants-of-convenience.
DEFAULT_N_BOOTSTRAP_MAP = 12
DEFAULT_BOOTSTRAP_SEED = 2026


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the train_bootstrap_replicates.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with four attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``),
        ``n_bootstrap_map`` (``int``, the per-map replicate count,
        default :data:`DEFAULT_N_BOOTSTRAP_MAP`) and
        ``bootstrap_seed`` (``int``, the seed for the single
        ``numpy.random.default_rng`` driving the match-resampling
        draws, default :data:`DEFAULT_BOOTSTRAP_SEED`). Together they
        locate the five input tables, the base
        ``ordinal_logit_model.json`` artifact and the output artifact
        ``<output_dir>/<version>/ordinal_bootstrap_replicates.json``.
        The flag names/defaults mirror
        ``evaluate_bootstrap_intervals.py``'s per-map section exactly.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-bootstrap-map``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Produce and persist the per-map bootstrap replicate "
            "models (M39.3): resample the train split at the match "
            "level (block bootstrap), refit the Stage-2 ordinal-logit "
            "map model once per replicate, and write "
            "ordinal_bootstrap_replicates.json (config + stripped "
            "replicate to_dict entries + the base model's thresholds "
            "provenance copy) so predict.py's make_predictor auto-loads "
            "it by default."
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
        "--n-bootstrap-map",
        type=int,
        default=DEFAULT_N_BOOTSTRAP_MAP,
        help=(
            "per-map bootstrap replicate count (default: "
            f"{DEFAULT_N_BOOTSTRAP_MAP} — a measured wall-clock choice "
            "on real v1 data: each replicate resamples the train split "
            "and refits the ordinal logit, so the full run at the "
            "default lands on the order of a minute or two; raise for "
            "smoother percentile bands in predict(), lower for a "
            "faster run)"
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=(
            "seed for numpy.random.default_rng, driving the match-"
            "resampling draws (the same seed reproduces byte-identical "
            f"resamples and therefore byte-identical refit models; "
            f"default: {DEFAULT_BOOTSTRAP_SEED})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train and persist the per-map bootstrap replicates end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The ``--n-bootstrap-map`` flag is validated (positive) before
    any I/O, the five input tables are loaded for the requested version
    (matches/maps/labels/splits/player_map_stats via the
    ``drivers.evaluate`` helpers), the base M20 artifact is loaded next
    (``ordinal_logit_model.json`` via
    :func:`models.ordinal_logit.from_dict` — the prerequisite, loaded
    *before* the refit loop so a missing artifact fails fast with the
    "run train_ordinal_logit.py first" signal; its ``thresholds``
    become the artifact's ``base_ordinal_thresholds`` provenance copy),
    and then one
    :func:`drivers.training_data.assemble_bootstrap_design_matrix`
    draw + ``models.ordinal_logit.fit`` per replicate runs over the
    ``"train"`` split, all draws consumed sequentially from the single
    ``numpy.random.default_rng(bootstrap_seed)`` (the duplicated per-
    map loop body of ``evaluate_bootstrap_intervals.py`` — module
    docstring decision 1). Non-converged replicates are kept (module
    docstring decision 3). The report dict — ``"config"``
    (``n_bootstrap_map``, ``bootstrap_seed``), ``"replicates"`` (each
    ``models.ordinal_logit.to_dict(model)`` with its derived
    ``"coefficient_report"`` key stripped), and
    ``"base_ordinal_thresholds"`` (the base model's 3 thresholds) — is
    written as
    ``<output_dir>/<version>/ordinal_bootstrap_replicates.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: version/output_dir, the
    replicate count with its converged count, and the seed.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        ValueError: If ``n_bootstrap_map`` is not positive, or if any
            input artifact/table is malformed (propagated from
            :func:`models.ordinal_logit.from_dict` / the feature
            modules / ``models.ordinal_logit.fit``, e.g. an empty
            resampled label vector).
        FileNotFoundError: If any of the five input tables or the base
            ``ordinal_logit_model.json`` artifact does not exist for
            the requested version (i.e. ``materialize.py`` /
            ``labels.py`` / ``splits.py`` / ``train_ordinal_logit.py``
            have not been run for it) — propagated unchanged as a clear
            "run the prerequisite first" signal. The base artifact is
            loaded before the refit loop, so a missing one fails fast
            rather than after the (expensive) resample/refit work.
        KeyError: If any input table or model artifact lacks a required
            key/column (propagated from the feature modules / the
            ``from_dict`` call).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.n_bootstrap_map < 1:
        raise ValueError(
            f"--n-bootstrap-map must be a positive integer, got "
            f"{args.n_bootstrap_map}"
        )

    output_dir = Path(args.output_dir)
    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The prerequisite (module docstring decision 2): load the final
    # M20 artifact before the refit loop so a missing artifact fails
    # fast. Its thresholds are the provenance copy the persisted
    # artifact carries (base_ordinal_thresholds — the staleness-guard
    # input for make_predictor's auto-load).
    base_path = output_dir / args.version / "ordinal_logit_model.json"
    with open(base_path, encoding="utf-8") as handle:
        base_model = ordinal_logit.from_dict(json.load(handle))

    # The one bootstrap rng, consumed sequentially by the per-map
    # resampling draws below (a fixed seed reproduces byte-identical
    # resamples and therefore byte-identical refit models).
    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)

    # The per-map refit loop (module docstring decision 1): one
    # match-level resample + one ordinal-logit fit per replicate, over
    # the train split — the same loop body
    # evaluate_bootstrap_intervals.py's per-map section runs, copied
    # (not imported) per that driver's own documented per-driver-
    # duplication precedent. Non-converged replicates are kept
    # (decision 3).
    replicate_models: list[OrdinalLogitModel] = []
    for _ in range(args.n_bootstrap_map):
        X_boot, y_boot = training_data.assemble_bootstrap_design_matrix(
            matches_df,
            maps_df,
            labels_df,
            splits_df,
            player_map_stats_df,
            bootstrap_rng,
        )
        replicate_models.append(ordinal_logit.fit(X_boot, y_boot))
    n_map_converged = sum(
        int(model.converged) for model in replicate_models
    )

    report = {
        "config": {
            "n_bootstrap_map": args.n_bootstrap_map,
            "bootstrap_seed": args.bootstrap_seed,
        },
        # Each replicate's to_dict output with the derived
        # "coefficient_report" key stripped (an explicit
        # dict-comprehension after calling to_dict — to_dict's
        # signature/behavior is unchanged; from_dict ignores the key on
        # read regardless, so stripping it only shrinks the file).
        "replicates": [
            {
                key: value
                for key, value in ordinal_logit.to_dict(model).items()
                if key != "coefficient_report"
            }
            for model in replicate_models
        ],
        "base_ordinal_thresholds": [
            float(threshold) for threshold in base_model.thresholds
        ],
    }

    artifact_path = (
        output_dir / args.version / "ordinal_bootstrap_replicates.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "persisted %d per-map bootstrap replicates (%d/%d converged) "
        "with bootstrap_seed=%d to %s (%s/%s)",
        args.n_bootstrap_map,
        n_map_converged,
        args.n_bootstrap_map,
        args.bootstrap_seed,
        artifact_path,
        output_dir,
        args.version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

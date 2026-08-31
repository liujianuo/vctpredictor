"""Command-line training of the temperature-scaling model (roadmap M24).

Thin command-line wrapper around :mod:`models.temperature_scaling`,
which owns the pure math (the ``eta / T`` category-probability formula,
the two-stage log-scale grid-search fit, serialization). This module
adds only the CLI/IO glue: argument parsing (:func:`parse_args`),
loading the five input tables (reusing the four ``load_*_table``
helpers from :mod:`drivers.evaluate` plus that module's
:func:`drivers.evaluate.load_player_map_stats_table`), assembling the
leakage-safe walk-forward out-of-fold calibration rows via
:func:`drivers.training_data.assemble_out_of_fold_eta_rows` (decision C
of the M24 plan: per-fold ordinal-logit refits, per-fold thresholds
during the fit), loading the existing final M20 artifact
``ordinal_logit_model.json``, calling
:func:`models.temperature_scaling.fit_temperature`, combining the fit
result with the base model's thresholds (the provenance copy of
decision E) and the OOF coverage dict into the full
:class:`models.temperature_scaling.TemperatureScaledModel`, and writing
the serialized artifact ``data/<version>/temperature_scaling_model.json``.

**Prerequisite:** ``drivers/train_ordinal_logit.py`` must already have
been run for the requested version (the final M20 artifact is loaded
for its ``thresholds`` provenance copy and as the "run
train_ordinal_logit.py first" signal). The base artifact is loaded
*before* the OOF assembly so a missing artifact fails fast rather than
after the (expensive) per-fold refit loop; this script never retrains
the base model.

Artifact written per run (scoped by dataset version):

- ``data/<version>/temperature_scaling_model.json`` — the
  :func:`models.temperature_scaling.to_dict` dict, written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing newline
  (the same serialization convention as every other artifact in this
  repo).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine: a missing
  input table, a missing base-model artifact, an empty OOF set, or a
  feature-computation failure all propagate as exceptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from drivers import evaluate, training_data
from models import ordinal_logit, temperature_scaling
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the train_temperature_scaling.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with two attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``) and ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``). Together
        they locate the five input tables, the base
        ``ordinal_logit_model.json`` artifact and the output artifact
        ``<output_dir>/<version>/temperature_scaling_model.json``.
        There are deliberately no hyperparameter flags: the grid
        defaults (``t_min=0.05, t_max=20.0, n_coarse=97, n_fine=61``)
        and the walk-forward fold defaults are the documented library
        defaults of :func:`models.temperature_scaling.fit_temperature` /
        :func:`drivers.training_data.assemble_out_of_fold_eta_rows`
        (decisions C/F of the M24 plan), matching the other drivers'
        no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit the one-parameter temperature scaling (eta / T, "
            "thresholds unchanged) on the walk-forward out-of-fold "
            "calibration set and write temperature_scaling_model.json."
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


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the temperature-scaling model end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table`), the base M20
    artifact is loaded (``ordinal_logit_model.json`` — the
    prerequisite; its ``thresholds`` become the calibration artifact's
    provenance copy, and its absence raises ``FileNotFoundError`` as
    the "run train_ordinal_logit.py first" signal), the walk-forward
    OOF calibration rows are assembled
    (:func:`drivers.training_data.assemble_out_of_fold_eta_rows` —
    per-fold refits with per-fold thresholds, the leakage-safe set),
    the temperature is fit
    (:func:`models.temperature_scaling.fit_temperature` on the OOF
    ``eta``/``thresholds_per_row``/``outcome_ordinal`` columns), the
    full :class:`models.temperature_scaling.TemperatureScaledModel` is
    constructed (fit result + the base model's thresholds + the OOF
    coverage dict), the artifact is written as
    ``<output_dir>/<version>/temperature_scaling_model.json``, and a
    one-line summary (``temperature``, ``n_calibration``,
    ``calibration_nll_at_t1``, ``calibration_nll_at_t_star``) is
    logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the five input tables, or the
            base ``ordinal_logit_model.json`` artifact, does not exist
            for the requested version — the artifact propagates as-is
            from ``json.load`` as a clear "run train_ordinal_logit.py
            first" signal (loaded before the OOF assembly so it fails
            fast); the temperature driver never retrains the base model.
        ValueError: If ``splits_df`` has no ``"train"`` rows, if the
            training region is too small to form one walk-forward fold,
            if a fold's validation block is empty, if the OOF
            submission fails the assembler's leak check, if a label or
            feature computation fails, or if the fit input is
            malformed (all propagated from
            :func:`drivers.training_data.assemble_out_of_fold_eta_rows`
            / :func:`models.temperature_scaling.fit_temperature` /
            :func:`models.ordinal_logit.from_dict`).
        KeyError: If any input table or model artifact lacks a required
            key/column (propagated from the feature modules / harness /
            the two ``from_dict`` functions).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The prerequisite: load the final M20 artifact before the (slower)
    # OOF assembly so a missing artifact fails fast. Its thresholds are
    # the provenance copy the calibration artifact carries (decision E).
    base_path = output_dir / args.version / "ordinal_logit_model.json"
    with open(base_path, encoding="utf-8") as handle:
        base_model = ordinal_logit.from_dict(json.load(handle))

    oof_df, coverage = training_data.assemble_out_of_fold_eta_rows(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )

    etas = oof_df["eta"].to_numpy(dtype=float)
    thresholds_per_row = oof_df[["theta1", "theta2", "theta3"]].to_numpy(
        dtype=float
    )
    y = oof_df["outcome_ordinal"].to_numpy(dtype=int)
    fit_result = temperature_scaling.fit_temperature(
        etas, thresholds_per_row, y
    )

    model = temperature_scaling.TemperatureScaledModel(
        temperature=fit_result["temperature"],
        thresholds=base_model.thresholds,
        n_calibration=fit_result["n_calibration"],
        oof_coverage=coverage,
        t_grid_min=fit_result["t_grid_min"],
        t_grid_max=fit_result["t_grid_max"],
        calibration_nll_at_t1=fit_result["calibration_nll_at_t1"],
        calibration_nll_at_t_star=fit_result["calibration_nll_at_t_star"],
    )

    artifact_path = (
        output_dir / args.version / "temperature_scaling_model.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(temperature_scaling.to_dict(model), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "fitted temperature scaling on %d OOF rows (%s/%s): "
        "temperature=%.6f calibration_nll_at_t1=%.6f "
        "calibration_nll_at_t_star=%.6f",
        model.n_calibration,
        output_dir,
        args.version,
        model.temperature,
        model.calibration_nll_at_t1,
        model.calibration_nll_at_t_star,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

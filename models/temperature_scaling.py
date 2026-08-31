"""One-parameter temperature scaling of the M20 ordinal latent score (roadmap M24).

A single fitted scalar ``T > 0`` that rescales the M20 ordinal logit's
latent score ``eta = x . beta`` *before* the three thresholds are
applied, fit on the leakage-safe walk-forward out-of-fold calibration
set (M10's :func:`utils.splits.assemble_out_of_fold_predictions`
assembly, driven from :func:`drivers.training_data.assemble_out_of_fold_eta_rows`).
This is the M24 implementation of the roadmap's "One-parameter
Platt-style rescaling of the ordinal latent score before applying
thresholds" — no resampling, no class reweighting anywhere (the
roadmap's explicit prohibition: "those improve accuracy while
destroying the base rate").

This module is deliberately pure math (like its sibling
``models.ordinal_logit``): it operates on already-computed ``eta`` /
``thresholds`` / ``y`` arrays, never on raw feature vectors or a fitted
``OrdinalLogitModel``. All file I/O, table joining, and the
per-fold refit-and-collect loop live in the drivers. The design
decisions below are recorded verbatim from the M24 plan so later
milestones do not re-derive them.

**Decision A — Rescaling form (recorded verbatim, do not re-derive).**
``eta' = eta / T`` with ``T > 0``; the three thresholds are *not*
rescaled — the literal reading of "rescale the latent score ... before
applying thresholds" (thresholds are applied to the already-rescaled
score, unchanged themselves). So ``C_j = sigmoid(theta_j + eta / T)``,
``P0 = C1``, ``P1 = C2 - C1``, ``P2 = C3 - C2``, ``P3 = 1 - C3``, each
clipped into ``[eps, 1-eps]`` via :data:`models._shared._PROB_CLIP_EPS`
— the exact formula in ``models/ordinal_logit.py``'s
``_category_probabilities`` with ``eta`` replaced by ``eta / T``. At
``T = 1`` this recovers the M20 model exactly (locked in by a test).
This is a **second, independent implementation** of that formula (not an
import of ``ordinal_logit._category_probabilities``, which is private
and which module-boundary rules forbid reaching into anyway) — matching
the established repo convention of independent reimplementation over
sibling-model reuse (``multinomial_logit``, ``binary_logit``,
``proportional_odds``'s cutpoint fits are all independent for the same
reason).

**Decision D — Module placement (module-boundary standard).** This
module is added to ``MODELS_MODULES`` in
``tests/test_module_boundaries.py``. It must **not** import
``models.ordinal_logit`` (only ``models._shared`` is an allowed lateral
import per the existing boundary test) — it operates generically on
already-computed ``eta`` / ``thresholds`` / ``y`` arrays, never on raw
feature vectors or a fitted ``OrdinalLogitModel``. The walk-forward
refit-and-collect loop (decision C of the M24 plan) lives in
``drivers/training_data.py``; the two driver scripts and the
``evaluation/temperature_calibration.py`` comparison module complete
the DAG:
``utils/splits, models/_shared -> models/temperature_scaling.py``;
``evaluation/harness, models/ordinal_logit, models/_shared, utils/splits
-> drivers/training_data.py``; drivers import freely.

**Decision F — Grid-search fit (no scipy in this repo; recorded
verbatim).** Reparameterize on a log scale to keep ``T > 0`` by
construction. Coarse grid: ``T_coarse =
concatenate(geomspace(0.05, 20.0, 97), [1.0])`` — ``1.0`` is included
explicitly so the fitted ``T*``'s OOF NLL is *guaranteed* ``<=`` the
uncalibrated (``T=1``) OOF NLL, a checkable invariant. Objective:
``NLL(T) = sum_i -log(clip(P_{y_i}(eta_i, thresholds_i, T)))`` over the
OOF rows. Take ``T0 = argmin`` over the coarse grid, then refine over a
fine grid ``geomspace(T0/1.5, T0*1.5, 61)`` clipped to ``[0.05, 20.0]``,
and return the overall argmin across both grids as ``T*``. Record
``t_grid_min=0.05``, ``t_grid_max=20.0``,
``calibration_nll_at_t1`` (the coarse grid's ``T=1.0`` evaluation) and
``calibration_nll_at_t_star`` in the artifact.

**Invariance is measured, not assumed (decision B of the M24 plan).**
The task text suggests temperature scaling is monotone so
accuracy/RPS-argmax are invariant; that is **not generally true for
this model** and must not be asserted as fact: ``eta / T`` is monotone
in ``eta`` for fixed ``T > 0``, but the marginal A/B decision boundary
is ``theta_2 + eta/T > 0``, i.e. ``eta > -theta_2 * T`` — a boundary
that itself *moves* with ``T`` (unless ``theta_2 == 0``, which it is
not: the M20 artifact's ``thresholds[1]`` is ``-0.0671``). Whether the
binary side call / 4-way argmax happen to stay fixed on the actual v1
test split is an empirical question that
:func:`evaluation.temperature_calibration.build_calibration_comparison_report`
measures and reports; this module only fits ``T`` against NLL and makes
no claim about accuracy/RPS invariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from models._shared import _PROB_CLIP_EPS, _sigmoid

# The four outcome categories in ordinal order, mirroring
# ``models._shared.OUTCOME_LABELS`` (kept in sync, deliberately not
# imported as a name from a sibling models/ module beyond _shared).
_N_CATEGORIES = 4

# Decision F's documented grid defaults, recorded verbatim (do not
# re-derive): the coarse grid spans geomspace(0.05, 20.0, 97) plus an
# explicit 1.0; the fine grid is 61 points over [T0/1.5, T0*1.5]
# clipped into [t_min, t_max].
_DEFAULT_T_MIN = 0.05
_DEFAULT_T_MAX = 20.0
_DEFAULT_N_COARSE = 97
_DEFAULT_N_FINE = 61


@dataclass(frozen=True)
class TemperatureScaledModel:
    """A fitted one-parameter temperature-scaled ordinal model.

    Holds the fitted scalar ``temperature`` plus the metadata needed to
    (a) apply the scaling at prediction time via
    :func:`predict_proba_with_temperature`, and (b) serialize/
    deserialize via :func:`to_dict` / :func:`from_dict`.
    ``thresholds`` is a **provenance copy** (decision E of the M24
    plan): the base M20 model's thresholds at calibration-fit time,
    used only as a staleness guard — any consumer must ``np.allclose``-
    check the loaded base model's thresholds against this stored copy
    and raise ``ValueError`` on mismatch rather than silently applying a
    stale ``T`` (that check lives in the consumer,
    :func:`drivers.evaluate._ordinal_logit_temperature_factory`, not
    here). ``oof_coverage`` is the coverage dict from
    :func:`utils.splits.assemble_out_of_fold_predictions` (plain
    int/list fields so it round-trips through JSON). The two NLL fields
    record decision F's checkable invariant
    ``calibration_nll_at_t_star <= calibration_nll_at_t1``.

    Attributes:
        temperature: The fitted scalar ``T > 0`` (the only free
            parameter).
        thresholds: The 3-vector of the base M20 model's thresholds at
            calibration-fit time (provenance copy, strictly increasing
            for the base model; used for the staleness guard).
        n_calibration: Number of OOF calibration rows ``T`` was fit on.
        oof_coverage: The coverage dict from
            :func:`utils.splits.assemble_out_of_fold_predictions`
            (``train_matches``/``covered_matches``/``warmup_excluded_ids``/
            ``warmup_excluded_count``), stored as plain int/list fields.
        t_grid_min: The lower grid bound (``0.05``).
        t_grid_max: The upper grid bound (``20.0``).
        calibration_nll_at_t1: The OOF NLL at ``T = 1`` (the
            uncalibrated baseline, from the coarse grid's explicit
            ``1.0`` point).
        calibration_nll_at_t_star: The OOF NLL at the fitted ``T*``
            (always ``<= calibration_nll_at_t1`` by construction).
    """

    temperature: float
    thresholds: np.ndarray
    n_calibration: int
    oof_coverage: dict
    t_grid_min: float
    t_grid_max: float
    calibration_nll_at_t1: float
    calibration_nll_at_t_star: float


def predict_proba_with_temperature(
    eta: float,
    thresholds: np.ndarray,
    temperature: float,
) -> tuple[float, float, float, float]:
    """Return the four category probabilities under temperature scaling.

    Implements decision A of the module docstring exactly: with
    ``scaled_eta = eta / T``, ``C_j = sigmoid(theta_j + scaled_eta)``,
    ``P0 = C1``, ``P1 = C2 - C1``, ``P2 = C3 - C2``, ``P3 = 1 - C3``.
    Each probability is clipped into ``[eps, 1 - eps]`` (see
    :data:`models._shared._PROB_CLIP_EPS`) before being returned, so
    the caller can take a log without hitting ``-inf``. At ``T = 1``
    this reproduces ``models.ordinal_logit._category_probabilities``
    exactly (same sigmoid calls, same clip) — locked in by a test.

    Args:
        eta: The scalar linear predictor ``x . beta`` of the base M20
            model (standardized feature vector dotted with the base
            model's coefficients).
        thresholds: The 3-vector of strictly increasing thresholds
            ``(theta_1, theta_2, theta_3)`` — the base M20 model's
            thresholds (decision E: never a fold model's thresholds at
            application time).
        temperature: The positive scalar ``T`` dividing ``eta`` before
            the thresholds are applied.

    Returns:
        A 4-tuple of ``float`` probabilities in
        :data:`models._shared.OUTCOME_LABELS` order, each in
        ``[eps, 1 - eps]``, summing to approximately 1.

    Raises:
        ValueError: If ``temperature <= 0`` (the scaled latent score
            would be undefined/order-inverted for a non-positive
            temperature), or if ``thresholds`` does not have exactly 3
            elements (a wrong threshold count would silently misalign
            the category boundaries).
    """
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be strictly positive, got {temperature!r}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"expected 3 thresholds, got {len(thresholds)}"
        )
    scaled_eta = eta / temperature
    c1 = _sigmoid(float(thresholds[0]) + scaled_eta)
    c2 = _sigmoid(float(thresholds[1]) + scaled_eta)
    c3 = _sigmoid(float(thresholds[2]) + scaled_eta)
    probs = np.clip(
        np.asarray([c1, c2 - c1, c3 - c2, 1.0 - c3], dtype=float),
        _PROB_CLIP_EPS,
        1.0 - _PROB_CLIP_EPS,
    )
    return (
        float(probs[0]),
        float(probs[1]),
        float(probs[2]),
        float(probs[3]),
    )


def _nll_at_temperature(
    etas: np.ndarray,
    thresholds_per_row: np.ndarray,
    y: np.ndarray,
    temperature: float,
) -> float:
    """Return the total OOF negative log-likelihood at one temperature.

    Computes ``NLL(T) = sum_i -log(clip(P_{y_i}(eta_i, thresholds_i,
    T)))`` (decision F's objective) by calling
    :func:`predict_proba_with_temperature` once per row — deliberately
    through the public prediction path so there is exactly one place
    that turns ``(eta, thresholds, T)`` into probabilities, exactly as
    :func:`models.ordinal_logit.total_log_likelihood` does for its own
    model. The caller has already validated the input shapes.

    Args:
        etas: The OOF linear predictors, ``(n,)`` floats.
        thresholds_per_row: The per-row thresholds, ``(n, 3)`` floats
            (each row's own fold model's thresholds during fitting —
            decision C of the M24 plan).
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        temperature: The candidate ``T > 0``.

    Returns:
        The total NLL as a non-negative ``float``.

    Raises:
        ValueError: If ``temperature <= 0`` or a row's thresholds are
            not length 3 (propagated from
            :func:`predict_proba_with_temperature`); the clip keeps
            every log finite, so no other error is possible.
    """
    total = 0.0
    for i in range(etas.shape[0]):
        probs = predict_proba_with_temperature(
            etas[i], thresholds_per_row[i], temperature
        )
        total += -math.log(probs[y[i]])
    return total


def fit_temperature(
    etas: np.ndarray,
    thresholds_per_row: np.ndarray,
    y: np.ndarray,
    t_min: float = _DEFAULT_T_MIN,
    t_max: float = _DEFAULT_T_MAX,
    n_coarse: int = _DEFAULT_N_COARSE,
    n_fine: int = _DEFAULT_N_FINE,
) -> dict:
    """Fit the one-parameter temperature by two-stage log-scale grid search.

    Implements decision F of the module docstring: builds the coarse
    grid ``concatenate(geomspace(t_min, t_max, n_coarse), [1.0])`` (the
    explicit ``1.0`` guarantees the uncalibrated ``T = 1`` point is
    always evaluated — so the returned ``calibration_nll_at_t_star`` is
    *guaranteed* ``<= calibration_nll_at_t1``, a checkable invariant),
    evaluates the OOF NLL (:func:`_nll_at_temperature`) at every coarse
    point, takes ``T0 = argmin``, refines over the fine grid
    ``clip(geomspace(T0 / 1.5, T0 * 1.5, n_fine), t_min, t_max)``, and
    returns the overall argmin across both grids as ``T*``. The
    returned dict is a plain JSON-serializable structure (no
    dataclass); the calling driver combines it with the base model's
    thresholds and the OOF coverage dict to build the full
    :class:`TemperatureScaledModel`.

    Args:
        etas: The OOF linear predictors, ``(n,)`` floats.
        thresholds_per_row: The per-row thresholds, ``(n, 3)`` floats
            (each row's own fold model's thresholds during fitting —
            decision C of the M24 plan: fold-specific thresholds while
            fitting ``T``, fixed final-model thresholds while applying
            it; this asymmetry is intentional and documented there).
        y: The true outcome ordinals, ``(n,)`` ints in ``{0, 1, 2, 3}``.
        t_min: The lower grid bound (default ``0.05``; strictly
            positive, since the grid is log-spaced).
        t_max: The upper grid bound (default ``20.0``; must exceed
            ``t_min``).
        n_coarse: The number of coarse grid points (default 97).
        n_fine: The number of fine grid points (default 61).

    Returns:
        A dict with keys ``temperature`` (the fitted ``T*``), 
        ``n_calibration`` (the OOF row count ``n``), ``t_grid_min``,
        ``t_grid_max``, ``calibration_nll_at_t1`` (the coarse grid's
        ``T = 1.0`` evaluation) and ``calibration_nll_at_t_star`` (the
        NLL at ``T*``; ``<= calibration_nll_at_t1`` by construction).
        Every value is a plain float/int, directly ``json.dumps``-
        serializable.

    Raises:
        ValueError: If ``etas``/``y`` are empty, if
            ``thresholds_per_row`` is not an ``(n, 3)`` array, if the
            three arrays have inconsistent row counts, if ``y`` contains
            a value outside ``{0, 1, 2, 3}``, if ``t_min``/``t_max``
            are not finite with ``0 < t_min < t_max``, or if either
            grid-point count is below 2 (a grid with fewer than two
            points cannot be searched). Also propagates
            :func:`predict_proba_with_temperature`'s ``ValueError`` for
            a non-positive candidate temperature (cannot occur: every
            candidate is positive by construction).
    """
    etas_arr = np.asarray(etas, dtype=float).ravel()
    thresh_arr = np.asarray(thresholds_per_row, dtype=float)
    y_arr = np.asarray(y, dtype=int).ravel()

    if etas_arr.size == 0:
        raise ValueError(
            "cannot fit a temperature on an empty eta vector"
        )
    if thresh_arr.ndim != 2 or thresh_arr.shape[1] != 3:
        raise ValueError(
            "thresholds_per_row must be an (n, 3) array of per-row "
            f"thresholds, got shape {thresh_arr.shape}"
        )
    if thresh_arr.shape[0] != etas_arr.shape[0] or y_arr.shape[0] != etas_arr.shape[0]:
        raise ValueError(
            f"etas ({etas_arr.shape[0]}), thresholds_per_row "
            f"({thresh_arr.shape[0]}) and y ({y_arr.shape[0]}) must all "
            "have the same row count"
        )
    if set(np.unique(y_arr).tolist()) - set(range(_N_CATEGORIES)):
        raise ValueError(
            f"y must contain only outcome ordinals 0..{_N_CATEGORIES - 1}, "
            f"got values {sorted(set(np.unique(y_arr).tolist()))}"
        )
    try:
        t_min_f = float(t_min)
        t_max_f = float(t_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"t_min/t_max must be real numbers, got t_min={t_min!r}, "
            f"t_max={t_max!r}"
        ) from exc
    if (
        not math.isfinite(t_min_f)
        or not math.isfinite(t_max_f)
        or not (0.0 < t_min_f < t_max_f)
    ):
        raise ValueError(
            f"t_min/t_max must be finite with 0 < t_min < t_max, got "
            f"t_min={t_min_f!r}, t_max={t_max_f!r}"
        )
    if int(n_coarse) < 2 or int(n_fine) < 2:
        raise ValueError(
            f"n_coarse ({n_coarse}) and n_fine ({n_fine}) must each be "
            "at least 2 grid points"
        )

    # Coarse grid: geomspace(t_min, t_max, n_coarse) plus the explicit
    # 1.0 appended at the end — the appended point guarantees the
    # uncalibrated NLL is always evaluated, making the returned
    # invariant (nll_at_t_star <= nll_at_t1) structural.
    coarse = np.concatenate(
        [np.geomspace(t_min_f, t_max_f, int(n_coarse)), np.asarray([1.0])]
    )
    coarse_nll = np.asarray(
        [
            _nll_at_temperature(etas_arr, thresh_arr, y_arr, t)
            for t in coarse
        ],
        dtype=float,
    )
    coarse_best_idx = int(np.argmin(coarse_nll))
    t0 = float(coarse[coarse_best_idx])
    # The coarse grid's T=1.0 evaluation: 1.0 was appended explicitly,
    # so this index is exact (a geomspace point that also equals 1.0
    # would only tie at the same value).
    t1_idx = int(np.argmin(np.abs(coarse - 1.0)))
    calibration_nll_at_t1 = float(coarse_nll[t1_idx])

    # Fine grid around T0, clipped back into [t_min, t_max].
    fine = np.clip(
        np.geomspace(t0 / 1.5, t0 * 1.5, int(n_fine)),
        t_min_f,
        t_max_f,
    )
    fine_nll = np.asarray(
        [
            _nll_at_temperature(etas_arr, thresh_arr, y_arr, t)
            for t in fine
        ],
        dtype=float,
    )
    fine_best_idx = int(np.argmin(fine_nll))

    # Overall argmin across both grids.
    if fine_nll[fine_best_idx] < coarse_nll[coarse_best_idx]:
        temperature = float(fine[fine_best_idx])
        calibration_nll_at_t_star = float(fine_nll[fine_best_idx])
    else:
        temperature = float(coarse[coarse_best_idx])
        calibration_nll_at_t_star = float(coarse_nll[coarse_best_idx])

    return {
        "temperature": temperature,
        "n_calibration": int(etas_arr.shape[0]),
        "t_grid_min": t_min_f,
        "t_grid_max": t_max_f,
        "calibration_nll_at_t1": calibration_nll_at_t1,
        "calibration_nll_at_t_star": calibration_nll_at_t_star,
    }


def to_dict(model: TemperatureScaledModel) -> dict:
    """Serialize a temperature-scaled model to a plain JSON-serializable dict.

    Produces the artifact dict the training driver writes:
    ``temperature``, ``thresholds`` (the provenance copy), 
    ``n_calibration``, ``oof_coverage``, ``t_grid_min``,
    ``t_grid_max``, ``calibration_nll_at_t1`` and
    ``calibration_nll_at_t_star``. Every value is a plain
    str/int/float/list/dict so ``json.dumps`` accepts the dict directly
    (with ``sort_keys=True`` for deterministic artifacts), mirroring
    ``models.ordinal_logit.to_dict``'s plain-type convention exactly.
    No file I/O happens here.

    Args:
        model: The temperature-scaled model to serialize.

    Returns:
        A plain dict as described, directly ``json.dumps``-serializable.

    Raises:
        Nothing (all fields are already plain types or numpy arrays
            converted here).
    """
    return {
        "temperature": float(model.temperature),
        "thresholds": [float(t) for t in model.thresholds],
        "n_calibration": int(model.n_calibration),
        "oof_coverage": {
            str(key): value for key, value in model.oof_coverage.items()
        },
        "t_grid_min": float(model.t_grid_min),
        "t_grid_max": float(model.t_grid_max),
        "calibration_nll_at_t1": float(model.calibration_nll_at_t1),
        "calibration_nll_at_t_star": float(model.calibration_nll_at_t_star),
    }


def from_dict(d: dict) -> TemperatureScaledModel:
    """Deserialize a temperature-scaled model from a to_dict-produced dict.

    Reconstructs a :class:`TemperatureScaledModel` from the plain dict
    :func:`to_dict` produces (or from ``json.loads`` of the artifact
    the training driver writes). Shape/range consistency is validated:
    ``temperature`` must be a positive finite float (a non-positive
    ``T`` would make :func:`predict_proba_with_temperature` raise at
    every call), and ``thresholds`` must have exactly 3 entries. No
    file I/O happens here.

    Args:
        d: The dict to load; must carry all eight documented keys.

    Returns:
        A :class:`TemperatureScaledModel` whose fields reproduce the
        serialized ones (``thresholds`` as a numpy array, ``oof_coverage``
        as a plain dict).

    Raises:
        KeyError: If a required key is absent (propagated from dict
            indexing, matching ``ordinal_logit.from_dict``'s error
            style).
        ValueError: If ``temperature`` is not a positive finite float,
            or if ``thresholds`` does not have exactly 3 entries (a
            shape or range inconsistency in a hand-built or stale
            artifact).
    """
    temperature = float(d["temperature"])
    thresholds = np.asarray(d["thresholds"], dtype=float)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"temperature must be a positive finite float, got "
            f"{temperature!r}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"expected 3 thresholds, got {len(thresholds)}"
        )
    return TemperatureScaledModel(
        temperature=temperature,
        thresholds=thresholds,
        n_calibration=int(d["n_calibration"]),
        oof_coverage=dict(d["oof_coverage"]),
        t_grid_min=float(d["t_grid_min"]),
        t_grid_max=float(d["t_grid_max"]),
        calibration_nll_at_t1=float(d["calibration_nll_at_t1"]),
        calibration_nll_at_t_star=float(d["calibration_nll_at_t_star"]),
    )

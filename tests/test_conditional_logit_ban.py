"""Tests for the conditional-logit ban model (M27).

Covers the 5-feature ban vector builder (hand-computed values on a
small synthetic league), the softmax scoring path (a hand-built model
must return exactly ``softmax(feature . beta)`` and the
no-intercept/internally-consistent softmax property across shrinking
candidate subsets), the analytic-vs-finite-difference gradient check
(the highest-risk correctness item, at multiple points), the
fit-converges-on-separable-synthetic-data regression (a dataset where
the banned map is always the lowest-``acting_map_win_rate`` map must
fit a negative coefficient on that feature), the ``to_dict``/
``from_dict`` round-trip and JSON serializability, the opponent
resolution's zero/ambiguous ``ValueError`` paths, the wrapped
predictor's ``action != "ban"`` rejection, and the ``fit`` input
validation guards.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from models.conditional_logit_ban import (
    FEATURE_NAMES,
    ConditionalLogitBanModel,
    _softmax,
    build_ban_feature_vector,
    fit,
    from_dict,
    make_veto_step_predictor_fn,
    predict_ban_distribution,
    to_dict,
)

# The as-of cutoff every synthetic feature test uses: the hour after
# the league's last strictly-prior match, so the as-of features see all
# eight earlier maps (and the queried match m9, dated exactly at the
# cutoff, is excluded by the strict-< boundary).
QUERY_DATE = "2026-01-01T08:00:00"

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Args:
        rows: A list of dicts, one per match, each carrying the keys
            in :data:`_MATCHES_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MATCHES_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is).
    """
    return pd.DataFrame(rows, columns=_MATCHES_COLS)


def _maps_df(rows):
    """Build a maps table with the fixed M8 column set.

    Args:
        rows: A list of dicts, one per map, each carrying the keys in
            :data:`_MAPS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MAPS_COLS` columns.

    Raises:
        Nothing.
    """
    return pd.DataFrame(rows, columns=_MAPS_COLS)


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        mid: The shared ``match_id`` for the new match and map.
        date: The match's ISO date string.
        team1_id: The match's team1 stable id.
        team2_id: The match's team2 stable id.
        map_name: The finished map's name.
        t1s: Rounds team1 won.
        t2s: Rounds team2 won.

    Returns:
        Nothing (appends in place).

    Raises:
        Nothing.
    """
    match_rows.append(
        {
            "match_id": mid,
            "date": date,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "status": "completed",
        }
    )
    map_rows.append(
        {
            "match_id": mid,
            "map_index": 0,
            "map_name": map_name,
            "team1_score": t1s,
            "team2_score": t2s,
            "winner": team1_id if t1s > t2s else team2_id,
        }
    )


def _stamp(i):
    """Return the ISO timestamp ``i`` hours after a fixed 2026-01-01 base.

    Args:
        i: The hour offset from the base.

    Returns:
        An ISO-8601 datetime string.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    return (base + pd.Timedelta(hours=i)).isoformat()


def _ban_league_tables():
    """Build the hand-computable ban-feature league.

    Team ``A`` plays Haven twice (1 win) and Split twice (0 wins):
    overall 4 games / 1 win -> prior 0.25, Haven mean
    ``(1 + 10*0.25) / 12 = 3.5/12``, Split mean ``2.5/12``. Team ``B``
    plays Ascent twice (2 wins) and Sunset twice (1 win): overall
    ``0.75``, and full shrinkage to ``0.75`` on any map it has never
    played (e.g. Haven). All eight matches are dated before
    :data:`QUERY_DATE`; a ninth match (m9, A vs B on Haven at exactly
    the query cutoff) is excluded from every as-of feature by the
    strict-``<`` boundary and exists so the opponent-resolution
    positive path has a live match to resolve.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 9 matches and 9 maps.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", _stamp(0), "A", "o1", "Haven", 13, 11)
    _add(match_rows, map_rows, "m2", _stamp(1), "A", "o2", "Haven", 8, 13)
    _add(match_rows, map_rows, "m3", _stamp(2), "A", "o3", "Split", 5, 13)
    _add(match_rows, map_rows, "m4", _stamp(3), "A", "o4", "Split", 9, 13)
    _add(match_rows, map_rows, "m5", _stamp(4), "B", "o5", "Ascent", 13, 9)
    _add(match_rows, map_rows, "m6", _stamp(5), "B", "o6", "Ascent", 13, 10)
    _add(match_rows, map_rows, "m7", _stamp(6), "B", "o7", "Sunset", 13, 11)
    _add(match_rows, map_rows, "m8", _stamp(7), "B", "o8", "Sunset", 6, 13)
    # The queried match itself, dated exactly at the cutoff: excluded
    # from features (strict <), present for opponent resolution.
    _add(match_rows, map_rows, "m9", QUERY_DATE, "A", "B", "Haven", 13, 10)
    return _matches_df(match_rows), _maps_df(map_rows)


def _small_fitted_model():
    """Fit a small deterministic synthetic model for serialization tests.

    Uses a fixed seed so the fitted parameters are stable across runs;
    the model itself is only exercised for round-tripping and closure
    agreement, not for predictive quality.

    Returns:
        A :class:`ConditionalLogitBanModel` fit on 80 synthetic groups
        of 4-6 candidate rows each.

    Raises:
        Nothing.
    """
    rng = np.random.default_rng(5)
    bounds = [0]
    ys = []
    rows = []
    for _ in range(80):
        n_s = int(rng.integers(4, 7))
        xs = rng.normal(scale=0.5, size=(n_s, len(FEATURE_NAMES)))
        rows.append(xs)
        ys.append(int(rng.integers(0, n_s)))
        bounds.append(bounds[-1] + n_s)
    X = np.vstack(rows)
    return fit(np.asarray(X), np.asarray(bounds), np.asarray(ys), max_iter=150)


def _ban_loss(Xs, bounds, ys, beta, l2_lambda):
    """Evaluate the ragged-group objective at one parameter point.

    A thin test-side wrapper over the module's own
    :func:`models.conditional_logit_ban._loss_and_gradient` returning
    only the scalar loss, so the finite-difference helper can
    differentiate exactly the objective the analytic gradient describes.

    Args:
        Xs: The (already-standardized) flattened design matrix.
        bounds: The group-offset array.
        ys: The per-group true-row indices.
        beta: The coefficient vector at which to evaluate.
        l2_lambda: The L2 strength.

    Returns:
        The scalar objective ``mean(NLL) + (l2_lambda/2) * sum(beta^2)``.

    Raises:
        ValueError: If shapes are inconsistent (propagated from
            :func:`models.conditional_logit_ban._loss_and_gradient`).
    """
    from models.conditional_logit_ban import _loss_and_gradient

    return _loss_and_gradient(Xs, bounds, ys, beta, l2_lambda)[0]


def _numeric_gradient(Xs, bounds, ys, beta, l2_lambda, eps):
    """Central finite-difference gradient of the batch objective.

    The independent numerical check the analytic gradient is tested
    against: perturbs each coefficient by ``±eps``, evaluates the
    objective via :func:`_ban_loss`, and takes the centered difference.
    The objective is the same clipped-NLL-plus-L2 function the analytic
    gradient differentiates, so at points where the clip is inactive
    (all test points) the two must agree to the finite-difference
    accuracy.

    Args:
        Xs: The (already-standardized) flattened design matrix.
        bounds: The group-offset array.
        ys: The per-group true-row indices.
        beta: The coefficient vector at which to differentiate.
        l2_lambda: The L2 strength.
        eps: The finite-difference step size.

    Returns:
        A ``(p,)`` numpy array of numerical gradient components.

    Raises:
        ValueError: If shapes are inconsistent (propagated from
            :func:`_ban_loss`).
    """
    beta_arr = np.asarray(beta, dtype=float).ravel()
    grad = np.empty(len(beta_arr))
    for i in range(len(beta_arr)):
        plus = beta_arr.copy()
        plus[i] += eps
        minus = beta_arr.copy()
        minus[i] -= eps
        grad[i] = (
            _ban_loss(Xs, bounds, ys, plus, l2_lambda)
            - _ban_loss(Xs, bounds, ys, minus, l2_lambda)
        ) / (2.0 * eps)
    return grad


def _separable_ban_dataset(rng, n_groups, n_maps):
    """Build the separable synthetic ban dataset for the fit regression.

    Constructs ``n_groups`` ban steps, each over ``n_maps`` candidate
    rows: every row is noise except feature 0 (``acting_map_win_rate``),
    which is fixed at ``1.0, 0.8, 0.6, 0.4, 0.2`` plus tiny noise; the
    banned map of each group is always the row with the lowest feature
    0. Because standardization is a per-column monotone transform, the
    lowest raw feature 0 row stays the lowest standardized feature 0
    row, so a correctly-fitting model must learn a *negative*
    coefficient on feature 0 (lower acting win rate -> more bannable).

    Args:
        rng: A seeded ``numpy.random.Generator``.
        n_groups: The number of ban steps to generate.
        n_maps: The candidate-set size of every step.

    Returns:
        A ``(X_flat, bounds, ys)`` tuple: the raw flattened design
        matrix ``(n_groups * n_maps, 5)``, the ``(n_groups + 1,)``
        group-offset array, and the ``(n_groups,)`` true-row indices.

    Raises:
        ValueError: If ``n_maps < 1`` or ``n_groups < 1`` (nothing to
            construct).
    """
    if n_maps < 1 or n_groups < 1:
        raise ValueError("separable dataset needs at least one group and one map per group")
    bounds = [0]
    ys = []
    rows = []
    base_values = np.linspace(1.0, 0.2, n_maps)
    for _ in range(n_groups):
        xs = rng.normal(scale=0.3, size=(n_maps, len(FEATURE_NAMES)))
        xs[:, 0] = base_values + rng.normal(scale=0.05, size=n_maps)
        rows.append(xs)
        ys.append(int(np.argmin(xs[:, 0])))
        bounds.append(bounds[-1] + n_maps)
    return np.vstack(rows), np.asarray(bounds), np.asarray(ys, dtype=int)


# --------------------------------------------------------------------------
# plan#3k: build_ban_feature_vector hand-computed values
# --------------------------------------------------------------------------


def test_build_ban_feature_vector_hand_computed():
    # The full 5-vector at QUERY_DATE, every entry hand-computed from
    # the fixture league:
    #   acting_map_win_rate 3.5/12 (A: 1 win / 2 Haven games, prior
    #     0.25, k=10),
    #   opponent_map_win_rate 0.75 (B has no Haven games -> full
    #     shrinkage to B's overall 0.75),
    #   acting_map_specialization 3.5/12 - 0.25 = 0.5/12,
    #   map_round_margin_variance 4.5 (Haven margins [2, 5],
    #     ddof=1),
    #   map_ot_rate 0.0 (no OT maps anywhere in the as-of pool).
    matches_df, maps_df = _ban_league_tables()
    vec = build_ban_feature_vector("A", "B", "Haven", QUERY_DATE, matches_df, maps_df)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec[0] == pytest.approx(3.5 / 12.0)
    assert vec[1] == pytest.approx(0.75)
    assert vec[2] == pytest.approx(0.5 / 12.0)
    assert vec[3] == pytest.approx(4.5)
    assert vec[4] == pytest.approx(0.0)


def test_build_ban_feature_vector_second_map_crosscheck():
    # The Split vector cross-checks the same league from a different
    # angle: A's Split mean 2.5/12, specialization -0.5/12 (Split is
    # weaker than A's baseline), variance 8.0 (margins [8, 4]).
    matches_df, maps_df = _ban_league_tables()
    vec = build_ban_feature_vector("A", "B", "Split", QUERY_DATE, matches_df, maps_df)
    assert vec[0] == pytest.approx(2.5 / 12.0)
    assert vec[1] == pytest.approx(0.75)
    assert vec[2] == pytest.approx(-0.5 / 12.0)
    assert vec[3] == pytest.approx(8.0)
    assert vec[4] == pytest.approx(0.0)


def test_ban_feature_vector_excludes_queried_match():
    # m9 (A vs B on Haven) is dated exactly at the cutoff: the strict-<
    # boundary must keep it out of every estimate. If it leaked, A's
    # overall would become 2/5 and the Haven mean would change.
    matches_df, maps_df = _ban_league_tables()
    vec = build_ban_feature_vector("A", "B", "Haven", QUERY_DATE, matches_df, maps_df)
    assert vec[0] == pytest.approx(3.5 / 12.0)


# --------------------------------------------------------------------------
# plan#3k: softmax / scoring path
# --------------------------------------------------------------------------


def test_predict_ban_distribution_equals_softmax_of_scores():
    # With an identity standardizer and hand-picked coefficients, the
    # returned distribution must equal the hand-computed stable softmax
    # of the raw feature . beta scores, aligned to the passed order and
    # summing to 1.
    matches_df, maps_df = _ban_league_tables()
    model = ConditionalLogitBanModel(
        coefficients=np.asarray([1.0, -0.5, 2.0, 0.1, -1.0]),
        standardizer_means=np.zeros(5),
        standardizer_stds=np.ones(5),
        feature_names=FEATURE_NAMES,
        converged=True,
        n_iter=10,
        final_loss=0.5,
        n_train=4,
        l2_lambda=1.0,
    )
    pool = sorted(["Haven", "Split", "Ascent", "Sunset"])
    probs = predict_ban_distribution(
        "A", "B", pool, QUERY_DATE, matches_df, maps_df, model
    )
    raw = [
        build_ban_feature_vector("A", "B", name, QUERY_DATE, matches_df, maps_df)
        for name in pool
    ]
    scores = [float(np.dot(row, model.coefficients)) for row in raw]
    assert probs == pytest.approx(_softmax(scores))
    assert sum(probs) == pytest.approx(1.0)


def test_predict_ban_distribution_rejects_empty_remaining_maps():
    # No remaining maps -> no distribution to form; fail loudly.
    matches_df, maps_df = _ban_league_tables()
    model = _small_fitted_model()
    with pytest.raises(ValueError, match="at least one remaining map"):
        predict_ban_distribution("A", "B", [], QUERY_DATE, matches_df, maps_df, model)


def test_softmax_internally_consistent_across_shrinking_candidate_sets():
    # The no-intercept/IIA property: because scoring is a pure feature
    # dot-product with no per-candidate intercept, the distribution over
    # a strict subset of the pool must be proportional to the
    # corresponding entries of the full-pool distribution — relative
    # preferences between two maps never depend on which other maps are
    # still in play. (With a per-map intercept this would still hold for
    # two *fixed* subsets, but the point is the model *constructs* no
    # per-map identity term at all — decision 1.)
    matches_df, maps_df = _ban_league_tables()
    model = _small_fitted_model()
    full_pool = sorted(["Ascent", "Haven", "Split", "Sunset"])
    subset = sorted(["Haven", "Split", "Sunset"])
    probs_full = predict_ban_distribution(
        "A", "B", full_pool, QUERY_DATE, matches_df, maps_df, model
    )
    probs_subset = predict_ban_distribution(
        "A", "B", subset, QUERY_DATE, matches_df, maps_df, model
    )
    assert sum(probs_subset) == pytest.approx(1.0)
    for name, p_sub in zip(subset, probs_subset):
        p_full = probs_full[full_pool.index(name)]
        # p_sub == p_full / sum(p_full over subset)
        assert p_sub == pytest.approx(p_full / sum(
            probs_full[full_pool.index(other)] for other in subset
        ))


# --------------------------------------------------------------------------
# plan#3d/3k: analytic vs finite-difference gradients
# --------------------------------------------------------------------------


def test_analytic_gradient_matches_finite_differences():
    # The single highest-risk correctness item: the analytic ragged-
    # group gradient must match a central finite-difference numerical
    # gradient (eps=1e-6) of the same batch objective, at multiple
    # points — the initialization point (beta=0) and two random
    # points — over a ragged group structure with variable group sizes.
    from models.conditional_logit_ban import _loss_and_gradient

    rng = np.random.default_rng(7)
    bounds = [0]
    ys = []
    rows = []
    for _ in range(12):
        n_s = int(rng.integers(2, 6))
        xs = rng.normal(scale=0.6, size=(n_s, len(FEATURE_NAMES)))
        rows.append(xs)
        ys.append(int(rng.integers(0, n_s)))
        bounds.append(bounds[-1] + n_s)
    Xs = np.vstack(rows)
    bounds_arr = np.asarray(bounds)
    ys_arr = np.asarray(ys)
    p = len(FEATURE_NAMES)
    points = [
        np.zeros(p),
        rng.normal(scale=0.4, size=p),
        rng.normal(scale=0.8, size=p),
    ]
    for beta in points:
        _loss, grad = _loss_and_gradient(Xs, bounds_arr, ys_arr, beta, 1.0)
        numeric = _numeric_gradient(Xs, bounds_arr, ys_arr, beta, 1.0, eps=1e-6)
        assert grad == pytest.approx(numeric, rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------
# plan#3k: fit convergence on separable synthetic data
# --------------------------------------------------------------------------


def test_fit_learns_negative_coefficient_on_banned_weak_map_feature():
    # The sign regression: a dataset where the banned map is always the
    # one with the lowest synthetic acting_map_win_rate must fit a
    # NEGATIVE coefficient on that feature (lower win rate -> more
    # bannable) — the decision-8 intuition "ban my weak maps" made
    # concrete — and it must be the dominant coefficient.
    rng = np.random.default_rng(42)
    X_flat, bounds, ys = _separable_ban_dataset(rng, n_groups=120, n_maps=5)
    model = fit(X_flat, bounds, ys, l2_lambda=0.1, max_iter=2000)
    assert model.coefficients[0] < 0.0
    assert model.coefficients[0] <= np.min(model.coefficients[1:])


def test_fit_loss_trace_non_increasing():
    # The Armijo line search guarantees every accepted step decreases
    # the objective, so the returned iteration trace must be
    # non-increasing and end at final_loss.
    rng = np.random.default_rng(3)
    X_flat, bounds, ys = _separable_ban_dataset(rng, n_groups=80, n_maps=5)
    model = fit(X_flat, bounds, ys, max_iter=300)
    assert len(model.loss_trace) == model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(model.loss_trace))
    assert model.final_loss == pytest.approx(model.loss_trace[-1])
    assert model.n_train == 80


def test_fit_validation_guards():
    # Wrong feature count, bad group boundaries, out-of-range true
    # indices, and empty design matrices are all hard errors, not silent
    # misalignments.
    X = np.zeros((20, len(FEATURE_NAMES)))
    bounds = np.arange(0, 21, 5)  # 0, 5, 10, 15, 20
    ys = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match="feature columns"):
        fit(np.zeros((10, 3)), bounds, ys)
    with pytest.raises(ValueError, match="must start at 0"):
        fit(X, bounds + 1, ys)
    with pytest.raises(ValueError, match="strictly increasing"):
        fit(X, np.array([0, 5, 5, 20]), ys)
    with pytest.raises(ValueError, match="groups; they must match"):
        fit(X, bounds, np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="outside its group"):
        fit(X, bounds, np.array([9, 0, 0, 0], dtype=int))
    with pytest.raises(ValueError, match="empty design matrix"):
        fit(np.zeros((0, len(FEATURE_NAMES))), np.array([0, 0]), np.zeros(1, dtype=int))


# --------------------------------------------------------------------------
# plan#3k: to_dict / from_dict round-trip and serializability
# --------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_and_json_serializable():
    # from_dict(to_dict(model)) must reproduce every serialized field,
    # and to_dict's output must be directly json.dumps-able (the
    # training driver writes it with json.dumps).
    model = _small_fitted_model()
    d = to_dict(model)
    serialized = json.dumps(d)
    restored = from_dict(json.loads(serialized))
    assert isinstance(restored, ConditionalLogitBanModel)
    assert restored.feature_names == model.feature_names == FEATURE_NAMES
    assert np.allclose(restored.coefficients, model.coefficients)
    assert np.allclose(restored.standardizer_means, model.standardizer_means)
    assert np.allclose(restored.standardizer_stds, model.standardizer_stds)
    assert restored.l2_lambda == model.l2_lambda
    assert restored.converged == model.converged
    assert restored.n_iter == model.n_iter
    assert restored.final_loss == model.final_loss
    assert restored.n_train == model.n_train
    # The loss trace is a live-fit diagnostic, deliberately not
    # persisted: a deserialized model carries an empty trace.
    assert model.loss_trace
    assert restored.loss_trace == ()


def test_coefficient_report_sorted_by_magnitude_with_directions():
    # The report entries must be sorted by |coefficient| descending and
    # carry the documented sign-derived direction strings.
    model = _small_fitted_model()
    report = to_dict(model)["coefficient_report"]
    assert [entry["feature"] for entry in report] == sorted(
        [entry["feature"] for entry in report],
        key=lambda name: abs(
            model.coefficients[list(FEATURE_NAMES).index(name)]
        ),
        reverse=True,
    )
    by_name = {entry["feature"]: entry for entry in report}
    for name, coefficient in zip(FEATURE_NAMES, model.coefficients):
        entry = by_name[name]
        assert entry["coefficient"] == pytest.approx(coefficient)
        if coefficient > 0.0:
            assert entry["direction"] == "favors more bannable"
        elif coefficient < 0.0:
            assert entry["direction"] == "favors less bannable"
        else:
            assert entry["direction"] == "no effect"


def test_from_dict_rejects_shape_mismatch():
    # A corrupt artifact (coefficients not matching feature_names) must
    # fail loudly rather than deserialize into a misaligned model.
    d = to_dict(_small_fitted_model())
    d["coefficients"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="must match"):
        from_dict(d)


# --------------------------------------------------------------------------
# plan#3k: opponent resolution and the wrapped predictor
# --------------------------------------------------------------------------


def test_resolve_opponent_id_returns_other_side():
    # The positive path: team A's queried match (m9, at QUERY_DATE)
    # resolves its opponent to B.
    from models.conditional_logit_ban import _resolve_opponent_id

    matches_df, _ = _ban_league_tables()
    assert _resolve_opponent_id("A", QUERY_DATE, matches_df) == "B"


def test_resolve_opponent_id_zero_matches_raises():
    # A team with no match on that date (or a None acting id) cannot
    # have its opponent resolved; fail loudly.
    from models.conditional_logit_ban import _resolve_opponent_id

    matches_df, _ = _ban_league_tables()
    with pytest.raises(ValueError, match="exactly one match"):
        _resolve_opponent_id("UNSEEN", QUERY_DATE, matches_df)


def test_resolve_opponent_id_ambiguous_matches_raise():
    # The same team playing twice at the same timestamp is ambiguous;
    # fail loudly rather than silently picking one opponent.
    from models.conditional_logit_ban import _resolve_opponent_id

    matches_df, _ = _ban_league_tables()
    extra = _matches_df(
        [
            {
                "match_id": "m10",
                "date": QUERY_DATE,
                "team1_id": "A",
                "team2_id": "Z",
                "status": "completed",
            }
        ]
    )
    dup = pd.concat([matches_df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one match"):
        _resolve_opponent_id("A", QUERY_DATE, dup)


def test_predictor_fn_rejects_non_ban_action():
    # The wrapped predictor only supports action == "ban" (M28 adds the
    # pick arm); anything else raises with the action named.
    matches_df, maps_df = _ban_league_tables()
    model = _small_fitted_model()
    fn = make_veto_step_predictor_fn(model)
    with pytest.raises(ValueError, match="only supports action 'ban'"):
        fn("A", "pick", ["Haven", "Split"], QUERY_DATE, matches_df, maps_df)


def test_predictor_fn_closure_matches_direct_predict():
    # The returned closure must (a) produce a length-K simplex for a
    # ban over K remaining maps and (b) agree with the pure path
    # predict_ban_distribution called with the opponent the closure
    # resolves itself — no drift between the interface bridge and the
    # pure scoring function.
    matches_df, maps_df = _ban_league_tables()
    model = _small_fitted_model()
    fn = make_veto_step_predictor_fn(model)
    pool = sorted(["Haven", "Split", "Ascent"])
    probs = fn("A", "ban", pool, QUERY_DATE, matches_df, maps_df)
    assert len(probs) == len(pool)
    assert sum(probs) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in probs)
    direct = predict_ban_distribution(
        "A", "B", pool, QUERY_DATE, matches_df, maps_df, model
    )
    assert probs == pytest.approx(direct)

"""Tests for the ordinal logistic regression model (M20).

Covers the 13-feature vector builder (a real-league integration check
plus one test per section-A missing-value fallback rule, asserting the
exact documented fallback value), the +eta sign-convention regression
(a synthetic single-informative-feature dataset must fit a positive
coefficient), the analytic-vs-finite-difference gradient check (the
highest-risk correctness item, for both ``beta`` and the raw threshold
parameters at multiple points), the monotonic-loss-decrease property on
synthetic and real v1 data, post-fit threshold ordering, the
``to_dict``/``from_dict`` round-trip and JSON serializability, the
``make_model_fn`` closure-vs-direct-predict agreement, the standardizer
zero-variance guard, and a skip-guarded real-``data/v1`` end-to-end run
that trains via the CLI driver, reloads the artifact, and scores the
real test split through the evaluation harness directly (recording the
resulting RPS/log-loss/accuracy numbers next to the M18 floor).
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate
from evaluation import harness
from features import closeness, h2h_context, player_form
from models.ordinal_logit import (
    FEATURE_NAMES,
    OUTCOME_LABELS,
    OrdinalLogitModel,
    apply_standardizer,
    build_feature_vector,
    fit,
    fit_standardizer,
    from_dict,
    make_model_fn,
    predict_proba,
    to_dict,
)
from tests._shared import _real_v1_available

# The as-of cutoff every synthetic feature test uses: the date of the
# league's A-vs-B match itself (m3), so the as-of features exclude the
# queried match (the real leakage boundary) and the event-stage lookup
# resolves the (A, B, date) triple to that same match.
QUERY_DATE = "2026-01-02T12:00:00"

_MATCHES_COLS = [
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "team1_name",
    "team2_name",
    "event_name",
    "status",
]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
    "team1_first_half_rounds",
    "team2_first_half_rounds",
    "team1_second_half_rounds",
    "team2_second_half_rounds",
    "team1_atk_rounds",
    "team1_def_rounds",
    "team2_atk_rounds",
    "team2_def_rounds",
]
_PMS_COLS = [
    "match_id",
    "map_index",
    "player_name",
    "team_name",
    "acs",
    "rating",
    "first_kills",
    "first_deaths",
]


def _league_tables():
    """Build the 3-match, 3-map synthetic league for feature tests.

    Three completed one-map matches: ``m1`` Alpha beats Xray on Haven
    13-11 (a close, non-OT scoreline), ``m2`` Bravo loses to Yankee on
    Haven 8-13, and ``m3`` Alpha beats Bravo on Bind 13-8 — the queried
    match itself, dated after the other two so the as-of features at its
    own date see exactly the two prior matches. Each team carries a
    full 5-player roster on each of its maps with fixed acs/rating
    values (Alpha 200/1.1 then 210/1.2, Bravo 250/1.3 then 240/1.25),
    giving the feature tests deterministic, hand-computable values.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple with the
        fixed column conventions above.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    match_rows = [
        {
            "match_id": "m1",
            "date": "2026-01-01T10:00:00",
            "team1_id": "A",
            "team2_id": "X",
            "team1_name": "Alpha",
            "team2_name": "Xray",
            "event_name": "VCT 2026: EMEA Stage 1",
            "status": "completed",
        },
        {
            "match_id": "m2",
            "date": "2026-01-02T10:00:00",
            "team1_id": "B",
            "team2_id": "Y",
            "team1_name": "Bravo",
            "team2_name": "Yankee",
            "event_name": "VCT 2026: EMEA Stage 1",
            "status": "completed",
        },
        {
            "match_id": "m3",
            "date": QUERY_DATE,
            "team1_id": "A",
            "team2_id": "B",
            "team1_name": "Alpha",
            "team2_name": "Bravo",
            "event_name": "VCT 2026: EMEA Stage 2",
            "status": "completed",
        },
    ]
    map_rows = [
        {
            "match_id": "m1",
            "map_index": 0,
            "map_name": "Haven",
            "team1_score": 13,
            "team2_score": 11,
            "winner": "Alpha",
            # Regulation 13-11: per-side atk+def == score (A 13 =
            # 7 atk + 6 def; X 11 = 6 atk + 5 def), pairings
            # 7+5=12 / 6+6=12 partition the 24 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 7,
            "team1_def_rounds": 6,
            "team2_atk_rounds": 6,
            "team2_def_rounds": 5,
        },
        {
            "match_id": "m2",
            "map_index": 0,
            "map_name": "Haven",
            "team1_score": 8,
            "team2_score": 13,
            "winner": "Yankee",
            # Regulation 8-13: team1 8 = 4 atk + 4 def; team2 13 =
            # 6 atk + 7 def; pairings 4+7=11 / 4+6=10 partition the
            # 21 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 4,
            "team1_def_rounds": 4,
            "team2_atk_rounds": 6,
            "team2_def_rounds": 7,
        },
        {
            "match_id": "m3",
            "map_index": 0,
            "map_name": "Bind",
            "team1_score": 13,
            "team2_score": 8,
            "winner": "Alpha",
            # Regulation 13-8: team1 13 = 7 atk + 6 def; team2 8 =
            # 4 atk + 4 def; pairings 7+4=11 / 6+4=10 partition the
            # 21 rounds.
            "team1_first_half_rounds": 12.0,
            "team2_first_half_rounds": 12.0,
            "team1_second_half_rounds": 12.0,
            "team2_second_half_rounds": 12.0,
            "team1_atk_rounds": 7,
            "team1_def_rounds": 6,
            "team2_atk_rounds": 4,
            "team2_def_rounds": 4,
        },
    ]
    pms_rows = []
    for mid, team, players, acs, rating, fk, fd in [
        ("m1", "Alpha", ["pA1", "pA2", "pA3", "pA4", "pA5"], 200.0, 1.1, 3, 2),
        ("m1", "Xray", ["pX1", "pX2", "pX3", "pX4", "pX5"], 180.0, 0.9, 2, 3),
        ("m2", "Bravo", ["pB1", "pB2", "pB3", "pB4", "pB5"], 250.0, 1.3, 2, 3),
        ("m2", "Yankee", ["pY1", "pY2", "pY3", "pY4", "pY5"], 170.0, 0.8, 3, 2),
        ("m3", "Alpha", ["pA1", "pA2", "pA3", "pA4", "pA5"], 210.0, 1.2, 3, 2),
        ("m3", "Bravo", ["pB1", "pB2", "pB3", "pB4", "pB5"], 240.0, 1.25, 2, 3),
    ]:
        for player in players:
            pms_rows.append(
                {
                    "match_id": mid,
                    "map_index": 0,
                    "player_name": player,
                    "team_name": team,
                    "acs": acs,
                    "rating": rating,
                    # Per-map conservation (sum FK == sum FD) holds
                    # per match: m1 5==5, m2 5==5, m3 5==5.
                    "first_kills": fk,
                    "first_deaths": fd,
                }
            )
    matches_df = pd.DataFrame(match_rows, columns=_MATCHES_COLS)
    maps_df = pd.DataFrame(map_rows, columns=_MAPS_COLS)
    pms_df = pd.DataFrame(pms_rows, columns=_PMS_COLS)
    return matches_df, maps_df, pms_df


def _zero_history_league_tables():
    """Build the 4-match league where the queried pair has zero history.

    Extends :func:`_league_tables`'s three matches (m1/m2/m3) with a
    fourth match ``m4`` — ``C`` (Charlie) beats ``D`` (Delta) on Lotus
    13-9, dated strictly after the others — where neither ``C`` nor
    ``D`` appears in any earlier match, so an as-of query at m4's own
    date sees an empty history for both sides. All three tables carry
    the same column conventions as :func:`_league_tables` (round detail
    present and internally consistent; first-kill/first-death
    conservation per map), so the three surviving M38.5 estimators
    run and each returns exactly its prior.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple with the
        fixed column conventions of :func:`_league_tables` plus the m4
        rows.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    matches_df, maps_df, pms_df = _league_tables()
    m4_date = "2026-01-01T16:00:00"
    m4_match = {
        "match_id": "m4",
        "date": m4_date,
        "team1_id": "C",
        "team2_id": "D",
        "team1_name": "Charlie",
        "team2_name": "Delta",
        "event_name": "VCT 2026: EMEA Stage 2",
        "status": "completed",
    }
    m4_map = {
        "match_id": "m4",
        "map_index": 0,
        "map_name": "Lotus",
        "team1_score": 13,
        "team2_score": 9,
        "winner": "Charlie",
        # Regulation 13-9: team1 13 = 7 atk + 6 def; team2 9 = 5 atk
        # + 4 def; pairings 7+4=11 / 6+5=11 partition the 22 rounds.
        "team1_first_half_rounds": 12.0,
        "team2_first_half_rounds": 12.0,
        "team1_second_half_rounds": 12.0,
        "team2_second_half_rounds": 12.0,
        "team1_atk_rounds": 7,
        "team1_def_rounds": 6,
        "team2_atk_rounds": 5,
        "team2_def_rounds": 4,
    }
    m4_pms = []
    for team, players, acs, rating, fk, fd in [
        ("Charlie", ["pC1", "pC2", "pC3", "pC4", "pC5"], 200.0, 1.1, 3, 2),
        ("Delta", ["pD1", "pD2", "pD3", "pD4", "pD5"], 180.0, 0.9, 2, 3),
    ]:
        for player in players:
            m4_pms.append(
                {
                    "match_id": "m4",
                    "map_index": 0,
                    "player_name": player,
                    "team_name": team,
                    "acs": acs,
                    "rating": rating,
                    # Conservation per map: 3+2 == 2+3 per side sums.
                    "first_kills": fk,
                    "first_deaths": fd,
                }
            )
    matches_df = pd.concat(
        [matches_df, pd.DataFrame([m4_match], columns=_MATCHES_COLS)],
        ignore_index=True,
    )
    maps_df = pd.concat(
        [maps_df, pd.DataFrame([m4_map], columns=_MAPS_COLS)],
        ignore_index=True,
    )
    pms_df = pd.concat(
        [pms_df, pd.DataFrame(m4_pms, columns=_PMS_COLS)],
        ignore_index=True,
    )
    return matches_df, maps_df, pms_df


def _sample_ordinal(X, beta, thresholds, rng):
    """Sample outcome ordinals from the model's own probabilities.

    Generates synthetic labels for the sign-convention test: for each
    row, computes ``eta = X @ beta``, the four category probabilities
    via the module's own :func:`_category_probabilities`, and draws one
    ordinal via ``numpy``'s categorical sampler. The fit is therefore
    tested against data generated *by the same model family*, so
    recovering the true coefficient sign is a well-posed check.

    Args:
        X: The design matrix, ``(n, p)`` floats.
        beta: The true coefficient vector, length ``p``.
        thresholds: The true 3-threshold vector (strictly increasing).
        rng: A seeded ``numpy.random.Generator``.

    Returns:
        An ``(n,)`` int array of outcome ordinals in ``{0, 1, 2, 3}``.

    Raises:
        ValueError: If the probabilities are not a valid distribution
            (propagated from numpy's sampler).
    """
    from models import ordinal_logit

    ys = []
    for eta in X @ beta:
        probs = ordinal_logit._category_probabilities(float(eta), thresholds)
        ys.append(int(rng.choice(4, p=probs)))
    return np.asarray(ys, dtype=int)


def _numeric_gradient(X, y, beta, raw, l2_lambda, eps):
    """Central finite-difference gradient of the batch objective.

    The independent numerical check the analytic gradients are tested
    against: for each parameter (``beta`` then ``raw``), perturbs it by
    ``±eps``, evaluates the objective via
    :func:`models.ordinal_logit._loss_and_gradient`'s loss component,
    and takes the centered difference. The objective is the same
    clipped-NLL-plus-L2 function the analytic gradient differentiates,
    so at points where the clip is inactive (all test points) the two
    must agree to the finite-difference accuracy.

    Args:
        X: The (already-standardized) design matrix.
        y: The true outcome ordinals.
        beta: The coefficient vector at which to differentiate.
        raw: The raw threshold parameters at which to differentiate.
        l2_lambda: The L2 strength.
        eps: The finite-difference step size.

    Returns:
        A ``(p + 3,)`` numpy array of numerical gradient components in
        ``[beta, raw]`` concatenation order.

    Raises:
        Nothing (the objective is total for finite inputs).
    """
    p = len(beta)
    params = np.concatenate([np.asarray(beta, dtype=float), np.asarray(raw, dtype=float)])
    grad = np.empty(p + 3)
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += eps
        minus = params.copy()
        minus[i] -= eps
        loss_plus = ordinal_logit_loss(X, y, plus[:p], plus[p:], l2_lambda)
        loss_minus = ordinal_logit_loss(X, y, minus[:p], minus[p:], l2_lambda)
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def ordinal_logit_loss(X, y, beta, raw, l2_lambda):
    """Evaluate the batch objective at one parameter point.

    A thin test-side wrapper over the module's own
    :func:`models.ordinal_logit._loss_and_gradient` returning only the
    scalar loss, so the finite-difference helper above can differentiate
    exactly the objective the analytic gradient describes.

    Args:
        X: The (already-standardized) design matrix.
        y: The true outcome ordinals.
        beta: The coefficient vector.
        raw: The raw threshold parameters.
        l2_lambda: The L2 strength.

    Returns:
        The scalar objective ``mean(NLL) + (l2_lambda/2) * sum(beta^2)``.

    Raises:
        ValueError: If shapes are inconsistent (propagated from
            :func:`models.ordinal_logit._loss_and_gradient`).
    """
    from models import ordinal_logit

    return ordinal_logit._loss_and_gradient(X, y, beta, raw, l2_lambda)[0]


@pytest.fixture(scope="module")
def real_v1_train_model(real_v1_train_design_matrix):
    """Fit the model on the real v1 train split once per module.

    Fits the ordinal logit with the documented defaults on the shared
    session-scoped design matrix. The expensive feature assembly (the
    five table reads plus the per-row :func:`build_feature_vector`
    loop) runs once per pytest session inside the
    ``real_v1_train_design_matrix`` fixture; this fixture only performs
    the cheap fit, cached per module for the real-data monotonic-loss
    and threshold-ordering tests.

    Args:
        real_v1_train_design_matrix: The session-scoped fixture
            providing ``(X, y_ordinal, train_rows, matches_df, maps_df,
            player_map_stats_df)``; only ``X`` and ``y_ordinal`` are
            consumed here, both read-only (never mutate them in place).

    Returns:
        The fitted :class:`OrdinalLogitModel`.

    Raises:
        pytest.skip: If the real v1 tables are absent (propagated from
            the session fixture's own skip guard; the fixture body
            itself raises nothing).
    """
    X, y_ordinal, _train_rows, _matches_df, _maps_df, _pms_df = (
        real_v1_train_design_matrix
    )
    return fit(X, y_ordinal)


# --------------------------------------------------------------------------
# plan#3: build_feature_vector real-league integration + fallback rules
# --------------------------------------------------------------------------


def test_build_feature_vector_real_league_exact_values():
    # The full 13-vector at m3's own date (as-of cutoff = the queried
    # match, so its own outcome is never in its feature history), with
    # every entry hand-computed from the fixture league:
    #   map_win_rate_diff 1.0 (A prior 1.0 full-shrink vs B prior 0.0),
    #   elo_differential 32.0 (A won m1 -> 1516, B lost m2 -> 1484),
    #   ot_rate_diff 0.0 (no OT maps in the as-of pool),
    #   map_round_margin_variance 0.0 (n=0 Bind maps -> NaN -> 0.0),
    #   acs_form_diff -50.0 (A 200 vs B 250, one map each),
    #   rating_form_diff -0.2 (A 1.1 vs B 1.3),
    #   h2h_win_rate_centered 0.0 (no prior A-vs-B games -> 0.5 prior),
    #   event_stage 2.0 (m3 is Stage 2),
    #   days_since_diff 1.0 (A last played 1 day before, B 0 days),
    #   roster_decay_diff 0.0 (both sides have 1 map -> 1.0 each).
    # The three surviving M38.5 additions (slots 10-12) are
    # Bind-at-m3 estimates: neither side has any prior Bind map, so
    # each estimator returns that side's shrunk overall prior (never
    # None — the ambiguity-3 no-fallback contract), and the diff is the
    # A-minus-B of two genuinely different team priors (A won its one
    # prior Haven map 13-11, B lost its one prior Haven map 8-13):
    #   attack_side_win_rate_diff 0.040572... (A overall-attack prior
    #     0.525090... minus B's 0.484517...),
    #   signed_margin_diff 7/11 (A's shrunk overall mean margin
    #     (+2 + 10*0)/(1+10) = 2/11 minus B's (-5 + 10*0)/(1+10) =
    #     -5/11),
    #   first_blood_diff 0.066667 (A 0.533333 minus B 0.466667).
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec.shape == (len(FEATURE_NAMES),)
    assert list(FEATURE_NAMES) == [
        "map_win_rate_diff",
        "elo_differential",
        "ot_rate_diff",
        "map_round_margin_variance",
        "acs_form_diff",
        "rating_form_diff",
        "h2h_win_rate_centered",
        "event_stage",
        "days_since_diff",
        "roster_decay_diff",
        "attack_side_win_rate_diff",
        "signed_margin_diff",
        "first_blood_diff",
    ]
    assert vec[0] == pytest.approx(1.0)
    assert vec[1] == pytest.approx(32.0, abs=1e-9)
    assert vec[2] == pytest.approx(0.0)
    assert vec[3] == pytest.approx(0.0)
    assert vec[4] == pytest.approx(-50.0)
    assert vec[5] == pytest.approx(-0.2)
    assert vec[6] == pytest.approx(0.0)
    assert vec[7] == pytest.approx(2.0)
    assert vec[8] == pytest.approx(1.0)
    assert vec[9] == pytest.approx(0.0)
    # The three surviving M38.5 additions (slots 10-12) at Bind on m3's
    # date: no side has prior Bind history, so each estimator returns
    # that side's shrunk *overall* prior as its map-level mean (never
    # None — the ambiguity-3 no-fallback contract), and the diff is the
    # A-minus-B of two genuinely different team priors (A won m1 13-11,
    # B lost m2 8-13):
    #   attack_side_win_rate_diff 0.040572... = A's shrunk overall
    #     attack prior (0.525090...) minus B's (0.484517...);
    #   signed_margin_diff 0.636363... = A's shrunk overall mean margin
    #     ((+2 + 10*0)/(1+10) = 2/11) minus B's ((-5 + 10*0)/(1+10) =
    #     -5/11): +7/11;
    #   first_blood_diff 0.066667 = A's overall first-blood mean
    #     (0.533333...) minus B's (0.466667...).
    assert vec[10] == pytest.approx(0.04057230154533176, abs=1e-9)
    assert vec[11] == pytest.approx(0.6363636363636362, abs=1e-9)
    assert vec[12] == pytest.approx(0.06666666666666665, abs=1e-9)


def test_build_feature_vector_zero_history_new_slots_are_zero():
    # M38.5 ambiguity 3, asserted in code: for a pair of teams with no
    # as-of history at all (C and D appear only in the queried match m4,
    # dated after every prior match), each of the three surviving M38.5
    # estimators returns exactly its (non-None) prior and the A-minus-B
    # difference is exactly 0.0 — no missing-value fallback branch is
    # needed. Also pins the ambiguity-1 name order for the three
    # surviving appended features and the length-13 output contract.
    matches_df, maps_df, pms_df = _zero_history_league_tables()
    query_date = "2026-01-01T16:00:00"  # m4's own date (strictly after m1-m3)
    vec = build_feature_vector(
        "C", "D", "Lotus", query_date, matches_df, maps_df, pms_df
    )
    assert vec.shape == (len(FEATURE_NAMES),)
    assert list(FEATURE_NAMES)[-3:] == [
        "attack_side_win_rate_diff",
        "signed_margin_diff",
        "first_blood_diff",
    ]
    assert vec[10] == 0.0
    assert vec[11] == 0.0
    assert vec[12] == 0.0


def test_form_diff_falls_back_to_zero_when_one_side_missing(monkeypatch):
    # Section A fallback: if either side's FormStat.mean is None (zero
    # qualifying maps), the acs/rating diff is exactly 0.0 — not a
    # partial diff against the other side's real value.
    def fake_form(team_id, date, matches_df, maps_df, pms_df, n, decay_rate):
        if team_id == "A":
            return player_form.PlayerFormResult(
                team_id,
                date,
                player_form.FormStat(None, 0, (), (), 0),
                player_form.FormStat(None, 0, (), (), 0),
                0,
                0,
            )
        return player_form.PlayerFormResult(
            team_id,
            date,
            player_form.FormStat(250.0, 1, (250.0,), (1.0,), 0),
            player_form.FormStat(1.3, 1, (1.3,), (1.0,), 0),
            1,
            0,
        )

    monkeypatch.setattr(player_form, "team_player_form", fake_form)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("acs_form_diff")] == 0.0
    assert vec[FEATURE_NAMES.index("rating_form_diff")] == 0.0


def test_form_diff_falls_back_to_zero_when_both_sides_missing(monkeypatch):
    # Both sides have no form signal: the diff is exactly 0.0, not NaN
    # and not a subtraction of None.
    def fake_form(team_id, date, matches_df, maps_df, pms_df, n, decay_rate):
        return player_form.PlayerFormResult(
            team_id,
            date,
            player_form.FormStat(None, 0, (), (), 0),
            player_form.FormStat(None, 0, (), (), 0),
            0,
            0,
        )

    monkeypatch.setattr(player_form, "team_player_form", fake_form)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("acs_form_diff")] == 0.0
    assert vec[FEATURE_NAMES.index("rating_form_diff")] == 0.0


def test_form_diff_is_real_difference_when_both_sides_present(monkeypatch):
    # Both sides have signal: the diff is the plain A-minus-B difference
    # of the two weighted means (A stubbed at 200, B at 250 -> -50).
    def fake_form(team_id, date, matches_df, maps_df, pms_df, n, decay_rate):
        mean = 200.0 if team_id == "A" else 250.0
        return player_form.PlayerFormResult(
            team_id,
            date,
            player_form.FormStat(mean, 1, (mean,), (1.0,), 0),
            player_form.FormStat(1.0, 1, (1.0,), (1.0,), 0),
            1,
            0,
        )

    monkeypatch.setattr(player_form, "team_player_form", fake_form)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("acs_form_diff")] == pytest.approx(-50.0)
    assert vec[FEATURE_NAMES.index("rating_form_diff")] == pytest.approx(0.0)


def test_days_since_diff_treats_none_side_as_zero(monkeypatch):
    # Section A fallback: a side with no strictly-prior match (None) is
    # treated as 0 before subtracting, so A-None vs B-5 gives 0 - 5.
    def fake_days(team_id, date, matches_df):
        return None if team_id == "A" else 5

    monkeypatch.setattr(h2h_context, "days_since_last_match", fake_days)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("days_since_diff")] == pytest.approx(-5.0)


def test_days_since_diff_both_none_is_zero(monkeypatch):
    # Both sides unseen: 0 - 0 == 0.0, not NaN.
    monkeypatch.setattr(
        h2h_context, "days_since_last_match", lambda team_id, date, matches_df: None
    )
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("days_since_diff")] == 0.0


def test_days_since_diff_is_real_difference_when_both_present(monkeypatch):
    # Both sides have rest gaps: the diff is the plain A-minus-B gap.
    def fake_days(team_id, date, matches_df):
        return {"A": 2, "B": 1}[team_id]

    monkeypatch.setattr(h2h_context, "days_since_last_match", fake_days)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("days_since_diff")] == pytest.approx(1.0)


def test_roster_decay_diff_treats_none_multiplier_as_one(monkeypatch):
    # Section A fallback: a side whose decay_multiplier is None (changed
    # is None or False) is treated as 1.0 (no penalty). Stub A as
    # changed=None (1 evaluable map) and B as changed=False: both map to
    # 1.0, so the diff is exactly 0.0.
    def fake_roster(team_id, date, matches_df, maps_df, pms_df, jaccard_threshold, half_life_days):
        return h2h_context.RosterChangeResult(
            team_id=team_id,
            date=date,
            changed=None if team_id == "A" else False,
            similarity=None if team_id == "A" else 1.0,
            decay_multiplier=None,
            changed_as_of_date=None,
        )

    monkeypatch.setattr(h2h_context, "team_roster_change", fake_roster)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("roster_decay_diff")] == 0.0


def test_roster_decay_diff_with_real_change_side(monkeypatch):
    # A side with a declared change carries its real decay multiplier
    # (stubbed at 0.8), the other side maps to 1.0, so the diff is
    # 0.8 - 1.0 == -0.2 — the None-fallback must not swallow the
    # changed side's value.
    def fake_roster(team_id, date, matches_df, maps_df, pms_df, jaccard_threshold, half_life_days):
        if team_id == "A":
            return h2h_context.RosterChangeResult(
                team_id, date, True, 0.4, 0.8, "2026-01-01T10:00:00"
            )
        return h2h_context.RosterChangeResult(
            team_id, date, False, 1.0, None, None
        )

    monkeypatch.setattr(h2h_context, "team_roster_change", fake_roster)
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("roster_decay_diff")] == pytest.approx(-0.2)


class _StubVariance:
    """Minimal stand-in for features.closeness.MapMarginVariance.

    Only the ``variance`` attribute is read by the fallback logic under
    test.

    Args:
        variance: The stub's variance value.

    Returns:
        Nothing (attribute holder).

    Raises:
        Nothing.
    """

    def __init__(self, variance):
        self.variance = variance


def test_margin_variance_nan_falls_back_to_zero(monkeypatch):
    # Section A fallback: a NaN variance (n <= 1 for that map) is
    # replaced with exactly 0.0 ("no observed-variance signal
    # contributes no information").
    monkeypatch.setattr(
        closeness,
        "map_round_margin_variance",
        lambda map_name, date, matches_df, maps_df: _StubVariance(float("nan")),
    )
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("map_round_margin_variance")] == 0.0


def test_margin_variance_real_value_passes_through(monkeypatch):
    # A finite variance is passed through unchanged (3.5 stays 3.5).
    monkeypatch.setattr(
        closeness,
        "map_round_margin_variance",
        lambda map_name, date, matches_df, maps_df: _StubVariance(3.5),
    )
    matches_df, maps_df, pms_df = _league_tables()
    vec = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    assert vec[FEATURE_NAMES.index("map_round_margin_variance")] == pytest.approx(3.5)


# --------------------------------------------------------------------------
# plan#4: standardizer
# --------------------------------------------------------------------------


def test_fit_standardizer_zero_variance_guard():
    # A constant training column must standardize to 0.0 for every row
    # (its std is replaced by 1.0) instead of dividing by zero or
    # raising; a varying column standardizes normally.
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    means, stds = fit_standardizer(X)
    assert means[0] == 1.0
    assert stds[0] == 1.0
    Xs = apply_standardizer(X, means, stds)
    assert Xs[:, 0] == pytest.approx(0.0)
    assert means[1] == pytest.approx(4.0)
    assert stds[1] == pytest.approx(math.sqrt(8.0 / 3.0))
    assert Xs[:, 1] == pytest.approx([-math.sqrt(3.0 / 2.0), 0.0, math.sqrt(3.0 / 2.0)])


def test_apply_standardizer_requires_matching_lengths():
    # A standardizer whose length does not match the matrix columns is a
    # hard error (a silent column misalignment would corrupt every
    # prediction).
    with pytest.raises(ValueError, match="must match"):
        apply_standardizer(np.ones((3, 5)), np.zeros(4), np.ones(4))


# --------------------------------------------------------------------------
# plan#12: sign convention regression (section C)
# --------------------------------------------------------------------------


def test_sign_convention_positive_coefficient_favors_team_a():
    # The +eta sign convention regression: a synthetic dataset where
    # feature 0 is strongly positive exactly on low-ordinal (A-win) rows
    # must fit a POSITIVE coefficient for it, not negative — i.e.
    # increasing the feature shifts probability mass toward A's
    # categories. Data is sampled from the model family itself with
    # beta_true[0] = 2.0, so recovering the sign is a well-posed check.
    rng = np.random.default_rng(42)
    n = 250
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    beta_true = np.zeros(len(FEATURE_NAMES))
    beta_true[0] = 2.0
    thresholds_true = np.asarray([-1.0, 0.5, 1.5])
    y = _sample_ordinal(X, beta_true, thresholds_true, rng)
    model = fit(X, y, l2_lambda=0.1)
    assert model.coefficients[0] > 0.0
    # And it must be the dominant coefficient: no noise feature should
    # pick up a stronger signal.
    assert model.coefficients[0] >= np.max(np.abs(model.coefficients[1:]))


# --------------------------------------------------------------------------
# plan#12: analytic vs finite-difference gradients (section D)
# --------------------------------------------------------------------------


def test_analytic_gradients_match_finite_differences():
    # The single highest-risk correctness item: the analytic gradients
    # (w.r.t. both beta and the raw threshold parameters) must match a
    # central finite-difference numerical gradient (eps=1e-6) of the
    # same batch objective, at multiple points — the initialization
    # point (beta=0, marginal thresholds) and a random point — not just
    # at one convenient location.
    from models import ordinal_logit

    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=30)
    p = len(FEATURE_NAMES)
    points = [
        np.concatenate([np.zeros(p), np.asarray([-0.3, 0.2, 0.4])]),
        rng.normal(scale=0.4, size=p + 3),
        rng.normal(scale=0.7, size=p + 3),
    ]
    for params in points:
        beta = params[:p]
        raw = params[p:]
        _loss, g_beta, g_raw = ordinal_logit._loss_and_gradient(
            X, y, beta, raw, 1.0
        )
        analytic = np.concatenate([g_beta, g_raw])
        numeric = _numeric_gradient(X, y, beta, raw, 1.0, eps=1e-6)
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------
# plan#12: monotonic loss decrease + threshold ordering (section E)
# --------------------------------------------------------------------------


def test_loss_trace_non_increasing_on_synthetic_batch():
    # The Armijo line search guarantees every accepted step decreases
    # the objective, so the returned iteration trace must be
    # non-increasing and end at final_loss.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=80)
    model = fit(X, y, max_iter=300)
    assert len(model.loss_trace) == model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(model.loss_trace))
    assert model.final_loss == pytest.approx(model.loss_trace[-1])


def test_thresholds_strictly_increasing_after_fit_synthetic():
    # The softplus reparameterization must keep theta_1 < theta_2 <
    # theta_3 after any fit, no matter where gradient descent lands.
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=60)
    model = fit(X, y, max_iter=200)
    assert np.all(np.diff(model.thresholds) > 0.0)


# --------------------------------------------------------------------------
# plan#12: fit validation
# --------------------------------------------------------------------------


def test_fit_rejects_wrong_feature_count():
    # The model is defined over exactly len(FEATURE_NAMES) features; a
    # mismatched matrix is a hard error, not a silent misalignment.
    X = np.ones((10, 7))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="feature columns"):
        fit(X, y)


def test_fit_rejects_invalid_labels():
    # A label outside 0..3 cannot be scored and must fail loudly.
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.full(10, 7, dtype=int)
    with pytest.raises(ValueError, match="outcome ordinals"):
        fit(X, y)


def test_fit_rejects_row_count_mismatch():
    # X rows and y entries must line up.
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.zeros(5, dtype=int)
    with pytest.raises(ValueError, match="must match"):
        fit(X, y)


# --------------------------------------------------------------------------
# plan#7/12: to_dict / from_dict round-trip and serializability
# --------------------------------------------------------------------------


def _small_fitted_model():
    """Fit a small deterministic synthetic model for serialization tests.

    Uses a fixed seed so the fitted parameters are stable across runs;
    the model itself is only exercised for round-tripping, not for its
    predictive quality.

    Returns:
        An :class:`OrdinalLogitModel` fit on 40 synthetic rows.

    Raises:
        Nothing.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=40)
    return fit(X, y, max_iter=150)


def test_to_dict_from_dict_round_trip_and_json_serializable():
    # from_dict(to_dict(model)) must reproduce every serialized field,
    # and to_dict's output must be directly json.dumps-able (the
    # training driver writes it with json.dumps).
    model = _small_fitted_model()
    d = to_dict(model)
    serialized = json.dumps(d)
    restored = from_dict(json.loads(serialized))
    assert isinstance(restored, OrdinalLogitModel)
    assert restored.feature_names == model.feature_names == FEATURE_NAMES
    assert np.allclose(restored.coefficients, model.coefficients)
    assert np.allclose(restored.thresholds, model.thresholds)
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
            assert entry["direction"] == "favors A"
        elif coefficient < 0.0:
            assert entry["direction"] == "favors B"
        else:
            assert entry["direction"] == "favors neither"


def test_from_dict_rejects_shape_mismatch():
    # A corrupt artifact (coefficients not matching feature_names) must
    # fail loudly rather than deserialize into a misaligned model.
    d = to_dict(_small_fitted_model())
    d["coefficients"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="must match"):
        from_dict(d)


# --------------------------------------------------------------------------
# plan#8/12: make_model_fn closure vs direct predict
# --------------------------------------------------------------------------


def test_make_model_fn_closure_matches_direct_predict():
    # The returned closure must (a) produce a length-4 simplex and (b)
    # agree exactly with the underlying pure call
    # predict_proba(build_feature_vector(...), model) for the same
    # inputs — no drift between the interface bridge and the pure path.
    model = _small_fitted_model()
    matches_df, maps_df, pms_df = _league_tables()
    model_fn = make_model_fn(model, pms_df)
    probs = model_fn("A", "B", "Bind", QUERY_DATE, matches_df, maps_df)
    assert len(probs) == 4
    assert sum(probs) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in probs)
    x = build_feature_vector(
        "A", "B", "Bind", QUERY_DATE, matches_df, maps_df, pms_df
    )
    direct = predict_proba(x, model)
    assert probs == pytest.approx(direct)


def test_predict_proba_accepts_both_vector_shapes():
    # predict_proba must work for both a 1-D vector and a 1-row matrix
    # (the closure path always hands it a 1-D vector; the pure path
    # accepts either).
    model = _small_fitted_model()
    x = np.linspace(-1.0, 1.0, len(FEATURE_NAMES))
    assert predict_proba(x, model) == pytest.approx(
        predict_proba(x.reshape(1, -1), model)
    )


# --------------------------------------------------------------------------
# plan#10/12: evaluate MODEL_REGISTRY factory shape
# --------------------------------------------------------------------------


def test_evaluate_registry_is_factory_dict_with_ordinal_logit():
    # plan#10: MODEL_REGISTRY values are (output_dir, version) ->
    # ModelFn factories; the four_way_baseline factory is the trivial
    # stateless one returning the harness adapter unchanged, and the
    # ordinal_logit key is present (so --model choices pick it up
    # automatically). The multinomial_logit key (added by task 024) is
    # asserted in tests/test_multinomial_logit.py.
    assert "four_way_baseline" in evaluate.MODEL_REGISTRY
    assert "ordinal_logit" in evaluate.MODEL_REGISTRY
    assert "multinomial_logit" in evaluate.MODEL_REGISTRY
    model_fn = evaluate.MODEL_REGISTRY["four_way_baseline"](Path("data"), "v1")
    assert model_fn is evaluate.harness.four_way_baseline_model


def test_ordinal_logit_factory_raises_on_missing_artifact(tmp_path):
    # The stateful ordinal_logit factory reads
    # <output_dir>/<version>/ordinal_logit_model.json; a missing
    # artifact must surface as a clear FileNotFoundError (the training
    # driver has not been run).
    with pytest.raises(FileNotFoundError):
        evaluate.MODEL_REGISTRY["ordinal_logit"](tmp_path, "v1")


# --------------------------------------------------------------------------
# plan#12: real v1 monotonic loss + threshold ordering (module fixture)
# --------------------------------------------------------------------------


def test_real_v1_loss_trace_non_increasing(real_v1_train_model):
    # The real assembled v1 training set (209 maps) must also produce a
    # non-increasing iteration trace, ending at the reported final_loss.
    trace = real_v1_train_model.loss_trace
    assert len(trace) == real_v1_train_model.n_iter
    assert all(b <= a for a, b in itertools.pairwise(trace))
    assert real_v1_train_model.final_loss == pytest.approx(trace[-1])


def test_real_v1_thresholds_strictly_increasing(real_v1_train_model):
    # Post-fit threshold ordering on the real training set: the softplus
    # reparameterization must keep theta_1 < theta_2 < theta_3.
    thresholds = real_v1_train_model.thresholds
    assert len(thresholds) == 3
    assert np.all(np.diff(thresholds) > 0.0)


# --------------------------------------------------------------------------
# plan#12: real v1 end-to-end via the training CLI + harness scoring
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_train_and_score_end_to_end():
    # The full M20 loop against real data/v1: run the training CLI
    # (which assembles the 209-map train matrix, fits with the
    # documented defaults, and writes data/v1/ordinal_logit_model.json),
    # reload the artifact through from_dict, wrap it with
    # make_model_fn over the real player_map_stats table, and score the
    # real 35-map test split through the evaluation harness directly
    # (not by shelling out to the evaluate CLI). Prints and records the
    # resulting mean_rps / mean_log_loss / marginal_binary_accuracy next
    # to the M18 floor (0.7317 / 0.9896 / 0.5143); beating the floor is
    # not a pass/fail gate for M20, but the numbers must be reported.
    from drivers import train_ordinal_logit

    rc = train_ordinal_logit.main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/ordinal_logit_model.json")
    assert artifact_path.exists()
    model = from_dict(json.loads(artifact_path.read_text(encoding="utf-8")))
    assert model.n_train == 209

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")

    held_out = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    assert len(held_out) == 35
    model_fn = make_model_fn(model, player_map_stats_df)
    scored = harness.score_held_out_maps(
        model_fn, held_out, matches_df, maps_df
    )
    report = harness.build_evaluation_report(scored)

    # The numbers are printed (and recorded in the BUILD status line and
    # this task's commit message); the sanity bounds below mirror the
    # M18-floor test's contract, not a floor-beating gate.
    print(
        "M20 ordinal-logit on real v1 test split (n_eval=35): "
        f"mean_rps={report['mean_rps']!r} "
        f"mean_log_loss={report['mean_log_loss']!r} "
        f"marginal_binary_accuracy={report['marginal_binary_accuracy']!r}"
    )
    assert report["n_eval"] == 35
    assert 0.0 <= report["mean_rps"] <= 3.0
    assert math.isfinite(report["mean_log_loss"])
    assert report["mean_log_loss"] > 0.0
    assert 0.0 <= report["marginal_binary_accuracy"] <= 1.0
    assert [entry["category"] for entry in report["calibration"]] == list(
        OUTCOME_LABELS
    )
    pred_matrix = scored[list(harness.PREDICTION_COLUMNS)].to_numpy()
    assert math.isfinite(pred_matrix.sum())
    assert (pred_matrix >= 0.0).all()
    assert all(
        abs(pred_matrix[i].sum() - 1.0) < 1e-9 for i in range(len(scored))
    )

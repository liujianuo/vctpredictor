# vctpredictor — Roadmap

Prioritized milestones derived from `description.txt` (revised, four-way outcome
spec). Each is intended to be implementable, reviewable, and shippable in a
single PLAN→BUILD→REVIEW→FIX pass.

Sizes: **S** ≈ under a pass with room to spare, **M** ≈ a full pass, **L** ≈ a
full pass with real risk of spilling (none are listed as L — anything that grew
that large has already been split below).

## Current state (already shipped, not repeated below)
- `scraper/` package: `models.py` (Team / PlayerStats / MapResult / Match
  dataclasses), `cache.py` (SQLite page + match cache, TTL helper), `vlr.py`
  (fetch/parse match, parse event match links), 26 fixture-based tests.
- `MapResult` already carries `team1_score` / `team2_score`, but nothing
  validates them and `PlayerStats` / `agent_picks` are unpopulated stubs.
- Not yet present: score validation, veto log parsing, half splits, outcome
  labels, persistent dataset, features, models, evaluation.

## Critical path
M1 → M9 → M10 → M13 → M14 → M20 → M21 → M22. M22 is the go/no-go gate: it tests
whether four-way granularity beats a plain binary model at v1 data scale. Build
toward it directly and defer anything that isn't on that path.

---

## Phase 0 — Data completeness

The description promotes the score parser from incidental to load-bearing: a
13-11 and a 16-14 are different training labels, so parsing errors corrupt the
labels themselves, not just a feature.

### M0. Config module for map pool + era [x]
Single source of truth for the active map pool, era date bounds, region filter,
and target event URLs.
- Size: **S**
- Depends on: —

### M1. Exact score parsing + validity assertions [x]
Capture the full final map score including OT rounds, and assert validity on
parse: winner ≥ 13, both teams ≥ 12 implies overtime, margin ≥ 2 in OT. Fail
loudly rather than storing a silently wrong label.
- Size: **S**
- Depends on: — (hardens existing `vlr.py` / `models.py`)

### M2. Score parser fixture suite [x]
Real match HTML fixtures covering regulation blowout (13-2), regulation close
(13-11), single OT (15-13) and multi-OT (16-14, 19-17), asserting exact scores
and the derived OT flag.
- Size: **S**
- Depends on: M1

### M3. Veto log parser [x]
Parse the ordered ban/pick/decider action sequence off a match page into a
`VetoAction` model (step index, acting team, action type, map). Fail loudly on
unrecognised phrasing rather than skipping the action.
- Size: **M**
- Depends on: —

### M4. Veto parser fixture suite [x]
Fixtures covering Bo1/Bo3/Bo5 veto formats and phrasing variants, asserting the
exact action sequence. Mitigates the veto-format-variation risk.
- Size: **S**
- Depends on: M3

### M5. Player-map stats parser [x]
Populate `PlayerStats` per player per map (ACS, K/D/A, ADR, KAST, FKPR, FDPR,
HS%, agent) plus `MapResult.agent_picks`.
- Size: **M**
- Depends on: —

### M6. Attack/defense half-split parsing [x]
Extract per-map first/second half and side (atk/def) round counts onto
`MapResult`. Feeds closeness features later.
- Size: **S**
- Depends on: M1

### M7. Scrape driver + rate limiting / robots [x]
CLI entry point that walks the configured events, applies a polite delay and a
robots.txt check, and writes every match through the cache. Idempotent re-runs.
- Size: **M**
- Depends on: M0, M1, M3

### M8. v1 dataset materialisation [x]
Turn cached matches into flat, versioned tables (`matches`, `maps`,
`veto_actions`, `player_map_stats`) on disk as Parquet, with a row-count and
sanity report (map count, OT rate, format mix).
- Size: **M**
- Depends on: M7

### M9. Four-way outcome labelling [x]
Derive the ordered label from each map score — A-regulation → A-OT → B-OT →
B-regulation — plus a signed round-margin column for the optional v2
formulation. This is the definition every downstream model trains against, so it
gets its own milestone and its own tests.
- Size: **S**
- Depends on: M8

---

## Phase 1 — Evaluation substrate

Built before any model, because RPS is the headline number and the split
protocol is a correctness requirement rather than a convention.

### M10. Chronological split + walk-forward CV [x]
Most recent ~15% held out as a two-way train/test split; expanding-window
(walk-forward) CV inside the training window; calibration is collected
out-of-fold rather than as a third static slice (M24 must not see train or
final-eval data). Shared by every stage.
- Size: **M**
- Depends on: M8

### M11. Proper scoring rules library [x]
RPS over ordered categories, multi-class log loss, Brier, and marginal binary
accuracy — as pure functions with unit tests, including the property that RPS
penalises adjacent-category misses less than distant ones.
- Size: **S**
- Depends on: M9

---

## Phase 2 — Feature layer (leakage-safe)

### M12. Point-in-time feature framework [x]
`features_as_of(team, date)` scaffolding: a strict as-of API plus a test proving
no row dated ≥ the match date can enter a feature. Everything downstream must go
through it.
- Size: **M**
- Depends on: M8

### M13. Bayesian-shrunk map win rates [x]
Partial-pooling estimator `(wins + k·prior)/(games + k)` shrinking map-specific
win rate toward the team's overall rate; expose the posterior, not just the point
estimate; choose k by cross-validation.
- Size: **S**
- Depends on: M12

### M14. Elo rating + team parity [x]
Sequential rating updated map-by-map in chronological order, queryable as-of a
date; exposes both the signed differential and its absolute value (the parity
feature the OT model needs).
- Size: **M**
- Depends on: M12

### M15. Closeness and overtime features [x]
Per-team frequency of close maps (≤2 round margin), heavily-shrunk per-team OT
rate, and per-map historical variance of round margins. The global OT base rate
is estimated from a wider pooled slice than the v1 era, since the team-level
signal is too sparse to stand alone.
- Size: **M**
- Depends on: M12, M9

### M16. Player-form features [x]
Recency-weighted rolling mean ACS/rating over the last N maps, aggregated to team
level.
- Size: **S**
- Depends on: M12, M5

### M17. Head-to-head + context features [x]
Heavily-shrunk H2H (overall and per-map), event stage, days since last match, and
a roster-change flag with post-change feature decay.
- Size: **M**
- Depends on: M12

---

## Phase 3 — Stage 2: per-map outcome

The heart of the project. Ordered before Stage 1 because the granularity gate
(M22) decides whether the output spec is viable at all.

### M18. Four-way baseline
P(A wins map) from shrunk win rates, split into regulation/OT by the global OT
base rate. Crude by design — it is the benchmark every later model must beat.
- Size: **S**
- Depends on: M13, M15

### M19. Map-outcome evaluation harness
RPS (headline) plus multi-class log loss, marginal binary accuracy, and
per-category calibration — including predicted vs observed OT rate, the category
most likely to be miscalibrated. Reports M18 as the floor.
- Size: **M**
- Depends on: M18, M10, M11

### M20. Ordinal logistic regression (proportional odds)
The primary model: one coefficient vector plus three thresholds over the M13–M17
features, with a coefficient report for interpretability.
- Size: **M**
- Depends on: M19, M14, M15, M16, M17

### M21. Multinomial comparison arm + proportional-odds diagnostic
Fit the four-class multinomial, compare against M20 on identical splits, and run
the Brant test (or per-category fit comparison) to check whether the shared-β
restriction holds. A violation here is what triggers M23.
- Size: **M**
- Depends on: M20

### M22. Granularity ablation — the viability gate
Marginalise the ordinal model back to a binary winner prediction and compare
against a logistic model trained directly on the binary target. If the granular
model loses substantially, the four-way spec is costing accuracy and the category
set or the ordinal structure has to change. Run this as early as M20 allows.
- Size: **S**
- Depends on: M20

### M23. Hurdle model fallback
Three binary components — P(OT), P(A wins | regulation), P(A wins | OT) — with
the OT-conditional pinned near 0.5 plus a small strength adjustment, since it
would train on only 10–20 rows. Separates closeness from direction, which the
proportional-odds form cannot. Build only if M21 shows a violation.
- Size: **M**
- Depends on: M19, M21

### M24. Latent-score temperature calibration
One-parameter Platt-style rescaling of the ordinal latent score before applying
thresholds, fit on the out-of-fold calibration set from M10. No resampling or
class reweighting anywhere — those improve accuracy while destroying the base
rate.
- Size: **M**
- Depends on: M20, M10

---

## Phase 4 — Stage 1: veto prediction

### M25. Rule-based greedy veto simulator
Ban the remaining map with the lowest shrunk win rate, pick the highest.
Deterministic, no training — and the hard-argmax limit of M27, which makes the
later comparison principled rather than arbitrary.
- Size: **S**
- Depends on: M13

### M26. Veto evaluation harness
Per-step cross-entropy and top-1/top-3 accuracy over veto actions, against both
the "most frequently played map" baseline and M25.
- Size: **M**
- Depends on: M25, M10

### M27. Conditional logit ban model
Feature-based scoring function over remaining maps with softmax normalisation
across the shrinking candidate set, trained by cross-entropy against observed
bans. Scored by features, never by map identity.
- Size: **M**
- Depends on: M26

### M28. Conditional logit pick model
The same structure with independent coefficients for picks, plus decider
handling (the last remaining map is forced, not chosen, and must be excluded
from the likelihood).
- Size: **M**
- Depends on: M27

### M29. Ancestral veto sampler
Sample N full sequences forward through the per-step distributions, each carrying
its own sequence probability, ready for Stage 3 marginalisation.
- Size: **S**
- Depends on: M28

---

## Phase 5 — Stage 3: series aggregation

### M30. Recursive series path enumeration
Exact scoreline probabilities from ordered per-map binary win probabilities,
written as recursion over `(a_wins, b_wins, map_index)` so Bo3, Bo5 and Bo7 share
one implementation. No simulation at this level — enumeration is exact and cheap,
and simulating would only add variance.
- Size: **S**
- Depends on: —

### M31. Veto-marginalised series prediction
Collapse each map's four-way distribution to a binary (regulation + OT per side),
run M30 per sampled sequence, and average across M29's samples weighted by
sequence probability. The four-way detail is retained for reporting, not
propagated into the scoreline.
- Size: **M**
- Depends on: M29, M30, M20

### M32. Flat series-scoreline baseline
A single classifier predicting the scoreline directly, ignoring maps entirely.
The decisive comparison — if the two-stage pipeline loses to this, that is the
headline result.
- Size: **S**
- Depends on: M10

### M33. Series evaluation harness
RPS and log loss over the 4-outcome (Bo3) and 6-outcome (Bo5) scoreline
distributions, plus marginal match-win accuracy, against M32.
- Size: **M**
- Depends on: M31, M32

### M34. Stage isolation
Evaluate Stage 2 on actual played maps vs. M29-predicted maps and report the gap
as the cost of Stage 1 error compounding.
- Size: **S**
- Depends on: M33

### M35. Compounding diagnostics
Two cheap checks on the assumptions Stage 3 rests on: predicted vs. observed
sweep rate (2-0 / 3-0 probabilities are products of same-direction terms, so they
expose Stage 2 overconfidence first), and a test of whether map-1's outcome
predicts map-2's beyond what features explain.
- Size: **S**
- Depends on: M33

---

## Phase 6 — Uncertainty

The three components are distinct in kind and are reported separately, never
collapsed into one number.

### M36. Bootstrap prediction intervals (epistemic)
Resample the training set, refit, and propagate the spread into per-map and
series intervals; report `n_games_backing` alongside each map.
- Size: **M**
- Depends on: M31

### M37. Veto-conditional variance (structural)
Spread of the series distribution across sampled veto sequences — the project's
most distinctive output, and the only uncertainty component that resolves the
moment the veto happens.
- Size: **S**
- Depends on: M31

### M38. Per-category reliability diagrams
Calibration assessed per outcome category rather than on one binary probability,
for both the four-way map output and the series scorelines.
- Size: **M**
- Depends on: M33

---

## Phase 7 — Deliverable surface

### M39. `predict()` public API
Wire the stages into the documented signature
`predict(team_a, team_b, best_of, map_pool, as_of_date)`, returning
`predicted_veto`, `per_map` (four probabilities + interval + `n_games_backing`),
`series`, and `veto_sensitivity`.
- Size: **M**
- Depends on: M31, M36, M37, M24

### M40. Template narrative generator
Deterministic plain-language explanation built from the structured output — no
LLM, so it is testable.
- Size: **S**
- Depends on: M39

### M41. Optional LLM scouting-report layer
An LLM consuming the structured output for prose only, with the constraint
enforced in code: it receives numbers and never produces them, or the calibration
guarantees are lost.
- Size: **M**
- Depends on: M40

### M42. Reproducible results report
One command that regenerates every metric table, baseline comparison, and
diagnostic — including the M22 granularity verdict and the "was the two-stage
pipeline worth it?" comparison against M32.
- Size: **M**
- Depends on: M33, M38, M22

---

## Notes and deferrals
- **Deferred to v2:** round-differential regression (model the margin
  continuously, integrate onto the four buckets). More statistically efficient in
  principle, but needs a correctly specified error distribution and an awkward
  margin→category mapping. Revisit once the categorical version works.
- **Deferred to post-v1:** sequence models (LSTM/transformer over partial veto
  state), and composition / comp-vs-comp features — M5 collects the agent data,
  but modelling it needs more maps than v1 provides.
- **Explicitly rejected, not deferred:** SMOTE or class reweighting for OT
  sparsity. They improve accuracy metrics by distorting the base rate, and
  calibration is the actual goal.
- **Risk checkpoints:** M2 covers score-parsing errors; M4 covers veto-format
  variation; M12 covers temporal leakage; M15 and M23 cover OT sparsity; M22
  covers multi-class dilution; M34 and M35 cover error compounding.
